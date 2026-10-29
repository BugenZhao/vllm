// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! [`EngineClient`] over the `vllm-proto` `GenerateTokens` API of a remote
//! `vllm-rs` server.
//!
//! The caller runs the text layer itself: requests arrive lowered, with model
//! defaults and stop tokens resolved, and outputs return raw tokens for the
//! caller to detokenize. The remote server only runs the engine.

mod request;
mod response;

use std::time::Duration;

use futures::StreamExt as _;
use thiserror::Error;
use tokio::sync::watch;
use tokio_util::task::AbortOnDropHandle;
use tonic::transport::Channel;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tracing::debug;
use vllm_llm::{BoxGenerateOutputStream, EngineClient, EngineInfo, GenerateRequest, Result};
use vllm_proto as pb;
use vllm_proto::control_client::ControlClient;
use vllm_proto::inference_client::InferenceClient;

use crate::response::status_to_error;

/// Delay before re-subscribing to server health after the watch stream ends.
const HEALTH_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// Failure to connect a [`RemoteEngineClient`].
#[derive(Debug, Error)]
pub enum ConnectError {
    #[error("failed to query the server: {0}")]
    Status(#[from] tonic::Status),
    #[error("server speaks vllm-proto {server}, but this client needs {client}")]
    ProtoVersion {
        server: String,
        client: &'static str,
    },
}

/// An [`EngineClient`] served by a remote `vllm-rs` gRPC server.
pub struct RemoteEngineClient {
    inference: InferenceClient<Channel>,
    control: ControlClient<Channel>,
    info: EngineInfo,
    server_info: pb::ServerInfo,
    health: watch::Receiver<bool>,
    _health_task: AbortOnDropHandle<()>,
}

impl RemoteEngineClient {
    /// Query the server behind `channel` and start watching its health.
    pub async fn connect(channel: Channel) -> std::result::Result<Self, ConnectError> {
        let mut control = ControlClient::new(channel.clone());
        let server_info = control.get_server_info(pb::GetServerInfoRequest {}).await?.into_inner();
        // The caller resolves engine sampling parameters from its own model
        // config and revision, so both sides are deployed from one revision.
        if server_info.proto_version != pb::VERSION {
            return Err(ConnectError::ProtoVersion {
                server: server_info.proto_version,
                client: pb::VERSION,
            });
        }
        let model_info = control.get_model_info(pb::GetModelInfoRequest {}).await?.into_inner();
        let info = EngineInfo {
            max_model_len: server_info.max_model_len,
            // TODO: report the model dtype so multimodal inputs can be
            // prepared on this side.
            model_dtype: None,
            supports_lora: model_info.supports_lora,
            vllm_version: server_info.engine_version.clone(),
        };

        let (health_tx, health) = watch::channel(true);
        let health_task = AbortOnDropHandle::new(tokio::spawn(watch_health(
            HealthClient::new(channel.clone()),
            health_tx,
        )));

        Ok(Self {
            inference: InferenceClient::new(channel),
            control,
            info,
            server_info,
            health,
            _health_task: health_task,
        })
    }

    /// Server facts reported at connection time.
    pub fn server_info(&self) -> &pb::ServerInfo {
        &self.server_info
    }

    /// A control client for the same server, e.g. to watch engine loads.
    pub fn control_client(&self) -> ControlClient<Channel> {
        self.control.clone()
    }
}

/// Mirror the server's `vllm.Inference` health status, re-subscribing after
/// the stream ends. Unreachable counts as unhealthy.
async fn watch_health(mut client: HealthClient<Channel>, health: watch::Sender<bool>) {
    let service = pb::inference_server::SERVICE_NAME.to_string();
    loop {
        match client
            .watch(HealthCheckRequest {
                service: service.clone(),
            })
            .await
        {
            Ok(response) => {
                let mut statuses = response.into_inner();
                while let Some(Ok(status)) = statuses.next().await {
                    health.send_replace(status.status() == ServingStatus::Serving);
                }
                debug!("server health stream ended");
            }
            Err(status) => debug!(%status, "failed to watch server health"),
        }
        health.send_replace(false);
        tokio::time::sleep(HEALTH_RETRY_INTERVAL).await;
    }
}

#[async_trait::async_trait]
impl EngineClient for RemoteEngineClient {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.clone()
    }

    async fn generate(&self, request: GenerateRequest) -> Result<BoxGenerateOutputStream> {
        let proto = request::to_proto(request)?;
        // The response stream owns the call: dropping it cancels the RPC, and
        // the server aborts the request.
        let responses = (self.inference.clone())
            .generate_tokens(proto)
            .await
            .map_err(status_to_error)?
            .into_inner();
        Ok(response::generate_outputs(responses).boxed())
    }

    async fn abort(&self, external_ids: &[String]) -> Result<()> {
        if external_ids.is_empty() {
            // TODO: abort all requests submitted through this client; the
            // server's `Abort` takes explicit IDs only.
            return Err(vllm_llm::Error::UnsupportedByEngine {
                field: "aborting all requests",
            });
        }
        (self.control.clone())
            .abort(pb::AbortRequest {
                request_ids: external_ids.to_vec(),
            })
            .await
            .map_err(status_to_error)?;
        Ok(())
    }

    async fn shutdown(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}
