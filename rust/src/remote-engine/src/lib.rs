// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The internal engine protocol between vLLM Rust frontends: the
//! [`EngineClient`] of a remote `vllm-rs` server, reached over gRPC with
//! [`GenerateRequest`] and [`GenerateOutput`](vllm_llm::GenerateOutput)
//! carried as MessagePack.
//!
//! The caller runs the text layer itself: requests arrive lowered, with model
//! defaults and stop tokens resolved, and outputs return raw tokens for the
//! caller to detokenize. The remote server only runs the engine.
//!
//! The payloads are the engine client's own types, not a published schema, so
//! both ends are expected to run the same vLLM revision. Fields are encoded by
//! name: a new `Option` or `#[serde(default)]` field keeps adjacent revisions
//! working, while renames, type changes, and new enum variants do not.

mod codec;

/// Generated client and server for the `vllm.internal.EngineRpc` service.
pub mod rpc {
    include!(concat!(env!("OUT_DIR"), "/vllm.internal.EngineRpc.rs"));
}

use std::time::Duration;

use futures::StreamExt as _;
use tokio::sync::watch;
use tokio_util::task::AbortOnDropHandle;
use tonic::transport::Channel;
use tonic::{Code, Status};
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tracing::debug;
use vllm_llm::{BoxGenerateOutputStream, EngineClient, EngineInfo, GenerateRequest, Result};
use vllm_proto as pb;
use vllm_proto::control_client::ControlClient;

use crate::rpc::engine_rpc_client::EngineRpcClient;
use crate::rpc::engine_rpc_server::SERVICE_NAME;

/// Delay before re-subscribing to server health after the watch stream ends.
const HEALTH_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// An [`EngineClient`] served by a remote `vllm-rs` gRPC server.
pub struct RemoteEngineClient {
    rpc: EngineRpcClient<Channel>,
    control: ControlClient<Channel>,
    info: EngineInfo,
    server_info: pb::ServerInfo,
    health: watch::Receiver<bool>,
    _health_task: AbortOnDropHandle<()>,
}

impl RemoteEngineClient {
    /// Query the server behind `channel` and start watching its health.
    pub async fn connect(channel: Channel) -> std::result::Result<Self, Status> {
        let mut control = ControlClient::new(channel.clone());
        let server_info = control.get_server_info(pb::GetServerInfoRequest {}).await?.into_inner();
        // TODO: check that the server runs a compatible revision.
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
            // Outputs come from a server of the same deployment, and can be as
            // large as the routed experts of a whole request.
            rpc: EngineRpcClient::new(channel).max_decoding_message_size(usize::MAX),
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

/// Mirror the server's `vllm.internal.EngineRpc` health status,
/// re-subscribing after the stream ends. Unreachable counts as unhealthy.
async fn watch_health(mut client: HealthClient<Channel>, health: watch::Sender<bool>) {
    let service = SERVICE_NAME.to_string();
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
        // The output stream owns the call: dropping it cancels the RPC, and
        // the server aborts the request.
        let outputs = (self.rpc.clone())
            .generate(request)
            .await
            .map_err(status_to_error)?
            .into_inner();
        Ok(outputs.map(|output| output.map_err(status_to_error)).boxed())
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

/// Map a gRPC status from the server onto the engine error categories.
fn status_to_error(status: Status) -> vllm_llm::Error {
    let message = status.message().to_string();
    match status.code() {
        Code::InvalidArgument | Code::NotFound | Code::FailedPrecondition | Code::OutOfRange => {
            vllm_llm::Error::RemoteInvalidRequest { message }
        }
        Code::Unavailable | Code::ResourceExhausted => vllm_llm::Error::Unavailable { message },
        _ => vllm_llm::Error::Remote { message },
    }
}
