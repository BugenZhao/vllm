// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! [`TextEngine`] over the `vllm-proto` gRPC API of a remote `vllm-rs` server.
//!
//! The remote server runs the text layer itself: it resolves sampling defaults
//! from its generation config, detokenizes, and applies stop strings, so a
//! request behaves as it would against a local `TextLlm` over the same model.

mod request;
mod response;

use std::time::Duration;

use futures::StreamExt as _;
use thiserror::Error;
use tokio::sync::watch;
use tokio_util::task::AbortOnDropHandle;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tracing::debug;
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputBackend;
use vllm_proto as pb;
use vllm_proto::control_client::ControlClient;
use vllm_proto::inference_client::InferenceClient;
use vllm_text::backend::DynTextBackend;
use vllm_text::{
    BoxTextOutputStream, EngineInfo, Result, TextEngine, TextRequest, TextRequestProcessor,
};

use crate::response::status_to_error;

/// gRPC metadata key that pins a request to one data parallel rank.
pub const DATA_PARALLEL_RANK_METADATA_KEY: &str = "x-data-parallel-rank";

/// Delay before re-subscribing to server health after the watch stream ends.
const HEALTH_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// Failure to connect a [`RemoteTextEngine`].
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

/// A [`TextEngine`] served by a remote `vllm-rs` gRPC server.
pub struct RemoteTextEngine {
    inference: InferenceClient<Channel>,
    control: ControlClient<Channel>,
    /// Request preparation over the same model as the server. Only its
    /// tokenizer is used here, to decode logprob token strings.
    processor: TextRequestProcessor,
    info: EngineInfo,
    server_info: pb::ServerInfo,
    health: watch::Receiver<bool>,
    _health_task: AbortOnDropHandle<()>,
}

impl RemoteTextEngine {
    /// Query the server behind `channel` and start watching its health.
    ///
    /// `backend` must load the same model and tokenizer as the server.
    pub async fn connect(
        channel: Channel,
        backend: DynTextBackend,
    ) -> std::result::Result<Self, ConnectError> {
        let mut control = ControlClient::new(channel.clone());
        let server_info = control.get_server_info(pb::GetServerInfoRequest {}).await?.into_inner();
        // The response side relies on token attributions, which older servers
        // do not send. Router and servers are deployed from one revision.
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
            processor: TextRequestProcessor::new(backend, server_info.max_model_len),
            info,
            server_info,
            health,
            _health_task: health_task,
        })
    }

    /// Override the maximum accepted logprobs count.
    pub fn with_max_logprobs(mut self, max_logprobs: Option<i32>) -> Self {
        self.processor = self.processor.with_max_logprobs(max_logprobs);
        self
    }

    /// Override the structured-output backend, xgrammar by default. It must
    /// match the server's.
    pub fn with_structured_outputs_backend(mut self, backend: StructuredOutputBackend) -> Self {
        self.processor = self.processor.with_structured_outputs_backend(backend);
        self
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
impl TextEngine for RemoteTextEngine {
    fn request_processor(&self) -> &TextRequestProcessor {
        &self.processor
    }

    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.clone()
    }

    async fn generate(&self, request: TextRequest) -> Result<BoxTextOutputStream> {
        let lowered = request::lower(request)?;
        let mut grpc_request = tonic::Request::new(lowered.proto);
        if let Some(rank) = lowered.data_parallel_rank {
            grpc_request
                .metadata_mut()
                .insert(DATA_PARALLEL_RANK_METADATA_KEY, MetadataValue::from(rank));
        }
        // The response stream owns the call: dropping it cancels the RPC, and
        // the server aborts the request.
        let responses = (self.inference.clone())
            .generate_stream(grpc_request)
            .await
            .map_err(status_to_error)?
            .into_inner();
        Ok(
            response::decoded_text_events(lowered.response, self.processor.tokenizer(), responses)
                .boxed(),
        )
    }

    async fn abort(&self, external_ids: &[String]) -> Result<()> {
        if external_ids.is_empty() {
            // TODO: abort all requests submitted through this client; the
            // server's `Abort` takes explicit IDs only.
            return Err(vllm_text::Error::UnsupportedByEngine {
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
