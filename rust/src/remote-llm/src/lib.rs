// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The internal protocol between vLLM Rust frontends: the [`Llm`] of a
//! remote `vllm-rs` server, reached over gRPC with its [`LlmInfo`],
//! [`GenerateRequest`], and [`GenerateOutput`](vllm_llm::GenerateOutput)
//! carried as MessagePack.
//!
//! The caller runs the text layer itself: requests arrive lowered, with model
//! defaults and stop tokens resolved, and outputs return raw tokens for the
//! caller to detokenize. The remote server only runs its [`Llm`].
//!
//! The payloads are `vllm-llm`'s own types, not a published schema, so
//! both ends are expected to run the same vLLM revision. Structs with named
//! fields are encoded by name, so a new `Option` or `#[serde(default)]` field
//! in one keeps adjacent revisions working. Engine-core types encoded as
//! tuples, such as `LoraRequest` and the tensor types, are positional, so a
//! new field there breaks compatibility, as do renames, type changes, and new
//! enum variants anywhere.

mod codec;

/// Generated client and server for the `vllm.internal.Llm` service.
pub mod rpc {
    include!(concat!(env!("OUT_DIR"), "/vllm.internal.Llm.rs"));
}

use std::num::NonZeroU32;
use std::time::Duration;

use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio_util::task::AbortOnDropHandle;
use tonic::transport::Channel;
use tonic::{Code, Status};
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tracing::debug;
use vllm_llm::{BoxGenerateOutputStream, GenerateRequest, Llm, LlmInfo, Result};

use crate::rpc::llm_client::LlmClient;
use crate::rpc::llm_server::SERVICE_NAME;

/// Delay before re-subscribing to server readiness after the watch stream
/// ends.
const READINESS_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// The request of `vllm.internal.Llm/Info`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct InfoRequest {}

/// The request of `vllm.internal.Llm/Abort`: the external request IDs to
/// abort, as for [`Llm::abort`], except that it must not be empty.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbortRequest {
    pub request_ids: Vec<String>,
}

/// The response of `vllm.internal.Llm/Abort`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AbortResponse {}

/// An [`Llm`] served by a remote `vllm-rs` gRPC server.
///
/// It uses only `vllm.internal.Llm` and the health service, so the server may
/// run any [`Llm`], not only an engine-core one.
pub struct RemoteLlm {
    rpc: LlmClient<Channel>,
    info: LlmInfo,
    /// Stays `true`: a failing server makes this `Llm` unready, not failed,
    /// since restarting this process would not fix it.
    health: watch::Sender<bool>,
    readiness: watch::Receiver<bool>,
    _readiness_task: AbortOnDropHandle<()>,
}

impl RemoteLlm {
    /// Query the server behind `channel` and start watching its readiness.
    pub async fn connect(channel: Channel) -> std::result::Result<Self, Status> {
        // Outputs come from a server of the same deployment, and can be as
        // large as the routed experts of a whole request.
        let mut rpc = LlmClient::new(channel.clone()).max_decoding_message_size(usize::MAX);
        let info = rpc.info(InfoRequest {}).await?.into_inner();

        let (readiness_tx, readiness) = watch::channel(true);
        let readiness_task = AbortOnDropHandle::new(tokio::spawn(watch_readiness(
            HealthClient::new(channel),
            readiness_tx,
        )));

        Ok(Self {
            rpc,
            info,
            health: watch::Sender::new(true),
            readiness,
            _readiness_task: readiness_task,
        })
    }
}

/// Mirror the server's `vllm.internal.Llm` health status, which follows its
/// readiness, re-subscribing after the stream ends. Unreachable counts as not
/// ready.
async fn watch_readiness(mut client: HealthClient<Channel>, readiness: watch::Sender<bool>) {
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
                    readiness.send_replace(status.status() == ServingStatus::Serving);
                }
                debug!("server health stream ended");
            }
            Err(status) => debug!(%status, "failed to watch server health"),
        }
        readiness.send_replace(false);
        tokio::time::sleep(READINESS_RETRY_INTERVAL).await;
    }
}

#[async_trait::async_trait]
impl Llm for RemoteLlm {
    fn info(&self) -> &LlmInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }

    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.readiness.clone()
    }

    async fn generate_n(
        &self,
        request: GenerateRequest,
        n: NonZeroU32,
    ) -> Result<Vec<BoxGenerateOutputStream>> {
        if n > NonZeroU32::MIN {
            // TODO: carry `n` to the server so its `Llm` fans out the samples.
            return Err(vllm_llm::Error::Unsupported { field: "n > 1" });
        }
        // The output stream owns the call: dropping it cancels the RPC, and
        // the server aborts the request.
        let outputs = (self.rpc.clone())
            .generate(request)
            .await
            .map_err(status_to_error)?
            .into_inner();
        Ok(vec![
            outputs.map(|output| output.map_err(status_to_error)).boxed(),
        ])
    }

    async fn abort(&self, external_ids: &[String]) -> Result<()> {
        if external_ids.is_empty() {
            // TODO: abort all requests submitted through this client. Passing
            // the empty slice on would abort other callers' requests too.
            return Err(vllm_llm::Error::Unsupported {
                field: "aborting all requests",
            });
        }
        (self.rpc.clone())
            .abort(AbortRequest {
                request_ids: external_ids.to_vec(),
            })
            .await
            .map_err(status_to_error)?;
        Ok(())
    }

    /// The remote server is not owned, so there is nothing to stop there;
    /// dropping `self` stops the readiness watch.
    async fn shutdown(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

/// Map a gRPC status from the server onto the [`vllm_llm::Error`] categories.
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
