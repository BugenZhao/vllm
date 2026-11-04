// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The internal protocol between vLLM Rust frontends: the [`Llm`] of a
//! remote `vllm-rs` server, reached over gRPC with [`GenerateRequest`] and
//! [`GenerateOutput`] carried as MessagePack.
//!
//! The caller runs the text layer itself: requests arrive lowered, with model
//! defaults and stop tokens resolved, and outputs return raw tokens for the
//! caller to detokenize. The remote server only runs its [`Llm`].
//!
//! The payloads are `vllm-llm`'s own types, not a published schema, so
//! both ends are expected to run the same vLLM revision. Fields are encoded by
//! name: a new `Option` or `#[serde(default)]` field keeps adjacent revisions
//! working, while renames, type changes, and new enum variants do not.

mod codec;

/// Generated client and server for the `vllm.internal.Llm` service.
pub mod rpc {
    include!(concat!(env!("OUT_DIR"), "/vllm.internal.Llm.rs"));
}

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};
use tokio_util::task::AbortOnDropHandle;
use tonic::transport::Channel;
use tonic::{Code, Status, Streaming};
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tracing::debug;
use vllm_llm::{BoxGenerateOutputStream, GenerateOutput, GenerateRequest, Llm, LlmInfo, Result};
use vllm_proto as pb;
use vllm_proto::control_client::ControlClient;

use crate::rpc::llm_client::LlmClient;
use crate::rpc::llm_server::SERVICE_NAME;

/// The request of `vllm.internal.Llm/Generate`: one lowered request submitted
/// as `n` parallel samples.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateNRequest {
    pub request: GenerateRequest,
    pub n: NonZeroU32,
}

/// A response of `vllm.internal.Llm/Generate`: one output of the sample at
/// `index`. Outputs of different samples interleave in one response stream.
#[derive(Debug, Serialize, Deserialize)]
pub struct SampleOutput {
    pub index: u32,
    pub output: GenerateOutput,
}

/// Delay before re-subscribing to server readiness after the watch stream
/// ends.
const READINESS_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// An [`Llm`] served by a remote `vllm-rs` gRPC server.
pub struct RemoteLlm {
    rpc: LlmClient<Channel>,
    control: ControlClient<Channel>,
    info: LlmInfo,
    server_info: pb::ServerInfo,
    /// Stays `true`: a failing server makes this `Llm` unready, not failed,
    /// since restarting this process would not fix it.
    health: watch::Sender<bool>,
    readiness: watch::Receiver<bool>,
    _readiness_task: AbortOnDropHandle<()>,
}

impl RemoteLlm {
    /// Query the server behind `channel` and start watching its readiness.
    pub async fn connect(channel: Channel) -> std::result::Result<Self, Status> {
        let mut control = ControlClient::new(channel.clone());
        let server_info = control.get_server_info(pb::GetServerInfoRequest {}).await?.into_inner();
        // TODO: check that the server runs a compatible revision.
        let model_info = control.get_model_info(pb::GetModelInfoRequest {}).await?.into_inner();
        let info = LlmInfo {
            max_model_len: server_info.max_model_len,
            // TODO: report the model dtype so multimodal inputs can be
            // prepared on this side.
            model_dtype: None,
            supports_lora: model_info.supports_lora,
            vllm_version: server_info.engine_version.clone(),
        };

        let (readiness_tx, readiness) = watch::channel(true);
        let readiness_task = AbortOnDropHandle::new(tokio::spawn(watch_readiness(
            HealthClient::new(channel.clone()),
            readiness_tx,
        )));

        Ok(Self {
            // Outputs come from a server of the same deployment, and can be as
            // large as the routed experts of a whole request.
            rpc: LlmClient::new(channel).max_decoding_message_size(usize::MAX),
            control,
            info,
            server_info,
            health: watch::Sender::new(true),
            readiness,
            _readiness_task: readiness_task,
        })
    }

    /// Server facts reported at connection time.
    pub fn server_info(&self) -> &pb::ServerInfo {
        &self.server_info
    }

    /// A control client for the same server.
    pub fn control_client(&self) -> ControlClient<Channel> {
        self.control.clone()
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

    /// The server's `Llm` fans out the samples, so their engine request IDs
    /// and seeds are derived as for a local one. All samples share one call:
    /// dropping every stream cancels it, and the server aborts the samples
    /// still running, but dropping only some streams does not abort those
    /// samples early.
    async fn generate_n(
        &self,
        request: GenerateRequest,
        n: NonZeroU32,
    ) -> Result<Vec<BoxGenerateOutputStream>> {
        let outputs = (self.rpc.clone())
            .generate(GenerateNRequest { request, n })
            .await
            .map_err(status_to_error)?
            .into_inner();
        if n == NonZeroU32::MIN {
            let stream =
                outputs.map(|item| item.map(|sample| sample.output).map_err(status_to_error));
            return Ok(vec![stream.boxed()]);
        }
        Ok(split_samples(outputs, n))
    }

    async fn abort(&self, external_ids: &[String]) -> Result<()> {
        if external_ids.is_empty() {
            // TODO: abort all requests submitted through this client; the
            // server's `Abort` takes explicit IDs only.
            return Err(vllm_llm::Error::Unsupported {
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

/// Split the interleaved outputs of one `Generate` call into a stream per
/// sample, in index order. A task forwards the outputs and keeps the call open
/// until every sample stream is dropped. A failed call ends every unfinished
/// sample with the error.
fn split_samples(
    mut outputs: Streaming<SampleOutput>,
    n: NonZeroU32,
) -> Vec<BoxGenerateOutputStream> {
    // Unbounded, so a sample the caller reads slowly does not hold back the
    // others; the server's generation rate bounds what they buffer.
    let (senders, receivers): (Vec<_>, Vec<_>) =
        (0..n.get()).map(|_| mpsc::unbounded_channel()).unzip();
    let mut senders: Vec<_> = senders.into_iter().map(Some).collect();
    let forward = async move {
        while let Some(item) = outputs.next().await {
            let SampleOutput { index, output } = match item {
                Ok(sample) => sample,
                Err(status) => {
                    for sender in senders.iter().flatten() {
                        let _ = sender.send(Err(status_to_error(status.clone())));
                    }
                    return;
                }
            };
            let Some(slot) = senders.get_mut(index as usize) else {
                let message = format!("server sent an output for sample {index} of {n}");
                for sender in senders.iter().flatten() {
                    let _ = sender.send(Err(vllm_llm::Error::Remote {
                        message: message.clone(),
                    }));
                }
                return;
            };
            let finished = output.finished();
            if let Some(sender) = slot {
                // A dropped sample stream only stops receiving.
                let _ = sender.send(Ok(output));
            }
            if finished {
                // End the sample's stream after its terminal output.
                *slot = None;
            }
        }
    };
    let call = Arc::new(AbortOnDropHandle::new(tokio::spawn(forward)));

    (receivers.into_iter())
        .map(|mut receiver| {
            let call = call.clone();
            futures::stream::poll_fn(move |cx| {
                // Hold the call open while this sample is read.
                let _call = &call;
                receiver.poll_recv(cx)
            })
            .boxed()
        })
        .collect()
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
