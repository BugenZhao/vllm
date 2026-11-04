// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::stream::{BoxStream, select_all};
use futures::{Stream, StreamExt as _, pin_mut};
use tonic::{Request, Response, Status};
use tracing::{Span, info, info_span};
use tracing_futures::Instrument as _;
use vllm_remote_llm::rpc::llm_server::{self, LlmServer};
use vllm_remote_llm::{GenerateNRequest, SampleOutput};

use super::inference::log_text_error;
use crate::state::AppState;

pub(crate) type LlmGrpcService = LlmServer<LlmServiceImpl>;

/// Internal gRPC service that serves this server's [`vllm_llm::Llm`] to other
/// Rust frontends, such as a router running its own text layer.
pub struct LlmServiceImpl {
    state: Arc<AppState>,
}

impl LlmServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl llm_server::Llm for LlmServiceImpl {
    type GenerateStream = Pin<Box<dyn Stream<Item = Result<SampleOutput, Status>> + Send>>;

    /// Submit an already lowered request to the `Llm` as `n` parallel samples
    /// and stream their raw outputs, interleaved and tagged with the sample
    /// index.
    async fn generate(
        &self,
        request: Request<GenerateNRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let started_at = Instant::now();
        let GenerateNRequest { request, n } = request.into_inner();
        let request_span = info_span!(
            "grpc_llm",
            request_id = %request.request_id,
            n,
            data_parallel_rank = ?request.data_parallel_rank,
        );
        info!(parent: &request_span, "gRPC Llm request received");

        let streams = (self.state.llm())
            .generate_n(request, n)
            .instrument(request_span.clone())
            .await
            .map_err(|error| {
                log_text_error(&request_span, started_at, "submission", error.into())
            })?;
        let stream = select_all(streams.into_iter().zip(0..).map(|(stream, index)| {
            stream.map(move |item| item.map(|output| SampleOutput { index, output }))
        }))
        .boxed();

        let response_stream =
            generate_responses(stream, request_span.clone(), started_at).instrument(request_span);
        Ok(Response::new(Box::pin(response_stream)))
    }
}

/// Forward the samples' output streams as gRPC responses.
///
/// The response body owns the output streams, so a client disconnect drops
/// them and aborts the samples in the engine. A failed sample ends the call,
/// which aborts the others.
#[try_stream]
async fn generate_responses(
    stream: BoxStream<'static, vllm_llm::Result<SampleOutput>>,
    request_span: Span,
    started_at: Instant,
    mut y: TryYielder<SampleOutput, Status>,
) -> Result<(), Status> {
    pin_mut!(stream);

    while let Some(output) = stream.next().await {
        let output = output
            .map_err(|error| log_text_error(&request_span, started_at, "stream", error.into()))?;
        y.yield_ok(output).await;
    }
    info!(
        parent: &request_span,
        elapsed_ms = started_at.elapsed().as_millis() as u64,
        "gRPC Llm stream closed"
    );
    Ok(())
}
