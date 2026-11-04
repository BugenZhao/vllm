// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use tracing::{Span, info, info_span};
use tracing_futures::Instrument as _;
use vllm_llm::{BoxGenerateOutputStream, GenerateOutput, GenerateRequest, LlmInfo};
use vllm_remote_llm::rpc::llm_server::{self, LlmServer};
use vllm_remote_llm::{AbortRequest, AbortResponse, InfoRequest};

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
    type GenerateStream = Pin<Box<dyn Stream<Item = Result<GenerateOutput, Status>> + Send>>;

    /// Report the `Llm`'s facts, which a remote caller adopts as its own.
    async fn info(&self, _request: Request<InfoRequest>) -> Result<Response<LlmInfo>, Status> {
        Ok(Response::new(self.state.llm().info().clone()))
    }

    /// Abort requests by their external IDs. Unlike [`vllm_llm::Llm::abort`],
    /// an empty list is rejected: the server is shared, so aborting every
    /// request is not one caller's to do.
    async fn abort(
        &self,
        request: Request<AbortRequest>,
    ) -> Result<Response<AbortResponse>, Status> {
        let request_ids = request.into_inner().request_ids;
        if request_ids.is_empty() {
            return Err(Status::invalid_argument("no request IDs to abort"));
        }
        (self.state.llm())
            .abort(&request_ids)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(AbortResponse {}))
    }

    /// Submit an already lowered request to the `Llm` and stream its raw
    /// outputs.
    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let started_at = Instant::now();
        let request = request.into_inner();
        let request_span = info_span!(
            "grpc_llm",
            request_id = %request.request_id,
            data_parallel_rank = ?request.data_parallel_rank,
        );
        info!(parent: &request_span, "gRPC Llm request received");

        let stream = (self.state.llm())
            .generate(request)
            .instrument(request_span.clone())
            .await
            .map_err(|error| {
                log_text_error(&request_span, started_at, "submission", error.into())
            })?;

        let response_stream =
            generate_responses(stream, request_span.clone(), started_at).instrument(request_span);
        Ok(Response::new(Box::pin(response_stream)))
    }
}

/// Forward one `Llm` output stream as gRPC responses.
///
/// The response body owns the output stream, so a client disconnect drops it
/// and aborts the request in the engine.
#[try_stream]
async fn generate_responses(
    stream: BoxGenerateOutputStream,
    request_span: Span,
    started_at: Instant,
    mut y: TryYielder<GenerateOutput, Status>,
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
