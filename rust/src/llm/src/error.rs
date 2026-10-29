// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust `llm` facade.
#[derive(Debug, Error)]
pub enum Error {
    #[error("generate request `{request_id}` has an empty prompt_token_ids")]
    EmptyPromptTokenIds { request_id: String },
    #[error("engine-core error")]
    EngineCoreClient(#[from] vllm_engine_core_client::Error),
    #[error(
        "sampling mask for request `{request_id}` has {row_count} rows for {token_count} generated tokens"
    )]
    SamplingMaskTokenCountMismatch {
        request_id: String,
        token_count: usize,
        row_count: usize,
    },
    #[error("routed experts for request `{request_id}` changed dtype or shape")]
    RoutedExpertsShapeMismatch { request_id: String },
    /// The request uses a field this engine does not carry out.
    #[error("`{field}` is not supported by this engine")]
    UnsupportedByEngine { field: &'static str },
    /// The engine cannot take the request now; retrying later may succeed.
    #[error("engine unavailable: {message}")]
    Unavailable { message: String },
    /// A remote engine rejected the request as invalid.
    #[error("{message}")]
    RemoteInvalidRequest { message: String },
    /// A remote engine failed the request.
    #[error("remote engine error: {message}")]
    Remote { message: String },
}
