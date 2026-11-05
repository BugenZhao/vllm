// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust `llm` facade.
#[derive(Debug, Error)]
pub enum Error {
    #[error("generate request has an empty prompt_token_ids")]
    EmptyPromptTokenIds,
    #[error("engine-core error")]
    EngineCoreClient(#[from] vllm_engine_core_client::Error),
    #[error("sampling mask has {row_count} rows for {token_count} generated tokens")]
    SamplingMaskTokenCountMismatch {
        token_count: usize,
        row_count: usize,
    },
    #[error("invalid auxiliary output rows")]
    AuxOutputRows(#[from] vllm_engine_core_client::protocol::tensor::ArrayRowsError),
    /// The request uses a field this implementation does not carry out.
    #[error("`{field}` is not supported")]
    Unsupported { field: &'static str },
    /// The request cannot be taken now; retrying later may succeed.
    #[error("unavailable: {message}")]
    Unavailable { message: String },
    /// A remote implementation rejected the request as invalid.
    #[error("{message}")]
    RemoteInvalidRequest { message: String },
    /// A remote implementation failed the request.
    #[error("remote error: {message}")]
    Remote { message: String },
}
