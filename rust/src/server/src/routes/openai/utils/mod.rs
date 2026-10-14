// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod logprobs;
pub mod metrics;
pub mod structured_outputs;
pub mod types;
pub mod usage;
pub mod validated_json;

use vllm_text::{PromptTruncation, TruncationSide};

use crate::error::{ApiError, bail_invalid_request};

/// Reject a non-empty requested model that this server does not serve. An
/// omitted or empty model selects the served model.
pub(crate) fn validate_requested_model(
    model: Option<&str>,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    if let Some(model) = model.filter(|model| !model.is_empty())
        && !served_model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.to_owned()));
    }
    Ok(())
}

pub(crate) fn validate_generation_prompt_truncation(
    limit: Option<i64>,
    echo: bool,
) -> Result<(), ApiError> {
    if let Some(limit) = limit
        && limit < -1
    {
        bail_invalid_request!(
            param = "truncate_prompt_tokens",
            "truncate_prompt_tokens must be >= -1."
        );
    }
    if echo && limit.is_some() {
        bail_invalid_request!(
            param = "echo",
            "`echo=true` is not supported with `truncate_prompt_tokens`."
        );
    }
    Ok(())
}

pub(crate) fn resolve_generation_prompt_truncation(
    limit: Option<i64>,
    side: Option<TruncationSide>,
) -> vllm_text::Result<Option<PromptTruncation>> {
    limit
        .map(|limit| PromptTruncation::from_wire(limit, side.unwrap_or(TruncationSide::Left)))
        .transpose()
}
