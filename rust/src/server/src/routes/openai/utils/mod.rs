// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod choices;
pub mod logprobs;
pub mod metrics;
pub mod structured_outputs;
pub mod types;
pub mod usage;
pub mod validated_json;

use std::collections::HashMap;
use std::num::NonZeroU32;

use serde_json::Value;
use vllm_text::{PromptTruncation, TruncationSide};

use crate::error::{ApiError, bail_invalid_request};

pub(crate) fn validate_generation_prompt_truncation(limit: Option<i64>) -> Result<(), ApiError> {
    if let Some(limit) = limit
        && limit < -1
    {
        bail_invalid_request!(
            param = "truncate_prompt_tokens",
            "truncate_prompt_tokens must be >= -1."
        );
    }
    Ok(())
}

/// Reject per-request multimodal kwargs overrides, like Python's
/// `validate_request_mm_kwargs` without `--trust-request-mm-kwargs`, which the
/// Rust frontend does not support. An empty map overrides nothing and passes.
pub(crate) fn validate_request_mm_kwargs(
    mm_processor_kwargs: Option<&HashMap<String, Value>>,
    media_io_kwargs: Option<&HashMap<String, Value>>,
) -> Result<(), ApiError> {
    if mm_processor_kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
        bail_invalid_request!(
            param = "mm_processor_kwargs",
            "Per-request mm_processor_kwargs are disabled by default because they can change \
             multimodal preprocessing resource usage. Start the server with \
             --trust-request-mm-kwargs only when clients are trusted."
        );
    }
    if media_io_kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
        bail_invalid_request!(
            param = "media_io_kwargs",
            "Per-request media_io_kwargs are disabled by default because they can change \
             multimodal media loading resource usage. Start the server with \
             --trust-request-mm-kwargs only when clients are trusted."
        );
    }
    Ok(())
}

/// Resolve the number of choices to generate, checked like Python's
/// `SamplingParams` checks `n`.
pub(crate) fn resolve_n(n: Option<u32>) -> Result<NonZeroU32, ApiError> {
    let n = n.unwrap_or(1);
    let Some(resolved) = NonZeroU32::new(n) else {
        bail_invalid_request!(param = "n", "n must be at least 1, got {n}.");
    };
    Ok(resolved)
}

/// Reject more choices than the server's `VLLM_MAX_N_SEQUENCES` allows, like
/// Python's `SamplingParams` checks `n`.
pub(crate) fn validate_n_limit(n: u32, max_n_sequences: i64) -> Result<(), ApiError> {
    if i64::from(n) > max_n_sequences {
        bail_invalid_request!(
            param = "n",
            "n must be at most {max_n_sequences}, got {n}. To increase this limit, set the \
             VLLM_MAX_N_SEQUENCES environment variable."
        );
    }
    Ok(())
}

/// Reject an output budget larger than the whole context window, like
/// Python's `TokenizeParams` check for chat and completions. A budget that
/// fits the window is still clamped to the room the prompt leaves when the
/// request is lowered; `/inference/v1/generate` only clamps, as in Python.
pub(crate) fn validate_max_output_tokens(
    param: &'static str,
    max_tokens: Option<u32>,
    max_model_len: u32,
) -> Result<(), ApiError> {
    if let Some(max_tokens) = max_tokens
        && max_tokens > max_model_len
    {
        bail_invalid_request!(
            param = param,
            "{param}={max_tokens} cannot be greater than max_model_len={max_model_len}. \
             Please request fewer output tokens."
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

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;

    use super::*;

    #[test]
    fn validate_request_mm_kwargs_rejects_only_non_empty_overrides() {
        let empty = HashMap::new();
        let processor = HashMap::from([("num_crops".to_string(), json!(4))]);
        let media = HashMap::from([("video".to_string(), json!({"num_frames": 8}))]);

        let results = [
            (None, None),
            (Some(&empty), Some(&empty)),
            (None, Some(&media)),
            (Some(&processor), None),
            // Python checks `mm_processor_kwargs` first.
            (Some(&processor), Some(&media)),
        ]
        .map(
            |(processor, media)| match validate_request_mm_kwargs(processor, media) {
                Ok(()) => "ok".to_string(),
                Err(ApiError::InvalidRequest { message, param }) => format!("{param:?}: {message}"),
                Err(error) => panic!("unexpected error: {error:?}"),
            },
        );

        expect![[r#"
            [
                "ok",
                "ok",
                "Some(\"media_io_kwargs\"): Per-request media_io_kwargs are disabled by default because they can change multimodal media loading resource usage. Start the server with --trust-request-mm-kwargs only when clients are trusted.",
                "Some(\"mm_processor_kwargs\"): Per-request mm_processor_kwargs are disabled by default because they can change multimodal preprocessing resource usage. Start the server with --trust-request-mm-kwargs only when clients are trusted.",
                "Some(\"mm_processor_kwargs\"): Per-request mm_processor_kwargs are disabled by default because they can change multimodal preprocessing resource usage. Start the server with --trust-request-mm-kwargs only when clients are trusted.",
            ]
        "#]].assert_debug_eq(&results);
    }
}
