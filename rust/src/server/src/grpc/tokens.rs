// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `GenerateTokens` conversions between the proto messages and the engine
//! client's request and output types. The request arrives already lowered by
//! the caller's text layer, so fields map one to one.

use std::num::NonZeroU32;

use tonic::Status;
use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs};
use vllm_engine_core_client::protocol::output::{RequestError, StopReason};
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_engine_core_client::protocol::sampling::{
    EngineCoreSamplingParams, RepetitionDetectionParams,
};
use vllm_engine_core_client::protocol::structured_outputs::{
    EngineCoreStructuredOutputsParams, StructuredOutputBackend, StructuredOutputConstraint,
    StructuredOutputOptions, StructuredOutputsParams,
};
use vllm_llm::{FinishReason, GenerateOutput, GeneratePromptInfo, GenerateRequest};

use super::convert::kv_hints_from_proto;
use super::pb;
use super::pb::json::{json_to_proto_struct, proto_value_to_json};

// ========================================================================================
// Request conversion
// ========================================================================================

/// Convert a `GenerateTokensRequest` into the engine client's request.
pub fn to_generate_request(request: pb::GenerateTokensRequest) -> Result<GenerateRequest, Status> {
    let pb::GenerateTokensRequest {
        request_id,
        prompt_token_ids,
        sampling_params,
        arrival_time,
        cache_salt,
        trace_headers,
        priority,
        data_parallel_rank,
        session_id,
        kv_hints,
        reasoning_gate,
    } = request;
    let sampling_params =
        sampling_params.ok_or_else(|| Status::invalid_argument("sampling_params is required"))?;
    let reasoning_gate = reasoning_gate.unwrap_or_default();

    Ok(GenerateRequest {
        request_id,
        prompt_token_ids,
        sampling_params: sampling_params_from_proto(sampling_params)?,
        mm_features: None,
        arrival_time,
        cache_salt,
        trace_headers: trace_headers.map(|headers| headers.headers.into_iter().collect()),
        priority,
        data_parallel_rank,
        session_id,
        kv_hints: kv_hints.map(kv_hints_from_proto),
        reasoning_parser_kwargs: reasoning_gate.chat_template_kwargs.map(|kwargs| {
            ReasoningParserKwargs {
                chat_template_kwargs: (kwargs.fields.iter())
                    .map(|(key, value)| (key.clone(), proto_value_to_json(value)))
                    .collect(),
            }
        }),
        reasoning_ended: reasoning_gate.reasoning_ended,
        lora_request: None,
    })
}

fn sampling_params_from_proto(
    params: pb::EngineSamplingParams,
) -> Result<EngineCoreSamplingParams, Status> {
    let pb::EngineSamplingParams {
        temperature,
        watermarking,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        thinking_token_budget,
        logprobs,
        prompt_logprobs,
        prompt_logprob_start,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_detection,
        stop_token_ids,
        eos_token_id,
        all_stop_token_ids,
        logit_bias,
        allowed_token_ids,
        bad_words_token_ids,
        structured_outputs,
        logprob_token_ids,
        skip_reading_prefix_cache,
        extra_args,
        routed_experts_prompt_start,
        stream_interval,
    } = params;
    let stream_interval = stream_interval
        .map(|interval| {
            NonZeroU32::new(interval)
                .ok_or_else(|| Status::invalid_argument("stream_interval must be positive"))
        })
        .transpose()?;

    Ok(EngineCoreSamplingParams {
        temperature,
        watermarking,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        thinking_token_budget,
        logprobs,
        prompt_logprobs,
        prompt_logprob_token_ids: None,
        prompt_logprob_start,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_detection: repetition_detection.map(|detection| RepetitionDetectionParams {
            max_pattern_size: detection.max_pattern_size,
            min_pattern_size: detection.min_pattern_size,
            min_count: detection.min_count,
        }),
        stop_token_ids,
        eos_token_id,
        all_stop_token_ids: all_stop_token_ids.into_iter().collect(),
        logit_bias: logit_bias.map(|bias| bias.bias.into_iter().collect()),
        allowed_token_ids: allowed_token_ids.map(|ids| ids.ids),
        bad_words_token_ids: bad_words_token_ids
            .map(|words| words.sequences.into_iter().map(|ids| ids.ids).collect()),
        structured_outputs: structured_outputs.map(structured_outputs_from_proto).transpose()?,
        logprob_token_ids: logprob_token_ids.map(|ids| ids.ids),
        skip_reading_prefix_cache,
        extra_args: extra_args.map(|args| {
            (args.fields.iter())
                .map(|(key, value)| (key.clone(), proto_value_to_json(value)))
                .collect()
        }),
        routed_experts_prompt_start,
        stream_interval,
    })
}

fn structured_outputs_from_proto(
    params: pb::EngineStructuredOutputs,
) -> Result<EngineCoreStructuredOutputsParams, Status> {
    use pb::engine_structured_outputs::{Backend, Constraint};

    let backend = match params.backend() {
        Backend::Xgrammar => StructuredOutputBackend::Xgrammar,
        Backend::Guidance => StructuredOutputBackend::Guidance,
        Backend::Outlines => StructuredOutputBackend::Outlines,
        Backend::LmFormatEnforcer => StructuredOutputBackend::LmFormatEnforcer,
    };
    let pb::EngineStructuredOutputs {
        constraint,
        disable_any_whitespace,
        disable_additional_properties,
        whitespace_pattern,
        backend: _,
    } = params;
    let constraint = match constraint
        .ok_or_else(|| Status::invalid_argument("structured_outputs.constraint is required"))?
    {
        Constraint::Json(json) => StructuredOutputConstraint::Json(
            serde_json::from_str(&json)
                .map_err(|e| Status::invalid_argument(format!("invalid json schema: {e}")))?,
        ),
        Constraint::Regex(regex) => StructuredOutputConstraint::Regex(regex),
        Constraint::Choice(choice) => StructuredOutputConstraint::Choice(choice.choices),
        Constraint::Grammar(grammar) => StructuredOutputConstraint::Grammar(grammar),
        Constraint::JsonObject(_) => StructuredOutputConstraint::JsonObject,
        Constraint::StructuralTag(tag) => StructuredOutputConstraint::StructuralTag(tag),
    };

    Ok(EngineCoreStructuredOutputsParams {
        params: StructuredOutputsParams {
            constraint,
            options: StructuredOutputOptions {
                disable_any_whitespace,
                disable_additional_properties,
                whitespace_pattern,
            },
        },
        backend,
    })
}

// ========================================================================================
// Output conversion
// ========================================================================================

/// Convert one engine output into a `GenerateTokensResponse`.
pub fn to_generate_tokens_response(
    output: GenerateOutput,
) -> Result<pb::GenerateTokensResponse, Status> {
    let GenerateOutput {
        request_id,
        prompt_info,
        token_ids,
        logprobs,
        finish_reason,
        cached_token_count,
        kv_transfer_params,
        ec_transfer_params,
        sampling_mask,
        spec_decode_metrics,
        weight_versions,
        routed_experts,
        timestamps,
    } = output;
    if routed_experts.is_some() {
        return Err(Status::unimplemented(
            "routed_experts are not supported over GenerateTokens",
        ));
    }
    let engine_request_id = if prompt_info.is_some() {
        request_id
    } else {
        String::new()
    };

    Ok(pb::GenerateTokensResponse {
        prompt_info: prompt_info.map(|info| prompt_info_to_proto(*info)).transpose()?,
        engine_request_id,
        token_ids,
        logprobs: logprobs.as_ref().map(logprobs_to_proto),
        finish: finish_reason.map(finish_to_proto),
        num_cached_tokens: cached_token_count as u64,
        kv_transfer_params: kv_transfer_params.as_deref().and_then(json_to_proto_struct),
        ec_transfer_params: ec_transfer_params.as_deref().and_then(json_to_proto_struct),
        sampling_mask: sampling_mask.map(|mask| pb::TokenIdSequences {
            sequences: (mask.rows.into_iter()).map(|ids| pb::TokenIds { ids }).collect(),
        }),
        spec_decode_metrics: spec_decode_metrics.map(|metrics| pb::SpecDecodeMetrics {
            num_spec_tokens: metrics.num_spec_tokens,
            histogram: metrics.histogram,
            num_draft_tokens: metrics.num_draft_tokens,
            per_step_accepted: metrics.per_step_accepted,
            per_step_drafted: metrics.per_step_drafted,
        }),
        weight_versions: weight_versions.map(|spans| pb::WeightVersions {
            spans: (spans.into_iter())
                .map(|span| pb::weight_versions::Span {
                    version: span.version.to_string(),
                    start: span.start as u64,
                    end: span.end as u64,
                })
                .collect(),
        }),
        timestamps: Some(pb::EngineTimestamps {
            queued: timestamps.queued_ts,
            scheduled: timestamps.scheduled_ts,
            first_token: timestamps.first_token_ts,
            last_token: timestamps.last_token_ts,
        }),
    })
}

fn prompt_info_to_proto(info: GeneratePromptInfo) -> Result<pb::TokenPromptInfo, Status> {
    let GeneratePromptInfo {
        prompt_token_ids,
        prompt_logprobs,
        prompt_token_id_logprobs,
    } = info;
    if prompt_token_id_logprobs.is_some() {
        return Err(Status::unimplemented(
            "prompt_token_id_logprobs are not supported over GenerateTokens",
        ));
    }
    Ok(pb::TokenPromptInfo {
        prompt_token_ids: prompt_token_ids.to_vec(),
        prompt_logprobs: prompt_logprobs.as_ref().map(logprobs_to_proto),
    })
}

fn logprobs_to_proto(logprobs: &Logprobs) -> pb::TokenLogprobs {
    pb::TokenLogprobs {
        positions: (logprobs.positions.iter())
            .map(
                |PositionLogprobs { entries }| pb::token_logprobs::Position {
                    token_ids: entries.iter().map(|entry| entry.token_id).collect(),
                    logprobs: entries.iter().map(|entry| entry.logprob).collect(),
                    ranks: entries.iter().map(|entry| entry.rank).collect(),
                },
            )
            .collect(),
    }
}

fn finish_to_proto(reason: FinishReason) -> pb::TokenFinish {
    use pb::token_finish::{Reason, StopReason as PbStopReason};

    let stop_reason = |reason: Option<StopReason>| {
        reason.map(|reason| match reason {
            StopReason::TokenId(id) => PbStopReason::StopTokenId(id),
            StopReason::Text(text) => PbStopReason::StopString(text),
        })
    };
    let (reason, stop_reason, error) = match reason {
        FinishReason::Stop(reason) => (Reason::Stop, stop_reason(reason), None),
        FinishReason::Length => (Reason::Length, None, None),
        FinishReason::Abort => (Reason::Abort, None, None),
        FinishReason::Error(error) => (
            Reason::Error,
            None,
            error.map(|error| {
                let RequestError {
                    code,
                    message,
                    retryable,
                } = *error;
                pb::RequestError {
                    code,
                    message,
                    retryable,
                }
            }),
        ),
        FinishReason::Repetition(reason) => (Reason::Repetition, stop_reason(reason), None),
    };
    pb::TokenFinish {
        reason: reason.into(),
        stop_reason,
        error,
    }
}
