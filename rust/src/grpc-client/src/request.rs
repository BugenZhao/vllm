// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! [`GenerateRequest`] to `GenerateTokensRequest`.
//!
//! Each field is destructured, so a new field fails to compile until it is
//! either carried or rejected here.

use vllm_engine_core_client::protocol::kv_hints::KvHintsEnvelope;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_engine_core_client::protocol::sampling::{
    EngineCoreSamplingParams, RepetitionDetectionParams,
};
use vllm_engine_core_client::protocol::structured_outputs::{
    EngineCoreStructuredOutputsParams, StructuredOutputBackend, StructuredOutputConstraint,
    StructuredOutputOptions, StructuredOutputsParams,
};
use vllm_llm::{Error, GenerateRequest, Result};
use vllm_proto as pb;
use vllm_proto::json::json_to_proto_value;

/// Convert a request to its wire form, rejecting fields the wire does not
/// carry yet.
pub(crate) fn to_proto(request: GenerateRequest) -> Result<pb::GenerateTokensRequest> {
    let GenerateRequest {
        request_id,
        prompt_token_ids,
        sampling_params,
        mm_features,
        arrival_time,
        cache_salt,
        trace_headers,
        priority,
        data_parallel_rank,
        session_id,
        kv_hints,
        reasoning_parser_kwargs,
        reasoning_ended,
        lora_request,
    } = request;
    if mm_features.is_some() {
        return Err(Error::UnsupportedByEngine {
            field: "mm_features",
        });
    }
    if lora_request.is_some() {
        return Err(Error::UnsupportedByEngine {
            field: "lora_request",
        });
    }

    let reasoning_gate =
        (reasoning_ended.is_some() || reasoning_parser_kwargs.is_some()).then(|| {
            pb::EngineReasoningGate {
                reasoning_ended,
                chat_template_kwargs: reasoning_parser_kwargs.map(
                    |ReasoningParserKwargs {
                         chat_template_kwargs,
                     }| prost_types::Struct {
                        fields: (chat_template_kwargs.iter())
                            .map(|(key, value)| (key.clone(), json_to_proto_value(value)))
                            .collect(),
                    },
                ),
            }
        });

    Ok(pb::GenerateTokensRequest {
        request_id,
        prompt_token_ids,
        sampling_params: Some(sampling_params_to_proto(sampling_params)?),
        arrival_time,
        cache_salt,
        trace_headers: trace_headers.map(|headers| pb::TraceHeaders {
            headers: headers.into_iter().collect(),
        }),
        priority,
        data_parallel_rank,
        session_id,
        kv_hints: kv_hints.map(kv_hints_to_proto),
        reasoning_gate,
    })
}

fn sampling_params_to_proto(params: EngineCoreSamplingParams) -> Result<pb::EngineSamplingParams> {
    let EngineCoreSamplingParams {
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
        prompt_logprob_token_ids,
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
    if prompt_logprob_token_ids.is_some() {
        return Err(Error::UnsupportedByEngine {
            field: "prompt_logprob_token_ids",
        });
    }

    Ok(pb::EngineSamplingParams {
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
        repetition_detection: repetition_detection.map(
            |RepetitionDetectionParams {
                 max_pattern_size,
                 min_pattern_size,
                 min_count,
             }| pb::RepetitionDetection {
                max_pattern_size,
                min_pattern_size,
                min_count,
            },
        ),
        stop_token_ids,
        eos_token_id,
        all_stop_token_ids: all_stop_token_ids.into_iter().collect(),
        logit_bias: logit_bias.map(|bias| pb::LogitBias {
            bias: bias.into_iter().collect(),
        }),
        allowed_token_ids: allowed_token_ids.map(|ids| pb::TokenIds { ids }),
        bad_words_token_ids: bad_words_token_ids.map(|words| pb::TokenIdSequences {
            sequences: words.into_iter().map(|ids| pb::TokenIds { ids }).collect(),
        }),
        structured_outputs: structured_outputs.map(structured_outputs_to_proto),
        logprob_token_ids: logprob_token_ids.map(|ids| pb::TokenIds { ids }),
        skip_reading_prefix_cache,
        extra_args: extra_args.map(|args| prost_types::Struct {
            fields: (args.iter())
                .map(|(key, value)| (key.clone(), json_to_proto_value(value)))
                .collect(),
        }),
        routed_experts_prompt_start,
        stream_interval: stream_interval.map(|interval| interval.get()),
    })
}

fn structured_outputs_to_proto(
    params: EngineCoreStructuredOutputsParams,
) -> pb::EngineStructuredOutputs {
    use pb::engine_structured_outputs::{Backend, Constraint};

    let EngineCoreStructuredOutputsParams {
        params:
            StructuredOutputsParams {
                constraint,
                options:
                    StructuredOutputOptions {
                        disable_any_whitespace,
                        disable_additional_properties,
                        whitespace_pattern,
                    },
            },
        backend,
    } = params;
    let constraint = match constraint {
        StructuredOutputConstraint::Json(schema) => Constraint::Json(schema.to_string()),
        StructuredOutputConstraint::Regex(regex) => Constraint::Regex(regex),
        StructuredOutputConstraint::Choice(choices) => {
            Constraint::Choice(pb::decoding_parameters::StringChoices { choices })
        }
        StructuredOutputConstraint::Grammar(grammar) => Constraint::Grammar(grammar),
        StructuredOutputConstraint::JsonObject => Constraint::JsonObject(true),
        StructuredOutputConstraint::StructuralTag(tag) => Constraint::StructuralTag(tag),
    };
    let backend = match backend {
        StructuredOutputBackend::Xgrammar => Backend::Xgrammar,
        StructuredOutputBackend::Guidance => Backend::Guidance,
        StructuredOutputBackend::Outlines => Backend::Outlines,
        StructuredOutputBackend::LmFormatEnforcer => Backend::LmFormatEnforcer,
    };

    pb::EngineStructuredOutputs {
        constraint: Some(constraint),
        disable_any_whitespace,
        disable_additional_properties,
        whitespace_pattern,
        backend: backend.into(),
    }
}

fn kv_hints_to_proto(hints: KvHintsEnvelope) -> pb::KvHintsEnvelope {
    pb::KvHintsEnvelope {
        protocol_version: hints.protocol_version,
        message_id: hints.message_id,
        actions: hints
            .actions
            .into_iter()
            .map(|action| pb::KvHintAction {
                action_id: action.action_id,
                action_type: action.action_type,
                action_version: action.action_version,
                payload: Some(prost_types::Struct {
                    fields: (action.payload.iter())
                        .map(|(key, value)| (key.clone(), json_to_proto_value(value)))
                        .collect(),
                }),
            })
            .collect(),
    }
}
