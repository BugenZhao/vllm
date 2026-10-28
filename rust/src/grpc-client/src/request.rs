// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Lowering a [`TextRequest`] into a `vllm-proto` `GenerateRequest`.
//!
//! The lowering is exact for every field it accepts: the server rebuilds the
//! same [`TextRequest`] and resolves defaults, detokenization, and stop strings
//! as a local `TextLlm` would. A field the wire cannot carry is rejected rather
//! than dropped.

use std::sync::Arc;

use vllm_engine_core_client::protocol::kv_hints::KvHintsEnvelope;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputConstraint, StructuredOutputOptions, StructuredOutputsParams,
};
use vllm_proto as pb;
use vllm_proto::json::{json_to_proto_struct, json_to_proto_value};
use vllm_text::{
    Error, Prompt, PromptTruncation, PromptTruncationLimit, Result, SamplingParams,
    TextDecodeOptions, TextRequest, TokenIdsError, TruncationSide,
};

/// `vllm_xargs` keys that the wire carries as `KVCacheParameters` fields.
const KV_TRANSFER_PARAMS: &str = "kv_transfer_params";
const EC_TRANSFER_PARAMS: &str = "ec_transfer_params";

/// One lowered request plus what the response side needs to rebuild events.
pub(crate) struct LoweredRequest {
    pub proto: pb::GenerateRequest,
    pub data_parallel_rank: Option<u32>,
    pub response: ResponseContext,
}

/// Request facts the response conversion needs.
pub(crate) struct ResponseContext {
    pub request_id: String,
    /// Final prompt token IDs, when the server sends the prompt unchanged.
    /// `None` asks the server to echo them.
    pub prompt_token_ids: Option<Arc<[u32]>>,
    pub intermediate: bool,
    pub skip_special_tokens: bool,
}

fn unsupported(field: &'static str) -> Error {
    Error::UnsupportedByEngine { field }
}

/// Lower one text request for `GenerateStream`.
pub(crate) fn lower(request: TextRequest) -> Result<LoweredRequest> {
    let TextRequest {
        request_id,
        prompt,
        mm_features,
        sampling_params,
        decode_options,
        intermediate,
        priority,
        cache_salt,
        prompt_truncation,
        add_special_tokens,
        data_parallel_rank,
        session_id,
        kv_hints,
        reasoning_parser_kwargs,
        reasoning_ended,
        lora_request,
        // The server stamps its own arrival time.
        arrival_time: _,
    } = request;
    request_id_is_set(&request_id)?;

    // TODO: send prepared features as `PreprocessedMediaFeatures`, or raw media
    // for the server to prepare.
    if mm_features.is_some() {
        return Err(unsupported("multimodal inputs"));
    }

    let truncate_prompt_tokens = match prompt_truncation {
        None => 0,
        Some(PromptTruncation {
            limit: PromptTruncationLimit::Fixed(limit),
            side: TruncationSide::Left,
        }) => u32::try_from(limit).map_err(|_| unsupported("truncate_prompt_tokens"))?,
        Some(_) => return Err(unsupported("truncate_prompt_tokens")),
    };

    // The server reuses token IDs as given, and tokenizes text with special
    // tokens added.
    let (prompt, prompt_token_ids) = match prompt {
        Prompt::TokenIds(ids) => {
            let unchanged = truncate_prompt_tokens == 0;
            let local = unchanged.then(|| Arc::<[u32]>::from(ids.as_slice()));
            (
                pb::generate_request::Prompt::TokenIds(pb::TokenIds { ids }),
                local,
            )
        }
        Prompt::Text(_) if !add_special_tokens => return Err(unsupported("add_special_tokens")),
        Prompt::Text(text) => (pb::generate_request::Prompt::Text(text), None),
    };

    let skip_special_tokens = decode_options.skip_special_tokens;
    let LoweredSampling {
        temperature,
        sampling,
        decoding,
        stopping,
        response,
        kv,
        watermarking,
    } = lower_sampling(sampling_params, decode_options, prompt_token_ids.is_none())?;

    let proto = pb::GenerateRequest {
        request_id: request_id.clone(),
        // Empty selects the server's primary served model.
        model: String::new(),
        prompt: Some(prompt),
        temperature,
        sampling: Some(sampling),
        decoding: Some(decoding),
        stopping: Some(stopping),
        response: Some(response),
        kv: Some(pb::KvCacheParameters {
            cache_salt: cache_salt.unwrap_or_default(),
            ..kv
        }),
        truncate_prompt_tokens,
        priority,
        session_id,
        media: vec![],
        lora_name: lora_request.map(|lora| lora.lora_name).unwrap_or_default(),
        watermarking,
        kv_hints: kv_hints.map(kv_hints_to_proto),
        engine_reasoning_gate: (reasoning_ended.is_some()
            || !reasoning_parser_kwargs.chat_template_kwargs.is_empty())
        .then(|| pb::EngineReasoningGate {
            reasoning_ended,
            chat_template_kwargs: Some(prost_types::Struct {
                fields: (reasoning_parser_kwargs.chat_template_kwargs.iter())
                    .map(|(key, value)| (key.clone(), json_to_proto_value(value)))
                    .collect(),
            }),
        }),
        omitted_temperature: pb::OmittedTemperature::ModelDefault.into(),
    };

    Ok(LoweredRequest {
        proto,
        data_parallel_rank,
        response: ResponseContext {
            request_id,
            prompt_token_ids,
            intermediate,
            skip_special_tokens,
        },
    })
}

/// The server replaces an empty request ID with a random one, which would not
/// match the ID callers abort by.
fn request_id_is_set(request_id: &str) -> Result<()> {
    if request_id.is_empty() {
        return Err(unsupported("empty request_id"));
    }
    Ok(())
}

struct LoweredSampling {
    temperature: Option<f32>,
    sampling: pb::RandomSampling,
    decoding: pb::DecodingParameters,
    stopping: pb::StoppingCriteria,
    response: pb::ResponseOptions,
    kv: pb::KvCacheParameters,
    watermarking: Option<bool>,
}

fn lower_sampling(
    params: SamplingParams,
    decode_options: TextDecodeOptions,
    echo_prompt_token_ids: bool,
) -> Result<LoweredSampling> {
    let SamplingParams {
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
        ignore_eos,
        logit_bias,
        allowed_token_ids,
        bad_words,
        bad_words_token_ids,
        logprob_token_ids,
        structured_outputs,
        skip_reading_prefix_cache,
        routed_experts_prompt_start,
        vllm_xargs,
        stream_interval,
    } = params;
    let TextDecodeOptions {
        skip_special_tokens,
        include_stop_str_in_output,
        stop_strings,
        min_tokens: decode_min_tokens,
    } = decode_options;

    // Fields `vllm-proto` has no place for yet.
    // TODO: tokenize `bad_words` like lowering does, and extend the proto for
    // the rest when a caller needs them.
    for (set, field) in [
        (repetition_detection.is_some(), "repetition_detection"),
        (
            prompt_logprob_token_ids.is_some(),
            "prompt_logprob_token_ids",
        ),
        (prompt_logprob_start.is_some(), "prompt_logprob_start"),
        (
            routed_experts_prompt_start.is_some(),
            "routed_experts_prompt_start",
        ),
        (stream_interval.is_some(), "stream_interval"),
        (bad_words.is_some(), "bad_words"),
    ] {
        if set {
            return Err(unsupported(field));
        }
    }

    // The wire has one `min_new_tokens` for both the sampler and stop-string
    // checking.
    if min_tokens.unwrap_or(0) != decode_min_tokens {
        return Err(unsupported("min_tokens"));
    }
    // Zero is the wire's "omitted".
    if max_tokens == Some(0) {
        return Err(unsupported("max_tokens=0"));
    }
    // Empty is the wire's "omitted", which would lift the restriction.
    if allowed_token_ids.as_ref().is_some_and(Vec::is_empty) {
        return Err(TokenIdsError::EmptyAllowedTokenIds.into());
    }

    let output_candidates = match (logprobs, logprob_token_ids) {
        (None, None) => None,
        (Some(-1), None) => Some(pb::candidate_tokens::Select::All(true)),
        (Some(count), None) => Some(pb::candidate_tokens::Select::TopN(
            u32::try_from(count).map_err(|_| unsupported("logprobs"))?,
        )),
        (None, Some(ids)) if !ids.is_empty() => {
            Some(pb::candidate_tokens::Select::TokenIds(pb::TokenIds { ids }))
        }
        (_, Some(_)) => return Err(unsupported("logprob_token_ids")),
    };
    let prompt_candidates = match prompt_logprobs {
        None => None,
        Some(-1) => Some(pb::candidate_tokens::Select::All(true)),
        Some(count) => Some(pb::candidate_tokens::Select::TopN(
            u32::try_from(count).map_err(|_| unsupported("prompt_logprobs"))?,
        )),
    };

    let mut kv = pb::KvCacheParameters {
        bypass_prefix_cache: skip_reading_prefix_cache == Some(true),
        ..Default::default()
    };
    for (key, value) in vllm_xargs.into_iter().flatten() {
        let field = match key.as_str() {
            KV_TRANSFER_PARAMS => &mut kv.kv_transfer_params,
            EC_TRANSFER_PARAMS => &mut kv.ec_transfer_params,
            _ => return Err(unsupported("vllm_xargs")),
        };
        *field = Some(json_to_proto_struct(&value).ok_or_else(|| unsupported("vllm_xargs"))?);
    }

    Ok(LoweredSampling {
        temperature,
        sampling: pb::RandomSampling {
            num_sequences: 0,
            top_k,
            top_p,
            min_p,
            seed,
        },
        decoding: pb::DecodingParameters {
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            logit_bias: logit_bias.unwrap_or_default(),
            allowed_token_ids: allowed_token_ids.unwrap_or_default(),
            structured_output: structured_outputs.map(lower_structured_output).transpose()?,
            bad_words_token_ids: (bad_words_token_ids.into_iter().flatten())
                .map(|ids| pb::TokenIds { ids })
                .collect(),
        },
        stopping: pb::StoppingCriteria {
            max_new_tokens: max_tokens.unwrap_or(0),
            min_new_tokens: decode_min_tokens,
            stop_token_ids: stop_token_ids.unwrap_or_default(),
            stop_strings: stop_strings.unwrap_or_default(),
            include_stop_strings: include_stop_str_in_output,
            ignore_eos,
            thinking_token_budget,
        },
        response: pb::ResponseOptions {
            prompt_token_ids: echo_prompt_token_ids,
            prompt_logprobs: prompt_candidates.is_some(),
            prompt_candidates: prompt_candidates.map(|select| pb::CandidateTokens {
                select: Some(select),
            }),
            output_text: Some(true),
            output_token_ids: true,
            output_logprobs: output_candidates.is_some(),
            output_candidates: output_candidates.map(|select| pb::CandidateTokens {
                select: Some(select),
            }),
            skip_special_tokens: Some(skip_special_tokens),
            output_token_attributions: true,
        },
        kv,
        watermarking,
    })
}

fn lower_structured_output(
    params: StructuredOutputsParams,
) -> Result<pb::decoding_parameters::StructuredOutput> {
    use pb::decoding_parameters::{StringChoices, StructuredOutput};

    if params.options != StructuredOutputOptions::default() {
        return Err(unsupported("structured_outputs options"));
    }
    Ok(match params.constraint {
        StructuredOutputConstraint::Json(schema) => StructuredOutput::Json(schema.to_string()),
        StructuredOutputConstraint::Regex(regex) => StructuredOutput::Regex(regex),
        StructuredOutputConstraint::Choice(choices) => {
            StructuredOutput::Choice(StringChoices { choices })
        }
        StructuredOutputConstraint::Grammar(grammar) => StructuredOutput::Grammar(grammar),
        StructuredOutputConstraint::JsonObject => StructuredOutput::JsonObject(true),
        StructuredOutputConstraint::StructuralTag(tag) => StructuredOutput::StructuralTag(tag),
    })
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
