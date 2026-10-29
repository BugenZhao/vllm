// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `GenerateTokensResponse` stream to [`GenerateOutput`]s.

use std::sync::Arc;

use futures::{Stream, StreamExt as _};
use tonic::{Code, Status};
use vllm_engine_core_client::protocol::output::{
    RequestError, RequestSpecDecodeMetrics, StopReason,
};
use vllm_engine_core_client::protocol::sampling_mask::SamplingMask;
use vllm_llm::{
    Error, FinishReason, GenerateOutput, GeneratePromptInfo, Logprobs, PositionLogprobs,
    RequestTimestamps, Result, TokenLogprob, WeightVersionSpan,
};
use vllm_proto as pb;
use vllm_proto::json::proto_struct_to_json;

/// Map a gRPC status from the server onto the engine error categories.
pub(crate) fn status_to_error(status: Status) -> Error {
    let message = status.message().to_string();
    match status.code() {
        Code::InvalidArgument | Code::NotFound | Code::FailedPrecondition | Code::OutOfRange => {
            Error::RemoteInvalidRequest { message }
        }
        Code::Unavailable | Code::ResourceExhausted => Error::Unavailable { message },
        _ => Error::Remote { message },
    }
}

/// Convert the server's response stream into engine outputs.
pub(crate) fn generate_outputs(
    responses: impl Stream<Item = std::result::Result<pb::GenerateTokensResponse, Status>>
    + Send
    + 'static,
) -> impl Stream<Item = Result<GenerateOutput>> + Send + 'static {
    // Only the first response names the engine request; later outputs repeat it.
    let mut engine_request_id = String::new();
    responses.map(move |response| {
        let response = response.map_err(status_to_error)?;
        if response.prompt_info.is_some() {
            engine_request_id.clone_from(&response.engine_request_id);
        }
        output_from_proto(response, engine_request_id.clone())
    })
}

fn output_from_proto(
    response: pb::GenerateTokensResponse,
    request_id: String,
) -> Result<GenerateOutput> {
    let pb::GenerateTokensResponse {
        prompt_info,
        engine_request_id: _,
        token_ids,
        logprobs,
        finish,
        num_cached_tokens,
        kv_transfer_params,
        ec_transfer_params,
        sampling_mask,
        spec_decode_metrics,
        weight_versions,
        timestamps,
    } = response;

    Ok(GenerateOutput {
        request_id,
        prompt_info: prompt_info
            .map(|info| -> Result<_> {
                Ok(Box::new(GeneratePromptInfo {
                    prompt_token_ids: Arc::from(info.prompt_token_ids),
                    prompt_logprobs: info.prompt_logprobs.map(logprobs_from_proto).transpose()?,
                    prompt_token_id_logprobs: None,
                }))
            })
            .transpose()?,
        token_ids,
        logprobs: logprobs.map(logprobs_from_proto).transpose()?,
        finish_reason: finish.map(finish_from_proto).transpose()?,
        cached_token_count: num_cached_tokens as usize,
        kv_transfer_params: kv_transfer_params
            .map(|params| Box::new(proto_struct_to_json(&params))),
        ec_transfer_params: ec_transfer_params
            .map(|params| Box::new(proto_struct_to_json(&params))),
        sampling_mask: sampling_mask.map(|mask| SamplingMask {
            rows: mask.sequences.into_iter().map(|ids| ids.ids).collect(),
        }),
        spec_decode_metrics: spec_decode_metrics.map(|metrics| {
            Box::new(RequestSpecDecodeMetrics {
                num_spec_tokens: metrics.num_spec_tokens,
                histogram: metrics.histogram,
                num_draft_tokens: metrics.num_draft_tokens,
                per_step_accepted: metrics.per_step_accepted,
                per_step_drafted: metrics.per_step_drafted,
            })
        }),
        weight_versions: weight_versions.map(|versions| {
            (versions.spans.into_iter())
                .map(|span| WeightVersionSpan {
                    version: Arc::from(span.version),
                    start: span.start as usize,
                    end: span.end as usize,
                })
                .collect()
        }),
        routed_experts: None,
        timestamps: timestamps.map_or_else(RequestTimestamps::default, |timestamps| {
            RequestTimestamps {
                queued_ts: timestamps.queued,
                scheduled_ts: timestamps.scheduled,
                first_token_ts: timestamps.first_token,
                last_token_ts: timestamps.last_token,
            }
        }),
    })
}

fn logprobs_from_proto(logprobs: pb::TokenLogprobs) -> Result<Logprobs> {
    let positions = (logprobs.positions.into_iter())
        .map(|position| {
            let pb::token_logprobs::Position {
                token_ids,
                logprobs,
                ranks,
            } = position;
            if logprobs.len() != token_ids.len() || ranks.len() != token_ids.len() {
                return Err(Error::Remote {
                    message: "logprob position has mismatched column lengths".to_string(),
                });
            }
            let entries = (token_ids.into_iter().zip(logprobs).zip(ranks))
                .map(|((token_id, logprob), rank)| TokenLogprob {
                    token_id,
                    logprob,
                    rank,
                })
                .collect();
            Ok(PositionLogprobs { entries })
        })
        .collect::<Result<_>>()?;
    Ok(Logprobs { positions })
}

fn finish_from_proto(finish: pb::TokenFinish) -> Result<FinishReason> {
    use pb::token_finish::{Reason, StopReason as PbStopReason};

    let reason = Reason::try_from(finish.reason).map_err(|_| Error::Remote {
        message: format!("unknown finish reason {}", finish.reason),
    })?;
    let stop_reason = finish.stop_reason.map(|reason| match reason {
        PbStopReason::StopTokenId(id) => StopReason::TokenId(id),
        PbStopReason::StopString(text) => StopReason::Text(text),
    });
    Ok(match reason {
        Reason::Stop => FinishReason::Stop(stop_reason),
        Reason::Length => FinishReason::Length,
        Reason::Abort => FinishReason::Abort,
        Reason::Error => FinishReason::Error(finish.error.map(|error| {
            Box::new(RequestError {
                code: error.code,
                message: error.message,
                retryable: error.retryable,
            })
        })),
        Reason::Repetition => FinishReason::Repetition(stop_reason),
    })
}
