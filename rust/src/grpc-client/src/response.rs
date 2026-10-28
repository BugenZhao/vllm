// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Rebuilding [`DecodedTextEvent`]s from a `GenerateStream` response stream.
//!
//! The server already detokenized and applied stop strings, so this only
//! restores the shapes: token attributions come from the wire, and logprob
//! token strings are decoded locally with the same tokenizer and helpers the
//! server uses.

use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _};
use tonic::{Code, Status};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::sampling_mask::SamplingMask;
use vllm_llm::{FinishReason, Logprobs, PositionLogprobs, TokenLogprob, TokenUsage};
use vllm_proto as pb;
use vllm_proto::json::proto_struct_to_json;
use vllm_text::output::{decode_logprobs, decode_prompt_logprobs};
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{
    DecodedLogprobs, DecodedText, DecodedTextEvent, Error, Finished, Result, SampledDelta,
    TokenAnchor, TokenAttribution,
};

use crate::request::ResponseContext;

/// Map a gRPC status from the server onto the text error categories.
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

/// Convert the server's response stream into decoded text events.
///
/// Requests with `intermediate == false` still stream from the server, so the
/// deltas are coalesced into one terminal event here, like a local `TextLlm`
/// emits.
#[try_stream]
pub(crate) async fn decoded_text_events(
    context: ResponseContext,
    tokenizer: DynTokenizer,
    mut responses: impl Stream<Item = std::result::Result<pb::GenerateResponse, Status>> + Unpin,
    mut y: TryYielder<DecodedTextEvent, Error>,
) -> Result<()> {
    let ResponseContext {
        request_id,
        prompt_token_ids: local_prompt_token_ids,
        intermediate,
        skip_special_tokens,
    } = context;
    let mut prompt_token_count = None;
    let mut pending: Option<PendingDelta> = None;

    while let Some(response) = responses.next().await {
        let pb::GenerateResponse {
            prompt_info,
            outputs,
        } = response.map_err(status_to_error)?;

        if let Some(prompt_info) = prompt_info {
            let prompt_token_ids = match &local_prompt_token_ids {
                Some(ids) => Arc::clone(ids),
                None => Arc::from(prompt_info.token_ids.as_slice()),
            };
            prompt_token_count = Some(prompt_info.num_prompt_tokens as usize);
            let prompt_logprobs = (!prompt_info.logprobs.is_empty())
                .then(|| {
                    decode_prompt_logprobs(
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        &prompt_logprobs(&prompt_token_ids, &prompt_info),
                        skip_special_tokens,
                    )
                })
                .transpose()?;
            y.yield_ok(DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            })
            .await;
        }

        let Some(output) = outputs else {
            continue;
        };
        let prompt_token_count = prompt_token_count.ok_or_else(|| Error::Remote {
            message: format!("request `{request_id}` streamed output before prompt info"),
        })?;
        let delta = PendingDelta::from_proto(output, &tokenizer, skip_special_tokens, |info| {
            finished(info, prompt_token_count)
        })?;
        let delta = match pending.take() {
            Some(mut pending) => {
                pending.append(delta);
                pending
            }
            None => delta,
        };
        if intermediate || delta.finished.is_some() {
            let done = delta.finished.is_some();
            y.yield_ok(delta.into_event()).await;
            if done {
                return Ok(());
            }
        } else {
            pending = Some(delta);
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })
}

/// One `TextDelta` under construction.
struct PendingDelta {
    decoded: DecodedText,
    sampled: SampledDelta,
    finished: Option<Box<Finished>>,
}

impl PendingDelta {
    fn from_proto(
        output: pb::SequenceOutput,
        tokenizer: &DynTokenizer,
        skip_special_tokens: bool,
        finished: impl FnOnce(pb::FinishInfo) -> Result<Finished>,
    ) -> Result<Self> {
        let pb::SequenceOutput {
            index: _,
            text,
            num_tokens: _,
            token_ids,
            logprobs,
            ranks,
            candidate_tokens,
            finish_info,
            sampling_mask,
            token_attributions,
        } = output;

        let logprobs = (!logprobs.is_empty())
            .then(|| {
                let positions = (token_ids.iter())
                    .zip(logprobs)
                    .zip(ranks)
                    .zip(candidate_tokens)
                    .map(|(((&token_id, logprob), rank), candidates)| {
                        position_logprobs(token_id, logprob, rank, candidates)
                    })
                    .collect();
                decode_logprobs(
                    tokenizer.as_ref(),
                    &Logprobs { positions },
                    skip_special_tokens,
                )
            })
            .transpose()?;

        let mut finished = finish_info.map(finished).transpose()?;
        if let Some(finished) = &mut finished
            && !sampling_mask.is_empty()
        {
            finished.sampling_mask = Some(SamplingMask {
                rows: sampling_mask.into_iter().map(|row| row.ids).collect(),
            });
        }

        Ok(Self {
            decoded: DecodedText {
                text,
                attributions: (token_attributions.into_iter())
                    .map(|attribution| TokenAttribution {
                        token_id: attribution.token_id,
                        anchor: match attribution.zero_width {
                            true => TokenAnchor::ZeroWidth {
                                byte_offset: attribution.byte_offset,
                            },
                            false => TokenAnchor::Visible {
                                byte_offset: attribution.byte_offset,
                            },
                        },
                    })
                    .collect(),
            },
            sampled: SampledDelta {
                token_ids,
                logprobs,
            },
            finished: finished.map(Box::new),
        })
    }

    /// Append a later delta, as the local decoder coalesces a non-intermediate
    /// request's output into its terminal event.
    fn append(&mut self, later: Self) {
        self.decoded.append(later.decoded);
        self.sampled.token_ids.extend(later.sampled.token_ids);
        if let Some(later_logprobs) = later.sampled.logprobs {
            (self.sampled.logprobs)
                .get_or_insert_with(|| DecodedLogprobs { positions: vec![] })
                .positions
                .extend(later_logprobs.positions);
        }
        self.finished = later.finished;
    }

    fn into_event(self) -> DecodedTextEvent {
        DecodedTextEvent::TextDelta {
            decoded: self.decoded,
            sampled: self.sampled,
            finished: self.finished,
        }
    }
}

/// The scored token first, then the extra candidates, as the engine orders
/// them.
fn position_logprobs(
    token_id: u32,
    logprob: f32,
    rank: u32,
    candidates: pb::CandidateTokenInfo,
) -> PositionLogprobs {
    PositionLogprobs {
        entries: std::iter::once(TokenLogprob {
            token_id,
            logprob,
            rank,
        })
        .chain(candidates.tokens.into_iter().map(|token| TokenLogprob {
            token_id: token.id,
            logprob: token.logprob,
            rank: token.rank,
        }))
        .collect(),
    }
}

/// Rebuild the scored prompt positions. The server pads the first position,
/// which has no left context, with zeros.
fn prompt_logprobs(prompt_token_ids: &[u32], prompt_info: &pb::PromptInfo) -> Logprobs {
    Logprobs {
        positions: (prompt_token_ids.iter().skip(1))
            .zip(prompt_info.logprobs.iter().skip(1))
            .zip(prompt_info.ranks.iter().skip(1))
            .zip(prompt_info.candidate_tokens.iter().skip(1))
            .map(|(((&token_id, &logprob), &rank), candidates)| {
                position_logprobs(token_id, logprob, rank, candidates.clone())
            })
            .collect(),
    }
}

fn finished(info: pb::FinishInfo, prompt_token_count: usize) -> Result<Finished> {
    use pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};

    let finish_reason = match info.finish_reason() {
        PbFinishReason::Stop => match info.stop_reason {
            Some(PbStopReason::StopTokenId(id)) => {
                FinishReason::Stop(Some(StopReason::TokenId(id)))
            }
            Some(PbStopReason::StopString(text)) => {
                FinishReason::Stop(Some(StopReason::Text(text)))
            }
            Some(PbStopReason::EosTokenId(_)) | None => FinishReason::stop_eos(),
        },
        PbFinishReason::Length => FinishReason::Length,
        // The server also reports a detected repetition as aborted.
        PbFinishReason::Aborted => FinishReason::Abort,
        PbFinishReason::NotFinished => {
            return Err(Error::Remote {
                message: "finish info without a finish reason".to_string(),
            });
        }
    };

    Ok(Finished {
        usage: TokenUsage {
            prompt_token_count,
            output_token_count: info.num_output_tokens as usize,
            cached_token_count: info.num_cached_tokens.unwrap_or(0) as usize,
        },
        finish_reason,
        kv_transfer_params: info
            .kv_transfer_params
            .as_ref()
            .map(|params| Box::new(proto_struct_to_json(params))),
        ec_transfer_params: info
            .ec_transfer_params
            .as_ref()
            .map(|params| Box::new(proto_struct_to_json(params))),
        sampling_mask: None,
        // TODO: carry the engine's request timestamps on the wire.
        timestamps: Default::default(),
    })
}
