// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Output processing helpers shared by text and chat layers.

pub use decoded::{
    DecodedTextEvent, Finished, SampledDelta, TextDecodeOptions, decoded_text_event_stream,
};
pub use logprobs::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};
pub use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};

mod decoded;
mod logprobs;

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt as _, pin_mut};
use vllm_engine_core_client::protocol::output::RequestSpecDecodeMetrics;
use vllm_engine_core_client::protocol::sampling_mask::SamplingMask;

use crate::{Error, FinishReason, Result, TextOutputStream};

/// Final decoded text plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedTextOutput {
    pub text: String,
    pub prompt_token_ids: Arc<[u32]>,
    pub prompt_logprobs: Option<DecodedPromptLogprobs>,
    pub logprobs: Option<DecodedLogprobs>,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub usage: vllm_llm::TokenUsage,
    pub timestamps: vllm_llm::RequestTimestamps,
    /// Connector-specific KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<serde_json::Value>,
    /// Connector-specific encoder cache transfer parameters for disaggregated
    /// serving.
    pub ec_transfer_params: Option<serde_json::Value>,
    /// Sampling support sets aligned with generated token positions.
    pub sampling_mask: Option<SamplingMask>,
    /// Per-request speculative-decoding statistics emitted by the engine.
    pub spec_decode_metrics: Option<RequestSpecDecodeMetrics>,
}

/// Per-request stream of decoded text events.
pub struct DecodedTextEventStream {
    prompt_token_ids: Arc<[u32]>,
    inner: Pin<Box<dyn Stream<Item = Result<DecodedTextEvent>> + Send>>,
}

impl DecodedTextEventStream {
    /// Wrap the decoded text events of one request whose prompt is
    /// `prompt_token_ids`.
    pub fn new(prompt_token_ids: Arc<[u32]>, inner: impl TextOutputStream) -> Self {
        Self {
            prompt_token_ids,
            inner: Box::pin(inner),
        }
    }

    /// Return the prompt token IDs submitted to the engine for this request.
    pub fn prompt_token_ids(&self) -> &Arc<[u32]> {
        &self.prompt_token_ids
    }

    /// Collect the stream to completion and return the final decoded text plus
    /// terminal metadata.
    #[allow(clippy::manual_async_fn, reason = "specify `Send` bound")]
    pub fn collect_output(self) -> impl Future<Output = Result<CollectedTextOutput>> + Send {
        async move {
            let prompt_token_ids = Arc::clone(&self.prompt_token_ids);
            let stream = self;
            pin_mut!(stream);
            let mut prompt_logprobs = None;
            let mut collected: Option<CollectedTextOutput> = None;

            while let Some(event) = stream.next().await.transpose()? {
                match event {
                    DecodedTextEvent::Start {
                        prompt_logprobs: start_prompt_logprobs,
                    } => {
                        prompt_logprobs = start_prompt_logprobs;
                    }
                    DecodedTextEvent::TextDelta {
                        decoded,
                        sampled:
                            SampledDelta {
                                token_ids: delta_token_ids,
                                logprobs: mut delta_logprobs,
                            },
                        finished,
                    } => {
                        if let Some(c) = collected.as_mut() {
                            c.text.push_str(&decoded.text);
                            c.token_ids.extend(delta_token_ids);
                            if let Some(dlp) = delta_logprobs.as_mut() {
                                if let Some(lp) = c.logprobs.as_mut() {
                                    lp.positions.extend_from_slice(&dlp.positions);
                                } else {
                                    c.logprobs = delta_logprobs;
                                }
                            }
                        } else {
                            collected = Some(CollectedTextOutput {
                                text: decoded.text,
                                prompt_token_ids: Arc::clone(&prompt_token_ids),
                                prompt_logprobs: prompt_logprobs.take(),
                                logprobs: delta_logprobs,
                                token_ids: delta_token_ids,
                                finish_reason: FinishReason::Error(None),
                                usage: vllm_llm::TokenUsage::default(),
                                timestamps: vllm_llm::RequestTimestamps::default(),
                                kv_transfer_params: None,
                                ec_transfer_params: None,
                                sampling_mask: None,
                                spec_decode_metrics: None,
                            })
                        };

                        if let Some(finished) = finished {
                            let mut collected = collected.unwrap();
                            collected.finish_reason = finished.finish_reason;
                            collected.usage = finished.usage;
                            collected.timestamps = finished.timestamps;
                            collected.kv_transfer_params =
                                finished.kv_transfer_params.map(|value| *value);
                            collected.ec_transfer_params =
                                finished.ec_transfer_params.map(|value| *value);
                            collected.sampling_mask = finished.sampling_mask;
                            collected.spec_decode_metrics = finished.spec_decode_metrics;
                            return Ok(collected);
                        }
                    }
                }
            }

            // Note: this is actually unreachable, as the underlying stream always emit an
            // error on unexpected close.
            Err(Error::StreamClosedBeforeTerminalOutput {
                request_id: "unknown".to_string(),
            })
        }
    }
}

impl Stream for DecodedTextEventStream {
    type Item = Result<DecodedTextEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use vllm_llm::FinishReason;

    use super::*;

    #[tokio::test]
    async fn collect_output_retains_prompt_and_sample_logprobs() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token_id: 0,
                    first_token: "o".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "p".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(DecodedTextEvent::TextDelta {
                decoded: DecodedText::unattributed("bc"),
                sampled: SampledDelta {
                    token_ids: vec![1, 2],
                    logprobs: Some(DecodedLogprobs {
                        positions: vec![
                            DecodedPositionLogprobs {
                                entries: vec![DecodedTokenLogprob {
                                    token_id: 0,
                                    token: "a".to_string(),
                                    logprob: -0.2,
                                    rank: 1,
                                }],
                            },
                            DecodedPositionLogprobs {
                                entries: vec![DecodedTokenLogprob {
                                    token_id: 0,
                                    token: "bc".to_string(),
                                    logprob: -0.3,
                                    rank: 1,
                                }],
                            },
                        ],
                    }),
                },
                finished: Some(Box::new(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 2,
                        prompt_cache_usage: Default::default(),
                    },
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    spec_decode_metrics: None,
                    timestamps: Default::default(),
                })),
            }),
        ]);

        let collected = DecodedTextEventStream::new(vec![10, 11].into(), stream)
            .collect_output()
            .await
            .unwrap();
        assert_eq!(collected.text, "bc");
        assert_eq!(
            collected.prompt_logprobs,
            Some(DecodedPromptLogprobs {
                first_token_id: 0,
                first_token: "o".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![DecodedTokenLogprob {
                        token_id: 0,
                        token: "p".to_string(),
                        logprob: -0.1,
                        rank: 1,
                    }],
                }],
            })
        );
        assert_eq!(
            collected.logprobs,
            Some(DecodedLogprobs {
                positions: vec![
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "a".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "bc".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    },
                ],
            })
        );
    }

    #[tokio::test]
    async fn collect_output_accumulates_intermediate_deltas() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                decoded: DecodedText::unattributed("he"),
                sampled: SampledDelta {
                    token_ids: vec![1, 2],
                    logprobs: Some(DecodedLogprobs {
                        positions: vec![
                            DecodedPositionLogprobs {
                                entries: vec![DecodedTokenLogprob {
                                    token_id: 0,
                                    token: "h".to_string(),
                                    logprob: -0.1,
                                    rank: 1,
                                }],
                            },
                            DecodedPositionLogprobs {
                                entries: vec![DecodedTokenLogprob {
                                    token_id: 0,
                                    token: "e".to_string(),
                                    logprob: -0.2,
                                    rank: 1,
                                }],
                            },
                        ],
                    }),
                },
                finished: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                decoded: DecodedText::unattributed("llo"),
                sampled: SampledDelta {
                    token_ids: vec![3, 4, 5],
                    logprobs: Some(DecodedLogprobs {
                        positions: vec![
                            DecodedPositionLogprobs {
                                entries: vec![DecodedTokenLogprob {
                                    token_id: 0,
                                    token: "l".to_string(),
                                    logprob: -0.3,
                                    rank: 1,
                                }],
                            },
                            DecodedPositionLogprobs {
                                entries: vec![DecodedTokenLogprob {
                                    token_id: 0,
                                    token: "l".to_string(),
                                    logprob: -0.4,
                                    rank: 1,
                                }],
                            },
                            DecodedPositionLogprobs {
                                entries: vec![DecodedTokenLogprob {
                                    token_id: 0,
                                    token: "o".to_string(),
                                    logprob: -0.5,
                                    rank: 1,
                                }],
                            },
                        ],
                    }),
                },
                finished: Some(Box::new(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 5,
                        prompt_cache_usage: Default::default(),
                    },
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    spec_decode_metrics: None,
                    timestamps: Default::default(),
                })),
            }),
        ]);

        let collected = DecodedTextEventStream::new(vec![10, 11].into(), stream)
            .collect_output()
            .await
            .unwrap();
        assert_eq!(collected.text, "hello");
        assert_eq!(collected.prompt_logprobs, None);
        assert_eq!(collected.token_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(
            collected.logprobs,
            Some(DecodedLogprobs {
                positions: vec![
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "h".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "e".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "l".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "l".to_string(),
                            logprob: -0.4,
                            rank: 1,
                        }],
                    },
                    DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "o".to_string(),
                            logprob: -0.5,
                            rank: 1,
                        }],
                    },
                ],
            })
        );
    }
}
