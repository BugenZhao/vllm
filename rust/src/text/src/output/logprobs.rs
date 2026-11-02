// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use vllm_llm::{Logprobs, PositionLogprobs};
use vllm_tokenizer::Tokenizer;

use crate::error::Error;

/// One decoded token candidate and its logprob metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedTokenLogprob {
    /// Original vocabulary token ID for this candidate.
    pub token_id: u32,
    /// Best-effort decoded token string for this candidate; empty when the
    /// request outputs no text.
    pub token: String,
    /// Log probability of this token candidate.
    pub logprob: f32,
    /// Vocabulary rank of this token candidate.
    pub rank: u32,
}

/// One position's decoded token candidates and their logprobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedPositionLogprobs {
    /// Candidate tokens for this position.
    pub entries: Vec<DecodedTokenLogprob>,
}

/// Decoded sample logprobs for generated token positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedLogprobs {
    /// Generated token positions covered by this payload.
    pub positions: Vec<DecodedPositionLogprobs>,
}

/// Decoded prompt logprobs for prompt token positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedPromptLogprobs {
    /// Original vocabulary token ID for the first prompt token.
    pub first_token_id: u32,
    /// Best-effort decoded string for the first prompt token; empty when the
    /// request outputs no text.
    ///
    /// The first prompt token has no left context to score against, so it is
    /// stored separately instead of appearing in `scored_positions`.
    pub first_token: String,
    /// Scored prompt positions after the first prompt token.
    ///
    /// `scored_positions[i]` corresponds to the prompt token at position `i +
    /// 1`.
    pub scored_positions: Vec<DecodedPositionLogprobs>,
}

/// The sampled tokens whose text is still incomplete, used to repair logprob
/// tokens of characters split across tokens.
///
/// Sample logprobs keep one context for the whole request, so the repair works
/// across engine updates; prompt logprobs start a fresh one.
#[derive(Debug, Default)]
pub(super) struct LogprobTokenContext {
    /// Sampled tokens since the sampled text was last complete, oldest first.
    /// Each of them reported `""`; the token that completes their text reports
    /// all of it.
    pending: Vec<u32>,
}

impl LogprobTokenContext {
    /// Pending tokens after which their text is reported as decoded, U+FFFD
    /// included, rather than waiting longer for a completing token. Real
    /// split characters, even overlapping ones, complete well before this.
    const MAX_PENDING: usize = 16;
}

/// Decode generated-token logprobs from the raw `llm` token-ID shape into the
/// text-layer decoded-token representation.
///
/// Each returned position corresponds to one generated token position from the
/// same `llm` update. `context` holds the request's earlier sampled tokens and
/// advances by this update's. `finished` marks the request's last update,
/// whose last position reports any text still pending.
pub(super) fn decode_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: Option<&T>,
    logprobs: &Logprobs,
    skip_special_tokens: bool,
    context: &mut LogprobTokenContext,
    finished: bool,
) -> Result<DecodedLogprobs, Error> {
    Ok(DecodedLogprobs {
        positions: decode_positions(
            tokenizer,
            &logprobs.positions,
            skip_special_tokens,
            context,
            finished,
        )?,
    })
}

/// Decode prompt logprobs from the raw `llm` token-ID shape into the text-layer
/// decoded-token representation.
///
/// The returned payload stores the first prompt token separately and decodes
/// the remaining scored prompt positions into `scored_positions`, matching
/// vLLM's prompt-logprobs semantics.
pub(super) fn decode_prompt_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: Option<&T>,
    prompt_token_ids: &[u32],
    logprobs: &Logprobs,
    skip_special_tokens: bool,
) -> Result<DecodedPromptLogprobs, Error> {
    let first_token_id = prompt_token_ids
        .first()
        .copied()
        .expect("prompt logprobs require at least one prompt token");
    let first_token = decode_token(tokenizer, first_token_id, skip_special_tokens)?;
    // Like Python, the context starts after the unscored first prompt token.
    let scored_positions = decode_positions(
        tokenizer,
        &logprobs.positions,
        skip_special_tokens,
        &mut LogprobTokenContext::default(),
        true,
    )?;

    Ok(DecodedPromptLogprobs {
        first_token_id,
        first_token,
        scored_positions,
    })
}

/// Decode consecutive token positions, advancing `context` by each position's
/// sampled token (its first entry). With `flush_last`, the last position
/// reports any text still pending, since no token follows to complete it.
fn decode_positions<T: Tokenizer + ?Sized>(
    tokenizer: Option<&T>,
    positions: &[PositionLogprobs],
    skip_special_tokens: bool,
    context: &mut LogprobTokenContext,
    flush_last: bool,
) -> Result<Vec<DecodedPositionLogprobs>, Error> {
    positions
        .iter()
        .enumerate()
        .map(|(index, position)| {
            let flush = flush_last && index + 1 == positions.len();
            decode_position_logprobs(tokenizer, position, skip_special_tokens, context, flush)
        })
        .try_collect()
}

/// Decode one token position's raw candidate set into decoded token strings
/// plus logprob metadata, then advance `context` by its sampled token (the
/// first entry).
///
/// With a tokenizer, every candidate token ID is decoded independently through
/// the active text backend, except where a character is split across tokens,
/// like Python's `LogprobsProcessor._correct_decoded_token`:
///
/// - A sampled token that leaves the sampled text incomplete reports `""`, and
///   the one that completes it reports everything since the text was last
///   complete, so the sampled tokens' strings join to the generated text.
///   Unlike Python's four-token window, the pending tokens are kept until the
///   text completes, so overlapping split characters also join, and a literal
///   U+FFFD (or bytes that never form a character) is reported as text once
///   more text follows or `flush` marks the last position.
/// - A top candidate with the sampled token's ID reports the same string. Any
///   other top candidate whose own text holds U+FFFD is decoded after the same
///   pending tokens and reports `""` unless it completes their text.
fn decode_position_logprobs<T: Tokenizer + ?Sized>(
    tokenizer: Option<&T>,
    position: &PositionLogprobs,
    skip_special_tokens: bool,
    context: &mut LogprobTokenContext,
    flush: bool,
) -> Result<DecodedPositionLogprobs, Error> {
    let mut sampled_pending = None;
    let mut entries = Vec::<DecodedTokenLogprob>::with_capacity(position.entries.len());
    for (index, entry) in position.entries.iter().enumerate() {
        let sampled = index == 0;
        // The sampled token also appears among the top candidates when it ranks
        // high enough.
        if let Some(first) = entries.first().filter(|first| first.token_id == entry.token_id) {
            let token = first.token.clone();
            entries.push(DecodedTokenLogprob {
                token_id: entry.token_id,
                token,
                logprob: entry.logprob,
                rank: entry.rank,
            });
            continue;
        }
        let mut token = decode_token(tokenizer, entry.token_id, skip_special_tokens)?;
        if let Some(tokenizer) = tokenizer
            && (token.contains(char::REPLACEMENT_CHARACTER)
                || (sampled && !context.pending.is_empty()))
        {
            let text = tokenizer.decode(
                &[context.pending.as_slice(), &[entry.token_id]].concat(),
                skip_special_tokens,
            )?;
            token = if sampled {
                let pending = text.ends_with(char::REPLACEMENT_CHARACTER)
                    && !flush
                    && context.pending.len() + 1 < LogprobTokenContext::MAX_PENDING;
                sampled_pending = Some(pending);
                if pending { String::new() } else { text }
            } else if text.contains(char::REPLACEMENT_CHARACTER) {
                String::new()
            } else {
                text
            };
        }
        entries.push(DecodedTokenLogprob {
            token_id: entry.token_id,
            token,
            logprob: entry.logprob,
            rank: entry.rank,
        });
    }

    match sampled_pending {
        Some(true) => context.pending.push(position.entries[0].token_id),
        Some(false) => context.pending.clear(),
        None => {}
    }
    Ok(DecodedPositionLogprobs { entries })
}

/// Without a tokenizer the request outputs no text, so the token string stays
/// empty.
fn decode_token<T: Tokenizer + ?Sized>(
    tokenizer: Option<&T>,
    token_id: u32,
    skip_special_tokens: bool,
) -> vllm_tokenizer::Result<String> {
    tokenizer.map_or(Ok(String::new()), |tokenizer| {
        tokenizer.decode(&[token_id], skip_special_tokens)
    })
}

#[cfg(test)]
mod tests {
    use vllm_llm::{Logprobs, PositionLogprobs, TokenLogprob};
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    #[test]
    fn decode_logprobs_decodes_every_candidate_token() {
        let tokenizer = TestTokenizer::new();
        let logprobs = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'a' as u32,
                        logprob: -0.1,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: b'b' as u32,
                        logprob: -0.2,
                        rank: 1,
                    },
                ],
            }],
        };

        assert_eq!(
            decode_logprobs(
                Some(&tokenizer),
                &logprobs,
                false,
                &mut LogprobTokenContext::default(),
                false,
            )
            .unwrap(),
            DecodedLogprobs {
                positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token_id: b'a' as u32,
                            token: "a".to_string(),
                            logprob: -0.1,
                            rank: 3,
                        },
                        DecodedTokenLogprob {
                            token_id: b'b' as u32,
                            token: "b".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        },
                    ],
                }],
            }
        );
    }

    #[test]
    fn decode_prompt_logprobs_separates_first_prompt_token() {
        let tokenizer = TestTokenizer::new();
        let logprobs = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![TokenLogprob {
                    token_id: b'x' as u32,
                    logprob: -0.4,
                    rank: 1,
                }],
            }],
        };

        assert_eq!(
            decode_prompt_logprobs(
                Some(&tokenizer),
                &[b'p' as u32, b'x' as u32],
                &logprobs,
                false
            )
            .unwrap(),
            DecodedPromptLogprobs {
                first_token_id: b'p' as u32,
                first_token: "p".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![DecodedTokenLogprob {
                        token_id: b'x' as u32,
                        token: "x".to_string(),
                        logprob: -0.4,
                        rank: 1,
                    }],
                }],
            }
        );
    }

    /// One position per sampled token, each with `alternatives` as top candidates.
    fn positions(sampled: &[u32], alternatives: &[u32]) -> Logprobs {
        Logprobs {
            positions: sampled
                .iter()
                .map(|&token_id| PositionLogprobs {
                    entries: std::iter::once(token_id)
                        .chain(alternatives.iter().copied())
                        .map(|token_id| TokenLogprob {
                            token_id,
                            logprob: -0.1,
                            rank: 1,
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Decoded strings per position, sampled token first.
    fn tokens(positions: &[DecodedPositionLogprobs]) -> Vec<Vec<String>> {
        positions
            .iter()
            .map(|position| position.entries.iter().map(|entry| entry.token.clone()).collect())
            .collect()
    }

    fn sampled_text(positions: &[DecodedPositionLogprobs]) -> String {
        positions.iter().map(|position| position.entries[0].token.as_str()).collect()
    }

    #[test]
    fn decode_logprobs_repairs_characters_split_across_tokens() {
        // Byte-level BPE tokens: 🍰 is F0 9F 8D B0, and 302 also carries the
        // following "이"; 303 is a space plus the first two bytes of “ (E2 80 9C);
        // 304 is a literal U+FFFD; 305 and 306 each complete one of 랤 (EB 9E A4),
        // 헠 (ED 97 A0) and 템 (ED 85 9C) and start the next.
        let tokenizer = TestTokenizer::new()
            .with_byte_token([0xf0, 0x9f], 301)
            .with_byte_token([0x8d, 0xb0, 0xec, 0x9d, 0xb4], 302)
            .with_byte_token([b' ', 0xe2, 0x80], 303)
            .with_regular_token("\u{fffd}", 304)
            .with_byte_token([0xa4, 0xed], 305)
            .with_byte_token([0x97, 0xa0, 0xed], 306);
        let cases: [(&[u32], &str); 8] = [
            // é split into two byte tokens.
            (&[b'a'.into(), 0xc3, 0xa9, b'b'.into()], "aéb"),
            // A token that starts with the rest of a character and continues
            // with more text.
            (&[301, 302], "🍰이"),
            // An incomplete token's other text goes to the completing token.
            (&[b'a'.into(), 303, 0x9c], "a “"),
            // Four byte tokens, the most a character can span.
            (&[0xf0, 0x9f, 0x8d, 0xb0], "🍰"),
            // Overlapping split characters spanning more than four tokens.
            (&[0xeb, 0x9e, 305, 306, 0x85, 0x9c], "랤헠템"),
            // A literal U+FFFD is text.
            (&[b'a'.into(), 304, b'b'.into()], "a\u{fffd}b"),
            // A trailing literal U+FFFD and bytes that never form a character
            // wait for more text until the request finishes.
            (&[b'a'.into(), 304], "a"),
            (&[b'a'.into(), 0xff, 0xc3], "a"),
        ];
        for (sampled, text) in cases {
            for finished in [false, true] {
                let decoded = decode_logprobs(
                    Some(&tokenizer),
                    &positions(sampled, &[]),
                    false,
                    &mut LogprobTokenContext::default(),
                    finished,
                )
                .unwrap();
                // Once finished, the tokens join to the full decoded text.
                let expected = match finished {
                    true => tokenizer.decode(sampled, false).unwrap(),
                    false => text.to_string(),
                };
                assert_eq!(
                    sampled_text(&decoded.positions),
                    expected,
                    "sampled {sampled:?}, finished {finished}"
                );
            }
        }

        let decoded = decode_logprobs(
            Some(&tokenizer),
            &positions(&[301, 302, b'a'.into(), 303, 0x9c, 304, b'b'.into()], &[]),
            false,
            &mut LogprobTokenContext::default(),
            false,
        )
        .unwrap();
        assert_eq!(
            tokens(&decoded.positions),
            [[""], ["🍰이"], ["a"], [""], [" “"], [""], ["\u{fffd}b"]]
        );
    }

    #[test]
    fn decode_logprobs_repairs_candidates_with_sampled_context_across_updates() {
        let tokenizer = TestTokenizer::new();
        let mut context = LogprobTokenContext::default();
        // The first update samples the lead byte of é; the second samples its
        // continuation byte, with è's continuation byte and `x` as candidates.
        let first = decode_logprobs(
            Some(&tokenizer),
            &positions(&[0xc3], &[b'x'.into()]),
            false,
            &mut context,
            false,
        )
        .unwrap();
        let second = decode_logprobs(
            Some(&tokenizer),
            &positions(&[0xa9], &[0xa8, b'x'.into()]),
            false,
            &mut context,
            true,
        )
        .unwrap();

        assert_eq!(tokens(&first.positions), [["", "x"]]);
        assert_eq!(tokens(&second.positions), [["é", "è", "x"]]);
    }

    #[test]
    fn decode_prompt_logprobs_repairs_split_characters() {
        let tokenizer = TestTokenizer::new();
        let decoded = decode_prompt_logprobs(
            Some(&tokenizer),
            &[b'p'.into(), 0xc3, 0xa9],
            &positions(&[0xc3, 0xa9], &[]),
            false,
        )
        .unwrap();

        assert_eq!(decoded.first_token, "p");
        assert_eq!(tokens(&decoded.scored_positions), [[""], ["é"]]);

        // The prompt is complete, so a trailing literal U+FFFD is text.
        let tokenizer = tokenizer.with_regular_token("\u{fffd}", 304);
        let decoded = decode_prompt_logprobs(
            Some(&tokenizer),
            &[b'p'.into(), b'a'.into(), 304],
            &positions(&[b'a'.into(), 304], &[]),
            false,
        )
        .unwrap();
        assert_eq!(tokens(&decoded.scored_positions), [["a"], ["\u{fffd}"]]);
    }

    #[test]
    fn decode_logprobs_reports_one_string_per_token_id() {
        // The sampled token appears again among the top candidates.
        let tokenizer = TestTokenizer::new().with_regular_token("\u{fffd}", 304);
        let decoded = decode_logprobs(
            Some(&tokenizer),
            &positions(&[b'a'.into(), 304, b'b'.into()], &[b'b'.into()]),
            false,
            &mut LogprobTokenContext::default(),
            false,
        )
        .unwrap();

        assert_eq!(
            tokens(&decoded.positions),
            [["a", "b"], ["", "b"], ["\u{fffd}b", "\u{fffd}b"]]
        );
    }
}
