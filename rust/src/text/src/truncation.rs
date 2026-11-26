// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Which side of the prompt to discard when truncation is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncationSide {
    /// Keep the end of the prompt, discarding its prefix.
    Left,
    /// Keep the start of the prompt, discarding its suffix.
    Right,
}

/// Maximum number of prompt tokens to retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptTruncationLimit {
    /// Retain `max_model_len - 1` tokens, holding one token of context back
    /// so generation always has room for at least one output token.
    ///
    /// Python's `TokenizeParams.max_truncation_tokens`; `max_tokens` is an
    /// upper bound that is clamped after tokenization, not a reservation.
    InputBudget,
    /// Retain at most the given number of prompt tokens.
    Fixed(u64),
}

/// Typed prompt-truncation policy used inside the text frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTruncation {
    /// Maximum number of prompt tokens to retain.
    pub limit: PromptTruncationLimit,
    /// Side from which excess prompt tokens are discarded.
    pub side: TruncationSide,
}

impl PromptTruncation {
    /// Convert the OpenAI-compatible integer representation into a typed policy.
    pub fn from_wire(limit: i64, side: TruncationSide) -> Result<Self> {
        let limit = match limit {
            -1 => PromptTruncationLimit::InputBudget,
            0.. => PromptTruncationLimit::Fixed(limit as u64),
            _ => return Err(Error::InvalidTruncatePromptTokens { value: limit }),
        };
        Ok(Self { limit, side })
    }

    pub(crate) fn apply(self, prompt_token_ids: &mut Vec<u32>, max_model_len: u32) -> Result<()> {
        if let Some(kept) = self.kept_range(prompt_token_ids.len(), max_model_len)? {
            prompt_token_ids.truncate(kept.end);
            prompt_token_ids.drain(..kept.start);
        }
        Ok(())
    }

    /// The range of a `len`-token prompt that truncation keeps, or `None` when
    /// the prompt already fits.
    pub(crate) fn kept_range(self, len: usize, max_model_len: u32) -> Result<Option<Range<usize>>> {
        let max_input_tokens = match self.limit {
            PromptTruncationLimit::InputBudget => max_model_len.saturating_sub(1),
            PromptTruncationLimit::Fixed(limit) => {
                if limit > u64::from(max_model_len) {
                    return Err(Error::TruncatePromptTokensExceedsMaxModelLen {
                        value: limit,
                        max_model_len,
                    });
                }
                limit as u32
            }
        } as usize;

        if len <= max_input_tokens {
            return Ok(None);
        }

        Ok(Some(match self.side {
            TruncationSide::Left => len - max_input_tokens..len,
            TruncationSide::Right => 0..max_input_tokens,
        }))
    }
}

/// The indices of a text prompt's tokens that truncation keeps, given the
/// range `kept` it keeps of the whole prompt and the tokens the tokenizer
/// `added`.
///
/// Like the tokenizer's own truncation, which Python applies when no side is
/// requested, the added special tokens (BOS, `[CLS]`, `[SEP]`) take slots
/// within the limit and are kept, and only the text is cut, from the same
/// side. With more of them than the limit, `kept` applies as it is.
pub(crate) fn kept_text_tokens(kept: Range<usize>, added: &[bool]) -> Vec<usize> {
    let num_added = added.iter().filter(|&&added| added).count();
    let Some(num_text) = kept.len().checked_sub(num_added) else {
        return kept.collect();
    };
    let text: Vec<_> = (0..added.len()).filter(|&index| !added[index]).collect();
    // A range from the start keeps the beginning of the text.
    let text = if kept.start == 0 {
        &text[..num_text]
    } else {
        &text[text.len() - num_text..]
    };
    let mut kept: Vec<_> = (0..added.len()).filter(|&index| added[index]).collect();
    kept.extend(text);
    kept.sort_unstable();
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wire_limits() {
        assert_eq!(
            PromptTruncation::from_wire(-1, TruncationSide::Left).unwrap(),
            PromptTruncation {
                limit: PromptTruncationLimit::InputBudget,
                side: TruncationSide::Left,
            }
        );
        assert_eq!(
            PromptTruncation::from_wire(0, TruncationSide::Right).unwrap(),
            PromptTruncation {
                limit: PromptTruncationLimit::Fixed(0),
                side: TruncationSide::Right,
            }
        );
        assert!(matches!(
            PromptTruncation::from_wire(-2, TruncationSide::Left),
            Err(Error::InvalidTruncatePromptTokens { value: -2 })
        ));
    }

    #[test]
    fn applies_fixed_limit_from_either_side() {
        let mut left = vec![1, 2, 3, 4, 5];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(3),
            side: TruncationSide::Left,
        }
        .apply(&mut left, 90)
        .unwrap();
        assert_eq!(left, vec![3, 4, 5]);

        let mut right = vec![1, 2, 3, 4, 5];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(3),
            side: TruncationSide::Right,
        }
        .apply(&mut right, 90)
        .unwrap();
        assert_eq!(right, vec![1, 2, 3]);
    }

    /// Like the tokenizer's own truncation, a text prompt keeps the `[CLS]`
    /// and `[SEP]` the tokenizer added and cuts only its text, unless they
    /// alone exceed the limit.
    #[test]
    fn text_truncation_keeps_added_special_tokens() {
        // [CLS] a b c [SEP]
        let added = [true, false, false, false, true];
        assert_eq!(kept_text_tokens(0..3, &added), [0, 1, 4]);
        assert_eq!(kept_text_tokens(2..5, &added), [0, 3, 4]);
        assert_eq!(kept_text_tokens(0..2, &added), [0, 4]);
        assert_eq!(kept_text_tokens(0..1, &added), [0]);
        assert_eq!(kept_text_tokens(4..5, &added), [4]);
    }

    #[test]
    fn input_budget_holds_back_one_token_for_output() {
        let mut prompt = vec![0; 100];
        PromptTruncation {
            limit: PromptTruncationLimit::InputBudget,
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap();
        assert_eq!(prompt.len(), 69);
    }

    #[test]
    fn fixed_limit_cannot_exceed_max_model_len() {
        let mut prompt = vec![1, 2, 3];
        let error = PromptTruncation {
            limit: PromptTruncationLimit::Fixed(80),
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap_err();
        assert!(matches!(
            error,
            Error::TruncatePromptTokensExceedsMaxModelLen {
                value: 80,
                max_model_len: 70
            }
        ));

        let mut prompt = vec![1, 2, 3];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(70),
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap();
        assert_eq!(prompt, vec![1, 2, 3]);
    }

    #[test]
    fn zero_limit_clears_prompt() {
        let mut prompt = vec![1, 2, 3];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(0),
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 90)
        .unwrap();
        assert!(prompt.is_empty());
    }

    #[test]
    fn limit_at_or_above_prompt_length_is_a_noop() {
        for limit in [3, 10] {
            let mut prompt = vec![1, 2, 3];
            PromptTruncation {
                limit: PromptTruncationLimit::Fixed(limit),
                side: TruncationSide::Left,
            }
            .apply(&mut prompt, 90)
            .unwrap();
            assert_eq!(prompt, vec![1, 2, 3]);
        }
    }
}
