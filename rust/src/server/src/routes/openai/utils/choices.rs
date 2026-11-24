// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Merging the per-choice streams or responses of a request with several
//! choices: `n > 1`, or a batch of prompts.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _, TryStreamExt as _, stream};

use super::types::Usage;
use crate::error::ApiError;

/// One streamed chunk of a single choice.
pub(crate) trait ChoiceChunk {
    /// Stamp the index of the choice this chunk belongs to.
    fn set_choice_index(&mut self, index: u32);

    /// The usage of the final usage-only chunk, or `None` for any other chunk.
    fn final_usage_mut(&mut self) -> Option<&mut Usage>;

    /// Drop the per-request metrics.
    fn clear_metrics(&mut self);
}

/// The non-streaming response of a single choice.
pub(crate) trait ChoiceResponse {
    /// Stamp the index of the choice this response holds.
    fn set_choice_index(&mut self, index: u32);

    /// The usage of this response, if it reports one.
    fn usage_mut(&mut self) -> Option<&mut Usage>;

    /// Drop the per-request metrics.
    fn clear_metrics(&mut self);

    /// Move the choices of `other` after the choices of this response.
    fn append_choices(&mut self, other: Self);
}

/// The final usages of a request's choices, folded per prompt.
struct PromptUsages(Vec<Option<Usage>>);

impl PromptUsages {
    fn new(num_prompts: usize) -> Self {
        Self(vec![None; num_prompts])
    }

    /// Fold in the usage of a choice of prompt `prompt_index` per
    /// [`Usage::add_choice`].
    fn add_choice(&mut self, prompt_index: usize, usage: &Usage) {
        match &mut self.0[prompt_index] {
            Some(prompt) => prompt.add_choice(usage),
            slot @ None => *slot = Some(usage.clone()),
        }
    }

    /// The request's usage, with each prompt counted once per
    /// [`Usage::add_prompt`], or `None` when no choice reported usage.
    fn total(self) -> Option<Usage> {
        (self.0.into_iter().flatten()).reduce(|mut total, prompt| {
            total.add_prompt(&prompt);
            total
        })
    }
}

/// Merge the chunk streams of a request's choices, in the order the choices
/// produce them, with each choice's chunks stamped with its index.
///
/// Like Python's streaming response for `n > 1`, the choices' final usages are
/// sent once, as one final chunk, folded per [`Usage::add_choice`], and
/// per-request metrics, which describe a single sample, are reported only for
/// `n = 1`.
pub(crate) fn merge_choice_chunk_streams<C: ChoiceChunk + Send + 'static>(
    choices: Vec<impl Stream<Item = Result<C, ApiError>> + Send + 'static>,
) -> impl Stream<Item = Result<C, ApiError>> + Send + 'static {
    merge_batch_choice_chunk_streams(vec![choices])
}

/// Like [`merge_choice_chunk_streams`], for a batch of prompts: the choices come
/// grouped by prompt, are indexed in prompt-major order, and each prompt counts
/// once in the final usage.
#[try_stream]
pub(crate) async fn merge_batch_choice_chunk_streams<C: ChoiceChunk + Send + 'static>(
    choices_by_prompt: Vec<Vec<impl Stream<Item = Result<C, ApiError>> + Send + 'static>>,
    mut y: TryYielder<C, ApiError>,
) -> Result<(), ApiError> {
    let mut usages = PromptUsages::new(choices_by_prompt.len());
    let choices =
        (choices_by_prompt.into_iter().enumerate()).flat_map(|(prompt_index, choices)| {
            choices.into_iter().map(move |choice| (prompt_index, choice))
        });
    let mut chunks = stream::select_all(choices.zip(0..).map(|((prompt_index, choice), index)| {
        choice
            .map_ok(move |mut chunk: C| {
                chunk.set_choice_index(index);
                (prompt_index, chunk)
            })
            .boxed()
    }));
    let single_choice = chunks.len() == 1;
    let mut final_usage_chunk: Option<C> = None;

    while let Some(chunk) = chunks.next().await {
        let (prompt_index, mut chunk) = chunk?;
        let Some(usage) = chunk.final_usage_mut() else {
            y.yield_ok(chunk).await;
            continue;
        };
        usages.add_choice(prompt_index, usage);
        if final_usage_chunk.is_none() {
            if !single_choice {
                chunk.clear_metrics();
            }
            final_usage_chunk = Some(chunk);
        }
    }
    if let Some(mut chunk) = final_usage_chunk {
        if let (Some(usage), Some(total)) = (chunk.final_usage_mut(), usages.total()) {
            *usage = total;
        }
        y.yield_ok(chunk).await;
    }
    Ok(())
}

/// Merge the responses of a request's choices, in index order.
///
/// Like Python's non-streaming response for `n > 1`, the prompt-level fields
/// come from the first choice, usage folds per [`Usage::add_choice`], and
/// per-request metrics, which describe a single sample, are reported only for
/// `n = 1`.
pub(crate) fn merge_choice_responses<R: ChoiceResponse>(responses: Vec<R>) -> R {
    merge_batch_choice_responses(vec![responses])
}

/// Like [`merge_choice_responses`], for a batch of prompts: the choices come
/// grouped by prompt, are indexed in prompt-major order, and each prompt counts
/// once in the usage per [`Usage::add_prompt`].
pub(crate) fn merge_batch_choice_responses<R: ChoiceResponse>(
    mut choices_by_prompt: Vec<Vec<R>>,
) -> R {
    if let [choices] = choices_by_prompt.as_mut_slice()
        && choices.len() == 1
    {
        return choices.pop().expect("one choice");
    }
    let mut usages = PromptUsages::new(choices_by_prompt.len());
    let mut responses =
        (choices_by_prompt.into_iter().enumerate()).flat_map(|(prompt_index, choices)| {
            choices.into_iter().map(move |response| (prompt_index, response))
        });
    let (prompt_index, mut merged) = responses.next().expect("a request has at least one choice");
    if let Some(usage) = merged.usage_mut() {
        usages.add_choice(prompt_index, usage);
    }
    for (index, (prompt_index, mut response)) in (1..).zip(responses) {
        merged.clear_metrics();
        if let Some(usage) = response.usage_mut() {
            usages.add_choice(prompt_index, usage);
        }
        response.set_choice_index(index);
        merged.append_choices(response);
    }
    if let (Some(usage), Some(total)) = (merged.usage_mut(), usages.total()) {
        *usage = total;
    }
    merged
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use serde_json::json;

    use super::*;
    use crate::routes::openai::utils::types::PromptTokenUsageInfo;

    fn usage(
        prompt_tokens: usize,
        cached_tokens: usize,
        created_cache_tokens: usize,
        completion_tokens: usize,
    ) -> Usage {
        Usage {
            prompt_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens: Some(completion_tokens),
            prompt_tokens_details: Some(PromptTokenUsageInfo {
                cached_tokens,
                created_cache_tokens,
                prefill_reported: true,
            }),
            completion_tokens_details: None,
        }
    }

    /// A sample aborted before its prefill: the engine reported no cache
    /// usage for it.
    fn aborted_before_prefill(prompt_tokens: usize) -> Usage {
        let mut usage = usage(prompt_tokens, 0, 0, 0);
        usage.prompt_tokens_details.as_mut().unwrap().prefill_reported = false;
        usage
    }

    /// Two prompts with several samples each. A sample prefilled after a
    /// sibling hits the prefix the sibling just cached, so the request's cache
    /// usage counts each prompt's prefilled sample with the fewest cached
    /// tokens, with its own created-cache count, wherever it comes in the
    /// prompt's samples.
    fn usages_by_prompt() -> Vec<Vec<Usage>> {
        vec![
            vec![usage(5, 0, 2, 3), usage(5, 4, 1, 2)],
            vec![
                aborted_before_prefill(7),
                usage(7, 6, 5, 1),
                usage(7, 2, 3, 1),
            ],
        ]
    }

    fn expected_usage() -> serde_json::Value {
        json!({
            "prompt_tokens": 12,
            "total_tokens": 19,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 2, "created_cache_tokens": 5},
            "completion_tokens_details": null,
        })
    }

    struct FakeResponse(Usage);

    impl ChoiceResponse for FakeResponse {
        fn set_choice_index(&mut self, _index: u32) {}

        fn usage_mut(&mut self) -> Option<&mut Usage> {
            Some(&mut self.0)
        }

        fn clear_metrics(&mut self) {}

        fn append_choices(&mut self, _other: Self) {}
    }

    struct FakeChunk(Option<Usage>);

    impl ChoiceChunk for FakeChunk {
        fn set_choice_index(&mut self, _index: u32) {}

        fn final_usage_mut(&mut self) -> Option<&mut Usage> {
            self.0.as_mut()
        }

        fn clear_metrics(&mut self) {}
    }

    #[test]
    fn responses_report_each_prompts_fewest_cached_tokens() {
        let responses = (usages_by_prompt().into_iter())
            .map(|usages| usages.into_iter().map(FakeResponse).collect())
            .collect();

        let merged = merge_batch_choice_responses(responses);

        assert_eq!(serde_json::to_value(merged.0).unwrap(), expected_usage());
    }

    #[tokio::test]
    async fn chunks_report_each_prompts_fewest_cached_tokens() {
        let streams = (usages_by_prompt().into_iter())
            .map(|usages| {
                (usages.into_iter())
                    .map(|usage| stream::iter([Ok(FakeChunk(None)), Ok(FakeChunk(Some(usage)))]))
                    .collect()
            })
            .collect();

        let chunks: Vec<_> = merge_batch_choice_chunk_streams(streams)
            .map(|chunk| chunk.unwrap().0)
            .collect()
            .await;

        let (usage, deltas) = chunks.split_last().unwrap();
        assert!(deltas.iter().all(Option::is_none));
        assert_eq!(serde_json::to_value(usage).unwrap(), expected_usage());
    }
}
