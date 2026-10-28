// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request input logging for `--enable-log-requests`, mirroring Python's
//! `RequestLogger.log_inputs`.

use tracing::{Level, debug, enabled, info};
use vllm_llm::GenerateRequest;
use vllm_tokenizer::Tokenizer;

/// Request input logging, enabled by `--enable-log-requests`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequestLogConfig {
    /// Maximum number of prompt characters or prompt token IDs logged
    /// (`--max-log-len`); `None` logs them in full.
    pub max_log_len: Option<usize>,
}

impl RequestLogConfig {
    /// Log one request's engine input like Python's `RequestLogger.log_inputs`:
    /// the sampling parameters and LoRA adapter at INFO, the prompt text and
    /// token IDs at DEBUG. The request ID comes from the enclosing request
    /// span.
    ///
    /// The prompt text is decoded from the final prompt token IDs, special
    /// tokens included, so it shows what the model sees. Python logs the
    /// request's own prompt text instead: the rendered template for chat, the
    /// raw text for completions, and none for token-ID prompts.
    pub(crate) fn log_inputs(self, request: &GenerateRequest, tokenizer: &dyn Tokenizer) {
        if enabled!(Level::DEBUG) {
            // Decode the whole prompt before cutting it to `max_log_len`
            // characters: cutting the IDs first could split a character
            // across byte-level tokens.
            let prompt = tokenizer.decode(&request.prompt_token_ids, false).ok();
            let prompt =
                prompt.as_deref().map(|prompt| truncate_log_text(prompt, self.max_log_len));
            let prompt_token_ids = truncate_log_ids(&request.prompt_token_ids, self.max_log_len);
            debug!(?prompt, ?prompt_token_ids, "request details");
        }
        info!(
            params = ?request.sampling_params,
            lora_request = ?request.lora_request,
            "received request"
        );
    }
}

/// The first `max_log_len` characters of logged text, or all of it when
/// `max_log_len` is `None`.
pub fn truncate_log_text(text: &str, max_log_len: Option<usize>) -> &str {
    match max_log_len.and_then(|max_log_len| text.char_indices().nth(max_log_len)) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

/// The first `max_log_len` logged token IDs, or all of them when
/// `max_log_len` is `None`.
pub fn truncate_log_ids<T>(ids: &[T], max_log_len: Option<usize>) -> &[T] {
    match max_log_len {
        Some(max_log_len) => &ids[..ids.len().min(max_log_len)],
        None => ids,
    }
}

#[cfg(test)]
mod tests {
    use super::{truncate_log_ids, truncate_log_text};

    #[test]
    fn truncate_log_text_counts_characters() {
        assert_eq!(truncate_log_text("héllo", Some(2)), "hé");
        assert_eq!(truncate_log_text("héllo", Some(10)), "héllo");
        assert_eq!(truncate_log_text("héllo", None), "héllo");
    }

    #[test]
    fn truncate_log_ids_keeps_leading_ids() {
        assert_eq!(truncate_log_ids(&[1, 2, 3], Some(2)), [1, 2]);
        assert_eq!(truncate_log_ids(&[1, 2, 3], Some(5)), [1, 2, 3]);
        assert_eq!(truncate_log_ids(&[1, 2, 3], None), [1, 2, 3]);
    }
}
