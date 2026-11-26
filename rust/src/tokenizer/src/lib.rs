// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use crate::incremental::DecodeStream;

mod byte_level_decode;
mod checkpoint_json;
#[macro_use]
mod error;
mod hf;
mod incremental;
mod tekken;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
mod tiktoken;
mod unavailable;

pub use checkpoint_json::{CheckpointJsonError, parse_checkpoint_json};
pub use error::{Result, TokenizerError};
pub use hf::HuggingFaceTokenizer;
pub use incremental::{DecodedText, IncrementalDecoder, TokenAnchor, TokenAttribution};
pub use tekken::TekkenTokenizer;
pub use tiktoken::TiktokenTokenizer;
pub use unavailable::UnavailableTokenizer;

pub trait Tokenizer: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Encode one prompt string like [`Self::encode`], also returning, per
    /// token, whether the tokenizer added it as a special token rather than
    /// reading it from the text, like Python's `special_tokens_mask`.
    ///
    /// By default, the added tokens are those around the text encoded without
    /// special tokens; none are marked when that encoding does not appear whole
    /// inside the full one.
    fn encode_with_special_tokens_mask(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<(Vec<u32>, Vec<bool>)> {
        let token_ids = self.encode(text, add_special_tokens)?;
        let mut added = vec![false; token_ids.len()];
        if add_special_tokens {
            let text_token_ids = self.encode(text, false)?;
            if let Some(start) = token_ids
                .windows(text_token_ids.len().max(1))
                .position(|window| text_token_ids.is_empty() || window == text_token_ids)
            {
                added[..start].fill(true);
                added[start + text_token_ids.len()..].fill(true);
            }
        }
        Ok((token_ids, added))
    }

    /// Equivalent to `encode(text, false)`, except that every added,
    /// special, and control-token matcher is bypassed.
    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Decode one token ID for per-token metadata, such as logprob token
    /// strings.
    ///
    /// Unlike sequence decoding, this preserves boundary whitespace encoded in
    /// token pieces when a backend strips it from an isolated token. Special
    /// tokens keep their spelling.
    fn decode_token(&self, token_id: u32) -> Result<String> {
        self.decode(&[token_id], false)
    }

    /// Convert one token string into a token ID, returning `None` if the token
    /// is not in the tokenizer vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Convert one token ID into the tokenizer's raw token string.
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// Borrow tokenizer added-vocabulary entries as `(token, id)` pairs.
    ///
    /// Backends that do not expose the distinction between model vocabulary
    /// and added vocabulary return an empty list.
    // TODO: add support to all tokenizer backends
    fn added_vocab(&self) -> &[(String, u32)] {
        &[]
    }

    /// Return the vocabulary size. Backends that cannot report it fall back to
    /// `usize::MAX`, an effectively unbounded value used only by test stubs.
    fn vocab_size(&self) -> usize {
        usize::MAX
    }

    /// Return whether the given token ID is special.
    fn is_special_id(&self, _token_id: u32) -> bool {
        false
    }

    /// Create a stateful incremental decoder primed with the given prompt
    /// tokens.
    ///
    /// The prompt tokens provide left context for the first generated token;
    /// the decoder does not re-emit prompt text.
    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        ))
    }
}

pub type DynTokenizer = Arc<dyn Tokenizer>;
