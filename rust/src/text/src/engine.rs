// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The text-generation engine boundary shared by local and remote engines.

use futures::stream::BoxStream;
use tokio::sync::watch;
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use crate::{DecodedTextEvent, Result, TextLlm, TextRequest, TextRequestProcessor};

/// Boxed decoded-text stream returned by [`TextEngine::generate`].
pub type BoxTextOutputStream = BoxStream<'static, Result<DecodedTextEvent>>;

/// Engine facts fixed when the engine connection is established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    /// Effective context length, after the engine auto-fitted its KV cache.
    pub max_model_len: u32,
    /// Effective model dtype, used to encode multimodal tensors. `None` when
    /// the engine does not report it, which disables multimodal inputs.
    pub model_dtype: Option<ModelDtype>,
    /// Whether the engine was started with LoRA support.
    pub supports_lora: bool,
    /// vLLM version reported by the engine.
    pub vllm_version: String,
}

/// Text generation: a [`TextRequest`] in, a [`DecodedTextEvent`] stream out.
///
/// [`TextLlm`] implements this over a local engine-core client. Other
/// implementations reach an engine over the network, or route among several
/// engines, under the same contract: prompt defaults, detokenization, and stop
/// strings are resolved by the engine side exactly as `TextLlm` resolves them.
#[async_trait::async_trait]
pub trait TextEngine: Send + Sync + 'static {
    /// Engine-free request preparation that is consistent with this engine's
    /// model and limits. Callers that need the final prompt tokens before
    /// submission tokenize with it first.
    fn request_processor(&self) -> &TextRequestProcessor;

    /// Engine facts fixed at connection time.
    fn info(&self) -> &EngineInfo;

    /// Subscribe to engine health; `true` means requests can be served.
    fn subscribe_health(&self) -> watch::Receiver<bool>;

    /// Whether the engine can currently serve requests.
    fn is_healthy(&self) -> bool {
        *self.subscribe_health().borrow()
    }

    /// Submit one request and stream incrementally decoded text.
    async fn generate(&self, request: TextRequest) -> Result<BoxTextOutputStream>;

    /// Abort in-flight requests by their external (user-supplied) request ids.
    /// An empty slice aborts every in-flight request.
    async fn abort(&self, external_ids: &[String]) -> Result<()>;

    /// The local engine, for routes that still need the engine-core client or
    /// raw token output directly.
    // TODO: move admin operations to an `EngineControl` trait so they work
    // against remote and routed engines too.
    fn local(&self) -> Option<&TextLlm> {
        None
    }

    /// Shut down the engine connection and its background tasks.
    async fn shutdown(self: Box<Self>) -> Result<()>;
}

/// Boxed [`TextEngine`] owned by a higher-level facade.
pub type DynTextEngine = Box<dyn TextEngine>;
