// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The token-generation boundary shared by local, remote, and routed engines.

use futures::StreamExt as _;
use futures::stream::BoxStream;
use tokio::sync::watch;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use crate::{GenerateOutput, GenerateRequest, Llm, Result};

/// Boxed output stream returned by [`EngineClient::generate`].
pub type BoxGenerateOutputStream = BoxStream<'static, Result<GenerateOutput>>;

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

/// Token generation: a [`GenerateRequest`] in, a [`GenerateOutput`] stream out.
///
/// This is the Rust counterpart of Python's `EngineClient` generate/abort
/// surface. [`Llm`] implements it over a local engine-core client; other
/// implementations reach an engine over the network or route among several
/// engines. Requests arrive fully lowered (sampling defaults resolved, stop
/// tokens set), and outputs carry raw tokens, so detokenization and stop
/// strings stay with the caller.
#[async_trait::async_trait]
pub trait EngineClient: Send + Sync + 'static {
    /// Engine facts fixed at connection time.
    fn info(&self) -> &EngineInfo;

    /// Subscribe to engine health; `true` means requests can be served.
    fn subscribe_health(&self) -> watch::Receiver<bool>;

    /// Whether the engine can currently serve requests.
    fn is_healthy(&self) -> bool {
        *self.subscribe_health().borrow()
    }

    /// Submit one tokenized request and stream its outputs. Dropping the
    /// stream before the terminal output aborts the request.
    async fn generate(&self, request: GenerateRequest) -> Result<BoxGenerateOutputStream>;

    /// Abort in-flight requests by their external (user-supplied) request ids.
    /// An empty slice aborts every in-flight request.
    async fn abort(&self, external_ids: &[String]) -> Result<()>;

    /// The local engine-core client, for admin and utility routes.
    // TODO: move admin operations to an `EngineControl` trait so they work
    // against remote and routed engines too.
    fn engine_core_client(&self) -> Option<&EngineCoreClient> {
        None
    }

    /// Shut down the engine connection and its background tasks.
    async fn shutdown(self: Box<Self>) -> Result<()>;
}

#[async_trait::async_trait]
impl EngineClient for Llm {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.client.subscribe_health()
    }

    async fn generate(&self, request: GenerateRequest) -> Result<BoxGenerateOutputStream> {
        Ok(Llm::generate(self, request).await?.boxed())
    }

    async fn abort(&self, external_ids: &[String]) -> Result<()> {
        Llm::abort(self, external_ids).await
    }

    fn engine_core_client(&self) -> Option<&EngineCoreClient> {
        Some(&self.client)
    }

    async fn shutdown(self: Box<Self>) -> Result<()> {
        Llm::shutdown(*self).await
    }
}
