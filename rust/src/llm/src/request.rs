// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use uuid::Uuid;
use vllm_engine_core_client::protocol::kv_hints::KvHintsEnvelope;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::multimodal::MmFeatures;
use vllm_engine_core_client::protocol::request::{EngineCoreRequest, ReasoningParserKwargs};
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::error::{Error, Result};
use crate::request_metrics::current_unix_timestamp_secs;

/// Tokenized decoder-only generate request accepted by [`crate::EngineCoreLlm`].
///
/// This is the first-stage Rust subset of the inputs that eventually flow into
/// Python `AsyncLLM.generate()`. The boundary is intentionally above
/// [`EngineCoreRequest`], but below higher-level text and multimodal
/// preprocessing.
///
/// Original Python API reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/protocol.py#L67-L84>
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    /// Unique ID of the request.
    pub request_id: String,
    /// Token IDs of the prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Sampling parameters forwarded to engine-core.
    pub sampling_params: EngineCoreSamplingParams,
    /// Optional multimodal features already prepared by `vllm-chat`.
    pub mm_features: Option<MmFeatures>,
    /// Unix timestamp, in seconds, when this request arrived at the frontend.
    ///
    /// Stamped at the frontend entry, before render and tokenization, to match
    /// Python's renderer-entry arrival_time. When omitted, it is filled as a
    /// fallback before the request is sent to engine-core.
    pub arrival_time: Option<f64>,
    /// Optional salt used to partition prefix-cache entries for this request.
    pub cache_salt: Option<String>,
    /// Optional tracing headers to forward to engine-core and downstream
    /// observability hooks.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// Request scheduling priority. Lower values are scheduled earlier.
    pub priority: i32,
    /// Optional data-parallel rank override for routing this request.
    pub data_parallel_rank: Option<u32>,
    /// Stable session identity shared by related requests.
    pub session_id: Option<String>,
    /// Optional orchestrator-originated KV hints.
    pub kv_hints: Option<KvHintsEnvelope>,
    /// Optional reasoning-parser kwargs forwarded to engine-side structured
    /// output logic.
    pub reasoning_parser_kwargs: Option<ReasoningParserKwargs>,
    /// Optional engine reasoning-gate override.
    ///
    /// `Some(true)` means the submitted grammar covers reasoning from the first
    /// generated token.
    pub reasoning_ended: Option<bool>,
    /// Optional LoRA adapter request applied to this generation.
    pub lora_request: Option<LoraRequest>,
}

#[derive(Debug)]
pub(crate) struct PreparedGenerateRequest {
    pub engine_request: EngineCoreRequest,
    /// The original prompt token IDs, also moved into the raw engine request.
    pub prompt_token_ids: Arc<[u32]>,
}

impl GenerateRequest {
    /// Validate and lower this request into the raw engine-core request format.
    ///
    /// With `tower_connector_lora`, the engine applies LoRA to the multimodal
    /// encoder, so encoder-cache identifiers are scoped to the request's
    /// adapter (see [`tower_connector_lora_scope`]).
    pub(crate) fn prepare(
        self,
        randomize_request_id: bool,
        tower_connector_lora: bool,
    ) -> Result<PreparedGenerateRequest> {
        if self.prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id,
            });
        }
        let GenerateRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            mut mm_features,
            arrival_time,
            cache_salt,
            trace_headers,
            priority,
            data_parallel_rank,
            session_id,
            kv_hints,
            reasoning_parser_kwargs,
            reasoning_ended,
            lora_request,
        } = self;

        if tower_connector_lora && let Some(features) = &mut mm_features {
            let scope = tower_connector_lora_scope(lora_request.as_ref());
            for feature in features {
                feature.identifier.insert_str(0, &scope);
            }
        }

        let external_request_id = request_id;
        let engine_request_id = if randomize_request_id {
            let random_suffix = Uuid::new_v4().simple().to_string();
            format!("{external_request_id}-{}", &random_suffix[..8])
        } else {
            external_request_id.clone()
        };
        Ok(PreparedGenerateRequest {
            prompt_token_ids: prompt_token_ids.as_slice().into(),
            engine_request: EngineCoreRequest {
                request_id: engine_request_id,
                prompt_token_ids: Some(prompt_token_ids.into()),
                mm_features,
                sampling_params: Some(sampling_params),
                pooling_params: None,
                arrival_time: arrival_time.unwrap_or_else(current_unix_timestamp_secs),
                lora_request,
                cache_salt,
                data_parallel_rank,
                prompt_embeds: None,
                prompt_is_token_ids: None,
                client_index: 0,
                current_wave: 0,
                priority,
                trace_headers,
                resumable: false,
                session_id,
                kv_hints,
                external_req_id: Some(external_request_id),
                reasoning_ended,
                reasoning_parser_kwargs,
                abort_immediately: false,
                // Computed by the engine-core client, which knows the engine's
                // prefix hash config.
                block_hashes: None,
            },
        })
    }
}

/// The prefix that scopes a request's multimodal encoder-cache identifiers to
/// its LoRA adapter, or to the base model without one, when the engine applies
/// LoRA to the multimodal tower and connector.
///
/// Python prefixes only adapter requests, with `{lora_name}:`. Tagging the base
/// model too keeps a client `uuid` from spelling an adapter's identifier
/// (vllm-project/vllm#66885), and the digest of the adapter's name and path
/// keeps names with `:` unambiguous and starts a fresh scope when a name is
/// reloaded from another path.
fn tower_connector_lora_scope(lora_request: Option<&LoraRequest>) -> String {
    let Some(lora) = lora_request else {
        return "base:".to_owned();
    };
    let adapter = serde_json::to_vec(&(&lora.lora_name, &lora.lora_path))
        .expect("a pair of strings serializes");
    format!("lora:{}:", hex::encode(Sha256::digest(adapter)))
}

/// The remote-prefill fields of `kv_transfer_params` read by the frontend.
///
/// The prefiller's scheduler fills `remote_prefill_cached_tokens` next to
/// `do_remote_prefill` in the parameters it returns, which the P/D proxy then
/// forwards to the decoder.
#[derive(Deserialize)]
struct RemotePrefillParams {
    #[serde(default)]
    do_remote_prefill: bool,
    remote_prefill_cached_tokens: Option<usize>,
}

impl PreparedGenerateRequest {
    /// Return the prefiller's cache hits when this is the decode side of a
    /// P/D request, so response usage reports them instead of the transferred
    /// KV that the decoder counts as cached.
    pub fn remote_prefill_cached_tokens(&self) -> Option<usize> {
        let params = (self.engine_request.sampling_params.as_ref()?.extra_args.as_ref()?)
            .get("kv_transfer_params")?;
        let params = RemotePrefillParams::deserialize(params).ok()?;
        if params.do_remote_prefill {
            params.remote_prefill_cached_tokens
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use expect_test::expect;
    use serde_json::json;
    use vllm_engine_core_client::protocol::kv_hints::{KvHintAction, KvHintsEnvelope};
    use vllm_engine_core_client::protocol::lora::LoraRequest;
    use vllm_engine_core_client::protocol::multimodal::{
        MmFeatureSpec, MmModality, PlaceholderRange,
    };
    use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
    use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
    use vllm_engine_core_client::test_utils::decode_prompt_token_ids;

    use super::GenerateRequest;
    use crate::error::Error;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: vec![11, 22, 33],
            sampling_params: EngineCoreSamplingParams::for_test(),
            mm_features: None,
            arrival_time: Some(42.5),
            cache_salt: Some("salt".to_string()),
            trace_headers: Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )])),
            priority: 3,
            data_parallel_rank: Some(2),
            session_id: Some("session-1".to_string()),
            kv_hints: Some(KvHintsEnvelope {
                protocol_version: "0.1".to_string(),
                message_id: "msg-1".to_string(),
                actions: vec![KvHintAction {
                    action_id: "action-1".to_string(),
                    action_type: "example.action".to_string(),
                    action_version: "1.0".to_string(),
                    payload: BTreeMap::new(),
                }],
            }),
            reasoning_parser_kwargs: Some(ReasoningParserKwargs {
                chat_template_kwargs: [(
                    "chat_template_kwargs".to_string(),
                    serde_json::json!({
                        "enable_thinking": true,
                    }),
                )]
                .into(),
            }),
            reasoning_ended: None,
            lora_request: None,
        }
    }

    #[test]
    fn prepare_builds_engine_core_request() {
        let prepared = sample_request().prepare(true, false).unwrap();

        assert_eq!(&*prepared.prompt_token_ids, &[11, 22, 33]);

        let request = prepared.engine_request;
        assert_eq!(request.external_req_id.as_deref(), Some("req-1"));
        assert!(request.request_id.starts_with("req-1-"));
        assert_ne!(request.request_id, "req-1");
        assert_eq!(
            decode_prompt_token_ids(&request).as_deref(),
            Some(&[11, 22, 33][..])
        );
        assert_eq!(request.arrival_time, 42.5);
        assert_eq!(request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(request.data_parallel_rank, Some(2));
        assert_eq!(request.session_id.as_deref(), Some("session-1"));
        assert_eq!(
            request.kv_hints.as_ref().map(|hints| hints.message_id.as_str()),
            Some("msg-1")
        );
        assert_eq!(
            request.trace_headers,
            Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )]))
        );
        assert_eq!(request.reasoning_ended, None);
        assert_eq!(
            request
                .reasoning_parser_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.chat_template_kwargs.get("chat_template_kwargs")),
            Some(&serde_json::json!({
                "enable_thinking": true
            }))
        );
    }

    #[test]
    fn prepare_rejects_empty_prompt_tokens() {
        let mut request = sample_request();
        request.prompt_token_ids.clear();

        let error = request.prepare(true, false).unwrap_err();
        assert!(matches!(
            error,
            Error::EmptyPromptTokenIds { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn prepare_can_preserve_external_request_id() {
        let prepared = sample_request().prepare(false, false).unwrap();

        let request = prepared.engine_request;
        assert_eq!(request.external_req_id.as_deref(), Some("req-1"));
        assert_eq!(request.request_id, "req-1");
    }

    #[test]
    fn prepare_forwards_multimodal_features() {
        let mut request = sample_request();
        request.mm_features = Some(Vec::new());

        let prepared = request.prepare(false, false).unwrap();

        assert_eq!(prepared.engine_request.mm_features, Some(Vec::new()));
    }

    /// Under tower and connector LoRA, encoder outputs depend on the adapter:
    /// identifiers must differ between the base model and each adapter, and a
    /// client `uuid` must not reproduce another scope's identifier.
    #[test]
    fn prepare_scopes_mm_identifiers_under_tower_connector_lora() {
        fn identifier(
            tower_connector_lora: bool,
            lora: Option<(&str, &str)>,
            uuid_or_hash: &str,
        ) -> String {
            let mut request = sample_request();
            request.mm_features = Some(vec![MmFeatureSpec {
                data: None,
                modality: MmModality::Image,
                identifier: uuid_or_hash.to_owned(),
                mm_position: PlaceholderRange {
                    offset: 0,
                    length: 1,
                    is_embed: None,
                },
                mm_hash: Some("image-hash".to_owned()),
            }]);
            request.lora_request = lora.map(|(lora_name, lora_path)| LoraRequest {
                lora_name: lora_name.to_owned(),
                lora_int_id: 1,
                lora_path: lora_path.to_owned(),
                base_model_name: None,
                load_inplace: false,
                is_3d_lora_weight: false,
            });
            let features = request
                .prepare(false, tower_connector_lora)
                .unwrap()
                .engine_request
                .mm_features
                .unwrap();
            assert_eq!(features[0].mm_hash.as_deref(), Some("image-hash"));
            features[0].identifier.clone()
        }

        let adapter = Some(("adapter", "/loras/adapter"));
        let adapter_identifier = identifier(true, adapter, "image-hash");
        let cases = [
            ("disabled, base", identifier(false, None, "image-hash")),
            (
                "disabled, adapter",
                identifier(false, adapter, "image-hash"),
            ),
            ("base", identifier(true, None, "image-hash")),
            ("adapter", adapter_identifier.clone()),
            (
                "adapter reloaded from another path",
                identifier(true, Some(("adapter", "/loras/adapter-v2")), "image-hash"),
            ),
            (
                "base with the adapter's identifier as uuid",
                identifier(true, None, &adapter_identifier),
            ),
            (
                "adapter name with `:`",
                identifier(true, Some(("adapter:x", "/loras/adapter")), "image-hash"),
            ),
        ];
        expect![[r#"
            [
                (
                    "disabled, base",
                    "image-hash",
                ),
                (
                    "disabled, adapter",
                    "image-hash",
                ),
                (
                    "base",
                    "base:image-hash",
                ),
                (
                    "adapter",
                    "lora:24c444a79b194a3edec30e58a828b7151838268589307ba39c8bcb04e0a3b2de:image-hash",
                ),
                (
                    "adapter reloaded from another path",
                    "lora:139742af0370888d090d45924e0efa6a4fa1da73254777f73f9e308c0d4591d8:image-hash",
                ),
                (
                    "base with the adapter's identifier as uuid",
                    "base:lora:24c444a79b194a3edec30e58a828b7151838268589307ba39c8bcb04e0a3b2de:image-hash",
                ),
                (
                    "adapter name with `:`",
                    "lora:825a94fcc81c6b0f77c3f101da09d7b1701d3191706e5c4d29baced549c2059f:image-hash",
                ),
            ]
        "#]].assert_debug_eq(&cases);
    }

    #[test]
    fn remote_prefill_cached_tokens_requires_remote_prefill() {
        let cases = [
            (
                "positive",
                Some(json!({"do_remote_prefill": true, "remote_prefill_cached_tokens": 5})),
                Some(5),
            ),
            (
                "zero",
                Some(json!({"do_remote_prefill": true, "remote_prefill_cached_tokens": 0})),
                Some(0),
            ),
            (
                "missing_count",
                Some(json!({"do_remote_prefill": true})),
                None,
            ),
            (
                "false_flag",
                Some(json!({"do_remote_prefill": false, "remote_prefill_cached_tokens": 5})),
                None,
            ),
            (
                "missing_flag",
                Some(json!({"remote_prefill_cached_tokens": 5})),
                None,
            ),
            ("no_kv_transfer_params", None, None),
        ];
        for (name, kv_transfer_params, expected) in cases {
            let mut request = sample_request();
            request.sampling_params.extra_args = kv_transfer_params
                .map(|params| [("kv_transfer_params".to_string(), params)].into());
            let actual = request.prepare(false, false).unwrap().remote_prefill_cached_tokens();
            assert_eq!(actual, expected, "{name}");
        }
    }
}
