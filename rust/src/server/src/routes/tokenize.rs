// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `POST /tokenize`, `POST /detokenize` and `GET /tokenizer_info` (root paths,
//! matching Python).
//!
//! Encode/decode runs entirely in-process via [`DynTokenizer`]; the inference
//! engine is not involved.

mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use serde_json::value::{RawValue, to_raw_value};
use thiserror_ext::AsReport as _;
use vllm_text::backend::hf::RawTokenizerConfig;

use crate::error::{ApiError, chat_submit_error, server_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::tokenize::types::{
    DetokenizeRequest, DetokenizeResponse, TokenizeChatRequest, TokenizeCompletionRequest,
    TokenizeRequest, TokenizeResponse,
};
use crate::state::AppState;
use crate::utils::resolve_base_request_id;

/// Match Python `tokenize-{base}` where base is `X-Request-Id` or a new UUID.
fn tokenize_request_id(headers: &HeaderMap) -> String {
    let base = resolve_base_request_id(
        headers.get("X-Request-Id").and_then(|value| value.to_str().ok()),
        None,
    );
    format!("tokenize-{base}")
}

/// Reject an unknown model name, matching the other handlers.
fn check_model(state: &AppState, model: Option<&str>) -> Result<(), ApiError> {
    if let Some(model) = model
        && !state.served_model_names().iter().any(|n| n == model)
    {
        return Err(ApiError::model_not_found(model.to_string()));
    }
    Ok(())
}

/// Build the `token_strs` vector when requested, via the tokenizer vocab.
fn token_strs(tokenizer: &vllm_text::tokenizer::DynTokenizer, ids: &[u32]) -> Vec<String> {
    // Unknown IDs yield "" — intentional; matches Python's convert_ids_to_tokens behaviour.
    ids.iter().map(|&id| tokenizer.id_to_token(id).unwrap_or_default()).collect()
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<TokenizeRequest>,
) -> Response {
    let request_id = tokenize_request_id(&headers);
    let tokenizer = state.chat.text().tokenizer();
    let max_model_len = state.chat.engine_core_client().max_model_len();

    let result = match body {
        TokenizeRequest::Completion(req) => tokenize_completion(&state, &request_id, req),
        TokenizeRequest::Chat(req) => tokenize_chat(&state, &request_id, req).await,
    };

    match result {
        Ok((tokens, want_strs)) => {
            let token_strs = want_strs.then(|| token_strs(&tokenizer, &tokens));
            Json(TokenizeResponse {
                count: tokens.len(),
                max_model_len,
                tokens,
                token_strs,
            })
            .into_response()
        }
        Err(error) => error.into_response(),
    }
}

fn tokenize_completion(
    state: &AppState,
    request_id: &str,
    req: TokenizeCompletionRequest,
) -> Result<(Vec<u32>, bool), ApiError> {
    check_model(state, req.model.as_deref())?;
    let return_token_strs = req.return_token_strs;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(req.into_text_request(request_id.to_string()))
        .map_err(|e| server_error!("tokenize failed: {}", e.to_report_string()))?;
    Ok((tokens, return_token_strs))
}

/// HTTP adapter for the chat-shaped `/tokenize` body.
///
/// Checks the model name, prepares without engine submission, and maps errors
/// to [`ApiError`].
async fn tokenize_chat(
    state: &AppState,
    request_id: &str,
    req: TokenizeChatRequest,
) -> Result<(Vec<u32>, bool), ApiError> {
    check_model(state, req.model.as_deref())?;
    let return_token_strs = req.return_token_strs;
    // `continue_final_message` / `add_generation_prompt` mutual exclusion is
    // enforced in `normalize_generation_prompt_mode` inside `into_chat_request`.
    let text_request = state
        .chat
        .request_processor()
        .prepare_for_tokenization(req.into_chat_request(request_id.to_string())?)
        .await
        .map_err(|error| chat_submit_error("tokenize failed", error))?;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(text_request)
        .map_err(|e| server_error!("tokenize failed: {}", e.to_report_string()))?;
    Ok((tokens, return_token_strs))
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<DetokenizeRequest>,
) -> Response {
    if let Err(error) = check_model(&state, body.model.as_deref()) {
        return error.into_response();
    }
    let tokenizer = state.chat.text().tokenizer();
    match tokenizer.decode(&body.tokens, /* skip_special_tokens = */ false) {
        Ok(prompt) => Json(DetokenizeResponse { prompt }).into_response(),
        Err(e) => server_error!("detokenize failed: {}", e.to_report_string()).into_response(),
    }
}

/// `GET /tokenizer_info`: the model's `tokenizer_config.json`, like Python's
/// `TokenizerInfoResponse`. Registered only with
/// `--enable-tokenizer-info-endpoint`.
pub async fn tokenizer_info(State(state): State<Arc<AppState>>) -> Response {
    match state.chat.tokenizer_config() {
        Some(config) => Json(tokenizer_info_response(config)).into_response(),
        None => server_error!("the loaded model has no tokenizer configuration").into_response(),
    }
}

/// Report `tokenizer_config.json` as Python does after Transformers loads it:
/// every `added_tokens_decoder` entry and every other object tagged
/// `"__type": "AddedToken"` becomes an `AddedToken`, which Python reports as
/// its `content`. Other values are passed through as written.
fn tokenizer_info_response(config: &RawTokenizerConfig) -> RawTokenizerConfig {
    config
        .iter()
        .map(|(key, raw)| {
            // Only values holding `AddedToken` objects go through `Value`;
            // the rest stay as written, since an integer beyond 64 bits would
            // lose precision there.
            let flattened = if key == "added_tokens_decoder" {
                match serde_json::from_str(raw.get()) {
                    Ok(Value::Object(tokens)) => Some(Value::Object(
                        tokens
                            .iter()
                            .map(|(id, token)| (id.clone(), added_token_content(token)))
                            .collect(),
                    )),
                    _ => None,
                }
            } else if raw.get().contains(r#""AddedToken""#) {
                serde_json::from_str(raw.get()).ok().map(|value| flatten_added_tokens(&value))
            } else {
                None
            };
            let raw = flattened.map_or_else(|| raw.clone(), |value| to_raw_json(&value));
            (key.clone(), raw)
        })
        .collect()
}

/// A JSON value as a raw value.
fn to_raw_json(value: &Value) -> Box<RawValue> {
    to_raw_value(value).expect("a JSON value serializes")
}

/// Replace each object tagged `"__type": "AddedToken"` with its content.
fn flatten_added_tokens(value: &Value) -> Value {
    match value {
        Value::Object(object)
            if object.get("__type").and_then(Value::as_str) == Some("AddedToken") =>
        {
            added_token_content(value)
        }
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), flatten_added_tokens(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(flatten_added_tokens).collect()),
        _ => value.clone(),
    }
}

/// The `content` of a serialized `AddedToken`, or the value itself if it has none.
fn added_token_content(token: &Value) -> Value {
    token.get("content").unwrap_or(token).clone()
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::tokenize_request_id;

    #[test]
    fn tokenize_request_id_prefers_x_request_id_header() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Request-Id", HeaderValue::from_static("client-req-1"));
        assert_eq!(tokenize_request_id(&headers), "tokenize-client-req-1");
    }

    #[test]
    fn tokenize_request_id_generates_uuid_when_header_missing() {
        let headers = HeaderMap::new();
        let id = tokenize_request_id(&headers);
        assert!(id.starts_with("tokenize-"));
        assert_ne!(id, "tokenize-");
    }
}
