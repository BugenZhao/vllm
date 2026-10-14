// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! SageMaker `/invocations` endpoint.
//!
//! Mirrors Python `vllm/entrypoints/serve/sagemaker/api_router.py`: the body is
//! routed to the first generate endpoint whose request type accepts it, chat
//! completions before completions.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse as _, Response};
use serde_json::Value;

use crate::error::ApiError;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::{self, ChatCompletionRequest, CompletionRequest};
use crate::state::AppState;

/// SageMaker header naming the LoRA adapter to serve the invocation with.
const ADAPTER_IDENTIFIER_HEADER: &str = "X-Amzn-SageMaker-Adapter-Identifier";

pub async fn invocations(
    state: State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Response {
    // Like Python's `inject_adapter_id(adapter_path="model")`: the adapter
    // header replaces the requested model.
    if let Some(adapter) =
        headers.get(ADAPTER_IDENTIFIER_HEADER).and_then(|value| value.to_str().ok())
        && let Some(body) = body.as_object_mut()
    {
        body.insert("model".to_string(), Value::from(adapter));
    }

    // A body that fails to deserialize or validate as one request type falls
    // through to the next, as with Python's pydantic validators.
    if let Ok(request) = ValidatedJson::<ChatCompletionRequest>::from_value(body.clone()) {
        return openai::chat_completions(state, headers, request).await;
    }
    if let Ok(request) = ValidatedJson::<CompletionRequest>::from_value(body) {
        return openai::completions(state, headers, request).await;
    }

    ApiError::invalid_request(
        "Cannot find suitable handler for request. Expected one of: \
         ['ChatCompletionRequest', 'CompletionRequest']",
        None,
    )
    .into_response()
}
