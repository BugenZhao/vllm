// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Error-response log for `--log-error-stack`.

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tracing::error;

use crate::error::ApiError;
use crate::routes::openai::utils::types::ErrorDetail;

/// Log each [`ApiError`] response at ERROR, like Python's exception handlers
/// under `--log-error-stack`.
///
/// Rust has no stack trace to print; the logged error is the one the client
/// receives, which for internal errors carries the error chain. An error sent
/// inside an already started stream is not an error response, so, as in
/// Python, it is not logged here. Unlike Python, which skips its
/// `GenerationError`s, engine-reported request failures are logged too.
pub async fn log_error_response(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let response = next.run(request).await;
    if let Some(error) = response.extensions().get::<ApiError>() {
        let ErrorDetail {
            message,
            error_type,
            param,
            code,
        } = error.to_error_response().error;
        error!(
            %method,
            path,
            status = code,
            error_type,
            param,
            error = message,
            "Error response"
        );
    }
    response
}
