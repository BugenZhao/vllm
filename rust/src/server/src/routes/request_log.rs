// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request body logging for `--enable-log-requests`, mirroring Python's
//! `RequestLogger.log_request_body`.

use serde::Serialize;
use thiserror_ext::AsReport as _;
use tracing::{Level, debug, enabled};
use vllm_text::truncate_log_text;

use crate::config::ApiServerOptions;

impl ApiServerOptions {
    /// Log a request's JSON body at DEBUG, like Python's
    /// `RequestLogger.log_request_body` at its default
    /// `VLLM_REQUEST_BODY_LOG_LEVEL`, truncated to `--max-log-len` characters.
    /// It shows the request before chat template rendering and lowering.
    pub(crate) fn log_request_body(self, request_id: &str, body: &impl Serialize) {
        if !self.enable_log_requests || !enabled!(Level::DEBUG) {
            return;
        }
        let body = serde_json::to_string(body).unwrap_or_else(|error| error.to_report_string());
        let body = truncate_log_text(&body, self.log_outputs.max_log_len);
        debug!(request_id, body, "request JSON body");
    }
}
