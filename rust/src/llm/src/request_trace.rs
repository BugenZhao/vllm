// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The per-request `llm_request` OpenTelemetry span.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use opentelemetry::propagation::TextMapPropagator as _;
use opentelemetry::trace::{Span as _, SpanKind, Tracer as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracer;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;

use crate::request_metrics::RequestMetricsTracker;

/// Span attribute names, copied from Python's `SpanAttributes`
/// (`vllm/tracing/utils.py`).
///
/// TODO: move to the current OTel GenAI semantic conventions (for example
/// `gen_ai.usage.input_tokens`, `gen_ai.request.choice.count` and the required
/// `gen_ai.operation.name`) together with the Python frontend.
mod attributes {
    pub const USAGE_COMPLETION_TOKENS: &str = "gen_ai.usage.completion_tokens";
    pub const USAGE_PROMPT_TOKENS: &str = "gen_ai.usage.prompt_tokens";
    pub const REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
    pub const REQUEST_TOP_P: &str = "gen_ai.request.top_p";
    pub const REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
    pub const REQUEST_ID: &str = "gen_ai.request.id";
    pub const REQUEST_N: &str = "gen_ai.request.n";
    pub const LATENCY_TIME_IN_QUEUE: &str = "gen_ai.latency.time_in_queue";
    pub const LATENCY_TIME_TO_FIRST_TOKEN: &str = "gen_ai.latency.time_to_first_token";
    pub const LATENCY_E2E: &str = "gen_ai.latency.e2e";
    pub const LATENCY_TIME_IN_MODEL_PREFILL: &str = "gen_ai.latency.time_in_model_prefill";
    pub const LATENCY_TIME_IN_MODEL_DECODE: &str = "gen_ai.latency.time_in_model_decode";
    pub const LATENCY_TIME_IN_MODEL_INFERENCE: &str = "gen_ai.latency.time_in_model_inference";
}

/// What one request's `llm_request` span reports besides its timings, fixed
/// when the request is submitted.
pub(crate) struct RequestTrace {
    tracer: SdkTracer,
    /// Parsed from the client's W3C trace headers; empty for a root span.
    parent: Context,
    /// External (user-facing) request ID.
    request_id: String,
    max_tokens: u32,
    top_p: f32,
    temperature: f32,
    n: u32,
}

impl RequestTrace {
    /// Capture the span's request parameters from one submitted engine
    /// request, one of the `n` samples of the client's request.
    pub(crate) fn new(
        tracer: SdkTracer,
        parent: Context,
        engine_request: &EngineCoreRequest,
        n: u32,
    ) -> Self {
        let sampling_params = (engine_request.sampling_params.as_ref())
            .expect("generate requests always carry sampling params");
        Self {
            tracer,
            parent,
            request_id: (engine_request.external_req_id.clone())
                .expect("prepare always sets external_req_id"),
            max_tokens: sampling_params.max_tokens,
            top_p: sampling_params.top_p,
            temperature: sampling_params.temperature,
            n,
        }
    }

    /// Export the span of a request that finished at `finished_at` (Unix
    /// seconds), like Python's `OutputProcessor.do_tracing`: it starts at
    /// arrival and is a server span under the client's trace context.
    pub(crate) fn emit(self, stats: &RequestMetricsTracker, finished_at: f64) {
        use attributes::*;

        let stats = stats.span_stats(finished_at);
        let mut attributes = vec![
            KeyValue::new(LATENCY_TIME_TO_FIRST_TOKEN, stats.time_to_first_token),
            KeyValue::new(LATENCY_E2E, stats.e2e),
            KeyValue::new(LATENCY_TIME_IN_QUEUE, stats.queued),
            KeyValue::new(USAGE_PROMPT_TOKENS, i64::from(stats.prompt_tokens)),
            KeyValue::new(USAGE_COMPLETION_TOKENS, i64::from(stats.completion_tokens)),
            KeyValue::new(LATENCY_TIME_IN_MODEL_PREFILL, stats.prefill),
            KeyValue::new(LATENCY_TIME_IN_MODEL_DECODE, stats.decode),
            KeyValue::new(LATENCY_TIME_IN_MODEL_INFERENCE, stats.inference),
            KeyValue::new(REQUEST_ID, self.request_id),
            KeyValue::new(REQUEST_TOP_P, f64::from(self.top_p)),
            KeyValue::new(REQUEST_MAX_TOKENS, i64::from(self.max_tokens)),
            KeyValue::new(REQUEST_N, i64::from(self.n)),
        ];
        // Python sets each request parameter only when it is truthy; of these,
        // only the temperature can be zero.
        if self.temperature != 0.0 {
            attributes.push(KeyValue::new(
                REQUEST_TEMPERATURE,
                f64::from(self.temperature),
            ));
        }

        let mut span = self
            .tracer
            .span_builder("llm_request")
            .with_kind(SpanKind::Server)
            .with_start_time(unix_time(stats.arrival_time))
            .with_attributes(attributes)
            .start_with_context(&self.tracer, &self.parent);
        span.end_with_timestamp(unix_time(finished_at));
    }
}

/// Parse the parent span context from W3C trace headers (`traceparent`,
/// `tracestate`), like Python's `extract_trace_context`.
pub(crate) fn extract_trace_context(trace_headers: &BTreeMap<String, String>) -> Context {
    let carrier: HashMap<String, String> = trace_headers.clone().into_iter().collect();
    TraceContextPropagator::new().extract(&carrier)
}

fn unix_time(seconds: f64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs_f64(seconds.max(0.0))
}
