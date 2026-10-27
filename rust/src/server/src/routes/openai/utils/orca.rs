// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! ORCA endpoint load reports on non-streaming OpenAI responses, like Python's
//! `vllm/entrypoints/serve/utils/orca_metrics.py`.
//!
//! ORCA proto: <https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto>

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::Response;
use itertools::Itertools as _;
use serde_json::{Map, json};
use tracing::warn;
use vllm_engine_core_client::EngineCoreClient;
use vllm_metrics::{EngineLabels, METRICS};

/// Request header that asks for a load report and selects its format.
const ENDPOINT_LOAD_METRICS_FORMAT: &str = "endpoint-load-metrics-format";
/// Response header carrying the load report.
const ENDPOINT_LOAD_METRICS: HeaderName = HeaderName::from_static("endpoint-load-metrics");

/// Wire format of the load report, chosen case-insensitively by the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrcaFormat {
    Text,
    Json,
}

/// Attach the `endpoint-load-metrics` header when the request asks for one
/// with a supported `endpoint-load-metrics-format`.
pub(crate) fn attach_endpoint_load_metrics(
    response: &mut Response,
    request_headers: &HeaderMap,
    client: &EngineCoreClient,
) {
    let Some(format) = request_headers.get(ENDPOINT_LOAD_METRICS_FORMAT) else {
        return;
    };
    let format = match format.to_str().unwrap_or_default() {
        "" => return,
        format if format.eq_ignore_ascii_case("text") => OrcaFormat::Text,
        format if format.eq_ignore_ascii_case("json") => OrcaFormat::Json,
        format => {
            warn!(format, "unsupported ORCA endpoint-load-metrics format");
            return;
        }
    };
    let report = orca_report(format, &named_metrics(client));
    let value = HeaderValue::from_str(&report).expect("ORCA report is a valid header value");
    response.headers_mut().insert(ENDPOINT_LOAD_METRICS, value);
}

/// Read the exported gauges that Python maps into ORCA named metrics, one
/// sample per engine, in Prometheus registration order.
fn named_metrics(client: &EngineCoreClient) -> Vec<(&'static str, f64)> {
    let labels = client
        .engine_indices()
        .into_iter()
        .map(|engine| EngineLabels {
            model_name: client.model_name().to_string(),
            engine,
        })
        .collect_vec();
    let scheduler = &METRICS.scheduler;
    let waiting = labels.iter().filter_map(|labels| {
        let waiting = scheduler.scheduler_waiting.get(labels)?.get();
        Some(("num_requests_waiting", waiting as f64))
    });
    let kv_cache_usage = labels.iter().filter_map(|labels| {
        let usage = scheduler.kv_cache_usage.get(labels)?.get();
        Some(("kv_cache_usage_perc", usage))
    });
    waiting.chain(kv_cache_usage).collect()
}

/// Render the report like Python's `create_orca_header`. With several engines,
/// TEXT lists each engine's sample and JSON keeps the last one per name.
fn orca_report(format: OrcaFormat, named_metrics: &[(&str, f64)]) -> String {
    match format {
        // `{:?}` prints floats like Python's `repr`, e.g. `3.0` rather than `3`.
        OrcaFormat::Text => format!(
            "TEXT {}",
            named_metrics
                .iter()
                .map(|(name, value)| format!("named_metrics.{name}={value:?}"))
                .join(", ")
        ),
        OrcaFormat::Json => {
            let named_metrics: Map<_, _> = named_metrics
                .iter()
                .map(|(name, value)| (name.to_string(), json!(value)))
                .collect();
            format!("JSON {}", json!({ "named_metrics": named_metrics }))
        }
    }
}
