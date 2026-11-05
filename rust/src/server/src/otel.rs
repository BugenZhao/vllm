// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OTLP trace export for `--otlp-traces-endpoint`, the counterpart of Python's
//! `init_tracer` (`vllm/tracing/otel.py`).

use std::str::FromStr;

use anyhow::{Context as _, Result, bail};
use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig as _};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider};
use tracing::warn;

/// Instrumenting module name of the frontend tracer, as in Python.
const INSTRUMENTING_MODULE_NAME: &str = "vllm.llm_engine";

/// The OTLP transport, chosen by `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` like in
/// Python, where it defaults to gRPC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OtlpProtocol {
    Grpc,
    HttpProtobuf,
}

impl FromStr for OtlpProtocol {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "grpc" => Ok(Self::Grpc),
            "http/protobuf" => Ok(Self::HttpProtobuf),
            other => bail!("Unsupported OTLP protocol '{other}' is configured"),
        }
    }
}

impl OtlpProtocol {
    fn from_env() -> Result<Self> {
        std::env::var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").map_or(Ok(Self::Grpc), |s| s.parse())
    }
}

/// The frontend's trace provider, exporting spans in batches to one OTLP
/// endpoint.
pub(crate) struct OtlpTracing {
    provider: SdkTracerProvider,
}

impl OtlpTracing {
    /// Build the exporter for `endpoint`, failing on an unsupported protocol
    /// or endpoint.
    pub(crate) fn init(endpoint: &str) -> Result<Self> {
        Self::init_with_protocol(endpoint, OtlpProtocol::from_env()?)
    }

    fn init_with_protocol(endpoint: &str, protocol: OtlpProtocol) -> Result<Self> {
        let exporter = match protocol {
            // Python's gRPC exporter is insecure, so a bare `host:port` means
            // plaintext.
            OtlpProtocol::Grpc => SpanExporter::builder()
                .with_tonic()
                .with_endpoint(if endpoint.contains("://") {
                    endpoint.to_owned()
                } else {
                    format!("http://{endpoint}")
                })
                .build(),
            OtlpProtocol::HttpProtobuf => SpanExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_endpoint(endpoint)
                .build(),
        }
        .with_context(|| format!("failed to build the OTLP span exporter for `{endpoint}`"))?;

        let resource = Resource::builder()
            .with_attributes([
                KeyValue::new("vllm.instrumenting_module_name", INSTRUMENTING_MODULE_NAME),
                KeyValue::new("vllm.process_id", std::process::id().to_string()),
            ])
            .build();
        let provider = SdkTracerProvider::builder()
            .with_resource(resource)
            .with_batch_exporter(exporter)
            .build();
        Ok(Self { provider })
    }

    pub(crate) fn tracer(&self) -> SdkTracer {
        self.provider.tracer(INSTRUMENTING_MODULE_NAME)
    }

    /// Export the spans still buffered and stop the exporter.
    pub(crate) async fn shutdown(self) {
        // The SDK bounds the blocking shutdown with its own timeout.
        let provider = self.provider;
        let result = tokio::task::spawn_blocking(move || provider.shutdown()).await;
        if let Ok(Err(error)) = result {
            warn!(%error, "failed to shut down the OpenTelemetry trace exporter");
        }
    }
}

#[cfg(test)]
mod tests;
