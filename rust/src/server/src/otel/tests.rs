// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::routing::post;
use opentelemetry::trace::{Span as _, Tracer as _};
use opentelemetry_proto::tonic::collector::trace::v1::trace_service_server::{
    TraceService, TraceServiceServer,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use opentelemetry_proto::tonic::common::v1::any_value::Value;
use prost::Message as _;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tonic::transport::server::TcpIncoming;
use tonic::{Request, Response, Status};

use super::{OtlpProtocol, OtlpTracing};

type Exports = mpsc::UnboundedSender<ExportTraceServiceRequest>;

/// In-process OTLP gRPC collector forwarding each export to a channel.
struct GrpcCollector(Exports);

#[tonic::async_trait]
impl TraceService for GrpcCollector {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        let _ = self.0.send(request.into_inner());
        Ok(Response::new(ExportTraceServiceResponse::default()))
    }
}

/// Start a collector for `protocol` and return the endpoint to export to.
async fn start_collector(protocol: OtlpProtocol, exports: Exports) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    match protocol {
        OtlpProtocol::Grpc => {
            tokio::spawn(
                tonic::transport::Server::builder()
                    .add_service(TraceServiceServer::new(GrpcCollector(exports)))
                    .serve_with_incoming(TcpIncoming::from(listener)),
            );
            // A bare `host:port`, as Python's gRPC exporter accepts.
            addr.to_string()
        }
        OtlpProtocol::HttpProtobuf => {
            async fn export(State(exports): State<Exports>, body: Bytes) -> Vec<u8> {
                let _ = exports.send(ExportTraceServiceRequest::decode(body).unwrap());
                ExportTraceServiceResponse::default().encode_to_vec()
            }
            let app = Router::new().route("/v1/traces", post(export)).with_state(exports);
            tokio::spawn(async move { axum::serve(listener, app).await });
            format!("http://{addr}/v1/traces")
        }
    }
}

/// Exporting through either protocol delivers the span, buffered by the batch
/// processor until shutdown, with Python's resource attributes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_exports_buffered_spans_over_each_protocol() {
    for protocol in [OtlpProtocol::Grpc, OtlpProtocol::HttpProtobuf] {
        let (exports, mut received) = mpsc::unbounded_channel();
        let endpoint = start_collector(protocol, exports).await;

        let tracing = OtlpTracing::init_with_protocol(&endpoint, protocol).unwrap();
        tracing.tracer().start("llm_request").end();
        tracing.shutdown().await;

        let export = tokio::time::timeout(Duration::from_secs(10), received.recv())
            .await
            .unwrap_or_else(|_| panic!("{protocol:?}: no export"))
            .unwrap();
        let [resource_spans] = export.resource_spans.as_slice() else {
            panic!("{protocol:?}: {export:?}");
        };
        let resource = resource_spans.resource.as_ref().unwrap();
        let module = (resource.attributes.iter())
            .find(|kv| kv.key == "vllm.instrumenting_module_name")
            .and_then(|kv| kv.value.as_ref()?.value.clone());
        assert_eq!(
            module,
            Some(Value::StringValue("vllm.llm_engine".to_string())),
            "{protocol:?}"
        );
        let names: Vec<_> = (resource_spans.scope_spans.iter())
            .flat_map(|scope| &scope.spans)
            .map(|span| span.name.as_str())
            .collect();
        assert_eq!(names, ["llm_request"], "{protocol:?}");
    }
}

#[test]
fn protocol_names_match_python() {
    assert_eq!("grpc".parse::<OtlpProtocol>().unwrap(), OtlpProtocol::Grpc);
    assert_eq!(
        "http/protobuf".parse::<OtlpProtocol>().unwrap(),
        OtlpProtocol::HttpProtobuf
    );
    let error = "http/json".parse::<OtlpProtocol>().unwrap_err();
    assert_eq!(
        error.to_string(),
        "Unsupported OTLP protocol 'http/json' is configured"
    );
}
