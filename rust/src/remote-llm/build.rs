// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

fn main() {
    let generate = tonic_build::manual::Method::builder()
        .name("generate")
        .route_name("Generate")
        .input_type("vllm_llm::GenerateRequest")
        .output_type("vllm_llm::GenerateOutput")
        .codec_path("crate::codec::MsgpackCodec")
        .server_streaming()
        .build();
    let info = tonic_build::manual::Method::builder()
        .name("info")
        .route_name("Info")
        .input_type("crate::InfoRequest")
        .output_type("vllm_llm::LlmInfo")
        .codec_path("crate::codec::MsgpackCodec")
        .build();
    let abort = tonic_build::manual::Method::builder()
        .name("abort")
        .route_name("Abort")
        .input_type("crate::AbortRequest")
        .output_type("crate::AbortResponse")
        .codec_path("crate::codec::MsgpackCodec")
        .build();
    let service = tonic_build::manual::Service::builder()
        .name("Llm")
        .package("vllm.internal")
        .comment("The `Llm` of a vLLM Rust frontend, served to other frontends.")
        .method(info)
        .method(generate)
        .method(abort)
        .build();
    tonic_build::manual::Builder::new().compile(&[service]);
}
