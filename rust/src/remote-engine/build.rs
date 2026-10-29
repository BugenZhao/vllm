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
    let service = tonic_build::manual::Service::builder()
        .name("EngineRpc")
        .package("vllm.internal")
        .comment("The engine client of a vLLM Rust frontend, served to other frontends.")
        .method(generate)
        .build();
    tonic_build::manual::Builder::new().compile(&[service]);
}
