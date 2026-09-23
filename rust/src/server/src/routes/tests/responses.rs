// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Route tests for the stateless `/v1/responses` subset.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use expect_test::{Expect, expect};
use serde_json::{Value, json};
use serial_test::serial;
use tower::Service as _;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;

use super::{
    FakeChatBackend, bytes_to_token_ids, test_app_with_backend_and_stream_output_specs,
    weather_tool_call_output_specs,
};

/// Engine outputs generating `text`, then one EOS token that is not decoded.
fn generate(chunks: &[&str]) -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
    (chunks.iter())
        .map(|chunk| (bytes_to_token_ids(chunk.as_bytes()), None))
        .chain([(vec![0], Some(EngineCoreFinishReason::Stop))])
        .collect()
}

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    })
}

fn apply_patch_tool() -> Value {
    json!({
        "type": "custom",
        "name": "apply_patch",
        "description": "Edit files. This is a FREEFORM tool.",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: begin_patch hunk+ end_patch"},
    })
}

/// Serve one request against a Qwen3-parsed fake model that generates the
/// given engine outputs, and return the status and body.
async fn call_responses(
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    body: Value,
) -> (StatusCode, String) {
    let (mut app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        output_specs,
    )
    .await;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .header("x-request-id", "test")
                .body(Body::from(body.to_string()))
                .expect("build request"),
        )
        .await
        .expect("call app");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    if status == StatusCode::OK {
        engine_task.await.expect("mock engine task");
    }
    (status, String::from_utf8(body.to_vec()).expect("utf8 body"))
}

/// Replace generated IDs and timestamps with stable placeholders.
fn normalize(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object.iter_mut() {
                match (key.as_str(), &*value) {
                    ("id" | "item_id" | "call_id", Value::String(id)) if id.contains('_') => {
                        let prefix = id.split('_').next().unwrap();
                        if prefix != "resp" {
                            *value = Value::from(format!("{prefix}_*"));
                        }
                    }
                    ("created_at", _) => *value = Value::from(0),
                    _ => normalize(value),
                }
            }
        }
        Value::Array(values) => values.iter_mut().for_each(normalize),
        _ => {}
    }
}

/// Parse an SSE body into `(event name, data)` pairs, checking that each event
/// is named after its payload type and sequence numbers count up from zero.
fn parse_sse(text: &str) -> Vec<Value> {
    text.split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .enumerate()
        .map(|(index, frame)| {
            let mut name = None;
            let mut data = None;
            for line in frame.lines() {
                if let Some(value) = line.strip_prefix("event: ") {
                    name = Some(value);
                } else if let Some(value) = line.strip_prefix("data: ") {
                    data = Some(serde_json::from_str::<Value>(value).expect("json data"));
                }
            }
            let data = data.unwrap_or_else(|| panic!("frame without data: {frame:?}"));
            assert_eq!(Some(data["type"].as_str().unwrap()), name, "{frame}");
            assert_eq!(data["sequence_number"], index, "{frame}");
            data
        })
        .collect()
}

/// Check the stream-level invariants shared by every streamed response, and
/// return the terminal response.
fn check_stream(events: &[Value], terminal_type: &str, expected_types: Expect) -> Value {
    let types: Vec<_> = events.iter().map(|event| event["type"].as_str().unwrap()).collect();
    expected_types.assert_debug_eq(&types);

    let terminal = events.last().expect("terminal event");
    assert_eq!(terminal["type"], terminal_type);
    let response = terminal["response"].clone();

    // Items streamed through `output_item.done` are exactly the final output,
    // and `output_item.added` announced the same item IDs in the same slots.
    let done: Vec<_> = (events.iter())
        .filter(|event| event["type"] == "response.output_item.done")
        .map(|event| {
            assert_eq!(
                event["item"],
                response["output"][event["output_index"].as_u64().unwrap() as usize]
            );
            event["item"].clone()
        })
        .collect();
    assert_eq!(Value::from(done), response["output"]);
    let added: Vec<_> = (events.iter())
        .filter(|event| event["type"] == "response.output_item.added")
        .map(|event| event["item"]["id"].clone())
        .collect();
    let output_ids: Vec<_> = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].clone())
        .collect();
    assert_eq!(added, output_ids);
    response
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_returns_reasoning_and_message_items() {
    let (status, body) = call_responses(
        generate(&["<think>Plan.</think>", "Hi!"]),
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "instructions": "Be brief.",
            "input": "Hello",
            "reasoning": {"effort": "high"},
            "metadata": {"trace": "1"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let mut response: Value = serde_json::from_str(&body).unwrap();
    normalize(&mut response);
    expect![[r#"
        {
          "id": "resp_test",
          "object": "response",
          "created_at": 0,
          "status": "completed",
          "error": null,
          "incomplete_details": null,
          "instructions": "Be brief.",
          "max_output_tokens": null,
          "model": "Qwen/Qwen1.5-0.5B-Chat",
          "output": [
            {
              "type": "reasoning",
              "id": "rs_*",
              "status": "completed",
              "summary": [],
              "content": [
                {
                  "type": "reasoning_text",
                  "text": "Plan."
                }
              ]
            },
            {
              "type": "message",
              "id": "msg_*",
              "role": "assistant",
              "status": "completed",
              "content": [
                {
                  "type": "output_text",
                  "text": "Hi!",
                  "annotations": [],
                  "logprobs": []
                }
              ]
            }
          ],
          "parallel_tool_calls": true,
          "previous_response_id": null,
          "reasoning": {
            "effort": "high",
            "summary": null
          },
          "store": false,
          "temperature": null,
          "text": {
            "format": {
              "type": "text"
            }
          },
          "tool_choice": "auto",
          "tools": [],
          "top_p": null,
          "truncation": "disabled",
          "usage": {
            "input_tokens": 40,
            "input_tokens_details": {
              "cached_tokens": 0
            },
            "output_tokens": 24,
            "output_tokens_details": {
              "reasoning_tokens": 5
            },
            "total_tokens": 64
          },
          "user": null,
          "metadata": {
            "trace": "1"
          }
        }"#]]
    .assert_eq(&serde_json::to_string_pretty(&response).unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_stream_emits_function_call_events() {
    let (status, body) = call_responses(
        weather_tool_call_output_specs(),
        json!({
            "input": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [weather_tool()],
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let mut events = parse_sse(&body);
    events.iter_mut().for_each(normalize);
    let response = check_stream(
        &events,
        "response.completed",
        expect![[r#"
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        "#]],
    );
    assert_eq!(events[0]["response"]["status"], "in_progress");
    assert!(events[0]["response"]["usage"].is_null());
    expect![[r#"
        [
          {
            "type": "reasoning",
            "id": "rs_*",
            "status": "completed",
            "summary": [],
            "content": [
              {
                "type": "reasoning_text",
                "text": "Need tool."
              }
            ]
          },
          {
            "type": "function_call",
            "id": "fc_*",
            "status": "completed",
            "call_id": "call_*",
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}"
          },
          {
            "type": "message",
            "id": "msg_*",
            "role": "assistant",
            "status": "completed",
            "content": [
              {
                "type": "output_text",
                "text": "}\n</tool_call",
                "annotations": [],
                "logprobs": []
              }
            ]
          }
        ]"#]]
    .assert_eq(&serde_json::to_string_pretty(&response["output"]).unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_raise_custom_tool_calls_from_function_shims() {
    let patch = "*** Begin Patch\n*** Add File: a.txt\n+say \"hi\"\n*** End Patch\n";
    let call = json!({"name": "apply_patch", "arguments": {"input": patch}});
    let output = format!("<tool_call>\n{call}\n</tool_call>");
    // Split the generation so arguments stream in several deltas, some of
    // them cutting through escape sequences.
    let chunks: Vec<String> = (output.as_bytes().chunks(7))
        .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap())
        .collect();
    let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();

    for stream in [false, true] {
        let (status, body) = call_responses(
            generate(&chunks),
            json!({
                "input": "Add a.txt.",
                "tools": [weather_tool(), apply_patch_tool()],
                "stream": stream,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");

        let response = if stream {
            let mut events = parse_sse(&body);
            events.iter_mut().for_each(normalize);
            let input_done = (events.iter())
                .find(|event| event["type"] == "response.custom_tool_call_input.done")
                .expect("custom tool call input done");
            assert_eq!(input_done["input"], patch);
            let deltas: Vec<_> = (events.iter())
                .filter(|event| event["type"] == "response.custom_tool_call_input.delta")
                .map(|event| event["delta"].as_str().unwrap())
                .collect();
            assert!(deltas.len() > 1, "input was not streamed: {deltas:?}");
            assert_eq!(deltas.concat(), patch);
            events.retain(|event| event["type"] != "response.custom_tool_call_input.delta");
            check_stream(
                &events,
                "response.completed",
                expect![[r#"
                    [
                        "response.created",
                        "response.in_progress",
                        "response.output_item.added",
                        "response.custom_tool_call_input.done",
                        "response.output_item.done",
                        "response.completed",
                    ]
                "#]],
            )
        } else {
            let mut response: Value = serde_json::from_str(&body).unwrap();
            normalize(&mut response);
            response
        };

        assert_eq!(
            response["output"],
            json!([{
                "type": "custom_tool_call",
                "id": "ctc_*",
                "status": "completed",
                "call_id": "call_*",
                "name": "apply_patch",
                "input": patch,
            }]),
            "stream={stream}"
        );
        // The custom tool is echoed back as declared.
        assert_eq!(response["tools"][1], apply_patch_tool());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_truncated_by_max_output_tokens_are_incomplete() {
    let output_specs = || {
        vec![
            (bytes_to_token_ids(b"Hel"), None),
            (
                bytes_to_token_ids(b"lo"),
                Some(EngineCoreFinishReason::Length),
            ),
        ]
    };

    let (status, body) = call_responses(
        output_specs(),
        json!({"input": "Hello", "max_output_tokens": 5, "stream": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = parse_sse(&body);
    let response = check_stream(
        &events,
        "response.incomplete",
        expect![[r#"
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.incomplete",
            ]
        "#]],
    );
    assert_eq!(response["status"], "incomplete");
    assert_eq!(
        response["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
    assert_eq!(response["max_output_tokens"], 5);
    assert_eq!(response["output"][0]["content"][0]["text"], "Hello");

    let (status, body) = call_responses(
        output_specs(),
        json!({"input": "Hello", "max_output_tokens": 5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["status"], "incomplete");
    assert_eq!(
        response["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn responses_reject_stateful_requests_before_generation() {
    let (status, body) = call_responses(
        Vec::new(),
        json!({"input": "Continue.", "previous_response_id": "resp_1"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert_eq!(error["error"]["param"], "previous_response_id");
}
