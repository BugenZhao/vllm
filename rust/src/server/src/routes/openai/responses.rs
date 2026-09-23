// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Stateless subset of the OpenAI Responses API.
//!
//! Every request carries the complete conversation history and produces one
//! model generation; tool calls are returned to the caller for execution.
//! Response storage, `previous_response_id`, conversations, server-side tool
//! execution, and context compaction are out of scope.

mod convert;
mod custom_tool;
mod output;
mod types;

use std::convert::Infallible;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use vllm_chat::{ChatEvent, ChatEventStreamTrait, CollectedAssistantMessage};

use self::convert::prepare_responses_request;
use self::custom_tool::CustomToolNames;
use self::output::{StreamAssembler, fail_response, finish_response, output_items};
pub(crate) use self::types::ResponsesRequest;
use self::types::{ResponseObject, ResponseStatus, SequencedEvent, StreamEvent};
use crate::error::{ApiError, chat_submit_error, server_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// Validate one Responses request and serve it through the shared `vllm-chat`
/// stack.
pub async fn responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;

    let prepared = match prepare_responses_request(
        body,
        &lora_resolution,
        request_context,
        unix_timestamp(),
    ) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "responses",
        request_id = %prepared.response.id,
        engine_request_id = tracing::field::Empty,
    );
    let enable_log_requests = state.api_server_options.enable_log_requests;

    let chat_stream =
        match state.chat.chat(prepared.chat_request).instrument(request_span.clone()).await {
            Ok(stream) => stream,
            Err(error) => {
                return chat_submit_error("failed to submit responses request", error)
                    .into_response();
            }
        };

    if stream {
        let event_stream = response_event_stream(
            chat_stream,
            prepared.response,
            prepared.custom_tools,
            enable_log_requests,
        );
        Sse::new(response_sse_stream(event_stream).instrument(request_span)).into_response()
    } else {
        match collect_response(
            chat_stream.collect_message(),
            prepared.response,
            &prepared.custom_tools,
            enable_log_requests,
        )
        .instrument(request_span)
        .await
        {
            Ok(response) => Json(response).into_response(),
            Err(error) => error.into_response(),
        }
    }
}

/// Collect one generation into a complete response object.
///
/// Failed generations are reported as HTTP errors, since no response has been
/// sent yet.
async fn collect_response(
    collected: impl Future<Output = vllm_chat::Result<CollectedAssistantMessage>>,
    mut response: ResponseObject,
    custom_tools: &CustomToolNames,
    enable_log_requests: bool,
) -> Result<ResponseObject, ApiError> {
    let CollectedAssistantMessage {
        message,
        usage,
        finish_reason,
        ..
    } = collected.await.map_err(|error| {
        server_error!("failed to collect response: {}", error.to_report_string())
    })?;
    finish_response(
        &mut response,
        output_items(message.content, custom_tools),
        usage,
        &finish_reason,
    );
    log_finished(&response, enable_log_requests);
    if response.status == ResponseStatus::Failed {
        let message = response.error.map(|error| error.message).unwrap_or_default();
        return Err(server_error!("{message}"));
    }
    Ok(response)
}

/// Convert one chat event stream into Responses stream events.
///
/// The stream always ends with exactly one terminal lifecycle event:
/// `response.completed`, `response.incomplete`, or `response.failed`. Errors
/// after streaming has started are reported through `response.failed`.
#[try_stream]
async fn response_event_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
    mut response: ResponseObject,
    custom_tools: CustomToolNames,
    enable_log_requests: bool,
    mut y: TryYielder<StreamEvent, Infallible>,
) -> Result<(), Infallible> {
    y.yield_ok(StreamEvent::Created {
        response: response.clone(),
    })
    .await;
    y.yield_ok(StreamEvent::InProgress {
        response: response.clone(),
    })
    .await;

    let mut assembler = StreamAssembler::new(custom_tools);
    let mut events = Vec::new();
    while let Some(next) = stream.next().await {
        match next {
            Ok(ChatEvent::Done {
                usage,
                finish_reason,
                ..
            }) => {
                finish_response(
                    &mut response,
                    assembler.take_output(),
                    usage,
                    &finish_reason,
                );
                log_finished(&response, enable_log_requests);
                y.yield_ok(StreamEvent::terminal(response)).await;
                return Ok(());
            }
            Ok(event) => {
                assembler.push(event, &mut events);
                for event in events.drain(..) {
                    y.yield_ok(event).await;
                }
            }
            Err(error) => {
                error!(error = %error.as_report(), "responses stream failed");
                fail_response(
                    &mut response,
                    assembler.take_output(),
                    error.to_report_string(),
                );
                y.yield_ok(StreamEvent::Failed { response }).await;
                return Ok(());
            }
        }
    }
    unreachable!("chat event streams end with `Done` or an error")
}

/// Serialize Responses stream events into SSE events.
///
/// Each SSE event is named after its payload `type` and carries its sequence
/// number. The stream ends after the terminal lifecycle event, with no
/// `[DONE]` sentinel.
fn response_sse_stream(
    events: impl Stream<Item = Result<StreamEvent, Infallible>>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    events.enumerate().map(|(sequence_number, event)| {
        let event = event?;
        let sequenced = SequencedEvent {
            event: &event,
            sequence_number: sequence_number as u64,
        };
        trace!(?sequenced, "responses emitting event");
        Ok(Event::default()
            .event(event.name())
            .json_data(&sequenced)
            .expect("StreamEvent must serialize to JSON"))
    })
}

fn log_finished(response: &ResponseObject, enable_log_requests: bool) {
    if enable_log_requests {
        let usage = response.usage.as_ref();
        info!(
            model = %response.model,
            status = ?response.status,
            input_tokens = usage.map_or(0, |usage| usage.input_tokens),
            output_tokens = usage.map_or(0, |usage| usage.output_tokens),
            "response finished"
        );
    }
}
