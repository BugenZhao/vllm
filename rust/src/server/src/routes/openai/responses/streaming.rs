// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Streaming event state machine for the Responses API.
//!
//! Maps the structured `vllm-chat` event stream (`ChatEvent`) onto the OpenAI
//! Responses SSE event protocol. Event shapes and emission order mirror the
//! simple (non-Harmony) streaming path of the Python frontend in
//! `vllm/entrypoints/openai/responses/streaming_events.py`, with one
//! deliberate improvement: item IDs stay consistent between streamed events
//! and the terminal response payload.

use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantToolCall, ChatEvent, FinishReason,
};

use super::types::{
    AssistantRole, ResponseItemStatus, ResponseOutputContentPart, ResponseOutputItem,
    ResponsesResponse, TextPart,
};
use crate::error::ApiError;
use crate::routes::openai::utils::logprobs::decoded_logprobs_to_openai_chat;
use crate::routes::openai::utils::types::ChatLogProbsContent;

/// One Responses API SSE event payload.
///
/// The wire `type` comes from [`Self::event_type`], which also names the SSE
/// `event:` line, so it is added by [`SequencedEvent`] rather than repeated
/// as a serde tag.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum ResponseStreamEvent {
    Created {
        response: Arc<ResponsesResponse>,
    },
    InProgress {
        response: Arc<ResponsesResponse>,
    },
    Completed {
        response: Arc<ResponsesResponse>,
    },
    Incomplete {
        response: Arc<ResponsesResponse>,
    },
    Failed {
        response: Arc<ResponsesResponse>,
    },
    OutputItemAdded {
        output_index: u32,
        item: ResponseOutputItem,
    },
    OutputItemDone {
        output_index: u32,
        item: ResponseOutputItem,
    },
    ContentPartAdded {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: ResponseStreamPart,
    },
    ContentPartDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: ResponseStreamPart,
    },
    OutputTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        /// Logprobs of the tokens of this delta's decoded update, when
        /// requested; empty otherwise.
        logprobs: Vec<TextDeltaLogprob>,
    },
    OutputTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
        /// Always empty, like Python; the content part carries the text's
        /// logprobs.
        logprobs: Vec<Value>,
    },
    ReasoningTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    ReasoningTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: u32,
        delta: String,
    },
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: u32,
        name: String,
        arguments: String,
    },
}

impl ResponseStreamEvent {
    /// Return the wire event type (used for `type` and the SSE `event:` line).
    pub(crate) fn event_type(&self) -> &'static str {
        match self {
            Self::Created { .. } => "response.created",
            Self::InProgress { .. } => "response.in_progress",
            Self::Completed { .. } => "response.completed",
            Self::Incomplete { .. } => "response.incomplete",
            Self::Failed { .. } => "response.failed",
            Self::OutputItemAdded { .. } => "response.output_item.added",
            Self::OutputItemDone { .. } => "response.output_item.done",
            Self::ContentPartAdded { .. } => "response.content_part.added",
            Self::ContentPartDone { .. } => "response.content_part.done",
            Self::OutputTextDelta { .. } => "response.output_text.delta",
            Self::OutputTextDone { .. } => "response.output_text.done",
            Self::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
            Self::ReasoningTextDone { .. } => "response.reasoning_text.done",
            Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
        }
    }
}

/// Wire form of one event: `type` and `sequence_number` next to the payload,
/// matching how the Python frontend serializes its typed SDK events.
/// `sequence_number` is assigned centrally by the SSE encoder for all events
/// of one request.
#[derive(Debug, Serialize)]
pub(crate) struct SequencedEvent<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    sequence_number: u64,
    #[serde(flatten)]
    event: &'a ResponseStreamEvent,
}

impl<'a> SequencedEvent<'a> {
    pub(crate) fn new(event: &'a ResponseStreamEvent, sequence_number: u64) -> Self {
        Self {
            event_type: event.event_type(),
            sequence_number,
            event,
        }
    }
}

/// One token's logprob on a `response.output_text.delta` event. The SDK's
/// delta shape, unlike the content part's, carries no `bytes`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct TextDeltaLogprob {
    token: String,
    logprob: f32,
    top_logprobs: Vec<TextDeltaTopLogprob>,
}

/// One alternative token of a [`TextDeltaLogprob`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct TextDeltaTopLogprob {
    token: String,
    logprob: f32,
}

impl From<&ChatLogProbsContent> for TextDeltaLogprob {
    fn from(content: &ChatLogProbsContent) -> Self {
        Self {
            token: content.token.clone(),
            logprob: content.logprob,
            top_logprobs: content
                .top_logprobs
                .iter()
                .map(|top| TextDeltaTopLogprob {
                    token: top.token.clone(),
                    logprob: top.logprob,
                })
                .collect(),
        }
    }
}

/// How to render output-text logprobs, when
/// `include=["message.output_text.logprobs"]` requests them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OutputLogprobsOptions {
    pub top_logprobs: i32,
    pub return_tokens_as_token_ids: bool,
}

/// Per-update state for attaching logprobs to output text.
///
/// Like Python, the logprobs of one decoded update go to that update's text
/// delta, and are dropped when the update outputs no text: per engine step
/// when streaming, and the whole output otherwise, since a non-streaming
/// request is one update. `vllm-chat` emits an update's `LogprobsDelta` after
/// its other events, so those events are held until it arrives.
struct OutputLogprobs {
    options: OutputLogprobsOptions,
    /// Chat events of the current decoded update.
    update: Vec<ChatEvent>,
    /// Logprobs of the update being processed, until its text takes them.
    update_logprobs: Vec<ChatLogProbsContent>,
}

impl OutputLogprobs {
    /// Hold one event of the current update. Text deltas of one block are
    /// merged so that the update's logprobs stay on a single delta event.
    fn hold(&mut self, event: &ChatEvent) {
        if let ChatEvent::BlockDelta {
            index,
            kind: AssistantBlockKind::Text,
            delta,
            token_count,
        } = event
            && let Some(ChatEvent::BlockDelta {
                index: held_index,
                kind: AssistantBlockKind::Text,
                delta: held_delta,
                token_count: held_token_count,
            }) = self.update.last_mut()
            && held_index == index
        {
            held_delta.push_str(delta);
            *held_token_count = held_token_count.zip(*token_count).map(|(a, b)| a + b);
            return;
        }
        self.update.push(event.clone());
    }
}

/// The kind of the block one event starts, extends, or ends.
fn block_kind(event: &ChatEvent) -> Option<AssistantBlockKind> {
    match event {
        ChatEvent::BlockStart { kind, .. } | ChatEvent::BlockDelta { kind, .. } => Some(*kind),
        ChatEvent::BlockEnd { block, .. } => Some(block.kind()),
        _ => None,
    }
}

/// Content part carried by `response.content_part.*` events.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum ResponseStreamPart {
    Output(ResponseOutputContentPart),
    Reasoning(TextPart),
}

/// State machine that maps one `vllm-chat` event stream onto Responses API
/// SSE events.
///
/// Items open lazily on the first non-empty delta so empty blocks never
/// appear in the stream, matching the Python state machine. Completed items
/// are the response's `output` for both streaming and non-streaming
/// requests, so streamed events and the terminal payload agree about items
/// and IDs.
pub(crate) struct OutputItemStreamer {
    /// The currently open output item, if any.
    current: Option<OpenItem>,
    /// Whether reasoning items appear in the stream and final output.
    include_reasoning: bool,
    /// Items completed so far, in output order.
    completed_items: Vec<ResponseOutputItem>,
    /// Whether the last completed item still owes its `output_item.done`
    /// event: its status is unknown until another item starts or generation
    /// ends.
    last_item_done_pending: bool,
    /// Output-text logprobs state, when requested.
    logprobs: Option<OutputLogprobs>,
}

/// The currently streamed assistant output item.
enum OpenItem {
    Reasoning {
        item_id: String,
        text: String,
    },
    Message {
        item_id: String,
        text: String,
        logprobs: Vec<ChatLogProbsContent>,
    },
    FunctionCall {
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

impl OutputItemStreamer {
    pub(crate) fn new(
        include_reasoning: bool,
        output_logprobs: Option<OutputLogprobsOptions>,
    ) -> Self {
        Self {
            current: None,
            include_reasoning,
            completed_items: Vec::new(),
            last_item_done_pending: false,
            logprobs: output_logprobs.map(|options| OutputLogprobs {
                options,
                update: Vec::new(),
                update_logprobs: Vec::new(),
            }),
        }
    }

    /// Process one chat event, returning the SSE events to emit.
    pub(crate) fn on_event(
        &mut self,
        event: &ChatEvent,
    ) -> Result<Vec<ResponseStreamEvent>, ApiError> {
        let Some(logprobs) = self.logprobs.as_mut() else {
            return Ok(self.process_event(event));
        };
        match event {
            ChatEvent::LogprobsDelta {
                logprobs: delta, ..
            } => {
                if let Some(delta) = delta {
                    let OutputLogprobsOptions {
                        top_logprobs,
                        return_tokens_as_token_ids,
                    } = logprobs.options;
                    let content = decoded_logprobs_to_openai_chat(
                        delta,
                        top_logprobs,
                        return_tokens_as_token_ids,
                    )?
                    .content;
                    logprobs.update_logprobs = content.unwrap_or_default();
                }
                Ok(self.flush_update())
            }
            // Events after the last `LogprobsDelta`, such as closing blocks,
            // form a last update.
            ChatEvent::Done { .. } => Ok(self.flush_update()),
            event => {
                logprobs.hold(event);
                Ok(vec![])
            }
        }
    }

    /// Process the held events of one decoded update with its logprobs.
    fn flush_update(&mut self) -> Vec<ResponseStreamEvent> {
        let Some(logprobs) = self.logprobs.as_mut() else {
            return vec![];
        };
        let update = std::mem::take(&mut logprobs.update);
        // As in Chat Completions, an update with hidden reasoning keeps its
        // token metadata hidden too.
        if !self.include_reasoning
            && update
                .iter()
                .any(|event| block_kind(event) == Some(AssistantBlockKind::Reasoning))
        {
            logprobs.update_logprobs.clear();
        }
        let mut events = Vec::new();
        for event in &update {
            events.extend(self.process_event(event));
        }
        if let Some(logprobs) = self.logprobs.as_mut() {
            logprobs.update_logprobs.clear();
        }
        events
    }

    /// Map one chat event onto SSE events.
    fn process_event(&mut self, event: &ChatEvent) -> Vec<ResponseStreamEvent> {
        let mut events = match event {
            ChatEvent::Start { .. } | ChatEvent::LogprobsDelta { .. } | ChatEvent::Done { .. } => {
                vec![]
            }
            ChatEvent::BlockStart { .. } => vec![],
            ChatEvent::BlockDelta { kind, delta, .. } => {
                if delta.is_empty() {
                    return vec![];
                }
                match kind {
                    AssistantBlockKind::Reasoning => {
                        if !self.include_reasoning {
                            return vec![];
                        }
                        let mut events = self.open_reasoning_if_needed();
                        events.push(self.reasoning_delta(delta.clone()));
                        events
                    }
                    AssistantBlockKind::Text => {
                        let mut events = self.open_message_if_needed();
                        events.push(self.text_delta(delta.clone()));
                        events
                    }
                    // Tool-calls flow through the dedicated events.
                    AssistantBlockKind::ToolCall => vec![],
                }
            }
            ChatEvent::BlockEnd { block, .. } => match block.kind() {
                AssistantBlockKind::Reasoning if !self.include_reasoning => vec![],
                AssistantBlockKind::Reasoning => self.close_reasoning(block_text(block)),
                AssistantBlockKind::Text => self.close_message(block_text(block)),
                // Tool-call blocks flow through the dedicated events.
                AssistantBlockKind::ToolCall => vec![],
            },
            ChatEvent::ToolCallStart { id, name, .. } => {
                let mut events = self.close_current(None);
                events.push(self.open_function_call(id, name));
                events
            }
            ChatEvent::ToolCallArgumentsDelta { delta, .. } => {
                if delta.is_empty() {
                    return vec![];
                }
                vec![self.function_call_delta(delta.clone())]
            }
            ChatEvent::ToolCallEnd { call, .. } => self.close_current(Some(call)),
        };
        if let Some(position) = events
            .iter()
            .position(|event| matches!(event, ResponseStreamEvent::OutputItemAdded { .. }))
            && let Some(done) = self.take_last_item_done()
        {
            events.insert(position, done);
        }
        events
    }

    /// Close any still-open item, returning the close events. Defensive: the
    /// chat event contract closes blocks and tool calls explicitly.
    pub(crate) fn on_stream_end(
        &mut self,
        finish_reason: Option<&FinishReason>,
    ) -> Vec<ResponseStreamEvent> {
        let mut events = self.flush_update();
        events.extend(self.close_current(None));
        if matches!(finish_reason, Some(FinishReason::Length))
            && let Some(last) = self.completed_items.last_mut()
        {
            last.mark_incomplete();
        }
        events.extend(self.take_last_item_done());
        events
    }

    /// Take the deferred `output_item.done` event of the last completed item.
    fn take_last_item_done(&mut self) -> Option<ResponseStreamEvent> {
        if !std::mem::take(&mut self.last_item_done_pending) {
            return None;
        }
        let item = self.completed_items.last()?.clone();
        Some(ResponseStreamEvent::OutputItemDone {
            output_index: self.output_index() - 1,
            item,
        })
    }

    /// Return the completed output items, the response's `output`.
    pub(crate) fn into_output_items(self) -> Vec<ResponseOutputItem> {
        self.completed_items
    }

    /// Index of the next output item.
    fn output_index(&self) -> u32 {
        self.completed_items.len() as u32
    }

    fn open_reasoning_if_needed(&mut self) -> Vec<ResponseStreamEvent> {
        if self.current.is_some() {
            return vec![];
        }
        let item_id = format!("rs_{}", Uuid::new_v4().simple());
        let item = ResponseOutputItem::Reasoning {
            id: item_id.clone(),
            summary: vec![],
            content: None,
            status: Some(ResponseItemStatus::InProgress),
        };
        let output_index = self.output_index();
        self.current = Some(OpenItem::Reasoning {
            item_id: item_id.clone(),
            text: String::new(),
        });
        vec![
            ResponseStreamEvent::OutputItemAdded { output_index, item },
            ResponseStreamEvent::ContentPartAdded {
                item_id,
                output_index,
                content_index: 0,
                part: ResponseStreamPart::Reasoning(TextPart::reasoning_text(String::new())),
            },
        ]
    }

    fn open_message_if_needed(&mut self) -> Vec<ResponseStreamEvent> {
        if self.current.is_some() {
            return vec![];
        }
        let item_id = format!("msg_{}", Uuid::new_v4().simple());
        let item = ResponseOutputItem::Message {
            id: item_id.clone(),
            role: AssistantRole,
            status: ResponseItemStatus::InProgress,
            content: vec![],
        };
        let output_index = self.output_index();
        self.current = Some(OpenItem::Message {
            item_id: item_id.clone(),
            text: String::new(),
            logprobs: Vec::new(),
        });
        vec![
            ResponseStreamEvent::OutputItemAdded { output_index, item },
            ResponseStreamEvent::ContentPartAdded {
                item_id,
                output_index,
                content_index: 0,
                part: ResponseStreamPart::Output(ResponseOutputContentPart::OutputText {
                    text: String::new(),
                    annotations: vec![],
                    logprobs: Some(vec![]),
                }),
            },
        ]
    }

    fn open_function_call(&mut self, id: &str, name: &str) -> ResponseStreamEvent {
        let item_id = format!("fc_{}", Uuid::new_v4().simple());
        let call_id = id.to_string();
        let output_index = self.output_index();
        self.current = Some(OpenItem::FunctionCall {
            item_id: item_id.clone(),
            call_id: call_id.clone(),
            name: name.to_string(),
            arguments: String::new(),
        });
        let item = ResponseOutputItem::FunctionCall {
            id: item_id,
            call_id,
            name: name.to_string(),
            arguments: String::new(),
            status: Some(ResponseItemStatus::InProgress),
        };
        ResponseStreamEvent::OutputItemAdded { output_index, item }
    }

    fn reasoning_delta(&mut self, delta: String) -> ResponseStreamEvent {
        let Some(OpenItem::Reasoning { item_id, text }) = self.current.as_mut() else {
            unreachable!("reasoning delta requires an open reasoning item");
        };
        text.push_str(&delta);
        ResponseStreamEvent::ReasoningTextDelta {
            item_id: item_id.clone(),
            output_index: self.output_index(),
            content_index: 0,
            delta,
        }
    }

    fn text_delta(&mut self, delta: String) -> ResponseStreamEvent {
        let delta_logprobs = self
            .logprobs
            .as_mut()
            .map(|logprobs| std::mem::take(&mut logprobs.update_logprobs))
            .unwrap_or_default();
        let output_index = self.output_index();
        let Some(OpenItem::Message {
            item_id,
            text,
            logprobs,
        }) = self.current.as_mut()
        else {
            unreachable!("text delta requires an open message item");
        };
        text.push_str(&delta);
        let event = ResponseStreamEvent::OutputTextDelta {
            item_id: item_id.clone(),
            output_index,
            content_index: 0,
            delta,
            logprobs: delta_logprobs.iter().map(TextDeltaLogprob::from).collect(),
        };
        logprobs.extend(delta_logprobs);
        event
    }

    fn function_call_delta(&mut self, delta: String) -> ResponseStreamEvent {
        let Some(OpenItem::FunctionCall {
            item_id, arguments, ..
        }) = self.current.as_mut()
        else {
            unreachable!("function call delta requires an open function call item");
        };
        arguments.push_str(&delta);
        ResponseStreamEvent::FunctionCallArgumentsDelta {
            item_id: item_id.clone(),
            output_index: self.output_index(),
            delta,
        }
    }

    /// Close the current item, emitting the done event sequence.
    ///
    /// `final_call` overrides the streamed tool-call payload with the parser's
    /// final tool-call block when available.
    fn close_current(
        &mut self,
        final_call: Option<&AssistantToolCall>,
    ) -> Vec<ResponseStreamEvent> {
        let Some(open) = self.current.take() else {
            return vec![];
        };
        let output_index = self.output_index();
        let (events, item) = match open {
            OpenItem::Reasoning { item_id, text } => {
                let part = TextPart::reasoning_text(text.clone());
                let item = ResponseOutputItem::Reasoning {
                    id: item_id.clone(),
                    summary: vec![],
                    content: Some(vec![part.clone()]),
                    status: Some(ResponseItemStatus::Completed),
                };
                (
                    vec![
                        ResponseStreamEvent::ReasoningTextDone {
                            item_id: item_id.clone(),
                            output_index,
                            content_index: 0,
                            text,
                        },
                        ResponseStreamEvent::ContentPartDone {
                            item_id,
                            output_index,
                            content_index: 0,
                            part: ResponseStreamPart::Reasoning(part),
                        },
                    ],
                    item,
                )
            }
            OpenItem::Message {
                item_id,
                text,
                logprobs,
            } => {
                let part = ResponseOutputContentPart::OutputText {
                    text: text.clone(),
                    annotations: vec![],
                    logprobs: (!logprobs.is_empty()).then_some(logprobs),
                };
                let item = ResponseOutputItem::Message {
                    id: item_id.clone(),
                    role: AssistantRole,
                    status: ResponseItemStatus::Completed,
                    content: vec![part.clone()],
                };
                (
                    vec![
                        ResponseStreamEvent::OutputTextDone {
                            item_id: item_id.clone(),
                            output_index,
                            content_index: 0,
                            text,
                            logprobs: vec![],
                        },
                        ResponseStreamEvent::ContentPartDone {
                            item_id,
                            output_index,
                            content_index: 0,
                            part: ResponseStreamPart::Output(part),
                        },
                    ],
                    item,
                )
            }
            OpenItem::FunctionCall {
                item_id,
                call_id,
                name,
                arguments,
            } => {
                let (call_id, name, arguments) = match final_call {
                    Some(call) => (call.id.clone(), call.name.clone(), call.arguments.clone()),
                    None => (call_id, name, arguments),
                };
                let events = vec![ResponseStreamEvent::FunctionCallArgumentsDone {
                    item_id: item_id.clone(),
                    output_index,
                    name: name.clone(),
                    arguments: arguments.clone(),
                }];
                let item = ResponseOutputItem::FunctionCall {
                    id: item_id,
                    call_id,
                    name,
                    arguments,
                    status: Some(ResponseItemStatus::Completed),
                };
                (events, item)
            }
        };
        self.completed_items.push(item);
        self.last_item_done_pending = true;
        events
    }

    /// Close the reasoning item if it is currently open. The block text
    /// assembled by the parser is authoritative over accumulated deltas.
    fn close_reasoning(&mut self, final_text: Option<&str>) -> Vec<ResponseStreamEvent> {
        if let Some(OpenItem::Reasoning { text, .. }) = self.current.as_mut()
            && let Some(final_text) = final_text
        {
            *text = final_text.to_string();
        }
        if !matches!(self.current, Some(OpenItem::Reasoning { .. })) {
            // Empty reasoning blocks never opened; nothing to close.
            return vec![];
        }
        self.close_current(None)
    }

    /// Close the message item if it is currently open. The block text
    /// assembled by the parser is authoritative over accumulated deltas.
    fn close_message(&mut self, final_text: Option<&str>) -> Vec<ResponseStreamEvent> {
        if let Some(OpenItem::Message { text, .. }) = self.current.as_mut()
            && let Some(final_text) = final_text
        {
            *text = final_text.to_string();
        }
        if !matches!(self.current, Some(OpenItem::Message { .. })) {
            // Empty text blocks never opened; nothing to close.
            return vec![];
        }
        self.close_current(None)
    }
}

/// Extract the text of one text/reasoning block.
fn block_text(block: &AssistantContentBlock) -> Option<&str> {
    match block {
        AssistantContentBlock::Reasoning { text } | AssistantContentBlock::Text { text } => {
            Some(text)
        }
        AssistantContentBlock::ToolCall(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::*;

    #[test]
    fn empty_function_call_emits_arguments_done() {
        let mut streamer = OutputItemStreamer::new(true, None);
        streamer
            .on_event(&ChatEvent::ToolCallStart {
                index: 0,
                id: "call_1".to_string(),
                name: "empty".to_string(),
            })
            .unwrap();

        let mut events = streamer
            .on_event(&ChatEvent::ToolCallEnd {
                index: 0,
                call: AssistantToolCall {
                    id: "call_1".to_string(),
                    name: "empty".to_string(),
                    arguments: String::new(),
                },
            })
            .unwrap();
        events.extend(streamer.on_stream_end(Some(&FinishReason::Stop(None))));
        let payloads: Vec<Value> = events
            .iter()
            .map(|event| serde_json::to_value(SequencedEvent::new(event, 0)).unwrap())
            .collect();

        assert_eq!(payloads[0]["type"], "response.function_call_arguments.done");
        assert_eq!(payloads[0]["arguments"], "");
        assert_eq!(payloads[1]["type"], "response.output_item.done");
    }

    #[test]
    fn empty_reasoning_block_is_not_an_output_item() {
        let mut streamer = OutputItemStreamer::new(true, None);
        for event in [
            ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
            },
            ChatEvent::BlockEnd {
                index: 0,
                block: AssistantContentBlock::Reasoning {
                    text: String::new(),
                },
            },
            ChatEvent::BlockStart {
                index: 1,
                kind: AssistantBlockKind::Text,
            },
            ChatEvent::BlockDelta {
                index: 1,
                kind: AssistantBlockKind::Text,
                delta: "hi".to_string(),
                token_count: None,
            },
            ChatEvent::BlockEnd {
                index: 1,
                block: AssistantContentBlock::Text {
                    text: "hi".to_string(),
                },
            },
        ] {
            streamer.on_event(&event).unwrap();
        }

        let items = streamer.into_output_items();
        assert!(
            matches!(&items[..], [ResponseOutputItem::Message { .. }]),
            "{items:?}"
        );
    }

    /// One decoded update's `LogprobsDelta` for `tokens`.
    fn logprobs_delta(tokens: &[&str]) -> ChatEvent {
        ChatEvent::LogprobsDelta {
            logprobs: Some(DecodedLogprobs {
                positions: tokens
                    .iter()
                    .map(|token| DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: token.to_string(),
                            logprob: -0.5,
                            rank: 1,
                        }],
                    })
                    .collect(),
            }),
            token_ids: vec![0; tokens.len()],
        }
    }

    fn text_delta(delta: &str) -> ChatEvent {
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Text,
            delta: delta.to_string(),
            token_count: None,
        }
    }

    /// Feed `events` to a streamer that includes output-text logprobs and
    /// return each text delta with the tokens of its logprobs.
    fn text_deltas_with_logprob_tokens(events: Vec<ChatEvent>) -> Vec<(String, Vec<String>)> {
        let mut streamer = OutputItemStreamer::new(
            true,
            Some(OutputLogprobsOptions {
                top_logprobs: 0,
                return_tokens_as_token_ids: false,
            }),
        );
        let mut emitted = Vec::new();
        for event in events {
            emitted.extend(streamer.on_event(&event).unwrap());
        }
        emitted
            .into_iter()
            .filter_map(|event| match event {
                ResponseStreamEvent::OutputTextDelta {
                    delta, logprobs, ..
                } => Some((
                    delta,
                    logprobs.into_iter().map(|logprob| logprob.token).collect(),
                )),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn update_without_text_drops_its_logprobs() {
        let deltas = text_deltas_with_logprob_tokens(vec![
            // A parser held `<` back, or a suppressed tool call: the update
            // outputs nothing.
            logprobs_delta(&["<"]),
            text_delta("<b"),
            logprobs_delta(&["b"]),
        ]);
        assert_eq!(deltas, [("<b".to_string(), vec!["b".to_string()])]);
    }

    #[test]
    fn text_deltas_of_one_update_merge_onto_one_delta() {
        let deltas = text_deltas_with_logprob_tokens(vec![
            text_delta("a"),
            text_delta("b"),
            logprobs_delta(&["a", "b"]),
        ]);
        assert_eq!(
            deltas,
            [("ab".to_string(), vec!["a".to_string(), "b".to_string()])]
        );
    }
}
