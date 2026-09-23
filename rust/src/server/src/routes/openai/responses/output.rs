// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Assembly of Responses output items from `vllm-chat` events.
//!
//! Every assistant block becomes one output item: text blocks become
//! `message` items, reasoning blocks become `reasoning` items, and tool calls
//! become `function_call` or `custom_tool_call` items. Streaming and
//! non-streaming responses share the item builders here, so an item streamed
//! by `response.output_item.done` is identical to the one in the final
//! response.

use tracing::debug;
use uuid::Uuid;
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantToolCall, ChatEvent, ChatTokenUsage,
    FinishReason,
};

use super::custom_tool::{self, CustomToolNames, InputDecoder};
use super::types::{
    IncompleteDetails, InputTokensDetails, ItemStatus, OutputContentPart, OutputItem,
    OutputTokensDetails, ResponseError, ResponseObject, ResponseStatus, ResponseUsage, StreamEvent,
};

/// Generate one output item ID with the given type prefix.
fn item_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

/// Build one completed output item from one finished assistant block.
fn completed_item(
    id: String,
    block: AssistantContentBlock,
    custom_tools: &CustomToolNames,
) -> OutputItem {
    match block {
        AssistantContentBlock::Text { text } => OutputItem::Message {
            id,
            role: "assistant",
            status: ItemStatus::Completed,
            content: vec![OutputContentPart::output_text(text)],
        },
        AssistantContentBlock::Reasoning { text } => OutputItem::Reasoning {
            id,
            status: ItemStatus::Completed,
            summary: Vec::new(),
            content: vec![OutputContentPart::ReasoningText { text }],
        },
        AssistantContentBlock::ToolCall(call) => {
            tool_call_item(id, ItemStatus::Completed, call, custom_tools)
        }
    }
}

/// Build one tool-call output item.
fn tool_call_item(
    id: String,
    status: ItemStatus,
    AssistantToolCall {
        id: call_id,
        name,
        arguments,
    }: AssistantToolCall,
    custom_tools: &CustomToolNames,
) -> OutputItem {
    if custom_tools.contains(&name) {
        OutputItem::CustomToolCall {
            id,
            status,
            call_id,
            name,
            input: custom_tool::raw_input(arguments),
        }
    } else {
        OutputItem::FunctionCall {
            id,
            status,
            call_id,
            name,
            arguments,
        }
    }
}

/// Item ID prefix for one tool call.
fn tool_call_id_prefix(name: &str, custom_tools: &CustomToolNames) -> &'static str {
    if custom_tools.contains(name) {
        "ctc"
    } else {
        "fc"
    }
}

/// Build the output items of one non-streaming response.
pub(super) fn output_items(
    blocks: Vec<AssistantContentBlock>,
    custom_tools: &CustomToolNames,
) -> Vec<OutputItem> {
    blocks
        .into_iter()
        .map(|block| {
            let prefix = match &block {
                AssistantContentBlock::Text { .. } => "msg",
                AssistantContentBlock::Reasoning { .. } => "rs",
                AssistantContentBlock::ToolCall(call) => {
                    tool_call_id_prefix(&call.name, custom_tools)
                }
            };
            completed_item(item_id(prefix), block, custom_tools)
        })
        .collect()
}

/// Fill in the terminal state of a response from the chat terminal metadata.
pub(super) fn finish_response(
    response: &mut ResponseObject,
    output: Vec<OutputItem>,
    usage: ChatTokenUsage,
    finish_reason: &FinishReason,
) {
    let (status, incomplete_details, error) = match finish_reason {
        FinishReason::Stop(_) => (ResponseStatus::Completed, None, None),
        FinishReason::Length => (
            ResponseStatus::Incomplete,
            Some(IncompleteDetails {
                reason: "max_output_tokens",
            }),
            None,
        ),
        FinishReason::Repetition(_) => (
            ResponseStatus::Incomplete,
            Some(IncompleteDetails {
                reason: "repetition",
            }),
            None,
        ),
        FinishReason::Abort => (
            ResponseStatus::Failed,
            None,
            Some(ResponseError {
                code: "server_error",
                message: "The request was aborted.".to_string(),
            }),
        ),
        FinishReason::Error => (
            ResponseStatus::Failed,
            None,
            Some(ResponseError {
                code: "server_error",
                message: "Internal server error".to_string(),
            }),
        ),
    };
    response.status = status;
    response.incomplete_details = incomplete_details;
    response.error = error;
    response.output = output;
    response.usage = Some(ResponseUsage {
        input_tokens: usage.prompt_token_count,
        input_tokens_details: InputTokensDetails {
            cached_tokens: usage.cached_token_count,
        },
        output_tokens: usage.output_token_count,
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: usage.reasoning_tokens,
        },
        total_tokens: usage.prompt_token_count + usage.output_token_count,
    });
}

/// Mark a response as failed after generation broke off mid-stream, keeping
/// the output items completed so far.
pub(super) fn fail_response(
    response: &mut ResponseObject,
    output: Vec<OutputItem>,
    message: String,
) {
    response.status = ResponseStatus::Failed;
    response.error = Some(ResponseError {
        code: "server_error",
        message,
    });
    response.output = output;
}

/// The output item currently being streamed.
#[derive(Debug)]
struct OpenItem {
    id: String,
    output_index: usize,
    kind: OpenItemKind,
}

#[derive(Debug)]
enum OpenItemKind {
    Text,
    Reasoning,
    FunctionCall,
    CustomToolCall {
        decoder: InputDecoder,
        /// Input text streamed so far.
        streamed: String,
    },
}

/// Per-response streaming state translating chat events into stream events.
///
/// `vllm-chat` opens at most one block or tool call at a time and closes it
/// before opening the next, so there is at most one open output item.
#[derive(Debug)]
pub(super) struct StreamAssembler {
    custom_tools: CustomToolNames,
    /// Completed output items, in output order.
    output: Vec<OutputItem>,
    open: Option<OpenItem>,
}

impl StreamAssembler {
    pub fn new(custom_tools: CustomToolNames) -> Self {
        Self {
            custom_tools,
            output: Vec::new(),
            open: None,
        }
    }

    /// Take the output items completed so far.
    pub fn take_output(&mut self) -> Vec<OutputItem> {
        std::mem::take(&mut self.output)
    }

    /// Open one output item at the next output index.
    fn open(&mut self, prefix: &str, kind: OpenItemKind) -> &OpenItem {
        self.open.insert(OpenItem {
            id: item_id(prefix),
            output_index: self.output.len(),
            kind,
        })
    }

    /// Translate one non-terminal chat event into stream events.
    pub fn push(&mut self, event: ChatEvent, events: &mut Vec<StreamEvent>) {
        match event {
            ChatEvent::BlockStart { kind, .. } => {
                let (prefix, open_kind, item, part): (_, _, fn(String) -> OutputItem, _) =
                    match kind {
                        AssistantBlockKind::Text => (
                            "msg",
                            OpenItemKind::Text,
                            |id| OutputItem::Message {
                                id,
                                role: "assistant",
                                status: ItemStatus::InProgress,
                                content: Vec::new(),
                            },
                            OutputContentPart::output_text(String::new()),
                        ),
                        AssistantBlockKind::Reasoning => (
                            "rs",
                            OpenItemKind::Reasoning,
                            |id| OutputItem::Reasoning {
                                id,
                                status: ItemStatus::InProgress,
                                summary: Vec::new(),
                                content: Vec::new(),
                            },
                            OutputContentPart::ReasoningText {
                                text: String::new(),
                            },
                        ),
                        AssistantBlockKind::ToolCall => {
                            unreachable!("tool calls flow through dedicated tool-call events")
                        }
                    };
                let open = self.open(prefix, open_kind);
                let (id, output_index) = (open.id.clone(), open.output_index);
                events.push(StreamEvent::OutputItemAdded {
                    output_index,
                    item: item(id.clone()),
                });
                events.push(StreamEvent::ContentPartAdded {
                    item_id: id,
                    output_index,
                    content_index: 0,
                    part,
                });
            }
            ChatEvent::BlockDelta { delta, .. } => {
                let open = self.open.as_ref().expect("block delta without an open block");
                let (item_id, output_index) = (open.id.clone(), open.output_index);
                events.push(match open.kind {
                    OpenItemKind::Text => StreamEvent::OutputTextDelta {
                        item_id,
                        output_index,
                        content_index: 0,
                        delta,
                        logprobs: Vec::new(),
                    },
                    OpenItemKind::Reasoning => StreamEvent::ReasoningTextDelta {
                        item_id,
                        output_index,
                        content_index: 0,
                        delta,
                    },
                    OpenItemKind::FunctionCall | OpenItemKind::CustomToolCall { .. } => {
                        unreachable!("block delta while a tool call is open")
                    }
                });
            }
            ChatEvent::BlockEnd { block, .. } => {
                let OpenItem {
                    id, output_index, ..
                } = self.open.take().expect("block end without an open block");
                let (text_done, part) = match &block {
                    AssistantContentBlock::Text { text } => (
                        StreamEvent::OutputTextDone {
                            item_id: id.clone(),
                            output_index,
                            content_index: 0,
                            text: text.clone(),
                            logprobs: Vec::new(),
                        },
                        OutputContentPart::output_text(text.clone()),
                    ),
                    AssistantContentBlock::Reasoning { text } => (
                        StreamEvent::ReasoningTextDone {
                            item_id: id.clone(),
                            output_index,
                            content_index: 0,
                            text: text.clone(),
                        },
                        OutputContentPart::ReasoningText { text: text.clone() },
                    ),
                    AssistantContentBlock::ToolCall(_) => {
                        unreachable!("tool calls flow through dedicated tool-call events")
                    }
                };
                events.push(text_done);
                events.push(StreamEvent::ContentPartDone {
                    item_id: id.clone(),
                    output_index,
                    content_index: 0,
                    part,
                });
                self.finish_item(completed_item(id, block, &self.custom_tools), events);
            }
            ChatEvent::ToolCallStart {
                id: call_id, name, ..
            } => {
                let (prefix, kind) = if self.custom_tools.contains(&name) {
                    let kind = OpenItemKind::CustomToolCall {
                        decoder: InputDecoder::default(),
                        streamed: String::new(),
                    };
                    ("ctc", kind)
                } else {
                    ("fc", OpenItemKind::FunctionCall)
                };
                let open = self.open(prefix, kind);
                let (id, output_index) = (open.id.clone(), open.output_index);
                let call = AssistantToolCall {
                    id: call_id,
                    name,
                    arguments: String::new(),
                };
                events.push(StreamEvent::OutputItemAdded {
                    output_index,
                    item: tool_call_item(id, ItemStatus::InProgress, call, &self.custom_tools),
                });
            }
            ChatEvent::ToolCallArgumentsDelta { delta, .. } => {
                let open = self.open.as_mut().expect("arguments delta without an open tool call");
                let (item_id, output_index) = (open.id.clone(), open.output_index);
                match &mut open.kind {
                    OpenItemKind::FunctionCall => {
                        events.push(StreamEvent::FunctionCallArgumentsDelta {
                            item_id,
                            output_index,
                            delta,
                        });
                    }
                    OpenItemKind::CustomToolCall { decoder, streamed } => {
                        let delta = decoder.push(&delta);
                        if !delta.is_empty() {
                            streamed.push_str(&delta);
                            events.push(StreamEvent::CustomToolCallInputDelta {
                                item_id,
                                output_index,
                                delta,
                            });
                        }
                    }
                    OpenItemKind::Text | OpenItemKind::Reasoning => {
                        unreachable!("arguments delta while a text block is open")
                    }
                }
            }
            ChatEvent::ToolCallEnd { call, .. } => {
                let OpenItem {
                    id,
                    output_index,
                    kind,
                } = self.open.take().expect("tool call end without an open tool call");
                let item =
                    tool_call_item(id.clone(), ItemStatus::Completed, call, &self.custom_tools);
                match &item {
                    OutputItem::FunctionCall {
                        name, arguments, ..
                    } => events.push(StreamEvent::FunctionCallArgumentsDone {
                        item_id: id,
                        output_index,
                        name: name.clone(),
                        arguments: arguments.clone(),
                    }),
                    OutputItem::CustomToolCall { input, .. } => {
                        let OpenItemKind::CustomToolCall { streamed, .. } = kind else {
                            unreachable!("custom tool calls open custom tool-call items")
                        };
                        // Stream what the decoder held back. Malformed shim
                        // arguments can make the final input diverge from the
                        // streamed text, in which case only `done` carries it.
                        match input.strip_prefix(streamed.as_str()) {
                            Some("") => {}
                            Some(remainder) => events.push(StreamEvent::CustomToolCallInputDelta {
                                item_id: id.clone(),
                                output_index,
                                delta: remainder.to_string(),
                            }),
                            None => debug!("custom tool call input diverged from streamed text"),
                        }
                        events.push(StreamEvent::CustomToolCallInputDone {
                            item_id: id,
                            output_index,
                            input: input.clone(),
                        });
                    }
                    OutputItem::Message { .. } | OutputItem::Reasoning { .. } => {
                        unreachable!("tool calls build tool-call items")
                    }
                }
                self.finish_item(item, events);
            }
            ChatEvent::Start { .. } | ChatEvent::LogprobsDelta { .. } => {}
            ChatEvent::Done { .. } => unreachable!("terminal events are handled by the caller"),
        }
    }

    /// Record one completed item and emit its `output_item.done` event.
    fn finish_item(&mut self, item: OutputItem, events: &mut Vec<StreamEvent>) {
        events.push(StreamEvent::OutputItemDone {
            output_index: self.output.len(),
            item: item.clone(),
        });
        self.output.push(item);
    }
}
