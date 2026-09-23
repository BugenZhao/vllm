// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Wire types for the stateless `/v1/responses` subset.
//!
//! The request side accepts the complete conversation history on every call and
//! silently ignores fields it does not recognize, so that existing Responses
//! clients can connect unchanged. Fields whose semantics require server-side
//! state are recognized explicitly so they can be rejected instead of ignored.
//!
//! OpenAI API reference: <https://platform.openai.com/docs/api-reference/responses>

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use llm_multimodal::ImageDetail;
use serde::de::{self, DeserializeOwned, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use validator::Validate;

use crate::routes::openai::utils::types::{
    Normalizable, ReasoningEffort, StringOrArray, default_true, deserialize_request_top_k,
    validate_stop, validate_top_p_value,
};

// ============================================================================
// Request
// ============================================================================

/// Request body of `POST /v1/responses`.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct ResponsesRequest {
    /// ID of the model to use.
    pub model: Option<String>,
    /// Complete conversation history for this generation.
    #[serde(default)]
    pub input: ResponseInput,
    /// System instructions inserted before the input history.
    pub instructions: Option<String>,
    /// Whether to stream the response as semantic server-sent events.
    #[serde(default)]
    pub stream: bool,
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,
    /// Upper bound on generated tokens, including reasoning tokens.
    #[validate(range(min = 1))]
    pub max_output_tokens: Option<u32>,
    #[validate(range(min = 0, max = 20))]
    pub top_logprobs: Option<u32>,
    /// Tool declarations. Built-in tool types are ignored.
    pub tools: Option<Vec<ResponseTool>>,
    pub tool_choice: Option<ResponseToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub reasoning: Option<ReasoningParam>,
    pub text: Option<TextParam>,
    /// Echoed back unchanged on the response object.
    pub metadata: Option<Value>,
    pub user: Option<String>,

    // -------- Stateful parameters, rejected when set --------
    pub previous_response_id: Option<String>,
    pub conversation: Option<Value>,
    pub background: Option<bool>,

    // -------- vLLM sampling parameters, as in the Python Responses API --------
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_request_top_k")]
    pub top_k: Option<u32>,
    pub seed: Option<i64>,
    pub logit_bias: Option<HashMap<String, f32>>,
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    #[serde(default = "default_true")]
    pub watermarking: bool,
    /// Explicit structured outputs, taking precedence over `text.format`.
    pub structured_outputs: Option<Value>,

    // -------- vLLM request parameters --------
    /// Caller-supplied request ID, overridden by the `X-Request-Id` header.
    pub request_id: Option<String>,
    /// Stable session identity shared by related requests.
    pub session_id: Option<String>,
    /// Scheduling priority, overridden by the `X-Vllm-Priority` header.
    pub priority: Option<i32>,
    #[validate(length(min = 1))]
    pub cache_salt: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    pub vllm_xargs: Option<HashMap<String, Value>>,
    pub kv_transfer_params: Option<HashMap<String, Value>>,
    pub ec_transfer_params: Option<HashMap<String, Value>>,
}

impl Normalizable for ResponsesRequest {}

/// Normalized `input`: a plain string is one user message.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResponseInput(pub Vec<InputItem>);

impl<'de> Deserialize<'de> for ResponseInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputVisitor;

        impl<'de> Visitor<'de> for InputVisitor {
            type Value = ResponseInput;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string or a list of input items")
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Ok(ResponseInput(vec![InputItem::Message(InputMessage {
                    role: InputRole::User,
                    content: TextOrList::Text(text.to_string()),
                })]))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(ResponseInput::default())
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(mut object) = seq.next_element::<serde_json::Map<String, Value>>()? {
                    // `EasyInputMessage` entries may omit the `type` tag.
                    object.entry("type").or_insert_with(|| Value::from("message"));
                    let item = InputItem::deserialize(Value::Object(object)).map_err(|error| {
                        de::Error::custom(format!("input[{}]: {error}", items.len()))
                    })?;
                    items.push(item);
                }
                Ok(ResponseInput(items))
            }
        }

        deserializer.deserialize_any(InputVisitor)
    }
}

/// Either a plain string or a list of typed entries, as used by message
/// `content` and tool-call `output`.
///
/// Deserialized by hand rather than with `#[serde(untagged)]` so that errors in
/// list entries keep their specific message.
#[derive(Debug, Clone, PartialEq)]
pub enum TextOrList<T> {
    Text(String),
    List(Vec<T>),
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for TextOrList<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TextOrListVisitor<T>(PhantomData<T>);

        impl<'de, T: DeserializeOwned> Visitor<'de> for TextOrListVisitor<T> {
            type Value = TextOrList<T>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string or a list of content parts")
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Ok(TextOrList::Text(text.to_string()))
            }

            fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
                Ok(TextOrList::Text(text))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut list = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(entry) = seq.next_element()? {
                    list.push(entry);
                }
                Ok(TextOrList::List(list))
            }
        }

        deserializer.deserialize_any(TextOrListVisitor(PhantomData))
    }
}

/// One input history item.
///
/// Item `id` and `status` fields are accepted and ignored: the history is
/// rebuilt from content on every request.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Message(InputMessage),
    Reasoning(InputReasoning),
    FunctionCall(InputFunctionCall),
    FunctionCallOutput(InputToolOutput),
    CustomToolCall(InputCustomToolCall),
    CustomToolCallOutput(InputToolOutput),
    /// Item types outside this subset, such as built-in tool calls, are
    /// ignored.
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InputMessage {
    pub role: InputRole,
    pub content: TextOrList<InputContentPart>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputRole {
    User,
    Assistant,
    System,
    Developer,
}

/// One content part of an input message or tool-call output.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentPart {
    InputText {
        text: String,
    },
    /// Assistant text replayed from an earlier response. Annotations and
    /// logprobs are ignored.
    OutputText {
        text: String,
    },
    InputImage {
        image_url: Option<String>,
        file_id: Option<String>,
        detail: Option<InputImageDetail>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputImageDetail {
    Auto,
    Low,
    High,
    Original,
}

impl From<InputImageDetail> for ImageDetail {
    fn from(detail: InputImageDetail) -> Self {
        match detail {
            InputImageDetail::Auto => Self::Auto,
            InputImageDetail::Low => Self::Low,
            // `original` asks for at least `high` fidelity.
            InputImageDetail::High | InputImageDetail::Original => Self::High,
        }
    }
}

/// Reasoning replayed from an earlier response.
///
/// Only plain-text `content` is restored. `summary` is not the reasoning itself
/// and `encrypted_content` is opaque to this server; both are ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InputReasoning {
    #[serde(default)]
    pub content: Option<Vec<ReasoningTextPart>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "reasoning_text")]
pub struct ReasoningTextPart {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InputFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InputCustomToolCall {
    pub call_id: String,
    pub name: String,
    pub input: String,
}

/// Output of a function or custom tool call, provided by the client.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InputToolOutput {
    pub call_id: String,
    pub output: TextOrList<InputContentPart>,
}

/// One tool declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTool {
    Function(FunctionTool),
    Custom(CustomTool),
    /// Built-in tools such as `web_search` or `mcp` are executed by the
    /// provider in the full API. This stateless subset ignores them.
    #[serde(other)]
    Unsupported,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionTool {
    pub name: String,
    pub description: Option<String>,
    /// JSON schema of the arguments object. Absent means no arguments.
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
}

/// A tool whose call input is free-form text instead of a JSON object.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomTool {
    pub name: String,
    pub description: Option<String>,
    pub format: Option<CustomToolFormat>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CustomToolFormat {
    /// Unconstrained text.
    Text,
    /// Text described by a grammar, e.g. Codex's Lark `apply_patch` grammar.
    Grammar { syntax: String, definition: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseToolChoice {
    Mode(ToolChoiceMode),
    Named(NamedToolChoice),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NamedToolChoice {
    Function { name: String },
    Custom { name: String },
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ReasoningParam {
    pub effort: Option<ReasoningEffort>,
    // `summary` is accepted, but no summary is generated.
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct TextParam {
    pub format: Option<TextFormat>,
    // `verbosity` is accepted and has no effect.
}

/// Output text format, echoed back on the response object.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: Value,
        strict: Option<bool>,
    },
}

// ============================================================================
// Response
// ============================================================================

/// The `response` object returned by non-streaming requests and carried by
/// lifecycle stream events.
///
/// Fields that depend on unsupported capabilities have fixed values, e.g.
/// `store: false` and `previous_response_id: null`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: ResponseStatus,
    pub error: Option<ResponseError>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    pub reasoning: ReasoningEcho,
    pub store: bool,
    pub temperature: Option<f32>,
    pub text: TextEcho,
    pub tool_choice: ResponseToolChoice,
    pub tools: Vec<ResponseTool>,
    pub top_p: Option<f32>,
    pub truncation: &'static str,
    pub usage: Option<ResponseUsage>,
    pub user: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResponseError {
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IncompleteDetails {
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReasoningEcho {
    pub effort: Option<ReasoningEffort>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextEcho {
    pub format: TextFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: usize,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: usize,
}

/// One generated output item.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message {
        id: String,
        role: &'static str,
        status: ItemStatus,
        content: Vec<OutputContentPart>,
    },
    Reasoning {
        id: String,
        status: ItemStatus,
        /// Always empty: summaries are not generated.
        summary: Vec<Value>,
        content: Vec<OutputContentPart>,
    },
    FunctionCall {
        id: String,
        status: ItemStatus,
        call_id: String,
        name: String,
        arguments: String,
    },
    CustomToolCall {
        id: String,
        status: ItemStatus,
        call_id: String,
        name: String,
        input: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    InProgress,
    Completed,
}

/// One content part of an output message or reasoning item.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentPart {
    OutputText {
        text: String,
        annotations: Vec<Value>,
        // TODO: output logprobs.
        logprobs: Vec<Value>,
    },
    ReasoningText {
        text: String,
    },
}

impl OutputContentPart {
    pub fn output_text(text: String) -> Self {
        Self::OutputText {
            text,
            annotations: Vec::new(),
            logprobs: Vec::new(),
        }
    }
}

// ============================================================================
// Streaming
// ============================================================================

/// One semantic server-sent event.
///
/// Every event carries a monotonically increasing `sequence_number`, assigned
/// when the event is serialized.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "response.created")]
    Created { response: ResponseObject },
    #[serde(rename = "response.in_progress")]
    InProgress { response: ResponseObject },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OutputContentPart,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OutputContentPart,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
        logprobs: Vec<Value>,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
        logprobs: Vec<Value>,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: usize,
        name: String,
        arguments: String,
    },
    #[serde(rename = "response.custom_tool_call_input.delta")]
    CustomToolCallInputDelta {
        item_id: String,
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.custom_tool_call_input.done")]
    CustomToolCallInputDone {
        item_id: String,
        output_index: usize,
        input: String,
    },
    #[serde(rename = "response.completed")]
    Completed { response: ResponseObject },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponseObject },
    #[serde(rename = "response.failed")]
    Failed { response: ResponseObject },
}

impl StreamEvent {
    /// Return the SSE `event:` name, which equals the payload `type`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Created { .. } => "response.created",
            Self::InProgress { .. } => "response.in_progress",
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
            Self::CustomToolCallInputDelta { .. } => "response.custom_tool_call_input.delta",
            Self::CustomToolCallInputDone { .. } => "response.custom_tool_call_input.done",
            Self::Completed { .. } => "response.completed",
            Self::Incomplete { .. } => "response.incomplete",
            Self::Failed { .. } => "response.failed",
        }
    }

    /// Terminal lifecycle event for a finished response with the given status.
    pub fn terminal(response: ResponseObject) -> Self {
        match response.status {
            ResponseStatus::Completed => Self::Completed { response },
            ResponseStatus::Incomplete => Self::Incomplete { response },
            ResponseStatus::Failed => Self::Failed { response },
            ResponseStatus::InProgress => unreachable!("terminal response must be finished"),
        }
    }
}

/// Wire form of one [`StreamEvent`] with its sequence number.
#[derive(Debug, Serialize)]
pub struct SequencedEvent<'a> {
    #[serde(flatten)]
    pub event: &'a StreamEvent,
    pub sequence_number: u64,
}
