// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Lowering of stateless Responses requests into `vllm-chat` requests.

use serde_json::json;
use tracing::debug;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatTool, ChatToolChoice, GenerationPromptMode, ResolvedToolContext,
    SamplingParams,
};
use vllm_text::output::TextDecodeOptions;

use super::custom_tool::{self, CustomToolNames};
use super::types::{
    CustomTool, FunctionTool, InputContentPart, InputItem, InputMessage, InputRole,
    InputToolOutput, NamedToolChoice, ReasoningEcho, ResponseObject, ResponseStatus, ResponseTool,
    ResponseToolChoice, ResponsesRequest, TextEcho, TextFormat, TextOrList, ToolChoiceMode,
};
use crate::error::{ApiError, bail_invalid_request, chat_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::structured_outputs::{
    JsonSchemaFormat, ResponseFormat, convert_from_response_format,
};
use crate::utils::{
    ResolvedRequestContext, convert_logit_bias, merge_ec_transfer_params, merge_kv_transfer_params,
    resolve_session_id,
};

/// Lowered chat request plus the response state shared by both delivery modes.
#[derive(Debug, Clone)]
pub(super) struct PreparedRequest {
    /// Lowered chat request for `vllm-chat`.
    pub chat_request: ChatRequest,
    /// In-progress response object echoing the request parameters.
    pub response: ResponseObject,
    /// Names of the declared custom tools.
    pub custom_tools: CustomToolNames,
}

/// Validate and lower one Responses request.
///
/// `lora_resolution.model_names` must be non-empty; the first entry is used as
/// the base `model` field in responses when no LoRA adapter is selected.
pub(super) fn prepare_responses_request(
    request: ResponsesRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    created_at: u64,
) -> Result<PreparedRequest, ApiError> {
    validate_request(&request, &lora_resolution.model_names)?;

    let request_id = format!("resp_{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());

    let tools: Vec<_> = request
        .tools
        .unwrap_or_default()
        .into_iter()
        .filter(|tool| {
            let supported = !matches!(tool, ResponseTool::Unsupported);
            if !supported {
                debug!("ignoring unsupported built-in tool");
            }
            supported
        })
        .collect();
    let custom_tools = CustomToolNames::new(tools.iter().filter_map(|tool| match tool {
        ResponseTool::Custom(tool) => Some(tool.name.clone()),
        ResponseTool::Function(_) | ResponseTool::Unsupported => None,
    }));
    let chat_tools = tools.iter().filter_map(convert_tool).collect();

    let messages = convert_input(request.instructions.clone(), request.input.0)?;
    if messages.is_empty() {
        bail_invalid_request!(
            param = "input",
            "At least one of `input` and `instructions` is required."
        );
    }

    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let tool_context = ResolvedToolContext::new(
        &messages,
        chat_tools,
        request.tool_choice.as_ref().and_then(convert_tool_choice),
        parallel_tool_calls,
    )
    .map_err(|error| chat_submit_error("failed to resolve request tools", error))?;

    let text_format = request.text.and_then(|text| text.format).unwrap_or_default();
    let response_format = convert_text_format(&text_format);
    let structured_outputs =
        convert_from_response_format(response_format.as_ref(), &request.structured_outputs)?;
    let response_format =
        response_format
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| {
                ApiError::invalid_request(
                    format!("failed to serialize text.format: {error}"),
                    Some("text"),
                )
            })?;
    let reasoning_effort = request.reasoning.and_then(|reasoning| reasoning.effort);
    let session_id = resolve_session_id(
        &ctx,
        request.session_id.as_deref(),
        request.vllm_xargs.as_ref(),
    );

    let chat_request = ChatRequest {
        request_id: request_id.clone(),
        messages,
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: request.seed,
            max_tokens: request.max_output_tokens,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            ignore_eos: request.ignore_eos,
            watermarking: request.watermarking,
            logit_bias: convert_logit_bias(request.logit_bias)?,
            structured_outputs,
            vllm_xargs: merge_kv_transfer_params(
                merge_ec_transfer_params(request.vllm_xargs, request.ec_transfer_params.as_ref()),
                request.kv_transfer_params.as_ref(),
            ),
            ..Default::default()
        },
        chat_options: ChatOptions {
            generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
            chat_template: None,
            reasoning_effort: reasoning_effort.map(|effort| effort.as_str().into()),
            response_format,
            template_kwargs: request.chat_template_kwargs.unwrap_or_default(),
        },
        tool_context,
        decode_options: TextDecodeOptions {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
            stop_strings: request.stop.map(|stop| stop.into_vec()),
            min_tokens: 0,
        },
        intermediate: request.stream,
        prompt_truncation: None,
        priority: ctx.priority.or(request.priority).unwrap_or(0),
        documents: None,
        cache_salt: request.cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id,
        lora_request: lora_resolution.lora_request.clone(),
    };

    let response = ResponseObject {
        id: request_id,
        object: "response",
        created_at,
        status: ResponseStatus::InProgress,
        error: None,
        incomplete_details: None,
        instructions: request.instructions,
        max_output_tokens: request.max_output_tokens,
        model: response_model,
        output: Vec::new(),
        parallel_tool_calls,
        previous_response_id: None,
        reasoning: ReasoningEcho {
            effort: reasoning_effort,
            summary: None,
        },
        store: false,
        temperature: request.temperature,
        text: TextEcho {
            format: text_format,
        },
        tool_choice: request.tool_choice.unwrap_or(ResponseToolChoice::Mode(ToolChoiceMode::Auto)),
        tools,
        top_p: request.top_p,
        truncation: "disabled",
        usage: None,
        user: request.user,
        metadata: request.metadata.unwrap_or_else(|| json!({})),
    };

    Ok(PreparedRequest {
        chat_request,
        response,
        custom_tools,
    })
}

/// Reject requests this stateless subset cannot serve faithfully.
///
/// Unrecognized fields are ignored by deserialization. The fields rejected here
/// are the ones whose omission would silently change the generated result.
fn validate_request(
    request: &ResponsesRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    if let Some(model) = request.model.as_ref().filter(|model| !model.is_empty())
        && !served_model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }
    if request.previous_response_id.is_some() {
        bail_invalid_request!(
            param = "previous_response_id",
            "`previous_response_id` is not supported: responses are not stored. Send the full conversation history in `input` instead."
        );
    }
    if request.conversation.is_some() {
        bail_invalid_request!(
            param = "conversation",
            "`conversation` is not supported: responses are not stored. Send the full conversation history in `input` instead."
        );
    }
    if request.background == Some(true) {
        bail_invalid_request!(param = "background", "`background` is not supported.");
    }
    if request.top_logprobs.is_some_and(|top_logprobs| top_logprobs > 0) {
        // TODO: output logprobs on `output_text` parts and deltas.
        bail_invalid_request!(
            param = "top_logprobs",
            "`top_logprobs` is not supported yet."
        );
    }
    Ok(())
}

/// Lower `instructions` plus the input history into chat messages.
///
/// Consecutive assistant-side items (assistant messages, reasoning, and tool
/// calls) are merged into one assistant message, in order. Each tool-call
/// output becomes one tool-response message.
fn convert_input(
    instructions: Option<String>,
    input: Vec<InputItem>,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut messages = Vec::with_capacity(input.len() + 1);
    if let Some(instructions) = instructions {
        messages.push(ChatMessage::system(instructions));
    }

    for item in input {
        match item {
            InputItem::Message(InputMessage { role, content }) => match role {
                InputRole::User => messages.push(ChatMessage::user(convert_content(
                    content,
                    ImagePolicy::Allowed,
                )?)),
                InputRole::Developer => messages.push(ChatMessage::developer(
                    convert_content(content, ImagePolicy::Allowed)?,
                    None,
                )),
                InputRole::System => messages.push(ChatMessage::system(convert_content(
                    content,
                    ImagePolicy::Rejected("system"),
                )?)),
                InputRole::Assistant => {
                    push_assistant_blocks(&mut messages, assistant_text_blocks(content)?);
                }
            },
            InputItem::Reasoning(reasoning) => {
                let text: String =
                    (reasoning.content.into_iter().flatten()).map(|part| part.text).collect();
                if !text.is_empty() {
                    push_assistant_blocks(
                        &mut messages,
                        [AssistantContentBlock::Reasoning { text }],
                    );
                }
            }
            InputItem::FunctionCall(call) => push_assistant_blocks(
                &mut messages,
                [AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: call.call_id,
                    name: call.name,
                    arguments: call.arguments,
                })],
            ),
            InputItem::CustomToolCall(call) => push_assistant_blocks(
                &mut messages,
                [AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: call.call_id,
                    name: call.name,
                    arguments: custom_tool::shim_arguments(call.input),
                })],
            ),
            InputItem::FunctionCallOutput(InputToolOutput { call_id, output })
            | InputItem::CustomToolCallOutput(InputToolOutput { call_id, output }) => messages
                .push(ChatMessage::tool_response(
                    convert_content(output, ImagePolicy::Allowed)?,
                    call_id,
                )),
            InputItem::Unsupported => debug!("ignoring unsupported input item"),
        }
    }

    Ok(messages)
}

/// Append blocks to the trailing assistant message, or start a new one.
fn push_assistant_blocks(
    messages: &mut Vec<ChatMessage>,
    blocks: impl IntoIterator<Item = AssistantContentBlock>,
) {
    if let Some(ChatMessage::Assistant { content }) = messages.last_mut() {
        content.extend(blocks);
    } else {
        messages.push(ChatMessage::assistant_blocks(blocks.into_iter().collect()));
    }
}

/// Whether `input_image` parts are accepted in one content position.
#[derive(Debug, Clone, Copy)]
enum ImagePolicy {
    Allowed,
    /// Rejected in messages of the given role.
    Rejected(&'static str),
}

fn convert_content(
    content: TextOrList<InputContentPart>,
    image_policy: ImagePolicy,
) -> Result<ChatContent, ApiError> {
    let parts = match content {
        TextOrList::Text(text) => return Ok(ChatContent::Text(text)),
        TextOrList::List(parts) => parts,
    };
    parts
        .into_iter()
        .map(|part| match part {
            InputContentPart::InputText { text } | InputContentPart::OutputText { text } => {
                Ok(ChatContentPart::text(text))
            }
            InputContentPart::InputImage {
                image_url,
                file_id,
                detail,
            } => {
                if let ImagePolicy::Rejected(role) = image_policy {
                    bail_invalid_request!(
                        param = "input",
                        "input_image is not supported in {role} messages."
                    );
                }
                let image_url = match (image_url, file_id) {
                    (Some(image_url), None) => image_url,
                    (None, None) => {
                        bail_invalid_request!(
                            param = "input",
                            "input_image must have image_url or file_id."
                        )
                    }
                    (Some(_), Some(_)) => {
                        bail_invalid_request!(
                            param = "input",
                            "input_image cannot have both image_url and file_id."
                        )
                    }
                    (None, Some(_)) => {
                        bail_invalid_request!(
                            param = "input",
                            "input_image file_id is not supported; pass the image as image_url."
                        )
                    }
                };
                Ok(ChatContentPart::ImageUrl {
                    image_url,
                    detail: detail.map(Into::into),
                    uuid: None,
                })
            }
        })
        .collect::<Result<_, _>>()
        .map(ChatContent::Parts)
}

/// Convert replayed assistant message content into assistant text blocks.
fn assistant_text_blocks(
    content: TextOrList<InputContentPart>,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    match content {
        TextOrList::Text(text) => Ok(vec![AssistantContentBlock::Text { text }]),
        TextOrList::List(parts) => parts
            .into_iter()
            .map(|part| match part {
                InputContentPart::InputText { text } | InputContentPart::OutputText { text } => {
                    Ok(AssistantContentBlock::Text { text })
                }
                InputContentPart::InputImage { .. } => bail_invalid_request!(
                    param = "input",
                    "input_image is not supported in assistant messages."
                ),
            })
            .collect(),
    }
}

/// Lower one supported tool declaration into a chat tool.
fn convert_tool(tool: &ResponseTool) -> Option<ChatTool> {
    Some(match tool {
        ResponseTool::Function(FunctionTool {
            name,
            description,
            parameters,
            strict,
        }) => ChatTool {
            name: name.clone(),
            description: description.clone(),
            parameters: parameters
                .clone()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            strict: *strict,
        },
        ResponseTool::Custom(CustomTool {
            name,
            description,
            format,
        }) => ChatTool {
            name: name.clone(),
            description: description.clone(),
            parameters: custom_tool::shim_parameters(format.as_ref()),
            strict: None,
        },
        ResponseTool::Unsupported => return None,
    })
}

/// Convert a requested tool choice. `auto` maps to the default, which falls
/// back to `none` when no supported tool is declared.
fn convert_tool_choice(tool_choice: &ResponseToolChoice) -> Option<ChatToolChoice> {
    match tool_choice {
        ResponseToolChoice::Mode(ToolChoiceMode::Auto) => None,
        ResponseToolChoice::Mode(ToolChoiceMode::None) => Some(ChatToolChoice::None),
        ResponseToolChoice::Mode(ToolChoiceMode::Required) => Some(ChatToolChoice::Required),
        ResponseToolChoice::Named(
            NamedToolChoice::Function { name } | NamedToolChoice::Custom { name },
        ) => Some(ChatToolChoice::Function { name: name.clone() }),
    }
}

/// Convert `text.format` into the shared response format, or `None` for plain
/// text.
fn convert_text_format(format: &TextFormat) -> Option<ResponseFormat> {
    match format {
        TextFormat::Text => None,
        TextFormat::JsonObject => Some(ResponseFormat::JsonObject),
        TextFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: name.clone(),
                description: description.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, StatusCode};
    use expect_test::expect;
    use serde_json::{Value, json};
    use vllm_chat::ChatToolChoice;

    use super::{PreparedRequest, prepare_responses_request};
    use crate::error::ApiError;
    use crate::lora::LoraModelResolution;
    use crate::routes::openai::responses::types::ResponsesRequest;
    use crate::utils::resolve_request_context;

    fn prepare(body: Value) -> Result<PreparedRequest, ApiError> {
        let request: ResponsesRequest = serde_json::from_value(body).expect("deserialize request");
        prepare_responses_request(
            request,
            &LoraModelResolution {
                model_names: vec!["test-model".to_string()],
                lora_request: None,
            },
            resolve_request_context(&HeaderMap::new(), Some("test")),
            1,
        )
    }

    fn error_message(body: Value) -> String {
        let error = prepare(body).expect_err("request should be rejected");
        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
        error.to_error_response().error.message
    }

    #[test]
    fn string_input_becomes_user_message_after_instructions() {
        let prepared = prepare(json!({
            "model": "test-model",
            "instructions": "Be brief.",
            "input": "Hi",
        }))
        .unwrap();

        assert_eq!(prepared.chat_request.request_id, "resp_test");
        expect![[r#"
            [
                System {
                    content: Text(
                        "Be brief.",
                    ),
                },
                User {
                    content: Text(
                        "Hi",
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&prepared.chat_request.messages);
    }

    #[test]
    fn instructions_alone_are_a_valid_request() {
        let prepared = prepare(json!({"instructions": "Say hi."})).unwrap();
        assert_eq!(prepared.chat_request.messages.len(), 1);
    }

    #[test]
    fn replayed_agent_history_merges_assistant_side_items() {
        let prepared = prepare(json!({
            "input": [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "Sandbox: read-only."},
                ]},
                {"role": "user", "content": "Fix the bug."},
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque",
                 "content": [{"type": "reasoning_text", "text": "Read the file first."}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "shell",
                 "arguments": "{\"command\":\"cat a.py\"}", "status": "completed"},
                {"type": "function_call_output", "call_id": "call_1", "output": "print(1)"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "ignored"}]},
                {"type": "custom_tool_call", "call_id": "call_2", "name": "apply_patch",
                 "input": "*** Begin Patch\n*** End Patch"},
                {"type": "custom_tool_call_output", "call_id": "call_2", "output": [
                    {"type": "input_text", "text": "Done."},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "original"},
                ]},
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Fixed.", "annotations": []},
                ]},
                {"type": "message", "role": "user", "content": "Thanks."},
            ],
        }))
        .unwrap();

        expect![[r#"
            [
                Developer {
                    content: Parts(
                        [
                            Text {
                                text: "Sandbox: read-only.",
                            },
                        ],
                    ),
                    tools: None,
                },
                User {
                    content: Text(
                        "Fix the bug.",
                    ),
                },
                Assistant {
                    content: [
                        Reasoning {
                            text: "Read the file first.",
                        },
                        ToolCall(
                            AssistantToolCall {
                                id: "call_1",
                                name: "shell",
                                arguments: "{\"command\":\"cat a.py\"}",
                            },
                        ),
                    ],
                },
                ToolResponse {
                    content: Text(
                        "print(1)",
                    ),
                    tool_call_id: "call_1",
                },
                Assistant {
                    content: [
                        ToolCall(
                            AssistantToolCall {
                                id: "call_2",
                                name: "apply_patch",
                                arguments: "{\"input\":\"*** Begin Patch\\n*** End Patch\"}",
                            },
                        ),
                    ],
                },
                ToolResponse {
                    content: Parts(
                        [
                            Text {
                                text: "Done.",
                            },
                            ImageUrl {
                                image_url: "data:image/png;base64,AAAA",
                                detail: Some(
                                    High,
                                ),
                                uuid: None,
                            },
                        ],
                    ),
                    tool_call_id: "call_2",
                },
                Assistant {
                    content: [
                        Text {
                            text: "Fixed.",
                        },
                    ],
                },
                User {
                    content: Text(
                        "Thanks.",
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&prepared.chat_request.messages);
    }

    #[test]
    fn custom_tools_are_offered_as_single_string_function_shims() {
        let prepared = prepare(json!({
            "input": "Edit the file.",
            "tools": [
                {"type": "function", "name": "shell", "description": "Run a command.",
                 "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                 "strict": false},
                {"type": "custom", "name": "apply_patch", "description": "Edit files.",
                 "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"patch\""}},
                {"type": "web_search", "external_web_access": false},
            ],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "parallel_tool_calls": false,
        }))
        .unwrap();

        assert!(prepared.custom_tools.contains("apply_patch"));
        assert!(!prepared.custom_tools.contains("shell"));
        let context = &prepared.chat_request.tool_context;
        assert_eq!(
            context.tool_choice,
            ChatToolChoice::Function {
                name: "apply_patch".to_string()
            }
        );
        assert!(!context.parallel_tool_calls);
        let [shell, apply_patch] = context.effective_tools.as_slice() else {
            panic!("expected two tools: {:?}", context.effective_tools);
        };
        assert_eq!(shell.strict, Some(false));
        expect![[r#"
            Tool {
                name: "apply_patch",
                description: Some(
                    "Edit files.",
                ),
                parameters: Object {
                    "type": String("object"),
                    "properties": Object {
                        "input": Object {
                            "type": String("string"),
                            "description": String("The raw text input of this tool, passed through verbatim. It must conform to the following lark grammar:\nstart: \"patch\""),
                        },
                    },
                    "required": Array [
                        String("input"),
                    ],
                },
                strict: None,
            }
        "#]]
        .assert_debug_eq(apply_patch);
        // Ignored built-in tools are not echoed back.
        assert_eq!(prepared.response.tools.len(), 2);
    }

    #[test]
    fn auto_tool_choice_without_supported_tools_disables_tools() {
        let prepared = prepare(json!({
            "input": "Hi",
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
        }))
        .unwrap();
        assert!(!prepared.chat_request.tool_context.parsing_enabled());
    }

    #[test]
    fn sampling_reasoning_and_text_format_are_lowered() {
        let prepared = prepare(json!({
            "input": "Hi",
            "temperature": 0.5,
            "top_p": 0.9,
            "max_output_tokens": 64,
            "reasoning": {"effort": "max", "summary": "auto"},
            "text": {"format": {"type": "json_schema", "name": "answer", "strict": true,
                                "schema": {"type": "object"}}, "verbosity": "low"},
            "stream": true,
        }))
        .unwrap();
        let request = &prepared.chat_request;

        assert_eq!(request.sampling_params.temperature, Some(0.5));
        assert_eq!(request.sampling_params.top_p, Some(0.9));
        assert_eq!(request.sampling_params.max_tokens, Some(64));
        assert!(request.sampling_params.structured_outputs.is_some());
        assert_eq!(request.chat_options.reasoning_effort, Some("max".into()));
        assert_eq!(
            request.chat_options.response_format,
            Some(json!({"type": "json_schema", "json_schema": {
                "name": "answer", "schema": {"type": "object"}, "strict": true,
            }}))
        );
        assert!(request.intermediate);
    }

    #[test]
    fn vllm_sampling_extensions_are_passed_through() {
        let prepared = prepare(json!({
            "input": "Hi",
            "top_k": 20,
            "seed": 7,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "logit_bias": {"42": -100},
            "stop": ["END"],
            "ignore_eos": true,
            "skip_special_tokens": false,
            "include_stop_str_in_output": true,
            "watermarking": false,
            "structured_outputs": {"regex": "[a-z]+"},
            "text": {"format": {"type": "json_object"}},
            "session_id": "session-1",
            "priority": 3,
            "cache_salt": "salt",
            "chat_template_kwargs": {"thinking": false},
            "kv_transfer_params": {"do_remote_decode": true},
        }))
        .unwrap();
        let request = &prepared.chat_request;
        let sampling = &request.sampling_params;

        assert_eq!(sampling.top_k, Some(20));
        assert_eq!(sampling.seed, Some(7));
        assert_eq!(sampling.repetition_penalty, Some(1.1));
        assert_eq!(sampling.frequency_penalty, Some(0.5));
        assert_eq!(sampling.presence_penalty, Some(-0.5));
        assert_eq!(sampling.logit_bias.as_ref().unwrap()[&42], -100.0);
        assert!(sampling.ignore_eos);
        assert!(!sampling.watermarking);
        // Explicit structured outputs take precedence over `text.format`.
        expect![[r#"
            Some(
                StructuredOutputsParams {
                    constraint: Regex(
                        "[a-z]+",
                    ),
                    options: StructuredOutputOptions {
                        disable_any_whitespace: false,
                        disable_additional_properties: false,
                        whitespace_pattern: None,
                    },
                    backend: Guidance,
                },
            )
        "#]]
        .assert_debug_eq(&sampling.structured_outputs);
        assert_eq!(
            sampling.vllm_xargs.as_ref().unwrap()["kv_transfer_params"],
            json!({"do_remote_decode": true})
        );
        assert_eq!(
            request.decode_options.stop_strings,
            Some(vec!["END".to_string()])
        );
        assert!(!request.decode_options.skip_special_tokens);
        assert!(request.decode_options.include_stop_str_in_output);
        assert_eq!(request.session_id.as_deref(), Some("session-1"));
        assert_eq!(request.priority, 3);
        assert_eq!(request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(
            request.chat_options.template_kwargs["thinking"],
            json!(false)
        );
    }

    #[test]
    fn stateful_and_unsupported_requests_are_rejected() {
        expect!["`previous_response_id` is not supported: responses are not stored. Send the full conversation history in `input` instead."]
            .assert_eq(&error_message(json!({"input": "Hi", "previous_response_id": "resp_1"})));
        expect!["`conversation` is not supported: responses are not stored. Send the full conversation history in `input` instead."]
            .assert_eq(&error_message(json!({"input": "Hi", "conversation": "conv_1"})));
        expect!["`background` is not supported."]
            .assert_eq(&error_message(json!({"input": "Hi", "background": true})));
        expect!["`top_logprobs` is not supported yet."]
            .assert_eq(&error_message(json!({"input": "Hi", "top_logprobs": 2})));
        expect!["At least one of `input` and `instructions` is required."]
            .assert_eq(&error_message(json!({"input": []})));
    }

    #[test]
    fn invalid_images_are_rejected() {
        let image_message = |role: &str, image: Value| {
            json!({"input": [{"role": role, "content": [
                {"type": "input_text", "text": "Look."},
                {"type": "input_image", "detail": "low"}
            ]}]})
            .pipe_image(image)
        };
        trait PipeImage {
            fn pipe_image(self, image: Value) -> Value;
        }
        impl PipeImage for Value {
            fn pipe_image(mut self, image: Value) -> Value {
                let part = &mut self["input"][0]["content"][1];
                for (key, value) in image.as_object().unwrap() {
                    part[key] = value.clone();
                }
                self
            }
        }

        let url = json!({"image_url": "https://example.com/a.png"});
        expect!["input_image is not supported in system messages."]
            .assert_eq(&error_message(image_message("system", url.clone())));
        expect!["input_image is not supported in assistant messages."]
            .assert_eq(&error_message(image_message("assistant", url.clone())));
        expect!["input_image must have image_url or file_id."]
            .assert_eq(&error_message(image_message("user", json!({}))));
        expect!["input_image cannot have both image_url and file_id."].assert_eq(&error_message(
            image_message(
                "user",
                json!({"image_url": "https://example.com/a.png", "file_id": "file-1"}),
            ),
        ));
        expect!["input_image file_id is not supported; pass the image as image_url."].assert_eq(
            &error_message(image_message("user", json!({"file_id": "file-1"}))),
        );
        assert!(prepare(image_message("user", url)).is_ok());
    }

    #[test]
    fn malformed_input_items_report_their_position() {
        let error = serde_json::from_value::<ResponsesRequest>(json!({
            "input": [{"role": "user", "content": "Hi"}, {"type": "function_call", "name": "f"}],
        }))
        .unwrap_err();
        expect!["input[1]: missing field `call_id`"].assert_eq(&error.to_string());
    }
}
