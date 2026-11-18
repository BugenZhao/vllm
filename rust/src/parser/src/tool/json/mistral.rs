// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::{Value, json};
use vllm_tokenizer::{DecodedText, DynTokenizer};
use xgrammar_structural_tag::NormalizedToolChoice;
use xgrammar_structural_tag::format::{Format, TagFormat, TriggeredTagsFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam, function_parameters};

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::output_grammar::{self, OutputGrammarContext, normalize_tool_choice};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

/// The control token that starts the call array.
const TOOL_CALLS: &str = "[TOOL_CALLS]";

/// The call array's opener, as the chat template writes it.
const CALLS_OPEN: &str = "[TOOL_CALLS] [";

const MISTRAL_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Mistral",
    start_marker: CALLS_OPEN,
    framed_start_marker: None,
    end_marker: "]",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for the pre-v11 Mistral JSON-array tool calls.
///
/// Example tool call content:
///
/// ```text
/// [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location":"Tokyo"}}]
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub(crate) struct MistralJsonArrayToolParser {
    inner: JsonToolCallParser,
}

impl MistralJsonArrayToolParser {
    /// Create a Mistral tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(MISTRAL_CONFIG),
        }
    }
}

impl ToolParser for MistralJsonArrayToolParser {
    fn create(tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// `[TOOL_CALLS]` is a special token in pre-v11 tokenizers too, so decoding
    /// must keep it for the start marker to reach the parser.
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// Constrain calls to the array this parser reads, `[TOOL_CALLS] [` then
    /// `{"name": ..., "arguments": ...}` objects joined by `, ` and `]`, as the
    /// chat template renders them. Python guides `required` and named choices
    /// to the bare array instead, because its JSON-schema path cannot write
    /// the control token.
    fn build_call_format(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<Format>> {
        let Some(NormalizedToolChoice {
            function_tools,
            choice,
            ..
        }) = normalize_tool_choice(ctx)?
        else {
            return Ok(None);
        };
        let single_call = choice == BuilderToolChoice::Forced || !ctx.parallel_tool_calls;
        let calls = TagFormat::new(
            CALLS_OPEN,
            Format::tags_with_separator(
                function_tools.iter().map(call_tag).collect(),
                ", ",
                true,
                single_call,
            ),
            "]",
        );
        Ok(Some(match choice {
            // The template ends the turn after the array, so there is one.
            BuilderToolChoice::Auto => Format::TriggeredTags(
                TriggeredTagsFormat::new(&[TOOL_CALLS], vec![calls]).with_stop_after_first(true),
            ),
            BuilderToolChoice::Required | BuilderToolChoice::Forced => Format::Tag(calls),
        }))
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

/// One call object: the tool's name, then arguments under its parameter schema.
fn call_tag(tool: &FunctionToolParam) -> TagFormat {
    let name = Value::from(tool.function.name.as_str());
    let arguments = match function_parameters(&tool.function) {
        // The parser reads the arguments as a JSON object, so an unpinned
        // schema still admits only objects.
        Value::Bool(true) => json!({ "type": "object" }),
        schema => schema,
    };
    TagFormat::new(
        format!(r#"{{"name": {name}, "arguments": "#),
        Format::json_schema(arguments),
        "}",
    )
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;
    use xgrammar_structural_tag::ToolChoice;

    use super::MistralJsonArrayToolParser;
    use crate::output_grammar::test_utils::outline;
    use crate::output_grammar::{OutputGrammarContext, ToolStrictLevel};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    /// Outline of the visible grammar for `tool_choice`, or `None`.
    fn call_grammar(
        tool_choice: ToolChoice,
        tool_strict_level: ToolStrictLevel,
        parallel_tool_calls: bool,
    ) -> Option<String> {
        let tools = test_tools();
        let parser = MistralJsonArrayToolParser::new(&tools);
        parser
            .build_visible_format(&OutputGrammarContext {
                tools: &tools,
                tool_choice: &tool_choice,
                tool_strict_level,
                parallel_tool_calls,
                answer: None,
            })
            .unwrap()
            .map(|format| outline(&format))
    }

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{function_name}","arguments":{arguments}}}"#)
    }

    fn build_tool_calls(tool_calls: &[String]) -> String {
        format!("[TOOL_CALLS] [{}]", tool_calls.join(","))
    }

    #[test]
    fn mistral_parse_complete_without_tool_call_keeps_text() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn mistral_parse_complete_extracts_raw_json_arguments() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let output = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_calls(&[build_tool_call("get_weather", arguments)])
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check.\n");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn mistral_parse_complete_extracts_pretty_multiple_tool_calls() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"I'll help.
[TOOL_CALLS] [
    {"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}}
    ,
    {"name": "add", "arguments": {"x": 1, "y": 2}}
]"#,
            )
            .unwrap();

        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "I'll help.\n",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\": \"Tokyo\", \"units\": \"celsius\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\": 1, \"y\": 2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn mistral_does_not_validate_or_normalize_arguments() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser
            .parse_complete(&build_tool_calls(&[build_tool_call(
                "get_weather",
                arguments,
            )]))
            .unwrap();

        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn mistral_streaming_emits_argument_deltas() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let chunks = [
            "preface [TOOL",
            "_CALLS] [{\"name\":\"get_weather\",\"arguments\":",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}] suffix",
        ];

        let mut output = ToolParserOutput::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_arguments.extend(
                next.calls()
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(output.normal_text(), "preface  suffix");
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn mistral_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_calls(&[build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn mistral_keeps_array_bracket_literal_inside_json_string() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let arguments = r#"{"text":"Array notation: arr[0] = value[1]"}"#;
        let output = parser
            .parse_complete(&build_tool_calls(&[build_tool_call("echo", arguments)]))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn mistral_finish_fails_incomplete_tool_call() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"[TOOL_CALLS] [{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["incomplete Mistral tool call"].assert_eq(&error.to_report_string());
    }

    #[test]
    fn mistral_malformed_field_order_fails_fast() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let error = parser
            .parse_chunk(r#"[TOOL_CALLS] [{"arguments":{},"name":"get_weather"}]"#)
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: near "{\"arguments\":{},\"name\":\"get_weather\"}]": invalid Mistral
            expected `name`"#]]
        .assert_eq(&error.to_report_string());
    }
    #[test]
    fn mistral_required_grammar_constrains_the_call_array() {
        expect![[r#"
            tag `[TOOL_CALLS] [` .. `]`
              tags_with_separator `, ` at_least_one
                tag `{"name": "get_weather", "arguments": ` json(object) `}`
                tag `{"name": "add", "arguments": ` json(object) `}`
                tag `{"name": "convert", "arguments": ` json(object) `}`
                tag `{"name": "calculate_area", "arguments": ` json(object) `}`
                tag `{"name": "update_record", "arguments": ` json(object) `}`
        "#]]
        .assert_eq(&call_grammar(ToolChoice::required(), ToolStrictLevel::Auto, true).unwrap());
    }

    #[test]
    fn mistral_named_grammar_allows_one_call_with_pinned_arguments() {
        expect![[r#"
            tag `[TOOL_CALLS] [` .. `]`
              tags_with_separator `, ` at_least_one stop_after_first
                tag `{"name": "add", "arguments": ` json({ x?: integer, y?: integer }) `}`
        "#]]
        .assert_eq(
            &call_grammar(
                ToolChoice::function("add"),
                ToolStrictLevel::Parameter,
                true,
            )
            .unwrap(),
        );
    }

    #[test]
    fn mistral_auto_grammar_triggers_on_the_control_token() {
        assert_eq!(
            call_grammar(ToolChoice::auto(), ToolStrictLevel::Auto, true),
            None
        );
        expect![[r#"
            triggered_tags [`[TOOL_CALLS]`] stop_after_first
              tag `[TOOL_CALLS] [` .. `]`
                tags_with_separator `, ` at_least_one stop_after_first
                  tag `{"name": "get_weather", "arguments": ` json(object) `}`
                  tag `{"name": "add", "arguments": ` json(object) `}`
                  tag `{"name": "convert", "arguments": ` json(object) `}`
                  tag `{"name": "calculate_area", "arguments": ` json(object) `}`
                  tag `{"name": "update_record", "arguments": ` json(object) `}`
        "#]]
        .assert_eq(&call_grammar(ToolChoice::auto(), ToolStrictLevel::Function, false).unwrap());
    }

    #[test]
    fn mistral_parses_calls_shaped_like_the_grammar() {
        let mut parser = MistralJsonArrayToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "add", "arguments": {"x": 1}}]"#,
            )
            .unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[1].name.as_deref(), Some("add"));
        assert_eq!(output.calls()[1].arguments, r#"{"x": 1}"#);
    }
}
