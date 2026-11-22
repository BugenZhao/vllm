// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::Value;
use vllm_tokenizer::{DecodedText, DynTokenizer};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, peek, seq};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::token::{any, literal};
use xgrammar_structural_tag::format::{Format, TagFormat, TriggeredTagsFormat};
use xgrammar_structural_tag::tool::function_parameters;

use super::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    tool_call_header_event,
};
use crate::output_grammar::{self, OutputGrammarContext};
use crate::schema::JsonType;
use crate::tool::utils::{
    JsonObjectScanState, JsonStringScanState, decode_json_str, parse_buffered_event, safe_text_len,
    take_json_object, take_json_string,
};
use crate::tool::{Result, Tool, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite4Mode {
    Text,
    Header,
    /// Parsing the arguments value:
    /// `None` until the first byte decides object vs string;
    /// `Some` while streaming the selected value shape.
    Args {
        args_scan: Option<Granite4ArgsScan>,
    },
    /// Arguments done; consume the object's closing `}` and `</tool_call>`.
    Close,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite4ArgsScan {
    Object(JsonObjectScanState),
    String(JsonStringScanState),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite4Event {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCallHeader {
        function_name: String,
    },
    /// Verbatim bytes of an object-valued arguments payload; `complete` once the
    /// object scan reaches its closing brace.
    ObjectArgsDelta {
        len: usize,
        complete: bool,
    },
    /// Decoded contents of a string-valued arguments payload.
    StringArgs {
        decoded: String,
    },
    ToolCallEnd,
}

/// Tool parser for Granite 4 `<tool_call>…</tool_call>` JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>{"name": "get_weather", "arguments": {"city": "Boston"}}</tool_call>
/// ```
///
/// Parallel calls are repeated `<tool_call>…</tool_call>` blocks with ordinary
/// content interleaved between them. This reuses the shared JSON helpers for
/// everything except one Granite 4 specific step (`args_event`): the `arguments`
/// value may be a JSON object (kept verbatim) **or** a JSON string whose decoded
/// contents are the arguments (the `# test granite behavior` case in Python).
pub struct Granite4ToolParser {
    buffer: String,
    mode: Granite4Mode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl Granite4ToolParser {
    /// Create a Granite 4 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Granite4Mode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Apply one parsed Granite 4 event to parser state and output.
    fn apply_event(&mut self, event: Granite4Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            Granite4Event::Text { len } => output.push_text(&self.buffer[..len]),
            Granite4Event::ToolCallStart => self.mode = Granite4Mode::Header,
            Granite4Event::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = Granite4Mode::Args { args_scan: None };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            Granite4Event::ObjectArgsDelta { len, complete } => {
                let arguments = self.buffer[..len].to_string();
                self.push_arguments(arguments, output)?;
                if complete {
                    self.mode = Granite4Mode::Close;
                }
            }
            Granite4Event::StringArgs { decoded } => {
                self.push_arguments(decoded, output)?;
                self.mode = Granite4Mode::Close;
            }
            Granite4Event::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = Granite4Mode::Text;
            }
        }
        Ok(())
    }

    /// Append one arguments delta to the active tool call.
    fn push_arguments(&self, arguments: String, output: &mut ToolParserOutput) -> Result<()> {
        let Some(tool_index) = self.active_tool_index else {
            return Err(parsing_failed!(
                "Granite4 arguments without an active tool call"
            ));
        };
        output.push_call(ToolCallDelta {
            tool_index,
            name: None,
            arguments,
        });
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = Granite4Mode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for Granite4ToolParser {
    fn create(tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Build the `<tool_call>` blocks the chat template renders: one or more
    /// for a tool choice that obliges a call, or any number amid free text
    /// when `auto` still asks for a grammar (a strict tool,
    /// `--tool-strict-level`, `parallel_tool_calls: false`).
    fn build_call_format(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<Format>> {
        let Some(tools) = output_grammar::normalize_tool_choice(ctx)? else {
            return Ok(None);
        };

        let calls = tools
            .function_tools
            .iter()
            .map(|tool| {
                // The template renders arguments as an object, so free
                // arguments still have to be one.
                let arguments = match function_parameters(&tool.function) {
                    Value::Bool(true) => JsonType::Object.any_schema(),
                    schema => schema,
                };
                TagFormat::new(
                    format!(
                        "{TOOL_CALL_START}\n{{\"name\": {}, \"arguments\": ",
                        Value::from(tool.function.name.as_str())
                    ),
                    Format::json_schema(arguments),
                    format!("}}\n{TOOL_CALL_END}"),
                )
            })
            .collect();
        let single_call = tools.choice.is_forced() || !ctx.parallel_tool_calls;
        Ok(Some(if tools.choice.requires_tool_call() {
            Format::tags_with_separator(calls, "\n", true, single_call)
        } else {
            Format::TriggeredTags(
                TriggeredTagsFormat::new(&[TOOL_CALL_START], calls)
                    .with_stop_after_first(single_call),
            )
        }))
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(&chunk.text);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_granite4_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            Granite4Mode::Text => output.push_text(&self.buffer),
            Granite4Mode::Header | Granite4Mode::Args { .. } | Granite4Mode::Close => {
                return Err(ToolParserError::IncompleteToolCall { parser: "Granite4" });
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Granite4ToolParser::reset(self)
    }
}

/// Parse a Granite 4 event for the current parser mode.
fn parse_next_granite4_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut Granite4Mode,
) -> ModalResult<Granite4Event> {
    match mode {
        Granite4Mode::Text => text_event(input),
        Granite4Mode::Header => header_event(input),
        Granite4Mode::Args { args_scan } => args_event(input, args_scan),
        Granite4Mode::Close => close_event(input),
    }
}

/// Parse content text or the start of a `<tool_call>` block. *(reuses `safe_text_len`)*
fn text_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite4Event> {
    alt((
        |input: &mut JsonToolInput<'_>| {
            seq!(_: literal(TOOL_CALL_START), _: ws0)
                .value(Granite4Event::ToolCallStart)
                .parse_next(input)
        },
        |input: &mut JsonToolInput<'_>| {
            safe_text_len(input, TOOL_CALL_START).map(|len| Granite4Event::Text { len })
        },
    ))
    .parse_next(input)
}

/// Parse the `{"name":"X","arguments":` header before the value. *(reuses `tool_call_header_event`)*
fn header_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite4Event> {
    const CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Granite4",
        start_marker: "",
        framed_start_marker: None,
        end_marker: "",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: None,
        name_key: "name",
        arguments_key: &["arguments"],
    };

    match tool_call_header_event(input, CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => {
            Ok(Granite4Event::ToolCallHeader { function_name })
        }
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Parse one arguments-value event.
///
/// GRANITE 4 SPECIFIC - the sole behavior that differs from the shared
/// `<tool_call>` JSON parsers. The value is either a JSON object (kept verbatim,
/// streamed incrementally via `take_json_object`) or an escaped JSON string
/// (decoded whole via `json_str`). The string form is why we cannot just forward
/// raw arg bytes like the sibling parsers do: an escaped string only resolves
/// once seen whole and unescaped.
fn args_event(
    input: &mut JsonToolInput<'_>,
    args_scan: &mut Option<Granite4ArgsScan>,
) -> ModalResult<Granite4Event> {
    if let Some(scan) = args_scan {
        return match scan {
            Granite4ArgsScan::Object(scan) => {
                let len = take_json_object(input, scan)?;
                Ok(Granite4Event::ObjectArgsDelta {
                    len,
                    complete: scan.complete(),
                })
            }
            Granite4ArgsScan::String(scan) => string_args_event(input, scan),
        };
    }

    match peek(any).parse_next(input)? {
        '{' => {
            let mut scan = JsonObjectScanState::default();
            let len = take_json_object(input, &mut scan)?;
            let complete = scan.complete();
            *args_scan = Some(Granite4ArgsScan::Object(scan));
            Ok(Granite4Event::ObjectArgsDelta { len, complete })
        }
        '"' => {
            *args_scan = Some(Granite4ArgsScan::String(JsonStringScanState::default()));
            let Some(Granite4ArgsScan::String(scan)) = args_scan else {
                unreachable!("Granite4 string scan state was just initialized")
            };
            string_args_event(input, scan)
        }
        _ => {
            let mut error = ContextError::new();
            error.push(StrContext::Label("Granite4 arguments"));
            Err(ErrMode::Cut(error))
        }
    }
}

fn string_args_event(
    input: &mut JsonToolInput<'_>,
    scan: &mut JsonStringScanState,
) -> ModalResult<Granite4Event> {
    let text = **input;
    let len = take_json_string(input, scan)?;
    Ok(Granite4Event::StringArgs {
        decoded: decode_json_str(&text[..len])?,
    })
}

/// Parse the tool-call object's closing `}` and the `</tool_call>` end marker.
fn close_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite4Event> {
    seq!(_: ws0, _: literal("}"), _: ws0, _: literal(TOOL_CALL_END))
        .value(Granite4Event::ToolCallEnd)
        .parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;
    use thiserror_ext::AsReport;
    use xgrammar_structural_tag::ToolChoice;
    use xgrammar_structural_tag::format::Format;

    use super::Granite4ToolParser;
    use crate::output_grammar::test_utils::outline;
    use crate::output_grammar::{OutputGrammarContext, ToolStrictLevel};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{Tool, ToolParser, ToolParserOutput, ToolParserTestExt as _};

    #[test]
    fn granite4_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn granite4_parse_complete_object_args() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<tool_call>{"name":"get_weather","arguments":{"city":"Boston"}}</tool_call>"#,
            )
            .unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"city":"Boston"}"#);
    }

    #[test]
    fn granite4_parse_complete_string_args() {
        // GRANITE4-SPECIFIC: `arguments` may be a pre-serialized JSON string; its
        // decoded contents become the arguments.
        let mut parser = Granite4ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<tool_call>{"name":"get_weather","arguments":"{\"city\":\"Boston\"}"}</tool_call>"#,
            )
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"city":"Boston"}"#);
    }

    #[test]
    fn granite4_extracts_interleaved_content_and_mixed_args() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"before <tool_call>{"name":"find_bbox","arguments":"{\"x\":1}"}</tool_call> middle <tool_call>{"name":"get_weather","arguments":{"city":"Boston"}}</tool_call> after"#,
            )
            .unwrap();

        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "before  middle  after",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "find_bbox",
                            ),
                            arguments: "{\"x\":1}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"Boston\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn granite4_streaming_handles_split_markers() {
        let input = r#"hello <tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call> bye"#;
        let chunks = split_by_chars(input, 5);
        let mut parser = Granite4ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello  bye");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"city":"Tokyo"}"#);
    }

    #[test]
    fn granite4_streaming_emits_object_argument_deltas() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let chunks = [
            r#"<tool_call>{"name":"get_weather","arguments":"#,
            r#"{"city":"#,
            r#""Beijing""#,
            r#"}"#,
            r#"}</tool_call>"#,
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

        assert_eq!(observed_arguments, [r#"{"city":"#, r#""Beijing""#, r#"}"#]);
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"city":"Beijing"}"#
        );
    }

    #[test]
    fn granite4_string_args_split_across_chunks() {
        let input = r#"<tool_call>{"name":"f","arguments":"{\"a\":1}"}</tool_call>"#;
        let chunks = split_by_chars(input, 3);
        let mut parser = Granite4ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("f"));
        assert_eq!(output.calls()[0].arguments, r#"{"a":1}"#);
    }

    #[test]
    fn granite4_long_string_args_stream_without_reparse() {
        let arguments = format!(r#"{{"data":"{}"}}"#, "x".repeat(64 * 1024));
        let encoded_arguments = serde_json::to_string(&arguments).unwrap();
        let input =
            format!(r#"<tool_call>{{"name":"f","arguments":{encoded_arguments}}}</tool_call>"#);
        let chunks = split_by_chars(&input, 7);
        let mut parser = Granite4ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("f"));
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn granite4_streaming_handles_marker_and_json_whitespace() {
        // Granite spaces the markers (`<tool_call> {…} </tool_call>`) and the JSON
        // (`"name": …`). Since `args_event` has no leading `ws0`, this guards that
        // the header consumes the whitespace before the arguments value.
        let input = concat!(
            "Here goes the bbox call: \n",
            r#"<tool_call> {"name": "find_bbox", "arguments": "{\"coordinates\": [[23.54, 43.1], [-12.2, 54.3], [4, 5]], \"coordinate_type\": \"latlong\"}"} </tool_call>"#,
            " Now the stock price call: \n ",
            r#"<tool_call> {"name": "get_stock_price", "arguments": {"symbol": "AAPL", "start_date": "2021-01-01", "end_date": "2021-12-31"}} </tool_call>"#,
            " Now another bbox call: \n ",
            r#"<tool_call> {"name": "find_bbox", "arguments": "{\"coordinates\": [[23.54, 43.1], [-12.2, 54.3], [4, 5]], \"coordinate_type\": \"latlong\"}"} </tool_call>"#,
            " See? I'm a helpful assistant.",
        );
        let chunks = split_by_chars(input, 3);
        let mut parser = Granite4ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "Here goes the bbox call: \n Now the stock price call: \n  Now another bbox call: \n  See? I'm a helpful assistant.",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "find_bbox",
                            ),
                            arguments: "{\"coordinates\": [[23.54, 43.1], [-12.2, 54.3], [4, 5]], \"coordinate_type\": \"latlong\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "get_stock_price",
                            ),
                            arguments: "{\"symbol\": \"AAPL\", \"start_date\": \"2021-01-01\", \"end_date\": \"2021-12-31\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 2,
                            name: Some(
                                "find_bbox",
                            ),
                            arguments: "{\"coordinates\": [[23.54, 43.1], [-12.2, 54.3], [4, 5]], \"coordinate_type\": \"latlong\"}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&output);
    }

    #[test]
    fn granite4_finish_fails_incomplete_tool_call() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<tool_call>{"name":"get_weather","arguments":{"city""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["incomplete Granite4 tool call"].assert_eq(&error.to_report_string());
    }

    #[test]
    fn granite4_rejects_non_object_non_string_args() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let error = parser
            .parse_chunk(r#"<tool_call>{"name":"f","arguments":42}</tool_call>"#)
            .unwrap_err();

        expect![[
            r#"tool parser parsing failed: near "42}</tool_call>": invalid Granite4 arguments"#
        ]]
        .assert_eq(&error.to_report_string());
    }

    #[test]
    fn granite4_preserve_special_tokens_is_false() {
        let parser = Granite4ToolParser::new(&test_tools());
        assert!(!parser.preserve_special_tokens());
    }

    #[test]
    fn granite4_builds_the_template_call_grammar() {
        let mut strict_tools = test_tools()[..2].to_vec();
        strict_tools[0].strict = Some(true);
        let build = |tools: &[Tool],
                     tool_choice: ToolChoice,
                     parallel_tool_calls,
                     answer: Option<&Format>| {
            Granite4ToolParser::new(tools)
                .build_visible_format(&OutputGrammarContext {
                    tools,
                    tool_choice: &tool_choice,
                    tool_strict_level: ToolStrictLevel::Auto,
                    parallel_tool_calls,
                    answer,
                })
                .unwrap()
                .map(|format| outline(&format))
        };

        // `required` holds the model to one or more calls as the template
        // renders them; a strict tool pins its arguments.
        expect![[r#"
            tags_with_separator `\n` at_least_one
              tag `<tool_call>\n{"name": "get_weather", "arguments": ` json({ location?: string, city?: string, state?: string, unit?: string, date?: string, days?: integer }) `}\n</tool_call>`
              tag `<tool_call>\n{"name": "add", "arguments": ` json({ ... }) `}\n</tool_call>`
        "#]]
            .assert_eq(&build(&strict_tools, ToolChoice::required(), true, None).unwrap());
        // A named tool allows exactly that one call.
        expect![[r#"
            tags_with_separator `\n` at_least_one stop_after_first
              tag `<tool_call>\n{"name": "add", "arguments": ` json({ ... }) `}\n</tool_call>`
        "#]]
        .assert_eq(&build(&strict_tools, ToolChoice::function("add"), true, None).unwrap());

        // `auto` builds no grammar unless something asks for one, such as a
        // strict tool, and then leaves text free around the calls.
        assert!(build(&test_tools()[..2], ToolChoice::auto(), true, None).is_none());
        expect![[r#"
            triggered_tags [`<tool_call>`]
              tag `<tool_call>\n{"name": "get_weather", "arguments": ` json({ location?: string, city?: string, state?: string, unit?: string, date?: string, days?: integer }) `}\n</tool_call>`
              tag `<tool_call>\n{"name": "add", "arguments": ` json({ ... }) `}\n</tool_call>`
        "#]].assert_eq(&build(&strict_tools, ToolChoice::auto(), true, None).unwrap());
        // Without parallel calls, both stop after the first call.
        expect![[r#"
            tags_with_separator `\n` at_least_one stop_after_first
              tag `<tool_call>\n{"name": "get_weather", "arguments": ` json({ ... }) `}\n</tool_call>`
              tag `<tool_call>\n{"name": "add", "arguments": ` json({ ... }) `}\n</tool_call>`
        "#]]
            .assert_eq(&build(&test_tools()[..2], ToolChoice::required(), false, None).unwrap());
        expect![[r#"
            triggered_tags [`<tool_call>`] stop_after_first
              tag `<tool_call>\n{"name": "get_weather", "arguments": ` json({ ... }) `}\n</tool_call>`
              tag `<tool_call>\n{"name": "add", "arguments": ` json({ ... }) `}\n</tool_call>`
        "#]]
            .assert_eq(&build(&test_tools()[..2], ToolChoice::auto(), false, None).unwrap());

        // With an answer constraint, `auto` holds either the calls or the answer.
        let answer = Format::json_schema(json!({"type": "object"}));
        expect![[r#"
            or
              tags_with_separator `\n` at_least_one
                tag `<tool_call>\n{"name": "get_weather", "arguments": ` json({ ... }) `}\n</tool_call>`
                tag `<tool_call>\n{"name": "add", "arguments": ` json({ ... }) `}\n</tool_call>`
              json(object)
        "#]].assert_eq(
            &build(&test_tools()[..2], ToolChoice::auto(), true, Some(&answer)).unwrap(),
        );
    }
}
