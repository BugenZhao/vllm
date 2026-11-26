// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};
use xgrammar_structural_tag::FunctionToolParam;
use xgrammar_structural_tag::format::{Format, JsonSchemaStyle, TagFormat, TriggeredTagsFormat};
use xgrammar_structural_tag::tool::function_parameters;

use super::qwen_coder::{Qwen3CoderToolParser, QwenCoderConfig};
use crate::output_grammar::{self, OutputGrammarContext};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const SEED_OSS_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "Seed-OSS",
    tool_call_start: "<seed:tool_call>",
    first_tool_call_start: "\n\n<seed:tool_call>",
    next_tool_call_start: "\n<seed:tool_call>",
    tool_call_end: "</seed:tool_call>",
    trim_parameter_newlines: true,
};

/// Tool parser for Seed-OSS XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <seed:tool_call>
/// <function=get_weather>
/// <parameter=location>杭州</parameter>
/// </function>
/// </seed:tool_call>
/// ```
///
/// Seed-OSS shares the Qwen3 Coder grammar exactly; only the two tool-call
/// wrapper tokens differ (`<seed:tool_call>` / `</seed:tool_call>`). The inner
/// `<function=...>` / `<parameter=...>` grammar and schema-driven argument
/// conversion are byte-identical, so this delegates to a
/// [`Qwen3CoderToolParser`] configured with the Seed-OSS markers. This mirrors
/// Python `SeedOssParser(Qwen3Parser)` in `vllm/parser/seed_oss.py`.
///
/// xgrammar has no Seed-OSS model (Python `SeedOssEngineToolParser` sets
/// `structural_tag_model = None`), so this parser builds the call grammar
/// from the calls the chat template renders.
///
/// The `<seed:tool_call>` wrapper tokens are added-vocabulary tokens
/// (`special = false`), not tokenizer special tokens, so they survive decoding
/// under the production default `skip_special_tokens = true`; the trait default
/// `preserve_special_tokens() == false` is therefore correct, matching the
/// sibling `SeedOssReasoningParser`, which relies on the same for `<seed:think>`.
pub struct SeedOssToolParser {
    inner: Qwen3CoderToolParser,
}

impl SeedOssToolParser {
    /// Create a Seed-OSS tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            inner: Qwen3CoderToolParser::with_config(tools, SEED_OSS_CONFIG),
        }
    }
}

impl ToolParser for SeedOssToolParser {
    fn create(tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Build the `<seed:tool_call>` blocks the chat template renders: one or
    /// more for a tool choice that obliges a call, or any number amid free
    /// text when `auto` still asks for a grammar (a strict tool,
    /// `--tool-strict-level`, `parallel_tool_calls: false`).
    fn build_call_format(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<Format>> {
        let Some(tools) = output_grammar::normalize_tool_choice(ctx)? else {
            return Ok(None);
        };

        let call = |prefix: &str, tool: &FunctionToolParam| {
            TagFormat::new(
                format!(
                    "{prefix}{}\n<function={}>\n",
                    SEED_OSS_CONFIG.tool_call_start, tool.function.name
                ),
                Format::json_schema_style(
                    function_parameters(&tool.function),
                    JsonSchemaStyle::QwenXml,
                ),
                format!("</function>\n{}", SEED_OSS_CONFIG.tool_call_end),
            )
        };
        let single_call = tools.choice.is_forced() || !ctx.parallel_tool_calls;
        Ok(Some(if tools.choice.requires_tool_call() {
            // The template opens every call on a new line, the first one
            // included.
            let calls = tools.function_tools.iter().map(|tool| call("\n", tool)).collect();
            Format::tags_with_separator(calls, "", true, single_call)
        } else {
            // Trigger on the bare opener, which is all the parser needs to
            // start a call; the newline before it stays free text.
            let calls = tools.function_tools.iter().map(|tool| call("", tool)).collect();
            Format::TriggeredTags(
                TriggeredTagsFormat::new(&[SEED_OSS_CONFIG.tool_call_start], calls)
                    .with_stop_after_first(single_call),
            )
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

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;
    use xgrammar_structural_tag::ToolChoice;
    use xgrammar_structural_tag::format::Format;

    use super::SeedOssToolParser;
    use crate::output_grammar::test_utils::outline;
    use crate::output_grammar::{OutputGrammarContext, ToolStrictLevel};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::tests::assert_tool_framing_preserves_body_whitespace;
    use crate::tool::{Tool, ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<parameter={name}>{value}</parameter>"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<seed:tool_call>\n<function={function_name}>\n{params}\n</function>\n</seed:tool_call>"
        )
    }

    #[test]
    fn seed_oss_parse_complete_without_tool_call_keeps_text() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn seed_oss_does_not_treat_plain_tool_call_marker_as_start() {
        // The Qwen Coder wrapper `<tool_call>` must NOT trigger the Seed-OSS
        // parser; only `<seed:tool_call>` does.
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser
            .parse_complete("<tool_call>\n<function=get_weather>\n</function>\n</tool_call>")
            .unwrap();

        assert_eq!(
            output.normal_text(),
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
        );
        assert!(output.calls().is_empty());
    }

    #[test]
    fn seed_oss_parse_complete_extracts_single_tool_call() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2026-04-29")],
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF", "date": "2026-04-29" })
        );
    }

    #[test]
    fn seed_oss_parse_complete_preserves_prefix_text() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let input = format!(
            "Thinking... {}",
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.normal_text(), "Thinking... ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn seed_oss_parse_complete_converts_schema_types() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "convert",
                &[
                    ("whole", "5.0"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "42"),
                ],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn seed_oss_streaming_extracts_multiple_tool_calls_in_order() {
        let text = format!(
            "{}\n{}",
            build_tool_call("get_weather", &[("location", "SF")]),
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let chunks = split_by_chars(&text, 7);
        let mut parser = SeedOssToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn seed_oss_streaming_handles_end_marker_split_across_chunks() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        output.append(
            parser
                .parse_chunk(
                    "<seed:tool_call>\n\
                     <function=get_weather>\n\
                     <parameter=location>SF</parameter>\n\
                     </function>\n\
                     </seed:tool",
                )
                .unwrap(),
        );

        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());

        output.append(parser.parse_chunk("_call>").unwrap());
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn seed_oss_finish_fails_incomplete_tool_call() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        parser
            .parse_chunk(
                "<seed:tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>",
            )
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert_eq!(error.to_report_string(), "incomplete Seed-OSS tool call");
    }

    #[test]
    fn tool_framing_preserves_body_whitespace_across_chunk_boundaries() {
        assert_tool_framing_preserves_body_whitespace::<SeedOssToolParser>(
            "\n\n",
            "<seed:tool_call>\n<function=get_weather>\n</function>\n</seed:tool_call>",
        );
    }

    #[test]
    fn seed_oss_builds_the_template_call_grammar() {
        let mut strict_tools = test_tools()[..2].to_vec();
        strict_tools[0].strict = Some(true);
        let build = |tools: &[Tool],
                     tool_choice: ToolChoice,
                     parallel_tool_calls,
                     answer: Option<&Format>| {
            SeedOssToolParser::new(tools)
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
            tags_with_separator `` at_least_one
              tag `\n<seed:tool_call>\n<function=get_weather>\n` qwen_xml({ location?: string, city?: string, state?: string, unit?: string, date?: string, days?: integer }) `</function>\n</seed:tool_call>`
              tag `\n<seed:tool_call>\n<function=add>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
        "#]].assert_eq(&build(&strict_tools, ToolChoice::required(), true, None).unwrap());
        // A named tool allows exactly that one call.
        expect![[r#"
            tags_with_separator `` at_least_one stop_after_first
              tag `\n<seed:tool_call>\n<function=add>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
        "#]].assert_eq(&build(&strict_tools, ToolChoice::function("add"), true, None).unwrap());

        // `auto` builds no grammar unless something asks for one, such as a
        // strict tool, and then leaves text free around the calls.
        assert!(build(&test_tools()[..2], ToolChoice::auto(), true, None).is_none());
        expect![[r#"
            triggered_tags [`<seed:tool_call>`]
              tag `<seed:tool_call>\n<function=get_weather>\n` qwen_xml({ location?: string, city?: string, state?: string, unit?: string, date?: string, days?: integer }) `</function>\n</seed:tool_call>`
              tag `<seed:tool_call>\n<function=add>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
        "#]].assert_eq(&build(&strict_tools, ToolChoice::auto(), true, None).unwrap());
        // Without parallel calls, both stop after the first call.
        expect![[r#"
            tags_with_separator `` at_least_one stop_after_first
              tag `\n<seed:tool_call>\n<function=get_weather>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
              tag `\n<seed:tool_call>\n<function=add>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
        "#]].assert_eq(&build(&test_tools()[..2], ToolChoice::required(), false, None).unwrap());
        expect![[r#"
            triggered_tags [`<seed:tool_call>`] stop_after_first
              tag `<seed:tool_call>\n<function=get_weather>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
              tag `<seed:tool_call>\n<function=add>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
        "#]].assert_eq(&build(&test_tools()[..2], ToolChoice::auto(), false, None).unwrap());

        // With an answer constraint, `auto` holds either the calls or the answer.
        let answer = Format::json_schema(json!({"type": "object"}));
        expect![[r#"
            or
              tags_with_separator `` at_least_one
                tag `\n<seed:tool_call>\n<function=get_weather>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
                tag `\n<seed:tool_call>\n<function=add>\n` qwen_xml(any) `</function>\n</seed:tool_call>`
              json(object)
        "#]].assert_eq(&build(&test_tools()[..2], ToolChoice::auto(), true, Some(&answer)).unwrap());
    }
}
