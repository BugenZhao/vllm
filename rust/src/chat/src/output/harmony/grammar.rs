// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Output grammar for Harmony generations.
//!
//! The renderer leaves the prompt inside an open `<|start|>assistant` header,
//! so the generation starts at the rest of a message header. Each message is
//! `<|channel|>CHANNEL[ to=RECIPIENT][ <|constrain|>TYPE]<|message|>TEXT` and
//! ends with `<|end|>`, after which the next message opens with
//! `<|start|>assistant`, or with a stop token that ends the generation:
//! `<|return|>` after a `final` message, `<|call|>` after a function call.
//!
//! The generation is a run of `analysis` messages followed by its last
//! message. Before a call, the run may also hold user-visible `commentary`
//! messages, in any order; before the `final` message it may not, since the
//! client would receive their text joined to the answer.
//!
//! | Request                     | Last message                    |
//! | --------------------------- | ------------------------------- |
//! | answer constraint, no tools | `final` with the answer         |
//! | `auto` with a tool grammar  | `final`, or one call            |
//! | `required` or named         | one call                        |
//!
//! A `<|call|>` ends the generation, so a generation holds at most one call
//! whatever `parallel_tool_calls` says. The answer constraint, or free text
//! without one, is the content of the `final` message. Headers are the ones
//! the renderer writes in history, which are also the ones gpt-oss generates.
//!
//! The grammar ends with the content of the last message: the engine masks
//! stop tokens until the grammar terminates, so the stop token that ends the
//! message cannot be part of the grammar.
//!
//! Message text and JSON strings exclude the Harmony markers: the Harmony
//! parser reads them as framing wherever they appear, so a marker inside an
//! answer would end it early and could open an undeclared call.

use vllm_parser::output_grammar::{
    self, BuiltOutputGrammar, OutputGrammarContext, normalize_tool_choice,
};
use xgrammar_structural_tag::NormalizedToolChoice;
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam, function_parameters};

const START_ASSISTANT: &str = "<|start|>assistant";
const END: &str = "<|end|>";
/// Markers that frame Harmony messages, kept out of their content.
const MARKERS: &[&str] = &[
    "<|start|>",
    END,
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
];

/// Build the grammar of a whole Harmony generation, or `None` when the request
/// asks for no tool grammar and has no answer constraint.
pub(super) fn build_output_grammar(
    ctx: &OutputGrammarContext<'_>,
) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
    let normalized =
        normalize_tool_choice(ctx)?.filter(|normalized| !normalized.function_tools.is_empty());
    if normalized.is_none() && ctx.answer.is_none() {
        return Ok(None);
    }

    let answer = Format::sequence(vec![
        Format::star(text_message("analysis")),
        last_message(
            "<|channel|>final<|message|>".to_string(),
            ctx.answer.map_or_else(|| Format::any_text_excluding(MARKERS), answer_content),
        ),
    ]);
    let Some(NormalizedToolChoice {
        function_tools,
        choice,
        ..
    }) = normalized
    else {
        return Ok(Some(BuiltOutputGrammar::from_token_zero(answer)));
    };

    let call = Format::sequence(vec![
        Format::star(Format::or(vec![
            text_message("analysis"),
            text_message("commentary"),
        ])),
        one_of(function_tools.iter().map(call_message).collect()),
    ]);
    let format = match choice {
        BuilderToolChoice::Auto => Format::or(vec![answer, call]),
        BuilderToolChoice::Required | BuilderToolChoice::Forced => call,
    };
    Ok(Some(BuiltOutputGrammar::from_token_zero(format)))
}

/// A whole `channel` message without a recipient, and the header opening the
/// next message.
fn text_message(channel: &str) -> Format {
    Format::sequence(vec![
        Format::tag(
            format!("<|channel|>{channel}<|message|>"),
            Format::any_text_excluding(MARKERS),
            END,
        ),
        Format::const_string(START_ASSISTANT),
    ])
}

/// One call to `tool`, with JSON arguments, as the renderer writes it in
/// history.
fn call_message(tool: &FunctionToolParam) -> Format {
    last_message(
        format!(
            "<|channel|>commentary to=functions.{} <|constrain|>json<|message|>",
            tool.function.name
        ),
        Format::JsonSchema(
            JsonSchemaFormat::new(function_parameters(&tool.function)).with_excludes(MARKERS),
        ),
    )
}

/// The user's answer constraint as `final` content, with the markers kept out
/// of the strings of a JSON answer. Other constraints are the user's language
/// and stay as given.
fn answer_content(answer: &Format) -> Format {
    match answer {
        Format::JsonSchema(json) => {
            let mut json = json.clone();
            json.excludes.extend(MARKERS.iter().map(|marker| marker.to_string()));
            Format::JsonSchema(json)
        }
        answer => answer.clone(),
    }
}

/// The message that a stop token ends, and with it the generation: its header
/// and `content`, which the grammar ends with.
fn last_message(header: String, content: Format) -> Format {
    Format::sequence(vec![Format::const_string(header), content])
}

/// `formats` as alternatives, without a wrapper for a single one.
fn one_of(mut formats: Vec<Format>) -> Format {
    if formats.len() == 1 {
        formats.pop().unwrap()
    } else {
        Format::or(formats)
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use serde_json::json;
    use vllm_parser::output_grammar::test_utils::outline;
    use vllm_parser::output_grammar::{GrammarCoverage, ToolStrictLevel};
    use xgrammar_structural_tag::ToolChoice;
    use xgrammar_structural_tag::tool::ToolChoiceValue;

    use super::*;
    use crate::request::ChatTool;

    fn tools(strict: Option<bool>) -> Vec<ChatTool> {
        ["get_weather", "add"]
            .map(|name| ChatTool {
                name: name.to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": { "x": { "type": "number" } },
                    "required": ["x"]
                }),
                strict,
                defer_loading: None,
            })
            .into()
    }

    fn build(
        tools: &[ChatTool],
        tool_choice: ToolChoice,
        answer: Option<Format>,
    ) -> Option<BuiltOutputGrammar> {
        build_output_grammar(&OutputGrammarContext {
            tools,
            tool_choice: &tool_choice,
            tool_strict_level: ToolStrictLevel::Auto,
            parallel_tool_calls: true,
            answer: answer.as_ref(),
        })
        .unwrap()
    }

    fn check(
        tools: &[ChatTool],
        tool_choice: ToolChoice,
        answer: Option<Format>,
        expected: Expect,
    ) {
        let grammar = build(tools, tool_choice, answer).unwrap();
        assert_eq!(grammar.coverage, GrammarCoverage::FromTokenZero);
        expected.assert_eq(&outline(&grammar.format));
    }

    fn answer() -> Option<Format> {
        Some(Format::json_schema(json!({ "type": "object" })))
    }

    #[test]
    fn no_grammar_without_tool_grammar_or_answer() {
        let auto = ToolChoice::Value(ToolChoiceValue::Auto);
        let none = ToolChoice::Value(ToolChoiceValue::None);
        assert_eq!(build(&[], auto.clone(), None), None);
        assert_eq!(build(&tools(None), auto, None), None);
        assert_eq!(build(&tools(Some(true)), none, None), None);
    }

    #[test]
    fn answer_constrains_final_message() {
        check(
            &tools(None),
            ToolChoice::Value(ToolChoiceValue::None),
            answer(),
            expect![[r#"
                sequence
                  star
                    sequence
                      tag `<|channel|>analysis<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                      `<|start|>assistant`
                  sequence
                    `<|channel|>final<|message|>`
                    json(object) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
            "#]],
        );
    }

    #[test]
    fn auto_with_strict_tools_allows_final_or_one_call() {
        check(
            &tools(Some(true)),
            ToolChoice::Value(ToolChoiceValue::Auto),
            None,
            expect![[r#"
                or
                  sequence
                    star
                      sequence
                        tag `<|channel|>analysis<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                        `<|start|>assistant`
                    sequence
                      `<|channel|>final<|message|>`
                      text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
                  sequence
                    star
                      or
                        sequence
                          tag `<|channel|>analysis<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                          `<|start|>assistant`
                        sequence
                          tag `<|channel|>commentary<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                          `<|start|>assistant`
                    or
                      sequence
                        `<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>`
                        json({ x: number }) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
                      sequence
                        `<|channel|>commentary to=functions.add <|constrain|>json<|message|>`
                        json({ x: number }) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
            "#]],
        );
    }

    #[test]
    fn auto_with_answer_keeps_calls_next_to_answer() {
        check(
            &tools(None),
            ToolChoice::Value(ToolChoiceValue::Auto),
            answer(),
            expect![[r#"
                or
                  sequence
                    star
                      sequence
                        tag `<|channel|>analysis<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                        `<|start|>assistant`
                    sequence
                      `<|channel|>final<|message|>`
                      json(object) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
                  sequence
                    star
                      or
                        sequence
                          tag `<|channel|>analysis<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                          `<|start|>assistant`
                        sequence
                          tag `<|channel|>commentary<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                          `<|start|>assistant`
                    or
                      sequence
                        `<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>`
                        json(any) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
                      sequence
                        `<|channel|>commentary to=functions.add <|constrain|>json<|message|>`
                        json(any) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
            "#]],
        );
    }

    #[test]
    fn required_allows_only_calls() {
        check(
            &tools(None),
            ToolChoice::Value(ToolChoiceValue::Required),
            None,
            expect![[r#"
                sequence
                  star
                    or
                      sequence
                        tag `<|channel|>analysis<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                        `<|start|>assistant`
                      sequence
                        tag `<|channel|>commentary<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                        `<|start|>assistant`
                  or
                    sequence
                      `<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>`
                      json(any) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
                    sequence
                      `<|channel|>commentary to=functions.add <|constrain|>json<|message|>`
                      json(any) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
            "#]],
        );
    }

    #[test]
    fn named_tool_allows_one_call_to_it() {
        check(
            &tools(Some(true)),
            ToolChoice::function("add"),
            answer(),
            expect![[r#"
                sequence
                  star
                    or
                      sequence
                        tag `<|channel|>analysis<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                        `<|start|>assistant`
                      sequence
                        tag `<|channel|>commentary<|message|>` text excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`] `<|end|>`
                        `<|start|>assistant`
                  sequence
                    `<|channel|>commentary to=functions.add <|constrain|>json<|message|>`
                    json({ x: number }) excluding [`<|start|>`, `<|end|>`, `<|channel|>`, `<|constrain|>`, `<|message|>`]
            "#]],
        );
    }
}
