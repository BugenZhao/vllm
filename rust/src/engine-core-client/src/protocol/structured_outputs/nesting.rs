// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Nesting cap on structured-output JSON, mirrored from Python's
//! `check_json_nesting`.

use serde_json::Value;

use crate::error::{Error, Result};

/// Upper bound on nested objects and arrays in a structured-output JSON schema
/// or structural tag. EngineCore's grammar conversion cost grows steeply with
/// nesting. OpenAI allows 10 levels of schema nesting.
const MAX_JSON_NESTING: usize = 64;

/// Extra nesting allowed for a structural tag, which wraps its JSON schemas in
/// tag structure, such as the tool-call tags built by output parsers.
const STRUCTURAL_TAG_WRAPPER_NESTING: usize = 16;

/// Reject a JSON schema nested more than [`MAX_JSON_NESTING`] levels deep.
pub(super) fn check_json_schema(schema: &Value) -> Result<()> {
    ensure_within(value_exceeds(schema, MAX_JSON_NESTING))
}

/// Reject a JSON schema string nested more than [`MAX_JSON_NESTING`] levels
/// deep, before it is parsed.
pub(super) fn check_json_schema_text(schema: &str) -> Result<()> {
    ensure_within(text_exceeds(schema, MAX_JSON_NESTING))
}

/// Reject a structural tag nested more than [`MAX_JSON_NESTING`] plus
/// [`STRUCTURAL_TAG_WRAPPER_NESTING`] levels deep.
pub(super) fn check_structural_tag(tag: &str) -> Result<()> {
    ensure_within(text_exceeds(
        tag,
        MAX_JSON_NESTING + STRUCTURAL_TAG_WRAPPER_NESTING,
    ))
}

fn ensure_within(exceeds: bool) -> Result<()> {
    if exceeds {
        return Err(Error::InvalidStructuredOutputsParams {
            message: format!(
                "The JSON schema is nested too deeply: objects and arrays may nest at most \
                 {MAX_JSON_NESTING} levels"
            ),
        });
    }
    Ok(())
}

/// Whether `value` nests objects and arrays deeper than `limit`, walked
/// iteratively so that the check itself cannot overflow the stack.
fn value_exceeds(value: &Value, limit: usize) -> bool {
    let mut stack = vec![(value, 1)];
    while let Some((value, depth)) = stack.pop() {
        match value {
            Value::Object(_) | Value::Array(_) if depth > limit => return true,
            Value::Object(object) => stack.extend(object.values().map(|child| (child, depth + 1))),
            Value::Array(array) => stack.extend(array.iter().map(|child| (child, depth + 1))),
            _ => {}
        }
    }
    false
}

/// Whether JSON `text` nests objects and arrays deeper than `limit`.
///
/// Scans brackets outside string literals instead of parsing, so a document
/// deeper than `serde_json`'s recursion limit is reported as too deep rather
/// than malformed. Malformed JSON is left to later validation.
fn text_exceeds(text: &str, limit: usize) -> bool {
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escaped = false;
    for byte in text.bytes() {
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > limit {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A JSON document with `depth` nested containers, alternating objects and
    /// arrays, with brackets inside a string that must not count.
    fn nested(depth: usize) -> Value {
        (0..depth).fold(json!("[{"), |inner, level| {
            if level % 2 == 0 {
                json!({ "a": inner })
            } else {
                json!([inner])
            }
        })
    }

    #[test]
    fn value_and_text_agree_at_the_limit() {
        for (depth, exceeds) in [(0, false), (3, false), (4, true), (8, true)] {
            let value = nested(depth);
            assert_eq!(value_exceeds(&value, 3), exceeds, "{value}");
            assert_eq!(text_exceeds(&value.to_string(), 3), exceeds, "{value}");
        }
    }

    #[test]
    fn text_ignores_escaped_quotes_in_strings() {
        assert!(!text_exceeds(r#"{"a": "\"[[[[\""}"#, 1));
        assert!(text_exceeds(r#"{"a": "\\", "b": [[]]}"#, 2));
    }
}
