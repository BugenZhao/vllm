// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::borrow::Cow;

use serde::de::DeserializeOwned;

/// Parse a JSON config file shipped with a checkpoint, such as `config.json`
/// or `tokenizer_config.json`.
///
/// Python's `json` module, which reads these files for the Python frontend and
/// the engine, accepts the non-standard literals `NaN`, `Infinity` and
/// `-Infinity`, and some checkpoints ship them (Mamba2-style configs set
/// `"time_step_limit": [0.0, Infinity]`). `serde_json` rejects them and
/// `serde_json::Value` cannot hold non-finite numbers, so each literal is read
/// as a string of its own spelling: a field the frontend ignores loads as
/// before, and a numeric field it reads fails to deserialize instead of
/// silently taking a default. Request JSON keeps strict parsing.
pub fn parse_checkpoint_json<T: DeserializeOwned>(content: &str) -> serde_json::Result<T> {
    serde_json::from_str(&quote_non_finite_literals(content))
}

/// Quote the `NaN`, `Infinity` and `-Infinity` literals in value position,
/// borrowing the input when it has none. A literal anywhere else, such as an
/// object key, is left for `serde_json` to reject, as Python does.
fn quote_non_finite_literals(content: &str) -> Cow<'_, str> {
    const LITERALS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];

    let bytes = content.as_bytes();
    let mut output = None::<String>;
    let mut copied = 0;
    let mut in_string = false;
    let mut escaped = false;
    // Open containers and the last non-whitespace byte outside strings, which
    // together tell whether the next token is a value.
    let mut containers = Vec::new();
    let mut last = None;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            index += 1;
            continue;
        }
        let in_value_position = match last {
            None | Some(b':' | b'[') => true,
            Some(b',') => containers.last() == Some(&b'['),
            _ => false,
        };
        if in_value_position
            && let Some(literal) = LITERALS
                .into_iter()
                .find(|literal| bytes[index..].starts_with(literal.as_bytes()))
        {
            // Literals are ASCII, so both slice bounds are char boundaries.
            let output = output.get_or_insert_with(|| String::with_capacity(content.len() + 2));
            output.push_str(&content[copied..index]);
            output.push('"');
            output.push_str(literal);
            output.push('"');
            index += literal.len();
            copied = index;
            last = Some(b'"');
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => containers.push(byte),
            b'}' | b']' => _ = containers.pop(),
            _ => {}
        }
        if !byte.is_ascii_whitespace() {
            last = Some(byte);
        }
        index += 1;
    }

    match output {
        Some(mut output) => {
            output.push_str(&content[copied..]);
            Cow::Owned(output)
        }
        None => Cow::Borrowed(content),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::parse_checkpoint_json;

    #[test]
    fn non_finite_literals_read_as_their_spelling() {
        let value: Value = parse_checkpoint_json(
            r#"{
                "time_step_limit": [0.0, Infinity],
                "low": -Infinity,
                "nan": NaN,
                "text": "NaN \"Infinity\" -Infinity\\",
                "vocab_size": 131072
            }"#,
        )
        .unwrap();

        assert_eq!(
            value,
            json!({
                "time_step_limit": [0.0, "Infinity"],
                "low": "-Infinity",
                "nan": "NaN",
                "text": "NaN \"Infinity\" -Infinity\\",
                "vocab_size": 131072,
            })
        );
        for invalid in [
            r#"{"limit": Inf}"#,
            r#"{NaN: 1}"#,
            r#"{"a": 1, Infinity: 2}"#,
        ] {
            assert!(
                parse_checkpoint_json::<Value>(invalid).is_err(),
                "{invalid}"
            );
        }
    }
}
