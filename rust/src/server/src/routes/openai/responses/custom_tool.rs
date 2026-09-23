// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Function shim through which custom tools are offered to the model.
//!
//! A custom tool takes free-form text instead of a JSON object. `vllm-chat`
//! and the model-specific tool parsers only know function tools, so each
//! custom tool is declared as a function taking one string parameter,
//! `{"input": "<text>"}`. Replayed custom tool calls are wrapped the same way,
//! and generated calls to a custom tool name are unwrapped back into raw text.

use std::collections::HashSet;

use serde_json::{Value, json};

use super::types::CustomToolFormat;

/// Name of the single string parameter of the function shim.
const INPUT: &str = "input";

/// Names of the custom tools declared by one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CustomToolNames(HashSet<String>);

impl CustomToolNames {
    pub fn new(names: impl IntoIterator<Item = String>) -> Self {
        Self(names.into_iter().collect())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
    }
}

/// JSON schema of the function shim offered to the model for a custom tool.
///
/// Grammar formats are described to the model but not enforced.
// TODO: constrain the `input` string with the declared grammar.
pub(super) fn shim_parameters(format: Option<&CustomToolFormat>) -> Value {
    let description = match format {
        None | Some(CustomToolFormat::Text) => {
            "The raw text input of this tool, passed through verbatim.".to_string()
        }
        Some(CustomToolFormat::Grammar { syntax, definition }) => format!(
            "The raw text input of this tool, passed through verbatim. \
             It must conform to the following {syntax} grammar:\n{definition}"
        ),
    };
    json!({
        "type": "object",
        "properties": {
            INPUT: {"type": "string", "description": description},
        },
        "required": [INPUT],
    })
}

/// Wrap the raw input of a replayed custom tool call into shim arguments.
pub(super) fn shim_arguments(input: String) -> String {
    json!({ INPUT: input }).to_string()
}

/// Recover the raw custom tool input from complete shim arguments.
///
/// Arguments that do not match the shim schema are passed through verbatim, so
/// the client can report the malformed call back to the model.
pub(super) fn raw_input(arguments: String) -> String {
    match serde_json::from_str::<Value>(&arguments) {
        Ok(Value::Object(mut object)) => match object.remove(INPUT) {
            Some(Value::String(input)) if object.is_empty() => input,
            _ => arguments,
        },
        _ => arguments,
    }
}

/// Incremental decoder of the raw input from streamed shim arguments.
///
/// While the arguments spell the shim prefix `{"input": "`, the decoder
/// unescapes the JSON string value as it arrives. Decoding stops at the closing
/// quote, or for good at the first byte that departs from the shim shape, in
/// which case [`raw_input`] of the complete arguments decides the final input.
/// Escape sequences split across deltas are held back until complete.
#[derive(Debug, Default)]
pub(super) struct InputDecoder {
    state: DecoderState,
    /// Arguments text not decoded yet: an incomplete prefix or escape.
    pending: String,
    /// A decoded high surrogate waiting for its low half.
    high_surrogate: Option<u16>,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum DecoderState {
    /// Matching the shim prefix up to the opening quote of the value.
    #[default]
    Prefix,
    /// Decoding the string value.
    Value,
    /// The value ended or the arguments departed from the shim shape.
    Stopped,
}

/// One escape sequence at the start of the undecoded value text.
enum Escape {
    /// A complete two-byte escape of one character.
    Char(char),
    /// A complete `\uXXXX` escape of one UTF-16 code unit.
    Unit(u16),
    /// More bytes are needed to decide.
    Incomplete,
    /// Not a valid JSON escape.
    Invalid,
}

impl InputDecoder {
    /// Consume one arguments delta and return the newly decoded input text.
    pub fn push(&mut self, delta: &str) -> String {
        let mut decoded = String::new();
        if self.state == DecoderState::Stopped {
            return decoded;
        }
        self.pending.push_str(delta);
        if self.state == DecoderState::Prefix {
            match match_prefix(&self.pending) {
                PrefixMatch::Complete(len) => {
                    self.pending.drain(..len);
                    self.state = DecoderState::Value;
                }
                PrefixMatch::Partial => return decoded,
                PrefixMatch::Mismatch => {
                    self.stop();
                    return decoded;
                }
            }
        }

        let pending = std::mem::take(&mut self.pending);
        let mut rest = pending.as_str();
        while let Some(special) = rest.find(['\\', '"']) {
            if special > 0 {
                self.push_str(&rest[..special], &mut decoded);
            }
            rest = &rest[special..];
            if rest.starts_with('"') {
                self.flush_surrogate(&mut decoded);
                self.stop();
                return decoded;
            }
            match parse_escape(rest) {
                Escape::Char(ch) => {
                    self.flush_surrogate(&mut decoded);
                    decoded.push(ch);
                    rest = &rest[2..];
                }
                Escape::Unit(unit) => {
                    self.push_unit(unit, &mut decoded);
                    rest = &rest[6..];
                }
                Escape::Incomplete => {
                    self.pending = rest.to_string();
                    return decoded;
                }
                Escape::Invalid => {
                    self.stop();
                    return decoded;
                }
            }
        }
        if !rest.is_empty() {
            self.push_str(rest, &mut decoded);
        }
        decoded
    }

    fn stop(&mut self) {
        self.state = DecoderState::Stopped;
        self.pending.clear();
    }

    fn push_str(&mut self, text: &str, decoded: &mut String) {
        self.flush_surrogate(decoded);
        decoded.push_str(text);
    }

    /// Decode one UTF-16 code unit, pairing surrogates across escapes.
    fn push_unit(&mut self, unit: u16, decoded: &mut String) {
        match unit {
            0xD800..=0xDBFF => {
                self.flush_surrogate(decoded);
                self.high_surrogate = Some(unit);
            }
            0xDC00..=0xDFFF => match self.high_surrogate.take() {
                Some(high) => decoded.extend(char::decode_utf16([high, unit]).flatten()),
                None => decoded.push(char::REPLACEMENT_CHARACTER),
            },
            _ => {
                self.flush_surrogate(decoded);
                decoded.extend(char::from_u32(unit.into()));
            }
        }
    }

    /// Emit a high surrogate that was not followed by its low half.
    fn flush_surrogate(&mut self, decoded: &mut String) {
        if self.high_surrogate.take().is_some() {
            decoded.push(char::REPLACEMENT_CHARACTER);
        }
    }
}

enum PrefixMatch {
    /// The prefix is complete and spans this many bytes.
    Complete(usize),
    /// The text so far is a proper prefix of the shim prefix.
    Partial,
    Mismatch,
}

/// Match `{"input": "` at the start of `text`, allowing JSON whitespace
/// between tokens.
fn match_prefix(text: &str) -> PrefixMatch {
    const TOKENS: [(bool, &str); 6] = [
        (true, "{"),
        (true, "\""),
        (false, INPUT),
        (false, "\""),
        (true, ":"),
        (true, "\""),
    ];

    let bytes = text.as_bytes();
    let mut pos = 0;
    for (whitespace_before, token) in TOKENS {
        if whitespace_before {
            while bytes.get(pos).is_some_and(|byte| b" \t\n\r".contains(byte)) {
                pos += 1;
            }
        }
        let rest = &bytes[pos..];
        if rest.len() < token.len() {
            return if token.as_bytes().starts_with(rest) {
                PrefixMatch::Partial
            } else {
                PrefixMatch::Mismatch
            };
        }
        if !rest.starts_with(token.as_bytes()) {
            return PrefixMatch::Mismatch;
        }
        pos += token.len();
    }
    PrefixMatch::Complete(pos)
}

/// Parse the escape sequence at the start of `text`, which starts with `\`.
fn parse_escape(text: &str) -> Escape {
    let bytes = text.as_bytes();
    let Some(&kind) = bytes.get(1) else {
        return Escape::Incomplete;
    };
    let ch = match kind {
        b'"' => '"',
        b'\\' => '\\',
        b'/' => '/',
        b'b' => '\u{8}',
        b'f' => '\u{c}',
        b'n' => '\n',
        b'r' => '\r',
        b't' => '\t',
        b'u' => {
            let hex = &bytes[2..bytes.len().min(6)];
            if !hex.iter().all(u8::is_ascii_hexdigit) {
                return Escape::Invalid;
            }
            if hex.len() < 4 {
                return Escape::Incomplete;
            }
            let hex = std::str::from_utf8(hex).expect("ASCII hex digits are UTF-8");
            return Escape::Unit(u16::from_str_radix(hex, 16).expect("four hex digits fit u16"));
        }
        _ => return Escape::Invalid,
    };
    Escape::Char(ch)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{InputDecoder, raw_input, shim_arguments};

    /// Decode `arguments` split into chunks of `size` characters.
    fn decode_in_chunks(arguments: &str, size: usize) -> String {
        let chars: Vec<char> = arguments.chars().collect();
        let mut decoder = InputDecoder::default();
        chars
            .chunks(size)
            .map(|chunk| decoder.push(&chunk.iter().collect::<String>()))
            .collect()
    }

    #[test]
    fn raw_input_recovers_text_from_shim_arguments() {
        let input = "*** Begin Patch\n*** Add File: a.txt\n+\"quoted\"\n*** End Patch";
        assert_eq!(raw_input(shim_arguments(input.to_string())), input);
        // Arguments outside the shim schema are passed through verbatim.
        for arguments in ["not json", r#"{"patch":"x"}"#, r#"{"input":"x","extra":1}"#] {
            assert_eq!(raw_input(arguments.to_string()), arguments);
        }
    }

    #[test]
    fn decoder_streams_the_final_input_at_every_split() {
        let input = "*** Begin Patch\n+say \"hi\" \\ tab\t/ \u{1}é中🦀\n*** End Patch\n";
        for arguments in [
            shim_arguments(input.to_string()),
            // Escaped non-ASCII, including a surrogate pair, and spaced tokens.
            format!(
                "{{ \"input\" :\n \"{}\" }}",
                serde_json::to_string(input)
                    .unwrap()
                    .trim_matches('"')
                    .replace('🦀', "\\ud83e\\udd80")
                    .replace('中', "\\u4e2d")
            ),
        ] {
            assert_eq!(raw_input(arguments.clone()), input, "{arguments}");
            for size in 1..=8 {
                assert_eq!(
                    decode_in_chunks(&arguments, size),
                    input,
                    "size={size} {arguments}"
                );
            }
        }
    }

    #[test]
    fn decoder_stops_when_arguments_depart_from_the_shim() {
        for arguments in [
            r#"{"patch": "x"}"#,
            r#"{"inputs": "x"}"#,
            r#"["x"]"#,
            r#"{"input": 1}"#,
        ] {
            assert_eq!(decode_in_chunks(arguments, 1), "", "{arguments}");
        }
        // Text before an invalid escape is kept; nothing after it.
        assert_eq!(decode_in_chunks(r#"{"input": "ab\qcd"}"#, 3), "ab");
        // A lone surrogate decodes to the replacement character.
        assert_eq!(
            decode_in_chunks(
                &json!({"input": "x"}).to_string().replace('x', "\\ud800x"),
                2
            ),
            "\u{fffd}x"
        );
    }
}
