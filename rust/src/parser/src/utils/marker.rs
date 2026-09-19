// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Token-aware marker matching for streaming parsers.
//!
//! Model protocols delimit channels with dedicated special tokens such as
//! `<|close|>` and `<|sep|>`. Ordinary BPE tokens can decode to the very same
//! spelling, so matching a marker by text alone mistakes model-written content
//! for structure. The helpers here keep text as the parse axis and consult the
//! generated-token attribution carried by [`DecodedText`] only where a marker's
//! bytes must have come from a special token.
//!
//! - [`Attributed`] is a winnow input over one `DecodedText` buffer: the text
//!   plus its buffer-local [`Anchors`], composed from winnow's own stream
//!   wrappers so every existing combinator keeps working.
//! - [`special`] matches a special token's spelling only when a single token
//!   with the expected ID produced it.
//! - [`Marker`] is a fixed marker string with guarded special-token segments,
//!   built once from tokenizer IDs. It is both the parser that consumes the
//!   marker and, through [`MarkerLike`], the definition the safe-text scanner in
//!   the parent module uses to stop in front of it, so the two never disagree.
//! - [`MarkerStream`] abstracts over `Partial<&str>` (text-only parsers) and
//!   [`Attributed`], letting one generic scanner serve both.

use std::ops::Range;

use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};
use winnow::Parser;
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::stream::{
    Compare, FindSlice, LocatingSlice, Location, Partial, Stateful, Stream, StreamIsPartial,
};
use winnow::token::literal;

use super::partial_prefix_len;

/// Whether parser input carries generated-token attribution.
///
/// This is a property of the caller's pipeline, known before the first delta,
/// so it is chosen once when the parser is constructed and never inferred from
/// the data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttributionMode {
    /// Every delta carries one [`TokenAttribution`] per generated token, as
    /// produced by the incremental detokenizer. Structural markers must be
    /// spelled by their dedicated special tokens; equal text from ordinary
    /// tokens stays content.
    #[default]
    Tokens,
    /// Deltas carry text only. Markers are matched by spelling alone, which
    /// cannot tell model-written marker text from structure.
    TextOnly,
}

/// Buffer-local token anchors of the text being parsed.
#[derive(Clone, Copy, Debug)]
pub enum Anchors<'i> {
    /// No attribution available: every spelling check passes.
    TextOnly,
    /// One record per generated token, sorted by byte offset. May be empty for
    /// an empty buffer.
    Tokens(&'i [TokenAttribution]),
}

impl<'i> Anchors<'i> {
    fn attributions(self) -> &'i [TokenAttribution] {
        match self {
            Anchors::TextOnly => &[],
            Anchors::Tokens(attributions) => attributions,
        }
    }

    /// Visible anchors within `range`, as `(byte_offset, token_id)`, in
    /// generation order.
    fn visible_in(self, range: Range<usize>) -> impl Iterator<Item = (usize, u32)> + 'i {
        let attributions = self.attributions();
        let first = attributions
            .partition_point(|attribution| anchor_offset(attribution.anchor) < range.start);
        attributions[first..]
            .iter()
            .take_while(move |attribution| anchor_offset(attribution.anchor) < range.end)
            .filter_map(|attribution| match attribution.anchor {
                TokenAnchor::Visible { byte_offset } => {
                    Some((byte_offset as usize, attribution.token_id))
                }
                TokenAnchor::ZeroWidth { .. } => None,
            })
    }

    /// Return whether `range` is exactly the visible span of the single token
    /// `token_id`: one visible anchor at `range.start` with that ID and no other
    /// visible anchor inside. Zero-width records are ignored.
    ///
    /// Special tokens never merge with neighbouring text, so this is the
    /// identity test for a special token's spelling. Anchors beyond the
    /// available input do not exist yet, so a partially arrived genuine
    /// spelling passes and is left to `literal` to report as incomplete.
    pub fn is_single_token(self, range: Range<usize>, token_id: u32) -> bool {
        match self {
            Anchors::TextOnly => true,
            Anchors::Tokens(_) => {
                let mut visible = self.visible_in(range.clone());
                visible.next() == Some((range.start, token_id)) && visible.next().is_none()
            }
        }
    }

    /// Earliest visible anchor at or after `from` whose token ID satisfies
    /// `accept`.
    fn find_visible(self, from: usize, accept: impl Fn(u32) -> bool) -> Option<usize> {
        self.visible_in(from..usize::MAX)
            .find(|&(_, token_id)| accept(token_id))
            .map(|(at, _)| at)
    }
}

fn anchor_offset(anchor: TokenAnchor) -> usize {
    match anchor {
        TokenAnchor::Visible { byte_offset } | TokenAnchor::ZeroWidth { byte_offset } => {
            byte_offset as usize
        }
    }
}

/// Streaming winnow input over one [`DecodedText`] buffer.
///
/// `LocatingSlice` reports the cursor's byte offset from the buffer start, which
/// is the coordinate system of the anchors; `Partial` keeps the incomplete-input
/// semantics streaming parsers rely on; `Stateful` carries the immutable anchors.
/// winnow forwards `Compare`, `FindSlice`, `Location`, and `UpdateSlice` through
/// all three, so `literal`, `alt`, `seq!`, `take_until`, and `rest` work unchanged.
pub type Attributed<'i> = Stateful<Partial<LocatingSlice<&'i str>>, Anchors<'i>>;

/// Create an [`Attributed`] input over `buffer` under `mode`.
pub fn attributed(buffer: &DecodedText, mode: AttributionMode) -> Attributed<'_> {
    Stateful {
        input: Partial::new(LocatingSlice::new(buffer.text.as_str())),
        state: match mode {
            AttributionMode::Tokens => Anchors::Tokens(&buffer.attributions),
            AttributionMode::TextOnly => Anchors::TextOnly,
        },
    }
}

/// A `&str`-sliced streaming input that can say where markers may begin.
///
/// Implemented for `Partial<&str>` (text-only parsers) and [`Attributed`]. The
/// difference between the two is only the *source of candidates*: text
/// occurrences of a spelling versus visible anchors of a marker's first special
/// token. The scanner built on top is shared.
pub trait MarkerStream<'i>:
    Stream<Slice = &'i str> + StreamIsPartial + Clone + for<'m> Compare<&'m str>
{
    /// Remaining input from the cursor.
    fn remaining(&self) -> &'i str;

    /// Byte offset of the cursor in the coordinate system used by
    /// [`Self::next_candidate`] and by resumable scan state.
    ///
    /// `Partial<&str>` tracks no origin and reports 0, so its offsets are
    /// relative to the cursor at the time of the call; [`Attributed`] reports
    /// the offset from the buffer start.
    fn offset(&self) -> usize;

    /// Token anchors of the buffer.
    fn anchors(&self) -> Anchors<'i>;

    /// Earliest offset at or after `from` where one of `markers` may begin:
    /// a complete spelling, a partial spelling at the end of the input, or a
    /// visible anchor of a marker's first special token.
    fn next_candidate<M: MarkerLike>(&self, markers: &[M], from: usize) -> Option<usize>;
}

impl<'i> MarkerStream<'i> for Partial<&'i str> {
    fn remaining(&self) -> &'i str {
        **self
    }

    fn offset(&self) -> usize {
        0
    }

    fn anchors(&self) -> Anchors<'i> {
        Anchors::TextOnly
    }

    fn next_candidate<M: MarkerLike>(&self, markers: &[M], from: usize) -> Option<usize> {
        text_candidate(
            self.remaining(),
            markers.iter().map(|marker| marker.text()),
            from,
        )
    }
}

impl<'i> MarkerStream<'i> for Attributed<'i> {
    fn remaining(&self) -> &'i str {
        ****self
    }

    fn offset(&self) -> usize {
        self.current_token_start()
    }

    fn anchors(&self) -> Anchors<'i> {
        self.state
    }

    fn next_candidate<M: MarkerLike>(&self, markers: &[M], from: usize) -> Option<usize> {
        let start = self.offset();
        let from = from.max(start);
        // Guarded markers can only begin at a visible anchor of their first
        // special token; the rest (and everything in text-only mode) fall back
        // to text occurrences.
        let text_only = matches!(self.state, Anchors::TextOnly);
        let anchored = (!text_only)
            .then(|| {
                self.state.find_visible(from, |token_id| {
                    markers.iter().any(|marker| marker.first_token_id() == Some(token_id))
                })
            })
            .flatten();
        let by_text = text_candidate(
            self.remaining(),
            markers
                .iter()
                .filter(|marker| text_only || marker.first_token_id().is_none())
                .map(|marker| marker.text()),
            from - start,
        )
        .map(|at| start + at);
        match (anchored, by_text) {
            (Some(anchored), Some(by_text)) => Some(anchored.min(by_text)),
            (anchored, by_text) => anchored.or(by_text),
        }
    }
}

/// Earliest text position at or after `from` where one of `spellings` occurs
/// in full, or where a proper prefix of one of them ends the input.
fn text_candidate<'m>(
    text: &str,
    spellings: impl Iterator<Item = &'m str> + Clone,
    from: usize,
) -> Option<usize> {
    let from = floor_char_boundary(text, from);
    let tail = &text[from..];
    if tail.is_empty() {
        return None;
    }
    if let Some(at) = find_earliest(tail, spellings.clone()) {
        return Some(from + at);
    }
    let keep_len = spellings.map(|spelling| partial_prefix_len(tail, spelling)).max().unwrap_or(0);
    (keep_len > 0).then(|| text.len() - keep_len)
}

/// Earliest occurrence of any spelling in `text`.
#[inline(always)]
fn find_earliest<'m>(text: &str, mut spellings: impl Iterator<Item = &'m str>) -> Option<usize> {
    let first = spellings.next()?;
    let second = spellings.next();
    let third = second.and_then(|_| spellings.next());
    let fourth = third.and_then(|_| spellings.next());
    // Use the fast specialized `FindSlice` impls for 1-3 spellings, and fall
    // back to a linear scan for 4+.
    let range = match (second, third, fourth) {
        (None, _, _) => text.find_slice(first),
        (Some(second), None, _) => text.find_slice((first, second)),
        (Some(second), Some(third), None) => text.find_slice((first, second, third)),
        (Some(second), Some(third), Some(fourth)) => {
            return [first, second, third, fourth]
                .into_iter()
                .chain(spellings)
                .filter_map(|spelling| text.find(spelling))
                .min();
        }
    };
    range.map(|range| range.start)
}

pub(super) fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// A fixed marker: pure text, or text with guarded special-token segments.
///
/// Implemented for `str` (and references), so text-only parsers keep passing
/// `&[&str]`, and for [`Marker`].
pub trait MarkerLike {
    /// The full spelling of the marker.
    fn text(&self) -> &str;

    /// The special token the marker begins with, if it begins with one.
    fn first_token_id(&self) -> Option<u32>;

    /// Parse the marker at the cursor, returning its spelling.
    fn parse_marker<'i, I: MarkerStream<'i>>(&self, input: &mut I) -> ModalResult<&'i str>;
}

impl MarkerLike for str {
    fn text(&self) -> &str {
        self
    }

    fn first_token_id(&self) -> Option<u32> {
        None
    }

    fn parse_marker<'i, I: MarkerStream<'i>>(&self, input: &mut I) -> ModalResult<&'i str> {
        literal(self).parse_next(input)
    }
}

impl MarkerLike for String {
    fn text(&self) -> &str {
        self
    }

    fn first_token_id(&self) -> Option<u32> {
        None
    }

    fn parse_marker<'i, I: MarkerStream<'i>>(&self, input: &mut I) -> ModalResult<&'i str> {
        self.as_str().parse_marker(input)
    }
}

impl<M: MarkerLike + ?Sized> MarkerLike for &M {
    fn text(&self) -> &str {
        (**self).text()
    }

    fn first_token_id(&self) -> Option<u32> {
        (**self).first_token_id()
    }

    fn parse_marker<'i, I: MarkerStream<'i>>(&self, input: &mut I) -> ModalResult<&'i str> {
        (**self).parse_marker(input)
    }
}

/// Parse the spelling of one special token, only when a single token with ID
/// `token_id` produced it.
///
/// The identity check runs before `literal`: when bytes are available at the
/// cursor and they do not belong to that token, the parser backtracks at once
/// instead of asking for more input, so an ordinary partial lookalike streams
/// as content immediately. A genuine spelling split by output holdback still
/// reports incomplete, because its anchor arrived with its first byte. Nothing
/// is consumed on either failure path. Under [`Anchors::TextOnly`] this is
/// plain `literal`.
pub fn special<'i, 's, I: MarkerStream<'i>>(
    spelling: &'s str,
    token_id: u32,
) -> impl Parser<I, &'i str, ErrMode<ContextError>> + 's {
    move |input: &mut I| {
        let start = input.offset();
        if input.eof_offset() > 0
            && !input.anchors().is_single_token(start..start + spelling.len(), token_id)
        {
            return Err(ErrMode::Backtrack(ContextError::new()));
        }
        literal(spelling).parse_next(input)
    }
}

/// A fixed marker string whose special-token segments are guarded by token
/// identity.
///
/// Build it once from tokenizer IDs, then use it both as the parser that
/// consumes the marker (it implements [`Parser`] by reference) and as the
/// definition the safe-text scanner stops in front of.
///
/// ```ignore
/// let think_close = Marker::special("<|close|>", close_id)
///     .then_text("think")
///     .then_special("<|sep|>", sep_id);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Marker {
    text: String,
    guards: Vec<Guard>,
}

/// One special-token segment of a [`Marker`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct Guard {
    /// Byte range of the spelling within `Marker::text`.
    range: Range<usize>,
    token_id: u32,
}

impl Marker {
    /// A marker matched by spelling alone.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            guards: Vec::new(),
        }
    }

    /// A marker beginning with the special token `token_id` spelled `spelling`.
    pub fn special(spelling: &str, token_id: u32) -> Self {
        Self::text(String::new()).then_special(spelling, token_id)
    }

    /// Append ordinary text, matched by spelling.
    #[must_use]
    pub fn then_text(mut self, text: &str) -> Self {
        self.text.push_str(text);
        self
    }

    /// Append the special token `token_id` spelled `spelling`.
    #[must_use]
    pub fn then_special(mut self, spelling: &str, token_id: u32) -> Self {
        let start = self.text.len();
        self.text.push_str(spelling);
        self.guards.push(Guard {
            range: start..self.text.len(),
            token_id,
        });
        self
    }

    /// The full spelling of the marker.
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl MarkerLike for Marker {
    fn text(&self) -> &str {
        &self.text
    }

    fn first_token_id(&self) -> Option<u32> {
        self.guards
            .first()
            .filter(|guard| guard.range.start == 0)
            .map(|guard| guard.token_id)
    }

    fn parse_marker<'i, I: MarkerStream<'i>>(&self, input: &mut I) -> ModalResult<&'i str> {
        let checkpoint = input.checkpoint();
        let mut end = 0;
        let parsed = self
            .guards
            .iter()
            .try_for_each(|guard| {
                literal(&self.text[end..guard.range.start]).parse_next(input)?;
                special(&self.text[guard.range.clone()], guard.token_id).parse_next(input)?;
                end = guard.range.end;
                Ok(())
            })
            .and_then(|()| literal(&self.text[end..]).parse_next(input));
        // Hand the whole spelling back as one slice.
        let len = input.offset_from(&checkpoint);
        input.reset(&checkpoint);
        parsed.map(|_| input.next_slice(len))
    }
}

impl<'i, I: MarkerStream<'i>> Parser<I, &'i str, ErrMode<ContextError>> for &Marker {
    fn parse_next(&mut self, input: &mut I) -> ModalResult<&'i str> {
        self.parse_marker(input)
    }
}

#[cfg(test)]
mod tests {
    use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};
    use winnow::Parser;
    use winnow::error::ErrMode;
    use winnow::stream::{Partial, Stream};

    use super::{AttributionMode, Marker, MarkerLike, MarkerStream, attributed, special};

    const CLOSE_ID: u32 = 256;
    const SEP_ID: u32 = 258;

    /// Build attributed text from `(piece, token_id)` pairs, one visible anchor
    /// at the first byte of each piece.
    fn pieces(pieces: &[(&str, u32)]) -> DecodedText {
        let mut decoded = DecodedText::default();
        for &(piece, token_id) in pieces {
            decoded.attributions.push(TokenAttribution {
                token_id,
                anchor: TokenAnchor::Visible {
                    byte_offset: decoded.text.len() as u32,
                },
            });
            decoded.text.push_str(piece);
        }
        decoded
    }

    fn think_close() -> Marker {
        Marker::special("<|close|>", CLOSE_ID)
            .then_text("think")
            .then_special("<|sep|>", SEP_ID)
    }

    /// `<|close|>` spelled by ordinary tokens.
    fn ordinary_close() -> Vec<(&'static str, u32)> {
        vec![("<", 60), ("|", 124), ("close", 900), ("|", 124), (">", 62)]
    }

    #[test]
    fn special_accepts_single_token_with_expected_id() {
        let buffer = pieces(&[("<|close|>", CLOSE_ID), ("x", 1)]);
        let mut input = attributed(&buffer, AttributionMode::Tokens);

        let matched = special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap();

        assert_eq!(matched, "<|close|>");
        assert_eq!(input.remaining(), "x");
    }

    #[test]
    fn special_rejects_ordinary_tokens_with_same_spelling() {
        let mut fixture = ordinary_close();
        fixture.push(("x", 1));
        let buffer = pieces(&fixture);
        let mut input = attributed(&buffer, AttributionMode::Tokens);

        let error = special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Backtrack(_)));
        assert_eq!(input.remaining(), "<|close|>x", "nothing consumed");
    }

    #[test]
    fn special_rejects_single_ordinary_token_with_wrong_id() {
        let buffer = pieces(&[("<|close|>", 900)]);
        let mut input = attributed(&buffer, AttributionMode::Tokens);

        let error = special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Backtrack(_)));
    }

    #[test]
    fn special_reports_incomplete_for_split_genuine_spelling() {
        // Output holdback delivered only the first bytes; the anchor came with them.
        let buffer = pieces(&[("<|clo", CLOSE_ID)]);
        let mut input = attributed(&buffer, AttributionMode::Tokens);

        let error = special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
    }

    #[test]
    fn special_rejects_partial_lookalike_without_waiting() {
        let buffer = pieces(&[("<", 60), ("|", 124), ("clo", 901)]);
        let mut input = attributed(&buffer, AttributionMode::Tokens);

        let error = special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Backtrack(_)));
    }

    #[test]
    fn special_ignores_zero_width_records_inside_span() {
        let mut buffer = pieces(&[("<|close|>", CLOSE_ID)]);
        buffer.attributions.push(TokenAttribution {
            token_id: 7,
            anchor: TokenAnchor::ZeroWidth { byte_offset: 3 },
        });
        buffer.attributions.sort_by_key(|attribution| match attribution.anchor {
            TokenAnchor::Visible { byte_offset } | TokenAnchor::ZeroWidth { byte_offset } => {
                byte_offset
            }
        });
        let mut input = attributed(&buffer, AttributionMode::Tokens);

        special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap();
    }

    #[test]
    fn special_is_literal_in_text_only_mode() {
        let buffer = pieces(&ordinary_close());
        let mut input = attributed(&buffer, AttributionMode::TextOnly);

        special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap();

        let mut input = Partial::new("<|close|>");
        special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap();
    }

    #[test]
    fn marker_requires_every_special_segment() {
        let marker = think_close();
        let genuine = pieces(&[("<|close|>", CLOSE_ID), ("think", 5), ("<|sep|>", SEP_ID)]);
        let mut input = attributed(&genuine, AttributionMode::Tokens);
        assert_eq!(
            (&marker).parse_next(&mut input).unwrap(),
            "<|close|>think<|sep|>"
        );

        // Real close, ordinary separator: the strict rule keeps it as content.
        let mixed = pieces(&[("<|close|>", CLOSE_ID), ("think", 5), ("<|sep|>", 902)]);
        let mut input = attributed(&mixed, AttributionMode::Tokens);
        assert!(matches!(
            (&marker).parse_next(&mut input).unwrap_err(),
            ErrMode::Backtrack(_)
        ));
        assert_eq!(
            input.remaining(),
            "<|close|>think<|sep|>",
            "nothing consumed"
        );
    }

    #[test]
    fn marker_first_token_id_requires_leading_special() {
        assert_eq!(think_close().first_token_id(), Some(CLOSE_ID));
        assert_eq!(Marker::text("</think>").first_token_id(), None);
        assert_eq!(
            Marker::text("x").then_special("<|sep|>", SEP_ID).first_token_id(),
            None
        );
        assert_eq!("</think>".first_token_id(), None);
    }

    #[test]
    fn attributed_next_candidate_uses_anchors_for_guarded_markers() {
        let mut fixture = vec![("say ", 1)];
        fixture.extend(ordinary_close()); // lookalike at 4
        fixture.extend([("think", 5), ("<|sep|>", 903), (" then ", 2)]);
        let genuine_at = fixture.iter().map(|(piece, _)| piece.len()).sum::<usize>();
        fixture.extend([("<|close|>", CLOSE_ID), ("think", 5), ("<|sep|>", SEP_ID)]);
        let buffer = pieces(&fixture);
        let input = attributed(&buffer, AttributionMode::Tokens);

        let marker = think_close();
        assert_eq!(input.next_candidate(&[&marker], 0), Some(genuine_at));
        assert_eq!(input.next_candidate(&[&marker], genuine_at + 1), None);
    }

    #[test]
    fn attributed_next_candidate_falls_back_to_text_for_unguarded_markers() {
        let buffer = pieces(&[("a", 1), ("</think>", 2), ("b", 3)]);
        let input = attributed(&buffer, AttributionMode::Tokens);

        assert_eq!(input.next_candidate(&["</think>"], 0), Some(1));
        assert_eq!(input.next_candidate(&["</thinking>"], 0), None);
        // Partial spelling at the end of the input is a candidate too.
        let buffer = pieces(&[("a", 1), ("</thi", 2)]);
        let input = attributed(&buffer, AttributionMode::Tokens);
        assert_eq!(input.next_candidate(&["</think>"], 0), Some(1));
    }

    #[test]
    fn text_next_candidate_matches_previous_scan_semantics() {
        let input = Partial::new("hello<not_marker><|tool");

        assert_eq!(
            input.next_candidate(&["<|tool_call>", "<|channel>thought\n"], 0),
            Some("hello<not_marker>".len())
        );
        assert_eq!(input.next_candidate(&["<|tool_call>"], 18), None);
        assert_eq!(
            Partial::new("hello<channel|><|tool_call>")
                .next_candidate(&["<|tool_call>", "<channel|>"], 0),
            Some(5)
        );
    }

    #[test]
    fn attributed_offsets_follow_the_cursor() {
        let buffer = pieces(&[("ab", 1), ("<|close|>", CLOSE_ID)]);
        let mut input = attributed(&buffer, AttributionMode::Tokens);
        input.next_slice(2);

        assert_eq!(input.offset(), 2);
        assert_eq!(input.remaining(), "<|close|>");
        special("<|close|>", CLOSE_ID).parse_next(&mut input).unwrap();
        assert_eq!(input.offset(), 11);
    }
}
