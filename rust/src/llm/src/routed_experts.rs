// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_engine_core_client::protocol::tensor::{ShapeExt as _, WireArrayData, WireNdArray};

use crate::error::{Error, Result};

/// Routed-expert rows along the token axis, kept as the engine sent them: one
/// `[rows, layers, top_k]` chunk per engine output.
///
/// The engine sends each output's new rows, so a consumer appends the rows of
/// every output it keeps and concatenates them once, when the request
/// finishes. Joining at the end keeps the cost linear in the request length.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoutedExpertsRows {
    chunks: Vec<WireNdArray>,
}

impl RoutedExpertsRows {
    /// Wrap the rows of one engine output.
    pub(crate) fn from_engine(rows: Option<Box<WireNdArray>>) -> Self {
        Self {
            chunks: rows.map(|rows| *rows).into_iter().collect(),
        }
    }

    /// Whether no rows were received.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Append the rows of a later output.
    pub fn append(&mut self, mut later: Self) {
        self.chunks.append(&mut later.chunks);
    }

    /// Drop up to `num_rows` rows from the end, like Python's
    /// `rows[:-num_rows]` for the tokens a stop string cut off. A chunk keeps
    /// its row shape even when no row is left in it.
    pub fn drop_trailing(&mut self, mut num_rows: usize) {
        for chunk in self.chunks.iter_mut().rev() {
            if num_rows == 0 {
                break;
            }
            let Some((&chunk_rows, row_shape)) = chunk.shape.split_first() else {
                continue;
            };
            let dropped = num_rows.min(chunk_rows);
            let kept = chunk_rows - dropped;
            let row_bytes =
                row_shape.checked_numel().unwrap_or(0) * chunk.dtype.scalar.element_size();
            if let WireArrayData::RawView(data) = &mut chunk.data {
                data.truncate(kept * row_bytes);
            }
            chunk.shape[0] = kept;
            num_rows -= dropped;
        }
    }

    /// Concatenate the rows along the token axis, like Python's
    /// `np.concatenate`; `None` when no output carried rows.
    pub fn concat(self, request_id: &str) -> Result<Option<WireNdArray>> {
        let mut chunks = self.chunks.into_iter();
        let Some(first) = chunks.next() else {
            return Ok(None);
        };
        let mut rest = chunks.peekable();
        if rest.peek().is_none() {
            return Ok(Some(first));
        }
        let shape_mismatch = || Error::RoutedExpertsShapeMismatch {
            request_id: request_id.to_string(),
        };
        let mut num_rows = *first.shape.first().ok_or_else(shape_mismatch)?;
        let mut data = first.data.as_raw_view().ok_or_else(shape_mismatch)?.to_vec();
        for chunk in rest {
            if chunk.dtype != first.dtype || chunk.shape.get(1..) != first.shape.get(1..) {
                return Err(shape_mismatch());
            }
            num_rows += chunk.shape[0];
            data.extend_from_slice(chunk.data.as_raw_view().ok_or_else(shape_mismatch)?);
        }
        let mut shape = first.shape;
        shape[0] = num_rows;
        Ok(Some(WireNdArray::from_raw(first.dtype, shape, data)))
    }
}

impl From<WireNdArray> for RoutedExpertsRows {
    fn from(rows: WireNdArray) -> Self {
        Self { chunks: vec![rows] }
    }
}

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::protocol::dtype::TensorDtype;

    use super::*;

    /// One `[1, 2]` routing row per value.
    fn rows(values: &[u8]) -> WireNdArray {
        WireNdArray::from_raw(
            TensorDtype::U8,
            vec![values.len(), 1, 2],
            values.iter().flat_map(|&value| [value, 0]).collect(),
        )
    }

    fn collected(rows: RoutedExpertsRows) -> Option<WireNdArray> {
        rows.concat("req").unwrap()
    }

    #[test]
    fn concat_joins_chunks_along_the_token_axis() {
        let mut all = RoutedExpertsRows::default();
        assert_eq!(collected(all.clone()), None);
        all.append(RoutedExpertsRows::from(rows(&[1, 2])));
        all.append(RoutedExpertsRows::default());
        all.append(RoutedExpertsRows::from(rows(&[3])));
        assert_eq!(collected(all), Some(rows(&[1, 2, 3])));
    }

    #[test]
    fn drop_trailing_cuts_rows_across_chunks() {
        let mut all = RoutedExpertsRows::from(rows(&[1, 2]));
        all.append(RoutedExpertsRows::from(rows(&[3, 4])));
        all.drop_trailing(3);
        assert_eq!(collected(all.clone()), Some(rows(&[1])));
        // Like `rows[:-n]` with `n` past the start, nothing is left.
        all.drop_trailing(5);
        assert_eq!(collected(all), Some(rows(&[])));
    }

    #[test]
    fn concat_rejects_a_changed_row_shape() {
        let mut all = RoutedExpertsRows::from(rows(&[1]));
        all.append(RoutedExpertsRows::from(WireNdArray::from_raw(
            TensorDtype::U8,
            vec![1, 2, 1],
            vec![0, 0],
        )));
        assert!(matches!(
            all.concat("req"),
            Err(Error::RoutedExpertsShapeMismatch { .. })
        ));
    }
}
