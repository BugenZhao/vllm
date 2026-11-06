// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Frontend admission limits on unfinished requests, the counterpart of
//! Python's `AsyncLLM.check_admission`.

use std::num::NonZeroU32;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;
use vllm_engine_core_client::RequestLifecycle;
use vllm_engine_core_client::protocol::output::EngineCoreOutput;
use vllm_metrics::{AdmissionRejectionLabels, METRICS, U64Counter};

use crate::error::{Error, Result};

/// Queue limits checked before a request is submitted to the engine.
///
/// A request over a limit fails with [`Error::Unavailable`] (HTTP 503), so
/// load balancers and clients can retry it on another instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AdmissionLimits {
    /// Maximum number of unfinished requests, where a request with `n`
    /// samples counts `n`.
    pub max_num_queued_reqs: Option<u64>,
    /// Maximum total prompt tokens of unfinished requests still in prefill,
    /// that is, before their first engine output.
    pub max_num_queued_tokens: Option<u64>,
}

/// Unfinished samples and the prompt tokens of those still in prefill.
#[derive(Debug, Default)]
struct Queued {
    requests: u64,
    prefill_tokens: u64,
}

/// Tracks unfinished requests against [`AdmissionLimits`].
pub(crate) struct Admission {
    limits: AdmissionLimits,
    queued: Arc<Mutex<Queued>>,
    /// `vllm:admission_rejections` for each limit.
    queued_reqs_rejections: U64Counter,
    queued_tokens_rejections: U64Counter,
}

impl Admission {
    /// Create the tracker, and the rejection series of every limit even when
    /// it is unset, like Python.
    pub(crate) fn new(model_name: &str, limits: AdmissionLimits) -> Self {
        let rejections = |reason| {
            METRICS
                .request
                .admission_rejections
                .get_or_create_owned(&AdmissionRejectionLabels {
                    model_name: model_name.to_string(),
                    reason,
                })
        };
        Self {
            limits,
            queued: Arc::default(),
            queued_reqs_rejections: rejections("max_num_queued_reqs"),
            queued_tokens_rejections: rejections("max_num_queued_tokens"),
        }
    }

    /// Admit the `n` samples of a request with `prompt_len` prompt tokens, or
    /// reject the request when a limit is reached. Each returned ticket keeps
    /// one sample counted until it is dropped.
    ///
    /// Like Python, the token limit rejects once the prefill backlog has
    /// reached it, regardless of the new prompt's length, and a sample counts
    /// its whole prompt until its first engine output, since chunked-prefill
    /// progress and prefix-cache hits are only reported with that output.
    pub(crate) fn admit(&self, n: NonZeroU32, prompt_len: u64) -> Result<Vec<AdmissionTicket>> {
        let n = n.get();
        let mut queued = self.queued.lock();
        if let Some(max) = self.limits.max_num_queued_reqs
            && queued.requests + u64::from(n) > max
        {
            self.queued_reqs_rejections.inc();
            return Err(Error::Unavailable {
                message: "The engine is currently busy and cannot accept new requests. \
                          Please try again later or on a different instance."
                    .to_string(),
            });
        }
        if let Some(max) = self.limits.max_num_queued_tokens
            && queued.prefill_tokens >= max
        {
            self.queued_tokens_rejections.inc();
            return Err(Error::Unavailable {
                message: "The engine has reached its prefill token backlog limit. \
                          Please try again later or on a different instance."
                    .to_string(),
            });
        }
        queued.requests += u64::from(n);
        queued.prefill_tokens += u64::from(n) * prompt_len;
        drop(queued);

        Ok((0..n)
            .map(|_| AdmissionTicket {
                queued: self.queued.clone(),
                prefill_tokens: prompt_len,
            })
            .collect())
    }
}

/// One admitted sample, counted as unfinished until dropped. The engine-core
/// client's request registry holds it from submission until the request
/// finishes or is aborted.
pub(crate) struct AdmissionTicket {
    queued: Arc<Mutex<Queued>>,
    /// Prompt tokens still counted as prefill backlog; zero once the first
    /// engine output arrived.
    prefill_tokens: u64,
}

impl AdmissionTicket {
    /// Stop counting the prompt as prefill backlog.
    fn end_prefill(&mut self) {
        if self.prefill_tokens > 0 {
            self.queued.lock().prefill_tokens -= std::mem::take(&mut self.prefill_tokens);
        }
    }
}

impl RequestLifecycle for AdmissionTicket {
    /// The sample's first engine output ends its prefill.
    fn observe_output(&mut self, _output: &EngineCoreOutput) {
        self.end_prefill();
    }
}

impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        let mut queued = self.queued.lock();
        queued.requests -= 1;
        queued.prefill_tokens -= self.prefill_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    fn is_unavailable(result: Result<Vec<AdmissionTicket>>) -> bool {
        matches!(result, Err(Error::Unavailable { .. }))
    }

    /// `vllm:admission_rejections` of `model_name` for both limits.
    fn rejections(model_name: &str) -> [u64; 2] {
        ["max_num_queued_reqs", "max_num_queued_tokens"].map(|reason| {
            METRICS
                .request
                .admission_rejections
                .get_or_create_owned(&AdmissionRejectionLabels {
                    model_name: model_name.to_string(),
                    reason,
                })
                .get()
        })
    }

    #[test]
    fn request_limit_counts_samples_until_tickets_drop() {
        let admission = Admission::new(
            "admission-request-limit",
            AdmissionLimits {
                max_num_queued_reqs: Some(3),
                max_num_queued_tokens: None,
            },
        );

        let two = admission.admit(n(2), 10).unwrap();
        assert_eq!(two.len(), 2);
        // Two samples are queued: two more would make four.
        assert!(is_unavailable(admission.admit(n(2), 10)));
        let one = admission.admit(n(1), 10).unwrap();
        assert!(is_unavailable(admission.admit(n(1), 10)));

        // Finishing one sample frees one slot; finishing a prefill frees none.
        let mut two = two;
        two[0].end_prefill();
        assert!(is_unavailable(admission.admit(n(1), 10)));
        two.pop();
        let _again = admission.admit(n(1), 10).unwrap();
        drop(one);
        let _last = admission.admit(n(1), 10).unwrap();
        assert_eq!(rejections("admission-request-limit"), [3, 0]);
    }

    #[test]
    fn token_limit_counts_prompts_until_first_output() {
        let admission = Admission::new(
            "admission-token-limit",
            AdmissionLimits {
                max_num_queued_reqs: None,
                max_num_queued_tokens: Some(100),
            },
        );

        // Below the limit, any prompt is admitted, even one that crosses it.
        let mut first = admission.admit(n(1), 60).unwrap();
        let mut second = admission.admit(n(1), 60).unwrap();
        assert!(is_unavailable(admission.admit(n(1), 1)));

        // A sample that produced its first output no longer counts.
        first[0].end_prefill();
        first[0].end_prefill();
        assert!(admission.admit(n(1), 1).is_ok());
        let _third = admission.admit(n(1), 40).unwrap();
        assert!(is_unavailable(admission.admit(n(1), 1)));

        // Dropping a ticket still in prefill releases its prompt.
        second.pop();
        // Each of `n` samples counts the prompt: 40 + 2 * 30 reaches the limit.
        let _pair = admission.admit(n(2), 30).unwrap();
        assert!(is_unavailable(admission.admit(n(1), 1)));
        assert_eq!(rejections("admission-token-limit"), [0, 3]);
    }

    #[test]
    fn no_limits_admit_everything() {
        let admission = Admission::new("admission-no-limits", AdmissionLimits::default());
        let tickets = admission.admit(n(4), u64::MAX / 8).unwrap();
        assert_eq!(tickets.len(), 4);
        assert!(admission.admit(n(1), 1).is_ok());
        // Both series are exported before the first rejection.
        let rendered = METRICS.render().unwrap();
        for reason in ["max_num_queued_reqs", "max_num_queued_tokens"] {
            let series = format!(
                "vllm:admission_rejections_total{{model_name=\"admission-no-limits\",reason=\"{reason}\"}} 0"
            );
            assert!(rendered.contains(&series), "{series}");
        }
    }
}
