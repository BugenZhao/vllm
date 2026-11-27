// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Whether the connected engines can serve generation requests now.
//!
//! Each engine answers a `check_health_generate` utility call between steps:
//! `sleeping` while paused or asleep, `weight_update_incomplete` while a weight
//! update has not finished, `busy` while it has work, and otherwise `ok` after
//! a forward probe. A wedged busy loop cannot report itself, so only a timed
//! round trip from the client shows it. The client therefore owns readiness,
//! fed by these probes and by the control calls it sends:
//!
//! - A control call that stops serving, such as a pause, withdraws readiness
//!   from before it is sent until it ends.
//! - Every control call that ends advances the epoch: an engine answers calls
//!   in order, so a probe sent earlier answers for the state before the call.
//! - A call sent to an engine cannot be taken back, and a wedged engine queues
//!   every call it is sent, so each engine has at most one probe in flight,
//!   shared by everyone who waits for it.
//!
//! Readiness is delivered two ways. [`Health::subscribe`] publishes it to
//! watchers such as the gRPC health services; while anyone subscribes, each
//! engine is probed after every control call and [`PROBE_INTERVAL`] after its
//! last answer. [`Health::check`] answers one caller from a fresh probe, as
//! Python's `AsyncLLM.check_health_generate`.

use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use parking_lot::Mutex;
use thiserror::Error;
use thiserror_ext::AsReport as _;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;

use crate::client::imp::ClientInner;
use crate::protocol::utility::EngineHealthState;
use crate::transport::{ConnectedEngine, EngineId};

/// How long after an engine's last answer subscribers get the next probe.
const PROBE_INTERVAL: Duration = Duration::from_secs(15);
/// How long a probe may stay unanswered before subscribers see its engine as
/// not ready.
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// Why the engines are not ready to serve generation requests.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NotReady {
    /// The engines have failed and can never serve again.
    #[error("Engine is dead")]
    EngineDead,
    /// Some engines are sleeping or paused.
    #[error("Engine is sleeping or paused (engine ranks: {engine_ranks:?})")]
    Sleeping { engine_ranks: Vec<u32> },
    /// A weight update started on some engines and has not finished
    /// successfully, so their weights may be incomplete.
    #[error("Weight update has not finished (engine ranks: {engine_ranks:?})")]
    WeightUpdateIncomplete { engine_ranks: Vec<u32> },
    /// Some engines did not answer the probe within the timeout.
    #[error("Health check timed out after {}s (engine ranks: {engine_ranks:?})", timeout.as_secs_f64())]
    ProbeTimeout {
        engine_ranks: Vec<u32>,
        timeout: Duration,
    },
    /// The probe failed on some engines; `error` describes the first failure.
    #[error("Health check failed (engine ranks: {engine_ranks:?})")]
    ProbeFailed {
        engine_ranks: Vec<u32>,
        error: String,
    },
}

type ProbeOutput = Result<EngineHealthState, String>;

/// One `check_health_generate` call to one engine.
#[derive(Clone)]
struct Probe {
    /// The epoch when the probe was sent.
    epoch: u64,
    sent_at: Instant,
    output: watch::Receiver<Option<ProbeOutput>>,
}

impl Probe {
    /// Send a probe to engine `engine`. Its own task records the answer, so
    /// the probe finishes, and is seen as finished, whether or not anyone
    /// waits for it.
    fn start(inner: &Arc<ClientInner>, engine: usize, engine_id: &EngineId, epoch: u64) -> Self {
        let call = inner.clone();
        let engine_id = engine_id.clone();
        let (output_tx, output) = watch::channel(None);
        inner.runtime_handle().spawn(async move {
            let output = call
                .call_utility_on(&[engine_id], "check_health_generate", (), None)
                .await
                .map(|mut states: Vec<EngineHealthState>| {
                    states.pop().expect("one result per engine")
                })
                .map_err(|error| error.to_report_string());
            // Finish the probe before recording its answer, so the probe the
            // record makes due is not refused for this one being in flight.
            output_tx.send_replace(Some(output.clone()));
            call.health().record(engine, epoch, &output);
        });
        Self {
            epoch,
            sent_at: Instant::now(),
            output,
        }
    }

    /// The probe's output, or `None` if it has none (yet).
    fn output(&self) -> Option<ProbeOutput> {
        self.output.borrow().clone()
    }

    /// Whether the probe has an output, or its task ended without one.
    fn is_finished(&self) -> bool {
        self.output.borrow().is_some() || self.output.has_changed().is_err()
    }

    async fn wait(&mut self) {
        let _ = self.output.wait_for(Option::is_some).await;
    }
}

/// An engine's latest answer to a probe.
struct Answer {
    /// The epoch the probe was sent at.
    epoch: u64,
    at: Instant,
    /// Whether the engine answered `ok` or `busy`.
    ready: bool,
}

struct EngineSlot {
    engine_id: EngineId,
    /// The data-parallel rank the engine reported in its ready response.
    rank: u32,
    /// The latest probe sent to this engine, finished or not.
    probe: Option<Probe>,
    answer: Option<Answer>,
}

impl EngineSlot {
    /// The probe in flight, if any.
    fn in_flight(&self) -> Option<&Probe> {
        self.probe.as_ref().filter(|probe| !probe.is_finished())
    }
}

struct HealthState {
    /// Control calls that stop serving and have not ended yet.
    withdrawals: usize,
    /// Advanced whenever a control call ends.
    epoch: u64,
    engines: Vec<EngineSlot>,
}

impl HealthState {
    /// Readiness to publish at `now`, or `None` to keep the published value
    /// until every engine answers a probe sent after the last control call.
    fn ready(&self, now: Instant) -> Option<bool> {
        if self.withdrawals > 0 {
            return Some(false);
        }
        let mut answered = true;
        for engine in &self.engines {
            if engine.in_flight().is_some_and(|probe| now >= probe.sent_at + PROBE_TIMEOUT) {
                return Some(false);
            }
            match &engine.answer {
                Some(answer) if answer.epoch == self.epoch => {
                    if !answer.ready {
                        return Some(false);
                    }
                }
                _ => answered = false,
            }
        }
        answered.then_some(true)
    }

    /// The engines to probe at `now`, and when to look again: when the next
    /// probe falls due or a probe in flight becomes overdue. Engines are probed
    /// only while `subscribed`, and never while a probe is in flight.
    fn due(&self, now: Instant, subscribed: bool) -> (Vec<usize>, Option<Instant>) {
        let mut due = Vec::new();
        let mut next: Option<Instant> = None;
        for (index, engine) in self.engines.iter().enumerate() {
            let wake_at = if let Some(probe) = engine.in_flight() {
                probe.sent_at + PROBE_TIMEOUT
            } else if subscribed {
                let due_at = match &engine.answer {
                    Some(answer) if answer.epoch == self.epoch => answer.at + PROBE_INTERVAL,
                    _ => now,
                };
                if due_at <= now {
                    due.push(index);
                    now + PROBE_TIMEOUT
                } else {
                    due_at
                }
            } else {
                continue;
            };
            if wake_at > now {
                next = Some(next.map_or(wake_at, |next| next.min(wake_at)));
            }
        }
        (due, next)
    }
}

/// Whether the connected engines can serve generation requests, as published
/// to subscribers and checked on demand. Control calls report through
/// [`RecheckGuard`] guards, and [`run_health_loop`] probes for subscribers.
pub(crate) struct Health {
    state: Mutex<HealthState>,
    tx: watch::Sender<bool>,
    /// Wakes [`run_health_loop`] when a probe may have fallen due or started.
    wake: Notify,
}

impl Health {
    /// Every engine starts as its ready response reports it: paused if it was
    /// prepared for a checkpoint on startup, ready otherwise.
    pub fn new(engines: &[ConnectedEngine]) -> Self {
        let now = Instant::now();
        let engines = engines
            .iter()
            .map(|engine| EngineSlot {
                engine_id: engine.engine_id.clone(),
                rank: engine.ready_response.data_parallel_rank,
                probe: None,
                answer: Some(Answer {
                    epoch: 0,
                    at: now,
                    ready: !engine.ready_response.starts_paused,
                }),
            })
            .collect();
        let state = HealthState {
            withdrawals: 0,
            epoch: 0,
            engines,
        };
        let ready = state.ready(now).unwrap_or(true);
        Self {
            state: Mutex::new(state),
            tx: watch::Sender::new(ready),
            wake: Notify::new(),
        }
    }

    /// Subscribe to the published readiness, probing the engines if their
    /// answer is unknown.
    pub fn subscribe(&self) -> watch::Receiver<bool> {
        let rx = self.tx.subscribe();
        self.wake.notify_one();
        rx
    }

    /// For a control call that may change readiness, such as a resume: leave
    /// readiness unchanged during the call and probe after it ends. Pass the
    /// guard to the call's [`ClientInner::call_utility`].
    pub fn recheck_after_call(self: &Arc<Self>) -> RecheckGuard {
        RecheckGuard {
            health: self.clone(),
            withdraws: false,
        }
    }

    /// For a control call that stops serving, such as a pause: withdraw
    /// readiness from before the call is sent until it ends and a probe after
    /// it answers. Pass the guard to the call's [`ClientInner::call_utility`].
    pub fn withdraw_until_recheck(self: &Arc<Self>) -> RecheckGuard {
        let mut state = self.state.lock();
        state.withdrawals += 1;
        self.publish(&state, Instant::now());
        RecheckGuard {
            health: self.clone(),
            withdraws: true,
        }
    }

    fn end(&self, withdraws: bool) {
        let mut state = self.state.lock();
        if withdraws {
            state.withdrawals -= 1;
        }
        state.epoch += 1;
        self.publish(&state, Instant::now());
        self.wake.notify_one();
    }

    /// Record an engine's answer to a probe sent at `epoch`.
    fn record(&self, engine: usize, epoch: u64, output: &ProbeOutput) {
        let now = Instant::now();
        let mut state = self.state.lock();
        state.engines[engine].answer = Some(Answer {
            epoch,
            at: now,
            ready: matches!(output, Ok(EngineHealthState::Ok | EngineHealthState::Busy)),
        });
        self.publish(&state, now);
        // An answer from before a control call ended leaves a probe due.
        self.wake.notify_one();
    }

    fn publish(&self, state: &HealthState, now: Instant) {
        if let Some(ready) = state.ready(now) {
            self.tx.send_if_modified(|current| std::mem::replace(current, ready) != ready);
        }
    }

    /// Start the probes due now and publish readiness; return when to poll
    /// again.
    fn poll(&self, inner: &Arc<ClientInner>) -> Option<Instant> {
        let now = Instant::now();
        let mut state = self.state.lock();
        let (due, next) = state.due(now, self.tx.receiver_count() > 0);
        let epoch = state.epoch;
        for index in due {
            let engine = &mut state.engines[index];
            engine.probe = Some(Probe::start(inner, index, &engine.engine_id, epoch));
        }
        self.publish(&state, now);
        next
    }

    fn ranks(&self) -> Vec<u32> {
        self.state.lock().engines.iter().map(|engine| engine.rank).collect()
    }

    fn is_withdrawn(&self) -> bool {
        self.state.lock().withdrawals > 0
    }

    /// The engine's probe in flight, or a new one if its last probe finished.
    fn current_or_start(&self, inner: &Arc<ClientInner>, index: usize) -> Probe {
        let mut state = self.state.lock();
        let epoch = state.epoch;
        let engine = &mut state.engines[index];
        if let Some(probe) = engine.in_flight() {
            return probe.clone();
        }
        let probe = Probe::start(inner, index, &engine.engine_id, epoch);
        engine.probe = Some(probe.clone());
        // Subscribers see it if it stays unanswered.
        self.wake.notify_one();
        probe
    }

    /// The engine's answer to a probe sent at `epoch` or later, or `None` if
    /// none arrives before `deadline`.
    async fn answer(
        &self,
        inner: &Arc<ClientInner>,
        index: usize,
        epoch: u64,
        deadline: Option<Instant>,
    ) -> Option<ProbeOutput> {
        loop {
            let mut probe = self.current_or_start(inner, index);
            match deadline {
                Some(deadline) => tokio::time::timeout_at(deadline, probe.wait()).await.ok()?,
                None => probe.wait().await,
            }
            if probe.epoch >= epoch {
                return probe.output();
            }
            // A stale probe finished: ask again for the current state.
        }
    }

    /// Probe every engine, waiting at most `timeout` in total. Unlike the
    /// published readiness, the answers must come from probes sent after
    /// the check started.
    pub async fn check(&self, inner: &Arc<ClientInner>, timeout: Duration) -> Result<(), NotReady> {
        if !inner.is_healthy() {
            return Err(NotReady::EngineDead);
        }
        // A call that stops serving is on its way to the engines, so they are
        // not ready whatever they answer now.
        let sleeping = || NotReady::Sleeping {
            engine_ranks: self.ranks(),
        };
        if self.is_withdrawn() {
            return Err(sleeping());
        }
        let epoch = self.state.lock().epoch;
        // A timeout too long to represent waits forever.
        let deadline = Instant::now().checked_add(timeout);
        let engines = self.state.lock().engines.len();
        let answers =
            join_all((0..engines).map(|index| self.answer(inner, index, epoch, deadline))).await;
        if !inner.is_healthy() {
            return Err(NotReady::EngineDead);
        }
        if self.is_withdrawn() {
            return Err(sleeping());
        }

        let mut sleeping = Vec::new();
        let mut weights_incomplete = Vec::new();
        let mut timed_out = Vec::new();
        let mut failed = Vec::new();
        for (rank, answer) in self.ranks().into_iter().zip(answers) {
            match answer {
                None => timed_out.push(rank),
                Some(Ok(EngineHealthState::Sleeping)) => sleeping.push(rank),
                Some(Ok(EngineHealthState::WeightUpdateIncomplete)) => {
                    weights_incomplete.push(rank)
                }
                Some(Ok(EngineHealthState::Ok | EngineHealthState::Busy)) => {}
                Some(Err(error)) => failed.push((rank, error)),
            }
        }
        if !sleeping.is_empty() {
            return Err(NotReady::Sleeping {
                engine_ranks: sleeping,
            });
        }
        if !weights_incomplete.is_empty() {
            return Err(NotReady::WeightUpdateIncomplete {
                engine_ranks: weights_incomplete,
            });
        }
        if !timed_out.is_empty() {
            return Err(NotReady::ProbeTimeout {
                engine_ranks: timed_out,
                timeout,
            });
        }
        if let Some((_, error)) = failed.first() {
            return Err(NotReady::ProbeFailed {
                error: error.clone(),
                engine_ranks: failed.into_iter().map(|(rank, _)| rank).collect(),
            });
        }
        Ok(())
    }
}

/// A control call in flight. Dropping it, whether the call succeeded or failed,
/// ends the call: the epoch advances and subscribers get a probe.
/// [`ClientInner::call_utility`] holds it until the engines reply, even if the
/// caller is cancelled.
pub(crate) struct RecheckGuard {
    health: Arc<Health>,
    /// Whether readiness is withdrawn until the call ends.
    withdraws: bool,
}

impl Drop for RecheckGuard {
    fn drop(&mut self) {
        self.health.end(self.withdraws);
    }
}

/// Background loop that probes the engines for subscribers and publishes
/// readiness, until the engines fail.
pub(crate) async fn run_health_loop(inner: Arc<ClientInner>) {
    let health = inner.health().clone();
    let mut healthy = inner.subscribe_health();
    loop {
        let next = health.poll(&inner);
        let sleep = async {
            match next {
                Some(at) => tokio::time::sleep_until(at).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            () = health.wake.notified() => {}
            () = sleep => {}
            _ = healthy.wait_for(|healthy| !*healthy) => {
                // Engine failure is permanent.
                health.tx.send_replace(false);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::poll;
    use rmpv::Value;
    use tokio::sync::oneshot;
    use zeromq::prelude::{SocketRecv as _, SocketSend as _};
    use zeromq::{DealerSocket, PushSocket, ZmqMessage};

    use super::NotReady;
    use crate::mock_engine::default_ready_response;
    use crate::protocol::output::{EngineCoreOutputs, UtilityCallOutput};
    use crate::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
    use crate::test_utils::{
        IpcNamespace, spawn_mock_engine_task, spawn_mock_engine_task_with_ready,
    };
    use crate::{EngineCoreClient, EngineCoreClientConfig, EngineId, TransportMode};

    /// Receive one `check_health_generate` call and return its call ID.
    async fn recv_probe(dealer: &mut DealerSocket) -> u64 {
        let frames = dealer.recv().await.expect("recv utility").into_vec();
        let payload: Value = rmp_serde::from_slice(&frames[1]).expect("decode utility payload");
        let payload = payload.as_array().expect("utility payload array");
        assert_eq!(payload[2], Value::from("check_health_generate"));
        payload[1].as_u64().expect("call id")
    }

    /// Answer the probe `call_id` with `state`.
    async fn answer(push: &mut PushSocket, call_id: u64, state: &str) {
        let output = EngineCoreOutputs::from(UtilityCallOutput {
            output: UtilityOutput {
                call_id: call_id.into(),
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(Value::from(state))),
            },
            ..Default::default()
        });
        push.send(ZmqMessage::from(
            rmp_serde::to_vec_named(&output).expect("encode output"),
        ))
        .await
        .expect("send output");
    }

    /// Whether the engine receives no further message for a while.
    async fn stays_quiet(dealer: &mut DealerSocket) -> bool {
        tokio::time::timeout(Duration::from_millis(200), dealer.recv()).await.is_err()
    }

    async fn connect(
        handshake_address: String,
        ipc: &IpcNamespace,
        engines: usize,
    ) -> EngineCoreClient {
        let mut config = EngineCoreClientConfig::new_single(handshake_address)
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            );
        if let TransportMode::HandshakeOwner {
            engine_count,
            data_parallel_size,
            ..
        } = &mut config.transport_mode
        {
            *engine_count = engines;
            *data_parallel_size = engines;
        }
        EngineCoreClient::connect(config).await.expect("connect client")
    }

    fn timed_out(timeout: Duration) -> Result<(), NotReady> {
        Err(NotReady::ProbeTimeout {
            engine_ranks: vec![0],
            timeout,
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probes_engines_by_identity() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        // Identities that encode no Python engine index: each engine still
        // gets, and answers, a probe of its own, reported by its rank.
        let engines =
            [("engine-0", 0, "ok"), ("engine-1", 1, "sleeping")].map(|(id, rank, state)| {
                let mut ready = default_ready_response();
                ready.data_parallel_rank = rank;
                spawn_mock_engine_task_with_ready(
                    handshake.clone(),
                    EngineId::from(id.as_bytes().to_vec()),
                    ready,
                    move |dealer, push| {
                        Box::pin(async move {
                            let call_id = recv_probe(dealer).await;
                            answer(push, call_id, state).await;
                        })
                    },
                )
            });
        let client = connect(handshake, &ipc, 2).await;

        let result = client.check_health_generate(Duration::from_secs(5)).await;
        assert_eq!(
            result,
            Err(NotReady::Sleeping {
                engine_ranks: vec![1]
            })
        );

        for (shutdown, task) in engines {
            let _ = shutdown.send(());
            task.await.unwrap();
        }
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn callers_share_the_in_flight_probe() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (quiet_tx, quiet_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    quiet_tx.send(stays_quiet(dealer).await).unwrap();
                    answer(push, call_id, "ok").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let timeout = Duration::from_millis(50);
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        // A third caller joins the probe while the engine still holds it.
        let mut check = Box::pin(client.check_health_generate(Duration::from_secs(5)));
        assert!(poll!(check.as_mut()).is_pending());
        go_tx.send(()).unwrap();
        // The one probe every caller shares is the one the engine answers.
        assert_eq!(check.await, Ok(()));
        assert!(quiet_rx.await.unwrap(), "a second probe was sent");

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_probe_finished_after_its_caller_left_is_not_reused() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    // Answer the first probe late, as `ok`; the engine then sleeps.
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    answer(push, call_id, "ok").await;
                    let call_id = recv_probe(dealer).await;
                    answer(push, call_id, "sleeping").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let timeout = Duration::from_millis(50);
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        go_tx.send(()).unwrap();
        // The first probe's own task records the answer, without any caller.
        let mut first = client.inner.health().state.lock().engines[0].probe.clone().unwrap();
        first.wait().await;
        assert!(first.is_finished());

        let result = client.check_health_generate(Duration::from_secs(5)).await;
        assert_eq!(
            result,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_withdrawing_call_in_flight_reports_every_rank_sleeping() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (quiet_tx, quiet_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, _push| {
                Box::pin(async move {
                    quiet_tx.send(stays_quiet(dealer).await).unwrap();
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        // A pause on its way to the engine.
        let pause = client.inner.health().withdraw_until_recheck();
        let result = client.check_health_generate(Duration::from_secs(5)).await;
        assert_eq!(
            result,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );
        assert!(quiet_rx.await.unwrap(), "the engine was probed");
        drop(pause);

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_withdrawing_call_started_during_a_check_overrides_its_answer() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    // The engine answers before the pause reaches it.
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    answer(push, call_id, "ok").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let mut check = Box::pin(client.check_health_generate(Duration::from_secs(5)));
        assert!(poll!(check.as_mut()).is_pending());
        let pause = client.inner.health().withdraw_until_recheck();
        go_tx.send(()).unwrap();
        assert_eq!(
            check.await,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );
        drop(pause);

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_probe_sent_before_a_control_call_ended_is_replaced() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (quiet_tx, quiet_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    // The first probe was answered before the control call ran,
                    // as `ok`; after the call the engine sleeps.
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    quiet_tx.send(stays_quiet(dealer).await).unwrap();
                    answer(push, call_id, "ok").await;
                    let call_id = recv_probe(dealer).await;
                    answer(push, call_id, "sleeping").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let timeout = Duration::from_millis(50);
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        // A control call ends while the probe is in flight.
        drop(client.inner.health().recheck_after_call());
        // The next caller finds the stale probe still in flight.
        let mut check = Box::pin(client.check_health_generate(Duration::from_secs(5)));
        assert!(poll!(check.as_mut()).is_pending());
        go_tx.send(()).unwrap();

        assert_eq!(
            check.await,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );
        // The stale probe was awaited, not doubled up.
        assert!(quiet_rx.await.unwrap(), "a second probe was sent");

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    /// Published readiness and probe scheduling, at chosen instants.
    mod state {
        use tokio::sync::watch;
        use tokio::time::Instant;

        use super::super::{
            Answer, EngineSlot, HealthState, PROBE_INTERVAL, PROBE_TIMEOUT, Probe, ProbeOutput,
        };
        use crate::EngineId;
        use crate::protocol::utility::EngineHealthState;

        /// One engine that answered `ready` at epoch 0 at `at`.
        fn one_engine(at: Instant, ready: bool) -> HealthState {
            HealthState {
                withdrawals: 0,
                epoch: 0,
                engines: vec![EngineSlot {
                    engine_id: EngineId::from(b"\x00\x00".to_vec()),
                    rank: 0,
                    probe: None,
                    answer: Some(Answer {
                        epoch: 0,
                        at,
                        ready,
                    }),
                }],
            }
        }

        /// A probe sent at `sent_at`, finished once its sender sends.
        fn probe(epoch: u64, sent_at: Instant) -> (watch::Sender<Option<ProbeOutput>>, Probe) {
            let (tx, output) = watch::channel(None);
            (
                tx,
                Probe {
                    epoch,
                    sent_at,
                    output,
                },
            )
        }

        #[test]
        fn readiness_waits_for_an_answer_after_the_last_control_call() {
            let now = Instant::now();
            let mut state = one_engine(now, true);
            assert_eq!(state.ready(now), Some(true));

            // A control call ends: the answer predates it.
            state.epoch += 1;
            assert_eq!(state.ready(now), None);
            state.engines[0].answer = Some(Answer {
                epoch: 1,
                at: now,
                ready: false,
            });
            assert_eq!(state.ready(now), Some(false));
            state.engines[0].answer = Some(Answer {
                epoch: 1,
                at: now,
                ready: true,
            });
            assert_eq!(state.ready(now), Some(true));
        }

        #[test]
        fn a_withdrawing_call_in_flight_hides_ready_answers() {
            let now = Instant::now();
            let mut state = one_engine(now, true);
            state.withdrawals = 1;
            assert_eq!(state.ready(now), Some(false));
        }

        #[test]
        fn an_unanswered_probe_withdraws_readiness_after_the_timeout() {
            let sent_at = Instant::now();
            let mut state = one_engine(sent_at, true);
            let (finish, probe) = probe(0, sent_at);
            state.engines[0].probe = Some(probe);

            let overdue = sent_at + PROBE_TIMEOUT;
            assert_eq!(state.ready(overdue - PROBE_TIMEOUT / 2), Some(true));
            assert_eq!(state.ready(overdue), Some(false));
            // The wedged engine gets no second probe; the loop wakes when the
            // probe falls overdue.
            assert_eq!(state.due(sent_at, true), (vec![], Some(overdue)));
            assert_eq!(state.due(overdue, true), (vec![], None));

            finish.send_replace(Some(Ok(EngineHealthState::Ok)));
            assert_eq!(state.ready(overdue), Some(true));
        }

        #[test]
        fn probes_fall_due_for_subscribers_only() {
            let at = Instant::now();
            let state = one_engine(at, true);
            let next = at + PROBE_INTERVAL;
            assert_eq!(state.due(at, false), (vec![], None));
            assert_eq!(state.due(at, true), (vec![], Some(next)));
            assert_eq!(state.due(next, false), (vec![], None));
            assert_eq!(state.due(next, true), (vec![0], Some(next + PROBE_TIMEOUT)));
        }

        #[test]
        fn a_control_call_makes_a_probe_due_once_the_stale_one_finishes() {
            let now = Instant::now();
            let mut state = one_engine(now, true);
            let (finish, probe) = probe(0, now);
            state.engines[0].probe = Some(probe);

            state.epoch += 1;
            assert_eq!(state.due(now, true), (vec![], Some(now + PROBE_TIMEOUT)));
            finish.send_replace(Some(Ok(EngineHealthState::Ok)));
            assert_eq!(state.due(now, true), (vec![0], Some(now + PROBE_TIMEOUT)));
        }
    }
}
