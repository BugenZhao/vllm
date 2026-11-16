// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Hot-reload of the TLS certificate material (`--enable-ssl-refresh`), mirroring
//! Python's `SSLCertRefresher`: when the cert/key/CA files change on disk, new
//! handshakes serve the new material without a restart.
//!
//! OpenSSL's [`SslContext`] is immutable once built, so a change rebuilds it and
//! swaps it into a [`ReloadableTls`] cell read per handshake; in-flight
//! connections keep the context they negotiated with.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};
use openssl::ssl::SslContext;
use thiserror_ext::AsReport as _;
use tls_listener::AsyncTls;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_util::task::AbortOnDropHandle;
use tracing::{info, warn};

use crate::config::TlsConfig;
use crate::tls;

/// Event-grouping windows mirroring watchfiles' defaults: reload once `STEP`
/// passes with no new event, but never group longer than `DEBOUNCE` in total.
const STEP: Duration = Duration::from_millis(50);
const DEBOUNCE: Duration = Duration::from_millis(1600);

/// A swappable OpenSSL context: the acceptor reads the current context per
/// handshake, so a reloader can install a rebuilt one at runtime.
#[derive(Clone)]
pub(crate) struct ReloadableTls(Arc<ArcSwap<SslContext>>);

impl ReloadableTls {
    pub(crate) fn new(context: SslContext) -> Self {
        Self(Arc::new(ArcSwap::from_pointee(context)))
    }

    fn store(&self, context: SslContext) {
        self.0.store(Arc::new(context));
    }

    /// The context new handshakes use.
    #[cfg(test)]
    pub(crate) fn current(&self) -> Arc<SslContext> {
        self.0.load_full()
    }
}

/// Delegate to the crate's own `SslContext` acceptor. `accept` snapshots the
/// context into a fresh `Ssl` and returns an owned future, so the loaded handle
/// may drop.
impl<C> AsyncTls<C> for ReloadableTls
where
    C: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    type Stream = tokio_openssl::SslStream<C>;
    type Error = openssl::ssl::Error;
    type AcceptFuture = <SslContext as AsyncTls<C>>::AcceptFuture;

    fn accept(&self, conn: C) -> Self::AcceptFuture {
        self.0.load_full().accept(conn)
    }
}

/// The TLS acceptors of one server: the HTTP listener's and, when the server
/// also serves gRPC, the gRPC listener's (same material, ALPN `h2`). With
/// [`TlsConfig::refresh`] a watcher keeps both current with the certificate
/// files until this is dropped.
pub(crate) struct ServerTls {
    pub(crate) http: ReloadableTls,
    pub(crate) grpc: Option<ReloadableTls>,
    _reloader: Option<AbortOnDropHandle<()>>,
}

impl ServerTls {
    /// Build the contexts once, up front, so a bad cert/key fails fast.
    pub(crate) fn build(tls: &TlsConfig, grpc: bool) -> anyhow::Result<Self> {
        // Read before building, so a change racing the build differs from it.
        let loaded = tls.refresh.then(|| CertFiles::new(tls)).map(|files| {
            let contents = files.read();
            (files, contents)
        });
        let http = ReloadableTls::new(tls::build_server_config(tls)?);
        let grpc = grpc
            .then(|| tls::build_grpc_server_config(tls))
            .transpose()?
            .map(ReloadableTls::new);
        let reloader = loaded.map(|(files, contents)| {
            spawn_cert_reloader(files, contents, tls.clone(), http.clone(), grpc.clone())
        });
        Ok(Self {
            http,
            grpc,
            _reloader: reloader,
        })
    }
}

/// The files a TLS context is built from.
struct CertFiles(Vec<PathBuf>);

impl CertFiles {
    /// The cert chain (always), plus the key when configured separately and
    /// the CA when client certificates are verified against it.
    fn new(tls: &TlsConfig) -> Self {
        Self(
            [
                tls.cert_file.as_deref(),
                tls.key_file.as_deref(),
                tls.ca_certs.as_deref().filter(|_| tls.cert_reqs != 0),
            ]
            .into_iter()
            .flatten()
            .map(PathBuf::from)
            .collect(),
        )
    }

    /// The files' contents, following symlinks; `None` for one that cannot be
    /// read (e.g. removed mid-rotation).
    fn read(&self) -> Vec<Option<Vec<u8>>> {
        self.0.iter().map(|file| std::fs::read(file).ok()).collect()
    }

    /// The directories to watch: each file's parent, since a symlink-swap
    /// rotation changes the leaf inode, which a file watch would stop
    /// following; and, for a symlinked file, the directory it resolves to, so
    /// a write to a target elsewhere is seen.
    fn dirs(&self) -> BTreeSet<PathBuf> {
        let parent = |file: &Path| match file.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        self.0
            .iter()
            .flat_map(|file| {
                let resolved = std::fs::canonicalize(file).ok().map(|file| parent(&file));
                std::iter::once(parent(file)).chain(resolved)
            })
            .collect()
    }
}

/// Spawn the certificate watcher; dropping the returned handle (on shutdown)
/// aborts it. The watches are registered before this returns, so no change
/// made afterwards is missed. `loaded` is what the current contexts were built
/// from.
fn spawn_cert_reloader(
    files: CertFiles,
    loaded: Vec<Option<Vec<u8>>>,
    tls: TlsConfig,
    http: ReloadableTls,
    grpc: Option<ReloadableTls>,
) -> AbortOnDropHandle<()> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let watcher = match notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        // A reload reads the files, which inotify reports as access events;
        // only creates, writes and removes are changes (as in watchfiles).
        if !matches!(&event, Ok(event) if event.kind.is_access()) {
            let _ = tx.send(event);
        }
    }) {
        Ok(watcher) => {
            let mut watcher = CertWatcher {
                files,
                watcher,
                dirs: BTreeSet::new(),
            };
            if watcher.watch_dirs() {
                info!(files = ?watcher.files.0, "watching SSL certificate files for changes");
                Some(watcher)
            } else {
                warn!("no SSL certificate directory could be watched; hot-reload is inactive");
                None
            }
        }
        Err(err) => {
            warn!(error = %err.as_report(), "failed to start the SSL certificate watcher; hot-reload disabled");
            None
        }
    };

    AbortOnDropHandle::new(tokio::spawn(async move {
        let Some(watcher) = watcher else {
            return;
        };
        run(watcher, rx, loaded, tls, http, grpc).await
    }))
}

/// The watcher with the directories it watches.
struct CertWatcher {
    files: CertFiles,
    watcher: RecommendedWatcher,
    dirs: BTreeSet<PathBuf>,
}

impl CertWatcher {
    /// Watch the files' directories, dropping those they no longer resolve to.
    /// Every directory is watched again each time: a directory removed and
    /// recreated (e.g. a link pointed away and back) loses its watch with the
    /// old inode. Returns whether any directory is watched.
    fn watch_dirs(&mut self) -> bool {
        let dirs = self.files.dirs();
        for dir in self.dirs.difference(&dirs) {
            let _ = self.watcher.unwatch(dir);
        }
        self.dirs.clear();
        for dir in dirs {
            match self.watcher.watch(&dir, RecursiveMode::NonRecursive) {
                Ok(()) => {
                    self.dirs.insert(dir);
                }
                Err(err) => warn!(
                    error = %err.as_report(),
                    dir = %dir.display(),
                    "failed to watch SSL certificate directory"
                ),
            }
        }
        !self.dirs.is_empty()
    }
}

async fn run(
    mut watcher: CertWatcher,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<notify::Result<notify::Event>>,
    mut loaded: Vec<Option<Vec<u8>>>,
    tls: TlsConfig,
    http: ReloadableTls,
    grpc: Option<ReloadableTls>,
) {
    // Events do not say whether the files changed: a watched directory may
    // hold other files (such as a log), so compare the contents instead.
    let mut failed = None;
    while let Some(item) = rx.recv().await {
        log_watch_error(&item);
        // Coalesce the burst a rotation emits into one reload.
        let ceiling = tokio::time::sleep(DEBOUNCE);
        tokio::pin!(ceiling);
        loop {
            tokio::select! {
                _ = tokio::time::sleep(STEP) => break,
                _ = &mut ceiling => break,
                item = rx.recv() => match item {
                    Some(item) => log_watch_error(&item),
                    None => return,
                },
            }
        }
        // OpenSSL reads the files again, so a write racing that read can fail a
        // rebuild of contents that are fine; retry until one succeeds, warning
        // once per contents so a log in the directory cannot feed a loop.
        let contents = watcher.files.read();
        if contents != loaded {
            match reload(&tls, &http, grpc.as_ref()) {
                Ok(()) => {
                    info!("Reloaded SSL certificate chain");
                    loaded = contents;
                    failed = None;
                }
                Err(err) => {
                    if failed.as_ref() != Some(&contents) {
                        warn!(
                            error = %format_args!("{err:#}"),
                            "SSL certificate reload failed; keeping the current certificate"
                        );
                    }
                    failed = Some(contents);
                }
            }
        }
        watcher.watch_dirs();
    }
}

/// Log a watcher-reported error (e.g. inotify overflow) instead of dropping it.
/// The reload that follows re-reads the files, so an error is noted, not fatal.
fn log_watch_error(item: &notify::Result<notify::Event>) {
    if let Err(err) = item {
        warn!(error = %err.as_report(), "SSL certificate watcher reported an error");
    }
}

/// Rebuild the active TLS context(s) and swap them in, all-or-nothing: if any
/// rebuild fails (a partial or malformed write), keep the current certificate,
/// never serving a half-updated pair.
pub(crate) fn reload(
    tls: &TlsConfig,
    http: &ReloadableTls,
    grpc: Option<&ReloadableTls>,
) -> anyhow::Result<()> {
    let http_context = tls::build_server_config(tls)?;
    let grpc_context = grpc.map(|_| tls::build_grpc_server_config(tls)).transpose()?;

    http.store(http_context);
    if let (Some(cell), Some(context)) = (grpc, grpc_context) {
        cell.store(context);
    }
    Ok(())
}
