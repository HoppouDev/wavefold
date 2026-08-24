//! Localhost TCP JSON-line automation server for UI testing / AI-driven
//! interaction, feature-gated (`automation`, off by default) and built as
//! a straight replacement for OS-level screenshot+input-injection: that
//! approach turned out to depend on window-manager focus timing, per-
//! compositor input-injection APIs (X11 vs native Wayland), and pixel-to-
//! screen coordinate mapping that isn't even stable across window
//! decorations - none of which exists on every OS. Driving the app through
//! its own real `Message` type sidesteps all of that and works identically
//! on Linux/Windows/macOS.
//!
//! Protocol: newline-delimited JSON on `127.0.0.1:{PORT}`. Each line is a
//! [`Request`] - either `"snapshot"` (read current state, no side effect)
//! or `{"inject": <Message>}` (inject a real `crate::Message`, exactly as
//! if a widget had produced it, and wait for it to actually be applied).
//! Every request gets exactly one JSON line back: the resulting
//! [`Snapshot`]. `PickInput`/`PickOutput` still open a real OS file dialog
//! when injected - automation should send `InputPicked`/`OutputPicked`
//! directly instead, which is exactly what those dialogs themselves
//! produce, so there's no separate "automation-only" message variant to
//! keep in sync with the real ones.
//!
//! **No authentication, by design, not oversight**: the loopback bind
//! keeps this off the network, but any other local process can still
//! connect and inject messages - including `Setup::Start`, which spawns a
//! real background encode with attacker-chosen input/output paths. The
//! `automation` Cargo feature gates *compilation*, not runtime access:
//! once a build with it is running, the socket is open to anyone on the
//! machine for as long as the process lives. That's the same tradeoff
//! Chrome's remote-debugging port and most local dev/test control planes
//! make - acceptable for a build nobody ships and that only runs when a
//! developer deliberately launches it for testing, not something to carry
//! into a build meant to run unattended or on a shared/multi-tenant host.
//!
//! **Every `inject` gets its own dedicated one-shot reply**, not a shared
//! "latest state" slot: `Handle::begin_request` hands out a fresh
//! correlation id paired with a `oneshot::Receiver`, and `Handle::publish`
//! - called from `App::update` via [`super::Envelope`] carrying that id -
//! delivers the resulting `Snapshot` straight to that specific receiver.
//! An earlier version tried to do this with `tokio::sync::watch` alone
//! (compare a correlation id embedded in the "current value" against the
//! one a client was waiting for), which looked correct but wasn't: `watch`
//! only ever holds the *latest* value, so a second, unrelated publish
//! landing between "my id showed up" and "I finished reading the value"
//! (a real window even with no explicit `.await` in between, since other
//! clients' tasks run on other OS threads and can genuinely race in) could
//! silently overwrite it first - confirmed empirically with a 4-client
//! concurrent stress test that reliably produced cross-client
//! contamination (client A reading client B's value) despite the
//! correlation id check. A one-shot channel has no shared slot to race
//! over: nothing but `App::update`'s one `send()` for this exact id can
//! ever write to it.

use super::{Envelope, Message};
use iced::futures::channel::mpsc;
use iced::futures::SinkExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, watch};
use tracing::{debug, error, info};

pub const PORT: u16 = 47624;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Request {
    Snapshot,
    Inject(Message),
}

/// Mirrors `ui::Screen`, but plain data - what a client actually receives.
#[derive(Serialize, Clone)]
#[serde(tag = "screen", rename_all = "snake_case")]
pub enum Snapshot {
    Setup(super::setup::Snapshot),
    Encoding(super::encoding::Snapshot),
}

struct Inner {
    // Current state for passive `"snapshot"` reads and `receiver_count()`
    // (see `has_subscribers`) - *not* used to correlate `inject` responses
    // anymore (see module doc for why that didn't work).
    state: watch::Sender<Snapshot>,
    next_id: AtomicU64,
    // One entry per in-flight `inject` request, removed either by
    // `publish` (the normal path: the message was applied, the reply was
    // sent) or by the requester itself on timeout (the message was never
    // dispatched - see `App::update`'s `dispatched` check - so nothing
    // will ever remove it otherwise, which would otherwise leak one entry
    // per wrong-screen/dropped injection for the life of the process).
    pending: Mutex<HashMap<u64, oneshot::Sender<Snapshot>>>,
}

/// Held by `App` (published to after every `update`, real or injected) and
/// cloned into the subscription's worker (read from, to answer clients).
/// `Hash`/`Eq` are by pointer identity, not contents - `App::subscription`
/// hands the *same* `Handle` back on every call, and iced uses this to
/// recognize the automation server as the same still-running subscription
/// instead of tearing it down and rebinding the port every frame.
#[derive(Clone)]
pub struct Handle(Arc<Inner>);

impl PartialEq for Handle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Handle {}

impl Hash for Handle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl Handle {
    pub fn new(initial: Snapshot) -> Self {
        let (state, _) = watch::channel(initial);
        Self(Arc::new(Inner { state, next_id: AtomicU64::new(0), pending: Mutex::new(HashMap::new()) }))
    }

    /// Always updates the broadcast "current state" (for passive `snapshot`
    /// reads), and - if `correlation_id` names a still-pending request -
    /// delivers this exact `Snapshot` to that request's one-shot receiver.
    /// A request only stays pending until its reply is sent or its
    /// requester gives up on timeout, so a stale/unknown id (already
    /// replied to, or never registered) is simply not found and ignored.
    pub fn publish(&self, correlation_id: Option<u64>, snapshot: Snapshot) {
        if let Some(id) = correlation_id {
            if let Some(tx) = self.0.pending.lock().expect("automation pending mutex poisoned").remove(&id) {
                let _ = tx.send(snapshot.clone());
            }
        }
        let _ = self.0.state.send(snapshot);
    }

    /// Whether any automation client is currently connected - lets
    /// `App::update` skip building a `Snapshot` at all (see its own doc
    /// comment) when nothing would ever read it.
    pub fn has_subscribers(&self) -> bool {
        self.0.state.receiver_count() > 0
    }

    /// Registers a fresh correlation id together with the one-shot channel
    /// `publish` will use to reply to it, once a message carrying this id
    /// is actually applied.
    fn begin_request(&self) -> (u64, oneshot::Receiver<Snapshot>) {
        let id = self.0.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.0.pending.lock().expect("automation pending mutex poisoned").insert(id, tx);
        (id, rx)
    }

    /// Cleans up a request's entry if it's still pending - called by the
    /// requester itself after giving up (timeout), since a message that
    /// was never dispatched (wrong screen, or the app never got to it)
    /// means `publish` will never be called for this id to remove it.
    fn cancel_request(&self, id: u64) {
        self.0.pending.lock().expect("automation pending mutex poisoned").remove(&id);
    }
}

pub fn subscription(handle: &Handle) -> iced::Subscription<Envelope> {
    iced::Subscription::run_with(handle.clone(), |handle| {
        let handle = handle.clone();
        iced::stream::channel(32, move |output: mpsc::Sender<Envelope>| async move {
            let listener = match TcpListener::bind(("127.0.0.1", PORT)).await {
                Ok(listener) => listener,
                Err(e) => {
                    error!("automation: failed to bind 127.0.0.1:{PORT}: {e}");
                    return;
                }
            };
            info!("automation: listening on 127.0.0.1:{PORT}");
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        tokio::spawn(handle_client(stream, handle.clone(), output.clone()));
                    }
                    Err(e) => {
                        // A transient OS-level error (e.g. EMFILE) would
                        // otherwise spin this loop with no delay and no
                        // diagnostic - log it and back off briefly instead
                        // of silently burning a core.
                        error!("automation: accept() failed: {e}");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        })
    })
}

async fn handle_client(stream: TcpStream, handle: Handle, mut output: mpsc::Sender<Envelope>) {
    let mut state = handle.0.state.subscribe();
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();

    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) if !line.trim().is_empty() => line,
            Ok(Some(_)) => continue,
            Ok(None) => return, // client closed the connection
            Err(e) => {
                debug!("automation: read error, dropping connection: {e}");
                return;
            }
        };

        let response = match serde_json::from_str::<Request>(&line) {
            Ok(Request::Snapshot) => serde_json::to_string(&*state.borrow_and_update()),
            Ok(Request::Inject(message)) => {
                let (id, rx) = handle.begin_request();
                if output.send(Envelope::from_automation(id, message)).await.is_err() {
                    // The stream side is gone (app shutting down) - nothing
                    // left to respond to a client about.
                    handle.cancel_request(id);
                    return;
                }
                match tokio::time::timeout(Duration::from_secs(5), rx).await {
                    Ok(Ok(snapshot)) => serde_json::to_string(&snapshot),
                    // Timed out (message never dispatched, or the app is
                    // stuck) or the sender was dropped without replying -
                    // clean up the registration and fall back to whatever
                    // the current state happens to be.
                    _ => {
                        handle.cancel_request(id);
                        serde_json::to_string(&*state.borrow_and_update())
                    }
                }
            }
            Err(e) => serde_json::to_string(&serde_json::json!({ "error": e.to_string() })),
        };

        let Ok(mut json) = response else { continue };
        json.push('\n');
        if write_half.write_all(json.as_bytes()).await.is_err() {
            return;
        }
    }
}
