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

use super::Message;
use iced::futures::channel::mpsc;
use iced::futures::SinkExt;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
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

/// Held by `App` (published to after every `update`, real or injected) and
/// cloned into the subscription's worker (read from, to answer clients).
///
/// Backed by `tokio::sync::watch` rather than a hand-rolled `Mutex` +
/// generation counter + `Notify`: an earlier version used exactly that
/// combination, and it had a real missed-wakeup race - `Notify::notified()`
/// only catches a `notify_waiters()` call made *after* the `Notified`
/// future is created, but the old code created it *after* checking the
/// counter, leaving a window where a `publish()` landing between the check
/// and the `notified()` call would go unseen until the next unrelated
/// update or a 5s timeout (confirmed against `tokio`'s own docs/source, not
/// just suspected). `watch::Receiver::changed()` doesn't have that gap - it
/// compares against an internally-tracked version under the same lock the
/// sender updates, so there's no separate "register interest" step to lose
/// a race with.
#[derive(Clone)]
pub struct Handle(Arc<watch::Sender<Snapshot>>);

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
        Self(Arc::new(watch::channel(initial).0))
    }

    /// `send` only errors when every receiver has been dropped (no
    /// automation client currently connected), which is fine to ignore -
    /// the new value is still stored and the next client to `subscribe`
    /// sees it.
    pub fn publish(&self, snapshot: Snapshot) {
        let _ = self.0.send(snapshot);
    }

    /// Whether any automation client is currently connected - lets
    /// `App::update` skip building a `Snapshot` at all (see its own doc
    /// comment) when nothing would ever read it.
    pub fn has_subscribers(&self) -> bool {
        self.0.receiver_count() > 0
    }
}

pub fn subscription(handle: &Handle) -> iced::Subscription<Message> {
    iced::Subscription::run_with(handle.clone(), |handle| {
        let handle = handle.clone();
        iced::stream::channel(32, move |output: mpsc::Sender<Message>| async move {
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

async fn handle_client(stream: TcpStream, handle: Handle, mut output: mpsc::Sender<Message>) {
    let mut state = handle.0.subscribe();
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
                if output.send(message).await.is_err() {
                    // The stream side is gone (app shutting down) - nothing
                    // left to respond to a client about.
                    return;
                }
                let _ = tokio::time::timeout(Duration::from_secs(5), state.changed()).await;
                serde_json::to_string(&*state.borrow_and_update())
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
