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

use super::Message;
use iced::futures::channel::mpsc;
use iced::futures::SinkExt;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
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
    snapshot: Mutex<Snapshot>,
    generation: AtomicU64,
    notify: Notify,
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
        Self(Arc::new(Inner { snapshot: Mutex::new(initial), generation: AtomicU64::new(0), notify: Notify::new() }))
    }

    pub fn publish(&self, snapshot: Snapshot) {
        *self.0.snapshot.lock().expect("automation snapshot mutex poisoned") = snapshot;
        self.0.generation.fetch_add(1, Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }

    fn snapshot(&self) -> Snapshot {
        self.0.snapshot.lock().expect("automation snapshot mutex poisoned").clone()
    }

    fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::SeqCst)
    }

    /// Waits until `generation()` has moved past `before`, or `timeout`
    /// elapses - `Notify` can miss a wakeup that fires between checking the
    /// counter and starting to wait, so this loops rather than awaiting
    /// `notified()` exactly once.
    async fn wait_for_generation_past(&self, before: u64, timeout: Duration) {
        let wait = async {
            while self.generation() == before {
                self.0.notify.notified().await;
            }
        };
        let _ = tokio::time::timeout(timeout, wait).await;
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
                let Ok((stream, _)) = listener.accept().await else { continue };
                tokio::spawn(handle_client(stream, handle.clone(), output.clone()));
            }
        })
    })
}

async fn handle_client(stream: TcpStream, handle: Handle, mut output: mpsc::Sender<Message>) {
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
            Ok(Request::Snapshot) => serde_json::to_string(&handle.snapshot()),
            Ok(Request::Inject(message)) => {
                let before = handle.generation();
                if output.send(message).await.is_err() {
                    // The stream side is gone (app shutting down) - nothing
                    // left to respond to a client about.
                    return;
                }
                handle.wait_for_generation_past(before, Duration::from_secs(5)).await;
                serde_json::to_string(&handle.snapshot())
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
