#[cfg(feature = "automation")]
pub mod automation;
pub mod encoding;
pub mod setup;

use iced::{Element, Task};

pub struct App {
    screen: Screen,
    #[cfg(feature = "automation")]
    automation: automation::Handle,
}

enum Screen {
    Setup(setup::State),
    Encoding(encoding::State),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "automation", derive(serde::Serialize, serde::Deserialize))]
pub enum Message {
    Setup(setup::Message),
    Encoding(encoding::Message),
}

/// The type iced actually operates on for `update`/`view`/`subscription` -
/// `Message` plus an optional automation correlation id, so `App::update`
/// can tell *which* message it's processing instead of only knowing *that*
/// some message arrived. Real messages (widget events, `Task` results) are
/// always `correlation_id: None` via `From<Message>`; only
/// `automation::handle_client` constructs one with `Some(id)`, for exactly
/// as long as it takes to correlate one injected message with the publish
/// it caused (see `automation.rs`'s module doc for why this exists - a
/// shared "did *something* change" signal isn't enough once other message
/// sources, like an active encode's own progress events, can interleave).
#[derive(Debug, Clone)]
pub struct Envelope {
    #[cfg(feature = "automation")]
    correlation_id: Option<u64>,
    message: Message,
}

impl From<Message> for Envelope {
    fn from(message: Message) -> Self {
        Self {
            #[cfg(feature = "automation")]
            correlation_id: None,
            message,
        }
    }
}

#[cfg(feature = "automation")]
impl Envelope {
    pub(crate) fn from_automation(correlation_id: u64, message: Message) -> Self {
        Self { correlation_id: Some(correlation_id), message }
    }
}

impl Default for App {
    fn default() -> Self {
        let screen = Screen::Setup(setup::State::default());
        Self {
            #[cfg(feature = "automation")]
            automation: automation::Handle::new(screen.snapshot()),
            screen,
        }
    }
}

impl App {
    pub fn update(&mut self, envelope: Envelope) -> Task<Envelope> {
        let Envelope { #[cfg(feature = "automation")] correlation_id, message } = envelope;

        let (task, dispatched) = match (&mut self.screen, message) {
            (Screen::Setup(state), Message::Setup(msg)) => (
                match state.update(msg) {
                    setup::Action::None => Task::none(),
                    setup::Action::Run(task) => task.map(Message::Setup),
                    setup::Action::Start { input, output, cutoff, encoder, backend } => {
                        let (encoding_state, task) = encoding::State::start(input, output, cutoff, encoder, backend);
                        self.screen = Screen::Encoding(encoding_state);
                        task.map(Message::Encoding)
                    }
                },
                true,
            ),
            (Screen::Encoding(state), Message::Encoding(msg)) => (
                match state.update(msg) {
                    encoding::Action::None => Task::none(),
                    encoding::Action::BackToSetup => {
                        self.screen = Screen::Setup(setup::State::default());
                        Task::none()
                    }
                },
                true,
            ),
            // A message meant for the screen we've since navigated away
            // from (e.g. a stray file-dialog result after leaving Setup) -
            // not dispatched anywhere, so nothing about `self.screen`
            // actually changed.
            _ => (Task::none(), false),
        };

        // `snapshot()` clones the whole screen's state (including
        // `encoding::State`'s unbounded log) - skip building one at all
        // when no automation client is even connected to read it, instead
        // of paying that cost on every single message (real UI or
        // pipeline progress) for the entire life of the app. Also skip it
        // for a message that wasn't actually dispatched: publishing
        // unconditionally here would let an automation client that injects
        // a message for the wrong screen (e.g. a stale assumption about
        // which screen is active) get back a fast, normal-looking
        // response indistinguishable from a message that really was
        // applied - better to let it wait out `wavefold_automation.py`'s
        // timeout and see the genuinely-unchanged snapshot than lie to it
        // quickly.
        #[cfg(not(feature = "automation"))]
        let _ = dispatched;
        #[cfg(feature = "automation")]
        if dispatched && self.automation.has_subscribers() {
            self.automation.publish(correlation_id, self.screen.snapshot());
        }
        task.map(Envelope::from)
    }

    pub fn view(&self) -> Element<'_, Envelope> {
        match &self.screen {
            Screen::Setup(state) => state.view().map(Message::Setup).map(Envelope::from),
            Screen::Encoding(state) => state.view().map(Message::Encoding).map(Envelope::from),
        }
    }

    #[cfg(feature = "automation")]
    pub fn subscription(&self) -> iced::Subscription<Envelope> {
        automation::subscription(&self.automation)
    }
}

#[cfg(feature = "automation")]
impl Screen {
    fn snapshot(&self) -> automation::Snapshot {
        match self {
            Screen::Setup(state) => automation::Snapshot::Setup(state.snapshot()),
            Screen::Encoding(state) => automation::Snapshot::Encoding(state.snapshot()),
        }
    }
}
