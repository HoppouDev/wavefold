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
    pub fn update(&mut self, message: Message) -> Task<Message> {
        let (task, dispatched) = match (&mut self.screen, message) {
            (Screen::Setup(state), Message::Setup(msg)) => (
                match state.update(msg) {
                    setup::Action::None => Task::none(),
                    setup::Action::Run(task) => task.map(Message::Setup),
                    setup::Action::Start { input, output, cutoff, encoder, backend, dct_algorithm } => {
                        let (encoding_state, task) = encoding::State::start(input, output, cutoff, encoder, backend, dct_algorithm);
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
            self.automation.publish(self.screen.snapshot());
        }
        task
    }

    pub fn view(&self) -> Element<'_, Message> {
        match &self.screen {
            Screen::Setup(state) => state.view().map(Message::Setup),
            Screen::Encoding(state) => state.view().map(Message::Encoding),
        }
    }

    #[cfg(feature = "automation")]
    pub fn subscription(&self) -> iced::Subscription<Message> {
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
