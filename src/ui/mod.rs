pub mod encoding;
pub mod setup;

use iced::{Element, Task};

pub struct App {
    screen: Screen,
}

enum Screen {
    Setup(setup::State),
    Encoding(encoding::State),
}

#[derive(Debug, Clone)]
pub enum Message {
    Setup(setup::Message),
    Encoding(encoding::Message),
}

impl Default for App {
    fn default() -> Self {
        Self { screen: Screen::Setup(setup::State::default()) }
    }
}

impl App {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match (&mut self.screen, message) {
            (Screen::Setup(state), Message::Setup(msg)) => match state.update(msg) {
                setup::Action::None => Task::none(),
                setup::Action::Run(task) => task.map(Message::Setup),
                setup::Action::Start { input, output, cutoff, encoder, backend } => {
                    let (encoding_state, task) = encoding::State::start(input, output, cutoff, encoder, backend);
                    self.screen = Screen::Encoding(encoding_state);
                    task.map(Message::Encoding)
                }
            },
            (Screen::Encoding(state), Message::Encoding(msg)) => match state.update(msg) {
                encoding::Action::None => Task::none(),
                encoding::Action::BackToSetup => {
                    self.screen = Screen::Setup(setup::State::default());
                    Task::none()
                }
            },
            // A message meant for the screen we've since navigated away
            // from (e.g. a stray file-dialog result after leaving Setup).
            _ => Task::none(),
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        match &self.screen {
            Screen::Setup(state) => state.view().map(Message::Setup),
            Screen::Encoding(state) => state.view().map(Message::Encoding),
        }
    }
}
