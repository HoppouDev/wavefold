use dctenc::dct_backend::ComputeBackend;
use dctenc::encoders::EncoderChoice;
use dctenc::pipeline::{self, PipelineMsg};
use iced::widget::{button, column, progress_bar, scrollable, text};
use iced::{Element, Fill, Task};
use std::path::PathBuf;
use tokio_stream::wrappers::UnboundedReceiverStream;

pub struct State {
    progress_current: u64,
    progress_total: u64,
    log: Vec<String>,
    running: bool,
}

#[derive(Debug, Clone)]
pub enum Message {
    Pipeline(PipelineMsg),
    WorkerDone,
    BackToSetup,
}

pub enum Action {
    None,
    BackToSetup,
}

impl State {
    /// Kicks off `pipeline::run` on a dedicated OS thread (it's a blocking
    /// call, not async) and returns a `Task` that streams its progress
    /// channel back reactively instead of the page polling every frame.
    pub fn start(input: PathBuf, output: PathBuf, cutoff: f32, encoder: EncoderChoice, backend: ComputeBackend) -> (Self, Task<Message>) {
        let state = Self { progress_current: 0, progress_total: 0, log: Vec::new(), running: true };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        std::thread::spawn(move || pipeline::run(&input, &output, cutoff, encoder, backend, tx));

        let task = Task::run(UnboundedReceiverStream::new(rx), Message::Pipeline).chain(Task::done(Message::WorkerDone));
        (state, task)
    }

    pub fn update(&mut self, message: Message) -> Action {
        match message {
            Message::Pipeline(msg) => {
                match msg {
                    PipelineMsg::Progress { current, total } => {
                        self.progress_current = current;
                        self.progress_total = total;
                    }
                    PipelineMsg::Log(line) => self.log.push(line),
                    PipelineMsg::Done => self.log.push("done.".into()),
                    PipelineMsg::Error(e) => self.log.push(format!("ERROR: {e}")),
                }
                Action::None
            }
            Message::WorkerDone => {
                self.running = false;
                Action::None
            }
            Message::BackToSetup => Action::BackToSetup,
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let progress: Element<'_, Message> = if self.progress_total > 0 {
            let frac = self.progress_current as f32 / self.progress_total as f32;
            column![
                progress_bar(0.0..=1.0, frac),
                text(format!("{}/{}", self.progress_current, self.progress_total)),
            ]
            .spacing(5)
            .into()
        } else if self.running {
            text(format!("frame {} (total unknown)", self.progress_current)).into()
        } else {
            column![].into()
        };

        let log_view = scrollable(column(self.log.iter().map(|line| text(line.clone()).font(iced::Font::MONOSPACE).into())).spacing(2))
            .height(300)
            .width(Fill);

        let back_button = button("New encode").on_press_maybe((!self.running).then_some(Message::BackToSetup));

        column![
            text("Encoding").size(24),
            progress,
            log_view,
            back_button,
        ]
        .spacing(12)
        .padding(16)
        .into()
    }
}
