use wavefold::codec::Codec;
use wavefold::dct_backend::ComputeBackend;
use iced::widget::{button, column, pick_list, row, slider, text};
use iced::{Element, Task};
use std::path::PathBuf;

pub struct State {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    cutoff: f32,
    encoder: Codec,
    backend: ComputeBackend,
}

impl Default for State {
    fn default() -> Self {
        Self { input: None, output: None, cutoff: 0.6, encoder: Codec::H264, backend: ComputeBackend::Gpu }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    PickInput,
    InputPicked(Option<PathBuf>),
    PickOutput,
    OutputPicked(Option<PathBuf>),
    CutoffChanged(f32),
    EncoderSelected(Codec),
    BackendSelected(ComputeBackend),
    Start,
}

/// What the parent (`crate::ui::App`) should do after handling a message on
/// this page — `Start` carries everything the encoding page needs to kick
/// off `pipeline::run`, since this page's own state doesn't survive the
/// transition.
pub enum Action {
    None,
    Run(Task<Message>),
    Start { input: PathBuf, output: PathBuf, cutoff: f32, encoder: Codec, backend: ComputeBackend },
}

impl State {
    pub fn update(&mut self, message: Message) -> Action {
        match message {
            Message::PickInput => Action::Run(Task::perform(pick_input_file(), Message::InputPicked)),
            Message::InputPicked(path) => {
                if path.is_some() {
                    self.input = path;
                }
                Action::None
            }
            Message::PickOutput => Action::Run(Task::perform(pick_output_file(), Message::OutputPicked)),
            Message::OutputPicked(path) => {
                if path.is_some() {
                    self.output = path;
                }
                Action::None
            }
            Message::CutoffChanged(cutoff) => {
                self.cutoff = cutoff;
                Action::None
            }
            Message::EncoderSelected(encoder) => {
                self.encoder = encoder;
                Action::None
            }
            Message::BackendSelected(backend) => {
                self.backend = backend;
                Action::None
            }
            Message::Start => {
                let (Some(input), Some(output)) = (self.input.clone(), self.output.clone()) else {
                    return Action::None;
                };
                Action::Start { input, output, cutoff: self.cutoff, encoder: self.encoder, backend: self.backend }
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let input_row = row![
            button("Select input video...").on_press(Message::PickInput),
            text(self.input.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".into())),
        ]
        .spacing(10);

        let output_row = row![
            button("Select output file...").on_press(Message::PickOutput),
            text(self.output.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".into())),
        ]
        .spacing(10);

        let cutoff_row = column![
            slider(0.0..=2.0, self.cutoff, Message::CutoffChanged).step(0.01f32),
            text(format!("DCT spectrum cutoff: {:.2}", self.cutoff)),
            text("0 = DC only (max distortion, strong global ringing/ghosting). 2.0 = full spectrum (lossless)."),
        ]
        .spacing(5);

        let encoder_row = row![
            text("Encoder:"),
            pick_list(Codec::ALL, Some(self.encoder), Message::EncoderSelected),
        ]
        .spacing(10);

        let backend_row = row![
            text("Compute backend:"),
            pick_list(ComputeBackend::ALL, Some(self.backend), Message::BackendSelected),
        ]
        .spacing(10);

        let can_start = self.input.is_some() && self.output.is_some();
        let start_button = button("Encode").on_press_maybe(can_start.then_some(Message::Start));

        column![
            text("Wavefold").size(24),
            text("Distortion effect: transforms each entire frame as one whole-image DCT (cosine basis), keeps only the lowest-frequency coefficients, then re-encodes with ffmpeg. Dropping detail this way produces global ringing/ghosting across the whole frame rather than blocky artifacts."),
            input_row,
            output_row,
            cutoff_row,
            encoder_row,
            backend_row,
            start_button,
        ]
        .spacing(12)
        .padding(16)
        .into()
    }
}

async fn pick_input_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("video", &["mp4", "mov", "mkv", "avi", "webm", "m4v"])
        .pick_file()
        .await
        .map(|f| f.path().to_path_buf())
}

async fn pick_output_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_file_name("output.mp4")
        .add_filter("mp4", &["mp4"])
        .save_file()
        .await
        .map(|f| f.path().to_path_buf())
}
