use dctenc::encoders::EncoderChoice;
use dctenc::pipeline::{self, PipelineMsg};
use iced::widget::{button, column, pick_list, progress_bar, row, scrollable, slider, text};
use iced::{Element, Fill, Task};
use std::path::PathBuf;
use tokio_stream::wrappers::UnboundedReceiverStream;

struct App {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    cutoff: f32,
    encoder: EncoderChoice,
    running: bool,
    progress_current: u64,
    progress_total: u64,
    log: Vec<String>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            input: None,
            output: None,
            cutoff: 0.6,
            encoder: EncoderChoice::H264,
            running: false,
            progress_current: 0,
            progress_total: 0,
            log: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    PickInput,
    InputPicked(Option<PathBuf>),
    PickOutput,
    OutputPicked(Option<PathBuf>),
    CutoffChanged(f32),
    EncoderSelected(EncoderChoice),
    StartEncode,
    Pipeline(PipelineMsg),
    WorkerDone,
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

impl App {
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PickInput => Task::perform(pick_input_file(), Message::InputPicked),
            Message::InputPicked(path) => {
                if path.is_some() {
                    self.input = path;
                }
                Task::none()
            }
            Message::PickOutput => Task::perform(pick_output_file(), Message::OutputPicked),
            Message::OutputPicked(path) => {
                if path.is_some() {
                    self.output = path;
                }
                Task::none()
            }
            Message::CutoffChanged(cutoff) => {
                self.cutoff = cutoff;
                Task::none()
            }
            Message::EncoderSelected(encoder) => {
                self.encoder = encoder;
                Task::none()
            }
            Message::StartEncode => {
                let (Some(input), Some(output)) = (self.input.clone(), self.output.clone()) else {
                    return Task::none();
                };
                self.running = true;
                self.progress_current = 0;
                self.progress_total = 0;
                self.log.clear();
                let cutoff = self.cutoff;
                let encoder = self.encoder;

                // `pipeline::run` internally spawns its own producer thread
                // and blocks the calling thread until the whole encode is
                // done, so it runs on a plain OS thread here rather than
                // inside this async Task; progress streams back over a
                // tokio channel, which `Task::run` turns into a `Message`
                // per item instead of requiring the UI to poll each frame.
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                std::thread::spawn(move || pipeline::run(&input, &output, cutoff, encoder, tx));

                Task::run(UnboundedReceiverStream::new(rx), Message::Pipeline)
                    .chain(Task::done(Message::WorkerDone))
            }
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
                Task::none()
            }
            Message::WorkerDone => {
                self.running = false;
                Task::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
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
            pick_list(EncoderChoice::ALL, Some(self.encoder), Message::EncoderSelected),
        ]
        .spacing(10);

        let can_start = self.input.is_some() && self.output.is_some() && !self.running;
        let encode_button = button("Encode").on_press_maybe(can_start.then_some(Message::StartEncode));

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
            .height(220)
            .width(Fill);

        column![
            text("DCT GPU Video Encoder").size(24),
            text("Distortion effect: transforms each entire frame as one whole-image DCT (cosine basis), keeps only the lowest-frequency coefficients, then re-encodes with ffmpeg. Dropping detail this way produces global ringing/ghosting across the whole frame rather than blocky artifacts."),
            input_row,
            output_row,
            cutoff_row,
            encoder_row,
            encode_button,
            progress,
            log_view,
        ]
        .spacing(12)
        .padding(16)
        .into()
    }
}

fn main() -> iced::Result {
    tracing_subscriber::fmt::init();
    iced::application(App::default, App::update, App::view)
        .title("DCT GPU Video Encoder")
        .window_size((640.0, 640.0))
        .run()
}
