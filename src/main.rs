mod ui;

use clap::{Parser, Subcommand};
use wavefold::dct_backend::ComputeBackend;
use wavefold::encoders::EncoderChoice;
use wavefold::pipeline::{self, PipelineMsg};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "wavefold", about = "Apply a whole-frame DCT distortion effect to video — GUI or headless")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Launch the desktop GUI (also the default when no subcommand is given)
    Gui,
    /// Run a headless encode, no display server required
    Encode {
        /// Input video path
        input: PathBuf,
        /// Output video path
        output: PathBuf,
        /// DCT spectrum cutoff: 0 = DC only (max distortion), 2.0 = full spectrum (lossless)
        #[arg(long, default_value_t = 0.6)]
        cutoff: f32,
        /// Output video encoder
        #[arg(long, value_enum, default_value_t = EncoderChoice::H264)]
        encoder: EncoderChoice,
        /// DCT compute backend. `cpu` needs no GPU at all (e.g. for CI runners).
        #[arg(long, value_enum, default_value_t = ComputeBackend::Gpu)]
        backend: ComputeBackend,
    },
}

fn main() {
    tracing_subscriber::fmt::init();
    match Cli::parse().command.unwrap_or(Command::Gui) {
        Command::Gui => {
            if let Err(e) = run_gui() {
                eprintln!("GUI error: {e}");
                std::process::exit(1);
            }
        }
        Command::Encode { input, output, cutoff, encoder, backend } => {
            run_encode(input, output, cutoff, encoder, backend);
        }
    }
}

fn run_gui() -> iced::Result {
    iced::application(ui::App::default, ui::App::update, ui::App::view)
        .title("Wavefold")
        .window_size((640.0, 640.0))
        .run()
}

fn run_encode(input: PathBuf, output: PathBuf, cutoff: f32, encoder: EncoderChoice, backend: ComputeBackend) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = std::thread::spawn(move || pipeline::run(&input, &output, cutoff, encoder, backend, tx));

    let mut had_error = false;
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            PipelineMsg::Progress { current, total } => {
                if total > 0 {
                    print!("\rframe {current}/{total}");
                } else {
                    print!("\rframe {current} (total unknown)");
                }
                let _ = std::io::stdout().flush();
            }
            PipelineMsg::Log(line) => println!("\n{line}"),
            PipelineMsg::Done => println!("\ndone"),
            PipelineMsg::Error(e) => {
                eprintln!("\nerror: {e}");
                had_error = true;
            }
        }
    }
    handle.join().expect("pipeline thread panicked");

    if had_error {
        std::process::exit(1);
    }
}
