use clap::Parser;
use dctenc::dct_backend::ComputeBackend;
use dctenc::encoders::EncoderChoice;
use dctenc::pipeline::{self, PipelineMsg};
use std::io::Write;
use std::path::PathBuf;

/// Headless CLI for the DCT GPU/CPU video distortion effect — the same
/// `pipeline::run` the GUI drives, without a display server, so this can
/// run in a GitHub Actions runner or any other headless environment.
#[derive(Parser)]
#[command(name = "dctenc-cli", about = "Apply the whole-frame DCT distortion effect to a video, headlessly")]
struct Cli {
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
}

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = std::thread::spawn(move || pipeline::run(&cli.input, &cli.output, cli.cutoff, cli.encoder, cli.backend, tx));

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
