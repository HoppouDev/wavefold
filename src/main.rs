// Suppresses the console window Windows otherwise pops up for every launch
// (double-clicking the exe, the Start Menu shortcut, etc.) since this is
// primarily a GUI app there. `main()` reattaches to the parent console (if
// any) up front so `wavefold encode ...` run from an existing terminal
// still prints normally - only launches with no parent console (double
// click) end up with no console at all, which is what we want.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod ui;

use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use wavefold::codec::Codec;
use wavefold::dct_backend::{ComputeBackend, DctAlgorithm};
use wavefold::media_backend::MediaBackendChoice;
use wavefold::pipeline::{self, PipelineMsg};

#[derive(Parser)]
#[command(
    name = "wavefold",
    about = "Apply a whole-frame DCT distortion effect to video — GUI or headless"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Launch the desktop GUI
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
        #[arg(long, value_enum, default_value_t = Codec::H264)]
        encoder: Codec,
        /// DCT compute backend. `cpu` needs no GPU at all (e.g. for CI runners).
        #[arg(long, value_enum, default_value_t = ComputeBackend::Gpu)]
        backend: ComputeBackend,
        /// DCT algorithm (GPU backend only): the O(N log N) FFT-based path,
        /// or the original O(N^2) matrix-multiply.
        #[arg(long, value_enum, default_value_t = DctAlgorithm::Fft)]
        dct_algorithm: DctAlgorithm,
        /// Decode/encode implementation. Only one exists today; this is the
        /// selector for when another is added.
        #[arg(long, value_enum, default_value_t = MediaBackendChoice::ALL[0])]
        media_backend: MediaBackendChoice,
    },
}

fn main() {
    // Re-attach to whatever console launched us (e.g. cmd.exe/PowerShell),
    // if any - `windows_subsystem = "windows"` above means we start with no
    // console at all, so without this `encode` output would silently go
    // nowhere when run from an existing terminal. No-op (and no console) on
    // a double-click launch, which has no parent console to attach to.
    #[cfg(windows)]
    unsafe {
        let _ = windows::Win32::System::Console::AttachConsole(
            windows::Win32::System::Console::ATTACH_PARENT_PROCESS,
        );
    }

    let base = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into());
    let filter = tracing_subscriber::EnvFilter::new(format!("{base},winit=warn,iced_winit=warn"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    match Cli::parse().command {
        Command::Gui => {
            if let Err(e) = run_gui() {
                eprintln!("GUI error: {e}");
                std::process::exit(1);
            }
        }
        Command::Encode {
            input,
            output,
            cutoff,
            encoder,
            backend,
            dct_algorithm,
            media_backend,
        } => {
            run_encode(
                input,
                output,
                cutoff,
                encoder,
                backend,
                dct_algorithm,
                media_backend,
            );
        }
    }
}

fn run_gui() -> iced::Result {
    iced::application(ui::App::default, ui::App::update, ui::App::view)
        .title("Wavefold")
        .window(iced::window::Settings {
            size: (640.0, 640.0).into(),
            icon: Some(load_icon()),
            ..Default::default()
        })
        .run()
}

fn load_icon() -> iced::window::Icon {
    let image = image::load_from_memory(include_bytes!("../assets/windows/icon.png"))
        .expect("embedded app icon is a valid PNG")
        .to_rgba8();
    let (width, height) = image.dimensions();
    iced::window::icon::from_rgba(image.into_raw(), width, height)
        .expect("embedded app icon has valid dimensions")
}

fn run_encode(
    input: PathBuf,
    output: PathBuf,
    cutoff: f32,
    encoder: Codec,
    backend: ComputeBackend,
    dct_algorithm: DctAlgorithm,
    media_backend: MediaBackendChoice,
) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = std::thread::spawn(move || {
        pipeline::run(
            &input,
            &output,
            cutoff,
            encoder,
            backend,
            dct_algorithm,
            media_backend,
            tx,
        )
    });

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
