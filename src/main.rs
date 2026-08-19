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
    use_bundled_gstreamer_plugins_if_present();
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

/// The Windows release (installer and .zip) bundles the GStreamer runtime
/// and its plugins in a `gstreamer-1.0` folder next to `wavefold.exe`,
/// since Windows has no system package manager providing it the way
/// Linux distros do. `gst::init()` only picks up plugins from
/// `GST_PLUGIN_PATH`/the system registry, so point it at that folder
/// when present. No-op on Linux/macOS, where GStreamer is expected to
/// already be installed system-wide (see README).
fn use_bundled_gstreamer_plugins_if_present() {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(plugin_dir) = exe.parent().map(|dir| dir.join("gstreamer-1.0")) {
            if plugin_dir.is_dir() {
                std::env::set_var("GST_PLUGIN_PATH", plugin_dir);
            }
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
