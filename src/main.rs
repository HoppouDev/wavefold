use dctenc::pipeline::{self, PipelineMsg};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

struct App {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    cutoff: f32,
    running: bool,
    progress_current: u64,
    progress_total: u64,
    log: Vec<String>,
    rx: Option<Receiver<PipelineMsg>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            input: None,
            output: None,
            cutoff: 0.6,
            running: false,
            progress_current: 0,
            progress_total: 0,
            log: Vec::new(),
            rx: None,
        }
    }
}

impl App {
    fn start_encode(&mut self) {
        let (Some(input), Some(output)) = (self.input.clone(), self.output.clone()) else { return };
        let cutoff = self.cutoff;
        let (tx, rx): (Sender<PipelineMsg>, Receiver<PipelineMsg>) = std::sync::mpsc::channel();
        self.rx = Some(rx);
        self.running = true;
        self.progress_current = 0;
        self.progress_total = 0;
        self.log.clear();
        std::thread::spawn(move || {
            pipeline::run(&input, &output, cutoff, tx);
        });
    }

    fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        loop {
            match rx.try_recv() {
                Ok(PipelineMsg::Progress { current, total }) => {
                    self.progress_current = current;
                    self.progress_total = total;
                }
                Ok(PipelineMsg::Log(l)) => self.log.push(l),
                Ok(PipelineMsg::Done) => {
                    self.log.push("done.".into());
                    self.running = false;
                }
                Ok(PipelineMsg::Error(e)) => {
                    self.log.push(format!("ERROR: {e}"));
                    self.running = false;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if self.running {
                        self.log.push("ERROR: encoder thread ended unexpectedly".into());
                        self.running = false;
                    }
                    break;
                }
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        if self.running {
            ui.ctx().request_repaint();
        }

        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("DCT GPU Video Encoder");
            ui.label("Distortion effect: transforms each entire frame as one whole-image DCT (cosine basis), keeps only the lowest-frequency coefficients, then re-encodes with ffmpeg (libx264). Dropping detail this way produces global ringing/ghosting across the whole frame rather than blocky artifacts.");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Select input video...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("video", &["mp4", "mov", "mkv", "avi", "webm", "m4v"])
                        .pick_file()
                    {
                        self.input = Some(path);
                    }
                }
                ui.label(self.input.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".into()));
            });

            ui.horizontal(|ui| {
                if ui.button("Select output file...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name("output.mp4")
                        .add_filter("mp4", &["mp4"])
                        .save_file()
                    {
                        self.output = Some(path);
                    }
                }
                ui.label(self.output.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".into()));
            });

            ui.separator();
            ui.add(egui::Slider::new(&mut self.cutoff, 0.0..=2.0).text("DCT spectrum cutoff"));
            ui.label("0 = DC only (max distortion, strong global ringing/ghosting). 2.0 = full spectrum (lossless).");

            ui.separator();
            let can_start = self.input.is_some() && self.output.is_some() && !self.running;
            if ui.add_enabled(can_start, egui::Button::new("Encode")).clicked() {
                self.start_encode();
            }

            if self.progress_total > 0 {
                let frac = self.progress_current as f32 / self.progress_total as f32;
                ui.add(egui::ProgressBar::new(frac).text(format!("{}/{}", self.progress_current, self.progress_total)));
            } else if self.running {
                ui.add(egui::ProgressBar::new(0.0).text(format!("frame {}", self.progress_current)).animate(true));
            }

            ui.separator();
            egui::ScrollArea::vertical().max_height(220.0).stick_to_bottom(true).show(ui, |ui| {
                for line in &self.log {
                    ui.monospace(line);
                }
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([640.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native("DCT GPU Video Encoder", options, Box::new(|_cc| Ok(Box::new(App::default()))))
}
