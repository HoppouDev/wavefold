use dctenc::pipeline::{self, PipelineMsg};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn unique_path(suffix: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("dctenc_integration_{}_{}_{}", std::process::id(), id, suffix))
}

fn make_test_clip(width: u32, height: u32, fps: u32, seconds: u32) -> std::path::PathBuf {
    let path = unique_path("in.mp4");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "lavfi"])
        .args(["-i", &format!("testsrc=size={width}x{height}:duration={seconds}:rate={fps}")])
        .args(["-pix_fmt", "yuv420p", "-y"])
        .arg(&path)
        .status()
        .expect("failed to spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed to generate test clip");
    path
}

fn probe_dims_and_frames(path: &std::path::Path) -> (u32, u32, u64) {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,nb_read_frames"])
        .args(["-count_frames", "-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .expect("failed to spawn ffprobe");
    assert!(out.status.success(), "ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let mut w = 0u32;
    let mut h = 0u32;
    let mut frames = 0u64;
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            match k {
                "width" => w = v.parse().unwrap_or(0),
                "height" => h = v.parse().unwrap_or(0),
                "nb_read_frames" => frames = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    (w, h, frames)
}

/// Full pipeline: synthetic clip -> GPU DCT encode -> valid mp4 with the same
/// dimensions and frame count as the source. Skips (rather than fails) if
/// this environment has no ffmpeg or no usable GPU adapter, since both are
/// legitimately absent in some CI sandboxes.
#[test]
fn encodes_synthetic_clip_end_to_end() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    if dctenc::gpu::DctGpu::new().is_err() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }

    let input = make_test_clip(48, 32, 8, 1); // 8 frames
    let output = unique_path("out.mp4");

    let (tx, rx) = mpsc::channel();
    let input2 = input.clone();
    let output2 = output.clone();
    let handle = std::thread::spawn(move || pipeline::run(&input2, &output2, 0.4, tx));

    let mut saw_done = false;
    let mut saw_error = None;
    let mut last_progress = (0u64, 0u64);
    for msg in rx {
        match msg {
            PipelineMsg::Progress { current, total } => last_progress = (current, total),
            PipelineMsg::Log(_) => {}
            PipelineMsg::Done => saw_done = true,
            PipelineMsg::Error(e) => saw_error = Some(e),
        }
    }
    handle.join().unwrap();

    if let Some(e) = saw_error {
        panic!("pipeline reported an error: {e}");
    }
    assert!(saw_done, "pipeline never sent Done");
    assert_eq!(last_progress, (8, 8), "expected to process all 8 frames");

    assert!(output.exists(), "output file was not created");
    let (w, h, frames) = probe_dims_and_frames(&output);
    assert_eq!(w, 48);
    assert_eq!(h, 32);
    assert_eq!(frames, 8);

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn reports_error_for_nonexistent_input() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let output = unique_path("should_not_exist.mp4");
    let (tx, rx) = mpsc::channel();
    let out2 = output.clone();
    let handle = std::thread::spawn(move || {
        pipeline::run(std::path::Path::new("/nonexistent/dctenc_missing_input.mp4"), &out2, 0.32, tx)
    });

    let mut saw_error = false;
    for msg in rx {
        if let PipelineMsg::Error(_) = msg {
            saw_error = true;
        }
    }
    handle.join().unwrap();
    assert!(saw_error, "expected an Error message for a missing input file");
    assert!(!output.exists());
}
