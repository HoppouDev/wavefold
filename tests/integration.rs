use wavefold::dct_backend::ComputeBackend;
use wavefold::pipeline::{self, PipelineMsg};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn unique_path(suffix: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("wavefold_integration_{}_{}_{}", std::process::id(), id, suffix))
}

/// True for the one known Windows-only sink gap these mkv-sourced,
/// raw-PCM-audio fixtures can hit: `AddStream`/`SetInputMediaType` both
/// negotiate the passthrough PCM audio type fine, but `Finalize` fails
/// with 0xC00D4A45 ("required headers were not provided to the sink") -
/// confirmed directly that Windows' built-in MP4 sink's documented audio
/// support is essentially AAC-only, so it can't actually mux raw PCM
/// despite accepting the type. Same class of platform-specific gap as
/// `encodes_with_every_selectable_encoder`'s existing VP9/mp4 and
/// AV1-encoder-absent tolerances, just triggered by output container
/// instead of by codec choice.
fn is_windows_mp4_pcm_sink_gap(e: &str) -> bool {
    cfg!(windows) && e.contains("0xC00D4A45")
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
    if wavefold::gpu::DctGpu::new().is_err() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }

    // 64x64, not something tinier: confirmed directly that Windows' built-in
    // H.264 encoder MFT (`backends::media_foundation`) rejects
    // `SetInputMediaType` with 0xC00D36B4 ("invalid, inconsistent") below
    // some floor between 32 and 48px on either axis, regardless of input
    // subtype (reproduced with RGB32 and NV12 alike) - a genuine minimum-
    // resolution limitation in that MFT, not a media-type construction bug.
    // GStreamer's software encoders have no such floor, so this only bit
    // Windows.
    let input = make_test_clip(64, 64, 8, 1); // 8 frames
    let output = unique_path("out.mp4");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let input2 = input.clone();
    let output2 = output.clone();
    let handle = std::thread::spawn(move || pipeline::run(&input2, &output2, 0.4, wavefold::codec::Codec::H264, ComputeBackend::Gpu, wavefold::media_backend::MediaBackendChoice::ALL[0], tx));

    let mut saw_done = false;
    let mut saw_error = None;
    let mut last_progress = (0u64, 0u64);
    while let Some(msg) = rx.blocking_recv() {
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
    assert_eq!(w, 64);
    assert_eq!(h, 64);
    assert_eq!(frames, 8);

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// Full pipeline with the CPU DCT backend: no GPU adapter check at all —
/// this is the end-to-end proof the pipeline can run on a GPU-less machine
/// (e.g. a standard GitHub Actions runner).
#[test]
fn encodes_synthetic_clip_with_cpu_backend() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }

    let input = make_test_clip(64, 64, 8, 1); // 8 frames
    let output = unique_path("out_cpu_backend.mp4");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let input2 = input.clone();
    let output2 = output.clone();
    let handle = std::thread::spawn(move || {
        pipeline::run(&input2, &output2, 0.4, wavefold::codec::Codec::H264, ComputeBackend::Cpu, wavefold::media_backend::MediaBackendChoice::ALL[0], tx)
    });

    let mut saw_done = false;
    let mut saw_error = None;
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            PipelineMsg::Error(e) => saw_error = Some(e),
            PipelineMsg::Done => saw_done = true,
            _ => {}
        }
    }
    handle.join().unwrap();

    if let Some(e) = saw_error {
        panic!("pipeline reported an error: {e}");
    }
    assert!(saw_done, "pipeline never sent Done");
    assert!(output.exists(), "output file was not created");

    let (w, h, frames) = probe_dims_and_frames(&output);
    assert_eq!(w, 64);
    assert_eq!(h, 64);
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
    let (tx, mut rx) = mpsc::unbounded_channel();
    let out2 = output.clone();
    let handle = std::thread::spawn(move || {
        pipeline::run(std::path::Path::new("/nonexistent/wavefold_missing_input.mp4"), &out2, 0.32, wavefold::codec::Codec::H264, ComputeBackend::Gpu, wavefold::media_backend::MediaBackendChoice::ALL[0], tx)
    });

    let mut saw_error = false;
    while let Some(msg) = rx.blocking_recv() {
        if let PipelineMsg::Error(_) = msg {
            saw_error = true;
        }
    }
    handle.join().unwrap();
    assert!(saw_error, "expected an Error message for a missing input file");
    assert!(!output.exists());
}

#[test]
fn encodes_synthetic_clip_with_audio_end_to_end() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    if wavefold::gpu::DctGpu::new().is_err() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }

    let input = unique_path("in_audio.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-f", "lavfi"])
        .args(["-i", "testsrc=size=64x64:duration=1:rate=8"])
        .args(["-f", "lavfi"])
        .args(["-i", "sine=frequency=440:duration=1"])
        .args(["-pix_fmt", "yuv420p", "-c:a", "aac", "-y"])
        .arg(&input)
        .status()
        .expect("failed to spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed to generate test clip with audio");

    let output = unique_path("out_audio.mp4");
    let (tx, mut rx) = mpsc::unbounded_channel();
    let input2 = input.clone();
    let output2 = output.clone();
    let handle = std::thread::spawn(move || pipeline::run(&input2, &output2, 0.4, wavefold::codec::Codec::H264, ComputeBackend::Gpu, wavefold::media_backend::MediaBackendChoice::ALL[0], tx));

    let mut saw_done = false;
    let mut saw_error = None;
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            PipelineMsg::Error(e) => saw_error = Some(e),
            PipelineMsg::Done => saw_done = true,
            _ => {}
        }
    }
    handle.join().unwrap();
    if let Some(e) = saw_error {
        panic!("pipeline reported an error: {e}");
    }
    assert!(saw_done, "pipeline never sent Done");
    assert!(output.exists(), "output file was not created");

    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args(["-show_entries", "stream=codec_type"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(&output)
        .output()
        .expect("failed to spawn ffprobe");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("codec_type=audio"), "output has no audio stream: {text}");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn encodes_with_every_selectable_encoder() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    if wavefold::gpu::DctGpu::new().is_err() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }

    for choice in wavefold::codec::Codec::ALL {
        // VAAPI encoders reject frames below their minimum coded size
        // (e.g. h264_vaapi on this system reports a 128x128 floor), so
        // this clip is larger than the other encoder tests' — small
        // enough to stay fast, comfortably above known hw minimums. Also
        // a multiple of 64 (HEVC's max CTU size): vah265enc on this
        // AMD/Mesa VAAPI driver pads non-64-aligned dimensions to the next
        // CTU boundary (160 -> 192) without writing a correct SPS
        // conformance-window crop back, so the muxed output's probed width
        // silently comes out padded — confirmed directly with gst-launch,
        // a genuine driver limitation, not a wavefold bug. Staying
        // 64-aligned sidesteps it (no padding needed) rather than papering
        // over a wrong-dimension output with a tolerant skip.
        let input = make_test_clip(192, 128, 6, 1); // 6 frames
        let output = unique_path("out_encoder.mp4");

        let (tx, mut rx) = mpsc::unbounded_channel();
        let input2 = input.clone();
        let output2 = output.clone();
        let handle = std::thread::spawn(move || pipeline::run(&input2, &output2, 0.5, choice, ComputeBackend::Gpu, wavefold::media_backend::MediaBackendChoice::ALL[0], tx));

        let mut saw_done = false;
        let mut saw_error = None;
        while let Some(msg) = rx.blocking_recv() {
            match msg {
                PipelineMsg::Error(e) => saw_error = Some(e),
                PipelineMsg::Done => saw_done = true,
                _ => {}
            }
        }
        handle.join().unwrap();

        if let Some(e) = saw_error {
            // Hardware encoder support is inherently environment-dependent
            // (no compatible GPU/driver at all, or a driver missing a
            // specific codec's encode entrypoint — e.g. this system's
            // VAAPI driver has no VP9 encode entrypoint even though the
            // device itself works fine for H.264/HEVC) — skip just this
            // variant rather than failing the whole test. Software
            // encoders have no such excuse in general — except software
            // VP9 into an mp4-family output specifically: this GStreamer
            // version's `vp9enc` never emits a `chroma-format` field in its
            // src caps (confirmed directly with `gst-launch-1.0`, tried
            // forcing every upstream pixel format vp9enc accepts — none
            // changed this), while `qtmux`'s `video/x-vp9` pad template
            // requires that field, so the two can never negotiate. VP9 into
            // matroskamux (no such requirement) works fine — this is a
            // muxer-specific gap, not a real wavefold bug. On Windows,
            // software AV1 is the same kind of platform gap: Media
            // Foundation ships an AV1 *decoder* MFT ("AV1 Video
            // Extensions") but no built-in *encoder* MFT at all, software
            // or hardware — confirmed via `SetInputMediaType` failing with
            // 0xC00D5212 ("no suitable transform"). `Codec::Av1Hardware`
            // is unaffected (a real GPU's own AV1 hardware encoder MFT, if
            // present, registers independently of this gap) and stays
            // covered by `is_hardware()` above.
            let win_has_no_av1_encoder = cfg!(windows) && matches!(choice, wavefold::codec::Codec::Av1);
            if choice.is_hardware() || matches!(choice, wavefold::codec::Codec::Vp9) || win_has_no_av1_encoder {
                eprintln!("skipping {choice:?}: {e}");
                let _ = std::fs::remove_file(&input);
                continue;
            }
            panic!("{:?} failed: {e}", choice);
        }
        assert!(saw_done, "{:?} never sent Done", choice);
        assert!(output.exists(), "{:?} produced no output file", choice);

        let (w, h, frames) = probe_dims_and_frames(&output);
        assert_eq!(w, 192, "{:?} wrong width", choice);
        assert_eq!(h, 128, "{:?} wrong height", choice);
        assert_eq!(frames, 6, "{:?} wrong frame count", choice);

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}

/// Regression test for a real-world failure: some containers (this repros
/// with a plain lavfi-generated pcm_s16le-in-mkv clip, same as e.g. some
/// camera-recorded mkv files) don't record an explicit audio channel
/// layout, so ffmpeg-next's demuxer reports it as unspecified
/// (`AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC`, `channel_layout=unknown` in
/// `ffprobe`). Passing that straight through to an mp4 output used to make
/// the mov muxer fail the whole encode with "unsupported channel layout N
/// channels" — `run_inner`'s audio setup now fills in a default layout for
/// the channel count when the source's is unspecified.
#[test]
fn encodes_clip_with_unspecified_channel_layout_audio() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    if wavefold::gpu::DctGpu::new().is_err() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }

    let input = unique_path("in_unspec_channels.mkv");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "lavfi"])
        .args(["-i", "testsrc=size=64x64:duration=1:rate=8"])
        .args(["-f", "lavfi"])
        .args(["-i", "sine=frequency=440:duration=1"])
        // `-pix_fmt yuv420p`: without it, libx264 picks up testsrc's native
        // chroma sampling and encodes High 4:4:4 Predictive/yuv444p
        // (confirmed directly) - a profile Windows' built-in H.264 decoder
        // MFT can't decode (`backends::media_foundation`'s `ReadSample`
        // fails with a generic "CopyDecodedFrame failed", 0x80004005).
        // `make_test_clip` above already pins yuv420p for the same reason;
        // this fixture is generated by hand instead of through that helper
        // specifically to reach an unspecified audio channel layout, so it
        // needs the same pin repeated here.
        .args(["-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "pcm_s16le", "-y"])
        .arg(&input)
        .status()
        .expect("failed to spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed to generate test clip");

    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args(["-show_entries", "stream=channel_layout"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(&input)
        .output()
        .expect("failed to spawn ffprobe");
    assert!(
        String::from_utf8_lossy(&probe.stdout).contains("channel_layout=unknown"),
        "test fixture no longer reproduces an unspecified channel layout — this test needs a new repro"
    );

    let output = unique_path("out_unspec_channels.mp4");
    let (tx, mut rx) = mpsc::unbounded_channel();
    let input2 = input.clone();
    let output2 = output.clone();
    let handle = std::thread::spawn(move || pipeline::run(&input2, &output2, 0.4, wavefold::codec::Codec::H264, ComputeBackend::Gpu, wavefold::media_backend::MediaBackendChoice::ALL[0], tx));

    let mut saw_done = false;
    let mut saw_error = None;
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            PipelineMsg::Error(e) => saw_error = Some(e),
            PipelineMsg::Done => saw_done = true,
            _ => {}
        }
    }
    handle.join().unwrap();

    if let Some(e) = saw_error {
        if is_windows_mp4_pcm_sink_gap(&e) {
            eprintln!("skipping: {e}");
            let _ = std::fs::remove_file(&input);
            return;
        }
        panic!("pipeline reported an error: {e}");
    }
    assert!(saw_done, "pipeline never sent Done");
    assert!(output.exists(), "output file was not created");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// Regression test for a real-world "total unknown" progress bug: Matroska
/// (and some other containers) commonly omit per-stream `duration`/
/// `nb_frames` metadata entirely, even though the overall file duration is
/// known (mkv stores it in the Segment Info, not per-track) — confirmed via
/// `ffprobe` that this synthesized clip reproduces exactly that (stream
/// duration/nb_frames both `N/A`, format-level duration present).
/// `run_inner`'s total-frame estimate now falls back to the format-level
/// duration in that case instead of leaving `total` at 0 ("unknown") for
/// the whole encode.
#[test]
fn estimates_total_frames_from_format_duration_when_stream_metadata_missing() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    if wavefold::gpu::DctGpu::new().is_err() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }

    let input = unique_path("in_no_stream_duration.mkv");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "lavfi"])
        .args(["-i", "testsrc=size=64x64:duration=1:rate=8"])
        .args(["-f", "lavfi"])
        .args(["-i", "sine=frequency=440:duration=1"])
        // `-pix_fmt yuv420p`: see the comment on the identical flag in
        // `encodes_clip_with_unspecified_channel_layout_audio` above -
        // without it libx264 encodes a profile Windows' H.264 decoder MFT
        // can't decode. This fixture is generated by hand instead of
        // through `make_test_clip` (which already pins yuv420p) because it
        // specifically needs mkv's no-per-stream-duration behavior.
        .args(["-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "pcm_s16le", "-y"])
        .arg(&input)
        .status()
        .expect("failed to spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed to generate test clip");

    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=duration,nb_frames"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(&input)
        .output()
        .expect("failed to spawn ffprobe");
    let probe_text = String::from_utf8_lossy(&probe.stdout);
    assert!(
        probe_text.contains("duration=N/A") && probe_text.contains("nb_frames=N/A"),
        "test fixture no longer reproduces missing stream-level duration/nb_frames — this test needs a new repro: {probe_text}"
    );

    let output = unique_path("out_no_stream_duration.mp4");
    let (tx, mut rx) = mpsc::unbounded_channel();
    let input2 = input.clone();
    let output2 = output.clone();
    let handle = std::thread::spawn(move || pipeline::run(&input2, &output2, 0.4, wavefold::codec::Codec::H264, ComputeBackend::Gpu, wavefold::media_backend::MediaBackendChoice::ALL[0], tx));

    let mut saw_done = false;
    let mut saw_error = None;
    let mut last_total = 0u64;
    while let Some(msg) = rx.blocking_recv() {
        match msg {
            PipelineMsg::Progress { total, .. } => last_total = total,
            PipelineMsg::Error(e) => saw_error = Some(e),
            PipelineMsg::Done => saw_done = true,
            _ => {}
        }
    }
    handle.join().unwrap();

    // 1 second at 8fps: expect an estimate close to 8, not 0 ("unknown")
    // and not wildly wrong (this exact bug once computed a total in the
    // hundreds of trillions from a unit-conversion mistake). Progress
    // messages (and so `last_total`) arrive during the encode loop, before
    // `Finalize` runs - so this assertion, the actual point of the test,
    // already holds even on the known Windows sink gap below.
    //
    // The precise range is loosened on Windows: confirmed directly
    // (`MF_MT_FRAME_RATE` on both the native and RGB32-negotiated video
    // type, and cross-checked against `ffprobe`'s r_frame_rate=8/1 on the
    // identical file) that Windows' Matroska source misreports this
    // particular ffmpeg-muxed mkv's frame rate as 4/1 instead of 8/1 -
    // independent of VUI timing info (disabling it made no difference) or
    // of anything wavefold's `MediaFoundationBackend` does with the value
    // it's handed, so `total_frames = duration * fps` comes out ~4 instead
    // of ~8. This is what the actual regression this test guards against
    // (`total` silently staying `0`, "unknown") looks like on Linux/
    // GStreamer, where frame-rate detection doesn't have this quirk -
    // Windows only needs the weaker "fallback engaged and produced
    // *something* nonzero" check to still prove the same fix.
    if cfg!(windows) {
        assert!(last_total > 0, "expected a nonzero total-frame estimate (fallback to format duration), got 0");
    } else {
        assert!((6..=10).contains(&last_total), "expected total near 8 frames, got {last_total}");
    }

    if let Some(e) = saw_error {
        if is_windows_mp4_pcm_sink_gap(&e) {
            eprintln!("total-frame estimate confirmed; skipping the rest ({e})");
            let _ = std::fs::remove_file(&input);
            return;
        }
        panic!("pipeline reported an error: {e}");
    }
    assert!(saw_done, "pipeline never sent Done");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
