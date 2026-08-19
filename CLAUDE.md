# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`dctenc` — a native Rust/egui desktop app that applies a whole-frame GPU DCT
"distortion" effect to video: decode → per-frame DCT compress/reconstruct on
the GPU → re-encode with libx264, with audio passed through untouched. This
is a visual effect tool, not a real codec — the point is the ringing/ghosting
artifact the DCT cutoff produces, not compression efficiency.

## Commands

```bash
cargo build --release        # release build (dev build works too, just slower at runtime)
cargo run --release          # launch the egui GUI
cargo check                  # fast typecheck, iterate with this first
cargo test --release         # unit tests (gpu.rs, pipeline.rs) + tests/integration.rs
cargo test --release <name>  # run a single test by substring, e.g. `cargo test --release roundtrip`
```

GPU tests (in `src/gpu.rs`) require a real wgpu adapter and skip gracefully
(printing to stderr, not failing) if `DctGpu::new()` can't find one — expect
skips in a headless/adapterless sandbox. `tests/integration.rs` shells out to
the `ffmpeg` CLI (not `ffmpeg-next`) purely to generate synthetic test-fixture
clips; it also skips gracefully if `ffmpeg` isn't on `PATH`.

No lint/format config is checked in; `cargo fmt` / `cargo clippy` apply with
their defaults if needed.

### System dependencies

`ffmpeg-next`/`ffmpeg-sys-next` bind against the system's libavcodec/
libavformat/etc. via `pkg-config` and generate bindings with `bindgen`
against the installed headers — **the crate's major version must match the
installed FFmpeg's major version** (this repo pins `ffmpeg-next = "9"` to
match FFmpeg 9 headers; an older/newer system FFmpeg will need the
dependency version bumped to match, or the build fails during bindgen, e.g.
on a missing/renamed header). `pkg-config`, FFmpeg dev headers, and libx264
must be installed and discoverable for a from-scratch build.

## Architecture

Pipeline, one direction: `main.rs` (GUI) → `pipeline.rs` (decode/encode
orchestration via ffmpeg-next) → `gpu.rs` (wgpu compute) → `shader.wgsl`.

- **`src/lib.rs`** just re-exports `pub mod gpu; pub mod pipeline;` so the
  binary (`main.rs`) and `tests/integration.rs` both link against the same
  public API — this split exists specifically so the pipeline can be
  integration-tested without a GUI.

- **`src/main.rs`** — egui/eframe app. Owns a background `std::thread`
  per encode run; the thread calls `pipeline::run` and streams
  `PipelineMsg::{Progress,Log,Done,Error}` back over an `mpsc` channel that
  `App::poll()` drains each frame. `TryRecvError::Disconnected` is treated
  as "the worker thread died without reporting" and surfaces an error rather
  than leaving the UI stuck in `running` state.

- **`src/pipeline.rs`** — all ffmpeg-next work: demux, decode the video
  stream, scale to `RGB24` (via `Scaler`), split into R/G/B planes, run each
  plane through the GPU DCT, reassemble, scale to `YUV420P`, encode with
  libx264 (`bf=0`, i.e. no B-frames — simpler timestamp handling since frames
  are processed independently anyway), and mux. Audio is a pure stream copy
  (no decode/re-encode), remuxed by rescaling packet timestamps into the
  output's time base. A few non-obvious things worth knowing before editing
  this file:
  - The mov/mp4 muxer's auto edit-list logic silently drops the last encoded
    video frame under some conditions; `write_header_with` passes
    `use_editlist=0` for mp4-family outputs to avoid that. That's an
    empirically-discovered fix, not something libav documents cleanly.
  - `decoder.receive_frame` / `encoder.receive_packet` loops must
    distinguish "no more output yet" (`Error::Eof` / `Error::Other{EAGAIN}`)
    from a real decode/encode error — treating any `Err` as "done" silently
    drops frames on genuine failures.
  - Video PTS comes from the decoded frame's own timestamp (rescaled into
    the encoder's time base), not a synthetic zero-based counter, so it
    stays aligned with the audio passthrough's original timeline.

- **`src/gpu.rs`** — `DctGpu`: the whole-frame (not block-based) separable
  2D DCT. Per plane, 4 GPU dispatches ping-pong two buffers: forward row →
  forward col (+ quality mask) → inverse col → inverse row (+ clamp to pixel
  range). Forward and inverse use the *same* orthonormal basis matrix but
  need it indexed transposed relative to each other — `dct_basis`/
  `transpose_square` precompute both `B` and `B^T` once on the CPU per
  (width, height), so the shader itself never branches on direction.
  `quality` (1..=100, a percent) maps to a normalized diagonal
  frequency cutoff in `[0, 2]`; a coefficient at `(u, v)` survives if
  `u/(width-1) + v/(height-1) <= threshold`.
  - `DctGpu` holds **3 independent buffer slots** (`buffers: [RefCell<...>; 3]`,
    one per RGB channel) so `process_rgb` can submit all three channels'
    GPU work before blocking on any of them — do not merge them back into a
    single shared buffer set, that would make one channel's `write_buffer`
    race with another's still-in-flight dispatch.
  - Per-slot buffers are cached keyed on `(width, height, quality)`, but a
    **quality-only** change (dimensions unchanged) rewrites just the
    `forward_mask` bind group's threshold uniform in place instead of
    rebuilding the O(width²+height²) basis matrices — that's what
    `mask_params_buf` and the `b.quality != quality` branch in
    `encode_plane` are for.
  - `process_plane` (single-channel) is kept for the unit tests /
    simple callers; `pipeline.rs` uses `process_rgb` for the actual encode
    path.
  - This is a **naive O(N) per-axis** transform (not a fast/FFT-based DCT),
    so cost scales roughly with `width·height·(width+height)` per plane per
    frame — `pipeline.rs` logs a warning above 640×480 because this gets
    slow fast at real video resolutions. Don't "simplify" this without
    accounting for that cost.

- **`src/shader.wgsl`** — the two WGSL compute entry points (`row_pass`,
  `col_pass`) that back the passes above. No shared/workgroup memory is
  used (each thread does its full O(width) or O(height) sum independently),
  which is what makes the whole-frame approach possible at all — an actual
  8×8-block-style design with workgroup-shared memory can't scale to
  arbitrary frame dimensions (workgroup size / shared memory limits), which
  is why this isn't block-based like a real video codec.

## Communication style (AGENTS.md / Cursor / Copilot / Windsurf / Cline rules)

This repo has generated per-tool "caveman mode" rule files (`AGENTS.md`,
`.cursor/rules/`, `.windsurf/rules/`, `.clinerules/`, `.github/copilot-instructions.md`)
via `caveman-init`: terse, fragment-heavy responses in chat, no filler/
pleasantries. The boundary that matters most: **code, commit messages, and
PR descriptions are still written in normal, full prose** — the terse style
applies to conversational responses only, not to anything persisted into the
repo or git history.
