# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`dctenc` — a native Rust/iced desktop app that applies a whole-frame GPU DCT
"distortion" effect to video: decode → per-frame DCT compress/reconstruct on
the GPU → re-encode with a user-selectable encoder (libx264/libx265/
libvpx-vp9/libaom-av1), with audio passed through untouched. This is a
visual effect tool, not a real codec — the point is the ringing/ghosting
artifact the DCT cutoff produces, not compression efficiency.

## Commands

```bash
cargo build --release        # release build (dev build works too, just slower at runtime)
cargo run --release          # launch the iced GUI
cargo check                  # fast typecheck, iterate with this first
cargo test --release         # unit tests (gpu.rs, pipeline.rs) + tests/integration.rs
cargo test --release <name>  # run a single test by substring, e.g. `cargo test --release roundtrip`
```

GPU tests (in `src/gpu.rs`) require a real wgpu adapter and skip gracefully
(printing to stderr, not failing) if `DctGpu::new()` can't find one — expect
skips in a headless/adapterless sandbox. `tests/integration.rs` shells out to
the `ffmpeg` CLI (not `ffmpeg-next`) purely to generate synthetic test-fixture
clips; it also skips gracefully if `ffmpeg` isn't on `PATH`.

`RUST_LOG=debug cargo run --release` (or any `tracing`-compatible env filter)
surfaces the `tracing` diagnostics described below — the app has no other
logging config checked in.

No lint/format config is checked in; `cargo fmt` / `cargo clippy` apply with
their defaults if needed.

### System dependencies

`ffmpeg-next`/`ffmpeg-sys-next` bind against the system's libavcodec/
libavformat/etc. via `pkg-config` and generate bindings with `bindgen`
against the installed headers — **the crate's major version must match the
installed FFmpeg's major version** (this repo pins `ffmpeg-next = "9"` to
match FFmpeg 9 headers; an older/newer system FFmpeg will need the
dependency version bumped to match, or the build fails during bindgen, e.g.
on a missing/renamed header). `pkg-config`, FFmpeg dev headers, and the
encoder libraries actually available on the system (libx264/libx265/
libvpx/libaom — see `src/encoders.rs`) must be installed and discoverable
for a from-scratch build; an `EncoderChoice` whose libav encoder isn't
present in the local FFmpeg build fails at encode time with a clear error,
not at compile time.

## Architecture

Pipeline, one direction: `main.rs` (iced bootstrap) → `ui/` (setup/encoding
pages) → `pipeline.rs` (decode/encode orchestration via ffmpeg-next, using
`encoders.rs` for the chosen output codec) → `gpu.rs` (wgpu compute) →
`shader.wgsl`.

- **`src/lib.rs`** re-exports `pub mod encoders; pub mod gpu; pub mod
  pipeline;` so the binary (`main.rs`/`ui/`) and `tests/integration.rs` both
  link against the same public API — this split exists specifically so the
  pipeline can be integration-tested without a GUI. `ui/` and `main.rs` are
  binary-only (not part of the library) since they're not exercised by
  `tests/integration.rs`.

- **`src/main.rs`** — a 9-line bootstrap: `mod ui;` plus
  `iced::application(ui::App::default, ui::App::update, ui::App::view)`.
  All actual UI logic lives in `src/ui/`.

- **`src/ui/`** — iced 0.14 app (`features = ["tokio"]`, so iced's executor
  is a real tokio runtime), split into two pages following iced's documented
  multi-screen pattern:
  - **`ui/mod.rs`** — `App` wraps `enum Screen { Setup(setup::State),
    Encoding(encoding::State) }` and a wrapper `Message` enum; `App::update`
    dispatches to whichever screen is active and applies the `Action` each
    screen's own `update` returns (`setup::Action::Start{..}` swaps
    `Screen` to `Encoding` and kicks off the encode; `encoding::
    Action::BackToSetup` swaps back to a fresh `Setup`). A screen never
    mutates the other screen's state or the top-level `Screen` directly —
    only `App::update` does, based on the `Action` it gets back. Stray
    messages for a screen you've since navigated away from (e.g. a
    file-dialog result resolving after leaving Setup) are silently dropped
    in the dispatch match.
  - **`ui/setup.rs`** — input/output file pickers (`rfd::AsyncFileDialog`
    via `Task::perform`, not the old synchronous `rfd::FileDialog`), the
    cutoff slider, the encoder `pick_list` (`EncoderChoice` implements
    `Display` in `encoders.rs` specifically for this), and the Encode
    button (`on_press_maybe`, only enabled once both paths are set).
  - **`ui/encoding.rs`** — `State::start(input, output, cutoff, encoder)`
    spawns `pipeline::run` on a plain `std::thread` (it's a blocking call,
    not async — spawning it as a tokio task would tie up an executor
    thread for the whole encode) and returns a `Task` that streams its
    `tokio::sync::mpsc` progress channel back via `Task::run(
    UnboundedReceiverStream::new(rx), Message::Pipeline).chain(Task::done(
    Message::WorkerDone))` — progress arrives reactively as messages
    instead of the old egui-era pattern of polling a channel every frame.
    Shows the progress bar, a scrollable log, and a "New encode" button
    (enabled once the worker reports done) that returns `Action::
    BackToSetup`.

- **`src/encoders.rs`** — `EncoderChoice` (H264/H265/Vp9/Av1) and its
  `profile()` → `EncoderProfile { codec_name, pixel_format, options }`:
  the libav encoder name for `ff::encoder::find_by_name` (not a codec-ID
  lookup — several of these codecs have more than one libav encoder),
  destination pixel format, and that encoder's own dictionary options.
  Different encoders take different option keys (`preset` for x264/x265;
  `deadline`+`cpu-used` for libvpx-vp9; `cpu-used` for libaom-av1) — this
  isn't a codec-ID swap. `bf=0` (the mp4 edit-list workaround, see
  `pipeline.rs` below) is applied uniformly in `pipeline.rs` regardless of
  which encoder is chosen, rather than special-cased per encoder;
  `avcodec_open2` silently ignores dictionary keys an encoder doesn't
  recognize, so this is harmless for encoders without B-frames.

- **`src/pipeline.rs`** — all ffmpeg-next work, split across two threads to
  overlap CPU and GPU work instead of serializing everything (see below):
  a producer thread demuxes and decodes; the caller's own thread (the
  "consumer") scales to RGB24, runs the GPU DCT, reassembles, scales to the
  target pixel format, encodes with the selected `EncoderChoice`, and muxes.
  Audio is a pure stream copy (no decode/re-encode), remuxed by rescaling
  packet timestamps into the output's time base. A few non-obvious things
  worth knowing before editing this file:
  - **Producer/consumer thread split**: `run_inner` spawns a producer
    `std::thread` that owns the input format context and decoder, sending
    a `WorkItem::{Video(VideoFrame), Audio(Packet), Eof, Error(String)}`
    per unit of work down a bounded `std::sync::mpsc::sync_channel(2)` to
    the consumer (this function's own calling thread). This means decode of
    frame N+1 overlaps with the consumer's GPU-wait + encode of frame N,
    instead of the two running one after another on a single thread (which
    is what previously produced "100% CPU, 65% GPU" — the GPU sat idle
    during every scale/encode step and vice versa). **`libswscale`'s
    `Scaler` (`software::scaling::context::Context`) is not `Send`** (a raw
    `SwsContext` pointer with no `unsafe impl Send`), so scaling stays on
    the consumer side — only `ff::format::context::Input`, `decoder::Video`,
    `ff::Packet`, and `ff::frame::Video` cross the channel, all of which
    ffmpeg-next explicitly marks `Send`. Don't try to move a `Scaler`
    across threads without re-checking that.
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
  - `split_rgb_planes`/`join_rgb_planes` parallelize their per-row loops
    with `rayon` (`par_chunks_mut`) — each row is a disjoint read/write, no
    aliasing between them.
  - `PipelineMsg` is the channel payload (`Progress`/`Log`/`Done`/`Error`),
    sent over a `tokio::sync::mpsc::UnboundedSender` (aliased as `Sender`
    via `use tokio::sync::mpsc::UnboundedSender as Sender` — `send()` is
    still a plain synchronous call, so the producer/consumer thread code
    didn't need to become async). This is a *different* channel from the
    internal producer→consumer `WorkItem` one above — `PipelineMsg` is the
    outward-facing progress/log feed the UI subscribes to.
  - `PipelineMsg::Log` (→ the UI's in-app log panel) and `tracing::{info,
    debug,warn,error}!` (→ stderr/whatever subscriber `main.rs` installs)
    fire at the same call sites deliberately — they're different audiences
    (end-user status vs. developer diagnostics), not a redundancy to clean
    up. GPU-test skip messages in `gpu.rs` stay on `eprintln!` rather than
    `tracing` on purpose: test binaries never call `tracing_subscriber::
    fmt::init()`, so a `tracing` call there would silently vanish instead
    of printing.

- **`src/gpu.rs`** — `DctGpu`: the whole-frame (not block-based) separable
  2D DCT. Per plane, 4 GPU dispatches ping-pong two buffers: forward row →
  forward col (+ cutoff mask) → inverse col → inverse row (+ clamp to pixel
  range). Forward and inverse use the *same* orthonormal basis matrix but
  need it indexed transposed relative to each other — `dct_basis`/
  `transpose_square` precompute both `B` and `B^T` once on the CPU per
  (width, height), so the shader itself never branches on direction.
  `cutoff` is a normalized diagonal frequency threshold in `0.0..=2.0`
  (passed straight through to the shader's `threshold` uniform — no
  percent-to-threshold rescaling layer); a coefficient at `(u, v)` survives
  if `u/(width-1) + v/(height-1) <= threshold`.
  - `DctGpu` holds **3 independent buffer slots** (`buffers: [RefCell<...>; 3]`,
    one per RGB channel) so `process_rgb` can submit all three channels'
    GPU work before blocking on any of them — do not merge them back into a
    single shared buffer set, that would make one channel's `write_buffer`
    race with another's still-in-flight dispatch. The GPU readback follows
    the same submit-all-before-blocking-any principle at a finer grain:
    `read_plane` is split into `begin_read` (issues `map_async` only, no
    blocking) and `finish_read` (blocks on one `device.poll` + copies out),
    so `process_rgb` does `encode_plane`×3 → `begin_read`×3 → one
    `device.poll` → `finish_read`×3, instead of polling once per channel in
    sequence.
  - Per-slot buffers are cached keyed on `(width, height, cutoff)`, but a
    **cutoff-only** change (dimensions unchanged) rewrites just the
    `forward_mask` bind group's threshold uniform in place instead of
    rebuilding the O(width²+height²) basis matrices — that's what
    `mask_params_buf` and the `b.cutoff != cutoff` branch in `encode_plane`
    are for.
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
