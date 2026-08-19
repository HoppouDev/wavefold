# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`dctenc` — a single native Rust binary (desktop GUI via iced, or headless via
a `clap` subcommand) that applies a whole-frame DCT "distortion" effect to
video: decode → per-frame DCT compress/reconstruct on a user-selectable
compute backend (GPU via wgpu, or a pure-CPU fallback) → re-encode with a
user-selectable encoder — software (libx264/libx265/libvpx-vp9/libaom-av1)
or VAAPI hardware — with audio passed through untouched. This is a visual
effect tool, not a real codec — the point is the ringing/ghosting artifact
the DCT cutoff produces, not compression efficiency.

## Commands

```bash
cargo build --release                             # release build: single dctenc binary
dctenc                                            # launch the iced GUI (default with no subcommand)
dctenc gui                                        # same, explicit
dctenc encode <in> <out> [--cutoff F] [--encoder ...] [--backend gpu|cpu]  # headless, no display server needed
cargo check                                       # fast typecheck, iterate with this first
cargo test --release                              # unit tests (gpu.rs, cpu.rs, dct_math.rs, pipeline.rs) + tests/integration.rs
cargo test --release <name>                       # run a single test by substring, e.g. `cargo test --release roundtrip`
```

GPU tests (in `src/gpu.rs`, plus the cross-check test in `src/cpu.rs`)
require a real wgpu adapter and skip gracefully (printing to stderr, not
failing) if `DctGpu::new()` can't find one — expect skips in a
headless/adapterless sandbox. CPU-backend tests (`src/cpu.rs`) have no such
guard and always run — this is deliberate, since it's the proof the CPU
backend needs no GPU at all. `tests/integration.rs` shells out to the
`ffmpeg` CLI (not `ffmpeg-next`) purely to generate synthetic test-fixture
clips; it also skips gracefully if `ffmpeg` isn't on `PATH`.

`.github/workflows/ci.yml` runs `dctenc encode --backend cpu` inside an
Arch Linux container on a standard (GPU-less) `ubuntu-latest` runner — see
the workflow file's comments for why Arch specifically (this repo's pinned
`ffmpeg-next` major version needs to match the installed FFmpeg's, and
Ubuntu's packaged FFmpeg is far behind what's pinned here).

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
not at compile time. The `*Vaapi` variants additionally need a VAAPI-capable
GPU/driver (`libva`, a `/dev/dri/renderD*` node) at *runtime* — there's no
build-time dependency on VAAPI since `pipeline.rs` drives it through raw FFI
against symbols FFmpeg's headers already provide, not a separate crate;
`av_hwdevice_ctx_create` failing (no compatible device) surfaces as a normal
pipeline error, not a build failure.

## Architecture

Pipeline, one direction: `main.rs` (clap entry point, dispatches to the iced
GUI or a headless encode) → `ui/` (setup/encoding pages, GUI path only) →
`pipeline.rs` (decode/encode orchestration via ffmpeg-next, using
`encoders.rs` for the chosen output codec) → `dct_backend.rs`'s
`ComputeBackend` (GPU or CPU) → `gpu.rs` (wgpu compute) / `cpu.rs` (plain
Rust + rayon), both driven by the same basis math in `dct_math.rs` → (GPU
only) `shader.wgsl`.

- **`src/lib.rs`** re-exports `pub mod cpu; pub mod dct_backend; mod
  dct_math; pub mod encoders; pub mod gpu; pub mod pipeline;` so both the
  `dctenc` binary (`main.rs`/`ui/`) and `tests/integration.rs` link against
  the same public API — this split exists specifically so the pipeline can
  be exercised without a GUI. `dct_math` is deliberately *not* `pub` — it's
  `pub(crate)` plumbing shared only between `gpu.rs` and `cpu.rs`, not part
  of the crate's public surface. `ui/` and `main.rs` are binary-only (not
  part of the library) since they're not exercised by `tests/integration.rs`.

- **`src/main.rs`** — single binary, one `clap::Parser` (`Cli { command:
  Option<Command> }`) with two subcommands: `Gui` (also the default when no
  subcommand is given, via `.unwrap_or(Command::Gui)` — preserves the old
  "just run the binary" muscle memory) and `Encode { input, output, cutoff,
  encoder, backend }`. `run_gui()` is the same
  `iced::application(ui::App::default, ui::App::update, ui::App::view).run()`
  bootstrap the binary always had; `run_encode(...)` is the former
  `dctenc-cli` binary's body verbatim — spawns `pipeline::run` on a
  `std::thread` exactly like `ui/encoding.rs` does (it's a blocking call,
  not async), then drives the `tokio::sync::mpsc` receiver on the main
  thread with `rx.blocking_recv()` (works fine with no tokio runtime
  present, which is exactly what `blocking_recv` is for), exiting `1` on
  `PipelineMsg::Error`. Merging both entry points into one binary means the
  `encode` subcommand now always links `iced`/`wgpu` too (they were
  previously a separate `dctenc-cli` binary that skipped that dependency
  weight) — harmless, since `run_gui()` is simply never called on that
  path; no window/GPU surface is touched unless the `Gui` branch runs. This
  is the binary `.github/workflows/ci.yml` runs as `dctenc encode --backend
  cpu` on a GPU-less runner.

- **`src/dct_backend.rs`** — `DctBackend` (the trait `gpu.rs`'s `DctGpu` and
  `cpu.rs`'s `DctCpu` both implement: one `process_rgb(r, g, b, width,
  height, cutoff)` method) and `ComputeBackend` (the user-facing `Gpu`/`Cpu`
  choice — same enum+`ALL`+`Display`+resolver shape as `EncoderChoice` in
  `encoders.rs`, and for the same reason: one switchable choice the GUI
  pick_list and the CLI's `--backend` flag both need). `ComputeBackend::
  build()` is the single place that turns the choice into a live
  `Box<dyn DctBackend>` (`DctGpu::new()`, which can fail with no compatible
  adapter, vs. `DctCpu::new()`, which can't fail). Also derives
  `clap::ValueEnum` (as does `EncoderChoice`) so the CLI's `--backend`/
  `--encoder` flags share one source of truth with the GUI's pick_lists
  instead of a separate hand-maintained CLI-side enum.

- **`src/dct_math.rs`** — `dct_basis`/`transpose_square`, the pure-CPU
  matrix-generation math shared verbatim by both `gpu.rs` (uploads the
  result to GPU buffers) and `cpu.rs` (uses it directly as the transform
  matrices) — pulled out specifically so the two compute backends can never
  silently drift onto different basis matrices.

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
    `Display` in `encoders.rs` specifically for this), a second `pick_list`
    for `ComputeBackend` (same `Display`-via-`label()` pattern, in
    `dct_backend.rs`), and the Encode button (`on_press_maybe`, only enabled
    once both paths are set).
  - **`ui/encoding.rs`** — `State::start(input, output, cutoff, encoder,
    backend)` spawns `pipeline::run` on a plain `std::thread` (it's a
    blocking call,
    not async — spawning it as a tokio task would tie up an executor
    thread for the whole encode) and returns a `Task` that streams its
    `tokio::sync::mpsc` progress channel back via `Task::run(
    UnboundedReceiverStream::new(rx), Message::Pipeline).chain(Task::done(
    Message::WorkerDone))` — progress arrives reactively as messages
    instead of the old egui-era pattern of polling a channel every frame.
    Shows the progress bar, a scrollable log, and a "New encode" button
    (enabled once the worker reports done) that returns `Action::
    BackToSetup`.

- **`src/encoders.rs`** — `EncoderChoice` (8 variants: H264/H265/Vp9/Av1,
  each with a `*Vaapi` hardware counterpart) and its `profile()` →
  `EncoderProfile { codec_name, sw_pixel_format, options, hardware }`:
  the libav encoder name for `ff::encoder::find_by_name` (not a codec-ID
  lookup — several of these codecs have more than one libav encoder), the
  pixel format frames are scaled into before encoding, that encoder's own
  dictionary options, and (for the `*Vaapi` variants) `Some(HwAccel {
  device_type, encoder_pixel_format })`. Different encoders take different
  option keys (`preset` for x264/x265; `deadline`+`cpu-used` for
  libvpx-vp9; `cpu-used` for libaom-av1; no options at all for the VAAPI
  variants) — this isn't a codec-ID swap. `bf=0` (the mp4 edit-list
  workaround, see `pipeline.rs` below) is applied uniformly in
  `pipeline.rs` regardless of which encoder is chosen, rather than
  special-cased per encoder; `avcodec_open2` silently ignores dictionary
  keys an encoder doesn't recognize, so this is harmless for encoders
  without B-frames. `sw_pixel_format` vs `HwAccel::encoder_pixel_format`
  are deliberately separate fields: for a hardware encoder, frames are
  scaled to a real software format (`Pixel::NV12`) and then *uploaded* to
  a hw-accel pixel format (`Pixel::VAAPI`) — see `pipeline.rs`.

- **`src/pipeline.rs`** — all ffmpeg-next work, split across **three**
  threads (decode / GPU DCT / encode+mux) so each stage's work overlaps
  the others instead of serializing on one thread. Audio is a pure stream
  copy (no decode/re-encode), remuxed by rescaling packet timestamps into
  the output's time base. A few non-obvious things worth knowing before
  editing this file:
  - **Three-stage pipeline**: a decode thread demuxes+decodes, sending
    `WorkItem::{Video(VideoFrame), Audio(Packet), Eof, Error(String)}`
    down a bounded `std::sync::mpsc::sync_channel(2)`. The GPU stage
    (scale→GPU DCT→reassemble→scale, PTS tagging, progress reporting)
    stays on `run_inner`'s own calling thread — **not** a spawned one —
    and forwards `EncodeItem::{Video(VideoFrame), Audio(Packet), Eof,
    Error(String)}` down a second channel to a *spawned* encode+mux
    thread, which owns `send_frame`/mux-write and the final
    `send_eof`/`write_trailer`, returning the final frame count through
    its `JoinHandle`. **Which stage is "the calling thread" vs "spawned"
    is dictated by `Send`, not by pipeline order**: `libswscale`'s
    `Scaler` (`software::scaling::context::Context`) is not `Send` (a raw
    `SwsContext` pointer, no `unsafe impl Send`) and so can *never* move
    into a newly spawned thread — that's why the GPU stage (the one
    holding the two `Scaler`s) is the thread that already exists rather
    than one that gets spawned, while `encoder`/`octx` (both confirmed
    `Send` via ffmpeg-next's `unsafe impl Send for Context`/`for Output`)
    are what moves into the new encode-stage thread instead. Don't
    "simplify" this by trying to spawn a GPU-stage thread and keep
    encode+mux on the caller — that's exactly the arrangement `Send`
    rules out. This three-way split is what fixed "100% CPU, 65% GPU"
    (an earlier two-thread version already overlapped decode with
    GPU+encode; this went further and overlaps GPU-DCT-of-frame-N+1 with
    encode-of-frame-N too).
  - **VAAPI hardware encoding** (`EncoderChoice::*Vaapi`): `ffmpeg-next`
    has no safe wrapper for hwdevice/hwframe APIs at all, so
    `setup_hw_frames_context`/`encode_hw_frame` in `pipeline.rs` drive
    them directly through `ff::sys` (raw FFI) — `av_hwdevice_ctx_create`
    (device=`NULL`, so libva auto-picks the render node) →
    `av_hwframe_ctx_alloc` → set `format`/`sw_format`/`width`/`height`/
    `initial_pool_size` on the `AVHWFramesContext` → `av_hwframe_ctx_init`
    → attach to `AVCodecContext.hw_frames_ctx` **before** opening the
    encoder (`avcodec_open2` reads it during init). Per frame:
    `av_hwframe_get_buffer` + `av_hwframe_transfer_data` upload the
    GPU-stage's software (NV12) frame into a hw frame before
    `send_frame`. The owning `HwFramesContext` (a thin RAII wrapper
    around the `*mut AVBufferRef`, `av_buffer_unref`'d on `Drop`) is
    created during setup on the GPU-stage thread but *used* on the
    encode-stage thread (per-frame `av_hwframe_get_buffer` calls need it
    alive there) — moving a raw pointer across that thread boundary needs
    its own `unsafe impl Send`, justified the same way ffmpeg-next
    justifies its own: the pointer is a plain refcounted heap handle with
    no thread affinity, and ownership fully transfers (never aliased
    across threads). Hardware encoder support is inherently
    environment/driver-dependent — e.g. on this dev system's AMD/Mesa
    VAAPI driver, H.264/HEVC/AV1 hw encode all work but VP9 hw encode
    fails with "no usable encoding entrypoint" (the driver just doesn't
    expose that capability) — `av_hwdevice_ctx_create` failure and
    encoder-open failure both surface as ordinary `anyhow` errors through
    the normal `PipelineMsg::Error` path, never a panic.
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
    still a plain synchronous call, so none of the three pipeline threads
    needed to become async). This is a *different* channel from the
    internal `WorkItem`/`EncodeItem` ones above — `PipelineMsg` is the
    outward-facing progress/log feed the UI subscribes to.
  - `PipelineMsg::Log` (→ the UI's in-app log panel) and `tracing::{info,
    debug,warn,error}!` (→ stderr/whatever subscriber `main.rs` installs)
    fire at the same call sites deliberately — they're different audiences
    (end-user status vs. developer diagnostics), not a redundancy to clean
    up. GPU-test skip messages in `gpu.rs` stay on `eprintln!` rather than
    `tracing` on purpose: test binaries never call `tracing_subscriber::
    fmt::init()`, so a `tracing` call there would silently vanish instead
    of printing.

- **`src/pipeline.rs`**'s `run`/`run_inner` take a `backend: ComputeBackend`
  parameter (alongside `encoder_choice`); `run_inner` resolves it once via
  `backend.build()?` into a `Box<dyn DctBackend>` before the pipeline
  starts, and the GPU-stage loop calls `.process_rgb(...)` on that trait
  object — the loop body doesn't know or care which concrete backend it's
  driving.

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

- **`src/cpu.rs`** — `DctCpu`: a direct transcription of `shader.wgsl`'s
  `row_pass`/`col_pass` into plain Rust, run on the CPU instead of the GPU —
  same four passes (forward row → forward col + cutoff mask → inverse col →
  inverse row + clamp), same basis matrices (from `dct_math.rs`, so it can't
  silently drift from the GPU path), same mask formula (`x/(w-1) + y/(h-1)
  <= threshold`). Exists purely so the effect can run with no GPU/wgpu
  adapter present — e.g. `.github/workflows/ci.yml`'s runner. Each pass
  parallelizes over independent output rows with `rayon::par_chunks_mut`
  (same technique as `pipeline.rs`'s `split_rgb_planes`/`join_rgb_planes`),
  so it's GPU-less but not single-threaded. Caches the basis matrices in a
  `RefCell<Option<Basis>>` keyed on `(width, height)`, mirroring `DctGpu`'s
  `PlaneBuffers` cache shape minus the GPU-specific bind-group machinery.
  `src/cpu.rs`'s test module includes `cpu_and_gpu_backends_agree`
  (GPU-guarded like `gpu.rs`'s tests) as a correctness cross-check between
  the two implementations — everything else in that module runs
  unconditionally, with no adapter guard, since that's the whole point.

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
