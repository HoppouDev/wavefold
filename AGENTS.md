# AGENTS.md

Guide AI coding agents (Claude Code, Cursor, Copilot, etc.) work this repo.

## What this is

`wavefold` — single native Rust binary (desktop GUI via iced, or headless via
`clap` subcommand). Apply whole-frame DCT "distortion" effect to video:
decode, per-frame DCT compress/reconstruct on user-selectable compute
backend (GPU via wgpu, or pure-CPU fallback), re-encode with
user-selectable encoder — software (x264/x265/vp9/av1 GStreamer elements)
or VAAPI hardware — audio pass through untouched. Visual effect tool, not
real codec — point is ringing/ghosting artifact from DCT cutoff, not
compression efficiency.

Media I/O behind `MediaBackend` trait (`media_backend.rs`) — decode/encode
implementation swappable, platform-gated: `backends::gstreamer`
(`cfg(not(windows))`) built on **GStreamer**
(`gstreamer`/`gstreamer-app`/`gstreamer-video`) on Linux/macOS,
`backends::media_foundation` (`cfg(windows)`) built on Windows's own
**Media Foundation** (`windows` crate) on Windows — no system package
manager there to install GStreamer from, but Media Foundation ships with
the OS, so that backend needs no bundled runtime at all. Exactly one of
the two ever compiled into given binary. GStreamer, not FFmpeg — replaced
earlier `ffmpeg-next`-based implementation specifically because
`ffmpeg-sys-next` declares `links = "ffmpeg"`, and Cargo hard-bans two
versions of `links`-crate coexisting one dependency graph — structurally
impossible for one `Cargo.toml` build against both this dev machine's
FFmpeg (Arch, rolling) and stock Ubuntu CI runner's older one same time.
GStreamer's C API/ABI stable since 1.0 (2012), so one `gstreamer` crate
version builds against wide range installed GStreamer versions, no
problem.

## Commands

```bash
cargo build --release                          # release build: single wavefold binary
wavefold                                       # launch the iced GUI (default with no subcommand)
wavefold gui                                   # same, explicit
wavefold encode <in> <out> [--cutoff F] [--encoder ...] [--backend gpu|cpu]  # headless, no display server needed
cargo check                                    # fast typecheck, iterate with this first
cargo test --release                           # unit tests (gpu.rs, cpu.rs, dct_math.rs, backends/gstreamer.rs) + tests/integration.rs
cargo test --release <name>                    # run a single test by substring, e.g. `cargo test --release roundtrip`
```

GPU tests (in `src/gpu.rs`, plus cross-check test `src/cpu.rs`) need real
wgpu adapter, skip gracefully (print stderr, not fail) if `DctGpu::new()`
finds none — expect skips headless/adapterless sandbox. CPU-backend tests
(`src/cpu.rs`) no such guard, always run — deliberate, proof CPU backend
needs no GPU at all. `tests/integration.rs` shells out `ffmpeg` CLI (not
`ffmpeg-next`) purely to generate synthetic test-fixture clips; also skips
gracefully if `ffmpeg` not on `PATH`.

`.github/workflows/ci.yml` runs `wavefold encode --backend cpu` on plain
`ubuntu-latest` runner (no container workaround needed — see "What this
is" section above why GStreamer rewrite specifically made that possible).

`RUST_LOG=debug cargo run --release` (or any `tracing`-compatible env filter)
surfaces `tracing` diagnostics described below — app has no other logging
config checked in.

No lint/format config checked in; `cargo fmt` / `cargo clippy` apply
defaults if needed.

### System dependencies

Linux/macOS only (`backends::gstreamer`) — Windows (`backends::
media_foundation`) needs nothing beyond MSVC toolchain, Media
Foundation ships with OS. `gstreamer-sys`/`gstreamer-app-sys`/
`gstreamer-video-sys` bind against
system's libgstreamer-1.0/libgstapp-1.0/libgstvideo-1.0 via `pkg-config` —
unlike old ffmpeg-next setup, no major-version-must-match constraint (see
"What this is" above), so any reasonably current GStreamer dev install
works. From-scratch build needs: `pkg-config`,
`libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev` (Debian/Ubuntu
naming; `gstreamer`/`gst-plugins-base` on Arch) for headers, **plus actual
plugins at runtime** — `gst-plugins-good` (`vp9enc`), `gst-plugins-bad`
(`x265enc`, `av1enc`, `va` VAAPI plugin), `gst-plugins-ugly` (`x264enc`) —
`Codec` whose element not installed fails at pipeline-construction
time, clear error (`gst::
ElementFactory::make` returning `None`), not
compile time. Also install `gstreamer1.0-libav` (or distro equivalent) for
broad-codec *decoding* — this repo's own dev/CI environment needed it,
default `openh264dec` decoder can't handle every H.264 profile FFmpeg
itself produces. `va`-plugin
VAAPI elements (`vah264enc`/`vah265enc`/`vaav1enc`) additionally need
VAAPI-capable GPU/driver (`libva`, `/dev/dri/renderD*` node) at *runtime*
to even register as available element factories — no such device,
`gst_inspect_1.0 va` still loads plugin but individual codec elements just
don't exist, so `ElementFactory::make("vah264enc")` fails same clean way
as genuinely-uninstalled plugin, not build failure.

## Architecture

Pipeline, one direction: `main.rs` (clap entry point, dispatch to iced
GUI or headless encode) → `ui/` (setup/encoding pages, GUI path only) →
`pipeline.rs` (thin dispatcher: resolves `ComputeBackend` → `Box<dyn
DctBackend>` once, resolves `MediaBackendChoice` → `Box<dyn MediaBackend>`,
calls its `run`) → `backends::gstreamer::GstreamerBackend` (only
`MediaBackend` impl today — `gst::Pipeline` built, driven via `gstreamer`/
`gstreamer-app`/`gstreamer-video`, mapping `Codec` to a GStreamer encoder
element internally) → `dct_backend.rs`'s `ComputeBackend` (GPU or CPU) →
`gpu.rs` (wgpu compute) / `cpu.rs` (plain Rust + rayon), both driven by
same basis math `dct_math.rs` → (GPU only) `shader.wgsl`.

- **`src/lib.rs`** re-exports `pub mod backends; pub mod codec; pub mod
  cpu; pub mod dct_backend; mod dct_math; pub mod gpu; pub mod
  media_backend; pub mod pipeline;` so both `wavefold` binary
  (`main.rs`/`ui/`) and `tests/integration.rs` link against same public
  API — split exists specifically so pipeline exercised without GUI.
  `dct_math` deliberately *not* `pub` — `pub(crate)` plumbing shared only
  between `gpu.rs` and `cpu.rs`, not part of crate's public surface. `ui/` and
  `main.rs` binary-only (not part of library) since not exercised by
  `tests/integration.rs`.

- **`src/main.rs`** — single binary, one `clap::Parser` (`Cli { command:
  Option<Command> }`) with two subcommands: `Gui` (default when none
  given, via `.unwrap_or(Command::Gui)`) and `Encode { input, output,
  cutoff, encoder, backend, media_backend }` (`encoder: Codec`,
  `media_backend: MediaBackendChoice` — CLI flag stays named `--encoder`
  for continuity even though the type is now backend-agnostic `Codec`,
  not a GStreamer-specific enum). `run_gui()` is the same
  `iced::application(...).run()` bootstrap the binary always had;
  `run_encode(...)` is the former `wavefold-cli` binary's body verbatim —
  spawns `pipeline::run` on `std::thread` like `ui/encoding.rs` does, then
  drives `tokio::sync::mpsc` on the main thread with `rx.blocking_recv()`
  (works fine with no tokio runtime present), exits `1` on
  `PipelineMsg::Error`. Merging both entry points means `encode` now
  always links `iced`/`wgpu` too — harmless, since `run_gui()` never
  called unless the `Gui` branch runs.

- **`src/dct_backend.rs`** — `DctBackend` trait (`gpu.rs`'s `DctGpu` and
  `cpu.rs`'s `DctCpu` both implement one `process_rgb(r, g, b, width,
  height, cutoff)` method) and `ComputeBackend` (user-facing `Gpu`/`Cpu`
  choice — same enum+`ALL`+`Display`+resolver shape as `Codec`/
  `MediaBackendChoice`, since GUI pick_list and CLI flag both need it).
  `ComputeBackend::build()` turns the choice into a live
  `Box<dyn DctBackend>` (`DctGpu::new()` can fail with no compatible
  adapter, `DctCpu::new()` can't). Also derives `clap::ValueEnum` so CLI's
  `--backend`/`--encoder`/`--media-backend` flags share one source of
  truth with the GUI's pick_lists.

- **`src/codec.rs`** — `Codec` (8 variants: H264/H265/Vp9/Av1, each with a
  `*Hardware` counterpart), `is_hardware()`, `ALL`/`Display`/
  `clap::ValueEnum` (same shape as `ComputeBackend`). Purely a
  user-facing choice of codec+hw-or-not, agnostic of GStreamer or any
  other backend — `backends::gstreamer` maps a `Codec` onto a concrete
  element. Renamed from old `encoders.rs`'s `EncoderChoice`
  (`H264Vaapi` → `H264Hardware` etc.) since "VAAPI" is
  GStreamer/Linux-specific vocabulary a non-GStreamer backend wouldn't
  share.

- **`src/media_backend.rs`** — `PipelineMsg`, `MediaBackend` trait (one
  `run(input, output, cutoff, codec, dct: Box<dyn DctBackend>, tx)` method
  — decode `input` to RGB frames, run `dct.process_rgb` over each,
  re-encode into `codec`, audio passthrough, report via `tx`), and
  `MediaBackendChoice` (`Gstreamer`/`MediaFoundation`, `cfg`-gated so
  exactly one compiled per target, same `ALL`/`Display`/
  `clap::ValueEnum`/`build()` shape as `ComputeBackend`/`Codec` —
  extension point for a third platform backend without touching
  `pipeline.rs`'s dispatcher).

- **`src/dct_math.rs`** — `dct_basis`/`transpose_square`, pure-CPU
  matrix-generation math shared verbatim by `gpu.rs` (uploads to GPU
  buffers) and `cpu.rs` (uses directly) — pulled out so the two compute
  backends never silently drift on different basis matrices.

- **`src/ui/`** — iced 0.14 app (`features = ["tokio"]`, so iced's
  executor is real tokio runtime), split into two pages following iced's
  documented multi-screen pattern:
  - **`ui/mod.rs`** — `App` wraps `enum Screen { Setup(setup::State),
    Encoding(encoding::State) }` and a wrapper `Message` enum;
    `App::update` dispatches to whichever screen is active, applies the
    `Action` each screen's own `update` returns (`setup::Action::Start{..}`
    swaps `Screen` to `Encoding`, kicks off encode;
    `encoding::Action::BackToSetup` swaps back to fresh `Setup`). Screen
    never mutates other screen's state or the top-level `Screen` directly
    — only `App::update` does. Stray messages for a screen navigated away
    from (e.g. a file-dialog result resolving after leaving Setup) are
    silently dropped in the dispatch match. Under the `automation` feature
    (`ui/automation.rs`), `update` publishes a state snapshot to any
    connected automation client after a message actually dispatched to a
    screen — skipped both when no client is connected
    (`Handle::has_subscribers()`) and for a dropped/stray message, so a
    client can't mistake a silently-ignored injection for a real one.
  - **`ui/setup.rs`** — input/output file pickers (`rfd::AsyncFileDialog`
    via `Task::perform`, not old synchronous `rfd::FileDialog`), cutoff
    slider, encoder `pick_list` (`Codec` implements `Display` in
    `codec.rs` specifically for this), second `pick_list` for
    `ComputeBackend` (same `Display`-via-`label()` pattern, in
    `dct_backend.rs`), Encode button (`on_press_maybe`, only enabled once
    both paths set). No `pick_list` for `MediaBackendChoice` — only one
    variant exists, picker with single disabled option would be dead
    UI; `ui/encoding.rs` passes `MediaBackendChoice::Gstreamer` straight
    through instead.
  - **`ui/encoding.rs`** — `State::start(input, output, cutoff, encoder,
    backend)` spawns `pipeline::run` on plain `std::thread` (blocking, not
    async — a tokio task would tie up the executor thread for the whole
    encode), returns a `Task` streaming the `tokio::sync::mpsc` progress
    channel back via `Task::run(UnboundedReceiverStream::new(rx),
    Message::Pipeline).chain(Task::done(Message::WorkerDone))` — progress
    arrives reactively instead of polling every frame. Shows a progress
    bar, scrollable log, "New encode" button (enabled once done) returning
    `Action::BackToSetup`.

- **`src/pipeline.rs`** — thin dispatcher: `run(input, output, cutoff,
  codec, compute_backend, media_backend, tx)` resolves `compute_backend`
  into a `Box<dyn DctBackend>` once via `.build()?` (shared regardless of
  which media backend runs — DCT compute choice is orthogonal to
  decode/encode choice), logs "initializing ... DCT backend", then calls
  `media_backend.build().run(...)` and turns an `Err` into
  `PipelineMsg::Error`.

- **`src/backends/gstreamer.rs`** — the only `MediaBackend` impl on
  Linux/macOS; builds and drives one `gst::Pipeline` per encode.
  `codec_profile(codec)` maps `Codec` to a GStreamer element factory name +
  its properties (`tune=zerolatency` for x264/x265, `deadline`+
  `lag-in-frames` for vp9, `cpu-used`+`lag-in-frames` for av1, none for
  hardware variants) + optional bitstream parser (`h264parse`/`h265parse`,
  required for H.265 into `qtmux`/`matroskamux`). `GstreamerBackend::run`
  takes an already-resolved `dct: Box<dyn DctBackend>` — decode, GPU DCT
  compute, and encode run on three threads connected by bounded
  `std::sync::mpsc::sync_channel`s (`decode_and_send` in `appsink`'s
  callbacks → `wavefold-gst-compute` thread → `wavefold-gst-encode` thread
  owning `appsrc.push_buffer(...)`) — needed because GStreamer's own
  `queue` elements only overlap decode with compute+encode, not compute
  with encode itself (fixed a real ~9-15%→~60% GPU-utilization regression
  on AV1 hardware encode). `PipelineMsg` (`Progress`/`Log`/`Done`/`Error`,
  defined in `media_backend.rs`) keeps the same outward contract the old
  ffmpeg-next implementation had, so `ui/`/`main.rs`/`tests/integration.rs`
  never had to change across either rewrite.
  Known encoder/muxer gaps tolerated at the test level: no `vavp9enc` in
  GStreamer's `va` plugin (VP9 hardware encode unavailable), `vp9enc`
  can't mux into `qtmux` (missing `chroma-format` cap — use `matroskamux`),
  VAAPI HEVC pads non-64-aligned frames to the CTU boundary without fixing
  up the SPS crop.
  Full pitfall log (pipeline wiring, EOS/preroll edge cases, reference-cycle
  leak, `set_state(Null)`-on-every-exit-path, progress-estimate quirks):
  [docs/gstreamer-notes.md](docs/gstreamer-notes.md).

- **`src/backends/media_foundation.rs`** — only `MediaBackend` impl on
  Windows (`cfg(windows)`), built on `IMFSourceReader`/`IMFSinkWriter`
  instead of `gst::Pipeline`. `codec_target(codec)` maps `Codec` to
  `{ subtype: GUID, hardware: bool }` (`MFVideoFormat_H264`/`_HEVC`/
  `_VP90`/`_AV1`; `hardware` feeds `MF_READWRITE_ENABLE_HARDWARE_
  TRANSFORMS` — MF auto-picks the matching registered encoder, no
  GStreamer-style explicit element-name selection). `MFVideoFormat_RGB32`
  is BGRA, not RGB, and can be a bottom-up DIB (negative stride) —
  `split_bgra_planes`/`join_bgra_planes` handle that. Audio passthrough
  (`setup_audio_passthrough`) adds the input's still-encoded audio type
  straight to the sink writer, same idea as `backends::gstreamer`'s
  `autoplug-continue` stream copy.
  Platform gaps tolerated at the test level (all confirmed on real
  Windows 11, not wavefold bugs): H.264 encoder MFT rejects inputs below
  ~32-48px on either axis; no software AV1 encoder MFT ships at all
  (`Codec::Av1Hardware` unaffected); the built-in MP4 sink accepts PCM
  audio negotiation but fails at `Finalize` (AAC-only in practice); MF
  ships no Matroska byte-stream handler, so `.mkv` output just fails.
  Full pitfall log (COM/MF lifecycle RAII guards, stride handling,
  container-support detail): [docs/media-foundation-notes.md](docs/media-foundation-notes.md).

- **`src/gpu.rs`** — `DctGpu`: the whole-frame (not block-based) separable
  2D DCT. Per plane, 4 GPU dispatches ping-pong two buffers: forward row →
  forward col (+ cutoff mask) → inverse col → inverse row (+ clamp to pixel
  range). Forward and inverse use *same* orthonormal basis matrix but
  need it indexed transposed relative to each other — `dct_basis`/
  `transpose_square` precompute both `B` and `B^T` once on CPU per
  (width, height), so shader itself never branches on direction.
  `cutoff` is a normalized diagonal frequency threshold in `0.0..=2.0`
  (passed straight through to shader's `threshold` uniform — no
  percent-to-threshold rescaling layer); a coefficient at `(u, v)` survives
  if `u/(width-1) + v/(height-1) <= threshold`.
  - `DctGpu` holds **3 independent buffer slots** (`buffers: [RefCell<...>; 3]`,
    one per RGB channel) so `process_rgb` can submit all three channels'
    GPU work before blocking on any of them — do not merge them back into a
    single shared buffer set, that would make one channel's `write_buffer`
    race with another's still-in-flight dispatch. GPU readback follows
    same submit-all-before-blocking-any principle at a finer grain:
    `read_plane` split into `begin_read` (issues `map_async` only, no
    blocking) and `finish_read` (blocks on one `device.poll` + copies out),
    so `process_rgb` does `encode_plane`×3 → `begin_read`×3 → one
    `device.poll` → `finish_read`×3, instead of polling once per channel in
    sequence.
  - Per-slot buffers cached keyed on `(width, height, cutoff)`, but a
    **cutoff-only** change (dimensions unchanged) rewrites just
    `forward_mask` bind group's threshold uniform in place instead of
    rebuilding O(width²+height²) basis matrices — that's what
    `mask_params_buf` and `b.cutoff != cutoff` branch in `encode_plane`
    are for.
  - `process_plane` (single-channel) kept for unit tests /
    simple callers; `backends/gstreamer.rs` uses `process_rgb` for
    actual encode path.
  - This is an **O(N) per-axis** transform (not a fast/FFT-based DCT), so
    FLOP count scales roughly with `width·height·(width+height)` per plane
    per frame regardless of tiling below — `backends/gstreamer.rs` logs
    a warning above 640×480 because this gets slow fast at real video
    resolutions. Don't "simplify" this without accounting for that cost.
  - `encode_plane`'s dispatch groups sized off a `TILE` constant
    (currently 16, mirroring `shader.wgsl`'s own `TILE`) rather than a bare
    workgroup size — two must stay in lockstep, since shader now
    tiles its memory access pattern through `workgroup` storage (see
    `shader.wgsl` below); a mismatch wouldn't fail to compile, just produce
    wrong output.
  - `DctGpu::poll_bounded` bounds `device.poll` to `GPU_POLL_TIMEOUT` (30s)
    instead of `wgpu::PollType::wait_indefinitely()`: a too-slow dispatch
    pushing past the driver's TDR window can get the GPU reset out from
    under the process mid-wait, leaving `wait_indefinitely()` blocked
    forever on a fence that'll never signal (reproduced at 1920×1082).
    Bounded wait turns that into a reported error instead of a silent
    freeze. Tiled shader below dropped a 1920×1082 `process_rgb` call to
    ~340ms, well under both the TDR window and this timeout — the timeout
    stays as a safety net for resolutions large enough to still cross it.

- **`src/cpu.rs`** — `DctCpu`: direct transcription of `shader.wgsl`'s
  `row_pass`/`col_pass` into plain Rust, run on CPU instead of GPU —
  same four passes (forward row → forward col + cutoff mask → inverse col →
  inverse row + clamp), same basis matrices (from `dct_math.rs`, so can't
  silently drift from GPU path), same mask formula (`x/(w-1) + y/(h-1)
  <= threshold`). Exists purely so effect can run with no GPU/wgpu
  adapter present — e.g. `.github/workflows/ci.yml`'s runner. Each pass
  parallelizes over independent output rows with `rayon::par_chunks_mut`
  (same technique as `backends/gstreamer.rs`'s `split_rgb_planes`/
  `join_rgb_planes`),
  so it's GPU-less but not single-threaded. Caches basis matrices in a
  `RefCell<Option<Basis>>` keyed on `(width, height)`, mirroring `DctGpu`'s
  `PlaneBuffers` cache shape minus GPU-specific bind-group machinery.
  `src/cpu.rs`'s test module includes `cpu_and_gpu_backends_agree`
  (GPU-guarded like `gpu.rs`'s tests) as correctness cross-check between
  two implementations — everything else in that module runs
  unconditionally, no adapter guard, since that's whole point.

- **`src/shader.wgsl`** — the two WGSL compute entry points (`row_pass`,
  `col_pass`) that back passes above. Each pass is a dense matrix
  multiply (`row_pass`: `SRC * B^T`, K = width; `col_pass`: `B * SRC`, K =
  height) tiled through `workgroup`-shared memory — standard blocked-GEMM
  technique: walk K dimension in `TILE`-sized (16) chunks, cache both
  operand tiles in shared storage per chunk so every value pulled from
  `storage` is reused `TILE` times instead of once, `workgroupBarrier()`
  between load and accumulate. Only the *memory access pattern* is tiled
  — the frame stays whole-frame, scaling to arbitrary dimensions the same
  as the untiled version did (this is GEMM tiling, unrelated to any fixed
  block size in the DCT itself). `TILE=16` gives 256 invocations/workgroup
  (the portable `max_compute_invocations_per_workgroup` limit, per
  `wgpu::Limits::default()`/`downlevel_defaults()`) and 2KB of shared
  storage, both under the portable 16KB
  `max_compute_workgroup_storage_size` floor.
  - Because `workgroupBarrier()` requires every invocation in the
    workgroup to reach it, out-of-range threads (frame dimensions not a
    multiple of `TILE`) can't early-return before the tile loop the way
    the untiled version could bounds-check up front — they stay in
    lockstep through every barrier with zero-padded loads, only skipping
    the final `dst` write. Covered by `cpu_and_gpu_backends_agree` and the
    GPU test suite, including the `width==1`/`height==1` edge case
    (`handles_one_pixel_wide_and_tall_frames`, where `num_tiles` still
    computes to 1 despite being far smaller than `TILE`).

## Communication style

This file (and `.cursor/rules/`, `.windsurf/rules/`, `.clinerules/`,
`.github/copilot-instructions.md`) generated via `caveman-init`: terse,
fragment-heavy chat responses, no filler/pleasantries. Boundary that
matters most: **code, commit messages, and PR descriptions still written
in normal, full prose** — terse style applies to conversational responses
only, not anything persisted into repo or git history. Caveman activation
rules for this file itself below.

---

Respond terse like smart caveman. All technical substance stay. Only fluff die.

Rules:
- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Not: "Sure! I'd be happy to help you with that."
- Yes: "Bug in auth middleware. Fix:"

Switch level: /caveman lite|full|ultra|wenyan-lite|wenyan-full|wenyan-ultra
Stop: "stop caveman" or "normal mode"

Auto-Clarity: drop caveman for security warnings, irreversible actions, user confused. Resume after.

Boundaries: code/commits/PRs written normal.
