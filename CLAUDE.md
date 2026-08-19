# CLAUDE.md

This file guide Claude Code (claude.ai/code) working code this repo.

## What this is

`wavefold` — single native Rust binary (desktop GUI via iced, or headless via
`clap` subcommand). Apply whole-frame DCT "distortion" effect to video:
decode, per-frame DCT compress/reconstruct on user-selectable compute
backend (GPU via wgpu, or pure-CPU fallback), re-encode with
user-selectable encoder — software (x264/x265/vp9/av1 GStreamer elements)
or VAAPI hardware — audio pass through untouched. Visual effect tool, not
real codec — point is ringing/ghosting artifact DCT cutoff produce, not
compression efficiency.

Media I/O (`pipeline.rs`/`encoders.rs`) built on **GStreamer**
(`gstreamer`/`gstreamer-app`/`gstreamer-video`), not FFmpeg — replaced
earlier `ffmpeg-next`-based implementation specifically because
`ffmpeg-sys-next` declares `links = "ffmpeg"`, and Cargo hard-bans two
versions of `links`-crate coexisting one dependency graph — made
structurally impossible for one `Cargo.toml` build against both this dev
machine's FFmpeg (Arch, rolling) and stock Ubuntu CI runner's older one
same time. GStreamer's C API/ABI stable since 1.0 (2012), so one
`gstreamer` crate version builds against wide range installed GStreamer
versions, no problem.

## Commands

```bash
cargo build --release                          # release build: single wavefold binary
wavefold                                       # launch the iced GUI (default with no subcommand)
wavefold gui                                   # same, explicit
wavefold encode <in> <out> [--cutoff F] [--encoder ...] [--backend gpu|cpu]  # headless, no display server needed
cargo check                                    # fast typecheck, iterate with this first
cargo test --release                           # unit tests (gpu.rs, cpu.rs, dct_math.rs, pipeline.rs) + tests/integration.rs
cargo test --release <name>                    # run a single test by substring, e.g. `cargo test --release roundtrip`
```

GPU tests (in `src/gpu.rs`, plus cross-check test `src/cpu.rs`) need real
wgpu adapter, skip gracefully (print stderr, not fail) if `DctGpu::new()`
find none — expect skips headless/adapterless sandbox. CPU-backend tests
(`src/cpu.rs`) no such guard, always run — deliberate, proof CPU backend
need no GPU at all. `tests/integration.rs` shells out `ffmpeg` CLI (not
`ffmpeg-next`) purely generate synthetic test-fixture clips; also skips
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

`gstreamer-sys`/`gstreamer-app-sys`/`gstreamer-video-sys` bind against
system's libgstreamer-1.0/libgstapp-1.0/libgstvideo-1.0 via `pkg-config` —
unlike old ffmpeg-next setup, no major-version-must-match constraint
(GStreamer's C ABI stable since 1.0), so any reasonably current GStreamer
dev install works. From-scratch build need: `pkg-config`,
`libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev` (Debian/Ubuntu
naming; `gstreamer`/`gst-plugins-base` on Arch) for headers, **plus actual
plugins at runtime** — `gst-plugins-good` (`vp9enc`), `gst-plugins-bad`
(`x265enc`, `av1enc`, `va` VAAPI plugin), `gst-plugins-ugly` (`x264enc`) —
`EncoderChoice` whose element not installed fails at pipeline-construction
time, clear error (`gst::
ElementFactory::make` returning `None`), not
compile time. Also install `gstreamer1.0-libav` (or distro equivalent) for
broad-codec *decoding* — this repo's own dev/CI environment needed it,
default `openh264dec` decoder can't handle every H.264 profile FFmpeg
itself produce (see `pipeline.rs`'s decode-side notes below). `va`-plugin
VAAPI elements (`vah264enc`/`vah265enc`/`vaav1enc`) additionally need
VAAPI-capable GPU/driver (`libva`, `/dev/dri/renderD*` node) at *runtime*
to even register as available element factories — no such device,
`gst_inspect_1.0 va` still loads plugin but individual codec elements just
don't exist, so `ElementFactory::make("vah264enc")` fails same clean way
as genuinely-uninstalled plugin, not build failure.

## Architecture

Pipeline, one direction: `main.rs` (clap entry point, dispatch to iced
GUI or headless encode) → `ui/` (setup/encoding pages, GUI path only) →
`pipeline.rs` (`gst::Pipeline` built, driven via `gstreamer`/
`gstreamer-app`/`gstreamer-video`, using `encoders.rs` for chosen output
codec's element) → `dct_backend.rs`'s `ComputeBackend` (GPU or CPU) →
`gpu.rs` (wgpu compute) / `cpu.rs` (plain Rust + rayon), both driven by
same basis math `dct_math.rs` → (GPU only) `shader.wgsl`.

- **`src/lib.rs`** re-exports `pub mod cpu; pub mod dct_backend; mod
  dct_math; pub mod encoders; pub mod gpu; pub mod pipeline;` so both
  `wavefold` binary (`main.rs`/`ui/`) and `tests/integration.rs` link
  against same public API — split exists specifically so pipeline
  exercised without GUI. `dct_math` deliberately *not* `pub` — `pub(crate)`
  plumbing shared only between `gpu.rs` and `cpu.rs`, not part crate's
  public surface. `ui/` and `main.rs` binary-only (not part library) since
  not exercised by `tests/integration.rs`.

- **`src/main.rs`** — single binary, one `clap::Parser` (`Cli { command:
  Option<Command> }`) with two subcommands: `Gui` (also default when no
  subcommand given, via `.unwrap_or(Command::Gui)` — preserves old "just
  run binary" muscle memory) and `Encode { input, output, cutoff,
  encoder, backend }`. `run_gui()` same
  `iced::application(ui::App::default, ui::App::update, ui::App::view).run()`
  bootstrap binary always had; `run_encode(...)` former `wavefold-cli`
  binary's body verbatim — spawns `pipeline::run` on `std::thread` exactly
  like `ui/encoding.rs` does (blocking call, not async), then drives
  `tokio::sync::mpsc` receiver on main thread with `rx.blocking_recv()`
  (works fine no tokio runtime present, exactly what `blocking_recv` for),
  exits `1` on `PipelineMsg::Error`. Merging both entry points one binary
  means `encode` subcommand now always links `iced`/`wgpu` too (previously
  separate `wavefold-cli` binary skipped that dependency weight) —
  harmless, since `run_gui()` simply never called that path; no window/GPU
  surface touched unless `Gui` branch runs. This binary
  `.github/workflows/ci.yml` runs as `wavefold encode --backend
  cpu` on
  GPU-less runner.

- **`src/dct_backend.rs`** — `DctBackend` (trait `gpu.rs`'s `DctGpu` and
  `cpu.rs`'s `DctCpu` both implement: one `process_rgb(r, g, b, width,
  height, cutoff)` method) and `ComputeBackend` (user-facing `Gpu`/`Cpu`
  choice — same enum+`ALL`+`Display`+resolver shape as `EncoderChoice` in
  `encoders.rs`, same reason: one switchable choice GUI pick_list and
  CLI's `--backend` flag both need). `ComputeBackend::
  build()` single
  place turns choice into live `Box<dyn DctBackend>` (`DctGpu::new()`, can
  fail no compatible adapter, vs `DctCpu::new()`, can't fail). Also
  derives `clap::ValueEnum` (as does `EncoderChoice`) so CLI's
  `--backend`/`--encoder` flags share one source of truth with GUI's
  pick_lists instead of separate hand-maintained CLI-side enum.

- **`src/dct_math.rs`** — `dct_basis`/`transpose_square`, pure-CPU
  matrix-generation math shared verbatim by both `gpu.rs` (uploads result
  GPU buffers) and `cpu.rs` (uses directly as transform matrices) — pulled
  out specifically so two compute backends never silently drift different
  basis matrices.

- **`src/ui/`** — iced 0.14 app (`features = ["tokio"]`, so iced's
  executor real tokio runtime), split into two pages following iced's
  documented multi-screen pattern:
  - **`ui/mod.rs`** — `App` wraps `enum Screen { Setup(setup::State),
    Encoding(encoding::State) }` and wrapper `Message` enum; `App::update`
    dispatches to whichever screen active, applies `Action` each screen's
    own `update` returns (`setup::Action::Start{..}` swaps `Screen` to
    `Encoding`, kicks off encode; `encoding::
    Action::BackToSetup` swaps
    back to fresh `Setup`). Screen never mutates other screen's state or
    top-level `Screen` directly — only `App::update` does, based on
    `Action` it gets back. Stray messages for screen navigated away from
    (e.g. file-dialog result resolving after leaving Setup) silently
    dropped in dispatch match.
  - **`ui/setup.rs`** — input/output file pickers (`rfd::AsyncFileDialog`
    via `Task::perform`, not old synchronous `rfd::FileDialog`), cutoff
    slider, encoder `pick_list` (`EncoderChoice` implements `Display` in
    `encoders.rs` specifically for this), second `pick_list` for
    `ComputeBackend` (same `Display`-via-`label()` pattern, in
    `dct_backend.rs`), Encode button (`on_press_maybe`, only enabled once
    both paths set).
  - **`ui/encoding.rs`** — `State::start(input, output, cutoff, encoder,
    backend)` spawns `pipeline::run` on plain `std::thread` (blocking
    call, not async — spawning as tokio task would tie up executor thread
    whole encode), returns `Task` streaming `tokio::sync::mpsc` progress
    channel back via `Task::run(
    UnboundedReceiverStream::new(rx), Message::Pipeline).chain(Task::done(
    Message::WorkerDone))` — progress arrives reactively as messages
    instead of old egui-era pattern polling channel every frame. Shows
    progress bar, scrollable log, "New encode" button (enabled once worker
    reports done) returning `Action::
    BackToSetup`.

- **`src/encoders.rs`** — `EncoderChoice` (8 variants: H264/H265/Vp9/Av1,
  each with `*Vaapi` hardware counterpart) and its `profile()` →
  `EncoderProfile { element_factory_name, properties, parser, hardware }`:
  GStreamer element factory name for `gst::ElementFactory::make` (not
  codec-ID lookup — several codecs have more than one GStreamer encoder
  element), that element's own properties (set via
  `set_property_from_str`, type-coerces from plain string regardless
  whether property itself string, enum, or integer), optional bitstream
  parser element name. Different encoders take entirely different
  property sets — `tune=zerolatency` for x264/x265; `deadline`+
  `lag-in-frames` for vp9; `cpu-used`+`lag-in-frames` for av1; no
  properties for VAAPI variants — not codec-ID swap.
  - **Every software encoder needs internal lookahead/B-frame buffering
    disabled** (frame-in, frame-out), or can fail emit first output
    packet fast enough for muxer (a `GstAggregator`) complete preroll on
    that pad — stalls **whole pipeline's** `Paused`->`Playing` transition
    forever, not just one branch (`appsink` only ever delivers one
    preroll buffer while stuck in `Paused`; confirmed reproducing this
    exact hang with `x264enc`'s defaults). Fix differs per encoder
    (`tune=zerolatency` for x264/x265; `lag-in-frames=0` for vp9/av1) so
    each set explicitly rather than assumed to share x264's property
    names. VAAPI encoders don't need this: `b-frames` already defaults to
    `0`.
  - **`parser: Option<&'static str>`** (`h264parse`/`h265parse`) is
    inserted between encoder and muxer's request pad — standard
    GStreamer practice, and *required* in practice for H.265: `x265enc`'s
    raw output caps don't have format `qtmux`/`matroskamux`'s video pad
    template accepts directly (fails with "Pads do not have common
    format" otherwise, confirmed). `h264parse` inserted uniformly too
    even though `x264enc` happened to link without one — cheap insurance
    against same class of failure on muxer/version this wasn't tested
    against. VP9/AV1 need no parser.
  - **No `vavp9enc` exists** in GStreamer's `va` plugin (confirmed against
    this machine's real AMD/Mesa VAAPI driver: only `vavp9dec` registers,
    no encoder) — same VP9-hw-encode gap this project already hit and
    tolerated via previous ffmpeg-next VAAPI path. `Vp9Vaapi` stays
    selectable (`ElementFactory::make` fails cleanly with `None` rather
    than panicking) so tolerant hardware-failure skip in
    `tests/integration.rs` still exercises that path.
  - **`hardware: bool`** is read only by `tests/integration.rs`'s
    tolerant-skip logic (hardware availability is environment/
    driver-dependent) — `pipeline.rs` itself doesn't need to know.

- **`src/pipeline.rs`** — builds and drives one `gst::Pipeline` per encode.
  `run`/`run_inner` take a `backend: ComputeBackend` parameter (alongside
  `encoder_choice`); `run_inner` resolves it once via `backend.build()?`
  into a `Box<dyn DctBackend>` (the trait has `Send` as a supertrait
  specifically so it can move into the `appsink` callback below, which
  GStreamer invokes from its own streaming thread) before building the
  pipeline. `PipelineMsg`'s shape (`Progress`/`Log`/`Done`/`Error` over a
  `tokio::sync::mpsc::UnboundedSender`) is unchanged from earlier
  ffmpeg-next implementation — whole point of this rewrite was to keep
  that outward contract identical so `ui/`, `main.rs`, and
  `tests/integration.rs` didn't have to change.
  - **Pipeline shape**: `filesrc ! decodebin`, whose `autoplug-continue`
    signal is told to keep decoding video (`true`) but stop the moment
    audio caps are no longer already `audio/x-raw` (`false`) — this is
    what makes audio a pure passthrough (no decode/re-encode) without a
    manual demux/remux step: `decodebin` just exposes the still-encoded
    pad directly instead of auto-plugging a decoder for it. The video pad
    goes `videoconvert ! appsink` (forced to `video/x-raw,format=RGB`);
    the DCT step runs inside `appsink`'s callbacks (see below), pushing
    into `appsrc ! videoconvert ! <encoder> ! [<parser> !] queue !
    <muxer>`. The audio pad (once linked) goes straight `queue !
    <muxer>`, no decode/encode element in between. No manual worker
    threads or channels are needed for overlap the way the old
    three-thread ffmpeg-next design required — GStreamer's own `queue`
    elements provide that buffering/overlap for free; `queue` elements
    exist in this graph for exactly that reason, not just as glue.
  - **The DCT step lives in `process_and_forward`**, called from both of
    `appsink`'s `new_preroll` and `new_sample` callbacks: pull the sample,
    extract RGB planes via `gst_video::VideoFrameRef` (stride-aware, same
    algorithm `split_rgb_planes`/`join_rgb_planes` always used, just
    reading/writing `VideoFrameRef` instead of an `ff::frame::Video`),
    call `DctBackend::process_rgb` (unchanged), reassemble into a new
    buffer, copy the original buffer's PTS/DTS/duration across unchanged
    (GStreamer expresses time in nanoseconds uniformly through the whole
    pipeline — unlike the old ffmpeg-next path, there is no timebase to
    rescale), and `appsrc.push_buffer(...)`.
  - **`appsink` delivers its very first buffer via `new_preroll` *and* the
    first `new_sample` call after reaching `Playing`** — both for the
    *same* buffer (confirmed empirically: both calls reported an identical
    PTS). `process_and_forward` dedups via a shared `last_pts: Mutex<
    Option<ClockTime>>`, skipping a call whose PTS matches the immediately
    preceding one. Skipping `new_preroll`'s forward entirely (only
    handling `new_sample`) is *not* a fix: without it, the video encode
    branch never gets a first buffer at all, which stalls the whole
    pipeline's `Paused`->`Playing` transition forever (the muxer, a
    `GstAggregator`, can't complete preroll on a starved pad) — confirmed
    by reproducing that hang with a no-op `new_preroll`. Both callbacks
    must forward identically; the dedup has to happen on the PTS instead.
  - **`appsink`'s EOS is not automatically forwarded to `appsrc`** —
    without an explicit `.eos(|_| appsrc.end_of_stream())` callback, the
    encode branch never learns decoding finished and the pipeline hangs
    forever after the last frame.
  - **`appsrc`'s caps must be fully specified** (`format`+`width`+
    `height`+`framerate`, not just `format`) before any buffer is pushed,
    or negotiation fails at `Playing` with "no width property given" —
    set once the decoded video pad's own caps are known, in
    `connect_pad_added`, since `appsrc` has no upstream to negotiate
    dimensions from the way `appsink` does.
  - **A *strong* clone of `pipeline` captured inside `decodebin`'s
    `pad-added` closure is a reference cycle**, not just a convenience:
    the closure is stored inside `decodebin`, which is a child element
    owned by `pipeline` itself, so `pipeline -> decodebin -> closure ->
    pipeline` never drops. That silently leaked *every* `tx` clone held
    by *any* callback on *any* element in the pipeline, which meant the
    `PipelineMsg` channel never closed and every caller's `while let
    Some(msg) = rx.recv()` loop hung forever — even after already
    receiving `Done`/`Error` — regardless of whether the underlying
    encode would've succeeded (confirmed: all of `tests/integration.rs`'s
    tests hung identically). `pipeline.downgrade()` (a `glib::WeakRef`,
    `.upgrade()`'d only where actually needed — the total-duration query)
    breaks the cycle.
  - **`pipeline.set_state(Null)` must run on *every* exit path, not just
    the success one.** Early on, a `set_state(Playing)` failure (e.g.
    missing input file) returned via `?` before ever reaching the
    Null-transition cleanup at the end of the function — and dropping a
    `gst::Pipeline` that was never cleanly transitioned to `Null` can
    itself hang during teardown (GStreamer's internal dispose logic
    forcing a state change while finalizing). Fixed by collecting the
    Playing-transition result into a local instead of using `?` on it
    directly, so the `Null` transition always runs before the function
    returns either way.
  - **`filesink` opens (creates/truncates) its output file as part of the
    pipeline's state transition, independent of whether upstream (e.g.
    `filesrc` on a missing input) ever actually succeeds** — so a failed
    encode can leave a zero-byte output file behind. `run_inner` removes
    `output_path` on any error return to match the old ffmpeg-next
    behavior (which never touched the file at all in that case).
  - **Total-frame estimate for progress reporting**: `query_duration`
    reliably returns `None` at `pad-added` time (confirmed — too early,
    before the demuxer has parsed enough to know it) and `DurationChanged`
    isn't reliably posted for every demuxer/container combination either
    (confirmed — never fired for either an mp4 or an mkv test fixture
    here). What actually works: block on `pipeline.state(timeout)` right
    after `set_state(Playing)` (waits for the async `Paused`->`Playing`
    transition to genuinely finish) and query once more at that point —
    by then the demuxer has necessarily parsed far enough to answer. Falls
    back to `0` ("unknown") if even that fails, same tolerant behavior as
    the old ffmpeg-next-based estimate.
  - **`qtmux`/`x265enc`+VAAPI-HEVC caveat**: `x265enc`'s raw output caps
    don't satisfy `qtmux`'s video pad template without a `h265parse`
    element in between (see `encoders.rs` above) — and this AMD/Mesa
    VAAPI driver's `vah265enc` pads non-64-aligned frame dimensions to the
    next HEVC CTU boundary without writing a correct SPS conformance-window
    crop back, so a probe of the muxed output can report the *padded*
    dimensions instead of the real ones (confirmed directly with
    `gst-launch-1.0`) — a genuine driver limitation, not something
    `pipeline.rs` can fix; `tests/integration.rs`'s shared multi-encoder
    fixture stays 64-aligned specifically to sidestep it.
  - **`vp9enc` cannot mux into `qtmux`** in this GStreamer version: it
    never emits a `chroma-format` field in its output caps regardless of
    upstream pixel format, while `qtmux`'s `video/x-vp9` pad template
    requires one (confirmed by testing every pixel format `vp9enc`
    accepts) — `matroskamux` has no such requirement and works fine.
    `tests/integration.rs` tolerates this specific combination as a known
    muxer gap, the same way it already tolerates hardware-encoder
    unavailability.
  - `split_rgb_planes`/`join_rgb_planes` parallelize their per-row loops
    with `rayon` (`par_chunks_mut`) — each row is a disjoint read/write,
    no aliasing between them; unchanged in spirit from the ffmpeg-next
    version, just reading/writing `gst_video::VideoFrameRef` now.
  - `PipelineMsg::Log` (to UI's in-app log panel) and `tracing::{info,
    debug,warn,error}!` (to stderr/whatever subscriber `main.rs` installs)
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
via `caveman-init`: terse, fragment-heavy responses in chat, no
filler/pleasantries. Boundary that matters most: **code, commit messages,
and PR descriptions still written in normal, full prose** — terse style
applies to conversational responses only, not to anything persisted into
the repo or git history.