# `backends::gstreamer` implementation notes

Deep pitfalls/war-stories behind `src/backends/gstreamer.rs`. Split out of
`AGENTS.md` to keep that file's per-session load light — pull this up only
when actually touching `backends/gstreamer.rs`.

`codec_profile(codec)` → private `EncoderProfile { element_factory_name,
properties, parser }`: GStreamer element factory name for
`gst::ElementFactory::make` (not codec-ID lookup — several codecs have more
than one GStreamer encoder element), that element's own properties (set via
`set_property_from_str`, type-coerces from plain string regardless whether
property itself string, enum, or integer), optional bitstream parser element
name. Different encoders take entirely different property sets —
`tune=zerolatency` for x264/x265; `deadline`+`lag-in-frames` for vp9;
`cpu-used`+`lag-in-frames` for av1; no properties for hardware variants —
not codec-ID swap.

- **Every software encoder needs internal lookahead/B-frame buffering
  disabled** (frame-in, frame-out), or can fail to emit first output packet
  fast enough for muxer (a `GstAggregator`) to complete preroll on that pad
  — stalls **whole pipeline's** `Paused`->`Playing` transition forever, not
  just one branch (`appsink` only ever delivers one preroll buffer while
  stuck in `Paused`; reproduced this exact hang with `x264enc`'s defaults).
  Fix differs per encoder (`tune=zerolatency` for x264/x265;
  `lag-in-frames=0` for vp9/av1) so each set explicitly rather than assumed
  to share x264's property names. Hardware (VAAPI) encoders don't need
  this: `b-frames` already defaults to `0`.
- **`parser: Option<&'static str>`** (`h264parse`/`h265parse`) inserted
  between encoder and muxer's request pad — standard GStreamer practice,
  and *required* in practice for H.265: `x265enc`'s raw output caps don't
  have format `qtmux`/`matroskamux`'s video pad template accepts directly
  (fails with "Pads do not have common format" otherwise). `h264parse`
  inserted uniformly too even though `x264enc` happened to link without one
  — cheap insurance against same class of failure on muxer/version this
  wasn't tested against. VP9/AV1 need no parser.
- **No `vavp9enc` exists** in GStreamer's `va` plugin (confirmed against
  this machine's real AMD/Mesa VAAPI driver: only `vavp9dec` registers, no
  encoder) — same VP9-hw-encode gap this project already hit and tolerated
  via previous ffmpeg-next VAAPI path. `Codec::Vp9Hardware` stays
  selectable (`ElementFactory::make` fails cleanly with `None` rather than
  panicking) so tolerant hardware-failure skip in `tests/integration.rs`
  still exercises that path.
- **`Codec::is_hardware()`** is read only by `tests/integration.rs`'s
  tolerant-skip logic (hardware availability is environment/
  driver-dependent) — `codec_profile` itself dispatches on the `Codec`
  variant directly, doesn't need it.

`GstreamerBackend::run` takes an already-resolved `dct: Box<dyn
DctBackend>` (alongside `codec`) — `pipeline.rs`'s dispatcher builds it
before calling in, this module never touches `ComputeBackend` at all.
`run_inner` moves it directly (no `Arc<Mutex<...>>` — trait's `Send`
supertrait is what lets it move whole into dedicated compute thread below,
only thread that ever touches it, so no shared mutable access needs
guarding) before building pipeline. `PipelineMsg`'s shape
(`Progress`/`Log`/`Done`/`Error` over a `tokio::sync::mpsc::UnboundedSender`,
now defined in `media_backend.rs` rather than here) unchanged from earlier
ffmpeg-next implementation — whole point of original GStreamer rewrite was
to keep that outward contract identical so `ui/`, `main.rs`, and
`tests/integration.rs` didn't have to change; later `MediaBackend` split
kept it unchanged again for same reason.

- **Pipeline shape**: `filesrc ! decodebin`, whose `autoplug-continue`
  signal told to keep decoding video (`true`) but stop the moment audio
  caps no longer already `audio/x-raw` (`false`) — this is what makes
  audio pure passthrough (no decode/re-encode) without a manual
  demux/remux step: `decodebin` just exposes still-encoded pad directly
  instead of auto-plugging a decoder for it. Video pad goes `dec_queue !
  videoconvert ! appsink` (forced to `video/x-raw,format=RGB`), pushing
  into `appsrc ! videoconvert ! <encoder> ! [<parser> !] queue ! <muxer>`.
  Audio pad (once linked) goes straight `queue ! <muxer>`, no
  decode/encode element in between.
- **Decode / GPU DCT compute / encode run on three threads connected by
  bounded `std::sync::mpsc::sync_channel`s** (`decode_and_send` in
  `appsink`'s callbacks → a `wavefold-gst-compute` thread → a
  `wavefold-gst-encode` thread that owns `appsrc.push_buffer(...)`) — not
  GStreamer's own `queue` elements providing that overlap for free (only
  true for `venc_queue`/`dec_queue`, protecting muxer/letting decode run
  ahead — not for the DCT compute step, which appsink's callback used to
  run inline). Confirmed via a real user report (~9-15% GPU utilization on
  Linux encoding to AV1 hardware despite DCT compute being real GPU work)
  that decode/compute/encode were fully serialized on one thread; adding
  `dec_queue` alone made no measurable difference — compute and encode
  were the two stages actually blocking each other. Fixed same way
  `backends::media_foundation` already had to (see
  [media-foundation-notes.md](media-foundation-notes.md)) — verified
  directly on real AMD/VAAPI hardware via
  `/sys/class/drm/card*/device/gpu_busy_percent`: ~9% before
  (single-callback), ~60% average (min 44, max 77) after, encoding
  1280x720 to AV1 hardware.
  - **Follow-up: thread-splitting alone still left a stall *inside* the
    compute stage.** `dct.process_rgb` is fully synchronous per frame
    (submit -> `poll_bounded` blocks the compute thread on the fence ->
    readback) before that same thread even starts preparing the *next*
    frame's GPU dispatch - so between decode/compute/encode running as
    overlapping stages, the GPU's own command queue still went empty on
    every frame's CPU-side `map_async` round trip. Fixed with a 2-deep
    frame pipeline: `DctBackend::submit_rgb`/`finish_rgb` (`DctGpu` uses 6
    buffer slots = 2 frame slots x 3 channels, `frame_slot*3+channel`;
    `DctCpu` just computes eagerly and stashes, no GPU queue to keep busy)
    let the compute thread submit frame N+1's work before blocking on
    frame N's readback. Ordering matters: a slot's prior occupant must be
    fully `finish_rgb`'d - unmapping its staging buffer - before
    `submit_rgb` reuses that slot's buffers for a new frame, or wgpu's
    mapping validation rejects the write; EOS drains both slots in
    original submission order (tracked via a monotonic frame counter
    alongside each slot) so PTS/DTS order into the encoder stays intact.
  - **Follow-up to the follow-up: naively deferring frame 0's completion
    deadlocked the whole pipeline.** First cut of the 2-deep pipeline above
    deferred *every* frame's `finish_rgb`, including frame 0's, until its
    slot got reused two frames later - reproduced as a hang (all threads
    idle) on `tests/integration.rs`'s `encodes_synthetic_clip_end_to_end`.
    Cause: GStreamer's PAUSED->PLAYING transition needs every sink -
    `filesink` (downstream of `appsrc`) included, not just `appsink` - to
    receive one preroll buffer before completing; `appsink` only delivers a
    *second* decoded buffer once the pipeline actually reaches PLAYING. With
    frame 0's result held back waiting for frame 2, `filesink` never got its
    preroll buffer, so PLAYING never completed, so frame 2 never got
    decoded - circular. Fix: frame 0 is always submitted *and* finished
    synchronously (one exception, tagged by `frame_counter == 0` in the
    compute thread), priming both branches' preroll before any deferral
    starts; every frame from 1 onward pipelines for real since PLAYING is
    already reached by then.
- **A plain `queue` element between `appsrc` and encoder was tried first
  and reverted** — this pipeline relies on `appsrc.push_buffer()`
  synchronously reflecting downstream encode/mux failures (see the
  `vp9enc`-into-`qtmux` gap below: it fails negotiation but never posts
  its own bus error, so *only* synchronous `GST_FLOW_NOT_NEGOTIATED`
  propagating back through `push_buffer` makes that failure visible at
  all). A queue there decouples that return value from real failure,
  which then happens asynchronously on the queue's own thread — confirmed
  this turns that specific failure into a silent hang
  (`rx.blocking_recv()` never unblocks) instead of clean, fast error
  `tests/integration.rs` already tolerates for that codec/muxer
  combination. Channel-based redesign above keeps the guarantee instead of
  discarding it: the *encode* thread still calls `appsrc.push_buffer()`
  synchronously (just off appsink-callback thread now), and explicitly
  `post_error_message`s onto pipeline's own bus on failure — which
  existing bus-reading loop already handles unchanged.
- **`decode_and_send`** (in `appsink`'s `new_preroll`/`new_sample`
  callbacks) does only fast half of old `process_and_forward`: pull
  sample, extract RGB planes via `gst_video::VideoFrameRef` (stride-aware,
  same algorithm `split_rgb_planes`/`join_rgb_planes` always used), and
  send `DecodedFrame` (planes + PTS/DTS/duration) through bounded channel
  to compute thread — deliberately *not* running DCT or touching `appsrc`
  itself, so callback returns fast and doesn't block
  `decodebin`/`dec_queue`'s own decode-side threading. Compute thread
  calls `DctBackend::process_rgb` (unchanged) and forwards `ComputedFrame`
  to encode thread, which reassembles buffer via `join_rgb_planes`,
  copies PTS/DTS/duration across (GStreamer expresses time in nanoseconds
  uniformly through whole pipeline — no timebase to rescale), and calls
  `appsrc.push_buffer(...)`.
- **`Eos` is explicit message sent through decode→compute→encode
  channels, not inferred from channel closure** — `appsink`'s
  `new_preroll`/`new_sample` closures each hold own clone of decode-side
  sender for as long as pipeline object exists, so channel never closes
  just because `eos` fired. `run_inner` also keeps one extra clone in its
  own scope and sends fallback `Eos` unconditionally after bus loop ends
  for *any* reason (including a `set_state(Playing)` failure, where
  `appsink`'s own `eos` callback never fires at all) — without it, joining
  compute/encode threads before returning could block forever.
- **`appsink` delivers its very first buffer via `new_preroll` *and* the
  first `new_sample` call after reaching `Playing`** — both for the *same*
  buffer (confirmed empirically: both calls reported identical PTS).
  `decode_and_send` dedups via shared `last_pts: Mutex<Option<ClockTime>>`,
  skipping a call whose PTS matches immediately preceding one. Skipping
  `new_preroll`'s forward entirely is *not* a fix: without it, video
  encode branch never gets first buffer at all, stalling whole pipeline's
  `Paused`->`Playing` transition forever (muxer can't complete preroll on
  starved pad). Both callbacks must forward identically; dedup has to
  happen on PTS instead.
- **`appsink`'s EOS not automatically forwarded to `appsrc`** — without
  explicit `.eos(|_| appsrc.end_of_stream())` callback, encode branch
  never learns decoding finished and pipeline hangs forever after last
  frame.
- **`appsrc`'s caps must be fully specified** (`format`+`width`+
  `height`+`framerate`, not just `format`) before any buffer pushed, or
  negotiation fails at `Playing` with "no width property given" — set
  once decoded video pad's own caps known, in `connect_pad_added`, since
  `appsrc` has no upstream to negotiate dimensions from the way `appsink`
  does.
- **A *strong* clone of `pipeline` captured inside `decodebin`'s
  `pad-added` closure is a reference cycle**, not just convenience:
  closure stored inside `decodebin`, which is a child element owned by
  `pipeline` itself, so `pipeline -> decodebin -> closure -> pipeline`
  never drops. That silently leaked every `tx` clone held by any callback
  on any element in pipeline, meaning `PipelineMsg` channel never closed
  and every caller's `while let Some(msg) = rx.recv()` loop hung forever.
  `pipeline.downgrade()` (a `glib::WeakRef`, `.upgrade()`'d only where
  actually needed — the total-duration query) breaks the cycle.
- **`pipeline.set_state(Null)` must run on *every* exit path, not just
  the success one.** A `set_state(Playing)` failure returning via `?`
  before ever reaching Null-transition cleanup left a `gst::Pipeline`
  never cleanly transitioned to `Null`, which can itself hang during
  teardown. Fixed by collecting Playing-transition result into a local
  instead of using `?` on it directly, so `Null` transition always runs
  before function returns either way.
- **`filesink` opens (creates/truncates) its output file as part of
  pipeline's state transition, independent of whether upstream (e.g.
  `filesrc` on missing input) ever actually succeeds** — so a failed
  encode can leave zero-byte output file behind. `run_inner` removes
  `output_path` on any error return to match old ffmpeg-next behavior.
- **Total-frame estimate for progress reporting**: `query_duration`
  reliably returns `None` at `pad-added` time (too early, before demuxer
  has parsed enough) and `DurationChanged` isn't reliably posted for
  every demuxer/container combination either. What works: block on
  `pipeline.state(timeout)` right after `set_state(Playing)` and query
  once more at that point — by then demuxer necessarily parsed far enough
  to answer. Falls back to `0` ("unknown") if even that fails.
- **`qtmux`/`x265enc`+VAAPI-HEVC caveat**: `x265enc`'s raw output caps
  don't satisfy `qtmux`'s video pad template without an `h265parse`
  element in between — and this AMD/Mesa VAAPI driver's `vah265enc` pads
  non-64-aligned frame dimensions to next HEVC CTU boundary without
  writing correct SPS conformance-window crop back, so a probe of muxed
  output can report *padded* dimensions instead of real ones (confirmed
  directly with `gst-launch-1.0`) — genuine driver limitation, not
  something `backends/gstreamer.rs` can fix;
  `tests/integration.rs`'s shared multi-encoder fixture stays 64-aligned
  specifically to sidestep it.
- **`vp9enc` cannot mux into `qtmux`** in this GStreamer version: it never
  emits a `chroma-format` field in output caps regardless of upstream
  pixel format, while `qtmux`'s `video/x-vp9` pad template requires one —
  `matroskamux` has no such requirement and works fine.
  `tests/integration.rs` tolerates this specific combination as known
  muxer gap, same way it already tolerates hardware-encoder unavailability.
- **Raw PCM audio input (e.g. some iOS/Instagram exports' 'ipcm' box)
  gets static/garbled on stream-copy passthrough — fixed by re-encoding to
  AAC instead.** Root cause: this GStreamer version's `qtmux`
  (gst-plugins-good 1.28.6, confirmed against its own source) has no code
  path to write anything but the legacy 'sowt'/'twos' fourcc for raw PCM
  audio — no 'ipcm'/'lpcm' support at all in the muxer (only in
  `qtdemux`'s read side). Output PCM bytes end up byte-identical to the
  source (verified), but common players apparently mishandle/misinterpret
  'sowt'/'twos' for >16-bit-depth PCM, producing static. Not fixable
  muxer-side. Fix: `connect_pad_added`'s audio branch now checks
  `name == "audio/x-raw"` — already-compressed audio (AAC/Opus/etc.)
  still links `aud_queue` straight to the muxer's `audio_%u` pad,
  byte-identical stream copy, completely unchanged; raw PCM instead links
  through a small always-present-but-conditionally-used re-encode chain,
  `aud_queue ! audioconvert ! avenc_aac !` muxer (`audioconvert` added
  since raw PCM caps like S24LE won't necessarily match whatever sample
  format `avenc_aac` wants). `avenc_aac` (GStreamer's `libav` plugin, from
  `gst-libav`/`gstreamer1.0-libav`) needs no new runtime dependency —
  AGENTS.md's system-dependencies section already mandates that package
  for decoding. Confirmed empirically via `gst-launch-1.0` that
  `avenc_aac`'s raw AAC output (`audio/mpeg, mpegversion=4,
  stream-format=raw`) links directly into both `qtmux` and `matroskamux`
  with no `aacparse` needed, unlike H.265/`h265parse`. `audioconvert`/
  `avenc_aac` are added to the pipeline and statically linked to
  `aud_queue` unconditionally at construction time (the raw-vs-compressed
  decision only happens later, inside `connect_pad_added`) — harmless if
  never used downstream, same tolerance already extended to `aud_queue`
  itself for audio-less inputs. Verified against a real Instagram/iOS
  export with 24-bit PCM audio (HEVC video, `.mov`→`.mp4`).
- `split_rgb_planes`/`join_rgb_planes` parallelize per-row loops with
  `rayon` (`par_chunks_mut`) — each row disjoint read/write, no aliasing
  between them; unchanged in spirit from ffmpeg-next version, just
  reading/writing `gst_video::VideoFrameRef` now.
- `PipelineMsg::Log` (to UI's in-app log panel) and `tracing::{info,
  debug,warn,error}!` (to stderr/whatever subscriber `main.rs` installs)
  fire at same call sites deliberately — different audiences (end-user
  status vs. developer diagnostics), not redundancy to clean up.
  GPU-test skip messages in `gpu.rs` stay on `eprintln!` rather than
  `tracing` on purpose: test binaries never call
  `tracing_subscriber::fmt::init()`, so a `tracing` call there would
  silently vanish instead of printing.
