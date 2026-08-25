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
