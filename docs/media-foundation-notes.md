# `backends::media_foundation` implementation notes

Deep pitfalls/war-stories behind `src/backends/media_foundation.rs` (only
`MediaBackend` impl on Windows, `cfg(windows)`, built on
`IMFSourceReader`/`IMFSinkWriter` instead of `gst::Pipeline`). Split out of
`AGENTS.md` to keep that file's per-session load light — pull this up only
when actually touching this backend.

Originally verified only by cross-compiling *whole* crate (GUI included)
for `x86_64-pc-windows-msvc` via `cargo xwin` and exercising decode/encode
under Wine's `winegstreamer`-backed MF implementation, no real Windows
machine available end-to-end; `tests/integration.rs` has since run for
real on Windows 11 (real GPU, real built-in codec MFTs) and surfaced three
genuine platform-specific gaps, all confirmed and now tolerated at test
level rather than treated as wavefold bugs:

- Windows' built-in H.264 encoder MFT rejects `SetInputMediaType` with
  0xC00D36B4 below some resolution floor between 32 and 48px on either
  axis (genuine minimum-size limitation in that MFT, not a media-type
  construction bug). `tests/integration.rs`'s small fixtures are 64x64,
  not smaller.
- Software `Codec::Av1` has no encoder MFT on Windows at all — only a
  decoder ships ("AV1 Video Extensions") — confirmed via
  `SetInputMediaType` failing with 0xC00D5212 ("no suitable transform").
  `Codec::Av1Hardware` unaffected (a real GPU's own AV1 hardware encoder
  MFT, if present, registers independently).
- Windows' built-in MP4 sink negotiates raw PCM audio passthrough fine
  (`AddStream`/`SetInputMediaType` both succeed) but fails at `Finalize`
  with 0xC00D4A45 ("required headers were not provided") — its documented
  MP4 audio support essentially AAC-only, so it can't actually mux PCM
  despite accepting the type.
- Separately (not a `wavefold`-code issue, just a fixture footgun):
  ffmpeg-generated synthetic clips need `-pix_fmt yuv420p` explicitly —
  without it `libx264` defaults to High 4:4:4 Predictive/yuv444p for
  `testsrc`'s native chroma sampling, which Windows' H.264 decoder MFT
  can't decode (`ReadSample` fails with generic "CopyDecodedFrame
  failed", 0x80004005). And Windows' Matroska source has been observed to
  misreport `MF_MT_FRAME_RATE` by exactly half for at least one
  ffmpeg-muxed mkv fixture (cross-checked against `ffprobe`'s correct
  `r_frame_rate` on the identical file) — `tests/integration.rs`'s
  total-frame-estimate assertion loosened accordingly on Windows.

- **`codec_target(codec)`** → `{ subtype: GUID, hardware: bool }`:
  `MFVideoFormat_H264`/`_HEVC`/`_VP90`/`_AV1` for subtype, `hardware`
  feeds `MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS` on sink writer (allow
  vs. force-software a hardware MFT — MF auto-picks whichever registered
  encoder matches subtype, no GStreamer-style explicit element-name
  selection). Windows has no built-in VP9 *encoder* MFT (only a decoder,
  via "VP9 Video Extensions") — same gap `backends::gstreamer` already has
  for `vavp9enc`, kept selectable and left to fail cleanly at
  `AddStream`/`SetInputMediaType` time.
- **`MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING` must be set on source
  reader's attributes** before requesting `MFVideoFormat_RGB32` output —
  without it, `SetCurrentMediaType` fails with `MF_E_INVALIDMEDIATYPE`.
  This is what lets source reader insert whatever colorspace-convert MFT
  native format needs, mirroring why `backends::gstreamer` forces
  `video/x-raw,format=RGB` through a `videoconvert` element.
- **`MFVideoFormat_RGB32` is BGRA, not RGB**, and can be a *bottom-up* DIB
  (negative stride from `IMF2DBuffer::Lock2D`) — classic Windows bitmap
  convention this format inherits. `split_bgra_planes` handles both
  channel order and negative stride's row-order flip; `join_bgra_planes`
  always writes top-down (`stride == width*4`) since it's a fresh buffer
  this backend allocates itself, no inherited orientation to preserve.
- **`IMF2DBuffer::Lock2D` used over plain `IMFMediaBuffer::Lock`** when
  available — correct, stride-aware way to read image data (pitch can
  exceed `width*4` for padding), same reasoning as `backends::gstreamer`'s
  `VideoFrameRef::plane_stride()`. Falls back to treating buffer as
  tightly packed for rare buffer that doesn't implement `IMF2DBuffer`.
- **Audio passthrough** (`setup_audio_passthrough`) adds input's native,
  still-encoded audio type straight to sink writer (no decode/re-encode)
  — `IMFSourceReader`-equivalent of `backends::gstreamer`'s
  `autoplug-continue`-based stream copy. Main read loop uses
  `MF_SOURCE_READER_ANY_STREAM`, letting reader itself decide which
  stream's next sample due, instead of manually interleaving
  video/audio reads.
- **Container support is real Windows Media Foundation's own limitation,
  not something this code can paper over**: MF reliably ships byte-stream
  handlers for mp4/mov/asf, not Matroska — unlike `backends::gstreamer`'s
  `matroskamux` fallback for non-mp4 outputs (needed there specifically
  for VP9), a `.mkv` output on MF backend just fails with whatever error
  MF gives for unresolvable handler.
- **`CoInitializeEx`/`MFStartup` and their matching
  `CoUninitialize`/`MFShutdown`** are RAII-guarded
  (`ComGuard`/`MfGuard`) so early `?` on any setup step still runs them on
  the way out — same every-exit-path reasoning as
  `backends::gstreamer`'s `pipeline.set_state(Null)`.
