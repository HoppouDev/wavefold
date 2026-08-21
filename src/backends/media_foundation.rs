//! Windows-only `MediaBackend` built on Media Foundation
//! (`IMFSourceReader`/`IMFSinkWriter`) instead of GStreamer — Windows has
//! no system package manager providing GStreamer the way Linux distros do,
//! but Media Foundation ships with the OS itself, so this backend needs no
//! bundled runtime at all (see `codec.rs`/`media_backend.rs` for why
//! `backends::gstreamer` is `cfg(not(windows))` and this is `cfg(windows)`
//! — never both compiled into the same binary).
//!
//! API shape verified against the real `windows` crate bindings compiled
//! for the actual `x86_64-pc-windows-msvc` target (via `cargo xwin`, which
//! cross-links against real Windows SDK import libraries) and exercised at
//! runtime under Wine's `winegstreamer`-backed Media Foundation
//! implementation: `IMFSourceReader` decode to `MFVideoFormat_RGB32`
//! (confirmed real BGRA32 pixel data, correct frame count, needs
//! `MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING` or negotiation fails with
//! `MF_E_INVALIDMEDIATYPE`) and `IMFSinkWriter` H264 setup
//! (`AddStream`/`SetInputMediaType`/`BeginWriting`/`WriteSample` all
//! confirmed to succeed; `Finalize` returned `E_NOTIMPL` under Wine's
//! encoder/mux implementation specifically, not something attributable to
//! this code) — real Windows was not available to verify end-to-end
//! either way.

use crate::codec::Codec;
use crate::dct_backend::DctBackend;
use crate::media_backend::{MediaBackend, PipelineMsg};
use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use tokio::sync::mpsc::UnboundedSender as Sender;
use tracing::{info, warn};
use windows::core::{Interface, GUID, PCWSTR};
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

pub struct MediaFoundationBackend;

impl MediaBackend for MediaFoundationBackend {
    fn run(
        &self,
        input: &Path,
        output: &Path,
        cutoff: f32,
        codec: Codec,
        dct: Box<dyn DctBackend>,
        tx: Sender<PipelineMsg>,
    ) -> Result<()> {
        run_inner(input, output, cutoff, codec, dct, &tx)
    }
}

/// Which MF video subtype a `Codec` encodes into, and whether to allow
/// (not force) a hardware MFT via `MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS`
/// - mirrors `backends::gstreamer`'s `codec_profile`. Unlike GStreamer's
/// explicit element-name selection, Media Foundation's transcode API picks
/// whichever registered encoder MFT matches the requested subtype; there's
/// no equivalent of "prefer x264enc specifically" - just "software only"
/// vs. "hardware allowed."
struct CodecTarget {
    subtype: GUID,
    hardware: bool,
}

fn codec_target(codec: Codec) -> CodecTarget {
    let subtype = match codec {
        Codec::H264 | Codec::H264Hardware => MFVideoFormat_H264,
        Codec::H265 | Codec::H265Hardware => MFVideoFormat_HEVC,
        // Windows ships an MF *decoder* for VP9 (the "VP9 Video
        // Extensions") but no built-in *encoder* MFT - the same
        // no-VP9-hardware-encode-style gap `backends::gstreamer` already
        // documents for `vavp9enc`. Kept selectable; `AddStream`/
        // `SetInputMediaType` below fail cleanly with an MF error code if
        // nothing registered can encode it, same tolerant-skip shape
        // `tests/integration.rs` already uses for the GStreamer backend.
        Codec::Vp9 | Codec::Vp9Hardware => MFVideoFormat_VP90,
        Codec::Av1 | Codec::Av1Hardware => MFVideoFormat_AV1,
    };
    CodecTarget { subtype, hardware: codec.is_hardware() }
}

fn to_wide_null(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
}

/// Finds the real (0-based) index of the first stream whose major type is
/// `major_type`. `IMFSourceReader::ReadSample`'s `pdwStreamIndex` out-param
/// is always a genuine per-stream index - unlike the
/// `MF_SOURCE_READER_FIRST_*_STREAM` pseudo-constants, which are valid only
/// as *inputs* to methods like `SetCurrentMediaType`/`GetNativeMediaType`
/// ("whichever stream turns out to be the first video/audio one"), never as
/// something to compare a real stream index against.
fn find_stream_index(reader: &IMFSourceReader, major_type: &GUID) -> Result<Option<u32>> {
    let mut i = 0u32;
    loop {
        let native = match unsafe { reader.GetNativeMediaType(i, 0) } {
            Ok(t) => t,
            Err(e) if e.code() == MF_E_INVALIDSTREAMNUMBER => return Ok(None),
            Err(e) => return Err(e).context("failed to enumerate source streams"),
        };
        if unsafe { native.GetGUID(&MF_MT_MAJOR_TYPE) }.ok().as_ref() == Some(major_type) {
            return Ok(Some(i));
        }
        i += 1;
    }
}

/// RAII guards for `CoInitializeEx`/`MFStartup` - both need a matching
/// uninit/shutdown call on every exit path, same reasoning as
/// `backends::gstreamer`'s `pipeline.set_state(Null)` needing to run on
/// every exit path (an early `?` skipping it would leak the COM apartment
/// / MF platform state for the process's remaining lifetime).
struct ComGuard;
impl Drop for ComGuard {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}

struct MfGuard;
impl Drop for MfGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = MFShutdown();
        }
    }
}

/// Deinterleaves a packed, already-top-down BGRA32 frame into three f32
/// planes (dropping alpha - Media Foundation's `MFVideoFormat_RGB32` is
/// always opaque for this use case). `stride` (from `IMF2DBuffer::Lock2D`)
/// can exceed `width*4` for row padding, but is always positive by the
/// time it reaches here - `read_sample_bgra` normalizes the bottom-up-DIB
/// case (negative pitch) into a top-down buffer itself, since only that
/// function has the raw pointer + signed pitch needed to walk rows
/// correctly (see its doc comment). Rows are independent, same
/// rayon-over-rows treatment as `backends::gstreamer`'s `split_rgb_planes`.
fn split_bgra_planes(data: &[u8], width: u32, height: u32, stride: i32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (w, h) = (width as usize, height as usize);
    debug_assert!(stride >= 0, "split_bgra_planes expects an already-normalized top-down buffer");
    let row_bytes = stride as usize;
    let mut r = vec![0f32; w * h];
    let mut g = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    r.par_chunks_mut(w).zip(g.par_chunks_mut(w)).zip(b.par_chunks_mut(w)).enumerate().for_each(|(y, ((r_row, g_row), b_row))| {
        let row = &data[y * row_bytes..y * row_bytes + w * 4];
        for x in 0..w {
            b_row[x] = row[x * 4] as f32;
            g_row[x] = row[x * 4 + 1] as f32;
            r_row[x] = row[x * 4 + 2] as f32;
        }
    });
    (r, g, b)
}

/// Reassembles three f32 planes into a top-down packed BGRA32 buffer
/// (`stride == width*4`, always positive - this is a fresh buffer this
/// backend allocates itself, so there's no inherited orientation/padding
/// to respect the way `split_bgra_planes` must for the source's buffer).
fn join_bgra_planes(width: u32, height: u32, r: &[f32], g: &[f32], b: &[f32]) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_mut(w * 4).take(h).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let o = x * 4;
            row[o] = b[i].round().clamp(0.0, 255.0) as u8;
            row[o + 1] = g[i].round().clamp(0.0, 255.0) as u8;
            row[o + 2] = r[i].round().clamp(0.0, 255.0) as u8;
            row[o + 3] = 255;
        }
    });
    out
}

/// Reads one `IMFSample`'s single buffer out as an already-top-down,
/// positive-stride byte buffer, via `IMF2DBuffer::Lock2D` when the buffer
/// supports it - falls back to `IMFMediaBuffer::Lock`, treating it as
/// tightly packed, for the rare buffer that doesn't implement
/// `IMF2DBuffer`.
///
/// Confirmed directly against Microsoft's own `Lock2D` documentation:
/// `scanline0` always points to the image's *top* row regardless of pitch
/// sign, and row `y`'s address is `scanline0 + y*pitch` using *signed*
/// pointer arithmetic - for a bottom-up DIB (negative pitch), row 1 sits
/// *behind* `scanline0` in memory, not ahead of it. The previous version
/// read `row_bytes * height` bytes forward from `scanline0` unconditionally
/// (wrong memory for negative pitch - not merely flipped, genuinely the
/// wrong bytes) and then applied a manual row-order flip in
/// `split_bgra_planes` on top of that already-wrong read, which is why
/// output came out upside down (and worse, was reading unrelated memory)
/// specifically on frames Media Foundation delivers bottom-up. Walking
/// each row via `scanline0.offset(y as isize * pitch as isize)` handles
/// both signs correctly and needs no downstream flip at all.
///
/// Also takes `height` directly (known by the caller from the negotiated
/// video type) rather than deriving it from `IMFMediaBuffer::
/// GetCurrentLength` - Microsoft's docs for `Lock2D` explicitly note that
/// `GetCurrentLength`/`GetMaxLength` "do not apply to the buffer that is
/// returned by the Lock2D method", so trusting it for the row count was
/// never guaranteed to be reliable.
unsafe fn read_sample_bgra(sample: &IMFSample, width: u32, height: u32) -> Result<(Vec<u8>, i32)> {
    let buffer = sample.ConvertToContiguousBuffer().context("failed to get contiguous sample buffer")?;
    if let Ok(buffer_2d) = buffer.cast::<IMF2DBuffer>() {
        let mut scanline0: *mut u8 = std::ptr::null_mut();
        let mut pitch: i32 = 0;
        buffer_2d.Lock2D(&mut scanline0, &mut pitch).context("IMF2DBuffer::Lock2D failed")?;
        let row_bytes = pitch.unsigned_abs() as usize;
        let mut data = vec![0u8; row_bytes * height as usize];
        for y in 0..height as usize {
            let src_row = scanline0.offset(y as isize * pitch as isize);
            std::ptr::copy_nonoverlapping(src_row, data[y * row_bytes..(y + 1) * row_bytes].as_mut_ptr(), row_bytes);
        }
        let _ = buffer_2d.Unlock2D();
        Ok((data, row_bytes as i32))
    } else {
        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let mut cur_len: u32 = 0;
        buffer.Lock(&mut data_ptr, None, Some(&mut cur_len)).context("IMFMediaBuffer::Lock failed")?;
        let data = std::slice::from_raw_parts(data_ptr, cur_len as usize).to_vec();
        let _ = buffer.Unlock();
        Ok((data, (width * 4) as i32))
    }
}

fn run_inner(
    input_path: &Path,
    output_path: &Path,
    cutoff: f32,
    codec: Codec,
    dct: Box<dyn DctBackend>,
    tx: &Sender<PipelineMsg>,
) -> Result<()> {
    unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.ok().context("failed to initialize COM")?;
    let _com_guard = ComGuard;
    unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL) }.context("failed to initialize Media Foundation")?;
    let _mf_guard = MfGuard;

    let result = encode(input_path, output_path, cutoff, codec, dct, tx);
    if result.is_err() {
        // `MFCreateSinkWriterFromURL` opens (creates/truncates) the output
        // file as part of sink-writer creation, independent of whether the
        // rest of the encode ever succeeds - so a failed encode can leave a
        // zero-byte/partial output file behind unless it's cleaned up here,
        // same reasoning as `backends::gstreamer`'s filesink cleanup.
        let _ = std::fs::remove_file(output_path);
    }
    result
}

fn encode(
    input_path: &Path,
    output_path: &Path,
    cutoff: f32,
    codec: Codec,
    dct: Box<dyn DctBackend>,
    tx: &Sender<PipelineMsg>,
) -> Result<()> {
    info!(path = %input_path.display(), "opening input");
    let _ = tx.send(PipelineMsg::Log("opening input...".into()));

    let target = codec_target(codec);

    // --- source: decode the first video stream to top-down BGRA32 ---
    let in_wide = to_wide_null(input_path);
    let reader = unsafe {
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 1).context("MFCreateAttributes failed")?;
        let attrs = attrs.context("MFCreateAttributes returned no attributes")?;
        // Without this, requesting RGB32 output fails with
        // MF_E_INVALIDMEDIATYPE - confirmed directly (see module doc
        // comment). This is what lets the source reader insert whatever
        // colorspace-convert MFT the native format needs.
        attrs.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1).context("failed to enable video processing")?;
        MFCreateSourceReaderFromURL(PCWSTR(in_wide.as_ptr()), &attrs)
    }
    .with_context(|| format!("failed to open input {}", input_path.display()))?;

    let rgb_type = unsafe {
        let t = MFCreateMediaType().context("failed to create media type")?;
        t.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        t.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
        t
    };
    unsafe { reader.SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, None, &rgb_type) }
        .context("input has no video stream Media Foundation can decode to RGB32")?;

    // `ReadSample`'s `pdwStreamIndex` output is a real stream index, so the
    // main loop below needs the video stream's actual index, not the
    // `MF_SOURCE_READER_FIRST_VIDEO_STREAM` selector used above.
    let video_stream_index =
        find_stream_index(&reader, &MFMediaType_Video)?.context("could not determine the real index of the negotiated video stream")?;

    let video_type = unsafe { reader.GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32) }
        .context("failed to read negotiated video type")?;
    let frame_size = unsafe { video_type.GetUINT64(&MF_MT_FRAME_SIZE) }.context("video type has no frame size")?;
    let (width, height) = ((frame_size >> 32) as u32, (frame_size & 0xFFFF_FFFF) as u32);
    let frame_rate = unsafe { video_type.GetUINT64(&MF_MT_FRAME_RATE) }.unwrap_or((30u64 << 32) | 1);
    let (fps_num, fps_den) = ((frame_rate >> 32) as u32, (frame_rate & 0xFFFF_FFFF).max(1) as u32);
    let fps = fps_num as f64 / fps_den as f64;

    info!(width, height, "decoded input parameters");
    let _ = tx.send(PipelineMsg::Log(format!("{width}x{height} @ {fps:.3} fps, cutoff={cutoff:.3}")));
    if (width as u64) * (height as u64) > 640 * 480 {
        warn!(width, height, "resolution is large for a whole-frame DCT; expect this to be slow");
        let _ = tx.send(PipelineMsg::Log(
            "warning: this resolution is large for a whole-frame DCT (cost grows ~quadratically); expect this to be slow".into(),
        ));
    }

    // Best-effort total-frame estimate for progress reporting, same
    // tolerant "stays 0 (unknown) if unavailable" fallback as
    // `backends::gstreamer`'s duration-query handling.
    let total_frames = unsafe { reader.GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE.0 as u32, &MF_PD_DURATION) }
        .ok()
        .and_then(|pv| duration_100ns(&pv))
        .map(|hns| ((hns as f64 / 10_000_000.0) * fps).round() as u64)
        .unwrap_or(0);

    // --- sink: encode into `codec`, container inferred from the output
    // extension by Media Foundation's own byte-stream-handler resolution
    // (same idea as `backends::gstreamer`'s extension-based muxer choice,
    // but MF only reliably ships handlers for mp4/mov/asf - unlike
    // GStreamer's `matroskamux` fallback, there's no built-in non-mp4
    // container to fall back to here, so a `.mkv` output just fails
    // cleanly with whatever error MF gives for an unresolvable
    // byte-stream handler). ---
    let out_wide = to_wide_null(output_path);
    let sink_attrs = unsafe {
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 1).context("MFCreateAttributes failed")?;
        let attrs = attrs.context("MFCreateAttributes returned no attributes")?;
        attrs
            .SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, target.hardware as u32)
            .context("failed to set hardware-transform preference")?;
        attrs
    };
    let writer = unsafe { MFCreateSinkWriterFromURL(PCWSTR(out_wide.as_ptr()), None, &sink_attrs) }
        .with_context(|| format!("failed to open output {}", output_path.display()))?;

    let out_type = unsafe {
        let t = MFCreateMediaType().context("failed to create media type")?;
        t.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        t.SetGUID(&MF_MT_SUBTYPE, &target.subtype)?;
        t.SetUINT32(&MF_MT_AVG_BITRATE, 4_000_000)?;
        t.SetUINT64(&MF_MT_FRAME_SIZE, frame_size)?;
        t.SetUINT64(&MF_MT_FRAME_RATE, frame_rate)?;
        t.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
        t
    };
    let stream_index = unsafe { writer.AddStream(&out_type) }
        .with_context(|| format!("no encoder registered for {codec} on this system"))?;

    let in_type = unsafe {
        let t = MFCreateMediaType().context("failed to create media type")?;
        t.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        t.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
        t.SetUINT64(&MF_MT_FRAME_SIZE, frame_size)?;
        t.SetUINT64(&MF_MT_FRAME_RATE, frame_rate)?;
        t.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
        t
    };
    unsafe { writer.SetInputMediaType(stream_index, &in_type, None) }.context("failed to negotiate encoder input type")?;

    // Audio passthrough: if the input has an audio stream, add it to the
    // sink writer with its *native* (still-encoded) type - no decode/
    // re-encode - same stream-copy behavior as `backends::gstreamer`'s
    // `autoplug-continue` audio handling.
    let audio_stream_index = setup_audio_passthrough(&reader, &writer).context("failed to set up audio passthrough")?;

    unsafe { writer.BeginWriting() }.context("failed to start writing output")?;

    let frame_duration_100ns = (10_000_000.0 / fps).round() as i64;

    // Decode / GPU DCT / encode run on three threads connected by bounded
    // channels, instead of one serial read->process->write loop - the
    // naive single-threaded version left the GPU idle while Media
    // Foundation blocked on `ReadSample`/`WriteSample` and vice versa
    // (confirmed: GPU utilization capped around 35%, no CPU core
    // saturated, despite the encode already being fast in wall-clock
    // terms). `backends::gstreamer` gets this overlap for free from its
    // pipeline's own `queue` elements; Media Foundation's reader/writer
    // API has no equivalent, so it has to be built by hand here.
    //
    // `reader`/`writer`/`IMFSample` are COM interfaces living in the
    // process's MTA (see `ComGuard` / `COINIT_MULTITHREADED` above) - MTA
    // objects have no thread affinity, only apartment affinity, so moving
    // one to a different OS thread is sound as long as that thread has
    // also joined the MTA before touching it (each worker below does its
    // own `CoInitializeEx`/`ComGuard` pair for exactly that reason).
    // Buffer by a fixed byte budget per channel rather than a fixed frame
    // count - a fixed frame count let buffered memory scale with
    // resolution unbounded (4 full-res f32 RGB frames at 3840x2160 is
    // ~380MB per channel), so pick a frame count that keeps each channel
    // within roughly `CHANNEL_BUDGET_BYTES`, clamped to stay >=2 (needed
    // for any decode/compute/encode overlap at all) and capped at the old
    // depth of 4 (no reason to buffer deeper than that just because a
    // frame happens to be tiny).
    const CHANNEL_BUDGET_BYTES: usize = 64 * 1024 * 1024;
    let frame_bytes = (width as usize) * (height as usize) * 3 * std::mem::size_of::<f32>();
    let channel_depth = (CHANNEL_BUDGET_BYTES / frame_bytes.max(1)).clamp(2, 4);

    /// Marker restricting `MtaSend`'s unsafe `Send` impl to the specific
    /// Media Foundation COM interfaces this module actually moves across
    /// threads, instead of blanket-asserting `Send` for any `T` - a type
    /// that isn't one of these (e.g. something genuinely not thread-safe
    /// reused here by a future refactor) fails to compile instead of
    /// silently getting an incorrect `Send` guarantee.
    trait MfComSend {}
    impl MfComSend for IMFSourceReader {}
    impl MfComSend for IMFSinkWriter {}
    impl MfComSend for IMFSample {}

    /// `windows-rs` COM interface wrappers are not `Send` by default (they
    /// leave thread-affinity judgment to the caller) - this asserts what
    /// the doc comment above already argues: an MTA object has no thread
    /// affinity, so moving one across threads is sound as long as the
    /// receiving thread has also joined the MTA (see the `ComGuard` pairs
    /// in the decode/encode closures below) before touching it.
    struct MtaSend<T: MfComSend>(T);
    unsafe impl<T: MfComSend> Send for MtaSend<T> {}

    // Audio and video both flow through this single decode->compute
    // channel, in the order `ReadSample` produced them, so the compute
    // thread forwards each message to `enc_tx` strictly in that same
    // order (processing video through the GPU, passing audio straight
    // through) - preserving the original single-threaded loop's
    // audio/video write interleave instead of letting audio (which needs
    // no GPU work) race ahead of video sitting in the compute queue.
    enum DecodeMsg {
        Audio(MtaSend<IMFSample>),
        Video { r: Vec<f32>, g: Vec<f32>, b: Vec<f32> },
    }

    enum EncodeMsg {
        Audio(MtaSend<IMFSample>),
        Video { r: Vec<f32>, g: Vec<f32>, b: Vec<f32> },
    }

    let (dec_tx, dec_rx) = std::sync::mpsc::sync_channel::<DecodeMsg>(channel_depth);
    let (enc_tx, enc_rx) = std::sync::mpsc::sync_channel::<EncodeMsg>(channel_depth);

    let decode_reader = MtaSend(reader);
    let decode_handle = std::thread::Builder::new()
        .name("wavefold-mf-decode".into())
        .spawn(move || -> Result<()> {
            unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.ok().context("decode thread: failed to join MTA")?;
            let _com_guard = ComGuard;
            // Forces capture of the whole `MtaSend` wrapper rather than
            // disjoint-capturing just its `.0` field directly (which would
            // bypass the wrapper's `unsafe impl Send` entirely - RFC 2229
            // precise capture only sees the sub-path actually used).
            let decode_reader = decode_reader;
            let reader = decode_reader.0;

            loop {
                let mut flags = 0u32;
                let mut stream_idx = 0u32;
                let mut sample: Option<IMFSample> = None;
                unsafe {
                    reader.ReadSample(
                        MF_SOURCE_READER_ANY_STREAM.0 as u32,
                        0,
                        Some(&mut stream_idx),
                        Some(&mut flags),
                        None,
                        Some(&mut sample),
                    )
                }
                .context("failed to read a sample")?;

                if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 && stream_idx == video_stream_index {
                    break;
                }
                let Some(sample) = sample else { continue };

                if Some(stream_idx) == audio_stream_index {
                    if dec_tx.send(DecodeMsg::Audio(MtaSend(sample))).is_err() {
                        break; // compute thread already gone (errored) - stop feeding it
                    }
                    continue;
                }
                if stream_idx != video_stream_index {
                    continue;
                }

                let (bgra, stride) = unsafe { read_sample_bgra(&sample, width, height) }?;
                let (r, g, b) = split_bgra_planes(&bgra, width, height, stride);
                if dec_tx.send(DecodeMsg::Video { r, g, b }).is_err() {
                    break; // compute thread already gone (errored) - stop feeding it
                }
            }
            Ok(())
        })
        .expect("failed to spawn decode thread");

    let compute_handle = std::thread::Builder::new()
        .name("wavefold-mf-compute".into())
        .spawn(move || -> Result<()> {
            for msg in dec_rx {
                let out = match msg {
                    DecodeMsg::Audio(sample) => EncodeMsg::Audio(sample),
                    DecodeMsg::Video { r, g, b } => {
                        let (r2, g2, b2) = dct.process_rgb(&r, &g, &b, width, height, cutoff)?;
                        EncodeMsg::Video { r: r2, g: g2, b: b2 }
                    }
                };
                if enc_tx.send(out).is_err() {
                    break; // encode thread already gone (errored) - stop feeding it
                }
            }
            Ok(())
        })
        .expect("failed to spawn compute thread");

    let progress_tx = tx.clone();
    let encode_writer = MtaSend(writer);
    let encode_handle = std::thread::Builder::new()
        .name("wavefold-mf-encode".into())
        .spawn(move || -> Result<u64> {
            unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.ok().context("encode thread: failed to join MTA")?;
            let _com_guard = ComGuard;
            let encode_writer = encode_writer;
            let writer = encode_writer.0;

            let mut frame_idx = 0u64;
            let mut pts_100ns = 0i64;
            for msg in enc_rx {
                match msg {
                    EncodeMsg::Audio(sample) => {
                        unsafe { writer.WriteSample(1, &sample.0) }.context("failed to write passthrough audio sample")?;
                    }
                    EncodeMsg::Video { r, g, b } => {
                        let out_bytes = join_bgra_planes(width, height, &r, &g, &b);
                        let out_sample = unsafe {
                            let buffer = MFCreateMemoryBuffer(out_bytes.len() as u32).context("failed to allocate output buffer")?;
                            let mut dest: *mut u8 = std::ptr::null_mut();
                            buffer.Lock(&mut dest, None, None).context("failed to lock output buffer")?;
                            std::ptr::copy_nonoverlapping(out_bytes.as_ptr(), dest, out_bytes.len());
                            buffer.Unlock().context("failed to unlock output buffer")?;
                            buffer.SetCurrentLength(out_bytes.len() as u32).context("failed to set output buffer length")?;

                            let out_sample = MFCreateSample().context("failed to create output sample")?;
                            out_sample.AddBuffer(&buffer).context("failed to attach output buffer")?;
                            out_sample.SetSampleTime(pts_100ns).context("failed to set sample time")?;
                            out_sample.SetSampleDuration(frame_duration_100ns).context("failed to set sample duration")?;
                            out_sample
                        };
                        unsafe { writer.WriteSample(stream_index, &out_sample) }.context("failed to write encoded sample")?;
                        pts_100ns += frame_duration_100ns;

                        frame_idx += 1;
                        let _ = progress_tx.send(PipelineMsg::Progress { current: frame_idx, total: total_frames });
                    }
                }
            }

            unsafe { writer.Finalize() }.context("failed to finalize output")?;
            Ok(frame_idx)
        })
        .expect("failed to spawn encode thread");

    // Join all three unconditionally before propagating any error - `?`
    // on the first `.join()` that comes back `Err` would return from this
    // function (dropping `run_inner`'s `_mf_guard`/`_com_guard`, which
    // call `MFShutdown`/`CoUninitialize`, and triggering its
    // failed-encode `remove_file` cleanup) while the other two threads
    // could still be mid-flight - e.g. the encode thread still inside
    // `writer.Finalize()` on a file `remove_file` is about to delete out
    // from under it. Collecting every `Result` first guarantees all three
    // threads have actually finished by the time any of that runs.
    let decode_result = decode_handle.join().expect("decode thread panicked");
    let compute_result = compute_handle.join().expect("compute thread panicked");
    let encode_result = encode_handle.join().expect("encode thread panicked");

    decode_result?;
    compute_result?;
    let frame_idx = encode_result?;

    info!(frames = frame_idx, path = %output_path.display(), "encode complete");
    let _ = tx.send(PipelineMsg::Log(format!("wrote {frame_idx} frames to {}", output_path.display())));
    let _ = tx.send(PipelineMsg::Done);
    Ok(())
}

/// Extracts a `VT_UI8` `PROPVARIANT` (what `MF_PD_DURATION` is documented
/// to be) as its raw 100ns-tick value, or `None` for anything else -
/// reading the wrong union field of an unrelated variant type would be UB,
/// so the `vt` tag is checked first.
fn duration_100ns(pv: &windows::core::PROPVARIANT) -> Option<i64> {
    const VT_UI8: u16 = 21;
    unsafe {
        let raw = pv.as_raw();
        if raw.Anonymous.Anonymous.vt == VT_UI8 {
            Some(raw.Anonymous.Anonymous.Anonymous.uhVal as i64)
        } else {
            None
        }
    }
}

/// Adds the input's first audio stream (if any) to `writer` with its
/// native, still-encoded media type - pure stream copy, no decode/
/// re-encode - and returns which `IMFSourceReader` stream index it came
/// from so the main read loop can route samples from it straight to
/// `writer` untouched. Returns `Ok(None)` if the input has no audio
/// stream, same "may simply never get used" tolerance as
/// `backends::gstreamer`'s `aud_queue`.
fn setup_audio_passthrough(reader: &IMFSourceReader, writer: &IMFSinkWriter) -> Result<Option<u32>> {
    let Some(audio_stream_index) = find_stream_index(reader, &MFMediaType_Audio)? else {
        return Ok(None); // no audio stream
    };
    let native_type = unsafe { reader.GetNativeMediaType(audio_stream_index, 0) }.context("failed to read native audio type")?;
    unsafe { reader.SetStreamSelection(audio_stream_index, true) }.context("failed to select audio stream")?;
    let audio_out_stream = unsafe { writer.AddStream(&native_type) }.context("failed to add passthrough audio stream to output")?;
    unsafe { writer.SetInputMediaType(audio_out_stream, &native_type, None) }.context("failed to set passthrough audio input type")?;
    if audio_out_stream != 1 {
        return Err(anyhow!("expected passthrough audio to land on sink writer stream 1, got {audio_out_stream}"));
    }
    Ok(Some(audio_stream_index))
}

#[cfg(test)]
mod tests {
    #[test]
    fn errors_on_missing_input() {
        let result = super::run_inner(
            std::path::Path::new("Z:\\nonexistent\\wavefold_test_input.mp4"),
            std::path::Path::new("Z:\\tmp\\wavefold_test_output_never_created.mp4"),
            0.6,
            crate::codec::Codec::H264,
            Box::new(crate::cpu::DctCpu::new()),
            &tokio::sync::mpsc::unbounded_channel().0,
        );
        assert!(result.is_err());
    }
}
