use crate::codec::Codec;
use crate::dct_backend::DctBackend;
use crate::media_backend::{MediaBackend, PipelineMsg};
use anyhow::{anyhow, Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use gstreamer_video::prelude::*;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::mpsc::UnboundedSender as Sender;
use tracing::{debug, error, info, warn};

pub struct GstreamerBackend;

impl MediaBackend for GstreamerBackend {
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

/// Concrete per-codec GStreamer settings: the element factory name (via
/// `gst::ElementFactory::make`, not a codec-ID lookup — several of these
/// codecs have more than one GStreamer encoder element), and that
/// element's own properties (set via `set_property_from_str`, which
/// type-coerces from the string regardless of whether the property is
/// itself a string, enum, or integer). Different encoders take entirely
/// different property sets (`tune`/`speed-preset` for x264/x265;
/// `deadline`/`lag-in-frames` for vp9; `cpu-used`/`lag-in-frames` for av1;
/// no extra properties for the VAAPI variants, whose defaults are already
/// zero-B-frame/low-latency) — this isn't a codec-ID swap.
struct EncoderProfile {
    element_factory_name: &'static str,
    properties: &'static [(&'static str, &'static str)],
    /// Bitstream parser element (`h264parse`/`h265parse`) to insert between
    /// the encoder and the muxer's request pad — standard GStreamer
    /// practice, and required in practice for H.265: `x265enc`'s raw output
    /// caps don't have a format `qtmux`/`matroskamux`'s video pad template
    /// will accept directly (linking fails with "Pads do not have common
    /// format" otherwise, confirmed). H.264 happened to link into `qtmux`
    /// without one in testing, but `h264parse` is inserted uniformly anyway
    /// — it's the standard pattern (`x264enc ! h264parse ! ...`) and cheap
    /// insurance against the same class of failure on a muxer/version this
    /// wasn't tested against. VP9/AV1 need no parser for muxing.
    parser: Option<&'static str>,
}

/// Every software encoder needs its lookahead/B-frame buffering disabled —
/// frame-in, frame-out — or it can fail to emit a first output packet fast
/// enough for the muxer's `GstAggregator` to complete preroll, stalling the
/// whole pipeline forever (found by spiking this exact issue with
/// x264enc's defaults; the fix differs per encoder, so each one is checked
/// and set explicitly rather than assumed to share x264's property names).
/// VAAPI encoders don't need this: `b-frames` already defaults to 0.
fn codec_profile(codec: Codec) -> EncoderProfile {
    match codec {
        Codec::H264 => {
            EncoderProfile { element_factory_name: "x264enc", properties: &[("tune", "zerolatency")], parser: Some("h264parse") }
        }
        Codec::H265 => {
            EncoderProfile { element_factory_name: "x265enc", properties: &[("tune", "zerolatency")], parser: Some("h265parse") }
        }
        Codec::Vp9 => {
            EncoderProfile { element_factory_name: "vp9enc", properties: &[("deadline", "1"), ("lag-in-frames", "0")], parser: None }
        }
        Codec::Av1 => {
            EncoderProfile { element_factory_name: "av1enc", properties: &[("cpu-used", "6"), ("lag-in-frames", "0")], parser: None }
        }
        Codec::H264Hardware => vaapi_profile("vah264enc", Some("h264parse")),
        Codec::H265Hardware => vaapi_profile("vah265enc", Some("h265parse")),
        // No `vavp9enc` exists in GStreamer's `va` plugin — confirmed
        // against this machine's real AMD/Mesa VAAPI driver (only
        // `vavp9dec` is registered, no encoder), the same VP9-hw-encode
        // gap this project already hit and tolerated via the previous
        // ffmpeg-next VAAPI path. Kept selectable so the tolerant
        // hardware-failure skip in `tests/integration.rs` still exercises
        // the "expected failure" path; `ElementFactory::make` fails
        // cleanly with `None` rather than panicking.
        Codec::Vp9Hardware => vaapi_profile("vavp9enc", None),
        Codec::Av1Hardware => vaapi_profile("vaav1enc", None),
    }
}

/// VAAPI encoders' own defaults are already frame-in/frame-out (`b-frames`
/// defaults to 0), so no extra properties are needed.
fn vaapi_profile(element_factory_name: &'static str, parser: Option<&'static str>) -> EncoderProfile {
    EncoderProfile { element_factory_name, properties: &[], parser }
}

/// Deinterleaves a packed RGB frame's plane 0 into three f32 planes,
/// respecting the frame's stride (linesize may exceed width*3). Rows are
/// independent (disjoint reads of `data`, disjoint writes into r/g/b), so
/// this is parallelized over rows with rayon.
fn split_rgb_planes(frame: &gst_video::VideoFrameRef<&gst::BufferRef>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let stride = frame.plane_stride()[0] as usize;
    let data = frame.plane_data(0).expect("RGB frame has no plane 0");
    let mut r = vec![0f32; w * h];
    let mut g = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    r.par_chunks_mut(w)
        .zip(g.par_chunks_mut(w))
        .zip(b.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((r_row, g_row), b_row))| {
            let row = &data[y * stride..y * stride + w * 3];
            for x in 0..w {
                r_row[x] = row[x * 3] as f32;
                g_row[x] = row[x * 3 + 1] as f32;
                b_row[x] = row[x * 3 + 2] as f32;
            }
        });
    (r, g, b)
}

/// Reassembles three f32 planes back into a packed RGB `gst::Buffer`,
/// respecting the destination frame's own stride (via `VideoFrameRef`, so
/// this stays correct regardless of any padding GStreamer negotiates).
/// Rows are independent, same rayon-over-rows treatment as `split_rgb_planes`.
fn join_rgb_planes(width: u32, height: u32, r: &[f32], g: &[f32], b: &[f32]) -> Result<gst::Buffer> {
    let info = gst_video::VideoInfo::builder(gst_video::VideoFormat::Rgb, width, height)
        .build()
        .context("failed to build output VideoInfo")?;
    let mut buffer = gst::Buffer::with_size(info.size()).context("failed to allocate output buffer")?;
    {
        let buffer_mut = buffer.get_mut().context("output buffer unexpectedly shared")?;
        let mut frame = gst_video::VideoFrameRef::from_buffer_ref_writable(buffer_mut, &info)
            .context("failed to map output frame writable")?;
        let (w, h) = (width as usize, height as usize);
        let stride = frame.plane_stride()[0] as usize;
        let data = frame.plane_data_mut(0).context("output frame has no plane 0")?;
        data.par_chunks_mut(stride).take(h).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                let o = x * 3;
                row[o] = r[i].round().clamp(0.0, 255.0) as u8;
                row[o + 1] = g[i].round().clamp(0.0, 255.0) as u8;
                row[o + 2] = b[i].round().clamp(0.0, 255.0) as u8;
            }
        });
    }
    Ok(buffer)
}

/// One decoded frame's RGB planes plus the original buffer's timing, handed
/// off from the decode side (appsink's callback) to the compute thread.
struct DecodedFrame {
    r: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
    width: u32,
    height: u32,
    pts: Option<gst::ClockTime>,
    dts: Option<gst::ClockTime>,
    duration: Option<gst::ClockTime>,
}

/// One DCT'd frame's RGB planes plus timing, handed off from the compute
/// thread to the encode thread.
struct ComputedFrame {
    r: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
    width: u32,
    height: u32,
    pts: Option<gst::ClockTime>,
    dts: Option<gst::ClockTime>,
    duration: Option<gst::ClockTime>,
}

/// One in-flight frame's dimensions+timing, stashed alongside a GPU frame
/// pipeline slot between `submit_rgb` and its later `finish_rgb` — the DCT
/// backend itself only holds pixel data, so this is what lets the compute
/// thread rebuild a `ComputedFrame` once a slot's result comes back.
struct FrameMeta {
    width: u32,
    height: u32,
    pts: Option<gst::ClockTime>,
    dts: Option<gst::ClockTime>,
    duration: Option<gst::ClockTime>,
}

/// Finishes GPU (or CPU) work for `slot`, already populated by an earlier
/// `submit_rgb`, builds a `ComputedFrame` from `meta`, and forwards it to
/// `enc_tx`. Returns `false` if either the DCT backend failed (already
/// reported onto `pipeline`'s bus, same as `submit_rgb`'s own failure path)
/// or the encode thread's receiver is gone — either way the compute thread
/// must stop.
fn finish_slot(
    dct: &dyn DctBackend,
    slot: usize,
    meta: FrameMeta,
    enc_tx: &SyncSender<EncodeMsg>,
    pipeline: &gst::Pipeline,
) -> bool {
    match dct.finish_rgb(slot) {
        Ok((r, g, b)) => {
            let out = ComputedFrame {
                r,
                g,
                b,
                width: meta.width,
                height: meta.height,
                pts: meta.pts,
                dts: meta.dts,
                duration: meta.duration,
            };
            enc_tx.send(EncodeMsg::Video(out)).is_ok()
        }
        Err(e) => {
            error!("DCT backend failed: {e:#}");
            pipeline.post_error_message(gst::error_msg!(gst::LibraryError::Failed, ["DCT backend failed: {e:#}"]));
            false
        }
    }
}

/// `Eos` is an explicit message rather than relying on channel closure,
/// since appsink's `new_preroll`/`new_sample` closures each hold their own
/// clone of the decode-side sender for as long as the pipeline exists — the
/// channel itself never naturally closes just because `eos` fired. See
/// `run_inner`'s decode/compute/encode threading doc comment for why these
/// three stages are split across threads at all.
enum DecodeMsg {
    Video(DecodedFrame),
    Eos,
}

enum EncodeMsg {
    Video(ComputedFrame),
    Eos,
}

/// Pulls one decoded RGB sample and sends its planes (plus timing) to the
/// compute thread — deliberately *not* running the DCT or pushing into
/// `appsrc` itself, so this callback returns fast and GStreamer's own
/// decode-side threading can keep decoding ahead instead of blocking on GPU
/// compute/encode (see `run_inner`'s threading doc comment).
///
/// NOTE: `appsink` redelivers its very first buffer twice — once via
/// `new_preroll` (to complete the pipeline's PAUSED preroll) and once more
/// via the first `new_sample` call after reaching PLAYING, both for the
/// *same* buffer (confirmed empirically: both calls reported an identical
/// PTS, while every subsequent `new_sample` call advanced normally) — so
/// every one of `tests/integration.rs`'s exact-frame-count assertions was
/// off by exactly one. `last_pts` dedups by skipping a call whose PTS
/// matches the immediately preceding one.
fn decode_and_send(
    sample: &gst::Sample,
    dec_tx: &SyncSender<DecodeMsg>,
    last_pts: &Mutex<Option<gst::ClockTime>>,
) -> Result<(), gst::FlowError> {
    let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
    {
        let mut last_pts = last_pts.lock().expect("last_pts mutex poisoned");
        let pts = buffer.pts();
        if pts.is_some() && *last_pts == pts {
            return Ok(());
        }
        *last_pts = pts;
    }
    let caps = sample.caps().ok_or(gst::FlowError::Error)?;
    let info = gst_video::VideoInfo::from_caps(caps).map_err(|_| gst::FlowError::Error)?;
    let in_frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buffer, &info).map_err(|_| gst::FlowError::Error)?;
    let (width, height) = (in_frame.width(), in_frame.height());

    let (r, g, b) = split_rgb_planes(&in_frame);
    let frame = DecodedFrame { r, g, b, width, height, pts: buffer.pts(), dts: buffer.dts(), duration: buffer.duration() };
    // A closed channel here means the compute thread already stopped
    // (its own error already reached the bus via `post_error_message`
    // below) — mapping to `FlowError::Error` just stops this branch from
    // trying to push more frames into a pipeline that's already winding
    // down, not a second, competing error report.
    dec_tx.send(DecodeMsg::Video(frame)).map_err(|_| gst::FlowError::Error)
}

fn run_inner(
    input_path: &Path,
    output_path: &Path,
    cutoff: f32,
    codec: Codec,
    dct: Box<dyn DctBackend>,
    tx: &Sender<PipelineMsg>,
) -> Result<()> {
    gst::init().context("failed to initialize GStreamer")?;

    info!(path = %input_path.display(), "opening input");
    let _ = tx.send(PipelineMsg::Log("opening input...".into()));

    let input_str = input_path.to_str().context("input path is not valid UTF-8")?;
    let output_str = output_path.to_str().context("output path is not valid UTF-8")?;

    let profile = codec_profile(codec);
    debug!(element = profile.element_factory_name, "video encoder selected");

    let pipeline = gst::Pipeline::new();

    let filesrc = gst::ElementFactory::make("filesrc")
        .property("location", input_str)
        .build()
        .context("failed to create filesrc")?;
    let decodebin = gst::ElementFactory::make("decodebin").build().context("failed to create decodebin")?;
    pipeline.add_many([&filesrc, &decodebin]).context("failed to add decode elements")?;
    filesrc.link(&decodebin).context("failed to link filesrc -> decodebin")?;

    // Encode side: appsrc (DCT'd frames pushed back in) -> videoconvert ->
    // encoder -> queue -> muxer -> filesink. appsrc's caps (width/height/
    // framerate) are set once the decoded video pad's caps are known, in
    // `connect_pad_added` below — appsrc has no upstream to negotiate
    // dimensions from, unlike appsink.
    //
    // Deliberately *no* queue between appsrc and videoconvert (tried once,
    // reverted): `process_and_forward`'s `appsrc.push_buffer()` relies on
    // the encode chain being synchronous to surface downstream failures —
    // e.g. `vp9enc` into `qtmux` (see `codec_profile`'s doc comment on that
    // gap) fails negotiation but never posts its own bus error, so without
    // a queue the `GST_FLOW_NOT_NEGOTIATED` return propagates synchronously
    // back through `push_buffer` into `process_and_forward`'s `Err`, which
    // is what makes that failure visible at all. A queue here decouples
    // that return value from the actual downstream failure, which starts
    // happening asynchronously on the queue's own thread — confirmed this
    // turns that specific failure into a silent hang (no bus error, no
    // flow-return propagation, `rx.blocking_recv()` never unblocks) instead
    // of the clean, fast error `tests/integration.rs` already relies on
    // tolerating for exactly this codec/muxer combination.
    let appsrc = gst_app::AppSrc::builder().format(gst::Format::Time).build();
    let venc_convert = gst::ElementFactory::make("videoconvert").build().context("failed to create videoconvert")?;
    let encoder = gst::ElementFactory::make(profile.element_factory_name)
        .build()
        .with_context(|| format!("encoder '{}' not available in this GStreamer install", profile.element_factory_name))?;
    for (key, value) in profile.properties {
        encoder.set_property_from_str(key, value);
    }
    // Bitstream parser (h264parse/h265parse) between encoder and muxer —
    // see `EncoderProfile::parser`'s doc comment for why this is needed.
    let parser = profile
        .parser
        .map(|name| {
            gst::ElementFactory::make(name)
                .build()
                .with_context(|| format!("parser '{name}' not available in this GStreamer install"))
        })
        .transpose()?;
    let venc_queue = gst::ElementFactory::make("queue").build().context("failed to create encode queue")?;

    // mp4/mov -> qtmux, everything else -> matroskamux, mirroring the old
    // ffmpeg-next path's extension-based muxer choice.
    let is_mp4_family = output_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("mp4") || e.eq_ignore_ascii_case("mov"))
        .unwrap_or(false);
    let muxer = gst::ElementFactory::make(if is_mp4_family { "qtmux" } else { "matroskamux" })
        .build()
        .context("failed to create muxer")?;
    let filesink = gst::ElementFactory::make("filesink")
        .property("location", output_str)
        .build()
        .context("failed to create filesink")?;

    pipeline
        .add_many([appsrc.upcast_ref::<gst::Element>(), &venc_convert, &encoder, &venc_queue, &muxer, &filesink])
        .context("failed to add encode elements")?;
    if let Some(p) = &parser {
        pipeline.add(p).context("failed to add parser")?;
    }
    let mut encode_chain: Vec<&gst::Element> = vec![appsrc.upcast_ref::<gst::Element>(), &venc_convert, &encoder];
    if let Some(p) = &parser {
        encode_chain.push(p);
    }
    encode_chain.push(&venc_queue);
    gst::Element::link_many(encode_chain).context("failed to link encode chain")?;
    let mux_video_sink = muxer.request_pad_simple("video_%u").context("muxer has no video pad template")?;
    venc_queue
        .static_pad("src")
        .context("encode queue has no src pad")?
        .link(&mux_video_sink)
        .context("failed to link encode queue -> muxer")?;
    muxer.link(&filesink).context("failed to link muxer -> filesink")?;

    // Audio passthrough queue: linked lazily once/if decodebin exposes a
    // (still-encoded — see autoplug-continue below) audio pad. Not every
    // input has an audio stream, so this may simply never get used.
    let aud_queue = gst::ElementFactory::make("queue").build().context("failed to create audio queue")?;
    // Decode-side conversion: whatever format the decoder outputs -> RGB,
    // before frames reach appsink. Distinct from `venc_convert` above,
    // which converts DCT'd RGB back to the encoder's format on the way out.
    let dec_convert = gst::ElementFactory::make("videoconvert").build().context("failed to create decode-side videoconvert")?;
    // `dec_queue` gives decoding its own thread, separate from the one
    // appsink's callback (GPU DCT dispatch) runs on - decodebin's video pad
    // links directly into a queue's sink pad below instead of straight into
    // `dec_convert`/`appsink`, so decodebin's internal decode thread can
    // keep decoding ahead instead of blocking on the appsink callback
    // returning. See `venc_pre_queue`'s doc comment above for the matching
    // encode-side half of this fix.
    let dec_queue = gst::ElementFactory::make("queue").build().context("failed to create decode queue")?;
    pipeline.add_many([&aud_queue, &dec_queue, &dec_convert]).context("failed to add audio/decode-convert elements")?;
    dec_queue.link(&dec_convert).context("failed to link decode queue -> videoconvert")?;

    let appsink = gst_app::AppSink::builder()
        .caps(&gst::Caps::builder("video/x-raw").field("format", "RGB").build())
        // `sync` defaults to `true` (inherited from `GstBaseSink`), which
        // paces every pulled sample to the pipeline clock — i.e. throttles
        // the whole decode/compute/encode chain to real-time (1x) playback
        // speed regardless of how fast decode, GPU DCT compute, and encode
        // could actually run. Confirmed: at 512x384 this alone accounted for
        // a ~5.9x slowdown (measured 33.6ms/frame, exactly 1/30fps, whether
        // decode/encode were hardware or software) and pinned GPU
        // utilization near ~20% despite the GPU frame-pipelining fix above —
        // disabling it let the same encode finish in ~5.7ms/frame with GPU
        // utilization around 74-97%. This is an offline batch transcode, not
        // playback, so nothing should be paced to wall-clock time at all.
        .sync(false)
        .build();
    pipeline.add(&appsink).context("failed to add appsink")?;
    dec_convert.link(&appsink).context("failed to link videoconvert -> appsink")?;

    let frame_idx = Arc::new(AtomicU64::new(0));
    let total_frames = Arc::new(AtomicU64::new(0));
    let last_pts: Arc<Mutex<Option<gst::ClockTime>>> = Arc::new(Mutex::new(None));
    let fps_holder: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));

    // Decode (appsink's own callback thread, driven by `dec_queue`) / GPU
    // DCT compute / encode (`appsrc.push_buffer` and everything
    // synchronously downstream of it) run on three separate stages
    // connected by bounded channels, instead of appsink's callback doing
    // all three inline — the naive single-callback version left the GPU
    // idle while waiting on the next decoded frame and vice versa
    // (confirmed: a user report of ~9-15% GPU utilization encoding to AV1
    // hardware despite the DCT compute step being real GPU work, and
    // `dec_queue` alone — decode overlapping with compute+encode, added
    // first as a narrower fix — made no measurable difference, meaning
    // compute and encode were the two stages actually serialized here).
    // `backends::media_foundation` hit the identical problem and fixed it
    // the identical way (see that module's own decode/compute/encode
    // threading doc comment) since Media Foundation's reader/writer API
    // has no `queue`-element equivalent to lean on either.
    //
    // A plain `queue` element between `appsrc` and the encoder (tried
    // once, reverted — see the doc comment above `venc_convert`'s
    // creation) can't be used for the same purpose: it breaks this
    // pipeline's reliance on `appsrc.push_buffer()` synchronously
    // reflecting downstream encode/mux failures. This channel-based
    // design keeps that guarantee by having the *encode* thread call
    // `appsrc.push_buffer()` — still synchronous, just on its own thread
    // now — and explicitly `post_error_message`ing onto the pipeline's own
    // bus on failure, which the existing bus-reading loop below already
    // picks up unchanged, instead of relying on a queue's implicit
    // (and in practice unreliable for at least one codec/muxer gap)
    // error propagation.
    //
    // Bounded (not unbounded) channels so a slow stage applies backpressure
    // instead of buffering the whole video's worth of decoded frames in
    // memory; depth 2 is enough for every adjacent pair of stages to have
    // one frame in flight and one buffered, the minimum for real overlap.
    const CHANNEL_DEPTH: usize = 2;
    let (dec_tx, dec_rx) = sync_channel::<DecodeMsg>(CHANNEL_DEPTH);
    let (enc_tx, enc_rx) = sync_channel::<EncodeMsg>(CHANNEL_DEPTH);

    let compute_pipeline = pipeline.clone();
    let compute_handle = thread::Builder::new()
        .name("wavefold-gst-compute".into())
        .spawn(move || {
            // 2-deep GPU frame pipeline: submit_rgb for the frame landing in
            // a slot happens BEFORE that slot is blocked on via finish_rgb,
            // one frame apart - so the GPU's command queue never goes empty
            // waiting on a CPU-side map_async round trip between frames, the
            // stall still left over after splitting decode/compute/encode
            // into their own threads alone (see docs/gstreamer-notes.md).
            // `pending[slot]` holds whichever frame currently occupies that
            // slot's GPU buffers, tagged with its submission order so EOS
            // draining below can finish both slots back in the right order.
            let mut pending: [Option<(u64, FrameMeta)>; 2] = [None, None];
            let mut frame_counter: u64 = 0;

            for msg in dec_rx {
                let frame = match msg {
                    DecodeMsg::Video(frame) => frame,
                    DecodeMsg::Eos => {
                        // Drain whichever slots still hold a submitted-but-
                        // unfinished frame, oldest submission first, so
                        // frame order into enc_tx (and thus PTS/DTS order
                        // into the encoder) stays correct regardless of
                        // which slot each of the last two frames landed in.
                        let mut order: Vec<usize> = (0..2).filter(|&s| pending[s].is_some()).collect();
                        order.sort_by_key(|&s| pending[s].as_ref().expect("filtered above").0);
                        for slot in order {
                            let (_, meta) = pending[slot].take().expect("filtered above");
                            if !finish_slot(dct.as_ref(), slot, meta, &enc_tx, &compute_pipeline) {
                                return;
                            }
                        }
                        let _ = enc_tx.send(EncodeMsg::Eos);
                        return;
                    }
                };

                let slot = (frame_counter % 2) as usize;
                let meta = FrameMeta { width: frame.width, height: frame.height, pts: frame.pts, dts: frame.dts, duration: frame.duration };

                // The very first frame must flow all the way through to
                // `enc_tx`/`appsrc` synchronously, not deferred - GStreamer's
                // PAUSED->PLAYING transition needs every sink (including
                // `filesink`, downstream of `appsrc`) to receive one preroll
                // buffer before completing, but `appsink` (decode side) only
                // delivers a *second* buffer once the pipeline actually
                // reaches PLAYING. Deferring frame 0's completion until frame
                // 2 arrives - this file's very first version of this
                // pipelining did exactly that - deadlocks: frame 2 can never
                // be decoded because PLAYING never completes because
                // `filesink` never prerolls because frame 0 never reached it.
                // Priming with one synchronous frame breaks that cycle; every
                // frame from here on is genuinely pipelined since the
                // preroll requirement is already satisfied.
                if frame_counter == 0 {
                    if let Err(e) = dct.submit_rgb(slot, &frame.r, &frame.g, &frame.b, frame.width, frame.height, cutoff) {
                        error!("DCT backend failed: {e:#}");
                        compute_pipeline.post_error_message(gst::error_msg!(gst::LibraryError::Failed, ["DCT backend failed: {e:#}"]));
                        return;
                    }
                    if !finish_slot(dct.as_ref(), slot, meta, &enc_tx, &compute_pipeline) {
                        return;
                    }
                    frame_counter += 1;
                    continue;
                }

                // Slot already holds an earlier frame (true once every slot
                // has been primed): finish and forward it first, so its
                // staging buffer is fully unmapped before submit_rgb below
                // reuses this slot's buffers - writing into a still-mapped
                // buffer is a wgpu validation error, not silently wrong
                // output.
                if let Some((_, prev_meta)) = pending[slot].take() {
                    if !finish_slot(dct.as_ref(), slot, prev_meta, &enc_tx, &compute_pipeline) {
                        return;
                    }
                }

                if let Err(e) = dct.submit_rgb(slot, &frame.r, &frame.g, &frame.b, frame.width, frame.height, cutoff) {
                    error!("DCT backend failed: {e:#}");
                    compute_pipeline.post_error_message(gst::error_msg!(gst::LibraryError::Failed, ["DCT backend failed: {e:#}"]));
                    return;
                }
                pending[slot] = Some((frame_counter, meta));
                frame_counter += 1;
            }

            // Defensive fallback: channel closed without an explicit Eos
            // (shouldn't happen - run_inner always sends one, see the
            // `dec_tx_eos` doc comment below - but drain rather than
            // silently drop any still-pending frame if it ever does).
            let mut order: Vec<usize> = (0..2).filter(|&s| pending[s].is_some()).collect();
            order.sort_by_key(|&s| pending[s].as_ref().expect("filtered above").0);
            for slot in order {
                let (_, meta) = pending[slot].take().expect("filtered above");
                if !finish_slot(dct.as_ref(), slot, meta, &enc_tx, &compute_pipeline) {
                    return;
                }
            }
            let _ = enc_tx.send(EncodeMsg::Eos);
        })
        .expect("failed to spawn compute thread");

    let encode_appsrc = appsrc.clone();
    let encode_pipeline = pipeline.clone();
    let encode_frame_idx = frame_idx.clone();
    let encode_total_frames = total_frames.clone();
    let encode_tx = tx.clone();
    let encode_handle = thread::Builder::new()
        .name("wavefold-gst-encode".into())
        .spawn(move || {
            for msg in enc_rx {
                let frame = match msg {
                    EncodeMsg::Video(frame) => frame,
                    EncodeMsg::Eos => {
                        let _ = encode_appsrc.end_of_stream();
                        return;
                    }
                };
                let out_buffer = match join_rgb_planes(frame.width, frame.height, &frame.r, &frame.g, &frame.b) {
                    Ok(mut buf) => {
                        if let Some(buf_mut) = buf.get_mut() {
                            buf_mut.set_pts(frame.pts);
                            buf_mut.set_dts(frame.dts);
                            buf_mut.set_duration(frame.duration);
                        }
                        buf
                    }
                    Err(e) => {
                        error!("failed to assemble output frame: {e:#}");
                        encode_pipeline.post_error_message(gst::error_msg!(gst::LibraryError::Failed, ["failed to assemble output frame: {e:#}"]));
                        return;
                    }
                };
                if let Err(e) = encode_appsrc.push_buffer(out_buffer) {
                    error!("failed to push encoded frame: {e:#}");
                    encode_pipeline.post_error_message(gst::error_msg!(gst::LibraryError::Failed, ["failed to push encoded frame: {e:#}"]));
                    return;
                }
                let current = encode_frame_idx.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = encode_tx.send(PipelineMsg::Progress { current, total: encode_total_frames.load(Ordering::Relaxed) });
            }
        })
        .expect("failed to spawn encode thread");

    // NOTE: appsink delivers its very first buffer via `new_preroll` (while
    // the pipeline is still PAUSED) and every subsequent one via
    // `new_sample` (once PLAYING) — both must forward identically, or the
    // video branch never gets its first buffer and the *whole pipeline's*
    // PAUSED->PLAYING transition stalls forever (the muxer can't complete
    // preroll on a starved pad). appsink's EOS is likewise not automatically
    // forwarded to appsrc and must be done here explicitly (via the compute/
    // encode threads above), or the encode branch never learns decoding
    // finished.
    // `dec_tx` itself (not just a clone) is kept alive in `run_inner`'s own
    // scope rather than moved into the `eos` closure below - the closures
    // registered on `appsink` live as long as `pipeline` does (until this
    // function returns), so if the pipeline never reaches a real EOS (e.g.
    // it fails before ever reaching `Playing`), nothing would otherwise ever
    // signal the compute/encode threads to stop, and joining them below
    // would block forever. This clone is `run_inner`'s explicit fallback:
    // sent unconditionally after the bus loop ends for any reason, so the
    // threads always wind down before this function returns. A second Eos
    // after the real one (the common case) is harmless — the compute thread
    // already returned after the first one, so this later send just fails
    // immediately with a disconnected-receiver error, which is ignored.
    {
        let dec_tx_preroll = dec_tx.clone();
        let dec_tx_sample = dec_tx.clone();
        let dec_tx_eos = dec_tx.clone();
        let last_pts_preroll = last_pts.clone();
        let last_pts_sample = last_pts.clone();
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_preroll(move |sink| {
                    let sample = sink.pull_preroll().map_err(|_| gst::FlowError::Eos)?;
                    decode_and_send(&sample, &dec_tx_preroll, &last_pts_preroll).map(|()| gst::FlowSuccess::Ok)
                })
                .new_sample(move |sink| {
                    let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    decode_and_send(&sample, &dec_tx_sample, &last_pts_sample).map(|()| gst::FlowSuccess::Ok)
                })
                .eos(move |_sink| {
                    let _ = dec_tx_eos.send(DecodeMsg::Eos);
                })
                .build(),
        );
    }

    // decodebin decides whether to keep autoplugging based on this signal:
    // false => stop and expose the pad as-is (passthrough), true => keep
    // looking for a decoder. Video always continues (gets decoded); audio
    // stops the moment it's no longer already raw, so it passes through to
    // the muxer encoded — no decode/re-encode, matching the old
    // ffmpeg-next path's pure stream-copy behavior.
    decodebin.connect("autoplug-continue", false, |values| {
        let caps = values[2].get::<gst::Caps>().expect("autoplug-continue arg 2 is not Caps");
        let name = caps.structure(0).map(|s| s.name()).unwrap_or_default();
        let should_continue = if name.starts_with("audio/") { name == "audio/x-raw" } else { true };
        Some(should_continue.to_value())
    });

    let width_height_logged = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let appsrc_for_pad = appsrc.clone();
        let dec_queue_for_pad = dec_queue.clone();
        let aud_queue_for_pad = aud_queue.clone();
        let muxer_for_pad = muxer.clone();
        // NOTE: a *strong* clone of `pipeline` here would create a
        // reference cycle — this closure is stored inside `decodebin`,
        // which is a child element owned by `pipeline` itself, so
        // `pipeline -> decodebin -> this closure -> pipeline` never drops.
        // That leaked every `tx` clone held by any callback on any element
        // in the pipeline, which meant the `PipelineMsg` channel never
        // closed and every caller's `while let Some(msg) = rx.recv()` loop
        // hung forever even after receiving `Done`/`Error` (confirmed: all
        // 8 integration tests hung identically, regardless of whether the
        // underlying encode would succeed or fail fast). A `WeakRef` breaks
        // the cycle; `.upgrade()` is `None` once the pipeline is really gone.
        let pipeline_weak = pipeline.downgrade();
        let total_frames_for_pad = total_frames.clone();
        let fps_holder_for_pad = fps_holder.clone();
        let tx_for_pad = tx.clone();
        let width_height_logged = width_height_logged.clone();

        decodebin.connect_pad_added(move |_dbin, src_pad| {
            let Some(caps) = src_pad.current_caps() else { return };
            let Some(structure) = caps.structure(0) else { return };
            let name = structure.name();

            if name.starts_with("video/x-raw") {
                let width: i32 = structure.get("width").unwrap_or(0);
                let height: i32 = structure.get("height").unwrap_or(0);
                let framerate: gst::Fraction = structure.get("framerate").unwrap_or(gst::Fraction::new(30, 1));

                if !width_height_logged.swap(true, Ordering::Relaxed) {
                    info!(width, height, "decoded input parameters");
                    let fps = framerate.numer() as f64 / framerate.denom().max(1) as f64;
                    let _ = tx_for_pad.send(PipelineMsg::Log(format!("{width}x{height} @ {fps:.3} fps, cutoff={cutoff:.3}")));
                    if (width as u64) * (height as u64) > 640 * 480 {
                        warn!(width, height, "resolution is large for a whole-frame DCT; expect this to be slow");
                        let _ = tx_for_pad.send(PipelineMsg::Log(
                            "warning: this resolution is large for a whole-frame DCT (cost grows ~quadratically); expect this to be slow".into(),
                        ));
                    }
                    // Total-frame estimate for progress reporting: `fps` is
                    // known now, but duration usually isn't yet (confirmed:
                    // `query_duration` reliably returns `None` this early,
                    // before the demuxer has parsed enough of the stream to
                    // know it) — stash `fps` and let the main bus loop's
                    // `DurationChanged` handler (the actual signal
                    // demuxers use to announce this) do the query once it
                    // fires. Stays at 0 ("unknown") if that never happens,
                    // same tolerant fallback as before.
                    *fps_holder_for_pad.lock().expect("fps_holder mutex poisoned") = Some(fps);
                    if let Some(pipeline) = pipeline_weak.upgrade() {
                        if let Some(dur) = pipeline.query_duration::<gst::ClockTime>() {
                            let secs = dur.seconds() as f64 + (dur.nseconds() % 1_000_000_000) as f64 / 1e9;
                            total_frames_for_pad.store((secs * fps).round() as u64, Ordering::Relaxed);
                        }
                    }
                }

                appsrc_for_pad.set_caps(Some(
                    &gst::Caps::builder("video/x-raw")
                        .field("format", "RGB")
                        .field("width", width)
                        .field("height", height)
                        .field("framerate", framerate)
                        .build(),
                ));

                let sinkpad = dec_queue_for_pad.static_pad("sink").expect("decode queue has no sink pad");
                if let Err(e) = src_pad.link(&sinkpad) {
                    error!("failed to link decoded video pad to decode queue: {e:?}");
                }
            } else if name.starts_with("audio/") {
                debug!("audio stream found: will pass through unchanged");
                let sinkpad = aud_queue_for_pad.static_pad("sink").expect("audio queue has no sink pad");
                if sinkpad.is_linked() {
                    return;
                }
                if let Err(e) = src_pad.link(&sinkpad) {
                    error!("failed to link passthrough audio pad to queue: {e:?}");
                    return;
                }
                let Some(mux_audio_sink) = muxer_for_pad.request_pad_simple("audio_%u") else {
                    error!("muxer has no audio pad template");
                    return;
                };
                let q_src = aud_queue_for_pad.static_pad("src").expect("audio queue has no src pad");
                if let Err(e) = q_src.link(&mux_audio_sink) {
                    error!("failed to link audio queue to muxer: {e:?}");
                }
            }
        });
    }

    if !width_height_logged.load(Ordering::Relaxed) {
        debug!("no audio stream found (or none yet) at pipeline construction time");
    }

    // NOTE: `pipeline.set_state(Null)` must run on *every* exit path, not
    // just the success one — dropping a `gst::Pipeline` that was never
    // cleanly transitioned to `Null` can itself hang during teardown
    // (GStreamer's internal dispose logic forcing a state change while
    // finalizing), which previously meant a `set_state(Playing)` failure
    // (e.g. missing input file) left the pipeline — and everything it
    // owns, including every `tx` clone held by the callbacks registered
    // above — stuck forever, so the channel never closed and callers'
    // `while let Some(msg) = rx.recv()` loops hung even after already
    // receiving the `Error` message. So: collect the Playing-transition
    // result instead of using `?` on it directly, always attempt the Null
    // transition afterward, and only propagate whichever error mattered.
    let mut final_error: Option<anyhow::Error> = None;
    if let Err(e) = pipeline.set_state(gst::State::Playing) {
        final_error = Some(anyhow!("failed to set pipeline to Playing: {e}"));
    } else {
        // One more best-effort duration query, blocking until the async
        // Paused->Playing transition truly completes first: `DurationChanged`
        // isn't reliably posted for every demuxer/container combination
        // (confirmed: never fired for either an mp4 or an mkv test fixture
        // here), but by the time the state change genuinely finishes the
        // demuxer has necessarily parsed far enough to have exposed pads
        // (`fps_holder` set) and, in practice, to answer a duration query.
        let _ = pipeline.state(gst::ClockTime::from_seconds(5));
        if total_frames.load(Ordering::Relaxed) == 0 {
            if let Some(fps) = *fps_holder.lock().expect("fps_holder mutex poisoned") {
                if let Some(dur) = pipeline.query_duration::<gst::ClockTime>() {
                    let secs = dur.seconds() as f64 + (dur.nseconds() % 1_000_000_000) as f64 / 1e9;
                    total_frames.store((secs * fps).round() as u64, Ordering::Relaxed);
                }
            }
        }

        let bus = pipeline.bus().context("pipeline has no bus")?;
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            use gst::MessageView;
            match msg.view() {
                MessageView::Eos(_) => break,
                MessageView::Error(err) => {
                    final_error = Some(anyhow!(
                        "pipeline error from {}: {} ({:?})",
                        err.src().map(|s| s.path_string().to_string()).unwrap_or_default(),
                        err.error(),
                        err.debug()
                    ));
                    break;
                }
                // The actual signal demuxers use to announce a
                // newly-known duration (see the pad-added handler above,
                // where an immediate `query_duration` reliably returns
                // `None` — too early in the PAUSED transition).
                MessageView::DurationChanged(_) => {
                    if let Some(fps) = *fps_holder.lock().expect("fps_holder mutex poisoned") {
                        if let Some(dur) = pipeline.query_duration::<gst::ClockTime>() {
                            let secs = dur.seconds() as f64 + (dur.nseconds() % 1_000_000_000) as f64 / 1e9;
                            total_frames.store((secs * fps).round() as u64, Ordering::Relaxed);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Unconditional fallback so the compute/encode threads always wind down
    // before this function returns, even on a path where the pipeline never
    // reached a real EOS (e.g. `set_state(Playing)` itself failed above) -
    // see the doc comment on `dec_tx_eos`'s clone for why this can't just
    // rely on `appsink`'s own `eos` callback alone.
    let _ = dec_tx.send(DecodeMsg::Eos);
    let _ = compute_handle.join();
    let _ = encode_handle.join();

    let _ = pipeline.set_state(gst::State::Null);

    if let Some(e) = final_error {
        // `filesink` opens (and thus creates/truncates) its output file as
        // part of the pipeline's state transition, independent of whether
        // upstream (e.g. `filesrc` on a missing input) ever actually
        // succeeds — so a failed encode can still leave a zero-byte output
        // file behind unless it's cleaned up here.
        let _ = std::fs::remove_file(output_path);
        return Err(e);
    }

    let final_frame_count = frame_idx.load(Ordering::Relaxed);
    info!(frames = final_frame_count, path = %output_path.display(), "encode complete");
    let _ = tx.send(PipelineMsg::Log(format!("wrote {final_frame_count} frames to {}", output_path.display())));
    let _ = tx.send(PipelineMsg::Done);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn errors_on_missing_input() {
        let result = super::run_inner(
            std::path::Path::new("/nonexistent/wavefold_test_input.mp4"),
            std::path::Path::new("/tmp/wavefold_test_output_never_created.mp4"),
            0.6,
            crate::codec::Codec::H264,
            Box::new(crate::cpu::DctCpu::new()),
            &tokio::sync::mpsc::unbounded_channel().0,
        );
        assert!(result.is_err());
    }
}
