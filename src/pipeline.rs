use crate::encoders::{EncoderChoice, HwAccel};
use crate::gpu::DctGpu;
use anyhow::{anyhow, bail, Context, Result};
use ffmpeg_next as ff;
use ff::format::Pixel;
use ff::frame::Video as VideoFrame;
use ff::media::Type as MediaType;
use ff::software::scaling::{context::Context as Scaler, flag::Flags as ScaleFlags};
use ff::Rescale;
use rayon::prelude::*;
use std::path::Path;
use tokio::sync::mpsc::UnboundedSender as Sender;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
pub enum PipelineMsg {
    Progress { current: u64, total: u64 },
    Log(String),
    Done,
    Error(String),
}

/// One unit of demuxed/decoded work handed from the decode stage to the GPU
/// stage, preserving the original interleave order via the channel's FIFO
/// delivery.
enum WorkItem {
    Video(VideoFrame),
    Audio(ff::Packet),
    Eof,
    Error(String),
}

/// One unit of GPU-processed work handed from the GPU stage to the
/// encode+mux stage. `Video` is already scaled into the encoder's pixel
/// format and PTS-tagged — the encode stage only has to send it.
enum EncodeItem {
    Video(VideoFrame),
    Audio(ff::Packet),
    Eof,
    Error(String),
}

/// Owns the VAAPI hw-frames-context buffer ref for the lifetime of the
/// encode: per-frame `av_hwframe_get_buffer` calls (in `encode_hw_frame`)
/// need it kept alive alongside the encoder itself, which is why this is
/// held by the encode+mux stage rather than dropped right after setup.
///
/// SAFETY: `*mut AVBufferRef` is a plain heap-allocated refcounted handle
/// with no thread-affinity of its own; moving ownership into the
/// encode-stage thread (this type is only ever used from that one thread
/// after construction, never shared/aliased across threads) is sound the
/// same way `ffmpeg-next` itself asserts `Send` for `codec::context::
/// Context`'s raw `AVCodecContext` pointer.
struct HwFramesContext(*mut ff::sys::AVBufferRef);

unsafe impl Send for HwFramesContext {}

impl Drop for HwFramesContext {
    fn drop(&mut self) {
        unsafe { ff::sys::av_buffer_unref(&mut self.0) }
    }
}

/// Creates a VAAPI hw device + hw frames context sized for `width`x`height`
/// software frames in `sw_format`, and attaches it to `enc_ctx.hw_frames_ctx`
/// — must run before the encoder is opened (`avcodec_open2` reads
/// `hw_frames_ctx` during init for hardware encoders). Sequence follows
/// FFmpeg's own `doc/examples/vaapi_encode.c`. `device` is left `NULL` in
/// `av_hwdevice_ctx_create` so libva auto-selects the render node, rather
/// than hardcoding e.g. `/dev/dri/renderD128`.
fn setup_hw_frames_context(
    enc_ctx: &mut ff::codec::encoder::video::Video,
    hw: &HwAccel,
    width: u32,
    height: u32,
    sw_format: Pixel,
) -> Result<HwFramesContext> {
    unsafe {
        let mut hw_device_ctx: *mut ff::sys::AVBufferRef = std::ptr::null_mut();
        let ret = ff::sys::av_hwdevice_ctx_create(
            &mut hw_device_ctx,
            hw.device_type,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        if ret < 0 {
            bail!("failed to create hw device context (no compatible GPU/driver available?): ffmpeg error {ret}");
        }

        let hw_frames_ref = ff::sys::av_hwframe_ctx_alloc(hw_device_ctx);
        ff::sys::av_buffer_unref(&mut hw_device_ctx); // hw_frames_ref holds its own ref to the device now
        if hw_frames_ref.is_null() {
            bail!("failed to allocate hw frames context");
        }

        let frames_ctx = (*hw_frames_ref).data as *mut ff::sys::AVHWFramesContext;
        (*frames_ctx).format = hw.encoder_pixel_format.into();
        (*frames_ctx).sw_format = sw_format.into();
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        (*frames_ctx).initial_pool_size = 4;

        let ret = ff::sys::av_hwframe_ctx_init(hw_frames_ref);
        if ret < 0 {
            let mut owned = hw_frames_ref;
            ff::sys::av_buffer_unref(&mut owned);
            bail!("failed to initialize hw frames context: ffmpeg error {ret}");
        }

        (*enc_ctx.as_mut_ptr()).hw_frames_ctx = ff::sys::av_buffer_ref(hw_frames_ref);

        Ok(HwFramesContext(hw_frames_ref))
    }
}

/// Uploads a software frame into a hw frame via the given context, copying
/// over the PTS, ready to hand to `encoder.send_frame`.
fn encode_hw_frame(hw_frames_ctx: &HwFramesContext, sw_frame: &VideoFrame) -> Result<VideoFrame> {
    let mut hw_frame = VideoFrame::empty();
    unsafe {
        let ret = ff::sys::av_hwframe_get_buffer(hw_frames_ctx.0, hw_frame.as_mut_ptr(), 0);
        if ret < 0 {
            bail!("av_hwframe_get_buffer failed: ffmpeg error {ret}");
        }
        let ret = ff::sys::av_hwframe_transfer_data(hw_frame.as_mut_ptr(), sw_frame.as_ptr(), 0);
        if ret < 0 {
            bail!("av_hwframe_transfer_data failed: ffmpeg error {ret}");
        }
        (*hw_frame.as_mut_ptr()).pts = (*sw_frame.as_ptr()).pts;
    }
    Ok(hw_frame)
}

pub fn run(input: &Path, output: &Path, cutoff: f32, encoder_choice: EncoderChoice, tx: Sender<PipelineMsg>) {
    if let Err(e) = run_inner(input, output, cutoff, encoder_choice, &tx) {
        error!("pipeline failed: {e:#}");
        let _ = tx.send(PipelineMsg::Error(format!("{e:#}")));
    }
}

/// True for the two libav return codes that mean "try again later" (need
/// more input) or "flush complete" — both expected loop-ending conditions
/// from `receive_frame`/`receive_packet`. Anything else is a real decode/
/// encode failure and must not be swallowed.
fn is_retry_or_eof(err: &ff::Error) -> bool {
    matches!(err, ff::Error::Eof) || matches!(err, ff::Error::Other { errno } if *errno == ff::error::EAGAIN)
}

/// Drains all packets an encoder currently has buffered (or flushes on EOF)
/// and writes them to the output container.
fn drain_encoder(
    encoder: &mut ff::encoder::Video,
    octx: &mut ff::format::context::Output,
    stream_index: usize,
    ost_time_base: ff::Rational,
) -> Result<()> {
    let mut packet = ff::Packet::empty();
    loop {
        match encoder.receive_packet(&mut packet) {
            Ok(()) => {
                packet.set_stream(stream_index);
                packet.rescale_ts(encoder.time_base(), ost_time_base);
                packet.write_interleaved(octx)?;
            }
            Err(e) if is_retry_or_eof(&e) => break,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Deinterleaves a packed RGB24 frame's plane 0 into three f32 planes,
/// respecting the frame's stride (linesize may exceed width*3). Rows are
/// independent (disjoint reads of `data`, disjoint writes into r/g/b), so
/// this is parallelized over rows with rayon.
fn split_rgb_planes(frame: &VideoFrame, width: u32, height: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (w, h) = (width as usize, height as usize);
    let stride = frame.stride(0);
    let data = frame.data(0);
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

/// Reassembles three f32 planes back into a packed RGB24 frame, respecting
/// the destination frame's own stride. Rows are independent, same
/// rayon-over-rows treatment as `split_rgb_planes`.
fn join_rgb_planes(width: u32, height: u32, r: &[f32], g: &[f32], b: &[f32]) -> VideoFrame {
    let (w, h) = (width as usize, height as usize);
    let mut frame = VideoFrame::new(Pixel::RGB24, width, height);
    let stride = frame.stride(0);
    let data = frame.data_mut(0);
    data.par_chunks_mut(stride).take(h).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let o = x * 3;
            row[o] = r[i].round().clamp(0.0, 255.0) as u8;
            row[o + 1] = g[i].round().clamp(0.0, 255.0) as u8;
            row[o + 2] = b[i].round().clamp(0.0, 255.0) as u8;
        }
    });
    frame
}

fn run_inner(
    input_path: &Path,
    output_path: &Path,
    cutoff: f32,
    encoder_choice: EncoderChoice,
    tx: &Sender<PipelineMsg>,
) -> Result<()> {
    ff::init()?;

    info!(path = %input_path.display(), "opening input");
    let _ = tx.send(PipelineMsg::Log("opening input...".into()));
    let mut ictx = ff::format::input(input_path)?;

    let video_stream_index = ictx
        .streams()
        .best(MediaType::Video)
        .ok_or_else(|| anyhow!("no video stream found in input"))?
        .index();
    let audio_stream_index = ictx.streams().best(MediaType::Audio).map(|s| s.index());

    let stream = ictx.stream(video_stream_index).context("video stream disappeared after being located")?;
    let time_base = stream.time_base();
    let raw_frame_rate = stream.avg_frame_rate();
    // Fall back to a sane 30fps when the container doesn't report a usable
    // rate (0/0 is common for VFR sources, raw streams, incomplete
    // metadata); used for both the logged fps and the encoder's time base
    // /frame rate below, so a degenerate rate can't leak into either.
    let frame_rate = if raw_frame_rate.denominator() != 0 && raw_frame_rate.numerator() != 0 {
        raw_frame_rate
    } else {
        ff::Rational::new(30, 1)
    };
    let fps: f64 = frame_rate.numerator() as f64 / frame_rate.denominator() as f64;
    let nb_frames = stream.frames().max(0) as u64;
    let duration_secs = if stream.duration() > 0 {
        stream.duration() as f64 * f64::from(time_base)
    } else {
        0.0
    };
    let total_frames = if nb_frames > 0 {
        nb_frames
    } else if duration_secs > 0.0 {
        (duration_secs * fps).round() as u64
    } else {
        0
    };

    let context_decoder = ff::codec::context::Context::from_parameters(stream.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;
    let width = decoder.width();
    let height = decoder.height();
    if width == 0 || height == 0 {
        bail!("could not read video dimensions (unsupported file?)");
    }

    info!(width, height, fps, cutoff, "decoded input parameters");
    let _ = tx.send(PipelineMsg::Log(format!(
        "{width}x{height} @ {fps:.3} fps, cutoff={cutoff:.3}"
    )));
    // The whole-frame DCT is a naive O(width) / O(height) sum per output
    // pixel (not a fast O(N log N) transform), so cost grows steeply with
    // resolution — warn rather than let it look like a hang.
    if (width as u64) * (height as u64) > 640 * 480 {
        warn!(width, height, "resolution is large for a whole-frame DCT; expect this to be slow");
        let _ = tx.send(PipelineMsg::Log(
            "warning: this resolution is large for a whole-frame DCT (cost grows ~quadratically); expect this to be slow".into(),
        ));
    }
    if audio_stream_index.is_some() {
        debug!("audio stream found: will pass through unchanged");
        let _ = tx.send(PipelineMsg::Log("audio stream found: will pass through unchanged".into()));
    } else {
        debug!("no audio stream found");
        let _ = tx.send(PipelineMsg::Log("no audio stream found".into()));
    }
    info!("initializing GPU DCT pipeline");
    let _ = tx.send(PipelineMsg::Log("initializing GPU DCT pipeline...".into()));
    let gpu = DctGpu::new().map_err(|e| anyhow!("GPU init failed: {e:#}"))?;

    let mut to_rgb = Scaler::get(
        decoder.format(),
        width,
        height,
        Pixel::RGB24,
        width,
        height,
        ScaleFlags::BILINEAR,
    )?;

    let mut octx = ff::format::output(output_path)?;
    let profile = encoder_choice.profile();
    let codec = ff::encoder::find_by_name(profile.codec_name)
        .ok_or_else(|| anyhow!("encoder '{}' not available in this ffmpeg build", profile.codec_name))?;
    debug!(codec = codec.name(), "video encoder selected");

    let enc_time_base = ff::Rational::new(frame_rate.denominator(), frame_rate.numerator().max(1));

    let mut enc_ctx = ff::codec::context::Context::new_with_codec(codec).encoder().video()?;
    enc_ctx.set_width(width);
    enc_ctx.set_height(height);
    enc_ctx.set_format(profile.hardware.as_ref().map_or(profile.sw_pixel_format, |hw| hw.encoder_pixel_format));
    enc_ctx.set_time_base(enc_time_base);
    enc_ctx.set_frame_rate(Some(frame_rate));
    // No B-frames, applied uniformly regardless of which encoder was picked:
    // frames are processed independently by the DCT pass anyway, and
    // B-frame reordering was producing a negative initial DTS that the mp4
    // muxer silently dropped on the first packet. AVCodecContext.max_b_frames
    // must agree with the encoder-specific option below, or the mp4 muxer
    // computes the wrong reorder delay and writes an edit list that
    // silently trims frames.
    enc_ctx.set_max_b_frames(0);
    if octx.format().flags().contains(ff::format::Flags::GLOBAL_HEADER) {
        enc_ctx.set_flags(ff::codec::Flags::GLOBAL_HEADER);
    }

    // Must happen before `open_as_with` below: `avcodec_open2` reads
    // `hw_frames_ctx` during hardware-encoder init.
    let hw_frames_ctx = match &profile.hardware {
        Some(hw) => {
            debug!(device_type = ?hw.device_type, "setting up hw frames context");
            Some(setup_hw_frames_context(&mut enc_ctx, hw, width, height, profile.sw_pixel_format)?)
        }
        None => None,
    };

    let mut encoder_opts = ff::Dictionary::new();
    for (key, value) in profile.options {
        encoder_opts.set(key, value);
    }
    encoder_opts.set("bf", "0");

    let mut encoder = enc_ctx.open_as_with(codec, encoder_opts)?;

    let mut vost = octx.add_stream(codec)?;
    vost.set_parameters(&encoder);
    let video_ost_index = vost.index();
    drop(vost);

    // audio: pure stream copy, no decode/re-encode.
    let audio_ost_index = if let Some(a_idx) = audio_stream_index {
        let aist = ictx.stream(a_idx).context("audio input stream disappeared after being located")?;
        let mut aost = octx.add_stream(ff::encoder::find(ff::codec::Id::None))?;
        aost.set_parameters(aist.parameters());
        unsafe {
            let params = aost.parameters().as_mut_ptr();
            // Zero the copied codec_tag: it's valid in the input's container
            // but may not be recognized by the output muxer, causing strict
            // players to misidentify or reject the passthrough track.
            (*params).codec_tag = 0;
            // Some containers (e.g. mkv from certain camera/phone encoders)
            // don't record an explicit channel layout, leaving it
            // AV_CHANNEL_ORDER_UNSPEC after a pure parameter copy. The mp4
            // muxer can't write a channel-layout box for an unspecified
            // layout and fails the whole encode with "unsupported channel
            // layout N channels" — fill in the standard layout for that
            // channel count (e.g. stereo for 2 channels) so passthrough
            // into mp4 always has something the muxer can write.
            if (*params).ch_layout.order == ff::sys::AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC {
                ff::sys::av_channel_layout_default(&mut (*params).ch_layout, (*params).ch_layout.nb_channels);
            }
        }
        Some(aost.index())
    } else {
        None
    };

    // Explicitly disable the mov muxer's auto edit-list: with an empty/auto
    // edit list it was mis-computing the segment duration one frame short,
    // causing players (including ffmpeg's own decoder) to silently drop the
    // last encoded frame on playback. `use_editlist` is a mov/mp4-only,
    // file-wide (not per-track) option, so only apply it when the resolved
    // muxer is actually mov/mp4. Verified this must apply regardless of
    // whether an audio track is present: leaving edit lists on "auto" for
    // an audio+video mux still drops the last video frame the same way, so
    // guaranteed video frame count wins over a possible audio-priming
    // edit-list gap on the passed-through track.
    let is_mp4_family = octx.format().name().split(',').any(|n| n == "mp4" || n == "mov");
    if is_mp4_family {
        let mut mov_opts = ff::Dictionary::new();
        mov_opts.set("use_editlist", "0");
        octx.write_header_with(mov_opts)?;
    } else {
        octx.write_header()?;
    }
    let ost_time_base = octx.stream(video_ost_index).context("video output stream disappeared after being added")?.time_base();

    let mut from_rgb = Scaler::get(
        Pixel::RGB24,
        width,
        height,
        profile.sw_pixel_format,
        width,
        height,
        ScaleFlags::BILINEAR,
    )?;

    // Three-stage pipeline — decode / GPU DCT / encode+mux — connected by
    // two bounded channels, so each stage's work overlaps with the others
    // instead of serializing on one thread (which is why the GPU previously
    // sat idle during scale/encode, and the encoder sat idle during the GPU
    // wait). Decode and encode+mux each get their own spawned thread below;
    // *this* calling thread stays as the GPU stage in the middle, because
    // `libswscale`'s `Context` (`to_rgb`/`from_rgb`) is not `Send` and so
    // can never move into a spawned closure — `encoder`/`octx` (needed
    // downstream) are `Send`, so it's the encode+mux stage that gets
    // spawned off instead, not the GPU stage. Both channels' FIFO order
    // preserves the original demux interleave for video and audio, so no
    // separate reordering is needed.
    //
    // `ictx`/`decoder` are only moved into the decode-stage closure below;
    // `encoder`/`octx` only into the encode-stage closure further down.
    // Neither is used again on this (GPU-stage) thread afterward.
    let audio_in_time_base = match audio_stream_index {
        Some(idx) => Some(ictx.stream(idx).context("audio input stream disappeared after being located")?.time_base()),
        None => None,
    };
    let a_ost_time_base = match audio_ost_index {
        Some(idx) => Some(octx.stream(idx).context("audio output stream disappeared after being added")?.time_base()),
        None => None,
    };

    let (work_tx, work_rx) = std::sync::mpsc::sync_channel::<WorkItem>(2);
    let producer = std::thread::spawn(move || -> Result<()> {
        let result = (|| -> Result<()> {
            for (stream_in, mut packet) in ictx.packets() {
                let in_index = stream_in.index();
                if in_index == video_stream_index {
                    decoder.send_packet(&packet)?;
                    let mut decoded = VideoFrame::empty();
                    loop {
                        match decoder.receive_frame(&mut decoded) {
                            Ok(()) => {
                                if work_tx.send(WorkItem::Video(decoded)).is_err() {
                                    return Ok(()); // consumer already stopped (errored)
                                }
                                decoded = VideoFrame::empty();
                            }
                            Err(e) if is_retry_or_eof(&e) => break,
                            Err(e) => return Err(e.into()),
                        }
                    }
                } else if Some(in_index) == audio_stream_index {
                    if let (Some(a_ost_index), Some(in_tb), Some(out_tb)) =
                        (audio_ost_index, audio_in_time_base, a_ost_time_base)
                    {
                        packet.rescale_ts(in_tb, out_tb);
                        packet.set_stream(a_ost_index);
                        if work_tx.send(WorkItem::Audio(packet)).is_err() {
                            return Ok(());
                        }
                    }
                }
            }

            decoder.send_eof()?;
            let mut decoded = VideoFrame::empty();
            loop {
                match decoder.receive_frame(&mut decoded) {
                    Ok(()) => {
                        if work_tx.send(WorkItem::Video(decoded)).is_err() {
                            return Ok(());
                        }
                        decoded = VideoFrame::empty();
                    }
                    Err(e) if is_retry_or_eof(&e) => break,
                    Err(e) => return Err(e.into()),
                }
            }
            Ok(())
        })();

        match &result {
            Ok(()) => {
                let _ = work_tx.send(WorkItem::Eof);
            }
            Err(e) => {
                let _ = work_tx.send(WorkItem::Error(format!("{e:#}")));
            }
        }
        result
    });

    // Encode+mux stage runs on a *new* spawned thread instead of this one:
    // `encoder`/`octx` are `Send` (unlike the `Scaler`s above), so they're
    // what moves. This stage owns everything downstream of GPU processing —
    // including the final `send_eof`/`write_trailer` — and reports the
    // final frame count back through its `JoinHandle` once this (GPU-stage)
    // thread joins it below.
    let (encode_tx, encode_rx) = std::sync::mpsc::sync_channel::<EncodeItem>(2);
    let encode_stage = std::thread::spawn(move || -> Result<i64> {
        let mut frame_idx: i64 = 0;
        loop {
            match encode_rx.recv() {
                Ok(EncodeItem::Video(yuv)) => {
                    match &hw_frames_ctx {
                        Some(hw) => {
                            let hw_frame = encode_hw_frame(hw, &yuv)?;
                            encoder.send_frame(&hw_frame)?;
                        }
                        None => encoder.send_frame(&yuv)?,
                    }
                    drain_encoder(&mut encoder, &mut octx, video_ost_index, ost_time_base)?;
                    frame_idx += 1;
                }
                Ok(EncodeItem::Audio(packet)) => {
                    packet.write_interleaved(&mut octx)?;
                }
                Ok(EncodeItem::Eof) => break,
                Ok(EncodeItem::Error(msg)) => bail!("{msg}"),
                Err(_) => bail!("GPU worker thread ended unexpectedly"),
            }
        }
        encoder.send_eof()?;
        drain_encoder(&mut encoder, &mut octx, video_ost_index, ost_time_base)?;
        octx.write_trailer()?;
        Ok(frame_idx)
    });

    // GPU stage: this (calling) thread. Scale to RGB24, run the GPU DCT,
    // reassemble, scale to the encoder's pixel format, tag the PTS — stays
    // here (rather than moving to a spawned thread) specifically because
    // `to_rgb`/`from_rgb`/`gpu` can't cross a thread boundary.
    let mut frame_idx: i64 = 0;
    let gpu_result = (|| -> Result<()> {
        loop {
            match work_rx.recv() {
                Ok(WorkItem::Video(decoded)) => {
                    let mut rgb = VideoFrame::empty();
                    to_rgb.run(&decoded, &mut rgb)?;

                    let (r, g, b) = split_rgb_planes(&rgb, width, height);
                    let (r2, g2, b2) = gpu.process_rgb(&r, &g, &b, width, height, cutoff)?;
                    let rgb_out = join_rgb_planes(width, height, &r2, &g2, &b2);

                    let mut yuv = VideoFrame::empty();
                    from_rgb.run(&rgb_out, &mut yuv)?;
                    // Keep the decoded frame's own timeline (rescaled into
                    // the encoder's time base) instead of a synthetic
                    // zero-based counter, so video stays aligned with the
                    // audio passthrough's original timestamps (e.g. inputs
                    // with a nonzero stream start_time). `frame_idx` is
                    // only a fallback for frames with no pts, and is
                    // already expressed directly in enc_time_base units (1
                    // tick = 1 frame), so unlike a real pts it must NOT be
                    // rescaled from the input's time_base.
                    let source_pts = match decoded.pts() {
                        Some(pts) => pts.rescale(time_base, enc_time_base),
                        None => frame_idx,
                    };
                    yuv.set_pts(Some(source_pts));
                    frame_idx += 1;

                    let _ = tx.send(PipelineMsg::Progress { current: frame_idx as u64, total: total_frames });

                    if encode_tx.send(EncodeItem::Video(yuv)).is_err() {
                        return Ok(()); // encode stage already stopped (errored)
                    }
                }
                Ok(WorkItem::Audio(packet)) => {
                    if encode_tx.send(EncodeItem::Audio(packet)).is_err() {
                        return Ok(());
                    }
                }
                Ok(WorkItem::Eof) => return Ok(()),
                Ok(WorkItem::Error(msg)) => bail!("{msg}"),
                Err(_) => bail!("decode worker thread ended unexpectedly"),
            }
        }
    })();

    match &gpu_result {
        Ok(()) => {
            let _ = encode_tx.send(EncodeItem::Eof);
        }
        Err(e) => {
            let _ = encode_tx.send(EncodeItem::Error(format!("{e:#}")));
        }
    }

    // Reaching here means `encode_stage` finished cleanly (it only ever
    // receives `EncodeItem::Eof`, never `Error`, when `gpu_result` is
    // `Ok`), so `gpu_result` is already known-`Ok` — nothing left to check.
    let final_frame_count = match encode_stage.join() {
        Ok(Ok(n)) => n,
        Ok(Err(e)) => return Err(e),
        Err(_) => bail!("encode worker thread panicked"),
    };
    match producer.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => bail!("decode worker thread panicked"),
    }

    info!(frames = final_frame_count, path = %output_path.display(), "encode complete");
    let _ = tx.send(PipelineMsg::Log(format!(
        "wrote {} frames to {}",
        final_frame_count,
        output_path.display()
    )));
    let _ = tx.send(PipelineMsg::Done);
    Ok(())
}

// Fixture-generating tests (spawning ffmpeg to build a synthetic clip, then
// running the pipeline against it) live in tests/integration.rs so there's
// one shared copy of that helper instead of two drifting implementations.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_on_missing_input() {
        let result = ff::format::input(Path::new("/nonexistent/dctenc_test_input.mp4"));
        assert!(result.is_err());
    }
}
