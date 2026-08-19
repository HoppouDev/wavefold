use crate::encoders::EncoderChoice;
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
use std::sync::mpsc::Sender;
use tracing::{debug, error, info, warn};

pub enum PipelineMsg {
    Progress { current: u64, total: u64 },
    Log(String),
    Done,
    Error(String),
}

/// One unit of demuxed/decoded work handed from the producer thread (demux +
/// decode) to the consumer (scale + GPU DCT + encode + mux), preserving the
/// original interleave order via the channel's FIFO delivery.
enum WorkItem {
    Video(VideoFrame),
    Audio(ff::Packet),
    Eof,
    Error(String),
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
    enc_ctx.set_format(profile.pixel_format);
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
        // Zero the copied codec_tag: it's valid in the input's container but
        // may not be recognized by the output muxer, causing strict players
        // to misidentify or reject the passthrough track.
        unsafe {
            (*aost.parameters().as_mut_ptr()).codec_tag = 0;
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
        profile.pixel_format,
        width,
        height,
        ScaleFlags::BILINEAR,
    )?;

    // Producer/consumer split: demuxing and decoding (stateful, CPU-bound
    // libavcodec work) run on a dedicated thread and feed a bounded channel,
    // so decode of frame N+1 can overlap with this thread's scale + GPU DCT
    // wait + encode of frame N instead of the two serializing on one thread
    // (which is why the GPU previously sat idle during every scale/encode
    // step). `libswscale`'s `Context` (`to_rgb`/`from_rgb`) is not `Send`,
    // so scaling stays here on the consumer side; the channel's FIFO order
    // preserves the original demux interleave for both video and audio, so
    // no separate reordering step is needed.
    //
    // `ictx`/`decoder` are only moved into the producer closure below, not
    // used again on this thread afterward.
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

    let mut frame_idx: i64 = 0;
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
                // Keep the decoded frame's own timeline (rescaled into the
                // encoder's time base) instead of a synthetic zero-based
                // counter, so video stays aligned with the audio
                // passthrough's original timestamps (e.g. inputs with a
                // nonzero stream start_time). `frame_idx` is only a
                // fallback for frames with no pts, and is already expressed
                // directly in enc_time_base units (1 tick = 1 frame), so
                // unlike a real pts it must NOT be rescaled from the
                // input's time_base.
                let source_pts = match decoded.pts() {
                    Some(pts) => pts.rescale(time_base, enc_time_base),
                    None => frame_idx,
                };
                yuv.set_pts(Some(source_pts));
                frame_idx += 1;

                encoder.send_frame(&yuv)?;
                drain_encoder(&mut encoder, &mut octx, video_ost_index, ost_time_base)?;

                let _ = tx.send(PipelineMsg::Progress { current: frame_idx as u64, total: total_frames });
            }
            Ok(WorkItem::Audio(packet)) => {
                packet.write_interleaved(&mut octx)?;
            }
            Ok(WorkItem::Eof) => break,
            Ok(WorkItem::Error(msg)) => bail!("{msg}"),
            Err(_) => bail!("decode worker thread ended unexpectedly"),
        }
    }

    match producer.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => bail!("decode worker thread panicked"),
    }

    encoder.send_eof()?;
    drain_encoder(&mut encoder, &mut octx, video_ost_index, ost_time_base)?;
    octx.write_trailer()?;

    info!(frames = frame_idx, path = %output_path.display(), "encode complete");
    let _ = tx.send(PipelineMsg::Log(format!(
        "wrote {} frames to {}",
        frame_idx,
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
