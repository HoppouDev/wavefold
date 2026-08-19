use ffmpeg_next as ff;

/// Concrete per-encoder settings: the libav encoder name to look up (via
/// `ff::encoder::find_by_name`, not a codec-ID lookup — several of these
/// codecs have more than one libav encoder), the pixel format it should
/// encode into, and the dictionary options that encoder itself understands.
/// Different encoders take entirely different option keys (`preset`/`bf`
/// are x264/x265-specific; libvpx-vp9 and libaom-av1 use their own knobs),
/// so selecting an encoder isn't just a codec-ID swap.
pub struct EncoderProfile {
    pub codec_name: &'static str,
    pub pixel_format: ff::format::Pixel,
    pub options: &'static [(&'static str, &'static str)],
}

/// User-selectable output video encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderChoice {
    H264,
    H265,
    Vp9,
    Av1,
}

impl EncoderChoice {
    pub const ALL: [EncoderChoice; 4] = [EncoderChoice::H264, EncoderChoice::H265, EncoderChoice::Vp9, EncoderChoice::Av1];

    pub fn label(&self) -> &'static str {
        match self {
            EncoderChoice::H264 => "H.264 (libx264)",
            EncoderChoice::H265 => "H.265 / HEVC (libx265)",
            EncoderChoice::Vp9 => "VP9 (libvpx-vp9)",
            EncoderChoice::Av1 => "AV1 (libaom-av1)",
        }
    }

    pub fn profile(&self) -> EncoderProfile {
        // `bf=0`/`set_max_b_frames(0)` (applied uniformly in pipeline.rs,
        // regardless of which of these is picked) is the mp4-muxer
        // edit-list workaround documented there — forcing it for all four
        // sidesteps re-litigating which encoders actually need it.
        match self {
            EncoderChoice::H264 => EncoderProfile {
                codec_name: "libx264",
                pixel_format: ff::format::Pixel::YUV420P,
                options: &[("preset", "medium")],
            },
            EncoderChoice::H265 => EncoderProfile {
                codec_name: "libx265",
                pixel_format: ff::format::Pixel::YUV420P,
                options: &[("preset", "medium")],
            },
            EncoderChoice::Vp9 => EncoderProfile {
                codec_name: "libvpx-vp9",
                pixel_format: ff::format::Pixel::YUV420P,
                options: &[("deadline", "good"), ("cpu-used", "4")],
            },
            EncoderChoice::Av1 => EncoderProfile {
                codec_name: "libaom-av1",
                pixel_format: ff::format::Pixel::YUV420P,
                options: &[("cpu-used", "6")],
            },
        }
    }
}

impl std::fmt::Display for EncoderChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
