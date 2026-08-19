use ffmpeg_next as ff;

/// GPU-accelerated encode via a libavutil hw device/frames context.
/// `ffmpeg-next` has no safe wrapper for these APIs at all, so
/// `pipeline.rs` drives them directly through `ff::sys` — see the doc
/// comment there for the full setup sequence.
pub struct HwAccel {
    pub device_type: ff::sys::AVHWDeviceType,
    /// What `enc_ctx.set_format(..)` is told (a hw-accel pixel format, e.g.
    /// `Pixel::VAAPI`) — distinct from `EncoderProfile::sw_pixel_format`,
    /// which is the real software format frames are uploaded *from*.
    pub encoder_pixel_format: ff::format::Pixel,
}

/// Concrete per-encoder settings: the libav encoder name to look up (via
/// `ff::encoder::find_by_name`, not a codec-ID lookup — several of these
/// codecs have more than one libav encoder), the pixel format frames are
/// scaled into before encoding, the dictionary options that encoder itself
/// understands, and (for hardware encoders) the hw-accel setup needed
/// before the encoder can be opened. Different encoders take entirely
/// different option keys (`preset`/`bf` are x264/x265-specific; libvpx-vp9
/// and libaom-av1 use their own knobs), so selecting an encoder isn't just
/// a codec-ID swap.
pub struct EncoderProfile {
    pub codec_name: &'static str,
    pub sw_pixel_format: ff::format::Pixel,
    pub options: &'static [(&'static str, &'static str)],
    pub hardware: Option<HwAccel>,
}

/// User-selectable output video encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderChoice {
    H264,
    H265,
    Vp9,
    Av1,
    H264Vaapi,
    H265Vaapi,
    Vp9Vaapi,
    Av1Vaapi,
}

impl EncoderChoice {
    pub const ALL: [EncoderChoice; 8] = [
        EncoderChoice::H264,
        EncoderChoice::H265,
        EncoderChoice::Vp9,
        EncoderChoice::Av1,
        EncoderChoice::H264Vaapi,
        EncoderChoice::H265Vaapi,
        EncoderChoice::Vp9Vaapi,
        EncoderChoice::Av1Vaapi,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            EncoderChoice::H264 => "H.264 (libx264)",
            EncoderChoice::H265 => "H.265 / HEVC (libx265)",
            EncoderChoice::Vp9 => "VP9 (libvpx-vp9)",
            EncoderChoice::Av1 => "AV1 (libaom-av1)",
            EncoderChoice::H264Vaapi => "H.264 (VAAPI hardware)",
            EncoderChoice::H265Vaapi => "H.265 / HEVC (VAAPI hardware)",
            EncoderChoice::Vp9Vaapi => "VP9 (VAAPI hardware)",
            EncoderChoice::Av1Vaapi => "AV1 (VAAPI hardware)",
        }
    }

    pub fn profile(&self) -> EncoderProfile {
        // `bf=0`/`set_max_b_frames(0)` (applied uniformly in pipeline.rs,
        // regardless of which of these is picked) is the mp4-muxer
        // edit-list workaround documented there — forcing it for all eight
        // sidesteps re-litigating which encoders actually need it.
        match self {
            EncoderChoice::H264 => EncoderProfile {
                codec_name: "libx264",
                sw_pixel_format: ff::format::Pixel::YUV420P,
                options: &[("preset", "medium")],
                hardware: None,
            },
            EncoderChoice::H265 => EncoderProfile {
                codec_name: "libx265",
                sw_pixel_format: ff::format::Pixel::YUV420P,
                options: &[("preset", "medium")],
                hardware: None,
            },
            EncoderChoice::Vp9 => EncoderProfile {
                codec_name: "libvpx-vp9",
                sw_pixel_format: ff::format::Pixel::YUV420P,
                options: &[("deadline", "good"), ("cpu-used", "4")],
                hardware: None,
            },
            EncoderChoice::Av1 => EncoderProfile {
                codec_name: "libaom-av1",
                sw_pixel_format: ff::format::Pixel::YUV420P,
                options: &[("cpu-used", "6")],
                hardware: None,
            },
            EncoderChoice::H264Vaapi => vaapi_profile("h264_vaapi"),
            EncoderChoice::H265Vaapi => vaapi_profile("hevc_vaapi"),
            EncoderChoice::Vp9Vaapi => vaapi_profile("vp9_vaapi"),
            EncoderChoice::Av1Vaapi => vaapi_profile("av1_vaapi"),
        }
    }
}

/// All four VAAPI encoders share the same shape: upload NV12 software
/// frames into a VAAPI hw frames context, no extra dictionary options
/// needed for a first cut (defaults are sane on every driver tested).
fn vaapi_profile(codec_name: &'static str) -> EncoderProfile {
    EncoderProfile {
        codec_name,
        sw_pixel_format: ff::format::Pixel::NV12,
        options: &[],
        hardware: Some(HwAccel {
            device_type: ff::sys::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            encoder_pixel_format: ff::format::Pixel::VAAPI,
        }),
    }
}

impl std::fmt::Display for EncoderChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
