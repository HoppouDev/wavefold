use std::fmt;

/// Which video codec to encode into, and whether to prefer a hardware
/// encoder. Backend-agnostic: a `MediaBackend` maps a `Codec` onto
/// whatever it needs internally (the GStreamer backend maps it onto a
/// GStreamer element name; the Media Foundation backend maps it onto an
/// `MFVideoFormat_*` subtype).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Codec {
    H264,
    H264Hardware,
    H265,
    H265Hardware,
    Vp9,
    Vp9Hardware,
    Av1,
    Av1Hardware,
}

impl Codec {
    pub const ALL: [Codec; 8] = [
        Codec::H264,
        Codec::H264Hardware,
        Codec::H265,
        Codec::H265Hardware,
        Codec::Vp9,
        Codec::Vp9Hardware,
        Codec::Av1,
        Codec::Av1Hardware,
    ];

    pub fn is_hardware(self) -> bool {
        match self {
            Codec::H264 | Codec::H265 | Codec::Vp9 | Codec::Av1 => false,
            Codec::H264Hardware | Codec::H265Hardware | Codec::Vp9Hardware | Codec::Av1Hardware => true,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Codec::H264 => "H.264",
            Codec::H264Hardware => "H.264 (hardware)",
            Codec::H265 => "H.265 / HEVC",
            Codec::H265Hardware => "H.265 / HEVC (hardware)",
            Codec::Vp9 => "VP9",
            Codec::Vp9Hardware => "VP9 (hardware)",
            Codec::Av1 => "AV1",
            Codec::Av1Hardware => "AV1 (hardware)",
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}
