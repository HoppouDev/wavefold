/// Concrete per-encoder settings: the GStreamer element factory name (via
/// `gst::ElementFactory::make`, not a codec-ID lookup — several of these
/// codecs have more than one GStreamer encoder element), and that
/// element's own properties (set via `set_property_from_str`, which
/// type-coerces from the string regardless of whether the property is
/// itself a string, enum, or integer). Different encoders take entirely
/// different property sets (`tune`/`speed-preset` for x264/x265;
/// `deadline`/`lag-in-frames` for vp9; `cpu-used`/`lag-in-frames` for av1;
/// no extra properties for the VAAPI variants, whose defaults are already
/// zero-B-frame/low-latency) — this isn't a codec-ID swap.
pub struct EncoderProfile {
    pub element_factory_name: &'static str,
    pub properties: &'static [(&'static str, &'static str)],
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
    pub parser: Option<&'static str>,
    /// Used for the integration tests' tolerant-skip semantics (hardware
    /// availability is environment/driver-dependent) — not consumed by
    /// `pipeline.rs` itself.
    pub hardware: bool,
}

/// User-selectable output video encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
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
            EncoderChoice::H264 => "H.264 (x264enc)",
            EncoderChoice::H265 => "H.265 / HEVC (x265enc)",
            EncoderChoice::Vp9 => "VP9 (vp9enc)",
            EncoderChoice::Av1 => "AV1 (av1enc)",
            EncoderChoice::H264Vaapi => "H.264 (VAAPI hardware)",
            EncoderChoice::H265Vaapi => "H.265 / HEVC (VAAPI hardware)",
            EncoderChoice::Vp9Vaapi => "VP9 (VAAPI hardware)",
            EncoderChoice::Av1Vaapi => "AV1 (VAAPI hardware)",
        }
    }

    pub fn profile(&self) -> EncoderProfile {
        // Every software encoder needs its lookahead/B-frame buffering
        // disabled — frame-in, frame-out — or it can fail to emit a first
        // output packet fast enough for the muxer's `GstAggregator` to
        // complete preroll, stalling the whole pipeline forever (found by
        // spiking this exact issue with x264enc's defaults; the fix
        // differs per encoder, so each one is checked and set explicitly
        // rather than assumed to share x264's property names). VAAPI
        // encoders don't need this: `b-frames` already defaults to 0.
        match self {
            EncoderChoice::H264 => EncoderProfile {
                element_factory_name: "x264enc",
                properties: &[("tune", "zerolatency")],
                parser: Some("h264parse"),
                hardware: false,
            },
            EncoderChoice::H265 => EncoderProfile {
                element_factory_name: "x265enc",
                properties: &[("tune", "zerolatency")],
                parser: Some("h265parse"),
                hardware: false,
            },
            EncoderChoice::Vp9 => EncoderProfile {
                element_factory_name: "vp9enc",
                properties: &[("deadline", "1"), ("lag-in-frames", "0")],
                parser: None,
                hardware: false,
            },
            EncoderChoice::Av1 => EncoderProfile {
                element_factory_name: "av1enc",
                properties: &[("cpu-used", "6"), ("lag-in-frames", "0")],
                parser: None,
                hardware: false,
            },
            EncoderChoice::H264Vaapi => vaapi_profile("vah264enc", Some("h264parse")),
            EncoderChoice::H265Vaapi => vaapi_profile("vah265enc", Some("h265parse")),
            // No `vavp9enc` exists in GStreamer's `va` plugin — confirmed
            // against this machine's real AMD/Mesa VAAPI driver (only
            // `vavp9dec` is registered, no encoder), the same VP9-hw-encode
            // gap this project already hit and tolerated via the previous
            // ffmpeg-next VAAPI path. Kept selectable so the tolerant
            // hardware-failure skip in `tests/integration.rs` still
            // exercises the "expected failure" path; `ElementFactory::make`
            // fails cleanly with `None` rather than panicking.
            EncoderChoice::Vp9Vaapi => vaapi_profile("vavp9enc", None),
            EncoderChoice::Av1Vaapi => vaapi_profile("vaav1enc", None),
        }
    }
}

/// VAAPI encoders' own defaults are already frame-in/frame-out
/// (`b-frames` defaults to 0), so no extra properties are needed.
fn vaapi_profile(element_factory_name: &'static str, parser: Option<&'static str>) -> EncoderProfile {
    EncoderProfile { element_factory_name, properties: &[], parser, hardware: true }
}

impl std::fmt::Display for EncoderChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
