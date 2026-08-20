use crate::codec::Codec;
use crate::dct_backend::DctBackend;
use anyhow::Result;
use std::fmt;
use std::path::Path;
use tokio::sync::mpsc::UnboundedSender as Sender;

#[derive(Debug, Clone)]
pub enum PipelineMsg {
    Progress { current: u64, total: u64 },
    Log(String),
    Done,
    Error(String),
}

/// One decode -> DCT -> encode implementation. `backends::gstreamer` (Linux/
/// macOS) and `backends::media_foundation` (Windows) are the two today, but
/// nothing about this trait is GStreamer- or Media-Foundation-specific: each
/// just needs its own way of decoding `input` to RGB frames, running
/// `dct.process_rgb` over each one, and re-encoding the result into `codec`,
/// with audio passed through untouched.
pub trait MediaBackend: Send + Sync {
    fn run(
        &self,
        input: &Path,
        output: &Path,
        cutoff: f32,
        codec: Codec,
        dct: Box<dyn DctBackend>,
        tx: Sender<PipelineMsg>,
    ) -> Result<()>;
}

/// User-selectable media backend (which decode/encode implementation to
/// use) - distinct from `Codec` (which codec) and `ComputeBackend` (which
/// DCT compute implementation). Same `ALL`/`Display`/`clap::ValueEnum`
/// shape as `Codec` and `ComputeBackend` so the CLI and GUI can pick it up
/// the same way. Platform-gated rather than a free user choice: GStreamer
/// has no system package manager to rely on on Windows, and Media
/// Foundation only exists on Windows, so exactly one variant is compiled
/// in per target - see `backends/mod.rs`. A third backend targeting, say,
/// macOS's AVFoundation would add a third `cfg`-gated variant the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum MediaBackendChoice {
    #[cfg(not(windows))]
    Gstreamer,
    #[cfg(windows)]
    MediaFoundation,
}

impl MediaBackendChoice {
    #[cfg(not(windows))]
    pub const ALL: [MediaBackendChoice; 1] = [MediaBackendChoice::Gstreamer];
    #[cfg(windows)]
    pub const ALL: [MediaBackendChoice; 1] = [MediaBackendChoice::MediaFoundation];

    fn label(self) -> &'static str {
        match self {
            #[cfg(not(windows))]
            MediaBackendChoice::Gstreamer => "GStreamer",
            #[cfg(windows)]
            MediaBackendChoice::MediaFoundation => "Media Foundation",
        }
    }

    pub fn build(self) -> Box<dyn MediaBackend> {
        match self {
            #[cfg(not(windows))]
            MediaBackendChoice::Gstreamer => Box::new(crate::backends::gstreamer::GstreamerBackend),
            #[cfg(windows)]
            MediaBackendChoice::MediaFoundation => Box::new(crate::backends::media_foundation::MediaFoundationBackend),
        }
    }
}

impl fmt::Display for MediaBackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}
