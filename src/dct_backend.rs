use crate::cpu::DctCpu;
use crate::gpu::DctGpu;
use anyhow::{anyhow, bail, Result};

/// One whole-frame separable DCT-II compute implementation, chosen at
/// runtime by `ComputeBackend`. `DctGpu` (wgpu compute) and `DctCpu` (plain
/// Rust + rayon) both implement this so `pipeline.rs` can treat either
/// uniformly after construction. `Send` is a supertrait (both impls
/// already are: `DctGpu` holds a `wgpu::Device`/`Queue`, both `Send`;
/// `DctCpu` holds a `RefCell`, `Send` since its contents are) so a
/// `Box<dyn DctBackend>` can move into the GStreamer appsink callback,
/// which runs on GStreamer's own streaming thread rather than the thread
/// that constructed the pipeline.
pub trait DctBackend: Send {
    fn process_rgb(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)>;

    /// Submits work for `frame_slot` (0 or 1) without blocking on
    /// completion, paired with a later `finish_rgb` on the same slot —
    /// lets a 2-deep frame pipeline (see `DctGpu::submit_rgb`) keep a
    /// backend's work queue non-empty across frames instead of a full
    /// stall-and-resume round trip per frame. Both real implementations
    /// (`DctGpu`, `DctCpu`) override this, since `backends/gstreamer.rs`'s
    /// compute thread calls `submit_rgb`/`finish_rgb` unconditionally on
    /// whatever `Box<dyn DctBackend>` it holds, with no fallback to
    /// `process_rgb` — a future backend that leaves this at its default
    /// no-op would need `finish_rgb`'s default `bail!` below to actually
    /// stop it, since these defaults exist only to keep the trait object
    /// safe to construct, not as a genuine no-pipelining fallback path.
    fn submit_rgb(
        &self,
        _frame_slot: usize,
        _r: &[f32],
        _g: &[f32],
        _b: &[f32],
        _width: u32,
        _height: u32,
        _cutoff: f32,
    ) -> Result<()> {
        Ok(())
    }

    /// Retrieves the result of a prior `submit_rgb` call for `frame_slot`.
    /// No default can do anything useful here (there's no generic state to
    /// fall back on), so this errors unless overridden — `DctGpu` overrides
    /// both methods with a real pipelined implementation; `DctCpu` overrides
    /// both too, eagerly computing in `submit_rgb` and stashing the result,
    /// since CPU has no separate submit/wait phase to actually defer. Any
    /// `DctBackend` impl that doesn't override both methods will fail here
    /// as soon as `backends/gstreamer.rs`'s compute thread calls it — see
    /// `submit_rgb`'s doc comment above.
    fn finish_rgb(&self, _frame_slot: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        bail!("finish_rgb: this DctBackend doesn't implement submit_rgb/finish_rgb pipelining")
    }
}

/// User-selectable DCT compute backend: GPU (wgpu, requires a compatible
/// adapter) or CPU (plain Rust, always available — this is what lets the
/// pipeline run on a GPU-less machine, e.g. a standard GitHub Actions
/// runner).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[cfg_attr(feature = "automation", derive(serde::Serialize, serde::Deserialize))]
pub enum ComputeBackend {
    Gpu,
    Cpu,
}

impl ComputeBackend {
    pub const ALL: [ComputeBackend; 2] = [ComputeBackend::Gpu, ComputeBackend::Cpu];

    pub fn label(&self) -> &'static str {
        match self {
            ComputeBackend::Gpu => "GPU (wgpu compute)",
            ComputeBackend::Cpu => "CPU (software, no GPU required)",
        }
    }

    /// Resolves the choice into a concrete backend. GPU construction can
    /// fail (no compatible adapter); CPU construction never does.
    pub fn build(&self) -> Result<Box<dyn DctBackend>> {
        match self {
            ComputeBackend::Gpu => Ok(Box::new(
                DctGpu::new().map_err(|e| anyhow!("GPU init failed: {e:#}"))?,
            )),
            ComputeBackend::Cpu => Ok(Box::new(DctCpu::new())),
        }
    }
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
