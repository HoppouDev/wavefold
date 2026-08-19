use crate::cpu::DctCpu;
use crate::gpu::DctGpu;
use anyhow::{anyhow, Result};

/// One whole-frame separable DCT-II compute implementation, chosen at
/// runtime by `ComputeBackend`. `DctGpu` (wgpu compute) and `DctCpu` (plain
/// Rust + rayon) both implement this so `pipeline.rs` can treat either
/// uniformly after construction.
pub trait DctBackend {
    fn process_rgb(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)>;
}

/// User-selectable DCT compute backend: GPU (wgpu, requires a compatible
/// adapter) or CPU (plain Rust, always available — this is what lets the
/// pipeline run on a GPU-less machine, e.g. a standard GitHub Actions
/// runner).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
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
            ComputeBackend::Gpu => {
                Ok(Box::new(DctGpu::new().map_err(|e| anyhow!("GPU init failed: {e:#}"))?))
            }
            ComputeBackend::Cpu => Ok(Box::new(DctCpu::new())),
        }
    }
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
