use crate::dct_backend::DctBackend;
use crate::dct_math::{dct_basis, transpose_square};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::cell::RefCell;

/// Cached basis matrices for one (width, height) combination, mirroring
/// `gpu.rs`'s `PlaneBuffers` cache shape — rebuilt only when dimensions
/// change, since the basis matrices don't depend on `cutoff`.
struct Basis {
    width: u32,
    height: u32,
    row_basis: Vec<f32>,
    row_basis_t: Vec<f32>,
    col_basis: Vec<f32>,
    col_basis_t: Vec<f32>,
}

/// Plain-Rust whole-frame DCT-II distortion effect — a direct transcription
/// of `shader.wgsl`'s `row_pass`/`col_pass` compute kernels, run on the CPU
/// instead of the GPU. Exists so the pipeline can run with no GPU/wgpu
/// adapter present at all (e.g. a standard GitHub Actions runner). Each
/// pass parallelizes over independent output rows with rayon, so this is
/// GPU-less but not single-threaded.
pub struct DctCpu {
    basis: RefCell<Option<Basis>>,
    // Backs `submit_rgb`/`finish_rgb`'s trait-level pipelining contract:
    // CPU has no separate submit/wait phase (no GPU queue to keep
    // non-empty), so `submit_rgb` just computes eagerly and stashes the
    // result here for `finish_rgb` to hand back — implementing both
    // (rather than relying on `DctBackend`'s stateless defaults) keeps
    // `backends/gstreamer.rs`'s compute thread backend-agnostic, calling
    // `submit_rgb`/`finish_rgb` uniformly regardless of which
    // `Box<dyn DctBackend>` it holds.
    stash: RefCell<[Option<(Vec<f32>, Vec<f32>, Vec<f32>)>; 2]>,
}

impl Default for DctCpu {
    fn default() -> Self {
        Self::new()
    }
}

impl DctCpu {
    pub fn new() -> Self {
        Self { basis: RefCell::new(None), stash: RefCell::new([None, None]) }
    }

    /// See the `stash` field's doc comment: eagerly computes (CPU has
    /// nothing to gain from deferring) and stashes the result for
    /// `finish_rgb` to pick up.
    pub fn submit_rgb(
        &self,
        frame_slot: usize,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<()> {
        if frame_slot >= 2 {
            bail!("submit_rgb: frame_slot must be 0 or 1 (got {frame_slot})");
        }
        let result = self.process_rgb(r, g, b, width, height, cutoff)?;
        self.stash.borrow_mut()[frame_slot] = Some(result);
        Ok(())
    }

    /// Takes back the result `submit_rgb` stashed for `frame_slot`. Errors
    /// if `submit_rgb` wasn't called first for that slot (or its result was
    /// already taken by an earlier `finish_rgb`).
    pub fn finish_rgb(&self, frame_slot: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if frame_slot >= 2 {
            bail!("finish_rgb: frame_slot must be 0 or 1 (got {frame_slot})");
        }
        self.stash.borrow_mut()[frame_slot]
            .take()
            .context("finish_rgb: no result stashed for this frame_slot - submit_rgb must be called first")
    }

    pub fn process_rgb(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let r_out = self.process_plane(r, width, height, cutoff)?;
        let g_out = self.process_plane(g, width, height, cutoff)?;
        let b_out = self.process_plane(b, width, height, cutoff)?;
        Ok((r_out, g_out, b_out))
    }

    /// Runs the whole-frame forward+quantize+inverse DCT on one
    /// single-channel plane (row-major, f32 0..255) — same contract as
    /// `DctGpu::process_plane`.
    pub fn process_plane(&self, pixels: &[f32], width: u32, height: u32, cutoff: f32) -> Result<Vec<f32>> {
        if width == 0 || height == 0 {
            bail!("process_plane: width and height must both be non-zero (got {width}x{height})");
        }
        if pixels.len() != width as usize * height as usize {
            bail!(
                "process_plane: pixel buffer has {} elements, expected width*height = {}",
                pixels.len(),
                width as usize * height as usize
            );
        }

        {
            let mut cache = self.basis.borrow_mut();
            let dims_match = matches!(&*cache, Some(b) if b.width == width && b.height == height);
            if !dims_match {
                let row_basis = dct_basis(width as usize);
                let row_basis_t = transpose_square(&row_basis, width as usize);
                let col_basis = dct_basis(height as usize);
                let col_basis_t = transpose_square(&col_basis, height as usize);
                *cache = Some(Basis { width, height, row_basis, row_basis_t, col_basis, col_basis_t });
            }
        }

        // Same epsilon nudge as the GPU path, so cutoff=2.0 stays lossless
        // despite floating-point rounding at the boundary.
        let threshold = cutoff.clamp(0.0, 2.0) + f32::EPSILON;

        let cache = self.basis.borrow();
        let basis = cache.as_ref().expect("basis cache populated above");

        // forward row transform: pixels -> row-frequency coefficients
        let a = row_transform(pixels, width, height, &basis.row_basis, false);
        // forward col transform + cutoff mask: -> full 2D frequency coefficients
        let b = col_transform(&a, width, height, &basis.col_basis, Some(threshold));
        // inverse col transform
        let c = col_transform(&b, width, height, &basis.col_basis_t, None);
        // inverse row transform + clamp to pixel range
        let d = row_transform(&c, width, height, &basis.row_basis_t, true);

        Ok(d)
    }
}

/// Transforms every row independently along the width axis — the CPU
/// equivalent of `shader.wgsl`'s `row_pass`. Used for both the forward row
/// transform (`basis` = B, `clamp` = false) and the inverse row transform
/// (`basis` = B^T, `clamp` = true).
fn row_transform(src: &[f32], width: u32, height: u32, basis: &[f32], clamp: bool) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let mut dst = vec![0f32; w * h];
    dst.par_chunks_mut(w).enumerate().take(h).for_each(|(y, out_row)| {
        let src_row = &src[y * w..y * w + w];
        for (x, out) in out_row.iter_mut().enumerate() {
            let basis_row = &basis[x * w..x * w + w];
            let sum: f32 = src_row.iter().zip(basis_row).map(|(s, c)| s * c).sum();
            *out = if clamp { sum.clamp(0.0, 255.0) } else { sum };
        }
    });
    dst
}

/// Transforms every column independently along the height axis — the CPU
/// equivalent of `shader.wgsl`'s `col_pass`. Used for the forward column
/// transform (`basis` = B, `mask` = the diagonal frequency cutoff) and the
/// inverse column transform (`basis` = B^T, `mask` = None).
fn col_transform(src: &[f32], width: u32, height: u32, basis: &[f32], mask: Option<f32>) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let denom_x = (w.max(2) - 1) as f32;
    let denom_y = (h.max(2) - 1) as f32;
    let mut dst = vec![0f32; w * h];
    dst.par_chunks_mut(w).enumerate().take(h).for_each(|(y, out_row)| {
        let basis_row = &basis[y * h..y * h + h];
        for (x, out) in out_row.iter_mut().enumerate() {
            let mut sum: f32 = (0..h).map(|i| src[i * w + x] * basis_row[i]).sum();
            if let Some(threshold) = mask {
                let rank = x as f32 / denom_x + y as f32 / denom_y;
                if rank > threshold {
                    sum = 0.0;
                }
            }
            *out = sum;
        }
    });
    dst
}

impl DctBackend for DctCpu {
    fn process_rgb(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        DctCpu::process_rgb(self, r, g, b, width, height, cutoff)
    }

    fn submit_rgb(
        &self,
        frame_slot: usize,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<()> {
        DctCpu::submit_rgb(self, frame_slot, r, g, b, width, height, cutoff)
    }

    fn finish_rgb(&self, frame_slot: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        DctCpu::finish_rgb(self, frame_slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::DctGpu;

    // These tests run unconditionally, on every machine — no GPU adapter
    // required — which is itself part of the proof this backend works
    // without one.

    #[test]
    fn full_cutoff_roundtrip_is_near_lossless() {
        let cpu = DctCpu::new();
        let (w, h) = (24u32, 18u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| ((i * 37 + 11) % 256) as f32).collect();

        let out = cpu.process_plane(&plane, w, h, 2.0).unwrap();
        assert_eq!(out.len(), plane.len());
        for (a, b) in plane.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1.0, "roundtrip drift too large: {a} vs {b}");
        }
    }

    #[test]
    fn very_low_cutoff_collapses_toward_the_frame_average() {
        let cpu = DctCpu::new();
        let (w, h) = (16u32, 16u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| (i % 256) as f32).collect();
        let mean = plane.iter().sum::<f32>() / plane.len() as f32;

        let out = cpu.process_plane(&plane, w, h, 0.02).unwrap();
        for v in out {
            assert!((v - mean).abs() < 5.0, "near-DC-only output {v} should be close to frame mean {mean}");
        }
    }

    #[test]
    fn lower_cutoff_reduces_output_variance() {
        let cpu = DctCpu::new();
        let (w, h) = (16u32, 16u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| ((i * 53) % 256) as f32).collect();

        let variance = |v: &[f32]| {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32
        };

        let high_c = cpu.process_plane(&plane, w, h, 1.8).unwrap();
        let low_c = cpu.process_plane(&plane, w, h, 0.2).unwrap();
        assert!(
            variance(&low_c) < variance(&high_c),
            "low cutoff should suppress detail more: var(low)={} var(high)={}",
            variance(&low_c),
            variance(&high_c)
        );
    }

    #[test]
    fn handles_one_pixel_wide_and_tall_frames() {
        let cpu = DctCpu::new();
        for (w, h) in [(1u32, 9u32), (9u32, 1u32), (1u32, 1u32)] {
            let plane: Vec<f32> = (0..(w * h)).map(|i| (i * 17 % 256) as f32).collect();
            let out = cpu.process_plane(&plane, w, h, 1.0).unwrap();
            assert_eq!(out.len(), plane.len());
            assert!(out.iter().all(|v| v.is_finite()), "{w}x{h} produced a non-finite pixel");
        }
    }

    #[test]
    fn rejects_zero_dimensions_and_mismatched_pixel_buffer() {
        let cpu = DctCpu::new();
        assert!(cpu.process_plane(&[], 0, 4, 1.0).is_err());
        assert!(cpu.process_plane(&[], 4, 0, 1.0).is_err());
        let wrong_len = vec![0f32; 3];
        assert!(cpu.process_plane(&wrong_len, 4, 4, 1.0).is_err());
    }

    #[test]
    fn cpu_and_gpu_backends_agree() {
        let Ok(gpu) = DctGpu::new() else {
            eprintln!("skipping GPU cross-check: no adapter available");
            return;
        };
        let cpu = DctCpu::new();
        let (w, h) = (20u32, 15u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| ((i * 41 + 7) % 256) as f32).collect();

        for cutoff in [0.1f32, 0.6, 1.3, 2.0] {
            let cpu_out = cpu.process_plane(&plane, w, h, cutoff).unwrap();
            let gpu_out = gpu.process_plane(&plane, w, h, cutoff).unwrap();
            for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
                assert!(
                    (a - b).abs() < 0.5,
                    "cutoff={cutoff}: CPU/GPU disagree: {a} vs {b}"
                );
            }
        }
    }
}
