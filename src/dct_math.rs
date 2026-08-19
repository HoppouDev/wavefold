use std::f32::consts::PI;

/// Orthonormal NxN DCT-II basis: `basis[k*n + i]` is basis function `k`
/// evaluated at position `i`. The same matrix serves as both the forward
/// and inverse transform for either compute backend (see shader.wgsl for
/// why, and dct_backend.rs for the CPU-side transcription of that logic).
pub(crate) fn dct_basis(n: usize) -> Vec<f32> {
    let mut b = vec![0f32; n * n];
    let n_f = n as f32;
    for k in 0..n {
        let alpha = if k == 0 { (1.0 / n_f).sqrt() } else { (2.0 / n_f).sqrt() };
        for i in 0..n {
            b[k * n + i] = alpha * ((PI / n_f) * (i as f32 + 0.5) * k as f32).cos();
        }
    }
    b
}

/// Transposes an NxN matrix stored row-major. Used to precompute B^T
/// (needed for the inverse DCT direction) once on the CPU instead of
/// branching on direction inside the per-element inner loop.
pub(crate) fn transpose_square(m: &[f32], n: usize) -> Vec<f32> {
    let mut t = vec![0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            t[col * n + row] = m[row * n + col];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_basis_rows_are_orthonormal() {
        for n in [8usize, 13, 32] {
            let b = dct_basis(n);
            for k in 0..n {
                for j in 0..n {
                    let dot: f32 = (0..n).map(|i| b[k * n + i] * b[j * n + i]).sum();
                    let expected = if k == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-3,
                        "n={n} basis rows {k},{j} not orthonormal: dot={dot}"
                    );
                }
            }
        }
    }
}
