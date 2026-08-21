// Mixed-radix (2/3/5/7) complex FFT, one Cooley-Tukey stage per dispatch.
// Ported from (and verified equivalent to) the iterative algorithm in
// fft_plan.rs::fft_mixed_radix - see that file's doc comment for the
// digit-reversal + growing-butterfly-stage derivation. This shader is
// just the inner per-stage butterfly loop; digit-reversal permutation is
// precomputed host-side (mirroring how shader.wgsl's row_basis/col_basis
// matrices are precomputed once and reused, not recomputed per-
// invocation) since it depends on the *full* factor list, not one stage.
// Both twiddle stages (gather and radix-combine) are cheap enough
// (radix <= 7) to compute inline via cos/sin rather than precomputing a
// table - this also means one `sign` uniform naturally covers both
// forward and inverse, matching fft_mixed_radix's single `sign` variable
// exactly instead of needing separately-baked forward/inverse tables.
//
// Batches `batch_count` independent length-`n` FFTs at once (one per row
// or one per column of the frame, laid out contiguously) since a single
// video frame needs many independent 1D FFTs, not just one.

struct StageParams {
    n: u32,            // total length of one FFT in the batch
    l: u32,            // sub-transform length *before* this stage
    radix: u32,        // this stage's radix (2, 3, 5, or 7)
    batch_count: u32,  // number of independent FFTs sharing this dispatch
    sign: f32,         // -1.0 forward, +1.0 inverse (matches fft_plan.rs)
};

@group(0) @binding(0) var<uniform> params: StageParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>; // n * batch_count, in place

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

const TAU: f32 = 6.283185307179586;

// One invocation handles one (batch, group, position-within-l) triplet:
// gathers `radix` inputs spaced `l` apart, applies this stage's twiddle
// factors, combines via a direct radix-point DFT, and writes `radix`
// outputs back in place. `groups = n / (l * radix)`.
@compute @workgroup_size(256, 1, 1)
fn fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l_new = params.l * params.radix;
    let groups = params.n / l_new;
    let work_per_batch = groups * params.l;
    let total_work = work_per_batch * params.batch_count;
    if (gid.x >= total_work) {
        return;
    }

    let batch = gid.x / work_per_batch;
    let within_batch = gid.x % work_per_batch;
    let g = within_batch / params.l;
    let p = within_batch % params.l;
    let base = batch * params.n + g * l_new;

    var inputs: array<vec2<f32>, 7>;
    for (var j: u32 = 0u; j < params.radix; j = j + 1u) {
        let tw_angle = params.sign * TAU * f32(j) * f32(p) / f32(l_new);
        let tw = vec2<f32>(cos(tw_angle), sin(tw_angle));
        inputs[j] = cmul(data[base + j * params.l + p], tw);
    }

    for (var k: u32 = 0u; k < params.radix; k = k + 1u) {
        var acc = vec2<f32>(0.0, 0.0);
        for (var j: u32 = 0u; j < params.radix; j = j + 1u) {
            let angle = params.sign * TAU * f32(j) * f32(k) / f32(params.radix);
            let w = vec2<f32>(cos(angle), sin(angle));
            acc = acc + cmul(inputs[j], w);
        }
        data[base + k * params.l + p] = acc;
    }
}

// Digit-reversal permutation, applied once before the first stage.
// `perm[i]` is the destination index for source index `i` (matches
// fft_plan.rs::digit_reverse_index exactly - precomputed host-side since
// it depends on the full factor list, not just one stage).
struct PermParams {
    n: u32,
    batch_count: u32,
};

@group(0) @binding(0) var<uniform> perm_params: PermParams;
@group(0) @binding(1) var<storage, read> perm: array<u32>; // length n
@group(0) @binding(2) var<storage, read> perm_src: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> perm_dst: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn digit_reverse_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = perm_params.n * perm_params.batch_count;
    if (gid.x >= total) {
        return;
    }
    let batch = gid.x / perm_params.n;
    let i = gid.x % perm_params.n;
    let base = batch * perm_params.n;
    perm_dst[base + perm[i]] = perm_src[gid.x];
}

// --- DCT-II/DCT-III construction (Makhoul's algorithm) on top of the FFT
// above - see fft_plan.rs::dct2_via_fft / dct3_via_fft's doc comments for
// the derivation this is a direct transcription of. `orthonormal_alpha`
// matches dct_math::dct_basis's alpha_k exactly.

fn orthonormal_alpha(k: u32, n: u32) -> f32 {
    if (k == 0u) {
        return sqrt(1.0 / f32(n));
    }
    return sqrt(2.0 / f32(n));
}

// Real input -> complex, scattered through a *combined* permutation table
// (Makhoul's even/odd interleave composed with digit-reversal, so the FFT
// stages can run directly on the result with no extra pass) - the real-
// input analog of `digit_reverse_permute` above, which is complex-only.
struct DctFwdPermParams {
    n: u32,
    batch_count: u32,
};

@group(0) @binding(0) var<uniform> dct_fwd_perm_params: DctFwdPermParams;
@group(0) @binding(1) var<storage, read> dct_fwd_perm: array<u32>; // length n, combined table
@group(0) @binding(2) var<storage, read> dct_fwd_perm_src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dct_fwd_perm_dst: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn dct_forward_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dct_fwd_perm_params.n * dct_fwd_perm_params.batch_count;
    if (gid.x >= total) {
        return;
    }
    let i = gid.x % dct_fwd_perm_params.n;
    let base = gid.x - i;
    dct_fwd_perm_dst[base + dct_fwd_perm[i]] = vec2<f32>(dct_fwd_perm_src[gid.x], 0.0);
}

// Post-FFT: V[k] (complex) -> F[k] (real, orthonormal-scaled DCT-II
// coefficient). One invocation per (batch, k).
struct Dct2PostParams {
    n: u32,
    batch_count: u32,
};

@group(0) @binding(0) var<uniform> dct2_post_params: Dct2PostParams;
@group(0) @binding(1) var<storage, read> dct2_post_src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dct2_post_dst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn dct2_post_twiddle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dct2_post_params.n * dct2_post_params.batch_count;
    if (gid.x >= total) {
        return;
    }
    let k = gid.x % dct2_post_params.n;
    let n = dct2_post_params.n;
    let angle = -TAU * f32(k) / (4.0 * f32(n)); // -pi*k/(2N), TAU=2pi
    let tw = vec2<f32>(cos(angle), sin(angle));
    let v = dct2_post_src[gid.x];
    let x_unnorm = 2.0 * (v.x * tw.x - v.y * tw.y); // 2*Re(v*tw)
    dct2_post_dst[gid.x] = x_unnorm * orthonormal_alpha(k, n) / 2.0;
}

// Pre-inverse-FFT: F[k] (real) -> V[k] (complex), the boundary-aware
// conjugate-symmetric reconstruction. Output is in *natural* order (not
// yet digit-reversed) - pair with `digit_reverse_permute` (using a plain,
// non-combined digit-reversal table) before running `fft_stage` with
// sign=+1.
struct Dct3PreParams {
    n: u32,
    batch_count: u32,
};

@group(0) @binding(0) var<uniform> dct3_pre_params: Dct3PreParams;
@group(0) @binding(1) var<storage, read> dct3_pre_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dct3_pre_dst: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn dct3_pre_twiddle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dct3_pre_params.n * dct3_pre_params.batch_count;
    if (gid.x >= total) {
        return;
    }
    let n = dct3_pre_params.n;
    let k = gid.x % n;
    let base = gid.x - k;
    let half = n / 2u;

    let f_k = dct3_pre_src[gid.x];
    let x_unnorm_k = f_k * 2.0 / orthonormal_alpha(k, n);

    if (k == 0u) {
        dct3_pre_dst[gid.x] = vec2<f32>(x_unnorm_k / 2.0, 0.0);
        return;
    }
    if (k == half) {
        dct3_pre_dst[gid.x] = vec2<f32>(x_unnorm_k / sqrt(2.0), 0.0);
        return;
    }
    if (k < half) {
        let f_nk = dct3_pre_src[base + (n - k)];
        let x_unnorm_nk = f_nk * 2.0 / orthonormal_alpha(n - k, n);
        let theta = -TAU * f32(k) / (4.0 * f32(n));
        let conj_tw = vec2<f32>(cos(theta), -sin(theta));
        let combined = vec2<f32>(x_unnorm_k, -x_unnorm_nk);
        let vk = cmul(combined, conj_tw);
        dct3_pre_dst[gid.x] = vec2<f32>(vk.x / 2.0, vk.y / 2.0);
        return;
    }
    // k > half: conjugate symmetry, mirrors the k < half case computed
    // for index (n - k) above.
    let f_mirror = dct3_pre_src[base + (n - k)];
    let x_unnorm_mirror = f_mirror * 2.0 / orthonormal_alpha(n - k, n);
    let theta_mirror = -TAU * f32(n - k) / (4.0 * f32(n));
    let conj_tw_mirror = vec2<f32>(cos(theta_mirror), -sin(theta_mirror));
    let combined_mirror = vec2<f32>(x_unnorm_mirror, -x_unnorm_k);
    let v_mirror = cmul(combined_mirror, conj_tw_mirror);
    dct3_pre_dst[gid.x] = vec2<f32>(v_mirror.x / 2.0, -v_mirror.y / 2.0);
}

// Post-inverse-FFT: v (complex, unscaled) -> x[n] (real), gathered through
// the same combined interleave index used by `dct_forward_permute` (the
// permutation is a bijection - forward *scatters into* that position,
// this *gathers from* it) and scaled by 1/N.
struct Dct3PostParams {
    n: u32,
    batch_count: u32,
};

@group(0) @binding(0) var<uniform> dct3_post_params: Dct3PostParams;
@group(0) @binding(1) var<storage, read> dct3_post_gather_idx: array<u32>; // length n
@group(0) @binding(2) var<storage, read> dct3_post_src: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> dct3_post_dst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn dct3_post_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dct3_post_params.n * dct3_post_params.batch_count;
    if (gid.x >= total) {
        return;
    }
    let n = dct3_post_params.n;
    let i = gid.x % n;
    let base = gid.x - i;
    dct3_post_dst[gid.x] = dct3_post_src[base + dct3_post_gather_idx[i]].x / f32(n);
}

// --- Plumbing to let gpu.rs mix FFT-based row/col passes (which need a
// contiguous length-n batch per row) with the GEMM-tiled column passes in
// shader.wgsl (which read columns strided through a row-major HxW
// buffer): transpose bridges the two layouts, and the mask/clamp passes
// mirror shader.wgsl's fused Params.apply_mask/clamp_output so the FFT
// path gets identical cutoff and pixel-range behavior.

struct TransposeParams {
    width: u32,  // src width (src is width x height row-major)
    height: u32,
};

@group(0) @binding(0) var<uniform> transpose_params: TransposeParams;
@group(0) @binding(1) var<storage, read> transpose_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> transpose_dst: array<f32>; // height x width row-major

@compute @workgroup_size(256, 1, 1)
fn transpose_real(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = transpose_params.width * transpose_params.height;
    if (gid.x >= total) {
        return;
    }
    let x = gid.x % transpose_params.width;
    let y = gid.x / transpose_params.width;
    transpose_dst[x * transpose_params.height + y] = transpose_src[gid.x];
}

// Applies shader.wgsl's exact cutoff-mask formula to a length-(width*height)
// row-major buffer of forward DCT-II coefficients (post row+col transform),
// zeroing any (x,y) above the diagonal frequency threshold.
struct MaskParams {
    width: u32,
    height: u32,
    threshold: f32,
};

@group(0) @binding(0) var<uniform> mask_params: MaskParams;
@group(0) @binding(1) var<storage, read_write> mask_data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn apply_cutoff_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = mask_params.width * mask_params.height;
    if (gid.x >= total) {
        return;
    }
    let x = gid.x % mask_params.width;
    let y = gid.x / mask_params.width;
    let denom_x = f32(max(mask_params.width, 2u) - 1u);
    let denom_y = f32(max(mask_params.height, 2u) - 1u);
    let rank = f32(x) / denom_x + f32(y) / denom_y;
    if (rank > mask_params.threshold) {
        mask_data[gid.x] = 0.0;
    }
}

@group(0) @binding(0) var<storage, read_write> clamp_data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn clamp_pixel_range(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&clamp_data)) {
        return;
    }
    clamp_data[gid.x] = clamp(clamp_data[gid.x], 0.0, 255.0);
}
