// Whole-frame separable DCT-II. Each 1D pass (row or column) is applied to
// the *entire* frame at once, not an 8x8 block, so a single dispatch here
// is one axis of one direction (forward or inverse) of a full-frame 2D DCT.
//
// The basis matrix B is orthonormal, so its inverse is its transpose: the
// forward transform is F(k) = sum_n f(n)*B(k,n) (output index k selects the
// basis row), while the inverse is f(n) = sum_k F(k)*B(k,n) (contracted
// index k selects the basis row instead). Rather than branch on direction
// per-thread, the host precomputes both B and its transpose and binds
// whichever one the current pass needs — both passes below use one plain
// `basis[out*n+i]` indexing formula regardless of direction.

struct Params {
    width: u32,
    height: u32,
    threshold: f32, // normalized diagonal frequency cutoff, 0..2
    apply_mask: u32,
    clamp_output: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_basis: array<f32>; // width x width
@group(0) @binding(2) var<storage, read> col_basis: array<f32>; // height x height

@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;

// Transforms every row independently along the width axis. Used both as the
// forward row transform (src = pixels, row_basis = B, out index = horizontal
// frequency) and the inverse row transform (src = coefficients, row_basis =
// B^T, out index = pixel x).
@compute @workgroup_size(8, 8, 1)
fn row_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let row_off = y * params.width;
    let basis_off = x * params.width;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.width; i = i + 1u) {
        sum = sum + src[row_off + i] * row_basis[basis_off + i];
    }
    if (params.clamp_output != 0u) {
        sum = clamp(sum, 0.0, 255.0);
    }
    dst[row_off + x] = sum;
}

// Transforms every column independently along the height axis. Used as both
// the forward column transform (col_basis = B; completes the 2D frequency-
// domain coefficients and, when apply_mask is set, drops everything above
// the quality cutoff) and the inverse column transform (col_basis = B^T).
@compute @workgroup_size(8, 8, 1)
fn col_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let basis_off = y * params.height;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.height; i = i + 1u) {
        sum = sum + src[i * params.width + x] * col_basis[basis_off + i];
    }
    if (params.apply_mask != 0u) {
        let denom_x = f32(max(params.width, 2u) - 1u);
        let denom_y = f32(max(params.height, 2u) - 1u);
        let rank = f32(x) / denom_x + f32(y) / denom_y;
        if (rank > params.threshold) {
            sum = 0.0;
        }
    }
    dst[y * params.width + x] = sum;
}
