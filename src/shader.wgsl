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
//
// Each pass is a dense matrix multiply (row_pass: SRC * B^T, K = width;
// col_pass: B * SRC, K = height) tiled through workgroup-shared memory: the
// classic blocked-GEMM technique, walking the K dimension in TILE-sized
// chunks and caching both operand tiles in `workgroup` storage so every
// element loaded from `storage` is reused TILE times instead of once. This
// is unrelated to (and not limited by) any fixed block size in the DCT
// itself — the frame stays whole-frame; only the *memory access pattern*
// is tiled, walking however many TILE-sized chunks `width`/`height` need,
// so it scales to arbitrary frame dimensions the same as the untiled
// version did. TILE=16 keeps the workgroup at 256 invocations (the
// portable `max_compute_invocations_per_workgroup` limit) and 2KB of
// shared storage (row/col tiles), both far under the portable 16KB
// `max_compute_workgroup_storage_size` floor.
//
// `workgroupBarrier()` requires every invocation in the workgroup to reach
// it — so unlike the untiled version, out-of-range threads (frame
// dimensions not a multiple of TILE) can't early-return before the tile
// loop. They stay in lockstep with in-range threads for every barrier,
// contribute zero-padded loads, and only skip the final `dst` write.

const TILE: u32 = 16u;

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

var<workgroup> tile_a: array<array<f32, TILE>, TILE>;
var<workgroup> tile_b: array<array<f32, TILE>, TILE>;

// Transforms every row independently along the width axis: DST = SRC * B^T,
// an (height x width)*(width x width) matmul with K = width. Used both as
// the forward row transform (src = pixels, row_basis = B, out index =
// horizontal frequency) and the inverse row transform (src = coefficients,
// row_basis = B^T, out index = pixel x).
@compute @workgroup_size(TILE, TILE, 1)
fn row_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let x = gid.x; // output column: frequency (forward) or pixel x (inverse)
    let y = gid.y; // row, independent of the transform
    let lx = lid.x;
    let ly = lid.y;

    var sum: f32 = 0.0;
    let num_tiles = (params.width + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_base = t * TILE;

        // tile_a[ly][lx] = SRC[y][k_base+lx]
        let k_a = k_base + lx;
        if (y < params.height && k_a < params.width) {
            tile_a[ly][lx] = src[y * params.width + k_a];
        } else {
            tile_a[ly][lx] = 0.0;
        }

        // tile_b[ly][lx] = B^T[k_base+ly][x] = row_basis[x][k_base+ly]
        let k_b = k_base + ly;
        if (x < params.width && k_b < params.width) {
            tile_b[ly][lx] = row_basis[x * params.width + k_b];
        } else {
            tile_b[ly][lx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tile_a[ly][k] * tile_b[k][lx];
        }

        workgroupBarrier();
    }

    if (x >= params.width || y >= params.height) {
        return;
    }
    if (params.clamp_output != 0u) {
        sum = clamp(sum, 0.0, 255.0);
    }
    dst[y * params.width + x] = sum;
}

// Transforms every column independently along the height axis: DST = B *
// SRC, an (height x height)*(height x width) matmul with K = height. Used
// as both the forward column transform (col_basis = B; completes the 2D
// frequency-domain coefficients and, when apply_mask is set, drops
// everything above the quality cutoff) and the inverse column transform
// (col_basis = B^T).
@compute @workgroup_size(TILE, TILE, 1)
fn col_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let x = gid.x;
    let y = gid.y;
    let lx = lid.x;
    let ly = lid.y;

    var sum: f32 = 0.0;
    let num_tiles = (params.height + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_base = t * TILE;

        // tile_a[ly][lx] = B[y][k_base+lx] = col_basis[y][k_base+lx]
        let k_a = k_base + lx;
        if (y < params.height && k_a < params.height) {
            tile_a[ly][lx] = col_basis[y * params.height + k_a];
        } else {
            tile_a[ly][lx] = 0.0;
        }

        // tile_b[ly][lx] = SRC[k_base+ly][x]
        let k_b = k_base + ly;
        if (k_b < params.height && x < params.width) {
            tile_b[ly][lx] = src[k_b * params.width + x];
        } else {
            tile_b[ly][lx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tile_a[ly][k] * tile_b[k][lx];
        }

        workgroupBarrier();
    }

    if (x >= params.width || y >= params.height) {
        return;
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
