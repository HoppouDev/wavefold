//! Mixed-radix (2/3/5/7) FFT planning: factorization, digit-reversal
//! permutation, and the iterative Cooley-Tukey butterfly algorithm — first
//! implemented and verified here in plain Rust (checked against `rustfft`
//! in tests) before being transcribed into `shader_fft.wgsl`, so algorithm
//! bugs get caught by a cheap CPU unit test instead of a GPU pipeline.
//! `src/gpu.rs` reuses `factor_mixed_radix` to decide, per axis, whether a
//! given width/height is FFT-eligible or must fall back to the existing
//! tiled-GEMM `shader.wgsl` path.

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

pub(crate) const RADICES: [usize; 4] = [2, 3, 5, 7];

/// Factors `n` into {2,3,5,7}, smallest radix first. `None` if `n` is 0,
/// 1 (too small to bother), or has a factor >7 remaining (e.g. 541) —
/// signals "not FFT-eligible, use the GEMM fallback for this axis."
pub(crate) fn factor_mixed_radix(n: usize) -> Option<Vec<usize>> {
    if n < 2 {
        return None;
    }
    let mut remaining = n;
    let mut factors = Vec::new();
    for &r in &RADICES {
        while remaining % r == 0 {
            factors.push(r);
            remaining /= r;
        }
    }
    if remaining == 1 {
        Some(factors)
    } else {
        None
    }
}

/// Whether `n` is eligible for the FFT-based DCT-II/III path: factors
/// completely into {2,3,5,7} *and* is even (`dct2_via_fft`/`dct3_via_fft`
/// assume even length - see their doc comments). Real video dimensions
/// are essentially always even, so this only excludes truly unusual axes.
pub(crate) fn fft_eligible(n: usize) -> Option<Vec<usize>> {
    if n % 2 != 0 {
        return None;
    }
    factor_mixed_radix(n)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct C32 {
    pub re: f32,
    pub im: f32,
}

impl C32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

/// Digit-reversal permutation index for mixed-radix decimation-in-time:
/// `n` decomposed into digits `d_1..d_m` (`d_1` least significant, using
/// `factors[0]` as its radix, `factors[1]` for `d_2`, etc — same order
/// `factor_mixed_radix` returns), reassembled with the digits *and* their
/// radix weights both reversed. Reduces to classic bit-reversal when every
/// factor is 2.
pub(crate) fn digit_reverse_index(mut n: usize, factors: &[usize]) -> usize {
    let mut digits = Vec::with_capacity(factors.len());
    for &r in factors {
        digits.push(n % r);
        n /= r;
    }
    let mut out = 0usize;
    let mut weight = 1usize;
    for (&d, &r) in digits.iter().rev().zip(factors.iter().rev()) {
        out += d * weight;
        weight *= r;
    }
    out
}

/// Makhoul's even/odd-interleave index map: for original index `i`
/// (`0..n`), returns its position in the permuted sequence `v` where
/// `v[n_] = x[2*n_]` (`n_ < n/2`), `v[n-1-n_] = x[2*n_+1]` (`n_ < n/2`) -
/// i.e. `v[dct_interleave_index(i)] == x[i]`. Used both to build the
/// combined forward scatter table (`dct_interleave_index` composed with
/// `digit_reverse_index`) and, in the inverse direction, as a gather
/// index directly (the same map, since it's a bijection).
pub(crate) fn dct_interleave_index(i: usize, n: usize) -> usize {
    if i % 2 == 0 {
        i / 2
    } else {
        n - 1 - (i - 1) / 2
    }
}

/// In-place iterative mixed-radix Cooley-Tukey FFT. `data.len()` must
/// equal `factors.iter().product()`. `inverse = true` computes the
/// (unscaled — caller divides by N if wanted) inverse transform via
/// conjugated twiddle signs.
pub(crate) fn fft_mixed_radix(data: &mut [C32], factors: &[usize], inverse: bool) {
    let n = data.len();
    debug_assert_eq!(n, factors.iter().product::<usize>());

    let mut permuted = vec![C32::new(0.0, 0.0); n];
    for (i, &v) in data.iter().enumerate() {
        permuted[digit_reverse_index(i, factors)] = v;
    }
    data.copy_from_slice(&permuted);

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut l = 1usize;
    for &r in factors.iter().rev() {
        let l_new = l * r;
        let groups = n / l_new;
        for g in 0..groups {
            let base = g * l_new;
            for p in 0..l {
                let mut inputs = [C32::new(0.0, 0.0); 7];
                for (j, slot) in inputs.iter_mut().enumerate().take(r) {
                    let tw_angle = sign * 2.0 * PI * (j as f32) * (p as f32) / (l_new as f32);
                    let tw = C32::new(tw_angle.cos(), tw_angle.sin());
                    *slot = data[base + j * l + p].mul(tw);
                }
                for k in 0..r {
                    let mut acc = C32::new(0.0, 0.0);
                    for (j, &inp) in inputs.iter().enumerate().take(r) {
                        let angle = sign * 2.0 * PI * (j as f32) * (k as f32) / (r as f32);
                        let w = C32::new(angle.cos(), angle.sin());
                        acc = acc.add(inp.mul(w));
                    }
                    data[base + k * l + p] = acc;
                }
            }
        }
        l = l_new;
    }
}

/// The orthonormal DCT-II basis coefficient `alpha_k`, matching
/// `dct_math::dct_basis` exactly (`alpha_0 = sqrt(1/n)`,
/// `alpha_{k>0} = sqrt(2/n)`) - both `dct2_via_fft` and `dct3_via_fft`
/// need this same scaling, so it's shared here instead of duplicated.
fn orthonormal_alpha(k: usize, n: usize) -> f32 {
    if k == 0 {
        (1.0 / n as f32).sqrt()
    } else {
        (2.0 / n as f32).sqrt()
    }
}

/// Real-input DCT-II via FFT (Makhoul's algorithm): permute into
/// even-then-reversed-odd order, FFT, then a per-bin twiddle + orthonormal
/// scale. `x.len()` must be even and factor completely into {2,3,5,7}
/// (see `factor_mixed_radix`) - verified against `dct_math::dct_basis`'s
/// direct matrix multiply in tests below.
pub(crate) fn dct2_via_fft(x: &[f32], factors: &[usize]) -> Vec<f32> {
    let n = x.len();
    debug_assert_eq!(n % 2, 0, "dct2_via_fft requires even length");
    let half = n / 2;

    let mut v = vec![C32::new(0.0, 0.0); n];
    for i in 0..half {
        v[i] = C32::new(x[2 * i], 0.0);
        v[n - 1 - i] = C32::new(x[2 * i + 1], 0.0);
    }

    fft_mixed_radix(&mut v, factors, false);

    let mut out = vec![0f32; n];
    for (k, slot) in out.iter_mut().enumerate() {
        let angle = -PI * (k as f32) / (2.0 * n as f32);
        let tw = C32::new(angle.cos(), angle.sin());
        let x_unnorm = 2.0 * v[k].mul(tw).re;
        *slot = x_unnorm * orthonormal_alpha(k, n) / 2.0;
    }
    out
}

/// Real-input DCT-III (the exact inverse of `dct2_via_fft` - `dct_basis`
/// is orthonormal, so `x = B^T * F` recovers `x` from `F = B * x`
/// exactly, no lossy round trip) via FFT.
///
/// Derivation: undoing `dct2_via_fft`'s scale step gives
/// `X_unnorm[k] = F[k]*2/alpha_k`. Since the intermediate `v` in the
/// forward direction is real, its FFT `V` has conjugate symmetry
/// (`V[N-k] = conj(V[k])`), which combined with
/// `X_unnorm[k] = 2*Re(V[k]*tw[k])` (`tw[k] = exp(-i*pi*k/(2N))`) and the
/// mirrored `X_unnorm[N-k] = -2*(a_k*sin(theta_k) + b_k*cos(theta_k))`
/// (`V[k] = a_k + i*b_k`, `theta_k` = `tw[k]`'s angle) gives two linear
/// equations in `(a_k, b_k)` - a 2D rotation, solved by applying the
/// inverse rotation - yielding
/// `V[k] = (X_unnorm[k] - i*X_unnorm[N-k]) * conj(tw[k]) / 2` for
/// `1 <= k < N/2`, with `V[0] = X_unnorm[0]/2` and (N even)
/// `V[N/2] = X_unnorm[N/2]/sqrt(2)` handled as the real-valued boundary
/// cases this general formula doesn't cover. Verified independently
/// against `dct_math::dct_basis`'s transpose matmul in tests below, not
/// just via round-trip (a round-trip-only check can hide compensating
/// errors).
pub(crate) fn dct3_via_fft(f: &[f32], factors: &[usize]) -> Vec<f32> {
    let n = f.len();
    debug_assert_eq!(n % 2, 0, "dct3_via_fft requires even length");
    let half = n / 2;

    let x_unnorm: Vec<f32> = (0..n).map(|k| f[k] * 2.0 / orthonormal_alpha(k, n)).collect();

    let mut v = vec![C32::new(0.0, 0.0); n];
    v[0] = C32::new(x_unnorm[0] / 2.0, 0.0);
    v[half] = C32::new(x_unnorm[half] / std::f32::consts::SQRT_2, 0.0);
    for k in 1..half {
        let theta = -PI * (k as f32) / (2.0 * n as f32);
        let conj_tw = C32::new(theta.cos(), -theta.sin());
        let combined = C32::new(x_unnorm[k], -x_unnorm[n - k]);
        let vk = combined.mul(conj_tw);
        v[k] = C32::new(vk.re / 2.0, vk.im / 2.0);
        v[n - k] = C32::new(v[k].re, -v[k].im);
    }

    fft_mixed_radix(&mut v, factors, true);
    let inv_n = 1.0 / n as f32;

    let mut x = vec![0f32; n];
    for i in 0..half {
        x[2 * i] = v[i].re * inv_n;
        x[2 * i + 1] = v[n - 1 - i].re * inv_n;
    }
    x
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct StageParams {
    pub(crate) n: u32,
    pub(crate) l: u32,
    pub(crate) radix: u32,
    pub(crate) batch_count: u32,
    pub(crate) sign: f32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PermParams {
    pub(crate) n: u32,
    pub(crate) batch_count: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}

/// GPU-resident mixed-radix FFT (`shader_fft.wgsl`), verified equivalent
/// to `fft_mixed_radix` above (itself verified against `rustfft`) — this
/// is the actual engine `gpu.rs`'s FFT-eligible row/col passes dispatch
/// into. Pipelines are built once and reused across every call; per-call
/// buffers (sized by `n`/`batch_count`, which vary per plane) are not
/// cached here — `gpu.rs`'s `PlaneBuffers`-style caching (keyed on
/// `(width, height)`) is the right place for that, added in a later
/// integration pass.
pub(crate) struct GpuFft {
    pub(crate) permute_pipeline: wgpu::ComputePipeline,
    pub(crate) stage_pipeline: wgpu::ComputePipeline,
    pub(crate) dct_forward_permute_pipeline: wgpu::ComputePipeline,
    pub(crate) dct2_post_pipeline: wgpu::ComputePipeline,
    pub(crate) dct3_pre_pipeline: wgpu::ComputePipeline,
    pub(crate) dct3_post_pipeline: wgpu::ComputePipeline,
    pub(crate) transpose_pipeline: wgpu::ComputePipeline,
    pub(crate) mask_pipeline: wgpu::ComputePipeline,
    pub(crate) clamp_pipeline: wgpu::ComputePipeline,
}

impl GpuFft {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mixed-radix fft"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_fft.wgsl").into()),
        });
        let make = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            permute_pipeline: make("digit_reverse_permute"),
            stage_pipeline: make("fft_stage"),
            dct_forward_permute_pipeline: make("dct_forward_permute"),
            dct2_post_pipeline: make("dct2_post_twiddle"),
            dct3_pre_pipeline: make("dct3_pre_twiddle"),
            dct3_post_pipeline: make("dct3_post_gather"),
            transpose_pipeline: make("transpose_real"),
            mask_pipeline: make("apply_cutoff_mask"),
            clamp_pipeline: make("clamp_pixel_range"),
        }
    }

    /// Runs `stage_pipeline` for every stage of `factors` (reverse order,
    /// matching `fft_mixed_radix`), in place on `buf` - shared by `run`,
    /// `run_dct2`, and `run_dct3`, all of which need this same stage loop
    /// after their own direction-specific setup.
    fn run_stages_in_place(&self, device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, n: usize, factors: &[usize], batch_count: usize, inverse: bool) {
        use wgpu::util::DeviceExt;
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut l = 1usize;
        for &r in factors.iter().rev() {
            let l_new = l * r;
            let groups = n / l_new;
            let work_per_batch = groups * l;
            let total_work = (work_per_batch * batch_count) as u32;

            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fft stage params"),
                contents: bytemuck::bytes_of(&StageParams {
                    n: n as u32,
                    l: l as u32,
                    radix: r as u32,
                    batch_count: batch_count as u32,
                    sign,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let stage_layout = self.stage_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft stage bind group"),
                layout: &stage_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fft stage encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fft stage pass"), timestamp_writes: None });
                pass.set_pipeline(&self.stage_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_work.div_ceil(256), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            l = l_new;
        }
    }

    /// Blocking readback of a `Vec<f32>`-shaped storage buffer.
    fn read_back_f32(&self, device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let byte_len = (len * std::mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft f32 staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fft f32 readback encoder") });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_len);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).expect("gpu poll failed");
        rx.recv().expect("map channel closed").expect("buffer map failed");
        let mapped = slice.get_mapped_range().expect("failed to map gpu buffer");
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }

    /// Real-input DCT-II via FFT, GPU version of `dct2_via_fft`. `x` is a
    /// concatenation of `batch_count` independent length-`n` real
    /// sequences (rows or columns of a plane).
    pub(crate) fn run_dct2(&self, device: &wgpu::Device, queue: &wgpu::Queue, x: &[f32], n: usize, factors: &[usize], batch_count: usize) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        assert_eq!(x.len(), n * batch_count);
        let complex_bytes = (n * batch_count * std::mem::size_of::<C32>()) as u64;
        let real_bytes = (n * batch_count * std::mem::size_of::<f32>()) as u64;

        let buf_in = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct2 real input"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_complex = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dct2 complex working buffer"),
            size: complex_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let combined_perm: Vec<u32> = (0..n).map(|i| digit_reverse_index(dct_interleave_index(i, n), factors) as u32).collect();
        let perm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct2 combined permutation"),
            contents: bytemuck::cast_slice(&combined_perm),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct2 fwd perm params"),
            contents: bytemuck::bytes_of(&PermParams { n: n as u32, batch_count: batch_count as u32, _pad0: 0, _pad1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        {
            let layout = self.dct_forward_permute_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dct2 fwd perm bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: perm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_complex.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dct2 fwd perm encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dct2 fwd perm pass"), timestamp_writes: None });
                pass.set_pipeline(&self.dct_forward_permute_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(((n * batch_count) as u32).div_ceil(256), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        self.run_stages_in_place(device, queue, &buf_complex, n, factors, batch_count, false);

        let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dct2 real output"),
            size: real_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct2 post params"),
            contents: bytemuck::bytes_of(&PermParams { n: n as u32, batch_count: batch_count as u32, _pad0: 0, _pad1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        {
            let layout = self.dct2_post_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dct2 post bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: post_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_complex.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dct2 post encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dct2 post pass"), timestamp_writes: None });
                pass.set_pipeline(&self.dct2_post_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(((n * batch_count) as u32).div_ceil(256), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        self.read_back_f32(device, queue, &buf_out, n * batch_count)
    }

    /// Real-input DCT-III via FFT, GPU version of `dct3_via_fft`.
    pub(crate) fn run_dct3(&self, device: &wgpu::Device, queue: &wgpu::Queue, f: &[f32], n: usize, factors: &[usize], batch_count: usize) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        assert_eq!(f.len(), n * batch_count);
        let complex_bytes = (n * batch_count * std::mem::size_of::<C32>()) as u64;
        let real_bytes = (n * batch_count * std::mem::size_of::<f32>()) as u64;

        let buf_in = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct3 real input"),
            contents: bytemuck::cast_slice(f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_natural = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dct3 natural-order complex buffer"),
            size: complex_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pre_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct3 pre params"),
            contents: bytemuck::bytes_of(&PermParams { n: n as u32, batch_count: batch_count as u32, _pad0: 0, _pad1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        {
            let layout = self.dct3_pre_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dct3 pre bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pre_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_natural.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dct3 pre encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dct3 pre pass"), timestamp_writes: None });
                pass.set_pipeline(&self.dct3_pre_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(((n * batch_count) as u32).div_ceil(256), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        // Plain digit-reversal (not the combined DCT table) before the
        // inverse FFT stages, matching dct3_via_fft calling
        // fft_mixed_radix directly on the natural-order reconstruction.
        let buf_reversed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dct3 digit-reversed complex buffer"),
            size: complex_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let plain_perm: Vec<u32> = (0..n).map(|i| digit_reverse_index(i, factors) as u32).collect();
        let plain_perm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct3 plain digit-reversal permutation"),
            contents: bytemuck::cast_slice(&plain_perm),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let perm_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct3 perm params"),
            contents: bytemuck::bytes_of(&PermParams { n: n as u32, batch_count: batch_count as u32, _pad0: 0, _pad1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        {
            let layout = self.permute_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dct3 digit reverse bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: perm_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: plain_perm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_natural.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_reversed.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dct3 digit reverse encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dct3 digit reverse pass"), timestamp_writes: None });
                pass.set_pipeline(&self.permute_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(((n * batch_count) as u32).div_ceil(256), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        self.run_stages_in_place(device, queue, &buf_reversed, n, factors, batch_count, true);

        let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dct3 real output"),
            size: real_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let gather_idx: Vec<u32> = (0..n).map(|i| dct_interleave_index(i, n) as u32).collect();
        let gather_idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct3 post gather index"),
            contents: bytemuck::cast_slice(&gather_idx),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let post_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dct3 post params"),
            contents: bytemuck::bytes_of(&PermParams { n: n as u32, batch_count: batch_count as u32, _pad0: 0, _pad1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        {
            let layout = self.dct3_post_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dct3 post bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: post_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gather_idx_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_reversed.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_out.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dct3 post encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dct3 post pass"), timestamp_writes: None });
                pass.set_pipeline(&self.dct3_post_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(((n * batch_count) as u32).div_ceil(256), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        self.read_back_f32(device, queue, &buf_out, n * batch_count)
    }

    /// Runs the full mixed-radix FFT (digit-reversal permutation followed
    /// by one dispatch per Cooley-Tukey stage, factors processed in
    /// reverse order exactly like `fft_mixed_radix`) on `data` - a
    /// concatenation of `batch_count` independent length-`n` complex
    /// sequences - blocking until the result is read back. `factors` must
    /// be `factor_mixed_radix(n)`'s result.
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[C32],
        n: usize,
        factors: &[usize],
        batch_count: usize,
        inverse: bool,
    ) -> Vec<C32> {
        assert_eq!(data.len(), n * batch_count);
        let total_elems = n * batch_count;
        let byte_len = (total_elems * std::mem::size_of::<C32>()) as u64;

        use wgpu::util::DeviceExt;
        let buf_in = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft input"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft working buffer"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let perm: Vec<u32> = (0..n).map(|i| digit_reverse_index(i, factors) as u32).collect();
        let perm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft digit-reverse permutation"),
            contents: bytemuck::cast_slice(&perm),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let perm_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft perm params"),
            contents: bytemuck::bytes_of(&PermParams { n: n as u32, batch_count: batch_count as u32, _pad0: 0, _pad1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fft encoder") });
        {
            let permute_layout = self.permute_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft permute bind group"),
                layout: &permute_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: perm_params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: perm_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_data.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fft permute pass"), timestamp_writes: None });
            pass.set_pipeline(&self.permute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (n * batch_count) as u32;
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        let sign = if inverse { 1.0 } else { -1.0 };
        let mut l = 1usize;
        for &r in factors.iter().rev() {
            let l_new = l * r;
            let groups = n / l_new;
            let work_per_batch = groups * l;
            let total_work = (work_per_batch * batch_count) as u32;

            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fft stage params"),
                contents: bytemuck::bytes_of(&StageParams {
                    n: n as u32,
                    l: l as u32,
                    radix: r as u32,
                    batch_count: batch_count as u32,
                    sign,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let stage_layout = self.stage_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft stage bind group"),
                layout: &stage_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_data.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fft stage encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fft stage pass"), timestamp_writes: None });
                pass.set_pipeline(&self.stage_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_work.div_ceil(256), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            l = l_new;
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fft readback encoder") });
        encoder.copy_buffer_to_buffer(&buf_data, 0, &staging, 0, byte_len);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).expect("gpu poll failed");
        rx.recv().expect("map channel closed").expect("buffer map failed");
        let mapped = slice.get_mapped_range().expect("failed to map gpu buffer");
        let result: Vec<C32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{num_complex::Complex32, FftPlanner};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn random_signal(n: usize, seed: &mut u64) -> Vec<C32> {
        (0..n).map(|_| C32::new(lcg(seed), lcg(seed))).collect()
    }

    /// Same skip-if-no-adapter tolerance as `gpu.rs`'s tests - GPU-adapter
    /// availability is legitimately absent in some sandboxes.
    fn skip_no_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }));
        let adapter = match adapter {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU FFT test: no adapter available ({e:#})");
                return None;
            }
        };
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wavefold fft test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            ..Default::default()
        })) {
            Ok(dq) => Some(dq),
            Err(e) => {
                eprintln!("skipping GPU FFT test: device request failed ({e:#})");
                None
            }
        }
    }

    #[test]
    fn gpu_fft_matches_rust_reference() {
        let Some((device, queue)) = skip_no_gpu() else { return };
        let gpu_fft = GpuFft::new(&device);

        let mut seed = 777u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            for &batch_count in &[1usize, 3] {
                let signal: Vec<C32> = (0..n * batch_count).map(|_| C32::new(lcg(&mut seed), lcg(&mut seed))).collect();

                for &inverse in &[false, true] {
                    let mut reference = signal.clone();
                    // fft_mixed_radix operates on one batch at a time.
                    for chunk in reference.chunks_mut(n) {
                        fft_mixed_radix(chunk, &factors, inverse);
                    }

                    let gpu_result = gpu_fft.run(&device, &queue, &signal, n, &factors, batch_count, inverse);

                    for (i, (g, r)) in gpu_result.iter().zip(reference.iter()).enumerate() {
                        assert!(
                            (g.re - r.re).abs() < 1e-1 && (g.im - r.im).abs() < 1e-1,
                            "n={n} batch_count={batch_count} inverse={inverse} index {i}: gpu={g:?} reference={r:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gpu_dct2_matches_rust_reference() {
        let Some((device, queue)) = skip_no_gpu() else { return };
        let gpu_fft = GpuFft::new(&device);

        let mut seed = 555u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            for &batch_count in &[1usize, 3] {
                let x: Vec<f32> = (0..n * batch_count).map(|_| lcg(&mut seed)).collect();

                let reference: Vec<f32> = x.chunks(n).flat_map(|chunk| dct2_via_fft(chunk, &factors)).collect();
                let gpu_result = gpu_fft.run_dct2(&device, &queue, &x, n, &factors, batch_count);

                for (i, (g, r)) in gpu_result.iter().zip(reference.iter()).enumerate() {
                    assert!((g - r).abs() < 1e-1, "n={n} batch_count={batch_count} index {i}: gpu={g} reference={r}");
                }
            }
        }
    }

    #[test]
    fn gpu_dct3_matches_rust_reference() {
        let Some((device, queue)) = skip_no_gpu() else { return };
        let gpu_fft = GpuFft::new(&device);

        let mut seed = 606u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            for &batch_count in &[1usize, 3] {
                let f: Vec<f32> = (0..n * batch_count).map(|_| lcg(&mut seed)).collect();

                let reference: Vec<f32> = f.chunks(n).flat_map(|chunk| dct3_via_fft(chunk, &factors)).collect();
                let gpu_result = gpu_fft.run_dct3(&device, &queue, &f, n, &factors, batch_count);

                for (i, (g, r)) in gpu_result.iter().zip(reference.iter()).enumerate() {
                    assert!((g - r).abs() < 1e-1, "n={n} batch_count={batch_count} index {i}: gpu={g} reference={r}");
                }
            }
        }
    }

    #[test]
    fn gpu_dct3_inverts_gpu_dct2() {
        let Some((device, queue)) = skip_no_gpu() else { return };
        let gpu_fft = GpuFft::new(&device);

        let mut seed = 707u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            let batch_count = 2usize;
            let x: Vec<f32> = (0..n * batch_count).map(|_| lcg(&mut seed)).collect();

            let f = gpu_fft.run_dct2(&device, &queue, &x, n, &factors, batch_count);
            let roundtrip = gpu_fft.run_dct3(&device, &queue, &f, n, &factors, batch_count);

            for (i, (o, r)) in x.iter().zip(roundtrip.iter()).enumerate() {
                assert!((o - r).abs() < 1e-1, "n={n} index {i}: orig={o} roundtrip={r}");
            }
        }
    }

    #[test]
    fn factor_mixed_radix_covers_common_video_dimensions() {
        assert_eq!(factor_mixed_radix(1920), Some(vec![2, 2, 2, 2, 2, 2, 2, 3, 5]));
        assert_eq!(factor_mixed_radix(1080), Some(vec![2, 2, 2, 3, 3, 3, 5]));
        assert_eq!(factor_mixed_radix(1280), Some(vec![2, 2, 2, 2, 2, 2, 2, 2, 5]));
        assert_eq!(factor_mixed_radix(720), Some(vec![2, 2, 2, 2, 3, 3, 5]));
        assert_eq!(factor_mixed_radix(3840), Some(vec![2, 2, 2, 2, 2, 2, 2, 2, 3, 5]));
        assert_eq!(factor_mixed_radix(2160), Some(vec![2, 2, 2, 2, 3, 3, 3, 5]));
    }

    #[test]
    fn factor_mixed_radix_rejects_large_prime_factors() {
        assert_eq!(factor_mixed_radix(1082), None); // 2 x 541, 541 prime
        assert_eq!(factor_mixed_radix(0), None);
        assert_eq!(factor_mixed_radix(1), None);
        assert_eq!(factor_mixed_radix(11), None); // 11 itself, prime > 7
    }

    #[test]
    fn forward_fft_matches_rustfft() {
        let mut seed = 12345u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            let signal = random_signal(n, &mut seed);

            let mut ours = signal.clone();
            fft_mixed_radix(&mut ours, &factors, false);

            let mut reference: Vec<Complex32> = signal.iter().map(|c| Complex32::new(c.re, c.im)).collect();
            let mut planner = FftPlanner::new();
            planner.plan_fft_forward(n).process(&mut reference);

            for (i, (o, r)) in ours.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (o.re - r.re).abs() < 1e-2 && (o.im - r.im).abs() < 1e-2,
                    "n={n} index {i}: ours={o:?} rustfft={r:?}"
                );
            }
        }
    }

    #[test]
    fn inverse_fft_matches_rustfft() {
        let mut seed = 999u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            let signal = random_signal(n, &mut seed);

            let mut ours = signal.clone();
            fft_mixed_radix(&mut ours, &factors, true);

            let mut reference: Vec<Complex32> = signal.iter().map(|c| Complex32::new(c.re, c.im)).collect();
            let mut planner = FftPlanner::new();
            planner.plan_fft_inverse(n).process(&mut reference);

            for (i, (o, r)) in ours.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (o.re - r.re).abs() < 1e-2 && (o.im - r.im).abs() < 1e-2,
                    "n={n} index {i}: ours={o:?} rustfft={r:?}"
                );
            }
        }
    }

    #[test]
    fn forward_then_inverse_recovers_original_scaled_by_n() {
        let mut seed = 42u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            let signal = random_signal(n, &mut seed);

            let mut roundtrip = signal.clone();
            fft_mixed_radix(&mut roundtrip, &factors, false);
            fft_mixed_radix(&mut roundtrip, &factors, true);

            for (i, (orig, rt)) in signal.iter().zip(roundtrip.iter()).enumerate() {
                let expected = C32::new(orig.re * n as f32, orig.im * n as f32);
                assert!(
                    (rt.re - expected.re).abs() < 1e-1 && (rt.im - expected.im).abs() < 1e-1,
                    "n={n} index {i}: roundtrip={rt:?} expected={expected:?}"
                );
            }
        }
    }

    fn dct_basis_matmul(x: &[f32], basis: &[f32], n: usize) -> Vec<f32> {
        (0..n).map(|k| (0..n).map(|i| x[i] * basis[k * n + i]).sum()).collect()
    }

    #[test]
    fn dct2_via_fft_matches_dct_basis_matmul() {
        let mut seed = 2024u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let basis = crate::dct_math::dct_basis(n);

            let expected = dct_basis_matmul(&x, &basis, n);
            let got = dct2_via_fft(&x, &factors);

            for (k, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!((g - e).abs() < 1e-2, "n={n} k={k}: got={g} expected={e}");
            }
        }
    }

    fn dct_basis_transpose_matmul(y: &[f32], basis: &[f32], n: usize) -> Vec<f32> {
        // x[n] = sum_k y[k] * B(k,n) = sum_k y[k] * basis[k*n_dim + n]
        (0..n).map(|pos| (0..n).map(|k| y[k] * basis[k * n + pos]).sum()).collect()
    }

    #[test]
    fn dct3_via_fft_matches_dct_basis_transpose_matmul() {
        // Deliberately arbitrary `y`, not a real dct2_via_fft output - a
        // round-trip-only test could hide compensating errors between the
        // forward and inverse constructions.
        let mut seed = 4096u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            let y: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let basis = crate::dct_math::dct_basis(n);

            let expected = dct_basis_transpose_matmul(&y, &basis, n);
            let got = dct3_via_fft(&y, &factors);

            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!((g - e).abs() < 1e-2, "n={n} i={i}: got={g} expected={e}");
            }
        }
    }

    #[test]
    fn dct3_inverts_dct2_exactly() {
        let mut seed = 8192u64;
        for &n in &[16usize, 24, 60, 210] {
            let factors = factor_mixed_radix(n).unwrap();
            let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();

            let f = dct2_via_fft(&x, &factors);
            let roundtrip = dct3_via_fft(&f, &factors);

            for (i, (o, r)) in x.iter().zip(roundtrip.iter()).enumerate() {
                assert!((o - r).abs() < 1e-2, "n={n} i={i}: orig={o} roundtrip={r}");
            }
        }
    }
}
