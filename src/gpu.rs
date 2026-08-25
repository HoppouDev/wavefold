use crate::dct_backend::DctBackend;
use crate::dct_math::{dct_basis, transpose_square};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::time::Duration;
use tracing::debug;

/// Bound on a single `device.poll` wait. Large enough frames (see
/// `DctGpu` docs) can still push one dispatch past the driver's TDR
/// window despite `shader.wgsl`'s tiling, at which point the GPU is reset
/// out from under this process; `wgpu::PollType::wait_indefinitely()`
/// then blocks forever on a fence that will never signal, with both GPU
/// and CPU sitting fully idle - confirmed reproducing this exact silent
/// hang at 1920x1082 (well past the 640x480 size this backend already
/// warns about) before the tiling rewrite. A bounded wait turns that into
/// a clean, reported error instead.
const GPU_POLL_TIMEOUT: Duration = Duration::from_secs(30);
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    threshold: f32,
    apply_mask: u32,
    clamp_output: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU resources for one fixed (width, height, cutoff) combination, reused
/// across every `process_plane` call instead of being allocated fresh each
/// time. Buffers `a` and `b` are ping-ponged across the 4 passes of a
/// separable whole-frame 2D DCT (forward row, forward col + mask, inverse
/// col, inverse row).
///
/// Several fields are only read to build the bind groups above and never
/// touched again, but must stay alive for as long as those bind groups do.
#[allow(dead_code)]
struct PlaneBuffers {
    width: u32,
    height: u32,
    cutoff: f32,
    row_basis_buf: wgpu::Buffer,
    row_basis_t_buf: wgpu::Buffer,
    col_basis_buf: wgpu::Buffer,
    col_basis_t_buf: wgpu::Buffer,
    input_buf: wgpu::Buffer,
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    mask_params_buf: wgpu::Buffer, // backs `forward_mask`; rewritten in place on a quality-only change
    forward_plain: wgpu::BindGroup, // basis=B, apply_mask=0, clamp_output=0
    forward_mask: wgpu::BindGroup, // basis=B, apply_mask=1, clamp_output=0
    inverse_plain: wgpu::BindGroup, // basis=B^T, apply_mask=0, clamp_output=0
    inverse_clamp: wgpu::BindGroup, // basis=B^T, apply_mask=0, clamp_output=1
    io_input_to_a: wgpu::BindGroup, // src=input, dst=a
    io_a_to_b: wgpu::BindGroup,    // src=a, dst=b
    io_b_to_a: wgpu::BindGroup,    // src=b, dst=a
}

/// GPU-resident whole-frame DCT-II distortion effect.
/// Runs a separable 2D DCT-II over an *entire* color plane at once (not
/// 8x8 blocks), keeps only the lowest-frequency coefficients within a
/// diagonal cutoff, and reconstructs via inverse DCT — since the transform
/// spans the whole frame, dropping high frequencies produces global
/// ringing/ghosting rather than the localized blockiness of block-based
/// codecs like JPEG.
pub struct DctGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    row_pipeline: wgpu::ComputePipeline,
    col_pipeline: wgpu::ComputePipeline,
    params_layout: wgpu::BindGroupLayout,
    io_layout: wgpu::BindGroupLayout,
    // One independent buffer set per channel slot so `process_rgb` can
    // submit all 3 channels' work before blocking on any of them, instead
    // of forcing a full stall-and-resume round trip per channel — slots
    // must stay physically separate since each channel's GPU work is
    // in flight concurrently.
    buffers: [RefCell<Option<PlaneBuffers>>; 3],
}

impl DctGpu {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .context("no compatible GPU adapter found")?;
        let adapter_info = adapter.get_info();
        debug!(name = %adapter_info.name, backend = ?adapter_info.backend, "GPU adapter selected");

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("wavefold device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            }))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dct shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dct params layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let io_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dct io layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dct pipeline layout"),
            bind_group_layouts: &[Some(&params_layout), Some(&io_layout)],
            immediate_size: 0,
        });

        let make_pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let row_pipeline = make_pipeline("row_pass");
        let col_pipeline = make_pipeline("col_pass");

        Ok(Self {
            device,
            queue,
            row_pipeline,
            col_pipeline,
            params_layout,
            io_layout,
            buffers: [RefCell::new(None), RefCell::new(None), RefCell::new(None)],
        })
    }

    /// Runs the whole-frame forward+quantize+inverse DCT on one single-channel
    /// plane (row-major, f32 0..255). `cutoff` is the normalized diagonal
    /// frequency cutoff in `0.0..=2.0` (see shader.wgsl): 2.0 keeps the whole
    /// spectrum ((near-)lossless), values near 0 keep only the coefficients
    /// closest to DC, producing strong global ringing.
    pub fn process_plane(
        &self,
        pixels: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<Vec<f32>> {
        self.encode_plane(0, pixels, width, height, cutoff)?;
        let rx = self.begin_read(0)?;
        self.poll_bounded()?;
        self.finish_read(0, rx)
    }

    /// Same transform as `process_plane`, but for all three color planes of
    /// one frame at once: all three channels' work is submitted to the GPU,
    /// and all three `map_async` readbacks are issued, before blocking on
    /// any of them — one `device.poll` drives all three to completion
    /// instead of a full stall-and-resume round trip per channel. Only the
    /// single-plane `process_plane` API still does one submit per channel.
    pub fn process_rgb(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.encode_plane(0, r, width, height, cutoff)?;
        self.encode_plane(1, g, width, height, cutoff)?;
        self.encode_plane(2, b, width, height, cutoff)?;
        let rx0 = self.begin_read(0)?;
        let rx1 = self.begin_read(1)?;
        let rx2 = self.begin_read(2)?;
        self.poll_bounded()?;
        let r_out = self.finish_read(0, rx0)?;
        let g_out = self.finish_read(1, rx1)?;
        let b_out = self.finish_read(2, rx2)?;
        Ok((r_out, g_out, b_out))
    }

    /// Validates inputs, (re)builds the channel's cached buffers if needed,
    /// uploads `pixels`, and submits all 4 DCT passes — but does not block
    /// on the result. Pair with `read_plane` on the same `channel`.
    fn encode_plane(
        &self,
        channel: usize,
        pixels: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<()> {
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
        let size_bytes = (pixels.len() * std::mem::size_of::<f32>()) as u64;

        // row_basis/col_basis scale as O(width^2)/O(height^2); reject
        // dimensions that would overflow the GPU's storage buffer binding
        // limit instead of letting wgpu panic deep inside buffer creation.
        let max_binding = self.device.limits().max_storage_buffer_binding_size;
        let row_basis_bytes = width as u64 * width as u64 * std::mem::size_of::<f32>() as u64;
        let col_basis_bytes = height as u64 * height as u64 * std::mem::size_of::<f32>() as u64;
        if row_basis_bytes > max_binding || col_basis_bytes > max_binding {
            bail!(
                "process_plane: {width}x{height} frame needs a basis buffer larger than this GPU allows ({max_binding} bytes max); try a lower resolution"
            );
        }

        {
            let mut cache = self.buffers[channel].borrow_mut();
            let dims_match = matches!(&*cache, Some(b) if b.width == width && b.height == height);
            if !dims_match {
                // Dimensions changed (or first use): full rebuild, including
                // the O(width^2+height^2) basis matrices.
                debug!(
                    channel,
                    width, height, "rebuilding GPU basis buffers (dimensions changed)"
                );
                *cache = Some(self.build_plane_buffers(width, height, cutoff, size_bytes));
            } else if let Some(b) = cache.as_mut() {
                if b.cutoff != cutoff {
                    // Same frame size, only the cutoff knob moved: rewrite
                    // just the threshold in place, skip regenerating/
                    // re-uploading the (unchanged) basis matrices.
                    debug!(
                        channel,
                        cutoff, "updating cutoff threshold in place (dimensions unchanged)"
                    );
                    let threshold = cutoff.clamp(0.0, 2.0) + f32::EPSILON;
                    let params = Params {
                        width,
                        height,
                        threshold,
                        apply_mask: 1,
                        clamp_output: 0,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    self.queue
                        .write_buffer(&b.mask_params_buf, 0, bytemuck::bytes_of(&params));
                    b.cutoff = cutoff;
                }
            }
        }

        let cache = self.buffers[channel].borrow();
        let buffers = cache
            .as_ref()
            .context("encode_plane: buffer cache missing after build/update")?;

        self.queue
            .write_buffer(&buffers.input_buf, 0, bytemuck::cast_slice(pixels));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dct encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dct pass"),
                timestamp_writes: None,
            });
            // Must match shader.wgsl's `TILE`/`@workgroup_size`.
            const TILE: u32 = 16;
            let groups_x = width.div_ceil(TILE);
            let groups_y = height.div_ceil(TILE);

            // forward row transform: input -> a
            pass.set_pipeline(&self.row_pipeline);
            pass.set_bind_group(0, &buffers.forward_plain, &[]);
            pass.set_bind_group(1, &buffers.io_input_to_a, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);

            // forward col transform + quality mask: a -> b
            pass.set_pipeline(&self.col_pipeline);
            pass.set_bind_group(0, &buffers.forward_mask, &[]);
            pass.set_bind_group(1, &buffers.io_a_to_b, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);

            // inverse col transform: b -> a
            pass.set_pipeline(&self.col_pipeline);
            pass.set_bind_group(0, &buffers.inverse_plain, &[]);
            pass.set_bind_group(1, &buffers.io_b_to_a, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);

            // inverse row transform + clamp to pixel range: a -> b
            pass.set_pipeline(&self.row_pipeline);
            pass.set_bind_group(0, &buffers.inverse_clamp, &[]);
            pass.set_bind_group(1, &buffers.io_a_to_b, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        encoder.copy_buffer_to_buffer(&buffers.buf_b, 0, &buffers.staging_buf, 0, size_bytes);
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Blocks until the most recent submission completes, or `GPU_POLL_TIMEOUT`
    /// elapses - whichever comes first. A bounded wait instead of
    /// `wait_indefinitely()` so a driver-level GPU reset (Windows TDR, most
    /// likely at resolutions well beyond the 640x480 warning threshold)
    /// surfaces as a clear error instead of hanging this thread forever.
    fn poll_bounded(&self) -> Result<()> {
        match self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(GPU_POLL_TIMEOUT) }) {
            Ok(_) => Ok(()),
            Err(wgpu::PollError::Timeout) => bail!(
                "GPU did not respond within {GPU_POLL_TIMEOUT:?} - likely a driver reset (Windows TDR) from a too-slow DCT dispatch at this resolution; try --backend cpu or a lower resolution"
            ),
            Err(e) => Err(e).context("gpu poll failed"),
        }
    }

    /// Issues the async map request for `channel`'s previously-`encode_plane`'d
    /// staging buffer without blocking. Pair with `finish_read` on the same
    /// `channel` after a `device.poll` — splitting the request from the wait
    /// is what lets `process_rgb` issue all three channels' map requests
    /// before blocking on any of them.
    fn begin_read(
        &self,
        channel: usize,
    ) -> Result<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> {
        let cache = self.buffers[channel].borrow();
        let buffers = cache
            .as_ref()
            .context("begin_read: encode_plane must be called first")?;
        let slice = buffers.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        Ok(rx)
    }

    /// Waits for `channel`'s `begin_read` map request (already driven by a
    /// `device.poll`) to complete, then reads back and returns the result.
    fn finish_read(
        &self,
        channel: usize,
        rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Result<Vec<f32>> {
        rx.recv().context("gpu map channel closed")??;

        let cache = self.buffers[channel].borrow();
        let buffers = cache
            .as_ref()
            .context("finish_read: encode_plane must be called first")?;
        let slice = buffers.staging_buf.slice(..);
        let data = slice
            .get_mapped_range()
            .context("failed to map gpu buffer")?;
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffers.staging_buf.unmap();

        Ok(result)
    }

    fn build_plane_buffers(
        &self,
        width: u32,
        height: u32,
        cutoff: f32,
        size_bytes: u64,
    ) -> PlaneBuffers {
        let row_basis = dct_basis(width as usize);
        let row_basis_t = transpose_square(&row_basis, width as usize);
        let col_basis = dct_basis(height as usize);
        let col_basis_t = transpose_square(&col_basis, height as usize);
        let make_basis_buf = |label: &str, data: &[f32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let row_basis_buf = make_basis_buf("row basis", &row_basis);
        let row_basis_t_buf = make_basis_buf("row basis T", &row_basis_t);
        let col_basis_buf = make_basis_buf("col basis", &col_basis);
        let col_basis_t_buf = make_basis_buf("col basis T", &col_basis_t);

        let input_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input"),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("a"),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b"),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Normalized diagonal frequency cutoff: 0 keeps only DC, 2.0 keeps
        // the entire spectrum (the maximum possible rank at the highest
        // corner frequency). A small epsilon keeps cutoff=2.0 lossless
        // despite floating-point rounding at that boundary.
        let threshold = cutoff.clamp(0.0, 2.0) + f32::EPSILON;

        let make_params = |row_basis: &wgpu::Buffer,
                           col_basis: &wgpu::Buffer,
                           apply_mask: bool,
                           clamp_output: bool| {
            let params = Params {
                width,
                height,
                threshold,
                apply_mask: apply_mask as u32,
                clamp_output: clamp_output as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            // COPY_DST so a quality-only change (see the `b.quality != quality`
            // branch above) can rewrite just the threshold here instead of
            // forcing a full rebuild of the (much more expensive)
            // O(width^2+height^2) basis buffers, which don't depend on
            // quality at all.
            let buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dct params bind group"),
                layout: &self.params_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: row_basis.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: col_basis.as_entire_binding(),
                    },
                ],
            });
            (buf, bind_group)
        };
        // Only `forward_mask` actually reads `threshold` (the others always
        // have apply_mask=0), so it's the only params buffer a quality-only
        // update needs to touch.
        let (_, forward_plain) = make_params(&row_basis_buf, &col_basis_buf, false, false);
        let (mask_params_buf, forward_mask) =
            make_params(&row_basis_buf, &col_basis_buf, true, false);
        let (_, inverse_plain) = make_params(&row_basis_t_buf, &col_basis_t_buf, false, false);
        let (_, inverse_clamp) = make_params(&row_basis_t_buf, &col_basis_t_buf, false, true);

        let make_io = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dct io bind group"),
                layout: &self.io_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                ],
            })
        };
        let io_input_to_a = make_io(&input_buf, &buf_a);
        let io_a_to_b = make_io(&buf_a, &buf_b);
        let io_b_to_a = make_io(&buf_b, &buf_a);

        PlaneBuffers {
            width,
            height,
            cutoff,
            row_basis_buf,
            row_basis_t_buf,
            col_basis_buf,
            col_basis_t_buf,
            input_buf,
            buf_a,
            buf_b,
            staging_buf,
            mask_params_buf,
            forward_plain,
            forward_mask,
            inverse_plain,
            inverse_clamp,
            io_input_to_a,
            io_a_to_b,
            io_b_to_a,
        }
    }
}

/// Forwards to the inherent `process_rgb` above — lets `pipeline.rs` hold a
/// `Box<dyn DctBackend>` and treat GPU/CPU compute uniformly.
impl DctBackend for DctGpu {
    fn process_rgb(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
        cutoff: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        DctGpu::process_rgb(self, r, g, b, width, height, cutoff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_no_gpu() -> Option<DctGpu> {
        match DctGpu::new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping GPU test: no adapter available ({e:#})");
                None
            }
        }
    }

    #[test]
    fn full_cutoff_roundtrip_is_near_lossless() {
        let Some(gpu) = skip_no_gpu() else { return };
        let (w, h) = (24u32, 18u32);
        // deterministic pseudo-random-ish pattern, no external RNG dependency
        let plane: Vec<f32> = (0..(w * h)).map(|i| ((i * 37 + 11) % 256) as f32).collect();

        let out = gpu.process_plane(&plane, w, h, 2.0).unwrap();
        assert_eq!(out.len(), plane.len());
        for (a, b) in plane.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1.0, "roundtrip drift too large: {a} vs {b}");
        }
    }

    #[test]
    fn very_low_cutoff_collapses_toward_the_frame_average() {
        let Some(gpu) = skip_no_gpu() else { return };
        let (w, h) = (16u32, 16u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| (i % 256) as f32).collect();
        let mean = plane.iter().sum::<f32>() / plane.len() as f32;

        let out = gpu.process_plane(&plane, w, h, 0.02).unwrap();
        for v in out {
            assert!(
                (v - mean).abs() < 5.0,
                "near-DC-only output {v} should be close to frame mean {mean}"
            );
        }
    }

    #[test]
    fn lower_cutoff_reduces_output_variance() {
        let Some(gpu) = skip_no_gpu() else { return };
        let (w, h) = (16u32, 16u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| ((i * 53) % 256) as f32).collect();

        let variance = |v: &[f32]| {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32
        };

        let high_c = gpu.process_plane(&plane, w, h, 1.8).unwrap();
        let low_c = gpu.process_plane(&plane, w, h, 0.2).unwrap();
        assert!(
            variance(&low_c) < variance(&high_c),
            "low cutoff should suppress detail more: var(low)={} var(high)={}",
            variance(&low_c),
            variance(&high_c)
        );
    }

    #[test]
    fn handles_one_pixel_wide_and_tall_frames() {
        let Some(gpu) = skip_no_gpu() else { return };
        // Exercises the mask-cutoff denominator guard (shader.wgsl:
        // `max(params.width, 2u) - 1u`) for the width==1 / height==1 edge
        // case, where a naive `width - 1u` would divide by zero.
        for (w, h) in [(1u32, 9u32), (9u32, 1u32), (1u32, 1u32)] {
            let plane: Vec<f32> = (0..(w * h)).map(|i| (i * 17 % 256) as f32).collect();
            let out = gpu.process_plane(&plane, w, h, 1.0).unwrap();
            assert_eq!(out.len(), plane.len());
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{w}x{h} produced a non-finite pixel"
            );
        }
    }

    #[test]
    fn rejects_zero_dimensions_and_mismatched_pixel_buffer() {
        let Some(gpu) = skip_no_gpu() else { return };
        assert!(gpu.process_plane(&[], 0, 4, 1.0).is_err());
        assert!(gpu.process_plane(&[], 4, 0, 1.0).is_err());
        let wrong_len = vec![0f32; 3];
        assert!(gpu.process_plane(&wrong_len, 4, 4, 1.0).is_err());
    }
}
