//! Common GPU SA dispatch operations.
//!
//! Reusable functions for GPU dispatch/readback, replica exchange, and
//! chain read/write. All operate on `GpuResources` + raw `u32` arrays —
//! no solver-specific types.

use crate::gpu_setup::GpuResources;
use crate::gpu_types::*;
use rand::rngs::SmallRng;
use rand::Rng;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Submit SA compute dispatch, poll GPU, read back best_costs and current_costs.
/// Returns `None` on timeout or GPU error.
pub fn dispatch_and_readback(
    gpu: &GpuResources,
    chain_count: u32,
    shutdown: &AtomicBool,
    dispatch_count: u64,
) -> Option<(Vec<u32>, Vec<u32>)> {
    let dispatch_start = Instant::now();
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SA"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SA Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.sa_pipeline);
        pass.set_bind_group(0, &gpu.sa_bg, &[]);
        pass.dispatch_workgroups(gpu.sa_workgroups, 1, 1);
    }
    let per_array_size = chain_count as u64 * 4;
    encoder.copy_buffer_to_buffer(&gpu.best_cost_buf, 0, &gpu.costs_readback_buf, 0, per_array_size);
    encoder.copy_buffer_to_buffer(&gpu.cost_buf, 0, &gpu.costs_readback_buf, per_array_size, per_array_size);
    gpu.queue.submit(Some(encoder.finish()));

    let costs_slice = gpu.costs_readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    costs_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });

    let poll_deadline = Instant::now() + std::time::Duration::from_secs(30);
    let mut poll_ok = false;
    loop {
        match gpu.device.poll(wgpu::PollType::Poll) {
            Ok(status) if status.is_queue_empty() => { poll_ok = true; break; }
            Ok(_) => {
                if shutdown.load(Ordering::Relaxed) { break; }
                if Instant::now() > poll_deadline {
                    eprintln!("GPU poll timed out after 30s at dispatch {} — device may be lost", dispatch_count);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(e) => {
                eprintln!("GPU poll error at dispatch {}: {:?}", dispatch_count, e);
                break;
            }
        }
    }
    if !poll_ok {
        gpu.costs_readback_buf.unmap();
        return None;
    }
    if rx.recv_timeout(std::time::Duration::from_secs(5)).is_err() {
        eprintln!("GPU map_async recv timed out at dispatch {}", dispatch_count);
        gpu.costs_readback_buf.unmap();
        return None;
    }
    if dispatch_count < 5 {
        eprintln!("dispatch {} completed in {}ms", dispatch_count, dispatch_start.elapsed().as_millis());
    }

    let n = chain_count as usize;
    let (best_costs, current_costs) = {
        let data = costs_slice.get_mapped_range();
        let all: &[u32] = bytemuck::cast_slice(&data);
        (all[..n].to_vec(), all[n..n * 2].to_vec())
    };
    gpu.costs_readback_buf.unmap();

    Some((best_costs, current_costs))
}

/// Execute replica exchange, restricted to within-island boundaries.
/// `island_size` = number of chains per island (e.g., 512). Pods in different islands never exchange.
/// Pass `island_size = chain_count` for unrestricted (original behavior).
/// Returns `(attempts, swaps_executed)`.
pub fn replica_exchange(
    gpu: &GpuResources,
    current_costs: &[u32],
    chain_count: u32,
    island_size: usize,
    temp_fn: fn(usize) -> f64,
    locked_chains: &HashSet<usize>,
    rng: &mut SmallRng,
    dispatch_count: u64,
) -> (u64, u64) {
    let parity = (dispatch_count % 2) as usize;
    let num_workgroups = chain_count as usize / TEMP_LEVELS;
    let pods_per_wg = TEMP_LEVELS / POD_SIZE;
    let mut swap_pairs: Vec<u32> = Vec::new();
    let mut attempts = 0u64;

    for wg in 0..num_workgroups {
        let wg_base = wg * TEMP_LEVELS;
        for pod in 0..pods_per_wg {
            let pod_base = wg_base + pod * POD_SIZE;

            // Check if this pod's chains would cross an island boundary
            let island_a = pod_base / island_size;
            let island_b = (pod_base + POD_SIZE - 1) / island_size;
            if island_a != island_b {
                continue; // Pod spans island boundary, skip entirely
            }

            let mut level = parity;
            while level + 1 < POD_SIZE {
                let chain_a = pod_base + level;
                let chain_b = pod_base + level + 1;
                if locked_chains.contains(&chain_a) || locked_chains.contains(&chain_b) {
                    level += 2;
                    continue;
                }
                let cost_a = current_costs[chain_a] as f64;
                let cost_b = current_costs[chain_b] as f64;
                let temp_a = temp_fn(level);
                let temp_b = temp_fn(level + 1);
                let delta = (1.0 / temp_a - 1.0 / temp_b) * (cost_a - cost_b);
                attempts += 1;
                if delta >= 0.0 || rng.random::<f64>() < delta.exp() {
                    swap_pairs.push(chain_a as u32);
                    swap_pairs.push(chain_b as u32);
                }
                level += 2;
            }
        }
    }

    let num_swaps = swap_pairs.len() / 2;

    if num_swaps > 0 {
        let params_data: [u32; 4] = [num_swaps as u32, 0, 0, 0];
        gpu.queue.write_buffer(&gpu.exchange_params_buf, 0, bytemuck::cast_slice(&params_data));
        gpu.queue.write_buffer(&gpu.swap_pairs_buf, 0, bytemuck::cast_slice(&swap_pairs));

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Exchange"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Exchange Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.exchange_pipeline);
            pass.set_bind_group(0, &gpu.exchange_bg, &[]);
            pass.dispatch_workgroups(((num_swaps as u32) + 63) / 64, 1, 1);
        }
        gpu.queue.submit(Some(encoder.finish()));
    }

    (attempts, num_swaps as u64)
}

/// Read one chain's packed assignment from GPU (blocking readback).
/// Returns the raw packed `u32` array, or `None` on timeout.
pub fn read_chain_raw(
    gpu: &GpuResources,
    chain_idx: usize,
    assign_u32s: usize,
) -> Option<Vec<u32>> {
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Read Chain"),
    });
    let offset = chain_idx as u64 * assign_u32s as u64 * 4;
    encoder.copy_buffer_to_buffer(
        &gpu.best_assign_buf, offset,
        &gpu.assign_readback_buf, 0,
        (assign_u32s * 4) as u64,
    );
    gpu.queue.submit(Some(encoder.finish()));

    let slice = gpu.assign_readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match gpu.device.poll(wgpu::PollType::Poll) {
            Ok(status) if status.is_queue_empty() => break,
            Ok(_) => {
                if Instant::now() > deadline {
                    gpu.assign_readback_buf.unmap();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(_) => { gpu.assign_readback_buf.unmap(); return None; }
        }
    }
    if rx.recv_timeout(std::time::Duration::from_secs(5)).is_err() {
        gpu.assign_readback_buf.unmap();
        return None;
    }
    let packed = {
        let data = slice.get_mapped_range();
        let s: &[u32] = bytemuck::cast_slice(&data);
        s.to_vec()
    };
    gpu.assign_readback_buf.unmap();
    Some(packed)
}

/// Write one chain's packed assignment + cost to GPU buffers (both current and best).
pub fn write_chain_raw(
    gpu: &GpuResources,
    chain_idx: usize,
    packed: &[u32],
    cost: u32,
    assign_u32s: usize,
) {
    let offset_assign = (chain_idx * assign_u32s * 4) as u64;
    let offset_cost = (chain_idx * 4) as u64;
    gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, bytemuck::cast_slice(packed));
    gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, bytemuck::cast_slice(packed));
    gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, bytemuck::bytes_of(&cost));
    gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, bytemuck::bytes_of(&cost));
}

/// Write a contiguous range of chains in bulk (2 write_buffer calls instead of 4×N).
/// `chain_start` is the first chain index. `assign_data` is packed assignments concatenated,
/// `cost_data` is the cost for each chain. Both slices must have the same chain count.
pub fn write_island_raw(
    gpu: &GpuResources,
    chain_start: usize,
    assign_data: &[u32],
    cost_data: &[u32],
    assign_u32s: usize,
) {
    let offset_assign = (chain_start * assign_u32s * 4) as u64;
    let offset_cost = (chain_start * 4) as u64;
    let assign_bytes = bytemuck::cast_slice(assign_data);
    let cost_bytes = bytemuck::cast_slice(cost_data);
    gpu.queue.write_buffer(&gpu.assign_buf, offset_assign, assign_bytes);
    gpu.queue.write_buffer(&gpu.best_assign_buf, offset_assign, assign_bytes);
    gpu.queue.write_buffer(&gpu.cost_buf, offset_cost, cost_bytes);
    gpu.queue.write_buffer(&gpu.best_cost_buf, offset_cost, cost_bytes);
}
