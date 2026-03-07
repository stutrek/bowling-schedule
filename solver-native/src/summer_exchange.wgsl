const ASSIGN_SIZE: u32 = 200u;

@group(0) @binding(0) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(1) var<storage, read_write> best_assignments: array<u32>;
@group(0) @binding(2) var<storage, read_write> costs: array<u32>;
@group(0) @binding(3) var<storage, read_write> best_costs: array<u32>;
@group(0) @binding(4) var<storage, read> swap_pairs: array<u32>;
@group(0) @binding(5) var<uniform> exchange_params: vec4<u32>;

@compute @workgroup_size(64)
fn exchange(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= exchange_params.x) { return; }

    let chain_a = swap_pairs[idx * 2u];
    let chain_b = swap_pairs[idx * 2u + 1u];
    let base_a = chain_a * ASSIGN_SIZE;
    let base_b = chain_b * ASSIGN_SIZE;

    for (var i = 0u; i < ASSIGN_SIZE; i++) {
        let tmp = assignments[base_a + i];
        assignments[base_a + i] = assignments[base_b + i];
        assignments[base_b + i] = tmp;
    }

    let tmp_cost = costs[chain_a];
    costs[chain_a] = costs[chain_b];
    costs[chain_b] = tmp_cost;

    if (costs[chain_a] < best_costs[chain_a]) {
        best_costs[chain_a] = costs[chain_a];
        for (var i = 0u; i < ASSIGN_SIZE; i++) {
            best_assignments[base_a + i] = assignments[base_a + i];
        }
    }
    if (costs[chain_b] < best_costs[chain_b]) {
        best_costs[chain_b] = costs[chain_b];
        for (var i = 0u; i < ASSIGN_SIZE; i++) {
            best_assignments[base_b + i] = assignments[base_b + i];
        }
    }
}
