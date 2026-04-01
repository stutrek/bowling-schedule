//! Island pool: manages 128 independent GPU islands with similarity-based
//! deduplication and stagnation eviction.

use crate::gpu_types_winter_fixed::WF_ASSIGN_U32S;
use solver_core::winter_fixed::{WF_TEAMS, WF_WEEKS, WF_POSITIONS};
use rand::rngs::SmallRng;
use rand::Rng;

pub const STAGNATION_DISPATCHES: u64 = 5000;
pub const DEDUP_INTERVAL: u64 = 500;
pub const REFINEMENT_ITERS: u64 = 10_000_000;

const ADAPT_INTERVAL: u64 = 100;
const MIN_DISTANCE_FLOOR: f64 = 2.0;
const MIN_DISTANCE_CEIL: f64 = 80.0;
const TARGET_OCCUPANCY: f64 = 0.75;
const DISTANCE_CROWD_FACTOR: f64 = 1.5;

/// Number of mapping u32s (excludes the flags word at index 24).
const MAPPING_U32S: usize = WF_ASSIGN_U32S - 1; // 24

pub struct IslandMeta {
    pub best_packed: [u32; WF_ASSIGN_U32S],
    pub best_cost: u32,
    pub normalized: [u32; MAPPING_U32S],
    pub last_improved_dispatch: u64,
    pub times_refined: u32,
    pub chain_start: usize,
}

pub struct IslandPool {
    pub islands: Vec<IslandMeta>,
    pub num_islands: usize,
    pub island_size: usize,
    pub min_distance: f64,
    pub dedup_resets: u64,
    pub stagnation_resets: u64,
    last_adapt_dispatch: u64,
    last_dedup_dispatch: u64,
}

impl IslandPool {
    pub fn new(num_islands: usize, island_size: usize, initial_min_distance: f64) -> Self {
        let islands = (0..num_islands)
            .map(|i| IslandMeta {
                best_packed: [0u32; WF_ASSIGN_U32S],
                best_cost: u32::MAX,
                normalized: [0u32; MAPPING_U32S],
                last_improved_dispatch: 0,
                times_refined: 0,
                chain_start: i * island_size,
            })
            .collect();
        IslandPool {
            islands,
            num_islands,
            island_size,
            min_distance: initial_min_distance,
            dedup_resets: 0,
            stagnation_resets: 0,
            last_adapt_dispatch: 0,
            last_dedup_dispatch: 0,
        }
    }

    /// Update an island's best if the new cost is better.
    /// Returns true if the island's best was updated.
    pub fn update_island_best(
        &mut self,
        island_idx: usize,
        packed: &[u32; WF_ASSIGN_U32S],
        cost: u32,
        dispatch: u64,
    ) -> bool {
        let island = &mut self.islands[island_idx];
        if cost < island.best_cost {
            island.best_packed = *packed;
            island.best_cost = cost;
            island.normalized = normalize_packed(packed);
            island.last_improved_dispatch = dispatch;
            true
        } else {
            false
        }
    }

    /// Pick the island most in need of refinement: lowest `times_refined`, ties by best cost.
    /// Skips islands currently being refined (passed as `busy` set).
    pub fn pick_for_refinement(&self, busy: &[usize]) -> Option<usize> {
        self.islands
            .iter()
            .enumerate()
            .filter(|(idx, island)| {
                island.best_cost < u32::MAX && !busy.contains(idx)
            })
            .min_by_key(|(_, island)| (island.times_refined, island.best_cost))
            .map(|(idx, _)| idx)
    }

    pub fn mark_refined(&mut self, island_idx: usize) {
        self.islands[island_idx].times_refined += 1;
    }

    /// Find pairs of islands whose normalized bests are within `min_distance`.
    /// Returns (worse_island_idx, better_island_idx, distance) for each duplicate pair.
    pub fn find_duplicates(&self) -> Vec<(usize, usize, u32)> {
        let threshold = self.min_distance as u32;
        let mut dupes = Vec::new();
        for i in 0..self.num_islands {
            if self.islands[i].best_cost == u32::MAX { continue; }
            for j in (i + 1)..self.num_islands {
                if self.islands[j].best_cost == u32::MAX { continue; }
                let d = nibble_distance(&self.islands[i].normalized, &self.islands[j].normalized);
                if d < threshold {
                    let (worse, better) = if self.islands[i].best_cost >= self.islands[j].best_cost {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    dupes.push((worse, better, d));
                }
            }
        }
        dupes
    }

    /// Return island indices that haven't improved in STAGNATION_DISPATCHES.
    pub fn find_stagnant(&self, dispatch_count: u64) -> Vec<usize> {
        self.islands
            .iter()
            .enumerate()
            .filter(|(_, island)| {
                island.best_cost < u32::MAX
                    && dispatch_count > island.last_improved_dispatch + STAGNATION_DISPATCHES
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Reset an island (caller should also reset GPU chains).
    pub fn reset_island(&mut self, island_idx: usize) {
        let island = &mut self.islands[island_idx];
        island.best_packed = [0u32; WF_ASSIGN_U32S];
        island.best_cost = u32::MAX;
        island.normalized = [0u32; MAPPING_U32S];
        island.last_improved_dispatch = 0;
        island.times_refined = 0;
    }

    /// Run periodic dedup + stagnation checks. Returns list of islands that were reset.
    pub fn periodic_maintenance(&mut self, dispatch_count: u64, busy: &[usize]) -> Vec<(usize, &'static str)> {
        let mut resets = Vec::new();

        if dispatch_count >= self.last_dedup_dispatch + DEDUP_INTERVAL {
            self.last_dedup_dispatch = dispatch_count;

            // Dedup — only reset each island once per round
            let dupes = self.find_duplicates();
            let mut already_reset = std::collections::HashSet::new();
            for (worse, _better, _d) in &dupes {
                if !busy.contains(worse) && !already_reset.contains(worse) {
                    self.reset_island(*worse);
                    self.dedup_resets += 1;
                    resets.push((*worse, "dedup"));
                    already_reset.insert(*worse);
                }
            }

            // Stagnation
            let stagnant = self.find_stagnant(dispatch_count);
            for idx in stagnant {
                if !busy.contains(&idx) && !resets.iter().any(|(i, _)| *i == idx) {
                    self.reset_island(idx);
                    self.stagnation_resets += 1;
                    resets.push((idx, "stagnation"));
                }
            }
        }

        resets
    }

    /// Adaptive min_distance tuning. Call every dispatch; acts every ADAPT_INTERVAL.
    pub fn maybe_adapt(&mut self, dispatch_count: u64, rng: &mut SmallRng) {
        if dispatch_count < self.last_adapt_dispatch + ADAPT_INTERVAL {
            return;
        }
        self.last_adapt_dispatch = dispatch_count;

        let active: Vec<usize> = self.islands
            .iter()
            .enumerate()
            .filter(|(_, island)| island.best_cost < u32::MAX)
            .map(|(idx, _)| idx)
            .collect();

        let occupancy = active.len() as f64 / self.num_islands as f64;
        let avg_d = self.sampled_avg_pairwise_distance(&active, rng);

        let old = self.min_distance;
        if occupancy < TARGET_OCCUPANCY {
            // Decay faster the worse occupancy is
            let decay = 0.5 + 0.4 * (occupancy / TARGET_OCCUPANCY);
            self.min_distance *= decay;
        } else if avg_d < self.min_distance * DISTANCE_CROWD_FACTOR {
            self.min_distance *= 1.1;
        }
        self.min_distance = self.min_distance.clamp(MIN_DISTANCE_FLOOR, MIN_DISTANCE_CEIL);

        if (self.min_distance - old).abs() > 0.01 {
            eprintln!(
                "  ADAPT min_distance: {:.1} → {:.1} (occ={:.0}% avg_d={:.1})",
                old, self.min_distance, occupancy * 100.0, avg_d,
            );
        }
    }

    /// Average pairwise distance across active islands (sampled if >30).
    pub fn sampled_avg_pairwise_distance(&self, active: &[usize], rng: &mut SmallRng) -> f64 {
        if active.len() < 2 { return f64::MAX; }
        let sample_pairs = 200usize;
        let mut total = 0u64;
        let mut count = 0u64;
        if active.len() <= 30 {
            // Exhaustive
            for i in 0..active.len() {
                for j in (i + 1)..active.len() {
                    total += nibble_distance(
                        &self.islands[active[i]].normalized,
                        &self.islands[active[j]].normalized,
                    ) as u64;
                    count += 1;
                }
            }
        } else {
            // Sampled
            for _ in 0..sample_pairs {
                let i = rng.random_range(0..active.len());
                let mut j = rng.random_range(0..active.len() - 1);
                if j >= i { j += 1; }
                total += nibble_distance(
                    &self.islands[active[i]].normalized,
                    &self.islands[active[j]].normalized,
                ) as u64;
                count += 1;
            }
        }
        if count == 0 { f64::MAX } else { total as f64 / count as f64 }
    }

    /// Stats for display.
    pub fn stats(&self, dispatch_count: u64) -> IslandPoolStats {
        let mut active = 0usize;
        let mut stagnant = 0usize;
        let mut best_cost = u32::MAX;
        let mut worst_cost = 0u32;
        let mut cost_sum = 0u64;

        for island in &self.islands {
            if island.best_cost < u32::MAX {
                active += 1;
                if island.best_cost < best_cost { best_cost = island.best_cost; }
                if island.best_cost > worst_cost { worst_cost = island.best_cost; }
                cost_sum += island.best_cost as u64;
                if dispatch_count > island.last_improved_dispatch + STAGNATION_DISPATCHES {
                    stagnant += 1;
                }
            }
        }

        IslandPoolStats {
            active,
            stagnant,
            dedup_resets: self.dedup_resets,
            stagnation_resets: self.stagnation_resets,
            best_cost,
            worst_cost,
            avg_cost: if active > 0 { cost_sum as f64 / active as f64 } else { 0.0 },
            min_distance: self.min_distance,
        }
    }
}

pub struct IslandPoolStats {
    pub active: usize,
    pub stagnant: usize,
    pub dedup_resets: u64,
    pub stagnation_resets: u64,
    pub best_cost: u32,
    pub worst_cost: u32,
    pub avg_cost: f64,
    pub min_distance: f64,
}

// ── Distance functions ──

/// Normalize team labels in a packed schedule so the first team encountered
/// gets label 0, the second gets label 1, etc. This makes two schedules
/// that differ only by team relabeling have identical normalized forms.
///
/// Operates on the mapping portion only (first 24 u32s, each containing
/// 8 × 4-bit nibbles = 8 team labels).
pub fn normalize_packed(packed: &[u32; WF_ASSIGN_U32S]) -> [u32; MAPPING_U32S] {
    // Build the permutation by scanning nibbles left to right
    let mut perm = [0xFFu8; WF_TEAMS]; // 0xFF = unassigned
    let mut next_label: u8 = 0;

    for w in 0..WF_WEEKS {
        for pos in 0..WF_POSITIONS {
            let u32_idx = w * 2 + pos / 8;
            let shift = (pos % 8) * 4;
            let team = ((packed[u32_idx] >> shift) & 0xF) as u8;
            if perm[team as usize] == 0xFF {
                perm[team as usize] = next_label;
                next_label += 1;
                if next_label as usize >= WF_TEAMS { break; }
            }
        }
        if next_label as usize >= WF_TEAMS { break; }
    }

    // Apply permutation
    let mut result = [0u32; MAPPING_U32S];
    for w in 0..WF_WEEKS {
        for pos in 0..WF_POSITIONS {
            let u32_idx = w * 2 + pos / 8;
            let shift = (pos % 8) * 4;
            let team = ((packed[u32_idx] >> shift) & 0xF) as u8;
            let new_label = if (team as usize) < WF_TEAMS && perm[team as usize] != 0xFF {
                perm[team as usize]
            } else {
                team
            };
            result[u32_idx] |= (new_label as u32 & 0xF) << shift;
        }
    }

    result
}

/// Hamming distance on 4-bit nibbles between two normalized packed arrays.
/// Each differing nibble = one team-position slot that differs.
/// Maximum possible = 192 (12 weeks × 16 positions).
pub fn nibble_distance(a: &[u32; MAPPING_U32S], b: &[u32; MAPPING_U32S]) -> u32 {
    let mut dist = 0u32;
    for i in 0..MAPPING_U32S {
        let xor = a[i] ^ b[i];
        if xor == 0 { continue; }
        // Count non-zero nibbles: OR all 4 bits of each nibble into the low bit,
        // then popcount on the masked result.
        let nz = (xor | (xor >> 1) | (xor >> 2) | (xor >> 3)) & 0x11111111;
        dist += nz.count_ones();
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_nibble_distance_identical() {
        let a = [0u32; MAPPING_U32S];
        let b = [0u32; MAPPING_U32S];
        assert_eq!(nibble_distance(&a, &b), 0);
    }

    #[test]
    fn test_nibble_distance_one_diff() {
        let mut a = [0u32; MAPPING_U32S];
        let mut b = [0u32; MAPPING_U32S];
        // Differ in one nibble (position 0 of word 0)
        a[0] = 0x00000001;
        b[0] = 0x00000002;
        assert_eq!(nibble_distance(&a, &b), 1);
    }

    #[test]
    fn test_nibble_distance_all_diff() {
        let mut a = [0u32; MAPPING_U32S];
        let mut b = [0u32; MAPPING_U32S];
        for i in 0..MAPPING_U32S {
            a[i] = 0x01010101; // nibbles: 1,0,1,0,1,0,1,0
            b[i] = 0x20202020; // nibbles: 0,2,0,2,0,2,0,2
        }
        // Every nibble differs
        assert_eq!(nibble_distance(&a, &b), (MAPPING_U32S * 8) as u32);
    }

    #[test]
    fn test_normalize_identity_permutation() {
        // Schedule where team labels already appear in order 0,1,2,...,15
        let mut packed = [0u32; WF_ASSIGN_U32S];
        for pos in 0..WF_POSITIONS {
            let u32_idx = pos / 8;
            let shift = (pos % 8) * 4;
            packed[u32_idx] |= (pos as u32 & 0xF) << shift;
        }
        // Fill remaining weeks with same pattern
        for w in 1..WF_WEEKS {
            packed[w * 2] = packed[0];
            packed[w * 2 + 1] = packed[1];
        }
        let norm = normalize_packed(&packed);
        // Should be unchanged (labels already canonical)
        for i in 0..MAPPING_U32S {
            assert_eq!(norm[i], packed[i], "word {} differs", i);
        }
    }

    #[test]
    fn test_normalize_swapped_teams() {
        // Two schedules that differ only by swapping team 0 and team 1
        let mut packed_a = [0u32; WF_ASSIGN_U32S];
        let mut packed_b = [0u32; WF_ASSIGN_U32S];

        for w in 0..WF_WEEKS {
            for pos in 0..WF_POSITIONS {
                let team = pos as u8;
                let u32_idx = w * 2 + pos / 8;
                let shift = (pos % 8) * 4;
                packed_a[u32_idx] |= (team as u32 & 0xF) << shift;
                // Swap teams 0 and 1
                let swapped = if team == 0 { 1 } else if team == 1 { 0 } else { team };
                packed_b[u32_idx] |= (swapped as u32 & 0xF) << shift;
            }
        }

        let norm_a = normalize_packed(&packed_a);
        let norm_b = normalize_packed(&packed_b);
        assert_eq!(nibble_distance(&norm_a, &norm_b), 0, "swapped teams should normalize to same form");
    }

    #[test]
    fn test_pool_pick_for_refinement() {
        let mut pool = IslandPool::new(20.0);
        pool.islands[0].best_cost = 500;
        pool.islands[0].times_refined = 3;
        pool.islands[1].best_cost = 400;
        pool.islands[1].times_refined = 1;
        pool.islands[2].best_cost = 600;
        pool.islands[2].times_refined = 1;

        // Should pick island 1 (times_refined=1, cost=400 beats island 2's 600)
        assert_eq!(pool.pick_for_refinement(&[]), Some(1));
        // If island 1 is busy, pick island 2
        assert_eq!(pool.pick_for_refinement(&[1]), Some(2));
    }

    #[test]
    fn test_stagnation_detection() {
        let mut pool = IslandPool::new(20.0);
        pool.islands[5].best_cost = 500;
        pool.islands[5].last_improved_dispatch = 100;
        pool.islands[10].best_cost = 600;
        pool.islands[10].last_improved_dispatch = 4000;

        let stagnant = pool.find_stagnant(5200);
        assert!(stagnant.contains(&5), "island 5 should be stagnant (5200 - 100 > 5000)");
        assert!(!stagnant.contains(&10), "island 10 should not be stagnant (5200 - 4000 < 5000)");
    }

    #[test]
    fn test_find_duplicates() {
        let mut pool = IslandPool::new(20.0);

        // Set two islands to identical normalized forms
        let mut packed = [0u32; WF_ASSIGN_U32S];
        for pos in 0..WF_POSITIONS {
            let u32_idx = pos / 8;
            let shift = (pos % 8) * 4;
            packed[u32_idx] |= (pos as u32 & 0xF) << shift;
        }
        for w in 1..WF_WEEKS {
            packed[w * 2] = packed[0];
            packed[w * 2 + 1] = packed[1];
        }

        pool.update_island_best(0, &packed, 400, 100);
        pool.update_island_best(1, &packed, 500, 100);

        let dupes = pool.find_duplicates();
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].0, 1, "worse island (cost 500) should be first");
        assert_eq!(dupes[0].1, 0, "better island (cost 400) should be second");
        assert_eq!(dupes[0].2, 0, "distance should be 0 for identical schedules");
    }

    #[test]
    fn test_adaptive_distance() {
        let mut pool = IslandPool::new(20.0);
        let mut rng = SmallRng::seed_from_u64(42);

        // With no active islands, occupancy is 0 → should decrease
        pool.maybe_adapt(ADAPT_INTERVAL, &mut rng);
        assert!(pool.min_distance < 20.0, "should decrease with low occupancy");
    }
}
