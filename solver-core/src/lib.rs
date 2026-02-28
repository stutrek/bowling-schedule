use rand::rngs::SmallRng;
use rand::Rng;
use serde::Deserialize;

pub const TEAMS: usize = 16;
pub const LANES: usize = 4;
pub const WEEKS: usize = 12;
pub const QUADS: usize = 4;
pub const POS: usize = 4;

pub type Assignment = [[[u8; POS]; QUADS]; WEEKS];

#[derive(Deserialize, Clone)]
pub struct Weights {
    pub matchup_zero: u32,
    pub matchup_triple: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: f64,
    pub early_late_alternation: u32,
    pub lane_balance: f64,
    pub lane_switch: f64,
    pub late_lane_balance: f64,
    pub commissioner_overlap: u32,
}

#[derive(Clone)]
pub struct CostBreakdown {
    pub matchup_balance: u32,
    pub consecutive_opponents: u32,
    pub early_late_balance: u32,
    pub early_late_alternation: u32,
    pub lane_balance: u32,
    pub lane_switch_balance: u32,
    pub late_lane_balance: u32,
    pub commissioner_overlap: u32,
    pub total: u32,
}

pub fn random_assignment(rng: &mut SmallRng) -> Assignment {
    let mut a = [[[0u8; POS]; QUADS]; WEEKS];
    for w in 0..WEEKS {
        let mut teams: [u8; TEAMS] = std::array::from_fn(|i| i as u8);
        for i in (1..TEAMS).rev() {
            let j = rng.random_range(0..=i);
            teams.swap(i, j);
        }
        for q in 0..QUADS {
            for p in 0..POS {
                a[w][q][p] = teams[q * POS + p];
            }
        }
    }
    a
}

pub fn evaluate(a: &Assignment, w8: &Weights) -> CostBreakdown {
    let mut matchups = [0i32; TEAMS * TEAMS];
    let mut week_matchup = [0u8; WEEKS * TEAMS * TEAMS];
    let mut lane_counts = [0i32; TEAMS * LANES];
    let mut late_lane_counts = [0i32; TEAMS * LANES];
    let mut stay_count = [0i32; TEAMS];
    let mut early_count = [0i32; TEAMS];
    let mut early_late = [0u8; TEAMS * WEEKS];

    for w in 0..WEEKS {
        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            let early: u8 = if q < 2 { 1 } else { 0 };
            let lane_off = (q % 2) * 2;

            let pairs: [(u8, u8); 4] = [(pa, pb), (pc, pd), (pa, pd), (pc, pb)];
            for &(t1, t2) in &pairs {
                let lo = t1.min(t2) as usize;
                let hi = t1.max(t2) as usize;
                matchups[lo * TEAMS + hi] += 1;
                week_matchup[w * TEAMS * TEAMS + lo * TEAMS + hi] = 1;
            }

            lane_counts[pa as usize * LANES + lane_off] += 2;
            lane_counts[pb as usize * LANES + lane_off] += 1;
            lane_counts[pb as usize * LANES + lane_off + 1] += 1;
            lane_counts[pc as usize * LANES + lane_off + 1] += 2;
            lane_counts[pd as usize * LANES + lane_off + 1] += 1;
            lane_counts[pd as usize * LANES + lane_off] += 1;

            if q >= 2 {
                late_lane_counts[pa as usize * LANES + lane_off] += 2;
                late_lane_counts[pb as usize * LANES + lane_off] += 1;
                late_lane_counts[pb as usize * LANES + lane_off + 1] += 1;
                late_lane_counts[pc as usize * LANES + lane_off + 1] += 2;
                late_lane_counts[pd as usize * LANES + lane_off + 1] += 1;
                late_lane_counts[pd as usize * LANES + lane_off] += 1;
            }

            stay_count[pa as usize] += 1;
            stay_count[pc as usize] += 1;

            for &t in &[pa, pb, pc, pd] {
                early_late[t as usize * WEEKS + w] = early;
                if early == 1 {
                    early_count[t as usize] += 1;
                }
            }
        }
    }

    let mut matchup_balance: u32 = 0;
    for i in 0..TEAMS {
        for j in (i + 1)..TEAMS {
            let c = matchups[i * TEAMS + j];
            if c == 0 {
                matchup_balance += w8.matchup_zero;
            } else if c >= 3 {
                matchup_balance += (c - 2) as u32 * w8.matchup_triple;
            }
        }
    }

    let mut consecutive_opponents: u32 = 0;
    for w in 0..(WEEKS - 1) {
        let b1 = w * TEAMS * TEAMS;
        let b2 = (w + 1) * TEAMS * TEAMS;
        for i in 0..TEAMS {
            for j in (i + 1)..TEAMS {
                let idx = i * TEAMS + j;
                if week_matchup[b1 + idx] != 0 && week_matchup[b2 + idx] != 0 {
                    consecutive_opponents += w8.consecutive_opponents;
                }
            }
        }
    }

    let mut early_late_balance: u32 = 0;
    let target_e: f64 = WEEKS as f64 / 2.0;
    for t in 0..TEAMS {
        let dev = (early_count[t] as f64 - target_e).abs();
        early_late_balance += (dev * dev * w8.early_late_balance) as u32;
    }

    let mut early_late_alternation: u32 = 0;
    for t in 0..TEAMS {
        for w in 0..(WEEKS - 2) {
            let base = t * WEEKS;
            if early_late[base + w] == early_late[base + w + 1]
                && early_late[base + w + 1] == early_late[base + w + 2]
            {
                early_late_alternation += w8.early_late_alternation;
            }
        }
    }

    let mut lane_balance: u32 = 0;
    let target_l: f64 = (WEEKS as f64 * 2.0) / LANES as f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            lane_balance +=
                ((lane_counts[t * LANES + l] as f64 - target_l).abs() * w8.lane_balance) as u32;
        }
    }

    let mut lane_switch_balance: u32 = 0;
    let target_stay: f64 = WEEKS as f64 / 2.0;
    for t in 0..TEAMS {
        let dev = (stay_count[t] as f64 - target_stay).abs();
        lane_switch_balance += (dev * w8.lane_switch) as u32;
    }

    let mut late_lane_balance: u32 = 0;
    let late_target_l: f64 = WEEKS as f64 / LANES as f64;
    for t in 0..TEAMS {
        for l in 0..LANES {
            late_lane_balance +=
                ((late_lane_counts[t * LANES + l] as f64 - late_target_l).abs() * w8.late_lane_balance) as u32;
        }
    }

    let mut min_overlap = WEEKS as u32;
    for i in 0..TEAMS {
        for j in (i + 1)..TEAMS {
            let mut overlap = 0u32;
            for w in 0..WEEKS {
                if early_late[i * WEEKS + w] == early_late[j * WEEKS + w] {
                    overlap += 1;
                }
            }
            if overlap < min_overlap {
                min_overlap = overlap;
            }
        }
    }
    let commissioner_overlap = w8.commissioner_overlap * min_overlap.saturating_sub(1);

    let total = matchup_balance
        + consecutive_opponents
        + early_late_balance
        + early_late_alternation
        + lane_balance
        + lane_switch_balance
        + late_lane_balance
        + commissioner_overlap;

    CostBreakdown {
        matchup_balance,
        consecutive_opponents,
        early_late_balance,
        early_late_alternation,
        lane_balance,
        lane_switch_balance,
        late_lane_balance,
        commissioner_overlap,
        total,
    }
}

pub fn perturb(a: &mut Assignment, rng: &mut SmallRng, n: usize) {
    for _ in 0..n {
        let w = rng.random_range(0..WEEKS);
        let q1 = rng.random_range(0..QUADS);
        let mut q2 = rng.random_range(0..(QUADS - 1));
        if q2 >= q1 {
            q2 += 1;
        }
        let p1 = rng.random_range(0..POS);
        let p2 = rng.random_range(0..POS);
        let tmp = a[w][q1][p1];
        a[w][q1][p1] = a[w][q2][p2];
        a[w][q2][p2] = tmp;
    }
}

pub fn assignment_to_tsv(a: &Assignment) -> String {
    let slot_names = ["Early 1", "Early 2", "Late 1", "Late 2"];
    let mut lines = vec![String::from("Week\tSlot\tLane 1\tLane 2\tLane 3\tLane 4")];

    for w in 0..WEEKS {
        let mut slots: [[String; LANES]; 4] = Default::default();

        for q in 0..QUADS {
            let [pa, pb, pc, pd] = a[w][q];
            let slot_base = if q < 2 { 0 } else { 2 };
            let lane_base = (q % 2) * 2;

            slots[slot_base][lane_base] = format!("{} v {}", pa + 1, pb + 1);
            slots[slot_base][lane_base + 1] = format!("{} v {}", pc + 1, pd + 1);
            slots[slot_base + 1][lane_base] = format!("{} v {}", pa + 1, pd + 1);
            slots[slot_base + 1][lane_base + 1] = format!("{} v {}", pc + 1, pb + 1);
        }

        for (s, slot_row) in slots.iter().enumerate() {
            lines.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                w + 1,
                slot_names[s],
                slot_row[0],
                slot_row[1],
                slot_row[2],
                slot_row[3]
            ));
        }
    }

    lines.join("\n")
}

pub fn cost_label(c: &CostBreakdown) -> String {
    format!(
        "total: {:>4} matchup: {:>3} consec: {:>3} el_bal: {:>3} el_alt: {:>3} lane: {:>3} switch: {:>3} ll_bal: {:>3} comm: {:>3}",
        c.total, c.matchup_balance, c.consecutive_opponents,
        c.early_late_balance, c.early_late_alternation, c.lane_balance,
        c.lane_switch_balance, c.late_lane_balance, c.commissioner_overlap,
    )
}

pub fn parse_tsv(content: &str) -> Option<Assignment> {
    let mut a: Assignment = [[[0u8; POS]; QUADS]; WEEKS];
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 2 { return None; }

    for w in 0..WEEKS {
        let base = 1 + w * 4;
        if base + 3 >= lines.len() { return None; }

        let e1: Vec<&str> = lines[base].split('\t').collect();
        let l1: Vec<&str> = lines[base + 2].split('\t').collect();
        if e1.len() < 6 || l1.len() < 6 { return None; }

        let parse_match = |s: &str| -> Option<(u8, u8)> {
            let parts: Vec<&str> = s.split(" v ").collect();
            if parts.len() != 2 { return None; }
            let a = parts[0].trim().parse::<u8>().ok()? - 1;
            let b = parts[1].trim().parse::<u8>().ok()? - 1;
            Some((a, b))
        };

        let (pa, pb) = parse_match(e1[2])?;
        let (pc, pd) = parse_match(e1[3])?;
        a[w][0] = [pa, pb, pc, pd];

        let (pa, pb) = parse_match(e1[4])?;
        let (pc, pd) = parse_match(e1[5])?;
        a[w][1] = [pa, pb, pc, pd];

        let (pa, pb) = parse_match(l1[2])?;
        let (pc, pd) = parse_match(l1[3])?;
        a[w][2] = [pa, pb, pc, pd];

        let (pa, pb) = parse_match(l1[4])?;
        let (pc, pd) = parse_match(l1[5])?;
        a[w][3] = [pa, pb, pc, pd];
    }
    Some(a)
}

pub fn reassign_commissioners(a: &mut Assignment) {
    let mut early_late = [0u8; TEAMS * WEEKS];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            let early: u8 = if q < 2 { 1 } else { 0 };
            for &t in &a[w][q] {
                early_late[t as usize * WEEKS + w] = early;
            }
        }
    }

    let mut best_i = 0usize;
    let mut best_j = 1usize;
    let mut min_overlap = WEEKS as u32;
    for i in 0..TEAMS {
        for j in (i + 1)..TEAMS {
            let mut overlap = 0u32;
            for w in 0..WEEKS {
                if early_late[i * WEEKS + w] == early_late[j * WEEKS + w] {
                    overlap += 1;
                }
            }
            if overlap < min_overlap {
                min_overlap = overlap;
                best_i = i;
                best_j = j;
            }
        }
    }

    if best_i == 0 && best_j == 1 {
        return;
    }

    let mut perm: [u8; TEAMS] = std::array::from_fn(|i| i as u8);
    perm.swap(0, best_i);
    perm.swap(1, best_j);
    let mut inv = [0u8; TEAMS];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u8;
    }

    for w in 0..WEEKS {
        for q in 0..QUADS {
            for p in 0..POS {
                a[w][q][p] = inv[a[w][q][p] as usize];
            }
        }
    }
}

pub fn flat_to_assignment(flat: &[u8]) -> Assignment {
    let mut a = [[[0u8; POS]; QUADS]; WEEKS];
    for w in 0..WEEKS {
        for q in 0..QUADS {
            for p in 0..POS {
                a[w][q][p] = flat[w * QUADS * POS + q * POS + p];
            }
        }
    }
    a
}

pub fn assignment_to_flat(a: &Assignment) -> Vec<u8> {
    let mut flat = Vec::with_capacity(WEEKS * QUADS * POS);
    for w in 0..WEEKS {
        for q in 0..QUADS {
            for p in 0..POS {
                flat.push(a[w][q][p]);
            }
        }
    }
    flat
}
