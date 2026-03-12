use solver_core::winter;
use solver_core::winter_fixed;

#[test]
fn validate_schedule_file() {
    let contents = match std::fs::read_to_string("results/gpu/0170-cpu1-20260311-221340-0400.tsv") {
        Ok(c) => c,
        Err(_) => { eprintln!("File not found, skipping"); return; }
    };
    let w8_str = std::fs::read_to_string("../weights.json").unwrap();
    let w8: winter::Weights = serde_json::from_str(&w8_str).unwrap();

    let a = winter::parse_tsv(&contents).expect("Failed to parse");
    let old_bd = winter::evaluate(&a, &w8);
    eprintln!("Old evaluate: {}", winter::cost_label(&old_bd));

    let sched = winter_fixed::from_assignment(&a);
    let wf8 = winter_fixed::WinterFixedWeights {
        matchup_zero: w8.matchup_zero, matchup_triple: w8.matchup_triple,
        consecutive_opponents: w8.consecutive_opponents, early_late_balance: w8.early_late_balance,
        early_late_alternation: w8.early_late_alternation,
        early_late_consecutive: w8.early_late_consecutive,
        lane_balance: w8.lane_balance, lane_switch: w8.lane_switch,
        late_lane_balance: w8.late_lane_balance,
        commissioner_overlap: w8.commissioner_overlap, half_season_repeat: w8.half_season_repeat,
    };
    let new_bd = winter_fixed::evaluate_fixed(&sched, &wf8);
    eprintln!("New evaluate: {}", winter_fixed::fixed_cost_label(&new_bd));

    // Check permutation validity
    for w in 0..12 {
        let mut seen = [false; 16];
        for pos in 0..16 {
            let t = sched.mapping[w][pos] as usize;
            assert!(t < 16, "Week {}: team {} out of range at pos {}", w, t, pos);
            assert!(!seen[t], "Week {}: team {} DUPLICATE", w, t);
            seen[t] = true;
        }
    }
    eprintln!("Permutation check: OK");

    assert_eq!(old_bd.total, new_bd.total,
        "MISMATCH: old={} new={}", old_bd.total, new_bd.total);
    eprintln!("MATCH: both evaluators agree on cost {}", old_bd.total);
}

#[test]
fn validate_all_gpu_results() {
    let w8_str = std::fs::read_to_string("../weights.json").unwrap();
    let w8: winter::Weights = serde_json::from_str(&w8_str).unwrap();
    let wf8 = winter_fixed::WinterFixedWeights {
        matchup_zero: w8.matchup_zero, matchup_triple: w8.matchup_triple,
        consecutive_opponents: w8.consecutive_opponents, early_late_balance: w8.early_late_balance,
        early_late_alternation: w8.early_late_alternation,
        early_late_consecutive: w8.early_late_consecutive,
        lane_balance: w8.lane_balance, lane_switch: w8.lane_switch,
        late_lane_balance: w8.late_lane_balance,
        commissioner_overlap: w8.commissioner_overlap, half_season_repeat: w8.half_season_repeat,
    };

    let entries = match std::fs::read_dir("results/gpu") {
        Ok(e) => e, Err(_) => { eprintln!("No results dir"); return; }
    };
    let mut checked = 0;
    let mut perm_errors = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "tsv") { continue; }
        let contents = std::fs::read_to_string(&path).unwrap();
        let a = match winter::parse_tsv(&contents) {
            Some(a) => a, None => continue,
        };
        let old_cost = winter::evaluate(&a, &w8).total;
        let sched = winter_fixed::from_assignment(&a);
        let new_cost = winter_fixed::evaluate_fixed(&sched, &wf8).total;
        assert_eq!(old_cost, new_cost,
            "MISMATCH in {:?}: old={} new={}", path.file_name(), old_cost, new_cost);
        let mut valid = true;
        for w in 0..12 {
            let mut seen = [false; 16];
            for q in 0..4 {
                for p in 0..4 {
                    let t = a[w][q][p] as usize;
                    if t >= 16 || seen[t] { valid = false; }
                    seen[t] = true;
                }
            }
        }
        if !valid {
            perm_errors += 1;
            eprintln!("  PERM ERROR: {:?} cost={}", path.file_name(), old_cost);
        }
        checked += 1;
    }
    eprintln!("Validated {} result files, {} with permutation errors", checked, perm_errors);
}

#[test]
fn rescore_result_files() {
    let w8_str = std::fs::read_to_string("../weights.json").unwrap();
    let w8: winter::Weights = serde_json::from_str(&w8_str).unwrap();

    let entries = match std::fs::read_dir("results/gpu") {
        Ok(e) => e, Err(_) => { eprintln!("No results dir"); return; }
    };
    let mut renamed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "tsv") { continue; }
        let contents = std::fs::read_to_string(&path).unwrap();
        let a = match winter::parse_tsv(&contents) {
            Some(a) => a, None => continue,
        };
        let cost = winter::evaluate(&a, &w8).total;
        let fname = path.file_name().unwrap().to_str().unwrap().to_string();
        // Current format: NNNN-source-timestamp.tsv
        // Replace the leading cost with the new cost
        let new_prefix = format!("{:04}", cost);
        let old_prefix = &fname[..4];
        if old_prefix != new_prefix {
            let new_fname = format!("{}{}", new_prefix, &fname[4..]);
            let new_path = path.with_file_name(&new_fname);
            std::fs::rename(&path, &new_path).unwrap();
            eprintln!("  {} -> {}", fname, new_fname);
            renamed += 1;
        }
    }
    eprintln!("Rescored: {} files renamed", renamed);
}
