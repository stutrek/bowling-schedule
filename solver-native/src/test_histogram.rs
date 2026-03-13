/// Standalone test: generate realistic GPU cost distributions and check for striping
/// Run with: cargo test -p solver-native --lib test_histogram -- --nocapture

#[cfg(test)]
mod tests {
    use crate::output_winter_fixed::build_histogram;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Generate costs resembling a GPU parallel tempering distribution:
    /// bell curve centered around `center` with some spread, plus a long tail
    fn generate_gpu_costs(rng: &mut SmallRng, n: usize, center: u32, spread: u32) -> Vec<u32> {
        let mut costs = Vec::with_capacity(n);
        for _ in 0..n {
            // Sum of 3 uniform samples → approximate bell curve
            let v: f64 = (0..3).map(|_| rng.random::<f64>()).sum::<f64>() / 3.0;
            let cost = center as f64 + (v - 0.5) * 2.0 * spread as f64;
            costs.push(cost.max(100.0) as u32);
        }
        // Sprinkle some outliers
        for _ in 0..n / 50 {
            costs.push(center + spread + rng.random_range(0..spread));
        }
        costs
    }

    fn count_stripes(row: &str) -> usize {
        // A stripe is a space flanked by non-space on both sides
        let chars: Vec<char> = row.chars().collect();
        let mut stripes = 0;
        for i in 1..chars.len() - 1 {
            if chars[i] == ' ' && chars[i - 1] != ' ' && chars[i + 1] != ' ' {
                stripes += 1;
            }
        }
        stripes
    }

    #[test]
    fn no_striping_in_dense_region() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Test several different distributions
        let scenarios: Vec<(&str, u32, u32, usize)> = vec![
            ("tight cluster", 2500, 300, 65536),
            ("wide spread", 3000, 1500, 65536),
            ("low cost cluster", 1000, 200, 65536),
            ("high cost spread", 5000, 2000, 65536),
            ("small population", 2500, 500, 1000),
        ];

        for (name, center, spread, n) in &scenarios {
            let costs = generate_gpu_costs(&mut rng, *n, *center, *spread);
            let hist = build_histogram(&costs, 140);

            println!("\n=== {} (center={}, spread={}, n={}) ===", name, center, spread, n);
            for row_idx in 0..3 {
                println!("  {} │{}", hist.labels[row_idx], hist.rows[row_idx]);
            }
            println!("       └{}", hist.legend);

            // Check for single-char gaps in each row (stripes)
            for (row_idx, row) in hist.rows.iter().enumerate() {
                let stripes = count_stripes(row);
                println!("  row {}: {} stripes", row_idx, stripes);
                // Bottom row (2) will naturally have the most stripes at the tails,
                // allow some there. Dense middle rows should have very few.
                if row_idx == 0 {
                    assert!(stripes <= 3, "{}: top row has {} stripes (>3)", name, stripes);
                }
            }
        }
    }

    #[test]
    fn stripe_test_print_raw_buckets() {
        // Show the raw bucket counts to diagnose striping
        let mut rng = SmallRng::seed_from_u64(99);
        let costs = generate_gpu_costs(&mut rng, 65536, 2500, 400);

        let hist = build_histogram(&costs, 140);
        println!("\n=== Raw histogram output ===");
        for row_idx in 0..3 {
            println!("  {} │{}", hist.labels[row_idx], hist.rows[row_idx]);
        }
        println!("       └{}", hist.legend);

        // Also print the bottom row char-by-char to see exactly where gaps are
        let chars: Vec<char> = hist.rows[2].chars().collect();
        let mut gap_positions = vec![];
        for i in 1..chars.len() - 1 {
            if chars[i] == ' ' && chars[i - 1] != ' ' && chars[i + 1] != ' ' {
                gap_positions.push(i);
            }
        }
        if !gap_positions.is_empty() {
            println!("  Bottom row single-char gaps at positions: {:?}", gap_positions);
        } else {
            println!("  No single-char gaps in bottom row");
        }
    }

    #[test]
    fn stripe_test_from_dump() {
        // Load a hist_dump_*.txt file if one exists in results/gpu/
        let dump_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results/gpu");
        let mut dump_files: Vec<_> = std::fs::read_dir(dump_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("hist_dump_"))
            .collect();
        dump_files.sort_by_key(|e| e.file_name());

        if dump_files.is_empty() {
            println!("No hist_dump files found in {} — press 'h' during a run to create one", dump_dir);
            return;
        }

        let path = dump_files.last().unwrap().path();
        println!("Loading dump: {}", path.display());
        let contents = std::fs::read_to_string(&path).unwrap();
        let costs: Vec<u32> = contents.lines()
            .filter_map(|l| l.trim().parse::<u32>().ok())
            .collect();
        println!("  {} costs, min={}, max={}", costs.len(),
            costs.iter().min().unwrap_or(&0), costs.iter().max().unwrap_or(&0));

        let hist = build_histogram(&costs, 140);
        println!();
        for row_idx in 0..3 {
            println!("  {} │{}", hist.labels[row_idx], hist.rows[row_idx]);
        }
        println!("       └{}", hist.legend);

        for (row_idx, row) in hist.rows.iter().enumerate() {
            let stripes = count_stripes(row);
            println!("  row {}: {} stripes", row_idx, stripes);
        }
    }
}
