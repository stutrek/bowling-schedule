use std::fs;
use std::path::Path;

use solver_core::winter;
use solver_core::summer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rescore [--league summer] <weights.json> <dir> [dir2 ...]");
        eprintln!("Re-evaluates all .tsv files and renames any whose score prefix is stale.");
        std::process::exit(1);
    }

    let league = args.iter()
        .position(|a| a == "--league")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("winter");

    // Skip --league and its argument in remaining args
    let filtered: Vec<&str> = {
        let mut v = Vec::new();
        let mut skip_next = false;
        for (_i, a) in args.iter().enumerate().skip(1) {
            if skip_next { skip_next = false; continue; }
            if a == "--league" { skip_next = true; continue; }
            v.push(a.as_str());
        }
        v
    };

    if filtered.len() < 2 {
        eprintln!("Usage: rescore [--league summer] <weights.json> <dir> [dir2 ...]");
        std::process::exit(1);
    }

    let weights_path = filtered[0];
    let dirs = &filtered[1..];

    match league {
        "winter" => rescore_winter(weights_path, dirs),
        "summer" => rescore_summer(weights_path, dirs),
        other => {
            eprintln!("Unknown league: {}. Use --league winter or --league summer", other);
            std::process::exit(1);
        }
    }
}

fn rescore_winter(weights_path: &str, dirs: &[&str]) {
    let weights_str = fs::read_to_string(weights_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", weights_path, e));
    let w8: winter::Weights = serde_json::from_str(&weights_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", weights_path, e));

    let mut checked = 0u32;
    let mut renamed = 0u32;
    let mut errors = 0u32;

    for dir_arg in dirs {
        let entries: Vec<_> = match fs::read_dir(dir_arg) {
            Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
            Err(e) => { eprintln!("Warning: could not read directory {}: {}", dir_arg, e); continue; }
        };

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("tsv") { continue; }

            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => { eprintln!("Warning: could not read {}: {}", path.display(), e); errors += 1; continue; }
            };

            let a = match winter::parse_tsv(&content) {
                Some(a) => a,
                None => { eprintln!("Warning: could not parse {}", path.display()); errors += 1; continue; }
            };

            checked += 1;
            let cost = winter::evaluate(&a, &w8);
            let filename = path.file_name().unwrap().to_string_lossy().to_string();
            let old_score = filename.get(..4).and_then(|s| s.parse::<u32>().ok());
            let new_score = cost.total;

            match old_score {
                Some(old) if old == new_score => {}
                Some(old) => {
                    let new_filename = format!("{:04}{}", new_score, &filename[4..]);
                    let new_path = path.parent().unwrap_or(Path::new(".")).join(&new_filename);
                    if new_path.exists() {
                        eprintln!("  CONFLICT {} -> {} (target exists)", filename, new_filename);
                        errors += 1;
                        continue;
                    }
                    match fs::rename(&path, &new_path) {
                        Ok(()) => {
                            eprintln!("  {} -> {} (was {}, now {} | {})", filename, new_filename, old, new_score, winter::cost_label(&cost));
                            renamed += 1;
                        }
                        Err(e) => { eprintln!("  ERROR renaming {}: {}", filename, e); errors += 1; }
                    }
                }
                None => {
                    eprintln!("  {} score={} ({})", filename, new_score, winter::cost_label(&cost));
                }
            }
        }
    }

    eprintln!("\nDone. Checked {} files, renamed {}, errors/skips {}.", checked, renamed, errors);
}

fn rescore_summer(weights_path: &str, dirs: &[&str]) {
    let weights_str = fs::read_to_string(weights_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", weights_path, e));
    let w8: summer::SummerWeights = serde_json::from_str(&weights_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", weights_path, e));

    let mut checked = 0u32;
    let mut renamed = 0u32;
    let mut errors = 0u32;

    for dir_arg in dirs {
        let entries: Vec<_> = match fs::read_dir(dir_arg) {
            Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
            Err(e) => { eprintln!("Warning: could not read directory {}: {}", dir_arg, e); continue; }
        };

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("tsv") { continue; }

            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => { eprintln!("Warning: could not read {}: {}", path.display(), e); errors += 1; continue; }
            };

            let a = match summer::parse_summer_tsv(&content) {
                Some(a) => a,
                None => { eprintln!("Warning: could not parse {}", path.display()); errors += 1; continue; }
            };

            checked += 1;
            let cost = summer::evaluate_summer(&a, &w8);
            let filename = path.file_name().unwrap().to_string_lossy().to_string();
            let old_score = filename.get(..4).and_then(|s| s.parse::<u32>().ok());
            let new_score = cost.total;

            match old_score {
                Some(old) if old == new_score => {}
                Some(old) => {
                    let new_filename = format!("{:04}{}", new_score, &filename[4..]);
                    let new_path = path.parent().unwrap_or(Path::new(".")).join(&new_filename);
                    if new_path.exists() {
                        eprintln!("  CONFLICT {} -> {} (target exists)", filename, new_filename);
                        errors += 1;
                        continue;
                    }
                    match fs::rename(&path, &new_path) {
                        Ok(()) => {
                            eprintln!("  {} -> {} (was {}, now {} | {})", filename, new_filename, old, new_score, summer::summer_cost_label(&cost));
                            renamed += 1;
                        }
                        Err(e) => { eprintln!("  ERROR renaming {}: {}", filename, e); errors += 1; }
                    }
                }
                None => {
                    eprintln!("  {} score={} ({})", filename, new_score, summer::summer_cost_label(&cost));
                }
            }
        }
    }

    eprintln!("\nDone. Checked {} files, renamed {}, errors/skips {}.", checked, renamed, errors);
}
