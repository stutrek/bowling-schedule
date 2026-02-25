use std::fs;
use std::path::Path;

use solver_native::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rescore <weights.json> <dir> [dir2 ...]");
        eprintln!("Re-evaluates all .tsv files and renames any whose score prefix is stale.");
        std::process::exit(1);
    }

    let weights_str = fs::read_to_string(&args[1])
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", args[1], e));
    let w8: Weights = serde_json::from_str(&weights_str)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", args[1], e));

    let mut checked = 0u32;
    let mut renamed = 0u32;
    let mut errors = 0u32;

    for dir_arg in &args[2..] {
        let entries: Vec<_> = match fs::read_dir(dir_arg) {
            Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
            Err(e) => {
                eprintln!("Warning: could not read directory {}: {}", dir_arg, e);
                continue;
            }
        };

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("tsv") {
                continue;
            }

            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Warning: could not read {}: {}", path.display(), e);
                    errors += 1;
                    continue;
                }
            };

            let a = match parse_tsv(&content) {
                Some(a) => a,
                None => {
                    eprintln!("Warning: could not parse {}", path.display());
                    errors += 1;
                    continue;
                }
            };

            checked += 1;
            let cost = evaluate(&a, &w8);
            let filename = path.file_name().unwrap().to_string_lossy().to_string();

            let old_score = filename.get(..4).and_then(|s| s.parse::<u32>().ok());
            let new_score = cost.total;

            match old_score {
                Some(old) if old == new_score => {}
                Some(old) => {
                    let new_filename = format!("{:04}{}", new_score, &filename[4..]);
                    let new_path = path.parent().unwrap_or(Path::new(".")).join(&new_filename);

                    if new_path.exists() {
                        eprintln!(
                            "  CONFLICT {} -> {} (target exists, skipping)",
                            filename, new_filename,
                        );
                        errors += 1;
                        continue;
                    }

                    match fs::rename(&path, &new_path) {
                        Ok(()) => {
                            eprintln!(
                                "  {} -> {} (was {}, now {} | {})",
                                filename, new_filename, old, new_score, cost_label(&cost),
                            );
                            renamed += 1;
                        }
                        Err(e) => {
                            eprintln!("  ERROR renaming {}: {}", filename, e);
                            errors += 1;
                        }
                    }
                }
                None => {
                    eprintln!(
                        "  {} score={} ({}), filename doesn't start with a score prefix",
                        filename, new_score, cost_label(&cost),
                    );
                }
            }
        }
    }

    eprintln!(
        "\nDone. Checked {} files, renamed {}, errors/skips {}.",
        checked, renamed, errors,
    );
}
