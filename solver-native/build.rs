fn main() {
    // Tell cargo to rerun if any WGSL shader files change,
    // since they're included via include_str! and cargo doesn't track them automatically.
    for entry in std::fs::read_dir("src").unwrap().flatten() {
        if entry.path().extension().map_or(false, |e| e == "wgsl") {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
}
