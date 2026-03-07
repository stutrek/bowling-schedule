use std::time::{Duration, Instant};

pub const FRESH_TABLE_INTERVAL_SECS: u64 = 600;

pub struct GlobalBestMeta {
    pub source: String,
    pub found_at: Instant,
}

pub fn fmt_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    format!("+{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

pub fn format_event(elapsed: Duration, msg: &str) -> String {
    format!("{:>9}  >>>  {}", fmt_elapsed(elapsed), msg)
}
