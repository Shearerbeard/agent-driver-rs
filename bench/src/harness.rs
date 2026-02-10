use std::fmt;
use std::future::Future;
use std::time::{Duration, Instant};

use serde::Serialize;
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};

/// Global tracking allocator — wraps System, counts bytes.
#[global_allocator]
pub static GLOBAL: &StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

/// A single timing sample from one iteration.
#[derive(Debug, Clone)]
pub struct Sample {
    pub duration: Duration,
    /// Total bytes allocated during this iteration.
    pub bytes_allocated: usize,
}

/// Aggregate statistics computed from a set of samples.
#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    #[serde(serialize_with = "ser_duration_us")]
    pub min: Duration,
    #[serde(serialize_with = "ser_duration_us")]
    pub median: Duration,
    #[serde(serialize_with = "ser_duration_us")]
    pub mean: Duration,
    #[serde(serialize_with = "ser_duration_us")]
    pub stddev: Duration,
    #[serde(serialize_with = "ser_duration_us")]
    pub max: Duration,
    /// Median total bytes allocated per iteration.
    pub alloc_median: usize,
}

fn ser_duration_us<S: serde::Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(d.as_micros() as u64)
}

/// Run a scenario `n` times (plus `warmup` discarded iterations) and collect samples.
pub async fn run_scenario<F, Fut>(
    name: &str,
    n: usize,
    warmup: usize,
    f: F,
) -> Vec<Sample>
where
    F: Fn() -> Fut,
    Fut: Future<Output = anyhow::Result<()>>,
{
    // Warmup iterations (discarded)
    for i in 0..warmup {
        eprintln!("  {name}: warmup {}/{warmup}", i + 1);
        if let Err(e) = f().await {
            eprintln!("  {name}: warmup error: {e}");
        }
    }

    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        eprintln!("  {name}: iteration {}/{n}", i + 1);
        let reg = Region::new(GLOBAL);
        let start = Instant::now();
        if let Err(e) = f().await {
            eprintln!("  {name}: iteration error: {e}");
            continue;
        }
        let duration = start.elapsed();
        let stats = reg.change();
        let bytes_allocated = stats.bytes_allocated as usize;
        samples.push(Sample {
            duration,
            bytes_allocated,
        });
    }
    samples
}

/// Compute aggregate statistics from a set of samples.
pub fn compute_stats(samples: &[Sample]) -> Option<Stats> {
    if samples.is_empty() {
        return None;
    }

    let mut durations: Vec<Duration> = samples.iter().map(|s| s.duration).collect();
    durations.sort();

    let n = durations.len();
    let min = durations[0];
    let max = durations[n - 1];
    let median = durations[n / 2];
    let mean_nanos = durations.iter().map(|d| d.as_nanos()).sum::<u128>() / n as u128;
    let mean = Duration::from_nanos(mean_nanos as u64);
    let variance = durations
        .iter()
        .map(|d| {
            let diff = d.as_nanos() as f64 - mean_nanos as f64;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let stddev = Duration::from_nanos(variance.sqrt() as u64);

    let mut allocs: Vec<usize> = samples.iter().map(|s| s.bytes_allocated).collect();
    allocs.sort();
    let alloc_median = allocs[n / 2];

    Some(Stats {
        min,
        median,
        mean,
        stddev,
        max,
        alloc_median,
    })
}

/// Result row for the report table.
#[derive(Serialize)]
pub struct ReportRow {
    pub scenario: String,
    pub library: String,
    #[serde(flatten)]
    pub stats: Stats,
}

/// Print the full benchmark report with per-scenario winner flags.
pub fn print_report(rows: &[ReportRow], model: &str, iterations: usize) {
    use std::collections::HashMap;

    let width = 108;
    let sep = "=".repeat(width);
    let thin = "─".repeat(width);

    // Build winner markers by pairing rows with the same scenario.
    // Winner is determined by median; marker shows speedup ratio.
    let mut scenario_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        scenario_indices.entry(&row.scenario).or_default().push(i);
    }

    let mut markers: Vec<String> = vec![String::new(); rows.len()];
    for indices in scenario_indices.values() {
        if indices.len() == 2 {
            let a_ns = rows[indices[0]].stats.median.as_nanos();
            let b_ns = rows[indices[1]].stats.median.as_nanos();
            if a_ns < b_ns && a_ns > 0 {
                let ratio = b_ns as f64 / a_ns as f64;
                markers[indices[0]] = format!("◀ {ratio:.1}x");
            } else if b_ns < a_ns && b_ns > 0 {
                let ratio = a_ns as f64 / b_ns as f64;
                markers[indices[1]] = format!("◀ {ratio:.1}x");
            }
        }
    }

    println!();
    println!("{sep}");
    println!(
        "  Benchmark: agent-driver-rs vs rig.rs (model: {model}, iterations: {iterations})"
    );
    println!("{sep}");
    println!();
    println!(
        "{:<17}{:<17}{:>9}{:>9}{:>9}{:>9}{:>9}  {:>10}  {}",
        "Scenario", "Library", "Min", "Median", "Mean", "StdDev", "Max", "Alloc", "vs"
    );
    println!("{thin}");

    let mut last_scenario = String::new();
    for (i, row) in rows.iter().enumerate() {
        if !last_scenario.is_empty() && row.scenario != last_scenario {
            println!("{thin}");
        }
        last_scenario.clone_from(&row.scenario);

        println!(
            "{:<17}{:<17}{}{}{}{}{}  {}  {}",
            row.scenario,
            row.library,
            FmtDuration(row.stats.min),
            FmtDuration(row.stats.median),
            FmtDuration(row.stats.mean),
            FmtDuration(row.stats.stddev),
            FmtDuration(row.stats.max),
            FmtBytes(row.stats.alloc_median),
            markers[i],
        );
    }
    println!("{sep}");
    println!();
}

/// Print benchmark results as JSON (one object with metadata + results array).
///
/// Durations are serialized as microseconds (u64) for easy numeric comparison.
pub fn print_json(rows: &[ReportRow], model: &str, iterations: usize) {
    #[derive(Serialize)]
    struct JsonReport<'a> {
        model: &'a str,
        iterations: usize,
        epoch_secs: u64,
        results: &'a [ReportRow],
    }

    let epoch_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let report = JsonReport {
        model,
        iterations,
        epoch_secs,
        results: rows,
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

/// Format a Duration for display (e.g. "1.5ms", "420ms", "1.2s").
/// Returns a pre-padded string to avoid multi-byte alignment issues.
struct FmtDuration(Duration);

impl FmtDuration {
    fn to_string_inner(&self) -> String {
        let us = self.0.as_micros();
        if us < 1_000 {
            format!("{us}us")
        } else if us < 1_000_000 {
            format!("{:.1}ms", us as f64 / 1_000.0)
        } else {
            format!("{:.2}s", us as f64 / 1_000_000.0)
        }
    }
}

impl fmt::Display for FmtDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.to_string_inner();
        write!(f, "{s:>9}")
    }
}

/// Format bytes for display (e.g. "1.2 KB", "3.4 MB").
struct FmtBytes(usize);

impl fmt::Display for FmtBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = if self.0 < 1_024 {
            format!("{} B", self.0)
        } else if self.0 < 1_024 * 1_024 {
            format!("{:.1} KB", self.0 as f64 / 1_024.0)
        } else {
            format!("{:.1} MB", self.0 as f64 / (1_024.0 * 1_024.0))
        };
        write!(f, "{s:>10}")
    }
}
