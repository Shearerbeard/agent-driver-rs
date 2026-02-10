mod harness;
mod ours;
mod rig_bench;

use clap::Parser;
use harness::ReportRow;

#[derive(Parser)]
#[command(name = "agent-driver-bench")]
#[command(about = "Benchmark agent-driver-rs vs rig.rs using OpenAI")]
struct Cli {
    /// Number of measured iterations per scenario
    #[arg(short = 'n', long = "iterations", default_value_t = 20)]
    iterations: usize,

    /// Number of warmup iterations (discarded before measurement)
    #[arg(short = 'w', long = "warmup", default_value_t = 3)]
    warmup: usize,

    /// Run only a specific scenario: cold_start, ttft, tool_roundtrip, mcp_discovery, mcp_roundtrip
    #[arg(short = 's', long = "scenario")]
    scenario: Option<String>,

    /// OpenAI model to use
    #[arg(short = 'm', long = "model", default_value = "gpt-4o-mini")]
    model: String,

    /// Output results as JSON (for CI regression tracking)
    #[arg(long = "json")]
    json: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();
    let model = &cli.model;
    let n = cli.iterations;
    let warmup = cli.warmup;
    let filter = cli.scenario.as_deref();

    // Validate OPENAI_API_KEY is set
    if std::env::var("OPENAI_API_KEY").is_err() {
        anyhow::bail!("OPENAI_API_KEY environment variable is required");
    }

    eprintln!("Benchmark: agent-driver-rs vs rig.rs");
    eprintln!("Model: {model}  |  Iterations: {n}  |  Warmup: {warmup}  |  Scenario: {}",
        filter.unwrap_or("all"));
    eprintln!();

    // Run scenarios, interleaving libraries per-scenario for fairness.
    let mut rows: Vec<ReportRow> = Vec::new();

    let scenarios: Vec<&str> = if let Some(s) = filter {
        vec![s]
    } else {
        vec!["cold_start", "ttft", "tool_roundtrip", "mcp_discovery", "mcp_roundtrip"]
    };

    for scenario in &scenarios {
        eprintln!("--- {scenario} ---");

        // Our library first
        let ours = ours::run_all(model, n, warmup, Some(scenario)).await;
        rows.extend(ours);

        // Rig second
        let rig = rig_bench::run_all(model, n, warmup, Some(scenario)).await;
        rows.extend(rig);

        eprintln!();
    }

    if cli.json {
        harness::print_json(&rows, model, n);
    } else {
        harness::print_report(&rows, model, n);
    }

    Ok(())
}
