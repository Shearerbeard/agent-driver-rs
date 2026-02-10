use std::hint::black_box;

use futures::StreamExt;
use rig::client::CompletionClient;
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::openai;
use rig::agent::MultiTurnStreamItem;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use rig::tool::Tool;
use rmcp::transport::ConfigureCommandExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::harness::{self, ReportRow, Sample};

/// Shared prompts (same as ours.rs).
const TTFT_PROMPT: &str = "Say hello in exactly three words.";
const TOOL_PROMPT: &str = "What is 47 multiplied by 89? Use the calculator tool.";
const SYSTEM_PROMPT: &str = "You are a helpful assistant. Be concise.";

// ---------------------------------------------------------------------------
// Calculator tool for rig
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CalculatorArgs {
    operation: String,
    x: f64,
    y: f64,
}

#[derive(Debug, Serialize)]
struct CalculatorOutput {
    result: f64,
}

#[derive(Debug)]
struct CalculatorError(String);

impl std::fmt::Display for CalculatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CalculatorError {}

struct Calculator;

impl Tool for Calculator {
    type Error = CalculatorError;
    type Args = CalculatorArgs;
    type Output = CalculatorOutput;

    const NAME: &'static str = "calculator";

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".into(),
            description: "Perform basic arithmetic operations (add, subtract, multiply, divide)"
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform"
                    },
                    "x": { "type": "number", "description": "First operand" },
                    "y": { "type": "number", "description": "Second operand" }
                },
                "required": ["operation", "x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = match args.operation.as_str() {
            "add" => args.x + args.y,
            "subtract" => args.x - args.y,
            "multiply" => args.x * args.y,
            "divide" => {
                if args.y == 0.0 {
                    return Err(CalculatorError("Division by zero".into()));
                }
                args.x / args.y
            }
            other => return Err(CalculatorError(format!("Unknown operation: {other}"))),
        };
        Ok(CalculatorOutput { result })
    }
}

/// Create a rig OpenAI client with explicit type.
fn make_client() -> anyhow::Result<openai::Client> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let client: openai::Client = openai::Client::new(&api_key)?;
    Ok(client)
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Scenario: cold_start — client + agent creation (no network).
pub async fn cold_start(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    let model = model.to_string();
    harness::run_scenario("rig/cold_start", n, warmup, || {
        let model = model.clone();
        async move {
            let client = make_client()?;
            let agent = client
                .agent(&model)
                .preamble(SYSTEM_PROMPT)
                .max_tokens(256)
                .tool(Calculator)
                .build();
            black_box(&agent);
            Ok(())
        }
    })
    .await
}

/// Scenario: ttft — time to first streaming text token.
pub async fn ttft(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    let model_str = model.to_string();
    harness::run_scenario("rig/ttft", n, warmup, || {
        let model_str = model_str.clone();
        async move {
            let client = make_client()?;
            let agent = client
                .agent(&model_str)
                .preamble(SYSTEM_PROMPT)
                .max_tokens(256)
                .build();

            let mut stream = agent.stream_prompt(TTFT_PROMPT).await;
            while let Some(event) = stream.next().await {
                match event {
                    Ok(MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::Text(_),
                    )) => break,
                    Ok(_) => continue,
                    Err(e) => anyhow::bail!("Stream error: {e}"),
                }
            }
            Ok(())
        }
    })
    .await
}

/// Scenario: tool_roundtrip — full agent loop with calculator tool (streaming).
///
/// Uses `stream_prompt().multi_turn(2)` so both libraries stream their tool
/// roundtrip, matching agent-driver-rs's `AgentLoop::run()`.
pub async fn tool_roundtrip(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    let model_str = model.to_string();
    harness::run_scenario("rig/tool_rt", n, warmup, || {
        let model_str = model_str.clone();
        async move {
            let client = make_client()?;
            let agent = client
                .agent(&model_str)
                .preamble(SYSTEM_PROMPT)
                .max_tokens(256)
                .tool(Calculator)
                .build();

            let mut stream = agent.stream_prompt(TOOL_PROMPT).multi_turn(2).await;
            let mut got_text = false;
            while let Some(event) = stream.next().await {
                match event {
                    Ok(MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::Text(_),
                    )) => {
                        got_text = true;
                    }
                    Ok(_) => continue,
                    Err(e) => anyhow::bail!("Stream error: {e}"),
                }
            }
            if !got_text {
                anyhow::bail!("Empty response from rig agent");
            }
            Ok(())
        }
    })
    .await
}

/// Scenario: mcp_discovery — tool enumeration on an already-connected MCP server.
pub async fn mcp_discovery(_model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    use rmcp::service::ServiceExt;
    use rmcp::transport::TokioChildProcess;
    use tokio::process::Command;

    // Setup: connect once before the timed loop
    let transport = TokioChildProcess::new(Command::new("npx").configure(|cmd| {
        cmd.args(["-y", "@modelcontextprotocol/server-filesystem", "."]);
    }))
    .expect("Failed to spawn MCP filesystem server");

    let mut client = rmcp::model::ClientInfo::default()
        .serve(transport)
        .await
        .expect("Failed to initialize MCP client");

    let samples = harness::run_scenario("rig/mcp_disc", n, warmup, || {
        let client = &client;
        async move {
            let result = client.list_tools(None).await?;
            assert!(!result.tools.is_empty(), "MCP server returned no tools");
            Ok(())
        }
    })
    .await;

    let _ = client
        .close_with_timeout(std::time::Duration::from_secs(3))
        .await;
    samples
}

/// Scenario: mcp_roundtrip — full agent loop with MCP-discovered tools.
///
/// Uses a single-tool-call prompt ("read Cargo.toml") to avoid depth-cap errors
/// from multi-round listing. Both sides cap at 2 tool rounds.
pub async fn mcp_roundtrip(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    use rmcp::service::ServiceExt;
    use rmcp::transport::TokioChildProcess;
    use tokio::process::Command;

    // Setup: connect + discover tools once
    let transport = TokioChildProcess::new(Command::new("npx").configure(|cmd| {
        cmd.args(["-y", "@modelcontextprotocol/server-filesystem", "."]);
    }))
    .expect("Failed to spawn MCP filesystem server");

    let mut client = rmcp::model::ClientInfo::default()
        .serve(transport)
        .await
        .expect("Failed to initialize MCP client");

    let tools = client
        .list_all_tools()
        .await
        .expect("Failed to list MCP tools");

    let model_str = model.to_string();
    let peer = client.peer().to_owned();
    let samples = harness::run_scenario("rig/mcp_rt", n, warmup, || {
        let model_str = model_str.clone();
        let tools = tools.clone();
        let peer = peer.clone();
        async move {
            let client = make_client()?;
            let agent = client
                .agent(&model_str)
                .preamble(SYSTEM_PROMPT)
                .max_tokens(512)
                .rmcp_tools(tools, peer)
                .build();

            let response: String = agent
                .prompt("Read the file named Cargo.toml and tell me the package name.")
                .max_turns(2)
                .await?;
            if response.is_empty() {
                anyhow::bail!("Empty response from rig agent");
            }
            Ok(())
        }
    })
    .await;

    let _ = client
        .close_with_timeout(std::time::Duration::from_secs(3))
        .await;
    samples
}

/// Run all rig scenarios and return report rows.
pub async fn run_all(model: &str, n: usize, warmup: usize, scenario_filter: Option<&str>) -> Vec<ReportRow> {
    let mut rows = Vec::new();
    let lib = "rig.rs";

    if scenario_filter.is_none() || scenario_filter == Some("cold_start") {
        let samples = cold_start(model, n, warmup).await;
        if let Some(stats) = harness::compute_stats(&samples) {
            rows.push(ReportRow {
                scenario: "cold_start".into(),
                library: lib.into(),
                stats,
            });
        }
    }

    if scenario_filter.is_none() || scenario_filter == Some("ttft") {
        let samples = ttft(model, n, warmup).await;
        if let Some(stats) = harness::compute_stats(&samples) {
            rows.push(ReportRow {
                scenario: "ttft".into(),
                library: lib.into(),
                stats,
            });
        }
    }

    if scenario_filter.is_none() || scenario_filter == Some("tool_roundtrip") {
        let samples = tool_roundtrip(model, n, warmup).await;
        if let Some(stats) = harness::compute_stats(&samples) {
            rows.push(ReportRow {
                scenario: "tool_roundtrip".into(),
                library: lib.into(),
                stats,
            });
        }
    }

    if scenario_filter.is_none() || scenario_filter == Some("mcp_discovery") {
        let samples = mcp_discovery(model, n, warmup).await;
        if let Some(stats) = harness::compute_stats(&samples) {
            rows.push(ReportRow {
                scenario: "mcp_discovery".into(),
                library: lib.into(),
                stats,
            });
        }
    }

    if scenario_filter.is_none() || scenario_filter == Some("mcp_roundtrip") {
        let samples = mcp_roundtrip(model, n, warmup).await;
        if let Some(stats) = harness::compute_stats(&samples) {
            rows.push(ReportRow {
                scenario: "mcp_roundtrip".into(),
                library: lib.into(),
                stats,
            });
        }
    }

    rows
}
