use std::hint::black_box;
use std::sync::Arc;

use agent_driver_rs::agent::{AgentLoop, AgentLoopConfig, MaxToolDepth};
use agent_driver_rs::config::{ApiKey, OpenAiConfig, OpenAiModel};
use agent_driver_rs::provider::OpenAiProvider;
use agent_driver_rs::tool::{DynTool, FnTool, ToolDefinition, ToolSchema};
use agent_driver_rs::ToolName;
use agent_driver_rs::{
    MaxTokens, ModelId, SessionBuilder, StreamDelta, StreamEvent, SystemPrompt, ToolResult,
};
use futures::StreamExt;
use serde_json::json;

use crate::harness::{self, ReportRow, Sample};

/// Shared prompts.
const TTFT_PROMPT: &str = "Say hello in exactly three words.";
const TOOL_PROMPT: &str = "What is 47 multiplied by 89? Use the calculator tool.";
const SYSTEM_PROMPT: &str = "You are a helpful assistant. Be concise.";

/// Build an OpenAI config for the given model string.
fn make_config(model: &str) -> OpenAiConfig {
    let api_key =
        ApiKey::new(std::env::var("OPENAI_API_KEY").unwrap());
    let model_enum = match model {
        "gpt-4o" => OpenAiModel::Gpt4o,
        "gpt-4o-mini" => OpenAiModel::Gpt4oMini,
        other => OpenAiModel::Custom(other.to_string()),
    };
    OpenAiConfig {
        api_key,
        model: model_enum,
        max_tokens: MaxTokens::new(256).unwrap(),
        temperature: None,
        reasoning: None,
        structured_output: false,
    }
}

/// Create the calculator tool matching the shared schema.
fn calculator_tool() -> DynTool {
    let schema = ToolSchema::from_value(json!({
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
    }))
    .unwrap();

    let def = ToolDefinition::new(
        ToolName::new("calculator").unwrap(),
        "Perform basic arithmetic operations (add, subtract, multiply, divide)",
        schema,
    );

    Arc::new(FnTool::new(def, |input, _ctx| {
        let op = input.get_str("operation").unwrap_or("add").to_string();
        let x = input.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let y = input.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
        Box::pin(async move {
            let result = match op.as_str() {
                "add" => x + y,
                "subtract" => x - y,
                "multiply" => x * y,
                "divide" => {
                    if y == 0.0 {
                        return Ok(ToolResult::error("Division by zero"));
                    }
                    x / y
                }
                _ => return Ok(ToolResult::error(format!("Unknown operation: {op}"))),
            };
            Ok(ToolResult::text(format!("{result}")))
        })
    }))
}

/// Scenario: cold_start — measure provider + session creation (no network).
pub async fn cold_start(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    let model = model.to_string();
    harness::run_scenario("ours/cold_start", n, warmup, || {
        let model = model.clone();
        async move {
            let config = make_config(&model);
            let provider = OpenAiProvider::new(config)?;
            let session = SessionBuilder::new()
                .with_provider(provider)
                .model(ModelId::new(model)?)
                .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
                .max_tokens(MaxTokens::new(256).unwrap())
                .build()
                .await?;
            black_box(&session);
            Ok(())
        }
    })
    .await
}

/// Scenario: ttft — time to first streaming text token.
pub async fn ttft(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    let model_str = model.to_string();
    harness::run_scenario("ours/ttft", n, warmup, || {
        let model_str = model_str.clone();
        async move {
            let config = make_config(&model_str);
            let provider = OpenAiProvider::new(config)?;
            let session = SessionBuilder::new()
                .with_provider(provider)
                .model(ModelId::new(&model_str)?)
                .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
                .max_tokens(MaxTokens::new(256).unwrap())
                .build()
                .await?;

            let handle = session.send_streaming(TTFT_PROMPT).await?;
            let mut stream = handle.into_stream();
            while let Some(event) = stream.next().await {
                match event {
                    Ok(StreamEvent::Delta(StreamDelta::TextDelta { .. })) => break,
                    Ok(_) => continue,
                    Err(e) => anyhow::bail!("Stream error: {e}"),
                }
            }
            // Explicit drop to release the stream. Note: StreamHandle does not
            // cancel via CancellationToken on drop, so the underlying connection
            // may linger briefly. Warmup iterations absorb any residual noise.
            drop(stream);
            Ok(())
        }
    })
    .await
}

/// Scenario: tool_roundtrip — full agent loop with calculator tool.
pub async fn tool_roundtrip(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    let model_str = model.to_string();
    harness::run_scenario("ours/tool_rt", n, warmup, || {
        let model_str = model_str.clone();
        async move {
            let config = make_config(&model_str);
            let provider = OpenAiProvider::new(config)?;
            let session = SessionBuilder::new()
                .with_provider(provider)
                .model(ModelId::new(&model_str)?)
                .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
                .max_tokens(MaxTokens::new(256).unwrap())
                .tool(calculator_tool())
                .build()
                .await?;

            let outcome = AgentLoop::new(&session).run(TOOL_PROMPT).await?;
            let text = outcome.final_response.text();
            if text.is_empty() {
                anyhow::bail!("Empty response from agent loop");
            }
            Ok(())
        }
    })
    .await
}

/// Scenario: mcp_discovery — tool enumeration on an already-connected MCP server.
///
/// Uses `list_raw_tools()` (raw protocol-level list) instead of `discover_tools()`
/// to match rig's `list_tools()` — both return raw tool definitions without
/// schema parsing or wrapper construction.
pub async fn mcp_discovery(_model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    use agent_driver_rs::tool::McpConnection;

    // Setup: connect once before the timed loop
    let conn = McpConnection::connect_stdio(
        "bench-fs",
        "npx",
        &["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    .await
    .expect("Failed to connect to MCP filesystem server");

    let samples = harness::run_scenario("ours/mcp_disc", n, warmup, || {
        let conn = &conn;
        async move {
            let tools = conn.list_raw_tools().await?;
            assert!(!tools.is_empty(), "MCP server returned no tools");
            Ok(())
        }
    })
    .await;

    conn.disconnect().await;
    samples
}

/// Scenario: mcp_roundtrip — full agent loop with MCP-discovered tools.
///
/// Caps `max_tool_depth` at 2 to match rig's `max_turns(2)`, ensuring both
/// libraries do at most 2 tool execution rounds. Uses a single-tool-call prompt
/// ("read Cargo.toml") to avoid depth-cap errors from multi-round listing.
pub async fn mcp_roundtrip(model: &str, n: usize, warmup: usize) -> Vec<Sample> {
    use agent_driver_rs::tool::McpConnection;

    // Setup: connect + discover tools once
    let conn = McpConnection::connect_stdio(
        "bench-fs",
        "npx",
        &["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    .await
    .expect("Failed to connect to MCP filesystem server");

    let mcp_tools = conn
        .discover_tools()
        .await
        .expect("Failed to discover MCP tools");

    let model_str = model.to_string();
    let samples = harness::run_scenario("ours/mcp_rt", n, warmup, || {
        let model_str = model_str.clone();
        let mcp_tools = mcp_tools.clone();
        async move {
            let config = make_config(&model_str);
            let provider = OpenAiProvider::new(config)?;
            let session = SessionBuilder::new()
                .with_provider(provider)
                .model(ModelId::new(&model_str)?)
                .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
                .max_tokens(MaxTokens::new(512).unwrap())
                .tools(mcp_tools)
                .build()
                .await?;

            let config = AgentLoopConfig {
                max_tool_depth: MaxToolDepth::new(2).unwrap(),
                ..Default::default()
            };
            let outcome = AgentLoop::new(&session)
                .with_config(config)
                .run("Read the file named Cargo.toml and tell me the package name.")
                .await?;
            let text = outcome.final_response.text();
            if text.is_empty() {
                anyhow::bail!("Empty response from agent loop");
            }
            Ok(())
        }
    })
    .await;

    conn.disconnect().await;
    samples
}

/// Run all our scenarios and return report rows.
pub async fn run_all(model: &str, n: usize, warmup: usize, scenario_filter: Option<&str>) -> Vec<ReportRow> {
    let mut rows = Vec::new();
    let lib = "agent-driver-rs";

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
