use std::hint::black_box;
use std::sync::Arc;

use agent_driver_rs::agent::{AgentLoop, AgentLoopConfig, MaxToolDepth};
use agent_driver_rs::config::{ApiKey, CustomModelCapabilities, OpenAiConfig, OpenAiModel};
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

/// Build an explicit custom capability declaration from the
/// `OPENAI_CUSTOM_TEMPERATURE`, `OPENAI_CUSTOM_REASONING`, and
/// `OPENAI_CUSTOM_STREAMING` environment variables.
///
/// All three must be present and strictly `true`/`false`; a partial or
/// malformed declaration fails loudly, matching the provider's config
/// contract. Custom capabilities are never inferred from the model name.
fn custom_capabilities_from_env(model: &str) -> CustomModelCapabilities {
    let flag = |var: &str| -> bool {
        match std::env::var(var).as_deref() {
            Ok("true") => true,
            Ok("false") => false,
            Ok(other) => panic!("{var} must be \"true\" or \"false\", got {other:?}"),
            Err(_) => panic!(
                "custom model {model:?} requires {var}; set OPENAI_CUSTOM_TEMPERATURE, \
                 OPENAI_CUSTOM_REASONING, and OPENAI_CUSTOM_STREAMING to \"true\" or \"false\""
            ),
        }
    };

    CustomModelCapabilities::new(
        flag("OPENAI_CUSTOM_TEMPERATURE"),
        flag("OPENAI_CUSTOM_REASONING"),
        flag("OPENAI_CUSTOM_STREAMING"),
    )
}

/// Build an OpenAI config for the given model string.
fn make_config(model: &str) -> OpenAiConfig {
    let api_key = ApiKey::new(std::env::var("OPENAI_API_KEY").unwrap())
        .expect("OPENAI_API_KEY must be set and non-empty");
    let custom_capabilities = match OpenAiModel::from_id(model) {
        OpenAiModel::Custom(_) => Some(custom_capabilities_from_env(model)),
        _ => None,
    };
    build_config(model, api_key, custom_capabilities)
}

/// Assemble a config from a model string and an explicit (optional)
/// custom capability declaration. Pure so the benchmark's model wiring
/// is testable without touching the environment.
fn build_config(
    model: &str,
    api_key: ApiKey,
    custom_capabilities: Option<CustomModelCapabilities>,
) -> OpenAiConfig {
    OpenAiConfig {
        api_key,
        api: Default::default(),
        base_url: None,
        model: OpenAiModel::from_id(model),
        max_tokens: MaxTokens::new(256).unwrap(),
        temperature: None,
        reasoning: None,
        structured_output: false,
        chat_reasoning_output: Default::default(),
        custom_capabilities,
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
pub async fn run_all(
    model: &str,
    n: usize,
    warmup: usize,
    scenario_filter: Option<&str>,
) -> Vec<ReportRow> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_api_key() -> ApiKey {
        ApiKey::new("bench-test-key").expect("test key is non-empty")
    }

    #[test]
    fn known_model_builds_without_custom_capabilities() {
        let config = build_config("gpt-4o-mini", test_api_key(), None);
        assert!(matches!(config.model, OpenAiModel::Gpt4oMini));
        assert!(config.custom_capabilities.is_none());
        OpenAiProvider::new(config).expect("known model config must construct");
    }

    /// A custom model with an explicit capability declaration builds and
    /// constructs — the benchmark's custom-model smoke path.
    #[test]
    fn custom_model_with_declared_capabilities_constructs() {
        let config = build_config(
            "org/local-llm",
            test_api_key(),
            Some(CustomModelCapabilities::new(true, false, true)),
        );
        assert!(matches!(config.model, OpenAiModel::Custom(ref m) if m == "org/local-llm"));
        OpenAiProvider::new(config).expect("custom model with capabilities must construct");
    }

    /// A custom model with no declaration must fail at provider
    /// construction — capability is never inferred from the model name.
    #[test]
    fn custom_model_without_capabilities_fails_loud() {
        let config = build_config("gpt-5.6-terra", test_api_key(), None);
        let err = match OpenAiProvider::new(config) {
            Ok(_) => panic!("missing declaration must fail"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(
            message.contains("custom_capabilities") || message.contains("capability"),
            "error must name the missing declaration, got: {message}"
        );
    }

}
