//! ClusterGuardian SRE agent example
//!
//! Demonstrates programmatic construction of a full agent configuration that
//! mirrors what `ClusterGuardian.config.toml` would specify:
//!
//! - Ollama provider with a custom model and 128K context
//! - Thinking/reasoning mode enabled
//! - 3 MCP HTTP servers (Kubernetes, memory, HITL)
//! - Fallback tool parsing for non-native tool-calling models
//! - Schema sanitization for strict function calling
//! - 100-turn tool depth for long SRE investigations
//! - Single-shot execution: stdin trigger → agent loop → stdout result
//!
//! # Usage
//!
//! ```bash
//! echo "Perform cluster health check" | \
//!   cargo run --example cluster_guardian --features "ollama mcp-http schema-sanitize"
//! ```
//!
//! Set `OLLAMA_HOST` to point at a remote Ollama instance (default: `http://localhost:11434`).
//! Requires a running Ollama instance and the 3 MCP servers to be reachable.

use std::io::{self, Read};

use agent_driver_rs::agent::{AgentEvent, AgentLoop, AgentLoopConfig, AgentObserver, MaxToolDepth};
use agent_driver_rs::config::{NumCtx, OllamaConfig, OllamaModel};
use agent_driver_rs::provider::{CompletionConfig, OllamaProvider};
use agent_driver_rs::tool::{McpHttpSpec, McpManager};
use agent_driver_rs::{MaxTokens, ModelId, SessionBuilder, SystemPrompt};

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

/// Observer tuned for single-shot / cron output.
///
/// Writes tool calls and lifecycle events to stderr so stdout is reserved for
/// the final response.
struct LoggingObserver;

#[async_trait::async_trait]
impl AgentObserver for LoggingObserver {
    async fn on_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::ThinkingDelta { thinking } => {
                eprint!("\x1b[2m{}\x1b[0m", thinking);
            }
            AgentEvent::ToolCallStart { name, .. } => {
                eprintln!("\x1b[33m[tool:start] {}\x1b[0m", name);
            }
            AgentEvent::ToolCallComplete {
                name,
                result,
                is_error,
                ..
            } => {
                if *is_error {
                    eprintln!(
                        "\x1b[31m[tool:error] {}: {}\x1b[0m",
                        name,
                        truncate(result, 200)
                    );
                } else {
                    eprintln!(
                        "\x1b[32m[tool:done]  {}: {}\x1b[0m",
                        name,
                        truncate(result, 200)
                    );
                }
            }
            AgentEvent::IterationStart { iteration } => {
                eprintln!("\x1b[2m[iteration {}]\x1b[0m", iteration);
            }
            AgentEvent::LoopComplete {
                reason,
                total_iterations,
            } => {
                eprintln!(
                    "\x1b[2m[complete: {} ({} iterations)]\x1b[0m",
                    reason, total_iterations
                );
            }
            _ => {}
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut boundary = max;
        while boundary > 0 && !s.is_char_boundary(boundary) {
            boundary -= 1;
        }
        format!("{}...", &s[..boundary])
    }
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = r#"You are ClusterGuardian, an expert Kubernetes SRE agent.

Your responsibilities:
1. Monitor cluster health and diagnose issues
2. Investigate alerts and correlate symptoms across services
3. Execute remediation actions when safe to do so
4. Document findings and actions in memory for future reference
5. Escalate to humans via HITL when unsure

Guidelines:
- Always check cluster state before making changes
- Use memory tools to recall previous incidents and resolutions
- Be thorough: check logs, events, pod status, and resource usage
- When in doubt, ask the human operator via the HITL server
- Never delete resources without human approval"#;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // ── 1. Ollama provider ───────────────────────────────────────────────
    let base_url: url::Url = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://localhost:11434".into())
        .parse()?;
    let max_tokens = MaxTokens::new(2000).expect("2000 is non-zero");

    let ollama_config = OllamaConfig {
        base_url,
        model: OllamaModel::Custom("glm-4.7-flash-128k".into()),
        num_ctx: NumCtx::new(128 * 1024)?, // 128K context window
        max_tokens: Some(max_tokens),
        temperature: None,
        keep_alive: None,
        think: Some(true), // Enable reasoning for complex diagnostics
    };
    let provider = OllamaProvider::new(ollama_config)?;

    // ── 2. Session ───────────────────────────────────────────────────────
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("glm-4.7-flash-128k")?)
        .completion_config(CompletionConfig {
            max_tokens,
            temperature: None,
            stop_sequences: vec![],
        })
        .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
        .sanitize_schemas(true)
        .build()
        .await?;

    // ── 3. MCP HTTP servers (parallel connect) ───────────────────────────
    let mut manager = McpManager::new();
    let errors = manager
        .connect_all_http(vec![
            McpHttpSpec {
                name: "kubernetes".into(),
                uri: "http://kubernetes-mcp-server.example:8080/mcp".into(),
            },
            McpHttpSpec {
                name: "memory".into(),
                uri: "http://basic-memory.example:8000/mcp".into(),
            },
            McpHttpSpec {
                name: "hitl".into(),
                uri: "http://hitl-mcp-server.example:8080/mcp".into(),
            },
        ])
        .await;

    for (name, err) in &errors {
        eprintln!(
            "Warning: failed to connect to MCP server '{}': {}",
            name, err
        );
    }

    // Sync tools from all connected servers
    let (total, sync_errors) = manager
        .sync_all_tools_concurrent(session.tool_registry())
        .await;
    if total > 0 {
        eprintln!(
            "Discovered {} tools from {} server(s)",
            total,
            manager.server_count()
        );
    }
    for err in sync_errors {
        eprintln!("Warning: tool discovery failed: {}", err);
    }

    // Keep MCP connections alive for the duration of the session
    let _mcp_keepalive = manager.into_connections();

    // List registered tools
    let tools = session.list_tools().await;
    if tools.is_empty() {
        eprintln!("Error: no tools available. Check MCP server connectivity.");
        std::process::exit(1);
    }
    eprintln!("Registered tools: {}", tools.len());
    for tool in &tools {
        eprintln!("  - {} ({})", tool.name, tool.description);
    }

    // ── 4. Agent loop config ─────────────────────────────────────────────
    let agent_config = AgentLoopConfig {
        // SRE investigations may chain many diagnostic steps
        max_tool_depth: MaxToolDepth::new(100)?,
        // GLM models emit tool calls as text, not native blocks
        fallback_tool_parsing: true,
        name: Some("ClusterGuardian".into()),
        continue_on_tool_error: true,
    };

    // ── 5. Read trigger from stdin ───────────────────────────────────────
    let mut trigger = String::new();
    io::stdin().read_to_string(&mut trigger)?;
    let trigger = trigger.trim();

    if trigger.is_empty() {
        eprintln!("No trigger provided on stdin. Provide a message, e.g.:");
        eprintln!("  echo \"Perform cluster health check\" | cargo run --example cluster_guardian --features \"ollama mcp-http schema-sanitize\"");
        std::process::exit(1);
    }

    eprintln!("Trigger: {}", trigger);

    // ── 6. Run agent loop ────────────────────────────────────────────────
    let outcome = AgentLoop::new(&session)
        .with_config(agent_config)
        .with_observer(LoggingObserver)
        .run(trigger)
        .await?;

    // Final response to stdout
    println!("{}", outcome.final_response.text());

    // ── 7. Shutdown ──────────────────────────────────────────────────────
    session.shutdown().await;

    Ok(())
}
