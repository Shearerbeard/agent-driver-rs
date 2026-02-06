//! CLI chat client with agentic tool loop and optional MCP server support
//!
//! This chat client demonstrates the agent-driver-rs library with:
//! - Streaming responses via the AgentLoop
//! - Dynamic tool calling (agent loop handles tool execution automatically)
//! - Optional MCP server connections for additional tools
//!
//! # Examples
//!
//! ```bash
//! # Basic chat
//! PROVIDER=openrouter cargo run --features openrouter --bin chat
//!
//! # With MCP server
//! PROVIDER=openrouter cargo run --features "openrouter mcp" --bin chat -- \
//!     --mcp "npx -y @anthropic/mcp-server-time"
//!
//! # With MCP config file
//! PROVIDER=openrouter cargo run --features "openrouter mcp" --bin chat -- \
//!     --mcp-config servers.json
//! ```

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use clap::Parser;

use agent_driver_rs::{
    agent::{AgentEvent, AgentLoop, AgentLoopConfig, AgentObserver, MaxToolDepth},
    config::ProviderConfig,
    provider::{CompletionConfig, Provider},
    ModelId, SessionBuilder, SystemPrompt,
};

/// CLI arguments
#[derive(Parser, Debug)]
#[command(name = "chat", about = "Agentic chat client for agent-driver-rs")]
struct Args {
    /// MCP server commands (e.g., "npx -y @anthropic/mcp-server-time")
    /// Can be specified multiple times for multiple servers
    #[arg(long = "mcp", value_name = "COMMAND")]
    mcp_servers: Vec<String>,

    /// Path to MCP server config file (JSON)
    #[cfg(feature = "mcp")]
    #[arg(long = "mcp-config", value_name = "PATH")]
    mcp_config: Option<String>,

    /// Maximum tool execution depth (default: 25)
    #[arg(long, default_value = "25")]
    max_tool_depth: u32,
}

/// Observer that prints streaming output to stdout
struct ChatObserver;

#[async_trait::async_trait]
impl AgentObserver for ChatObserver {
    async fn on_event(&self, event: &AgentEvent) {
        let mut stdout = io::stdout();
        match event {
            AgentEvent::TextDelta { text } => {
                print!("{}", text);
                let _ = stdout.flush();
            }
            AgentEvent::ThinkingDelta { thinking } => {
                // Show thinking in dim style
                print!("\x1b[2m{}\x1b[0m", thinking);
                let _ = stdout.flush();
            }
            AgentEvent::ToolCallStart { name, .. } => {
                eprintln!("\x1b[33m[calling tool: {}]\x1b[0m", name);
            }
            AgentEvent::ToolCallComplete {
                name,
                result,
                is_error,
                ..
            } => {
                if *is_error {
                    eprintln!(
                        "\x1b[31m[tool {} error: {}]\x1b[0m",
                        name,
                        truncate(result, 200)
                    );
                } else {
                    eprintln!(
                        "\x1b[32m[tool {} done: {}]\x1b[0m",
                        name,
                        truncate(result, 200)
                    );
                }
            }
            AgentEvent::IterationStart { iteration } => {
                eprintln!("\x1b[2m[tool iteration {}]\x1b[0m", iteration);
            }
            AgentEvent::LoopComplete {
                reason,
                total_iterations,
            } => {
                if *total_iterations > 0 {
                    eprintln!(
                        "\x1b[2m[loop done: {}, {} tool iteration(s)]\x1b[0m",
                        reason, total_iterations
                    );
                }
            }
            _ => {}
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        // Find a char boundary at or before `max` to avoid panicking on multi-byte UTF-8
        let mut boundary = max;
        while boundary > 0 && !s.is_char_boundary(boundary) {
            boundary -= 1;
        }
        format!("{}...", &s[..boundary])
    }
}

/// MCP server config file format
#[cfg(feature = "mcp")]
#[derive(serde::Deserialize)]
struct McpConfigFile {
    servers: Vec<McpServerEntry>,
}

#[cfg(feature = "mcp")]
#[derive(serde::Deserialize)]
struct McpServerEntry {
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

/// Keepalive container for MCP connections so child processes don't get dropped
#[cfg(feature = "mcp")]
struct McpKeepAlive {
    connections: Vec<agent_driver_rs::tool::McpConnection>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Logging ──────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    // ── Configuration ─────────────────────────────────────────────────
    let config = ProviderConfig::from_env().map_err(|e| {
        eprintln!("Configuration error: {}", e);
        eprintln!(
            "\nSet PROVIDER env var to one of: anthropic, openai, bedrock, openrouter, ollama"
        );
        eprintln!("Then set the corresponding API key (e.g., ANTHROPIC_API_KEY)");
        e
    })?;

    config.validate()?;

    println!("Using provider: {}", config.provider_kind());

    // ── Provider creation ──────────────────────────────────────────────
    let (provider, model_id, completion_config) = create_provider(config).await?;

    println!("Using model: {}", model_id);

    // ── Session setup ──────────────────────────────────────────────────
    let session = SessionBuilder::new()
        .provider(provider)
        .model(model_id)
        .completion_config(completion_config)
        .system_prompt(SystemPrompt::new(
            "You are a helpful assistant. Be concise and direct in your responses. \
             When you have tools available, use them to answer questions accurately.",
        ))
        .build()
        .await?;

    // ── MCP server connections (optional) ─────────────────────────────
    #[cfg(feature = "mcp")]
    let _mcp_keepalive = setup_mcp_connections(&args, &session).await?;

    #[cfg(not(feature = "mcp"))]
    if !args.mcp_servers.is_empty() {
        eprintln!(
            "Warning: --mcp flag requires the 'mcp' feature. Recompile with --features mcp"
        );
    }

    // ── Show registered tools ─────────────────────────────────────────
    let tools = session.list_tools().await;
    if !tools.is_empty() {
        println!("Registered tools: {}", tools.len());
        for tool in &tools {
            println!("  - {} ({})", tool.name, tool.description);
        }
    }

    println!("Type your messages below. Press Ctrl+C to exit.\n");

    // ── Agent loop config ─────────────────────────────────────────────
    let agent_config = AgentLoopConfig {
        max_tool_depth: MaxToolDepth::new(args.max_tool_depth)?,
        continue_on_tool_error: true,
    };

    // ── Main chat REPL ────────────────────────────────────────────────
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            // EOF
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle special commands
        if input == "/quit" || input == "/exit" {
            break;
        }
        if input == "/clear" {
            session.clear_messages().await;
            println!("Conversation cleared.");
            continue;
        }
        if input == "/tools" {
            let tools = session.list_tools().await;
            if tools.is_empty() {
                println!("No tools registered.");
            } else {
                println!("Registered tools ({}):", tools.len());
                for tool in &tools {
                    println!("  - {} — {}", tool.name, tool.description);
                }
            }
            continue;
        }
        if input == "/help" {
            println!("Commands:");
            println!("  /quit, /exit - Exit the chat");
            println!("  /clear       - Clear conversation history");
            println!("  /tools       - List registered tools");
            println!("  /help        - Show this help");
            continue;
        }

        // Run the agent loop
        println!();
        match AgentLoop::new(&session)
            .with_config(agent_config.clone())
            .with_observer(ChatObserver)
            .run(input)
            .await
        {
            Ok(outcome) => {
                if let Some(usage) = outcome.final_response.metadata.usage {
                    println!(
                        "\n\x1b[2m[{} input, {} output tokens]\x1b[0m",
                        usage.input_tokens, usage.output_tokens
                    );
                }
                println!();
            }
            Err(e) => {
                eprintln!("\nError: {}", e);
                println!();
            }
        }
    }

    println!("Goodbye!");
    session.shutdown().await;

    Ok(())
}

/// Create a provider, model ID, and completion config from the unified config enum.
async fn create_provider(
    config: ProviderConfig,
) -> Result<(Arc<dyn Provider>, ModelId, CompletionConfig), Box<dyn std::error::Error>> {
    match config {
        ProviderConfig::Anthropic(cfg) => {
            let model_id = ModelId::new(cfg.model.as_str())?;
            let completion_config = CompletionConfig {
                max_tokens: cfg.max_tokens,
                temperature: cfg.temperature,
                stop_sequences: vec![],
            };
            let provider = agent_driver_rs::provider::AnthropicProvider::new(cfg)?;
            Ok((Arc::new(provider), model_id, completion_config))
        }
        #[cfg(feature = "openai")]
        ProviderConfig::OpenAi(cfg) => {
            let model_id = ModelId::new(cfg.model.as_str())?;
            let completion_config = CompletionConfig {
                max_tokens: cfg.max_tokens,
                temperature: cfg.temperature,
                stop_sequences: vec![],
            };
            let provider = agent_driver_rs::provider::OpenAiProvider::new(cfg)?;
            Ok((Arc::new(provider), model_id, completion_config))
        }
        #[cfg(feature = "bedrock")]
        ProviderConfig::Bedrock(cfg) => {
            let model_id = ModelId::new(cfg.model.model_id())?;
            let completion_config = CompletionConfig {
                max_tokens: cfg.max_tokens,
                temperature: cfg.temperature,
                stop_sequences: vec![],
            };
            let provider = agent_driver_rs::provider::BedrockProvider::new(cfg).await?;
            Ok((Arc::new(provider), model_id, completion_config))
        }
        #[cfg(feature = "openrouter")]
        ProviderConfig::OpenRouter(cfg) => {
            let model_id = ModelId::new(cfg.model.as_str())?;
            let completion_config = CompletionConfig {
                max_tokens: cfg.max_tokens,
                temperature: cfg.temperature,
                stop_sequences: vec![],
            };
            let provider = agent_driver_rs::provider::OpenRouterProvider::new(cfg)?;
            Ok((Arc::new(provider), model_id, completion_config))
        }
        #[cfg(feature = "ollama")]
        ProviderConfig::Ollama(cfg) => {
            let model_id = ModelId::new(cfg.model.as_str())?;
            let completion_config = CompletionConfig {
                max_tokens: cfg
                    .max_tokens
                    // Safe: 4096 is a hardcoded non-zero constant
                    .unwrap_or_else(|| agent_driver_rs::MaxTokens::new(4096).expect("4096 is non-zero")),
                temperature: cfg.temperature,
                stop_sequences: vec![],
            };
            let provider = agent_driver_rs::provider::OllamaProvider::new(cfg)?;
            Ok((Arc::new(provider), model_id, completion_config))
        }
        #[allow(unreachable_patterns)]
        _ => Err("Provider not enabled in features".into()),
    }
}

/// Connect to MCP servers from CLI args and config file.
///
/// Returns a keepalive handle that must be held for the duration of the session
/// to prevent the child processes from being dropped.
#[cfg(feature = "mcp")]
async fn setup_mcp_connections(
    args: &Args,
    session: &agent_driver_rs::Session,
) -> Result<McpKeepAlive, Box<dyn std::error::Error>> {
    let mut keepalive = McpKeepAlive {
        connections: Vec::new(),
    };

    // From CLI --mcp args
    for (i, server_cmd) in args.mcp_servers.iter().enumerate() {
        let parts: Vec<&str> = server_cmd.split_whitespace().collect();
        if parts.is_empty() {
            eprintln!("Warning: empty MCP server command, skipping");
            continue;
        }
        let name = format!("mcp-{}", i);
        let command = parts[0];
        let cmd_args: Vec<&str> = parts[1..].to_vec();
        eprintln!("Connecting to MCP server '{}': {}", name, server_cmd);
        match agent_driver_rs::tool::McpConnection::connect_stdio(&name, command, &cmd_args).await {
            Ok(conn) => {
                match conn.sync_tools(session.tool_registry()).await {
                    Ok(count) => eprintln!("  Discovered {} tools from '{}'", count, name),
                    Err(e) => eprintln!(
                        "  Warning: failed to discover tools from '{}': {}",
                        name, e
                    ),
                }
                keepalive.connections.push(conn);
            }
            Err(e) => {
                eprintln!(
                    "  Warning: failed to connect to MCP server '{}': {}",
                    name, e
                );
            }
        }
    }

    // From --mcp-config file
    if let Some(config_path) = &args.mcp_config {
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read MCP config '{}': {}", config_path, e))?;
        let config_file: McpConfigFile = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse MCP config '{}': {}", config_path, e))?;

        for entry in &config_file.servers {
            let cmd_args: Vec<&str> = entry.args.iter().map(|s| s.as_str()).collect();
            eprintln!(
                "Connecting to MCP server '{}': {} {}",
                entry.name,
                entry.command,
                entry.args.join(" ")
            );
            match agent_driver_rs::tool::McpConnection::connect_stdio(
                &entry.name,
                &entry.command,
                &cmd_args,
            )
            .await
            {
                Ok(conn) => {
                    match conn.sync_tools(session.tool_registry()).await {
                        Ok(count) => {
                            eprintln!("  Discovered {} tools from '{}'", count, entry.name)
                        }
                        Err(e) => eprintln!(
                            "  Warning: failed to discover tools from '{}': {}",
                            entry.name, e
                        ),
                    }
                    keepalive.connections.push(conn);
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: failed to connect to MCP server '{}': {}",
                        entry.name, e
                    );
                }
            }
        }
    }

    Ok(keepalive)
}
