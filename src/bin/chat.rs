//! Simple CLI chat client
//!
//! This is a minimal chat client that demonstrates the agent-driver-rs library.
//! It loads configuration from environment variables and streams completions to stdout.

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use agent_driver_rs::{
    config::ProviderConfig,
    provider::{CompletionConfig, Provider},
    streaming::{StreamDelta, StreamEvent},
    ModelId, SessionBuilder, SystemPrompt,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Load configuration from environment
    let config = ProviderConfig::from_env().map_err(|e| {
        eprintln!("Configuration error: {}", e);
        eprintln!("\nSet PROVIDER env var to one of: anthropic, openai, bedrock, openrouter, ollama");
        eprintln!("Then set the corresponding API key (e.g., ANTHROPIC_API_KEY)");
        e
    })?;

    config.validate()?;

    println!("Using provider: {}", config.provider_name());

    // Create provider based on config
    let (provider, model_id, completion_config): (Arc<dyn Provider>, ModelId, CompletionConfig) =
        match config {
            ProviderConfig::Anthropic(cfg) => {
                let model_id = ModelId::new(cfg.model.as_str())?;
                let completion_config = CompletionConfig {
                    max_tokens: cfg.max_tokens,
                    temperature: cfg.temperature,
                    stop_sequences: vec![],
                };
                let provider = agent_driver_rs::provider::AnthropicProvider::new(cfg)?;
                (Arc::new(provider), model_id, completion_config)
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
                (Arc::new(provider), model_id, completion_config)
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
                (Arc::new(provider), model_id, completion_config)
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
                (Arc::new(provider), model_id, completion_config)
            }
            #[cfg(feature = "ollama")]
            ProviderConfig::Ollama(cfg) => {
                let model_id = ModelId::new(cfg.model.as_str())?;
                let completion_config = CompletionConfig {
                    max_tokens: cfg
                        .max_tokens
                        .unwrap_or_else(|| agent_driver_rs::MaxTokens::new(4096).unwrap()),
                    temperature: cfg.temperature,
                    stop_sequences: vec![],
                };
                let provider = agent_driver_rs::provider::OllamaProvider::new(cfg)?;
                (Arc::new(provider), model_id, completion_config)
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err("Provider not enabled in features".into());
            }
        };

    println!("Using model: {}", model_id);
    println!("Type your messages below. Press Ctrl+C to exit.\n");

    // Build session
    let session = SessionBuilder::new()
        .provider(provider)
        .model(model_id)
        .completion_config(completion_config)
        .system_prompt(SystemPrompt::new(
            "You are a helpful assistant. Be concise and direct in your responses.",
        ))
        .build()
        .await?;

    // Main chat loop
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
        if input == "/help" {
            println!("Commands:");
            println!("  /quit, /exit - Exit the chat");
            println!("  /clear       - Clear conversation history");
            println!("  /help        - Show this help");
            continue;
        }

        // Send message and stream response
        match session.send_streaming(input).await {
            Ok(stream) => {
                print!("\n");
                futures::pin_mut!(stream);

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) => {
                            print!("{}", text);
                            stdout.flush()?;
                        }
                        Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking })) => {
                            // Show thinking in a different style
                            print!("\x1b[2m{}\x1b[0m", thinking);
                            stdout.flush()?;
                        }
                        Ok(StreamEvent::Completed { metadata }) => {
                            if let Some(usage) = metadata.usage {
                                println!(
                                    "\n\x1b[2m[{} input, {} output tokens]\x1b[0m",
                                    usage.input_tokens, usage.output_tokens
                                );
                            }
                            break;
                        }
                        Ok(StreamEvent::Error { error }) => {
                            eprintln!("\nStream error: {}", error);
                            break;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("\nError: {}", e);
                            break;
                        }
                    }
                }
                println!();
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    println!("Goodbye!");
    session.shutdown().await;

    Ok(())
}
