//! Phoenix Integration Tests - Ollama Provider
//!
//! Tests Ollama provider with OTEL tracing to Phoenix.
//!
//! Prerequisites:
//!   1. Start Ollama server: ollama serve
//!   2. Set Phoenix endpoint: export PHOENIX_ENDPOINT=http://your-phoenix:4317
//!   3. Run: cargo run --features "phoenix ollama" --example phoenix_integration_ollama

use agent_driver_rs::agent::AgentLoop;
use agent_driver_rs::config::OllamaConfig;
use agent_driver_rs::provider::ollama::OllamaProvider;
use agent_driver_rs::session::SessionBuilder;
use agent_driver_rs::types::ModelId;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Phoenix Integration - Ollama Provider\n");

    let tracer = agent_driver_rs::otel::init_phoenix().unwrap_or_else(|e| {
        eprintln!("Warning: Phoenix init failed ({}), using no-op tracer", e);
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::default();
        agent_driver_rs::otel::init_tracer_provider(Arc::new(provider));
        Arc::new(agent_driver_rs::otel::get_tracer("phoenix-ollama").unwrap())
    });

    let config = OllamaConfig::from_env()?;
    let model_name = config.model.as_str().to_string();
    let provider = OllamaProvider::new(config)?;

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new(&model_name)?)
        .otel_tracer(tracer)
        .build()
        .await?;

    let outcome = AgentLoop::new(&session)
        .run("What is 2 + 2? Think carefully about the answer and explain your reasoning.")
        .await?;

    println!("Ollama test passed");
    println!("Response: {}", outcome.final_response.text());

    session.shutdown().await;
    agent_driver_rs::otel::shutdown_phoenix();

    Ok(())
}
