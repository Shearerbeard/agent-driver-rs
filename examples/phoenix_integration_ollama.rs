//! Phoenix Integration Tests - Ollama Provider
//!
//! Tests Ollama provider with reasoning capabilities and OTEL tracing.
//!
//! Prerequisites:
//!   1. Start OTel Collector: docker compose -f docker-compose.phoenix.yaml up -d
//!   2. Start Ollama server: ollama serve
//!   3. Run test: cargo run --features "phoenix ollama" --example phoenix_integration_ollama
//!
//! Check spans: docker compose -f docker-compose.phoenix.yaml logs -f otel-collector

use agent_driver_rs::agent::AgentLoop;
use agent_driver_rs::config::OllamaConfig;
use agent_driver_rs::provider::ollama::OllamaProvider;
use agent_driver_rs::session::SessionBuilder;
use agent_driver_rs::types::ModelId;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Phoenix Integration Tests - Ollama Provider");
    println!("═══════════════════════════════════════════════\n");

    // Test Ollama provider with reasoning
    println!("🧪 Testing Ollama Provider with Reasoning");

    let config = OllamaConfig::from_env()?;
    let model_name = config.model.as_str().to_string();
    let provider = OllamaProvider::new(config)?;

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new(&model_name)?)
        .build()
        .await?;

    let outcome = AgentLoop::new(&session)
        .run("What is 2 + 2? Think carefully about the answer and explain your reasoning.")
        .await?;

    println!("✅ Ollama test passed");
    println!("Response: {}", outcome.final_response.text());

    session.shutdown().await;

    Ok(())
}
