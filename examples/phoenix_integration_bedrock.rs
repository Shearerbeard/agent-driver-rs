//! Phoenix Integration Tests - Bedrock Provider
//!
//! Tests Bedrock provider with reasoning capabilities and OTEL tracing.
//!
//! Prerequisites:
//!   1. Start OTel Collector: docker compose -f docker-compose.phoenix.yaml up -d
//!   2. Configure AWS credentials: aws sso login
//!   3. Run test: cargo run --features "phoenix bedrock" --example phoenix_integration_bedrock
//!
//! Check spans: docker compose -f docker-compose.phoenix.yaml logs -f otel-collector

use agent_driver_rs::agent::AgentLoop;
use agent_driver_rs::config::BedrockConfig;
use agent_driver_rs::provider::bedrock::BedrockProvider;
use agent_driver_rs::session::SessionBuilder;
use agent_driver_rs::types::ModelId;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Phoenix Integration Tests - Bedrock Provider");
    println!("═══════════════════════════════════════════════\n");

    // Test Bedrock provider with reasoning
    println!("🧪 Testing Bedrock Provider with Reasoning");

    let config = BedrockConfig::from_env()?;
    let provider = BedrockProvider::new(config).await?;

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("anthropic.claude-sonnet-4-20250514-v1:0")?)
        .build()
        .await?;

    let outcome = AgentLoop::new(&session)
        .run("What is 2 + 2? Think carefully about the answer and explain your reasoning.")
        .await?;

    println!("✅ Bedrock test passed");
    println!("Response: {}", outcome.final_response.text());

    session.shutdown().await;

    Ok(())
}
