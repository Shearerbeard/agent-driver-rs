//! Phoenix Integration Tests - Bedrock Provider
//!
//! Tests Bedrock provider with OTEL tracing to Phoenix.
//!
//! Prerequisites:
//!   1. Configure AWS credentials: aws sso login
//!   2. Set Phoenix endpoint: export PHOENIX_ENDPOINT=http://your-phoenix:4317
//!   3. Run: cargo run --features "phoenix bedrock" --example phoenix_integration_bedrock

use agent_driver_rs::agent::AgentLoop;
use agent_driver_rs::config::BedrockConfig;
use agent_driver_rs::provider::bedrock::BedrockProvider;
use agent_driver_rs::session::SessionBuilder;
use agent_driver_rs::types::ModelId;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Phoenix Integration - Bedrock Provider\n");

    let tracer = agent_driver_rs::otel::init_phoenix().unwrap_or_else(|e| {
        eprintln!("Warning: Phoenix init failed ({}), using no-op tracer", e);
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::default();
        agent_driver_rs::otel::init_tracer_provider(Arc::new(provider));
        Arc::new(agent_driver_rs::otel::get_tracer("phoenix-bedrock").unwrap())
    });

    let config = BedrockConfig::from_env()?;
    let provider = BedrockProvider::new(config).await?;

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("anthropic.claude-sonnet-4-20250514-v1:0")?)
        .otel_tracer(tracer)
        .build()
        .await?;

    let outcome = AgentLoop::new(&session)
        .run("What is 2 + 2? Think carefully about the answer and explain your reasoning.")
        .await?;

    println!("Bedrock test passed");
    println!("Response: {}", outcome.final_response.text());

    session.shutdown().await;
    agent_driver_rs::otel::shutdown_phoenix();

    Ok(())
}
