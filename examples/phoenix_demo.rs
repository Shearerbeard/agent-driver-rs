//! Example: Arize Phoenix tracing integration
//!
//! Demonstrates agent-driver-rs with Phoenix tracing via OTLP.
//!
//! Run with:
//! ```bash
//! PHOENIX_ENDPOINT=http://your-phoenix:4317 cargo run --features phoenix --example phoenix_demo
//! ```

#[cfg(feature = "phoenix")]
use agent_driver_rs::SessionBuilder;
#[cfg(feature = "phoenix")]
use std::sync::Arc;

#[cfg(feature = "phoenix")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Phoenix-enabled agent execution...");

    let tracer = agent_driver_rs::otel::init_phoenix().unwrap_or_else(|e| {
        eprintln!("Warning: Phoenix init failed ({}), using no-op tracer", e);
        let provider = opentelemetry_sdk::trace::TracerProvider::default();
        agent_driver_rs::otel::init_tracer_provider(Arc::new(provider));
        Arc::new(agent_driver_rs::otel::get_tracer("agent-driver-rs").unwrap())
    });

    println!(
        "Phoenix tracer initialized (endpoint: {})",
        std::env::var("PHOENIX_ENDPOINT").unwrap_or_else(|_| "localhost:4317".into())
    );

    let provider = agent_driver_rs::provider::MockProvider::new(vec![vec![
        agent_driver_rs::streaming::StreamEvent::Delta(
            agent_driver_rs::streaming::StreamDelta::TextDelta {
                text: "I can help you find weather information. What city are you interested in?"
                    .into(),
            },
        ),
    ]]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(agent_driver_rs::types::ModelId::new("gpt-4").unwrap())
        .otel_tracer(tracer)
        .build()
        .await
        .map_err(|e| format!("Failed to build session: {}", e))?;

    let outcome = agent_driver_rs::agent::AgentLoop::new(&session)
        .run("What is the weather in San Francisco?")
        .await
        .map_err(|e| format!("Agent loop failed: {}", e))?;

    println!("Agent loop completed!");
    println!("Final response: {}", outcome.final_response.text());
    println!("Total iterations: {}", outcome.iterations);
    println!("Session ID: {}", session.session_id());

    session.shutdown().await;
    agent_driver_rs::otel::shutdown_phoenix();

    Ok(())
}

#[cfg(not(feature = "phoenix"))]
fn main() {
    println!("Phoenix feature is not enabled.");
    println!("Run with: PHOENIX_ENDPOINT=http://your-phoenix:4317 cargo run --features phoenix --example phoenix_demo");
}
