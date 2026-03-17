//! Example: Arize Phoenix tracing integration
//!
//! This example demonstrates how to use Arize Phoenix with agent-driver-rs
//! to trace and visualize LLM agent execution.
//!
//! Run with:
//! ```bash
//! cargo run --example phoenix_demo
//! ```
//!
//! Prerequisites:
//! 1. Start Phoenix server: `cargo install arize-phoenix && phoenix --server`
//! 2. Set environment variable: `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317`
//!
//! View traces: https://phoenix.arize.com

#[cfg(feature = "phoenix")]
use agent_driver_rs::SessionBuilder;
#[cfg(feature = "phoenix")]
use opentelemetry_sdk::trace::TracerProvider;
#[cfg(feature = "phoenix")]
use std::sync::Arc;

#[cfg(feature = "phoenix")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Phoenix-enabled agent execution...");

    // Initialize Phoenix exporter
    // Note: This is a simplified example. In production, use:
    //
    // ```ignore
    // use opentelemetry_otlp::SpanExporter;
    // let exporter = SpanExporter::builder()
    //     .with_http()
    //     .with_endpoint("http://localhost:4317")
    //     .build()?;
    // let tracer_provider = TracerProvider::builder()
    //     .with_batch_exporter(exporter)
    //     .build();
    // ```
    //
    // For gRPC (port 4317):
    // ```ignore
    // use opentelemetry_otlp::SpanExporter;
    // let exporter = SpanExporter::builder()
    //     .with_grpc()
    //     .with_endpoint("http://localhost:4317")
    //     .build()?;
    // ```

    // For this example, we just create a simple provider
    let tracer_provider = TracerProvider::default();

    // Initialize Phoenix tracer provider globally
    agent_driver_rs::otel::init_tracer_provider(Arc::new(tracer_provider));

    println!("✅ Phoenix tracer provider initialized");

    // Create a mock provider (replace with real provider in production)
    let provider = agent_driver_rs::provider::MockProvider::new(vec![
        vec![agent_driver_rs::streaming::StreamEvent::Delta(
            agent_driver_rs::streaming::StreamDelta::TextDelta {
                text: "I can help you find weather information. What city are you interested in?".into(),
            },
        )],
    ]);

    // Build session with Phoenix tracing enabled
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(agent_driver_rs::types::ModelId::new("gpt-4").unwrap())
        .otel_tracer(
            Arc::new(agent_driver_rs::otel::get_tracer("agent-driver-rs").unwrap()),
        )
        .build()
        .await
        .map_err(|e| format!("Failed to build session: {}", e))?;

    println!("🤖 Starting agent loop with Phoenix tracing...");

    // Run agent loop - traces are automatically captured
    let outcome = agent_driver_rs::agent::AgentLoop::new(&session)
        .run("What is the weather in San Francisco?")
        .await
        .map_err(|e| format!("Agent loop failed: {}", e))?;

    println!("✅ Agent loop completed!");
    println!("📊 Final response: {}", outcome.final_response.text());
    println!("🔄 Total iterations: {}", outcome.iterations);

    // Verify traces were sent to Phoenix
    #[cfg(feature = "phoenix")]
    {
        println!("\n🔍 Verify traces in Phoenix UI:");
        println!("   Visit: https://phoenix.arize.com");
        println!("   Search for session ID: {}", session.session_id());
        println!("   Look for:");
        println!("   - 'agent_loop' root span");
        println!("   - 'session.send' and 'session.continue' spans");
        println!("   - 'llm.completion' spans for provider calls");
        println!("   - 'tool_execution' spans (if tools are used)");
    }

    // Clean up
    session.shutdown().await;
    println!("👋 Session shut down gracefully");

    Ok(())
}

// For non-phoenix builds, provide a placeholder main
#[cfg(not(feature = "phoenix"))]
fn main() {
    println!("⚠️  Phoenix feature is not enabled.");
    println!("   Enable with: cargo run --example phoenix_demo --features phoenix");
    println!("\n   To use Phoenix:");
    println!("   1. Install Phoenix: cargo install arize-phoenix");
    println!("   2. Start Phoenix server: phoenix --server");
    println!("   3. Run with environment variable:");
    println!("      PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317 cargo run --example phoenix_demo --features phoenix");
}
