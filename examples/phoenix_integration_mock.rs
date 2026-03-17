//! Phoenix Integration Tests - Complete Suite
//!
//! Tests trace scenarios with Mock, Bedrock, and Ollama providers.
//!
//! Prerequisites:
//!   1. Start OTel Collector: docker compose -f docker-compose.phoenix.yaml up -d
//!   2. Run specific test: cargo run --features "phoenix" --example phoenix_integration_mock
//!
//! Check spans: docker compose -f docker-compose.phoenix.yaml logs otel-collector | grep -E "span|attributes"

use std::sync::Arc;
use futures::FutureExt;

use agent_driver_rs::provider::mock::{
    mock_content_filter_response, mock_mixed_text_tool_response,
    mock_multi_tool_response, mock_text_response, mock_thinking_response, mock_tool_call_response,
    MockProvider,
};
use agent_driver_rs::session::SessionBuilder;
use agent_driver_rs::tool::{FnTool, ToolDefinition, ToolInput, ToolResult, ToolSchema};
use agent_driver_rs::types::ModelId;
use agent_driver_rs::{agent::AgentLoop, otel};

fn echo_tool() -> agent_driver_rs::tool::DynTool {
    let def = ToolDefinition::new(
        agent_driver_rs::types::ToolName::new("echo").unwrap(),
        "Echos the input back",
        ToolSchema::empty(),
    );
    Arc::new(FnTool::new(def, |_input: &ToolInput, _ctx: &agent_driver_rs::tool::ToolContext| {
        async move { Ok(ToolResult::text("echoed!")) }.boxed()
    }))
}

fn add_tool() -> agent_driver_rs::tool::DynTool {
    let def = ToolDefinition::new(
        agent_driver_rs::types::ToolName::new("add").unwrap(),
        "Adds two numbers",
        ToolSchema::new(
            serde_json::from_str(r#"{
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" }
                },
                "required": ["a", "b"]
            }"#).unwrap()
        ),
    );
    Arc::new(FnTool::new(def, |input: &ToolInput, _ctx: &agent_driver_rs::tool::ToolContext| {
        let a: f64 = input.get("a").unwrap().as_f64().unwrap();
        let b: f64 = input.get("b").unwrap().as_f64().unwrap();
        async move { Ok(ToolResult::text(&(a + b).to_string())) }.boxed()
    }))
}

fn init_tracer() -> Arc<opentelemetry_sdk::trace::Tracer> {
    let provider = opentelemetry_sdk::trace::TracerProvider::default();
    otel::init_tracer_provider(Arc::new(provider));
    Arc::new(otel::get_tracer("phoenix-integration").unwrap())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Phoenix Integration Tests - Mock Provider");
    println!("═══════════════════════════════════════════════\n");

    println!("📡 Using default OTEL tracer (stdout logging)\n");

    let tracer = init_tracer();

    println!("✅ OTel tracer initialized\n");

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Basic text call
    println!("🧪 Test 1: Basic text call");
    match test_basic_text(tracer.clone()).await {
        Ok(_) => { println!("   ✅ Passed\n"); passed += 1; }
        Err(e) => { println!("   ❌ Failed: {}\n", e); failed += 1; }
    }

    // Test 2: Single tool call
    println!("🧪 Test 2: Single tool call");
    match test_single_tool_call(tracer.clone()).await {
        Ok(_) => { println!("   ✅ Passed\n"); passed += 1; }
        Err(e) => { println!("   ❌ Failed: {}\n", e); failed += 1; }
    }

    // Test 3: Multi-tool call
    println!("🧪 Test 3: Multi-tool call (parallel)");
    match test_multi_tool_call(tracer.clone()).await {
        Ok(_) => { println!("   ✅ Passed\n"); passed += 1; }
        Err(e) => { println!("   ❌ Failed: {}\n", e); failed += 1; }
    }

    // Test 4: Tool depth (2 iterations)
    println!("🧪 Test 4: Tool depth (2 iterations)");
    match test_tool_depth(tracer.clone()).await {
        Ok(_) => { println!("   ✅ Passed\n"); passed += 1; }
        Err(e) => { println!("   ❌ Failed: {}\n", e); failed += 1; }
    }

    // Test 5: Thinking/reasoning
    println!("🧪 Test 5: Reasoning/thinking");
    match test_thinking(tracer.clone()).await {
        Ok(_) => { println!("   ✅ Passed\n"); passed += 1; }
        Err(e) => { println!("   ❌ Failed: {}\n", e); failed += 1; }
    }

    // Test 6: Mixed text + tool
    println!("🧪 Test 6: Mixed text + tool");
    match test_mixed_text_tool(tracer.clone()).await {
        Ok(_) => { println!("   ✅ Passed\n"); passed += 1; }
        Err(e) => { println!("   ❌ Failed: {}\n", e); failed += 1; }
    }

    // Test 7: Content filter
    println!("🧪 Test 7: Content filter");
    match test_content_filter(tracer.clone()).await {
        Ok(_) => { println!("   ✅ Passed\n"); passed += 1; }
        Err(e) => { println!("   ❌ Failed: {}\n", e); failed += 1; }
    }

    println!("═══════════════════════════════════════════════");
    println!("📊 Results: {} passed, {} failed", passed, failed);
    println!("═══════════════════════════════════════════════\n");

    if failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

async fn test_basic_text(tracer: Arc<opentelemetry_sdk::trace::Tracer>) -> Result<(), Box<dyn std::error::Error>> {
    let provider = MockProvider::new(vec![mock_text_response("Hello, world!")]);
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("mock-model")?)
        .otel_tracer(tracer)
        .build()
        .await?;
    let outcome = AgentLoop::new(&session).run("Say hello").await?;
    assert_eq!(outcome.final_response.text(), "Hello, world!");
    session.shutdown().await;
    Ok(())
}

async fn test_single_tool_call(tracer: Arc<opentelemetry_sdk::trace::Tracer>) -> Result<(), Box<dyn std::error::Error>> {
    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_text_response("Tool executed!"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("mock-model")?)
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await?;
    let outcome = AgentLoop::new(&session).run("Use the echo tool").await?;
    assert_eq!(outcome.final_response.text(), "Tool executed!");
    session.shutdown().await;
    Ok(())
}

async fn test_multi_tool_call(tracer: Arc<opentelemetry_sdk::trace::Tracer>) -> Result<(), Box<dyn std::error::Error>> {
    let provider = MockProvider::new(vec![
        mock_multi_tool_response(&[
            ("call_1", "add", "{\"a\": 1, \"b\": 2}"),
            ("call_2", "add", "{\"a\": 3, \"b\": 4}"),
        ]),
        mock_text_response("Both tools executed!"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("mock-model")?)
        .tool(add_tool())
        .otel_tracer(tracer)
        .build()
        .await?;
    let outcome = AgentLoop::new(&session).run("Add 1+2 and 3+4").await?;
    assert_eq!(outcome.final_response.text(), "Both tools executed!");
    session.shutdown().await;
    Ok(())
}

async fn test_tool_depth(tracer: Arc<opentelemetry_sdk::trace::Tracer>) -> Result<(), Box<dyn std::error::Error>> {
    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_tool_call_response("call_2", "echo", "{}"),
        mock_text_response("Done after 2 iterations!"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("mock-model")?)
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await?;
    let outcome = AgentLoop::new(&session).run("Echo twice").await?;
    assert_eq!(outcome.final_response.text(), "Done after 2 iterations!");
    session.shutdown().await;
    Ok(())
}

async fn test_thinking(tracer: Arc<opentelemetry_sdk::trace::Tracer>) -> Result<(), Box<dyn std::error::Error>> {
    let provider = MockProvider::new(vec![mock_thinking_response(
        "Let me think about this carefully...",
        "The answer is 42.",
    )]);
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("mock-model")?)
        .otel_tracer(tracer)
        .build()
        .await?;
    let outcome = AgentLoop::new(&session).run("What is the answer?").await?;
    assert_eq!(outcome.final_response.text(), "The answer is 42.");
    session.shutdown().await;
    Ok(())
}

async fn test_mixed_text_tool(tracer: Arc<opentelemetry_sdk::trace::Tracer>) -> Result<(), Box<dyn std::error::Error>> {
    let provider = MockProvider::new(vec![
        mock_mixed_text_tool_response("I'll use a tool:", "call_1", "echo", "{}"),
        mock_text_response("Tool result received!"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("mock-model")?)
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await?;
    let outcome = AgentLoop::new(&session).run("Use echo please").await?;
    assert_eq!(outcome.final_response.text(), "Tool result received!");
    session.shutdown().await;
    Ok(())
}

async fn test_content_filter(tracer: Arc<opentelemetry_sdk::trace::Tracer>) -> Result<(), Box<dyn std::error::Error>> {
    let provider = MockProvider::new(vec![mock_content_filter_response("Partial output...")]);
    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(ModelId::new("mock-model")?)
        .otel_tracer(tracer)
        .build()
        .await?;
    let outcome = AgentLoop::new(&session).run("Generate content").await?;
    assert!(outcome.final_response.text().contains("Partial output"));
    session.shutdown().await;
    Ok(())
}
