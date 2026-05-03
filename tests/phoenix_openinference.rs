//! OpenInference spec conformance tests for Phoenix tracing.
//!
//! Uses InMemorySpanExporter to capture spans without a running collector.
//! Asserts that all spans carry spec-correct attribute keys and values.
//!
//! Run: cargo test --features "phoenix test-support" --test phoenix_openinference

use std::sync::Arc;

use futures::FutureExt;
use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::export::trace::SpanData;
use opentelemetry_sdk::testing::trace::InMemorySpanExporterBuilder;
use opentelemetry_sdk::trace::{
    SimpleSpanProcessor, TracerProvider as SdkTracerProvider,
};

use agent_driver_rs::agent::AgentLoop;
use agent_driver_rs::otel::attr;
use agent_driver_rs::provider::mock::{
    mock_text_response, mock_tool_call_response, MockProvider,
};
use agent_driver_rs::session::SessionBuilder;
use agent_driver_rs::tool::{FnTool, ToolDefinition, ToolInput, ToolResult, ToolSchema};
use agent_driver_rs::types::{ModelId, ToolName};

fn test_tracer() -> (
    Arc<opentelemetry_sdk::trace::Tracer>,
    opentelemetry_sdk::testing::trace::InMemorySpanExporter,
    Arc<SdkTracerProvider>,
) {
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter.clone())))
        .build();
    let tracer = Arc::new(provider.tracer("test"));
    (tracer, exporter, Arc::new(provider))
}

fn find_span<'a>(spans: &'a [SpanData], name: &str) -> &'a SpanData {
    spans
        .iter()
        .find(|s| s.name.as_ref() == name)
        .unwrap_or_else(|| panic!("span '{}' not found in {:?}", name, spans.iter().map(|s| s.name.as_ref()).collect::<Vec<_>>()))
}

fn get_attr<'a>(span: &'a SpanData, key: &str) -> Option<&'a opentelemetry::Value> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| &kv.value)
}

fn assert_span_kind(span: &SpanData, expected: &str) {
    let kind = get_attr(span, attr::SPAN_KIND)
        .unwrap_or_else(|| panic!("span '{}' missing {}", span.name, attr::SPAN_KIND));
    assert_eq!(
        kind.as_str(),
        expected,
        "span '{}' has wrong span kind: {:?}",
        span.name,
        kind
    );
}

fn echo_tool() -> agent_driver_rs::tool::DynTool {
    let def = ToolDefinition::new(
        ToolName::new("echo").unwrap(),
        "Echos the input back",
        ToolSchema::empty(),
    );
    Arc::new(FnTool::new(
        def,
        |_input: &ToolInput, _ctx: &agent_driver_rs::tool::ToolContext| {
            async move { Ok(ToolResult::text("echoed!")) }.boxed()
        },
    ))
}

fn flush_and_collect(
    provider: &SdkTracerProvider,
    exporter: &opentelemetry_sdk::testing::trace::InMemorySpanExporter,
) -> Vec<SpanData> {
    provider.force_flush();
    exporter.get_finished_spans().unwrap()
}

#[tokio::test]
async fn test_agent_span_is_agent_kind() {
    let (tracer, exporter, provider) = test_tracer();
    let mock = MockProvider::new(vec![mock_text_response("hello")]);
    let session = SessionBuilder::new()
        .with_provider(mock)
        .model(ModelId::new("test-model").unwrap())
        .otel_tracer(tracer)
        .build()
        .await
        .unwrap();

    let _outcome = AgentLoop::new(&session).run("hi").await.unwrap();
    session.shutdown().await;

    let spans = flush_and_collect(&provider, &exporter);
    let agent = find_span(&spans, "agent_loop");
    assert_span_kind(agent, "AGENT");
    assert!(get_attr(agent, attr::AGENT_NAME).is_some());
}

#[tokio::test]
async fn test_session_spans_are_chain_kind() {
    let (tracer, exporter, provider) = test_tracer();
    let mock = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_text_response("done"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(mock)
        .model(ModelId::new("test-model").unwrap())
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await
        .unwrap();

    let _outcome = AgentLoop::new(&session).run("use echo").await.unwrap();
    session.shutdown().await;

    let spans = flush_and_collect(&provider, &exporter);
    let send = find_span(&spans, "session.send");
    assert_span_kind(send, "CHAIN");

    let cont = find_span(&spans, "session.continue");
    assert_span_kind(cont, "CHAIN");
}

#[tokio::test]
async fn test_tool_span_is_tool_kind() {
    let (tracer, exporter, provider) = test_tracer();
    let mock = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_text_response("done"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(mock)
        .model(ModelId::new("test-model").unwrap())
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await
        .unwrap();

    let _outcome = AgentLoop::new(&session).run("use echo").await.unwrap();
    session.shutdown().await;

    let spans = flush_and_collect(&provider, &exporter);
    let tool = find_span(&spans, "tool.echo");
    assert_span_kind(tool, "TOOL");

    let tool_name = get_attr(tool, attr::TOOL_NAME)
        .expect("tool span missing tool.name attribute");
    assert_eq!(tool_name.as_str(), "echo");
}

#[tokio::test]
async fn test_all_spans_have_span_kind() {
    let (tracer, exporter, provider) = test_tracer();
    let mock = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_text_response("done"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(mock)
        .model(ModelId::new("test-model").unwrap())
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await
        .unwrap();

    let _outcome = AgentLoop::new(&session).run("use echo").await.unwrap();
    session.shutdown().await;

    let spans = flush_and_collect(&provider, &exporter);
    assert!(!spans.is_empty(), "no spans exported");
    for span in &spans {
        assert!(
            get_attr(span, attr::SPAN_KIND).is_some(),
            "span '{}' missing {}",
            span.name,
            attr::SPAN_KIND
        );
    }
}

#[tokio::test]
async fn test_span_hierarchy() {
    let (tracer, exporter, provider) = test_tracer();
    let mock = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_text_response("done"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(mock)
        .model(ModelId::new("test-model").unwrap())
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await
        .unwrap();

    let _outcome = AgentLoop::new(&session).run("use echo").await.unwrap();
    session.shutdown().await;

    let spans = flush_and_collect(&provider, &exporter);
    let agent = find_span(&spans, "agent_loop");

    assert!(
        !agent.span_context.span_id().to_string().is_empty(),
        "agent span should have a valid span id"
    );

    let agent_span_id = agent.span_context.span_id();
    for span in &spans {
        if span.name.as_ref() == "agent_loop" {
            continue;
        }
        assert_eq!(
            span.parent_span_id, agent_span_id,
            "span '{}' should be a child of agent_loop",
            span.name
        );
    }
}

#[tokio::test]
async fn test_agent_outcome_attributes() {
    let (tracer, exporter, provider) = test_tracer();
    let mock = MockProvider::new(vec![mock_text_response("hello")]);
    let session = SessionBuilder::new()
        .with_provider(mock)
        .model(ModelId::new("test-model").unwrap())
        .otel_tracer(tracer)
        .build()
        .await
        .unwrap();

    let _outcome = AgentLoop::new(&session).run("hi").await.unwrap();
    session.shutdown().await;

    let spans = flush_and_collect(&provider, &exporter);
    let agent = find_span(&spans, "agent_loop");

    assert!(
        get_attr(agent, attr::AGENT_STOP_REASON).is_some(),
        "agent span missing stop_reason"
    );
    assert!(
        get_attr(agent, attr::AGENT_ITERATIONS).is_some(),
        "agent span missing iterations"
    );
}

#[tokio::test]
async fn test_no_legacy_attribute_keys() {
    let legacy_keys = [
        "llm.model",
        "llm.provider",
        "llm.prompt_tokens",
        "llm.completion_tokens",
        "llm.total_tokens",
        "llm.temperature",
        "agent.iteration",
        "agent.error.tool",
    ];

    let (tracer, exporter, provider) = test_tracer();
    let mock = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_text_response("done"),
    ]);
    let session = SessionBuilder::new()
        .with_provider(mock)
        .model(ModelId::new("test-model").unwrap())
        .tool(echo_tool())
        .otel_tracer(tracer)
        .build()
        .await
        .unwrap();

    let _outcome = AgentLoop::new(&session).run("use echo").await.unwrap();
    session.shutdown().await;

    let spans = flush_and_collect(&provider, &exporter);
    for span in &spans {
        for key in &legacy_keys {
            assert!(
                get_attr(span, key).is_none(),
                "span '{}' has legacy attribute key '{}'",
                span.name,
                key
            );
        }
    }
}
