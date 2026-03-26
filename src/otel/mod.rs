//! OpenTelemetry / Arize Phoenix tracing support for agent-driver-rs
//!
//! This module provides integration with Arize Phoenix, an open-source LLM
//! observability platform built on OpenTelemetry. It enables tracing of:
//! - LLM completions (provider-level)
//! - Agent loop execution (multi-turn orchestration)
//! - Tool execution (individual tool calls)
//! - Session operations (send/receive)
//!
//! ## Usage
//!
//! Enable the `phoenix` feature to use Phoenix tracing:
//!
//! ```toml
//! [dependencies]
//! agent-driver-rs = { version = "0.1.0", features = ["phoenix"] }
//! ```
//!
//! Then initialize Phoenix in your application:
//!
//! ```ignore
//! use agent_driver_rs::SessionBuilder;
//! use opentelemetry_sdk::trace::TracerProvider;
//! use opentelemetry_otlp::SpanExporter;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize Phoenix exporter (HTTP on port 4318)
//!     let exporter = SpanExporter::builder()
//!         .with_http()
//!         .with_endpoint("http://phoenix-collector:4318")
//!         .build()?;
//!
//!     let tracer_provider = TracerProvider::builder()
//!         .with_batch_exporter(exporter)
//!         .build();
//!
//!     // Build session with Phoenix tracing enabled
//!     let session = SessionBuilder::new()
//!         .with_provider(my_provider)
//!         .build()
//!         .await?;
//!
//!     // Run agent loop - traces are automatically captured
//!     let outcome = agent_driver_rs::agent::AgentLoop::new(&session)
//!         .run("Search current weather")
//!         .await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Design Principles
//!
//! - **Finite types**: All OTEL configurations use enums and newtypes
//! - **Split locks**: Tracing operations don't block session state
//! - **RAII guards**: Automatic span lifecycle management
//! - **Streaming-aware**: Spans properly handle streaming operations
//! - **Non-invasive**: No changes to provider implementations needed

pub mod instrumentation;
pub mod types;

#[cfg(feature = "phoenix")]
use crate::Session;
#[cfg(feature = "phoenix")]
use opentelemetry::trace::TracerProvider;
#[cfg(feature = "phoenix")]
use opentelemetry_sdk::trace::TracerProvider as SdkTracerProvider;

#[cfg(feature = "phoenix")]
pub use instrumentation::*;
#[cfg(feature = "phoenix")]
pub use types::{
    AgentLoopAttributes, AgentStopReason, CompletionAttributes, CompletionStatus, LlmFinishReason,
    OtelEndpoint, OtlpExporterConfig, ProviderKind, SessionAttributes, SpanKind, SpanName,
    ToolExecutionAttributes, ToolExecutionResult,
};

/// Global tracer provider for Phoenix traces
///
/// This singleton is initialized once and used across all sessions.
/// In production, this should be configured with a Phoenix exporter.
#[cfg(feature = "phoenix")]
pub static PHOENIX_TRACER_PROVIDER: std::sync::OnceLock<std::sync::Arc<SdkTracerProvider>> =
    std::sync::OnceLock::new();

/// Initialize the global tracer provider with Phoenix exporter
///
/// This should be called once at application startup.
///
/// # Arguments
///
/// * `endpoint` - Phoenix collector endpoint URL (e.g., "http://localhost:4317")
/// * `headers` - Optional headers for the OTLP request
///
/// # Example
///
/// ```ignore
/// use opentelemetry_otlp::SpanExporter;
/// use opentelemetry_sdk::trace::TracerProvider;
/// use std::sync::Arc;
///
/// let exporter = SpanExporter::builder()
///     .with_http()
///     .with_endpoint("http://localhost:4318")
///     .build()?;
///
/// let provider = TracerProvider::builder()
///     .with_batch_exporter(exporter)
///     .build();
///
/// agent_driver_rs::otel::init_tracer_provider(Arc::new(provider));
/// ```
#[cfg(feature = "phoenix")]
pub fn init_tracer_provider(provider: std::sync::Arc<SdkTracerProvider>) {
    PHOENIX_TRACER_PROVIDER.set(provider).ok();
}

/// Get the global tracer provider
///
/// Returns `None` if Phoenix is not initialized or the feature is disabled.
#[cfg(feature = "phoenix")]
pub fn get_tracer_provider() -> Option<std::sync::Arc<SdkTracerProvider>> {
    PHOENIX_TRACER_PROVIDER.get().cloned()
}

/// Get a tracer from the global provider
///
/// Returns a tracing error if Phoenix is not initialized.
#[cfg(feature = "phoenix")]
pub fn get_tracer(name: &str) -> Result<opentelemetry_sdk::trace::Tracer, String> {
    let provider = get_tracer_provider()
        .ok_or_else(|| "Phoenix tracer provider not initialized".to_string())?;

    Ok(provider.tracer(name.to_string()))
}

/// Enable Phoenix tracing for a session
///
/// This wraps the session in a span that tracks all operations.
#[cfg(feature = "phoenix")]
pub fn enable_session_tracing(_session: &Session) -> Option<SessionOperationSpan> {
    get_tracer("agent-driver-rs")
        .ok()
        .and_then(|tracer| SessionOperationSpan::new(&tracer, "session.operation").ok())
}

#[cfg(all(test, feature = "phoenix"))]
mod tests {
    use super::*;

    #[test]
    fn test_types_imports() {
        // Verify all types can be imported
        let _ = OtelEndpoint::new("http://localhost:4317");
        let _ = SpanKind::Client;
        let _ = CompletionStatus::Success;
        let _ = AgentStopReason::Normal;
        let _ = ProviderKind::Anthropic;
        let _ = ToolExecutionResult::success("test");
        let _ = LlmFinishReason::Stop;
    }

    #[test]
    fn test_instrumentation_imports() {
        // Verify all instrumentation types can be imported
        let _ = CompletionSpan::new(
            &opentelemetry_sdk::trace::TracerProvider::default().tracer("test"),
            "test",
            CompletionAttributes {
                model: "test".to_string(),
                provider: "openai".to_string(),
                completion_status: CompletionStatus::Success,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                temperature: None,
            },
        );
    }
}
