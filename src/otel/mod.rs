//! OpenTelemetry / Arize Phoenix tracing support for agent-driver-rs
//!
//! Implements the [OpenInference](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md)
//! semantic conventions so spans display correctly in Phoenix.
//!
//! ## Usage
//!
//! ```ignore
//! use agent_driver_rs::{SessionBuilder, otel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tracer = otel::init_phoenix()?;
//!     let session = SessionBuilder::new()
//!         .with_provider(my_provider)
//!         .otel_tracer(tracer)
//!         .build()
//!         .await?;
//!
//!     let outcome = agent_driver_rs::agent::AgentLoop::new(&session)
//!         .run("Search current weather")
//!         .await?;
//!
//!     otel::shutdown_phoenix();
//!     Ok(())
//! }
//! ```

pub mod instrumentation;

#[cfg(feature = "phoenix")]
use opentelemetry::trace::TracerProvider as _;
#[cfg(feature = "phoenix")]
use opentelemetry_sdk::trace::SdkTracerProvider;

#[cfg(feature = "phoenix")]
pub use instrumentation::{AgentLoopSpan, SessionOperationSpan, SpanKind, ToolSpan, attr};

#[cfg(feature = "phoenix")]
pub static PHOENIX_TRACER_PROVIDER: std::sync::OnceLock<std::sync::Arc<SdkTracerProvider>> =
    std::sync::OnceLock::new();

#[cfg(feature = "phoenix")]
pub fn init_tracer_provider(provider: std::sync::Arc<SdkTracerProvider>) {
    drop(PHOENIX_TRACER_PROVIDER.set(provider));
}

#[cfg(feature = "phoenix")]
pub fn get_tracer_provider() -> Option<std::sync::Arc<SdkTracerProvider>> {
    PHOENIX_TRACER_PROVIDER.get().cloned()
}

#[cfg(feature = "phoenix")]
pub fn get_tracer(name: &str) -> Result<opentelemetry_sdk::trace::Tracer, String> {
    let provider = get_tracer_provider()
        .ok_or_else(|| "Phoenix tracer provider not initialized".to_owned())?;
    Ok(provider.tracer(name.to_owned()))
}

/// Initialize Phoenix tracing with OTLP gRPC exporter.
///
/// Reads `PHOENIX_ENDPOINT` env var (defaults to `http://localhost:4317`).
/// Must be called from within a Tokio runtime.
#[cfg(feature = "phoenix")]
pub fn init_phoenix() -> Result<std::sync::Arc<opentelemetry_sdk::trace::Tracer>, String> {
    use opentelemetry_otlp::WithExportConfig as _;

    let endpoint =
        std::env::var("PHOENIX_ENDPOINT").unwrap_or_else(|_| "http://localhost:4317".to_owned());

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .map_err(|e| format!("Failed to build OTLP exporter: {e}"))?;

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();

    let tracer = provider.tracer("agent-driver-rs");
    let tracer_arc = std::sync::Arc::new(tracer);
    init_tracer_provider(std::sync::Arc::new(provider));
    Ok(tracer_arc)
}

/// Flush remaining spans and shut down the global tracer provider.
#[cfg(feature = "phoenix")]
pub fn shutdown_phoenix() {
    if let Some(provider) = get_tracer_provider() {
        let _shutdown = provider.shutdown();
    }
}
