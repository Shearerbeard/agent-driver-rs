//! OpenTelemetry instrumentation helpers
//!
//! Provides RAII guards for automatic span lifecycle management following
//! the project's coding principles: finite types, explicit resource management,
//! and proper error handling.

#[cfg(feature = "phoenix")]
use crate::otel::types::{
    AgentLoopAttributes, AgentStopReason, CompletionAttributes, CompletionStatus, LlmFinishReason,
    ProviderKind, SpanKind, SpanName,
};
#[cfg(feature = "phoenix")]
use crate::Session;
#[cfg(feature = "phoenix")]
use opentelemetry::trace::{Span as SpanTrait, Status, Tracer as TracerTrait};
#[cfg(feature = "phoenix")]
use opentelemetry::Context;
#[cfg(feature = "phoenix")]
use opentelemetry::Value as AttributeValue;
#[cfg(feature = "phoenix")]
use std::sync::Arc;

#[cfg(feature = "phoenix")]
/// Completion span RAII guard
///
