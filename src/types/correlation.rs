//! Correlation tracking types for request tracing

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for correlating related operations
///
/// Used to track requests across async boundaries and link related events.
/// Only created via `generate()` - no arbitrary construction allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CorrelationId(Uuid);

impl CorrelationId {
    /// Generate a new unique correlation ID
    #[must_use]
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the underlying UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }

    /// Get the correlation ID as a string
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl std::fmt::Display for CorrelationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full = self.0.to_string();
        let short = full.get(..8).unwrap_or(&full);
        write!(f, "{short}")
    }
}

/// Context for correlating operations within a request scope
#[derive(Debug, Clone)]
pub struct CorrelationContext {
    /// Primary correlation ID for this context
    pub id: CorrelationId,
    /// Optional parent correlation ID for hierarchical tracing
    pub parent_id: Option<CorrelationId>,
    /// Optional span name for structured logging
    pub span_name: Option<String>,
}

impl CorrelationContext {
    /// Create a new correlation context
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: CorrelationId::generate(),
            parent_id: None,
            span_name: None,
        }
    }

    /// Create a child context with this context as parent
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            id: CorrelationId::generate(),
            parent_id: Some(self.id),
            span_name: None,
        }
    }

    /// Create a child context with a span name
    #[must_use]
    pub fn child_with_span(&self, span_name: impl Into<String>) -> Self {
        Self {
            id: CorrelationId::generate(),
            parent_id: Some(self.id),
            span_name: Some(span_name.into()),
        }
    }

    /// Set the span name
    #[must_use]
    pub fn with_span(mut self, span_name: impl Into<String>) -> Self {
        self.span_name = Some(span_name.into());
        self
    }
}

impl Default for CorrelationContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_id_unique() {
        let id1 = CorrelationId::generate();
        let id2 = CorrelationId::generate();
        assert_ne!(id1, id2);
    }

    #[test]
    fn correlation_id_display_short() {
        let id = CorrelationId::generate();
        let display = format!("{id}");
        assert_eq!(display.len(), 8);
    }

    #[test]
    fn correlation_context_child() {
        let parent = CorrelationContext::new();
        let child = parent.child();
        assert_eq!(child.parent_id, Some(parent.id));
        assert_ne!(child.id, parent.id);
    }
}
