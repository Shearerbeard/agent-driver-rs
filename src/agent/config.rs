//! Agent loop configuration

use std::num::NonZeroU32;

use crate::error::AgentLoopError;

/// Maximum number of tool execution rounds before the loop stops.
///
/// Counts tool execution rounds, not total model responses. A final
/// text-only response doesn't count. This is the safety limit against
/// runaway tool calling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxToolDepth(NonZeroU32);

impl MaxToolDepth {
    /// Create a new MaxToolDepth with validation
    pub fn new(depth: u32) -> Result<Self, AgentLoopError> {
        NonZeroU32::new(depth)
            .map(Self)
            .ok_or_else(|| AgentLoopError::InvalidConfig("max_tool_depth must be > 0".into()))
    }

    /// Get the inner value
    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

impl Default for MaxToolDepth {
    fn default() -> Self {
        // SAFETY: 25 is non-zero
        Self(NonZeroU32::new(25).unwrap())
    }
}

/// Configuration for the agent loop
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// Maximum number of tool execution rounds (default: 25)
    pub max_tool_depth: MaxToolDepth,
    /// Whether to continue when a tool returns an error (default: true)
    ///
    /// When true, tool errors are sent back to the model as error results,
    /// allowing it to recover or try a different approach.
    /// When false, the loop stops immediately on tool error.
    pub continue_on_tool_error: bool,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_tool_depth: MaxToolDepth::default(),
            continue_on_tool_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_tool_depth_valid() {
        let depth = MaxToolDepth::new(10).unwrap();
        assert_eq!(depth.get(), 10);
    }

    #[test]
    fn max_tool_depth_zero() {
        assert!(MaxToolDepth::new(0).is_err());
    }

    #[test]
    fn max_tool_depth_default() {
        assert_eq!(MaxToolDepth::default().get(), 25);
    }

    #[test]
    fn config_default() {
        let config = AgentLoopConfig::default();
        assert_eq!(config.max_tool_depth.get(), 25);
        assert!(config.continue_on_tool_error);
    }
}
