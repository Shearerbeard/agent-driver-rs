//! Model-related newtypes with validated construction.
//!
//! All types in this module use the newtype pattern with protected constructors
//! to enforce domain invariants at compile time:
//! - [`ModelId`] -- non-empty, restricted character set
//! - [`MaxTokens`] -- guaranteed positive via `NonZeroU32`
//! - [`Temperature`] -- clamped to \[0.0, 2.0\] to cover all provider ranges

use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

use crate::error::{ModelIdError, TemperatureError};

/// Validated model identifier
///
/// Model IDs must be non-empty and contain only alphanumeric characters,
/// hyphens, underscores, forward slashes, colons, and dots.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    /// Create a new ModelId with validation
    #[must_use = "this returns a Result that should be checked"]
    pub fn new(id: impl Into<String>) -> Result<Self, ModelIdError> {
        let id = id.into();
        if id.is_empty() {
            return Err(ModelIdError::Empty);
        }
        if !id
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '/' | ':' | '.'))
        {
            return Err(ModelIdError::InvalidCharacters);
        }
        Ok(Self(id))
    }

    /// Get the model ID as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ModelId {
    type Error = ModelIdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl TryFrom<&str> for ModelId {
    type Error = ModelIdError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Maximum tokens for completion response
///
/// Wraps NonZeroU32 to ensure the value is always positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MaxTokens(NonZeroU32);

impl MaxTokens {
    /// Create a new MaxTokens value
    ///
    /// Returns None if tokens is 0.
    #[must_use]
    pub fn new(tokens: u32) -> Option<Self> {
        NonZeroU32::new(tokens).map(Self)
    }

    /// Get the token count
    #[must_use]
    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

impl std::fmt::Display for MaxTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for MaxTokens {
    fn default() -> Self {
        let Some(tokens) = NonZeroU32::new(4096) else {
            unreachable!("4096 is non-zero")
        };
        Self(tokens)
    }
}

/// Sampling temperature
///
/// Valid range is [0.0, 2.0] to accommodate both Anthropic (0-1) and OpenAI (0-2).
/// Provider implementations should validate against their specific ranges.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Temperature(f32);

impl Temperature {
    /// Create a new Temperature with validation
    ///
    /// Returns error if value is outside [0.0, 2.0].
    #[must_use = "this returns a Result that should be checked"]
    pub fn new(temp: f32) -> Result<Self, TemperatureError> {
        if !(0.0..=2.0).contains(&temp) {
            return Err(TemperatureError::OutOfRange(temp));
        }
        Ok(Self(temp))
    }

    /// Get the temperature value
    #[must_use]
    pub fn get(&self) -> f32 {
        self.0
    }
}

impl std::fmt::Display for Temperature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'de> Deserialize<'de> for Temperature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let f = f32::deserialize(deserializer)?;
        Self::new(f).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_valid() {
        ModelId::new("claude-sonnet-4-20250514").unwrap();
        ModelId::new("gpt-4o").unwrap();
        ModelId::new("anthropic/claude-sonnet-4").unwrap();
        ModelId::new("llama3.2:3b").unwrap();
    }

    #[test]
    fn model_id_invalid() {
        assert!(matches!(ModelId::new(""), Err(ModelIdError::Empty)));
        assert!(matches!(
            ModelId::new("model with spaces"),
            Err(ModelIdError::InvalidCharacters)
        ));
    }

    #[test]
    fn max_tokens_valid() {
        assert!(MaxTokens::new(4096).is_some());
        assert!(MaxTokens::new(1).is_some());
        assert!(MaxTokens::new(0).is_none());
    }

    #[test]
    fn temperature_valid() {
        Temperature::new(0.0).unwrap();
        Temperature::new(1.0).unwrap();
        Temperature::new(2.0).unwrap();
        Temperature::new(-0.1).unwrap_err();
        Temperature::new(2.1).unwrap_err();
    }
}
