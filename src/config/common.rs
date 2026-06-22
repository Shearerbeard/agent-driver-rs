//! Common configuration types shared across providers

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::str::FromStr;

use crate::error::ConfigError;

/// Parse an optional environment variable into `T` using `FromStr`.
///
/// Returns `Ok(None)` when the variable is absent, and `Err` when the variable
/// is present but cannot be parsed. This prevents silent fallback to defaults
/// when a user provides an invalid value.
pub fn env_parse_opt<T>(var: &str) -> Result<Option<T>, ConfigError>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(var) {
        Ok(s) => s
            .parse::<T>()
            .map(Some)
            .map_err(|e| ConfigError::InvalidValue {
                field: var.to_owned(),
                reason: format!("{var}: {e}"),
            }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ConfigError::from(e)),
    }
}

/// Parse an optional environment variable into `T`, falling back to the default.
///
/// Use this when the variable is optional and the type has a sensible default.
pub fn env_parse_or_default<T>(var: &str) -> Result<T, ConfigError>
where
    T: FromStr + Default,
    T::Err: std::fmt::Display,
{
    env_parse_opt(var).map(Option::unwrap_or_default)
}

/// API key newtype (redacted in Debug output)
#[derive(Clone, Serialize)]
#[serde(transparent)]
pub struct ApiKey(String);

impl ApiKey {
    /// Create a new API key.
    ///
    /// Rejects empty strings so that missing keys are surfaced as config errors
    /// rather than runtime auth failures.
    pub fn new(key: impl Into<String>) -> Result<Self, ConfigError> {
        let key = key.into();
        if key.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "api_key".to_owned(),
                reason: "API key cannot be empty".into(),
            });
        }
        Ok(Self(key))
    }

    /// Get the API key as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ApiKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ApiKey([REDACTED])")
    }
}

/// AWS region newtype
///
/// Uses Cow for static constants while allowing runtime values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct AwsRegion(Cow<'static, str>);

impl AwsRegion {
    /// US East (N. Virginia).
    pub const US_EAST_1: Self = Self(Cow::Borrowed("us-east-1"));
    /// US West (Oregon).
    pub const US_WEST_2: Self = Self(Cow::Borrowed("us-west-2"));
    /// EU West (Ireland).
    pub const EU_WEST_1: Self = Self(Cow::Borrowed("eu-west-1"));
    /// EU Central (Frankfurt).
    pub const EU_CENTRAL_1: Self = Self(Cow::Borrowed("eu-central-1"));
    /// Asia Pacific (Tokyo).
    pub const AP_NORTHEAST_1: Self = Self(Cow::Borrowed("ap-northeast-1"));

    /// Create a new region from a runtime string.
    ///
    /// Rejects empty strings so that an unset `AWS_REGION` does not silently
    /// propagate as an empty region value.
    pub fn new(region: impl Into<String>) -> Result<Self, ConfigError> {
        let region = region.into();
        if region.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "aws_region".to_owned(),
                reason: "AWS region cannot be empty".into(),
            });
        }
        Ok(Self(Cow::Owned(region)))
    }

    /// Get the region as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AwsRegion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for AwsRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for AwsRegion {
    fn default() -> Self {
        Self::US_EAST_1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_redacted_in_debug() {
        let key = ApiKey::new("sk-secret-key").unwrap();
        let debug = format!("{key:?}");
        assert!(!debug.contains("sk-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn api_key_rejects_empty() {
        ApiKey::new("").unwrap_err();
    }

    #[test]
    fn aws_region_static_constants() {
        assert_eq!(AwsRegion::US_EAST_1.as_str(), "us-east-1");
        assert_eq!(AwsRegion::US_WEST_2.as_str(), "us-west-2");
    }

    #[test]
    fn aws_region_runtime() {
        let region = AwsRegion::new("eu-west-2").unwrap();
        assert_eq!(region.as_str(), "eu-west-2");
    }

    #[test]
    fn aws_region_rejects_empty() {
        AwsRegion::new("").unwrap_err();
    }
}
