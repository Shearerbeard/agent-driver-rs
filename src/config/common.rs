//! Common configuration types shared across providers

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// API key newtype (redacted in Debug output)
#[derive(Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct ApiKey(String);

impl ApiKey {
    /// Create a new API key
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Get the API key as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
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

    /// Create a new region from a runtime string
    pub fn new(region: impl Into<String>) -> Self {
        Self(Cow::Owned(region.into()))
    }

    /// Get the region as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
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
        let key = ApiKey::new("sk-secret-key");
        let debug = format!("{:?}", key);
        assert!(!debug.contains("sk-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn aws_region_static_constants() {
        assert_eq!(AwsRegion::US_EAST_1.as_str(), "us-east-1");
        assert_eq!(AwsRegion::US_WEST_2.as_str(), "us-west-2");
    }

    #[test]
    fn aws_region_runtime() {
        let region = AwsRegion::new("eu-west-2");
        assert_eq!(region.as_str(), "eu-west-2");
    }
}
