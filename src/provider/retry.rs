//! Retry logic with exponential backoff for rate limiting.
//!
//! Provides [`with_retry`] for wrapping provider calls with automatic retries on
//! [`ProviderError::RateLimited`](crate::error::ProviderError::RateLimited).
//! Non-rate-limit errors are returned immediately without retrying.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::error::ProviderError;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff interval
    pub initial_interval: Duration,
    /// Maximum backoff interval
    pub max_interval: Duration,
    /// Backoff multiplier
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_interval: Duration::from_millis(500),
            max_interval: Duration::from_secs(30),
            multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of retries
    #[must_use]
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Set the initial backoff interval
    #[must_use]
    pub fn initial_interval(mut self, d: Duration) -> Self {
        self.initial_interval = d;
        self
    }

    /// Set the maximum backoff interval
    #[must_use]
    pub fn max_interval(mut self, d: Duration) -> Self {
        self.max_interval = d;
        self
    }

    /// Set the backoff multiplier
    #[must_use]
    pub fn multiplier(mut self, m: f64) -> Self {
        self.multiplier = m;
        self
    }

    /// Compute the backoff delay for a given attempt (0-indexed)
    fn backoff_delay(&self, attempt: u32) -> Duration {
        let mut delay = self.initial_interval;
        for _ in 0..attempt {
            delay = delay.mul_f64(self.multiplier);
            if delay > self.max_interval {
                return self.max_interval;
            }
        }
        delay
    }
}

/// Execute an async operation with retry logic.
///
/// Only retries on rate limit errors. Other errors are returned immediately.
/// Retry sleeps are cancellation-aware: if the token is cancelled during the
/// backoff sleep, the function returns `ProviderError::Cancelled`.
pub async fn with_retry<F, Fut, T>(
    config: &RetryConfig,
    cancellation: CancellationToken,
    mut f: F,
) -> Result<T, ProviderError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, ProviderError>>,
{
    let mut attempts = 0;

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e @ ProviderError::RateLimited { .. }) => {
                attempts += 1;
                if attempts >= config.max_retries {
                    return Err(e);
                }

                // Use retry_after if provided, otherwise use backoff
                let delay = e
                    .retry_after()
                    .unwrap_or_else(|| config.backoff_delay(attempts - 1));

                tracing::debug!(
                    attempts = attempts,
                    delay_ms = delay.as_millis(),
                    "Rate limited, retrying after delay"
                );

                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => return Err(ProviderError::Cancelled),
                    _ = tokio::time::sleep(delay) => {}
                }
            }
            Err(e) => return Err(e), // Don't retry other errors
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn success_on_first_try() {
        let config = RetryConfig::default();
        let result = with_retry(&config, CancellationToken::new(), || async {
            Ok::<_, ProviderError>(42)
        })
        .await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn success_after_retry() {
        let config = RetryConfig::new()
            .max_retries(3)
            .initial_interval(Duration::from_millis(10));

        let attempts = Arc::new(AtomicU32::new(0));

        let result = with_retry(&config, CancellationToken::new(), || {
            let attempts = Arc::clone(&attempts);
            async move {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(ProviderError::RateLimited {
                        provider: crate::provider::ProviderKind::Anthropic,
                        retry_after: None,
                    })
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn fail_after_max_retries() {
        let config = RetryConfig::new()
            .max_retries(2)
            .initial_interval(Duration::from_millis(10));

        let attempts = Arc::new(AtomicU32::new(0));

        let result = with_retry(&config, CancellationToken::new(), || {
            let attempts = Arc::clone(&attempts);
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(ProviderError::RateLimited {
                    provider: crate::provider::ProviderKind::Anthropic,
                    retry_after: None,
                })
            }
        })
        .await;

        assert!(matches!(result, Err(ProviderError::RateLimited { .. })));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn no_retry_on_other_errors() {
        let config = RetryConfig::default();
        let attempts = Arc::new(AtomicU32::new(0));

        let result = with_retry(&config, CancellationToken::new(), || {
            let attempts = Arc::clone(&attempts);
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(ProviderError::Auth {
                    provider: crate::provider::ProviderKind::Anthropic,
                    kind: crate::error::AuthErrorKind::Rejected,
                    message: "Invalid API key".into(),
                })
            }
        })
        .await;

        assert!(matches!(result, Err(ProviderError::Auth { .. })));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    /// If the cancellation token fires during the retry backoff sleep,
    /// with_retry should return ProviderError::Cancelled promptly.
    #[tokio::test]
    async fn cancellation_aborts_retry_sleep() {
        let config = RetryConfig::new()
            .max_retries(3)
            .initial_interval(Duration::from_secs(60));

        let cancellation = CancellationToken::new();

        let start = tokio::time::Instant::now();
        let retry_handle = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                with_retry(&config, cancellation, || async {
                    Err::<i32, _>(ProviderError::RateLimited {
                        provider: crate::provider::ProviderKind::Anthropic,
                        retry_after: None,
                    })
                })
                .await
            }
        });

        // Cancel shortly after the first attempt has entered the backoff sleep.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancellation.cancel();

        let result = retry_handle.await.unwrap();

        assert!(matches!(result, Err(ProviderError::Cancelled)));
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "cancellation should abort the 60s sleep"
        );
    }
}
