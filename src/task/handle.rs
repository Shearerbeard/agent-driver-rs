//! Task handle wrapper with correlation tracking

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::types::CorrelationId;

/// Wrapper around JoinHandle with correlation tracking and cancellation
#[derive(Debug)]
pub struct TaskHandle<T> {
    pub(crate) inner: JoinHandle<T>,
    pub(crate) correlation_id: CorrelationId,
    pub(crate) cancellation: CancellationToken,
    pub(crate) name: &'static str,
}

impl<T> TaskHandle<T> {
    /// Get the correlation ID for this task
    pub fn correlation_id(&self) -> CorrelationId {
        self.correlation_id
    }

    /// Get the task name
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Request cancellation of this task
    ///
    /// The task must check the cancellation token to respond to this.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Abort the task immediately
    ///
    /// This is more forceful than `cancel()` - the task will be aborted
    /// even if it doesn't check the cancellation token.
    pub fn abort(&self) {
        self.inner.abort();
    }

    /// Check if the task has finished
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Get a reference to the cancellation token
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
}

// Note: JoinHandle<T> is Unpin regardless of T, so this projection is safe
impl<T> Future for TaskHandle<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safe: JoinHandle implements Unpin
        Pin::new(&mut self.inner).poll(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::task::TaskTracker;

    #[tokio::test]
    async fn task_handle_await() {
        let tracker = TaskTracker::new();
        let cancellation = CancellationToken::new();
        let correlation_id = CorrelationId::generate();

        let handle = TaskHandle {
            inner: tracker.spawn(async { 42 }),
            correlation_id,
            cancellation,
            name: "test",
        };

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn task_handle_cancel() {
        let tracker = TaskTracker::new();
        let cancellation = CancellationToken::new();
        let correlation_id = CorrelationId::generate();

        let token = cancellation.clone();
        let handle = TaskHandle {
            inner: tracker.spawn(async move {
                loop {
                    if token.is_cancelled() {
                        return "cancelled";
                    }
                    tokio::task::yield_now().await;
                }
            }),
            correlation_id,
            cancellation: cancellation.clone(),
            name: "test",
        };

        assert!(!handle.is_cancelled());
        handle.cancel();
        assert!(handle.is_cancelled());

        let result = handle.await.unwrap();
        assert_eq!(result, "cancelled");
    }
}
