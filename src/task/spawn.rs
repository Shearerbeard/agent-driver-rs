//! Tracked spawning helper for correlation-based task spawning

use std::future::Future;
use std::sync::Arc;

use crate::error::TaskPoolError;
use crate::types::CorrelationId;

use super::{TaskHandle, TaskPool};

/// Spawn a tracked task using the given correlation ID.
///
/// This is a thin wrapper around [`TaskPool::spawn`] that keeps the caller
/// from needing to import both `CorrelationId` and `TaskPool` methods.
pub fn spawn_tracked<F, T>(
    pool: &Arc<TaskPool>,
    correlation_id: CorrelationId,
    name: &'static str,
    future: F,
) -> Result<TaskHandle<T>, TaskPoolError>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    pool.spawn(correlation_id, name, future)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tracked_spawn_via_free_function() {
        let pool = TaskPool::new();
        let cid = CorrelationId::generate();

        let handle = spawn_tracked(&pool, cid, "test", async { 42 }).unwrap();
        assert_eq!(handle.correlation_id(), cid);

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }
}
