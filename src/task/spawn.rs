//! TrackedSpawn trait for correlation-based spawning

use std::future::Future;
use std::sync::Arc;

use crate::error::TaskPoolError;
use crate::types::CorrelationId;

use super::{TaskHandle, TaskPool};

/// Extension trait for tracked spawning with proper bounds
pub trait TrackedSpawn {
    /// Spawn a tracked task using this correlation ID
    fn spawn_tracked<F, T>(
        &self,
        pool: &Arc<TaskPool>,
        name: &'static str,
        future: F,
    ) -> Result<TaskHandle<T>, TaskPoolError>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static;
}

impl TrackedSpawn for CorrelationId {
    fn spawn_tracked<F, T>(
        &self,
        pool: &Arc<TaskPool>,
        name: &'static str,
        future: F,
    ) -> Result<TaskHandle<T>, TaskPoolError>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        pool.spawn(*self, name, future)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tracked_spawn_via_trait() {
        let pool = TaskPool::new();
        let cid = CorrelationId::generate();

        let handle = cid.spawn_tracked(&pool, "test", async { 42 }).unwrap();
        assert_eq!(handle.correlation_id(), cid);

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }
}
