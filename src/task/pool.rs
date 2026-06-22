//! Task pool for tracked spawning with cascading cancellation

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::error::TaskPoolError;
use crate::types::CorrelationId;

use super::handle::TaskHandle;

/// Internal task registration info
struct RegisteredTask {
    cancellation: CancellationToken,
    #[allow(
        dead_code,
        reason = "retained for diagnostic logging of registered tasks"
    )]
    name: &'static str,
}

/// Pool for tracked task spawning
///
/// All tasks should be spawned through this pool to ensure:
/// - Proper correlation tracking
/// - Cascading cancellation via the root token
/// - Graceful shutdown that waits for all tasks
///
/// Registration happens BEFORE task execution to prevent race conditions.
///
/// # Example
///
/// ```no_run
/// use agent_driver_rs::task::TaskPool;
/// use agent_driver_rs::CorrelationId;
///
/// # async fn example() {
/// let pool = TaskPool::new();
/// let handle = pool.spawn(CorrelationId::generate(), "my_task", async { 42 }).unwrap();
/// let result = handle.await.unwrap();
/// pool.shutdown().await;
/// # }
/// ```
pub struct TaskPool {
    // parking_lot::RwLock is fine here - lock held only for HashMap ops, never across await
    tasks: RwLock<HashMap<CorrelationId, RegisteredTask>>,
    root_token: CancellationToken,
    tracker: TaskTracker,
    /// Gate flag: true while the pool accepts new tasks.
    ///
    /// `TaskTracker` tracks in-flight tasks but has no "closed to new spawns"
    /// state, so a separate flag is required to reject spawns after
    /// `shutdown()` has been called. The flag is checked with `Acquire` and
    /// cleared with `Release` ordering to synchronize with `spawn`.
    accepting: AtomicBool,
}

impl TaskPool {
    /// Create a new task pool
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            tasks: RwLock::new(HashMap::new()),
            root_token: CancellationToken::new(),
            tracker: TaskTracker::new(),
            accepting: AtomicBool::new(true),
        })
    }

    /// Spawn a tracked task
    ///
    /// Registration happens BEFORE task starts executing to prevent race conditions.
    /// Uses a oneshot channel to gate task start until after registration.
    ///
    /// # TOCTOU safety
    ///
    /// There is an inherent race between the `accepting` check and the task
    /// insertion: `shutdown()` can flip `accepting` to `false` between our
    /// load and our insert.  To close this gap, we re-check `accepting` after
    /// insertion. If shutdown has started in the meantime, we eagerly cancel
    /// the task's token (so the future observes cancellation immediately) and
    /// remove it from the map. The task is still tracked by `TaskTracker` and
    /// will be awaited during `shutdown()`, but it will see cancellation and
    /// exit promptly.
    pub fn spawn<F, T>(
        self: &Arc<Self>,
        correlation_id: CorrelationId,
        name: &'static str,
        future: F,
    ) -> Result<TaskHandle<T>, TaskPoolError>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        // Use Acquire ordering to synchronize with shutdown's Release store
        if !self.accepting.load(Ordering::Acquire) {
            return Err(TaskPoolError::Shutdown);
        }

        let task_token = self.root_token.child_token();
        let pool = Arc::clone(self);
        let cid = correlation_id;

        // Gate to ensure registration completes before task executes
        let (start_tx, start_rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn via TaskTracker - task waits for gate before running
        let handle = self.tracker.spawn(async move {
            // Wait for registration to complete (ignore error if sender dropped)
            drop(start_rx.await);
            let result = future.await;
            // Cleanup on completion
            pool.tasks.write().remove(&cid);
            result
        });

        // Register BEFORE task can execute
        self.tasks.write().insert(
            correlation_id,
            RegisteredTask {
                cancellation: task_token.clone(),
                name,
            },
        );

        // TOCTOU guard: if shutdown() raced between our initial check and now,
        // the task snuck in. Cancel it eagerly and remove from the map so
        // shutdown proceeds cleanly.
        if !self.accepting.load(Ordering::Acquire) {
            task_token.cancel();
            self.tasks.write().remove(&correlation_id);
        }

        // Release the gate - task can now execute
        let _sent = start_tx.send(());

        Ok(TaskHandle {
            inner: handle,
            correlation_id,
            cancellation: task_token,
            name,
        })
    }

    /// Cancel a specific task by correlation ID
    ///
    /// Returns true if the task was found and cancelled.
    pub fn cancel(&self, id: CorrelationId) -> bool {
        if let Some(task) = self.tasks.read().get(&id) {
            task.cancellation.cancel();
            true
        } else {
            false
        }
    }

    /// Cancel all tasks
    pub fn cancel_all(&self) {
        self.root_token.cancel();
    }

    /// Shut down the pool gracefully
    ///
    /// This will:
    /// 1. Stop accepting new tasks
    /// 2. Cancel all existing tasks
    /// 3. Wait for all tasks to complete
    pub async fn shutdown(&self) {
        // Use Release ordering to synchronize with spawn's Acquire load
        self.accepting.store(false, Ordering::Release);
        self.root_token.cancel();
        self.tracker.close();
        self.tracker.wait().await;
    }

    /// Get the number of active tasks
    pub fn task_count(&self) -> usize {
        self.tasks.read().len()
    }

    /// Check if the pool is accepting new tasks
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire)
    }

    /// Get a child token of the root cancellation token
    pub fn child_token(&self) -> CancellationToken {
        self.root_token.child_token()
    }

    /// Get a clone of the task tracker
    pub fn tracker(&self) -> TaskTracker {
        self.tracker.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_and_await() {
        let pool = TaskPool::new();
        let handle = pool
            .spawn(CorrelationId::generate(), "test", async { 42 })
            .unwrap();
        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn spawn_and_cancel() {
        let pool = TaskPool::new();
        let cid = CorrelationId::generate();

        // Get a child token to pass into the task
        let token = pool.child_token();
        let handle = pool
            .spawn(cid, "test", async move {
                // Check the root token (via child) for cancellation
                loop {
                    if token.is_cancelled() {
                        return "cancelled";
                    }
                    tokio::task::yield_now().await;
                }
            })
            .unwrap();

        // Cancel all tasks (cancels the root token)
        pool.cancel_all();

        let result = handle.await.unwrap();
        assert_eq!(result, "cancelled");
    }

    #[tokio::test]
    async fn shutdown_cancels_all() {
        let pool = TaskPool::new();

        let token = pool.child_token();
        let _handle = pool
            .spawn(CorrelationId::generate(), "test", async move {
                loop {
                    if token.is_cancelled() {
                        return;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .unwrap();

        pool.shutdown().await;
        assert!(!pool.is_accepting());
    }

    #[tokio::test]
    async fn spawn_after_shutdown_fails() {
        let pool = TaskPool::new();
        pool.shutdown().await;

        let result = pool.spawn(CorrelationId::generate(), "test", async { 42 });
        assert!(matches!(result, Err(TaskPoolError::Shutdown)));
    }
}
