//! Task tracking system with cancellation support
//!
//! This module provides:
//! - `TaskPool` for tracked task spawning
//! - `TaskHandle` wrapper with correlation tracking
//! - `TrackedSpawn` trait for convenient spawning

mod handle;
mod pool;
mod spawn;

pub use handle::TaskHandle;
pub use pool::TaskPool;
pub use spawn::TrackedSpawn;
