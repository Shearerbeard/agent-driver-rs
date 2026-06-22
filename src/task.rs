//! Task tracking system with cancellation support
//!
//! This module provides:
//! - `TaskPool` for tracked task spawning
//! - `TaskHandle` wrapper with correlation tracking
//! - `spawn_tracked` free function for convenient spawning

mod handle;
mod pool;
mod spawn;

pub use handle::TaskHandle;
pub use pool::TaskPool;
pub use spawn::spawn_tracked;
