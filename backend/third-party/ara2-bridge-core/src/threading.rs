//! ARA model-thread identity checks.

use crate::AraError;
use std::thread::ThreadId;

/// Identity of the thread that created an ARA document runtime.
#[derive(Clone, Copy, Debug)]
pub struct ModelThread {
    id: ThreadId,
}

impl ModelThread {
    /// Captures the current thread as the model thread.
    pub fn current() -> Self {
        Self {
            id: std::thread::current().id(),
        }
    }

    /// Returns whether the caller is currently on the captured model thread.
    pub fn is_current(self) -> bool {
        std::thread::current().id() == self.id
    }

    /// Rejects calls made from any other thread.
    pub fn require_current(self) -> Result<(), AraError> {
        if self.is_current() {
            Ok(())
        } else {
            Err(AraError::InvalidThread(
                "operation requires the ARA model thread",
            ))
        }
    }
}
