//! Shared poison state for panic and invariant containment.

use crate::Diagnostic;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Thread-safe poison state retaining the first causal diagnostic.
#[derive(Debug, Default)]
pub struct PoisonState {
    poisoned: AtomicBool,
    diagnostic: Mutex<Option<Diagnostic>>,
}

impl PoisonState {
    /// Marks the runtime poisoned and retains the first causal diagnostic.
    pub fn poison(&self, diagnostic: Diagnostic) {
        if !self.poisoned.swap(true, Ordering::AcqRel) {
            *self.diagnostic.lock() = Some(diagnostic);
        }
    }

    /// Returns whether an unrecoverable failure has poisoned the runtime.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Returns the first causal diagnostic, when poisoned.
    pub fn diagnostic(&self) -> Option<Diagnostic> {
        self.diagnostic.lock().clone()
    }
}
