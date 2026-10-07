//! Checked scoped states for ARA editing, restoration, access, rendering, and teardown.

use crate::{AraError, Diagnostic, ModelThread, PoisonState};
use parking_lot::Mutex;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RestoreMode {
    None,
    Ara1,
    Ara2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Active,
    TearingDown,
    Destroyed,
}

#[derive(Debug)]
struct State {
    phase: Phase,
    editing: bool,
    restore: RestoreMode,
    sample_accesses: usize,
    content_call: bool,
    render_activations: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            phase: Phase::Active,
            editing: false,
            restore: RestoreMode::None,
            sample_accesses: 0,
            content_call: false,
            render_activations: 0,
        }
    }
}

/// Shared checked lifecycle for one ARA document runtime.
#[derive(Debug)]
pub struct Lifecycle {
    model_thread: ModelThread,
    state: Mutex<State>,
    poison: PoisonState,
}

impl Lifecycle {
    /// Creates an active lifecycle and captures the current model thread.
    pub fn new_on_current_thread() -> Self {
        Self {
            model_thread: ModelThread::current(),
            state: Mutex::new(State::default()),
            poison: PoisonState::default(),
        }
    }

    /// Begins a normal model-editing interval.
    pub fn begin_editing(&self) -> Result<EditGuard<'_>, AraError> {
        self.require_model_operation()?;
        let mut state = self.state.lock();
        if state.editing {
            return Err(AraError::InvalidState("editing is already active"));
        }
        if state.restore != RestoreMode::None {
            return Err(AraError::InvalidState("restoration is active"));
        }
        state.editing = true;
        Ok(EditGuard {
            lifecycle: self,
            active: true,
        })
    }

    /// Begins a generation-1 restore interval without an edit interval.
    pub fn begin_ara1_restore(&self) -> Result<RestoreGuard<'_>, AraError> {
        self.require_model_operation()?;
        let mut state = self.state.lock();
        if state.editing || state.restore != RestoreMode::None {
            return Err(AraError::InvalidState(
                "editing or restoration is already active",
            ));
        }
        state.restore = RestoreMode::Ara1;
        Ok(RestoreGuard {
            lifecycle: self,
            mode: RestoreMode::Ara1,
            active: true,
        })
    }

    /// Begins an ARA2 restore interval together with its required edit interval.
    pub fn begin_ara2_restore(&self) -> Result<RestoreGuard<'_>, AraError> {
        self.require_model_operation()?;
        let mut state = self.state.lock();
        if state.editing || state.restore != RestoreMode::None {
            return Err(AraError::InvalidState(
                "editing or restoration is already active",
            ));
        }
        state.editing = true;
        state.restore = RestoreMode::Ara2;
        Ok(RestoreGuard {
            lifecycle: self,
            mode: RestoreMode::Ara2,
            active: true,
        })
    }

    /// Begins one sample-access enablement interval.
    pub fn begin_sample_access(&self) -> Result<SampleAccessGuard<'_>, AraError> {
        self.require_model_operation()?;
        let mut state = self.state.lock();
        state.sample_accesses = state
            .sample_accesses
            .checked_add(1)
            .ok_or(AraError::InvalidState("sample-access count overflow"))?;
        Ok(SampleAccessGuard {
            lifecycle: self,
            active: true,
        })
    }

    /// Begins the exclusive controller content-call interval.
    pub fn begin_content_call(&self) -> Result<ContentCallGuard<'_>, AraError> {
        self.require_model_operation()?;
        let mut state = self.state.lock();
        if state.content_call {
            return Err(AraError::InvalidState("content call is already active"));
        }
        state.content_call = true;
        Ok(ContentCallGuard {
            lifecycle: self,
            active: true,
        })
    }

    /// Begins a render activation. Activation itself is allowed off the model thread.
    pub fn begin_render_activation(&self) -> Result<RenderActivationGuard<'_>, AraError> {
        self.require_usable()?;
        let mut state = self.state.lock();
        state.render_activations = state
            .render_activations
            .checked_add(1)
            .ok_or(AraError::InvalidState("render activation count overflow"))?;
        Ok(RenderActivationGuard {
            lifecycle: self,
            active: true,
        })
    }

    /// Begins teardown after all scoped activity has ended.
    ///
    /// Teardown remains legal after poisoning so owned foreign resources can always be released.
    pub fn begin_teardown(&self) -> Result<TeardownGuard<'_>, AraError> {
        self.model_thread.require_current()?;
        let mut state = self.state.lock();
        if state.phase != Phase::Active {
            return Err(AraError::InvalidState("teardown is already active"));
        }
        if state.editing
            || state.restore != RestoreMode::None
            || state.sample_accesses != 0
            || state.content_call
            || state.render_activations != 0
        {
            return Err(AraError::InvalidState(
                "cannot teardown while scoped activity is active",
            ));
        }
        state.phase = Phase::TearingDown;
        Ok(TeardownGuard {
            lifecycle: self,
            active: true,
        })
    }

    /// Marks the runtime poisoned with a causal diagnostic.
    pub fn poison(&self, diagnostic: Diagnostic) {
        self.poison.poison(diagnostic);
    }

    /// Returns whether the runtime is poisoned.
    pub fn is_poisoned(&self) -> bool {
        self.poison.is_poisoned()
    }

    /// Returns the first poison diagnostic.
    pub fn poison_diagnostic(&self) -> Option<Diagnostic> {
        self.poison.diagnostic()
    }

    fn require_model_operation(&self) -> Result<(), AraError> {
        self.model_thread.require_current()?;
        self.require_usable()
    }

    fn require_usable(&self) -> Result<(), AraError> {
        if self.poison.is_poisoned() {
            return Err(AraError::Poisoned);
        }
        if self.state.lock().phase != Phase::Active {
            return Err(AraError::InvalidState("runtime is tearing down"));
        }
        Ok(())
    }

    fn finish_edit(&self) -> Result<(), AraError> {
        self.model_thread.require_current()?;
        let mut state = self.state.lock();
        if !state.editing || state.restore != RestoreMode::None {
            return Err(AraError::InvalidState("editing is not active"));
        }
        state.editing = false;
        Ok(())
    }

    fn finish_restore(&self, mode: RestoreMode) -> Result<(), AraError> {
        self.model_thread.require_current()?;
        let mut state = self.state.lock();
        if state.restore != mode {
            return Err(AraError::InvalidState("matching restoration is not active"));
        }
        state.restore = RestoreMode::None;
        if mode == RestoreMode::Ara2 {
            state.editing = false;
        }
        Ok(())
    }

    fn finish_sample_access(&self) -> Result<(), AraError> {
        self.model_thread.require_current()?;
        let mut state = self.state.lock();
        state.sample_accesses = state
            .sample_accesses
            .checked_sub(1)
            .ok_or(AraError::InvalidState("sample access is not active"))?;
        Ok(())
    }

    fn finish_content_call(&self) -> Result<(), AraError> {
        self.model_thread.require_current()?;
        let mut state = self.state.lock();
        if !state.content_call {
            return Err(AraError::InvalidState("content call is not active"));
        }
        state.content_call = false;
        Ok(())
    }

    fn finish_render_activation(&self) -> Result<(), AraError> {
        let mut state = self.state.lock();
        state.render_activations = state
            .render_activations
            .checked_sub(1)
            .ok_or(AraError::InvalidState("render activation is not active"))?;
        Ok(())
    }

    fn finish_teardown(&self) -> Result<(), AraError> {
        self.model_thread.require_current()?;
        let mut state = self.state.lock();
        if state.phase != Phase::TearingDown {
            return Err(AraError::InvalidState("teardown is not active"));
        }
        state.phase = Phase::Destroyed;
        Ok(())
    }
}

macro_rules! simple_guard {
    ($name:ident, $finish:ident, $docs:literal) => {
        #[doc = $docs]
        pub struct $name<'a> {
            lifecycle: &'a Lifecycle,
            active: bool,
        }

        impl $name<'_> {
            /// Explicitly balances the interval and reports lifecycle errors.
            pub fn finish(mut self) -> Result<(), AraError> {
                self.lifecycle.$finish()?;
                self.active = false;
                Ok(())
            }
        }

        impl Drop for $name<'_> {
            fn drop(&mut self) {
                if self.active {
                    let _ = self.lifecycle.$finish();
                    self.active = false;
                }
            }
        }
    };
}

simple_guard!(EditGuard, finish_edit, "A scoped model-editing interval.");
simple_guard!(
    SampleAccessGuard,
    finish_sample_access,
    "A scoped sample-access enablement interval."
);
simple_guard!(
    ContentCallGuard,
    finish_content_call,
    "An exclusive scoped content callback interval."
);
simple_guard!(
    RenderActivationGuard,
    finish_render_activation,
    "A scoped renderer activation interval."
);
simple_guard!(
    TeardownGuard,
    finish_teardown,
    "A scoped runtime teardown interval."
);

/// A scoped generation-specific restoration interval.
pub struct RestoreGuard<'a> {
    lifecycle: &'a Lifecycle,
    mode: RestoreMode,
    active: bool,
}

impl RestoreGuard<'_> {
    /// Explicitly balances restoration and reports lifecycle errors.
    pub fn finish(mut self) -> Result<(), AraError> {
        self.lifecycle.finish_restore(self.mode)?;
        self.active = false;
        Ok(())
    }
}

impl Drop for RestoreGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.lifecycle.finish_restore(self.mode);
            self.active = false;
        }
    }
}
