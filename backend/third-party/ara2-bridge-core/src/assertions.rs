//! Process-coordinated ARA assertion function-pointer cells.

use crate::{ApiGeneration, AraError};
use ara2_bridge_sys::ARAAssertFunction;
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

struct AssertEntry {
    cell: Box<ARAAssertFunction>,
    active: usize,
}

#[derive(Default)]
struct AssertState {
    entries: BTreeMap<ApiGeneration, AssertEntry>,
}

fn state() -> &'static Mutex<AssertState> {
    static STATE: OnceLock<Mutex<AssertState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(AssertState::default()))
}

fn lock_state() -> MutexGuard<'static, AssertState> {
    state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Process-scoped coordinator for ARA assertion function-pointer addresses.
///
/// All coordinator values use the same process state. Concurrent factory initializations selecting
/// the same generation therefore receive the same cell address, while generation selection remains
/// stored in each [`FactoryInitialization`].
#[derive(Clone, Copy, Debug, Default)]
pub struct AssertCoordinator {
    _private: (),
}

impl AssertCoordinator {
    /// Returns the number of active factory initializations for a generation.
    pub fn active_count(&self, generation: ApiGeneration) -> usize {
        lock_state()
            .entries
            .get(&generation)
            .map_or(0, |entry| entry.active)
    }

    fn acquire(&self, generation: ApiGeneration) -> Result<*mut ARAAssertFunction, AraError> {
        if !generation.supported_on_target() {
            return Err(AraError::Unsupported(
                "API generation is unavailable on this target",
            ));
        }

        let mut state = lock_state();
        let entry = state
            .entries
            .entry(generation)
            .or_insert_with(|| AssertEntry {
                cell: Box::new(None),
                active: 0,
            });
        entry.active = entry.active.checked_add(1).ok_or(AraError::InvalidState(
            "factory initialization count overflow",
        ))?;
        Ok(std::ptr::from_mut(entry.cell.as_mut()))
    }

    fn release(&self, generation: ApiGeneration) -> Result<(), AraError> {
        let mut state = lock_state();
        let remove = {
            let entry = state
                .entries
                .get_mut(&generation)
                .ok_or(AraError::InvalidState("factory is not initialized"))?;
            entry.active = entry
                .active
                .checked_sub(1)
                .ok_or(AraError::InvalidState("factory is not initialized"))?;
            entry.active == 0
        };
        if remove {
            state.entries.remove(&generation);
        }
        Ok(())
    }
}

/// One balanced ARA factory initialization interval.
///
/// This guard is intentionally not cloneable. Dropping an active guard balances the initialization;
/// callers that need to observe teardown failures can call [`Self::uninitialize`] explicitly.
pub struct FactoryInitialization<'a> {
    generation: ApiGeneration,
    assert_address: *mut ARAAssertFunction,
    coordinator: &'a AssertCoordinator,
    active: bool,
}

impl<'a> FactoryInitialization<'a> {
    /// Begins an initialization interval for one factory entry.
    pub fn begin(
        generation: ApiGeneration,
        coordinator: &'a AssertCoordinator,
    ) -> Result<Self, AraError> {
        let assert_address = coordinator.acquire(generation)?;
        Ok(Self {
            generation,
            assert_address,
            coordinator,
            active: true,
        })
    }

    /// Returns the generation selected by this factory.
    pub const fn generation(&self) -> ApiGeneration {
        self.generation
    }

    /// Returns the stable pointer to the generation's process-shared assertion callback cell.
    pub const fn assert_address(&self) -> *mut ARAAssertFunction {
        self.assert_address
    }

    /// Ends this initialization interval and reports duplicate teardown.
    pub fn uninitialize(&mut self) -> Result<(), AraError> {
        if !self.active {
            return Err(AraError::InvalidState("factory is not initialized"));
        }
        self.coordinator.release(self.generation)?;
        self.active = false;
        Ok(())
    }

    /// Returns whether this interval still requires uninitialization.
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

impl Drop for FactoryInitialization<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.coordinator.release(self.generation);
            self.active = false;
        }
    }
}
