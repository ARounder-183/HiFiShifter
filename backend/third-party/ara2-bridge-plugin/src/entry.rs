//! Per-factory ARA initialization state.

use crate::factory::ControllerCreator;
use ara2_bridge_core::{ApiGeneration, AraError};
use ara2_bridge_sys::{
    ARAAssertFunction, ARADocumentControllerHostInstance, ARADocumentControllerInstance,
    ARADocumentProperties, ARAFactory,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

#[derive(Clone, Copy, Debug)]
struct Selection {
    generation: ApiGeneration,
    assert_address: usize,
}

#[derive(Clone, Copy, Debug)]
struct SharedAssertion {
    address: usize,
    users: usize,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn assertions() -> &'static Mutex<BTreeMap<ApiGeneration, SharedAssertion>> {
    static ASSERTIONS: OnceLock<Mutex<BTreeMap<ApiGeneration, SharedAssertion>>> = OnceLock::new();
    ASSERTIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn acquire_assertion(generation: ApiGeneration, address: usize) -> Result<(), AraError> {
    let mut assertions = lock(assertions());
    match assertions.get_mut(&generation) {
        Some(shared) if shared.address != address => Err(AraError::InvalidState(
            "factories using one API generation must share the assert-function address",
        )),
        Some(shared) => {
            shared.users = shared.users.checked_add(1).ok_or(AraError::InvalidState(
                "assert-function user count overflow",
            ))?;
            Ok(())
        }
        None => {
            assertions.insert(generation, SharedAssertion { address, users: 1 });
            Ok(())
        }
    }
}

fn release_assertion(selection: Selection) {
    let mut assertions = lock(assertions());
    let remove = match assertions.get_mut(&selection.generation) {
        Some(shared) if shared.address == selection.assert_address && shared.users > 1 => {
            shared.users -= 1;
            false
        }
        Some(shared) if shared.address == selection.assert_address => true,
        _ => false,
    };
    if remove {
        assertions.remove(&selection.generation);
    }
}

/// Shared state reached by both the safe entry API and its C callback trampolines.
pub(crate) struct CallbackState {
    lowest: ApiGeneration,
    highest: ApiGeneration,
    selection: Mutex<Option<Selection>>,
    create_controller: Option<Arc<ControllerCreator>>,
    factory: AtomicPtr<ARAFactory>,
}

impl CallbackState {
    pub(crate) fn new(
        lowest: ApiGeneration,
        highest: ApiGeneration,
        create_controller: Option<Arc<ControllerCreator>>,
    ) -> Self {
        Self {
            lowest,
            highest,
            selection: Mutex::new(None),
            create_controller,
            factory: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub(crate) fn attach_factory(&self, factory: *const ARAFactory) {
        self.factory.store(factory.cast_mut(), Ordering::Release);
    }

    pub(crate) fn create_document_controller(
        &self,
        host: *const ARADocumentControllerHostInstance,
        properties: *const ARADocumentProperties,
    ) -> Result<*const ARADocumentControllerInstance, AraError> {
        let generation = self
            .generation()
            .ok_or(AraError::InvalidState("factory is not initialized"))?;
        let create = self
            .create_controller
            .as_ref()
            .ok_or(AraError::Unsupported(
                "factory has no document-controller constructor",
            ))?;
        let factory = self.factory.load(Ordering::Acquire).cast_const();
        if factory.is_null() {
            return Err(AraError::InvalidState(
                "factory ABI storage is not attached",
            ));
        }
        create(generation, factory, host, properties)
    }

    pub(crate) fn initialize(
        &self,
        generation: ApiGeneration,
        assert_address: *mut ARAAssertFunction,
    ) -> Result<(), AraError> {
        if assert_address.is_null() {
            return Err(AraError::InvalidArgument(
                "assert-function address must not be null",
            ));
        }
        if generation < self.lowest || generation > self.highest {
            return Err(AraError::InvalidArgument(
                "API generation is outside the factory range",
            ));
        }
        if !generation.supported_on_target() {
            return Err(AraError::Unsupported(
                "API generation is unavailable on this target",
            ));
        }

        let mut selection = lock(&self.selection);
        if selection.is_some() {
            return Err(AraError::InvalidState("factory is already initialized"));
        }
        let selected = Selection {
            generation,
            assert_address: assert_address as usize,
        };
        acquire_assertion(generation, selected.assert_address)?;
        *selection = Some(selected);
        Ok(())
    }

    pub(crate) fn uninitialize(&self) -> Result<(), AraError> {
        let mut selection = lock(&self.selection);
        let selected = selection
            .take()
            .ok_or(AraError::InvalidState("factory is not initialized"))?;
        release_assertion(selected);
        Ok(())
    }

    pub(crate) fn generation(&self) -> Option<ApiGeneration> {
        lock(&self.selection).map(|selection| selection.generation)
    }
}

impl Drop for CallbackState {
    fn drop(&mut self) {
        if let Some(selection) = *self
            .selection
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            release_assertion(selection);
        }
    }
}

/// Independent initialization handle for one exported ARA factory.
pub struct PluginEntry {
    pub(crate) state: std::sync::Arc<CallbackState>,
}

impl PluginEntry {
    pub(crate) fn new(state: std::sync::Arc<CallbackState>) -> Self {
        Self { state }
    }

    /// Initializes this factory for the selected released API generation.
    pub fn initialize(
        &self,
        generation: ApiGeneration,
        assert_address: *mut ARAAssertFunction,
    ) -> Result<(), AraError> {
        self.state.initialize(generation, assert_address)
    }

    /// Balances a successful initialization of this factory.
    pub fn uninitialize(&self) -> Result<(), AraError> {
        self.state.uninitialize()
    }

    /// Returns the generation currently selected for this factory.
    pub fn generation(&self) -> Option<ApiGeneration> {
        self.state.generation()
    }
}
