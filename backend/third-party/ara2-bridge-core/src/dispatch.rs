//! Panic-contained adapters from safe Rust operations to ARA ABI sentinels.

use crate::{AraBool, AraError, Diagnostic, DocumentId, InstanceId};
use ara2_bridge_sys::ARABool;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;

/// Runtime services required by common callback dispatch adapters.
pub trait DispatchRuntime {
    /// Returns whether a previous unrecoverable failure poisoned the runtime.
    fn is_poisoned(&self) -> bool;
    /// Poisons the runtime while retaining the causal diagnostic.
    fn poison(&self, diagnostic: Diagnostic);
    /// Records a diagnostic for deferred reporting.
    fn record_diagnostic(&self, diagnostic: Diagnostic);
    /// Returns the affected document identity when available.
    fn document_id(&self) -> Option<DocumentId> {
        None
    }
    /// Returns the affected runtime instance identity when available.
    fn instance_id(&self) -> Option<InstanceId> {
        None
    }
}

fn contextualize(
    runtime: &impl DispatchRuntime,
    interface: &'static str,
    method: &'static str,
    error: AraError,
) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(error).at(interface, method);
    if let Some(document) = runtime.document_id() {
        diagnostic = diagnostic.with_document(document);
    }
    if let Some(instance) = runtime.instance_id() {
        diagnostic = diagnostic.with_instance(instance);
    }
    diagnostic
}

fn dispatch<T>(
    runtime: &impl DispatchRuntime,
    interface: &'static str,
    method: &'static str,
    operation: impl FnOnce() -> Result<T, AraError>,
) -> Option<T> {
    if runtime.is_poisoned() {
        return None;
    }
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(error)) => {
            runtime.record_diagnostic(contextualize(runtime, interface, method, error));
            None
        }
        Err(_) => {
            let diagnostic = contextualize(runtime, interface, method, AraError::Poisoned)
                .with_message("panic contained at ARA callback boundary");
            runtime.record_diagnostic(diagnostic.clone());
            runtime.poison(diagnostic);
            None
        }
    }
}

/// Dispatches a void callback, recording errors and containing panics.
pub fn dispatch_void(
    runtime: &impl DispatchRuntime,
    interface: &'static str,
    method: &'static str,
    operation: impl FnOnce() -> Result<(), AraError>,
) {
    let _ = dispatch(runtime, interface, method, operation);
}

/// Dispatches an ARA boolean callback with canonical false failure sentinel.
pub fn dispatch_bool(
    runtime: &impl DispatchRuntime,
    interface: &'static str,
    method: &'static str,
    operation: impl FnOnce() -> Result<bool, AraError>,
) -> ARABool {
    AraBool::new(dispatch(runtime, interface, method, operation).unwrap_or(false)).into_raw()
}

/// Dispatches a nullable-reference callback with null failure sentinel.
pub fn dispatch_ref<T>(
    runtime: &impl DispatchRuntime,
    interface: &'static str,
    method: &'static str,
    operation: impl FnOnce() -> Result<Option<NonNull<T>>, AraError>,
) -> *mut T {
    dispatch(runtime, interface, method, operation)
        .flatten()
        .map_or(std::ptr::null_mut(), NonNull::as_ptr)
}

/// Dispatches a signed integer callback with zero failure sentinel.
pub fn dispatch_i32(
    runtime: &impl DispatchRuntime,
    interface: &'static str,
    method: &'static str,
    operation: impl FnOnce() -> Result<i32, AraError>,
) -> i32 {
    dispatch(runtime, interface, method, operation).unwrap_or(0)
}

/// Dispatches a playback head/tail query with zero-pair failure sentinel.
pub fn dispatch_time_pair(
    runtime: &impl DispatchRuntime,
    interface: &'static str,
    method: &'static str,
    operation: impl FnOnce() -> Result<(f64, f64), AraError>,
) -> (f64, f64) {
    dispatch(runtime, interface, method, operation).unwrap_or((0.0, 0.0))
}
