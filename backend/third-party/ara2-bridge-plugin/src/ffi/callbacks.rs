//! Small shared pieces used by generated callback shells.

#![allow(dead_code)]

use super::generated_callbacks::ControllerDelegate;
use super::ControllerInterface;
use ara2_bridge_sys::{ARADocumentControllerInstance, ARADocumentControllerRef};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

/// Coverage entry joining one C callback name to its generated delegate index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Delegate {
    /// Exact field name in `ARADocumentControllerInterface`.
    pub c_name: &'static str,
    /// Zero-based header-order slot index.
    pub slot: usize,
}

impl Delegate {
    pub(crate) const fn new(c_name: &'static str, slot: usize) -> Self {
        Self { c_name, slot }
    }
}

/// Required conformance-test classification for one generated callback shell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CallbackContract {
    /// Exact field name in `ARADocumentControllerInterface`.
    pub c_name: &'static str,
    /// Zero-based header-order slot index used by ABI dispatch tests.
    pub slot: usize,
}

impl CallbackContract {
    pub(crate) const fn new(c_name: &'static str, slot: usize) -> Self {
        Self { c_name, slot }
    }
}

pub(crate) struct ControllerCell {
    delegate: Mutex<Box<dyn ControllerDelegate>>,
    interface: ControllerInterface,
    instance: AtomicPtr<ARADocumentControllerInstance>,
}

impl Drop for ControllerCell {
    fn drop(&mut self) {
        let instance = self.instance.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !instance.is_null() {
            // SAFETY: `controller_instance` transfers exactly one boxed allocation into this raw
            // pointer, and the enclosing controller cell is its sole lifetime owner.
            drop(unsafe { Box::from_raw(instance) });
        }
    }
}

pub(crate) fn controller_ref(delegate: Box<dyn ControllerDelegate>) -> ARADocumentControllerRef {
    Box::into_raw(Box::new(ControllerCell {
        delegate: Mutex::new(delegate),
        interface: super::document_controller_interface(
            ara2_bridge_core::ApiGeneration::V23Final,
            super::ControllerCapabilities::default(),
        )
        .expect("2.3 base interface is supported on test targets"),
        instance: AtomicPtr::new(std::ptr::null_mut()),
    }))
    .cast()
}

pub(crate) fn controller_instance(
    delegate: Box<dyn ControllerDelegate>,
    interface: ControllerInterface,
) -> *const ARADocumentControllerInstance {
    let cell = Box::new(ControllerCell {
        delegate: Mutex::new(delegate),
        interface,
        instance: AtomicPtr::new(std::ptr::null_mut()),
    });
    let cell = Box::into_raw(cell);
    let controller_ref = cell.cast();
    // SAFETY: `cell` is a live uniquely owned allocation that is not exposed until the instance
    // is returned; this shared access only obtains its stable interface pointer.
    let interface = unsafe { (*cell).interface.as_raw() };
    let instance = Box::new(ARADocumentControllerInstance {
        structSize: std::mem::size_of::<ARADocumentControllerInstance>(),
        documentControllerRef: controller_ref,
        documentControllerInterface: interface,
    });
    let pointer = Box::into_raw(instance);
    // SAFETY: `cell` remains live and its atomic field permits initialization through a shared
    // reference without invalidating the controller reference embedded in `instance`.
    unsafe { (*cell).instance.store(pointer, Ordering::Release) };
    pointer.cast_const()
}

pub(crate) unsafe fn destroy_controller_ref(controller: ARADocumentControllerRef) {
    if !controller.is_null() {
        // SAFETY: the caller passes an exclusively owned pointer created by `controller_ref`.
        drop(unsafe { Box::from_raw(controller.cast::<ControllerCell>()) });
    }
}

pub(crate) fn dispatch<R: Copy>(
    controller: ARADocumentControllerRef,
    fallback: R,
    callback: impl FnOnce(&mut dyn ControllerDelegate) -> R,
) -> R {
    if controller.is_null() {
        return fallback;
    }
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: generated callbacks receive a live runtime-owned controller reference.
        let cell = unsafe { &*controller.cast::<ControllerCell>() };
        let mut delegate = cell
            .delegate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        callback(delegate.as_mut())
    }))
    .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::generated_callbacks;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Fixture {
        calls: Arc<AtomicUsize>,
        panic_once: Arc<AtomicBool>,
    }

    impl ControllerDelegate for Fixture {
        fn begin_editing(&mut self) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.panic_once.swap(false, Ordering::SeqCst) {
                panic!("fixture panic");
            }
        }

        fn get_processing_algorithms_count(&mut self) -> i32 {
            7
        }
    }

    #[test]
    fn generated_shells_call_named_delegate_and_contain_panics() {
        let calls = Arc::new(AtomicUsize::new(0));
        let panic_once = Arc::new(AtomicBool::new(true));
        let controller = controller_ref(Box::new(Fixture {
            calls: calls.clone(),
            panic_once,
        }));
        // SAFETY: `controller` is runtime-owned and remains live until the final destroy callback.
        unsafe { generated_callbacks::begin_editing(controller) };
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // SAFETY: the poisoned delegate lock is recovered and the same controller remains live.
        let algorithm_count =
            unsafe { generated_callbacks::get_processing_algorithms_count(controller) };
        assert_eq!(algorithm_count, 7);
        // SAFETY: this is the unique terminal callback and consumes the controller allocation.
        unsafe { generated_callbacks::destroy_document_controller(controller) };
    }
}
