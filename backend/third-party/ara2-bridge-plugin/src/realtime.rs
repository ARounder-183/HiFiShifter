//! Allocation-free head/tail callback adapter with model-thread snapshot replacement.

use ara2_bridge_core::{
    AraError, HeadTailEntry, RealtimeFailureCode, RealtimeFailureQueue, RealtimeHeadTailView,
};
use ara2_bridge_sys::{ARAPlaybackRegionRef, ARATimeDuration};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

/// Owns immutable snapshots while realtime queries use one atomically selected view.
pub struct RealtimeHeadTailAdapter {
    snapshots: Mutex<Vec<Arc<RealtimeHeadTailView>>>,
    current: AtomicPtr<RealtimeHeadTailView>,
    capacity: usize,
    failures: RealtimeFailureQueue,
}

#[allow(dead_code)] // Constructed by the controller adapter added after capability composition.
pub(crate) struct HeadTailDispatch {
    pub(crate) adapter: Arc<RealtimeHeadTailAdapter>,
}

impl crate::ffi::generated_callbacks::ControllerDelegate for HeadTailDispatch {
    fn get_playback_region_head_and_tail_time(
        &mut self,
        playback_region: ARAPlaybackRegionRef,
        head_time: *mut ARATimeDuration,
        tail_time: *mut ARATimeDuration,
    ) {
        if playback_region.is_null() || head_time.is_null() || tail_time.is_null() {
            let _ = self
                .adapter
                .failures
                .report(RealtimeFailureCode::InvalidState);
            return;
        }
        let key = playback_region as usize as u64;
        let Some((head, tail)) = self.adapter.query(key) else {
            return;
        };
        // SAFETY: ARA supplies both output pointers writable for one duration during this callback.
        unsafe {
            head_time.write(head);
            tail_time.write(tail);
        }
    }
}

impl RealtimeHeadTailAdapter {
    /// Creates an adapter and its preallocated deferred-failure queue.
    pub fn new(capacity: usize) -> Result<Self, AraError> {
        if capacity == 0 {
            return Err(AraError::InvalidArgument(
                "head/tail snapshot capacity must be nonzero",
            ));
        }
        let initial = Arc::new(RealtimeHeadTailView::new([], capacity)?);
        let pointer = Arc::as_ptr(&initial).cast_mut();
        Ok(Self {
            snapshots: Mutex::new(vec![initial]),
            current: AtomicPtr::new(pointer),
            capacity,
            failures: RealtimeFailureQueue::new(capacity)?,
        })
    }

    /// Builds and atomically installs a validated immutable snapshot off the realtime path.
    pub fn install(
        &self,
        entries: impl IntoIterator<Item = HeadTailEntry>,
    ) -> Result<(), AraError> {
        let snapshot = Arc::new(RealtimeHeadTailView::new(entries, self.capacity)?);
        let pointer = Arc::as_ptr(&snapshot).cast_mut();
        self.snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(snapshot);
        self.current.store(pointer, Ordering::Release);
        Ok(())
    }

    /// Performs one allocation-free and blocking-lock-free head/tail lookup.
    pub fn query(&self, region_key: u64) -> Option<(f64, f64)> {
        let pointer = self.current.load(Ordering::Acquire);
        // SAFETY: every installed snapshot remains owned in `snapshots` until this adapter drops;
        // replacement never mutates or frees a view observed by a concurrent query.
        let result = unsafe { &*pointer }.query(region_key);
        if result.is_none() {
            let _ = self.failures.report(RealtimeFailureCode::MissingRegion);
        }
        result
    }

    /// Pops one deferred fixed-size realtime failure for model-thread diagnostics.
    pub fn pop_failure(&self) -> Option<RealtimeFailureCode> {
        self.failures.pop()
    }

    /// Returns the number of snapshots retained to protect concurrent readers.
    pub fn retained_snapshot_count(&self) -> usize {
        self.snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{callbacks, generated_callbacks};

    #[test]
    fn query_does_not_acquire_snapshot_storage_lock() {
        let adapter = RealtimeHeadTailAdapter::new(2).unwrap();
        adapter
            .install([HeadTailEntry::new(1, 0.1, 0.2).unwrap()])
            .unwrap();
        let storage = adapter.snapshots.lock().unwrap();
        assert_eq!(adapter.query(1), Some((0.1, 0.2)));
        drop(storage);
    }

    #[test]
    fn abi_head_tail_shell_uses_snapshot_while_storage_lock_is_held() {
        let adapter = Arc::new(RealtimeHeadTailAdapter::new(2).unwrap());
        let identity = Box::new(0_u8);
        let region: ARAPlaybackRegionRef = std::ptr::from_ref(identity.as_ref()).cast_mut().cast();
        adapter
            .install([HeadTailEntry::new(region as usize as u64, 0.25, 0.5).unwrap()])
            .unwrap();
        let controller = callbacks::controller_ref(Box::new(HeadTailDispatch {
            adapter: adapter.clone(),
        }));
        let storage = adapter.snapshots.lock().unwrap();
        let mut head = 0.0;
        let mut tail = 0.0;
        // SAFETY: all opaque identities, output pointers, and controller storage are live.
        unsafe {
            generated_callbacks::get_playback_region_head_and_tail_time(
                controller,
                region,
                &raw mut head,
                &raw mut tail,
            )
        };
        assert_eq!((head, tail), (0.25, 0.5));
        drop(storage);
        // SAFETY: this uniquely consumes the controller allocation after its callback completes.
        unsafe { callbacks::destroy_controller_ref(controller) };
    }
}
