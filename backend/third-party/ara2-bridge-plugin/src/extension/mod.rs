//! ARA plug-in extension role resolution and shared-lifetime interface storage.

mod editor;
mod playback;
mod view;

use ara2_bridge_core::{ApiGeneration, AraError, ContentTimeRange};
use ara2_bridge_sys::*;
use bitflags::bitflags;
use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::null;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;

bitflags! {
    /// ARA 2 renderer/view roles known or assigned by a companion API.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ExtensionRoles: i32 {
        /// Playback renderer assignment.
        const PLAYBACK_RENDERER = kARAPlaybackRendererRole as i32;
        /// Editor renderer assignment.
        const EDITOR_RENDERER = kARAEditorRendererRole as i32;
        /// Editor view assignment.
        const EDITOR_VIEW = kARAEditorViewRole as i32;
    }
}

impl ExtensionRoles {
    /// Resolves enabled roles using `supported & (!known | assigned)` and rejects unknown assigns.
    pub fn resolve(known: Self, assigned: Self, supported: Self) -> Result<Self, AraError> {
        if !(assigned & !known).is_empty() {
            return Err(AraError::InvalidArgument(
                "assigned extension roles must be declared known",
            ));
        }
        Ok(supported & (!known | assigned))
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) struct ExtensionState {
    controller_alive: AtomicBool,
    companion_alive: AtomicBool,
    enabled: ExtensionRoles,
    legacy: bool,
    playback_regions: Mutex<HashSet<(i32, usize)>>,
    region_sequences: Mutex<HashSet<usize>>,
    model_thread: ThreadId,
    selection: Mutex<Option<ExtensionViewSelection>>,
    hidden_sequences: Mutex<Vec<usize>>,
}

/// Owned copy of one editor-view selection notification.
#[derive(Clone, Debug)]
pub struct ExtensionViewSelection {
    playback_regions: Vec<usize>,
    region_sequences: Vec<usize>,
    time_range: Option<ContentTimeRange>,
}

impl ExtensionViewSelection {
    /// Returns selected playback-region pointer keys in caller order.
    pub fn playback_regions(&self) -> &[usize] {
        &self.playback_regions
    }

    /// Returns selected region-sequence pointer keys in caller order.
    pub fn region_sequences(&self) -> &[usize] {
        &self.region_sequences
    }

    /// Returns the copied optional selected time range.
    pub const fn time_range(&self) -> Option<&ContentTimeRange> {
        self.time_range.as_ref()
    }
}

impl ExtensionState {
    fn require_controller(&self) -> Result<(), AraError> {
        if std::thread::current().id() != self.model_thread {
            Err(AraError::InvalidState(
                "extension assignment must run on the binding thread",
            ))
        } else if self.controller_alive.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(AraError::InvalidState(
                "document controller has been destroyed",
            ))
        }
    }

    fn add_playback_region(&self, role: ExtensionRoles, region: usize) -> Result<(), AraError> {
        self.require_controller()?;
        if region == 0 || !self.enabled.contains(role) || role.bits().count_ones() != 1 {
            return Err(AraError::InvalidArgument(
                "playback region role is not enabled",
            ));
        }
        if !lock(&self.playback_regions).insert((role.bits(), region)) {
            return Err(AraError::InvalidState(
                "playback region is already assigned to this role",
            ));
        }
        Ok(())
    }

    fn remove_playback_region(&self, role: ExtensionRoles, region: usize) -> Result<(), AraError> {
        self.require_controller()?;
        if !lock(&self.playback_regions).remove(&(role.bits(), region)) {
            return Err(AraError::InvalidState(
                "playback region is not assigned to this role",
            ));
        }
        Ok(())
    }
}

struct ExtensionAllocation {
    state: Arc<ExtensionState>,
    _legacy_interface: Arc<ARAPlugInExtensionInterface>,
    _playback_interface: Arc<ARAPlaybackRendererInterface>,
    _editor_interface: Arc<ARAEditorRendererInterface>,
    _view_interface: Arc<ARAEditorViewInterface>,
    instance: Arc<ARAPlugInExtensionInstance>,
}

// SAFETY: all raw records are immutable after construction, their nested pointers target the
// allocation's retained `ExtensionState`, and every mutable callback collection is mutex-guarded.
unsafe impl Send for ExtensionAllocation {}
// SAFETY: the same immutable-backing and synchronized-state invariants permit shared role calls.
unsafe impl Sync for ExtensionAllocation {}

/// Companion-owned extension binding with stable ARA interface storage.
pub struct ExtensionBinding {
    allocation: Arc<ExtensionAllocation>,
}

impl ExtensionBinding {
    /// Resolves roles and creates stable legacy/ARA 2 extension interfaces.
    pub fn new(
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
        supported: ExtensionRoles,
    ) -> Result<(Self, ExtensionControllerLease), AraError> {
        if !generation.supported_on_target() {
            return Err(AraError::Unsupported(
                "extension generation is unavailable on this target",
            ));
        }
        let legacy = generation < ApiGeneration::V2Draft;
        let enabled = if legacy {
            ExtensionRoles::empty()
        } else {
            ExtensionRoles::resolve(known, assigned, supported)?
        };
        let state = Arc::new(ExtensionState {
            controller_alive: AtomicBool::new(true),
            companion_alive: AtomicBool::new(true),
            enabled,
            legacy,
            playback_regions: Mutex::new(HashSet::new()),
            region_sequences: Mutex::new(HashSet::new()),
            model_thread: std::thread::current().id(),
            selection: Mutex::new(None),
            hidden_sequences: Mutex::new(Vec::new()),
        });
        let state_pointer = Arc::as_ptr(&state).cast_mut();
        let legacy_interface = Arc::new(ARAPlugInExtensionInterface {
            structSize: std::mem::size_of::<ARAPlugInExtensionInterface>(),
            setPlaybackRegion: Some(playback::legacy_set_playback_region),
            removePlaybackRegion: Some(playback::legacy_remove_playback_region),
        });
        let playback_interface = Arc::new(ARAPlaybackRendererInterface {
            structSize: std::mem::size_of::<ARAPlaybackRendererInterface>(),
            addPlaybackRegion: Some(playback::add_playback_region),
            removePlaybackRegion: Some(playback::remove_playback_region),
        });
        let editor_interface = Arc::new(ARAEditorRendererInterface {
            structSize: std::mem::size_of::<ARAEditorRendererInterface>(),
            addPlaybackRegion: Some(editor::add_playback_region),
            removePlaybackRegion: Some(editor::remove_playback_region),
            addRegionSequence: Some(editor::add_region_sequence),
            removeRegionSequence: Some(editor::remove_region_sequence),
        });
        let view_interface = Arc::new(ARAEditorViewInterface {
            structSize: std::mem::size_of::<ARAEditorViewInterface>(),
            notifySelection: Some(view::notify_selection),
            notifyHideRegionSequences: Some(view::notify_hide_region_sequences),
        });
        // Raw ARA interface records contain opaque pointers but are immutable after publication;
        // `ExtensionAllocation` supplies the audited cross-thread synchronization contract.
        #[allow(clippy::arc_with_non_send_sync)]
        let instance = Arc::new(ARAPlugInExtensionInstance {
            structSize: std::mem::size_of::<ARAPlugInExtensionInstance>(),
            plugInExtensionRef: if legacy {
                state_pointer.cast()
            } else {
                std::ptr::null_mut()
            },
            plugInExtensionInterface: if legacy {
                Arc::as_ptr(&legacy_interface)
            } else {
                null()
            },
            playbackRendererRef: if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                state_pointer.cast()
            } else {
                std::ptr::null_mut()
            },
            playbackRendererInterface: if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                Arc::as_ptr(&playback_interface)
            } else {
                null()
            },
            editorRendererRef: if enabled.contains(ExtensionRoles::EDITOR_RENDERER) {
                state_pointer.cast()
            } else {
                std::ptr::null_mut()
            },
            editorRendererInterface: if enabled.contains(ExtensionRoles::EDITOR_RENDERER) {
                Arc::as_ptr(&editor_interface)
            } else {
                null()
            },
            editorViewRef: if enabled.contains(ExtensionRoles::EDITOR_VIEW) {
                state_pointer.cast()
            } else {
                std::ptr::null_mut()
            },
            editorViewInterface: if enabled.contains(ExtensionRoles::EDITOR_VIEW) {
                Arc::as_ptr(&view_interface)
            } else {
                null()
            },
        });
        let allocation = Arc::new(ExtensionAllocation {
            state,
            _legacy_interface: legacy_interface,
            _playback_interface: playback_interface,
            _editor_interface: editor_interface,
            _view_interface: view_interface,
            instance,
        });
        Ok((
            Self {
                allocation: allocation.clone(),
            },
            ExtensionControllerLease {
                allocation: Some(allocation),
            },
        ))
    }

    /// Returns the enabled ARA 2 role set.
    pub fn enabled_roles(&self) -> ExtensionRoles {
        self.allocation.state.enabled
    }

    /// Returns whether the deprecated ARA 1 extension pair is represented.
    pub fn has_legacy_extension(&self) -> bool {
        self.allocation.state.legacy
    }

    /// Returns the stable raw extension-instance pointer.
    pub fn as_raw(&self) -> *const ARAPlugInExtensionInstance {
        Arc::as_ptr(&self.allocation.instance)
    }

    /// Returns whether this owner still retains interface storage.
    pub const fn storage_is_alive(&self) -> bool {
        true
    }

    /// Assigns a playback region to one enabled renderer role.
    pub fn add_playback_region(
        &self,
        role: ExtensionRoles,
        region_key: usize,
    ) -> Result<(), AraError> {
        self.allocation.state.add_playback_region(role, region_key)
    }

    /// Removes a playback region from one enabled renderer role.
    pub fn remove_playback_region(
        &self,
        role: ExtensionRoles,
        region_key: usize,
    ) -> Result<(), AraError> {
        self.allocation
            .state
            .remove_playback_region(role, region_key)
    }

    /// Returns the most recent fully copied editor-view selection.
    pub fn view_selection(&self) -> Option<ExtensionViewSelection> {
        lock(&self.allocation.state.selection).clone()
    }

    /// Returns the most recent copied hidden-sequence pointer keys.
    pub fn hidden_region_sequences(&self) -> Vec<usize> {
        lock(&self.allocation.state.hidden_sequences).clone()
    }

    /// Returns current playback-region and region-sequence assignment counts.
    pub fn assignment_counts(&self) -> (usize, usize) {
        (
            lock(&self.allocation.state.playback_regions).len(),
            lock(&self.allocation.state.region_sequences).len(),
        )
    }
}

impl Drop for ExtensionBinding {
    fn drop(&mut self) {
        self.allocation
            .state
            .companion_alive
            .store(false, Ordering::Release);
    }
}

/// Controller-side owner retaining extension storage after companion teardown.
pub struct ExtensionControllerLease {
    allocation: Option<Arc<ExtensionAllocation>>,
}

impl ExtensionControllerLease {
    /// Tombstones controller-facing graph calls and releases this owner.
    pub fn destroy(mut self) {
        if let Some(allocation) = self.allocation.take() {
            allocation
                .state
                .controller_alive
                .store(false, Ordering::Release);
        }
    }

    /// Returns whether this controller owner still retains interface storage.
    pub fn storage_is_alive(&self) -> bool {
        self.allocation.is_some()
    }
}

impl Drop for ExtensionControllerLease {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.as_ref() {
            allocation
                .state
                .controller_alive
                .store(false, Ordering::Release);
        }
    }
}

pub(crate) unsafe fn with_state<R>(
    reference: *mut std::ffi::c_void,
    fallback: R,
    callback: impl FnOnce(&ExtensionState) -> R,
) -> R {
    if reference.is_null() {
        return fallback;
    }
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: role references are the retained `Arc<ExtensionState>` pointee.
        callback(unsafe { &*reference.cast::<ExtensionState>() })
    }))
    .unwrap_or(fallback)
}
