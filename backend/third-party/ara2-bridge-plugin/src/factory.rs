//! Immutable, ABI-facing ARA factory storage.

use crate::builder::FactorySpec;
use crate::entry::{CallbackState, PluginEntry};
use ara2_bridge_core::{ApiGeneration, AraError};
use ara2_bridge_sys::{
    ARADocumentControllerHostInstance, ARADocumentControllerInstance, ARADocumentProperties,
    ARAFactory, ARAInterfaceConfiguration,
};
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::pin::Pin;
use std::ptr::{addr_of, null, NonNull};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

pub(crate) type ControllerCreator = dyn Fn(
        ApiGeneration,
        *const ARAFactory,
        *const ARADocumentControllerHostInstance,
        *const ARADocumentProperties,
    ) -> Result<*const ARADocumentControllerInstance, AraError>
    + Send
    + Sync;

const CALLBACK_SLOT_COUNT: usize = 64;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn slots() -> &'static [Mutex<Option<Weak<CallbackState>>>] {
    static SLOTS: OnceLock<Vec<Mutex<Option<Weak<CallbackState>>>>> = OnceLock::new();
    SLOTS
        .get_or_init(|| (0..CALLBACK_SLOT_COUNT).map(|_| Mutex::new(None)).collect())
        .as_slice()
}

struct SlotRegistration {
    index: usize,
    state: Weak<CallbackState>,
}

impl SlotRegistration {
    fn allocate(state: &Arc<CallbackState>) -> Result<Self, AraError> {
        for (index, slot) in slots().iter().enumerate() {
            let mut registered = lock(slot);
            if registered.as_ref().and_then(Weak::upgrade).is_none() {
                let state = Arc::downgrade(state);
                *registered = Some(state.clone());
                return Ok(Self { index, state });
            }
        }
        Err(AraError::Unsupported(
            "this binary already exports the maximum of 64 ARA factories",
        ))
    }
}

impl Drop for SlotRegistration {
    fn drop(&mut self) {
        let mut registered = lock(&slots()[self.index]);
        if registered
            .as_ref()
            .is_some_and(|current| Weak::ptr_eq(current, &self.state))
        {
            *registered = None;
        }
    }
}

fn callback_state(slot: usize) -> Option<Arc<CallbackState>> {
    slots()
        .get(slot)
        .and_then(|entry| lock(entry).as_ref()?.upgrade())
}

unsafe extern "C" fn initialize<const SLOT: usize>(config: *const ARAInterfaceConfiguration) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(state) = callback_state(SLOT) else {
            return;
        };
        if config.is_null() {
            return;
        }
        // SAFETY: the non-null pointer is caller-owned for the call. Each packed field is copied
        // unaligned, and the size field is checked before accessing either tail field.
        let size = unsafe { addr_of!((*config).structSize).read_unaligned() };
        if size < ara2_bridge_sys::layout::ARAINTERFACE_CONFIGURATION_ASSERT_FUNCTION_ADDRESS {
            return;
        }
        // SAFETY: the checked byte extent covers both fields and both are copied unaligned.
        let generation = unsafe { addr_of!((*config).desiredApiGeneration).read_unaligned() };
        // SAFETY: same checked extent as above.
        let assert_address = unsafe { addr_of!((*config).assertFunctionAddress).read_unaligned() };
        if let Ok(generation) = ApiGeneration::try_from_raw(generation) {
            let _ = state.initialize(generation, assert_address);
        }
    }));
}

unsafe extern "C" fn uninitialize<const SLOT: usize>() {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(state) = callback_state(SLOT) {
            let _ = state.uninitialize();
        }
    }));
}

unsafe extern "C" fn create_document_controller<const SLOT: usize>(
    host_instance: *const ARADocumentControllerHostInstance,
    properties: *const ARADocumentProperties,
) -> *const ARADocumentControllerInstance {
    catch_unwind(AssertUnwindSafe(|| {
        callback_state(SLOT)
            .and_then(|state| {
                state
                    .create_document_controller(host_instance, properties)
                    .ok()
            })
            .unwrap_or_else(null)
    }))
    .unwrap_or(null())
}

type Initialize = unsafe extern "C" fn(*const ARAInterfaceConfiguration);
type Uninitialize = unsafe extern "C" fn();
type Create = unsafe extern "C" fn(
    *const ARADocumentControllerHostInstance,
    *const ARADocumentProperties,
) -> *const ARADocumentControllerInstance;

macro_rules! callbacks {
    ($callback:ident, $type:ty) => {
        [
            $callback::<0> as $type,
            $callback::<1>,
            $callback::<2>,
            $callback::<3>,
            $callback::<4>,
            $callback::<5>,
            $callback::<6>,
            $callback::<7>,
            $callback::<8>,
            $callback::<9>,
            $callback::<10>,
            $callback::<11>,
            $callback::<12>,
            $callback::<13>,
            $callback::<14>,
            $callback::<15>,
            $callback::<16>,
            $callback::<17>,
            $callback::<18>,
            $callback::<19>,
            $callback::<20>,
            $callback::<21>,
            $callback::<22>,
            $callback::<23>,
            $callback::<24>,
            $callback::<25>,
            $callback::<26>,
            $callback::<27>,
            $callback::<28>,
            $callback::<29>,
            $callback::<30>,
            $callback::<31>,
            $callback::<32>,
            $callback::<33>,
            $callback::<34>,
            $callback::<35>,
            $callback::<36>,
            $callback::<37>,
            $callback::<38>,
            $callback::<39>,
            $callback::<40>,
            $callback::<41>,
            $callback::<42>,
            $callback::<43>,
            $callback::<44>,
            $callback::<45>,
            $callback::<46>,
            $callback::<47>,
            $callback::<48>,
            $callback::<49>,
            $callback::<50>,
            $callback::<51>,
            $callback::<52>,
            $callback::<53>,
            $callback::<54>,
            $callback::<55>,
            $callback::<56>,
            $callback::<57>,
            $callback::<58>,
            $callback::<59>,
            $callback::<60>,
            $callback::<61>,
            $callback::<62>,
            $callback::<63>,
        ]
    };
}

static INITIALIZE: [Initialize; CALLBACK_SLOT_COUNT] = callbacks!(initialize, Initialize);
static UNINITIALIZE: [Uninitialize; CALLBACK_SLOT_COUNT] = callbacks!(uninitialize, Uninitialize);
static CREATE: [Create; CALLBACK_SLOT_COUNT] = callbacks!(create_document_controller, Create);

struct FactoryBacking {
    _factory_id: CString,
    _archive_id: CString,
    _name: CString,
    _manufacturer: CString,
    _information_url: CString,
    _version: CString,
    _compatible_ids: Vec<CString>,
    _analyzable_content_types: Vec<i32>,
}

/// One immutable ARA factory and its independent initialization entry.
pub struct Factory {
    id: String,
    entry: PluginEntry,
    _slot: SlotRegistration,
    backing: Pin<Box<FactoryBacking>>,
    compatible_id_pointers: Pin<Box<[*const c_char]>>,
    raw: NonNull<ARAFactory>,
}

impl Factory {
    pub(crate) fn new(spec: FactorySpec) -> Result<Self, AraError> {
        let state = Arc::new(CallbackState::new(
            spec.lowest,
            spec.highest,
            spec.create_controller,
        ));
        let slot = SlotRegistration::allocate(&state)?;
        let id = spec
            .factory_id
            .to_str()
            .expect("validated ASCII")
            .to_owned();
        let backing = Box::pin(FactoryBacking {
            _factory_id: spec.factory_id,
            _archive_id: spec.archive_id,
            _name: spec.name,
            _manufacturer: spec.manufacturer,
            _information_url: spec.information_url,
            _version: spec.version,
            _compatible_ids: spec.compatible_ids,
            _analyzable_content_types: spec.capabilities.analyzable_content_types,
        });
        let compatible_id_pointers = backing
            ._compatible_ids
            .iter()
            .map(|id| id.as_ptr())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let compatible_id_pointers = Pin::new(compatible_id_pointers);
        let compatible_ids_pointer = if compatible_id_pointers.is_empty() {
            null()
        } else {
            compatible_id_pointers.as_ptr()
        };
        let analyzable_pointer = if backing._analyzable_content_types.is_empty() {
            null()
        } else {
            backing._analyzable_content_types.as_ptr()
        };
        let struct_size = if spec.capabilities.stores_audio_file_chunks {
            ara2_bridge_sys::layout::ARAFACTORY_SUPPORTS_STORING_AUDIO_FILE_CHUNKS
        } else {
            ara2_bridge_sys::layout::ARAFACTORY_SUPPORTED_PLAYBACK_TRANSFORMATION_FLAGS
        };
        let raw = Box::new(ARAFactory {
            structSize: struct_size,
            lowestSupportedApiGeneration: spec.lowest.as_raw(),
            highestSupportedApiGeneration: spec.highest.as_raw(),
            factoryID: backing._factory_id.as_ptr(),
            initializeARAWithConfiguration: Some(INITIALIZE[slot.index]),
            uninitializeARA: Some(UNINITIALIZE[slot.index]),
            plugInName: backing._name.as_ptr(),
            manufacturerName: backing._manufacturer.as_ptr(),
            informationURL: backing._information_url.as_ptr(),
            version: backing._version.as_ptr(),
            createDocumentControllerWithDocument: Some(CREATE[slot.index]),
            documentArchiveID: backing._archive_id.as_ptr(),
            compatibleDocumentArchiveIDsCount: compatible_id_pointers.len(),
            compatibleDocumentArchiveIDs: compatible_ids_pointer,
            analyzeableContentTypesCount: backing._analyzable_content_types.len(),
            analyzeableContentTypes: analyzable_pointer,
            supportedPlaybackTransformationFlags: spec.capabilities.playback_transformations.bits()
                as i32,
            supportsStoringAudioFileChunks: if spec.capabilities.stores_audio_file_chunks {
                ara2_bridge_sys::kARATrue
            } else {
                ara2_bridge_sys::kARAFalse
            },
        });
        let raw = NonNull::new(Box::into_raw(raw)).expect("Box pointers are non-null");
        state.attach_factory(raw.as_ptr());
        Ok(Self {
            id,
            entry: PluginEntry::new(state),
            _slot: slot,
            backing,
            compatible_id_pointers,
            raw,
        })
    }

    /// Returns this factory's persistent identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns this factory's independent initialization entry.
    pub fn entry(&self) -> &PluginEntry {
        &self.entry
    }

    /// Returns the stable raw factory pointer valid for this value's lifetime.
    pub fn as_raw(&self) -> *const ARAFactory {
        self.raw.as_ptr()
    }

    /// Copies the packed raw factory record for inspection or registration.
    pub fn raw_copy(&self) -> ARAFactory {
        // SAFETY: `raw` points to a fully initialized live record; unaligned copying supports the
        // generated packed representation on every target.
        unsafe { self.raw.as_ptr().read_unaligned() }
    }
}

impl Drop for Factory {
    fn drop(&mut self) {
        let _keep_backing_live = (&self.backing, &self.compatible_id_pointers);
        // SAFETY: construction transfers exactly one boxed factory allocation into `raw`.
        drop(unsafe { Box::from_raw(self.raw.as_ptr()) });
    }
}
