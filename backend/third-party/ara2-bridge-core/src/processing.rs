//! Processing-algorithm backing and license-capability validation.

use crate::AraError;
use bitflags::bitflags;
use std::collections::BTreeSet;
use std::ffi::{c_char, CString};

bitflags! {
    /// ARA playback-transformation flags, retaining future unknown bits.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PlaybackTransformationFlags: u32 {
        /// Time stretching.
        const TIMESTRETCH = 1 << 0;
        /// Time stretching follows musical-context tempo.
        const REFLECT_TEMPO = 1 << 1;
        /// Content-based tail fade.
        const CONTENT_FADE_TAIL = 1 << 2;
        /// Content-based head fade.
        const CONTENT_FADE_HEAD = 1 << 3;
        /// Both content-based fade borders.
        const CONTENT_FADES = Self::CONTENT_FADE_TAIL.bits() | Self::CONTENT_FADE_HEAD.bits();
    }
}

/// Controller-lifetime owned processing-algorithm strings.
#[derive(Debug)]
pub struct ProcessingAlgorithmProperties {
    persistent_id: CString,
    name: CString,
}

impl ProcessingAlgorithmProperties {
    /// Creates one algorithm description.
    pub fn new(persistent_id: &str, name: &str) -> Result<Self, AraError> {
        if persistent_id.is_empty() || !persistent_id.is_ascii() {
            return Err(AraError::InvalidArgument(
                "processing algorithm ID must be nonempty ASCII",
            ));
        }
        if name.is_empty() {
            return Err(AraError::InvalidArgument(
                "processing algorithm name must be nonempty",
            ));
        }
        Ok(Self {
            persistent_id: CString::new(persistent_id)
                .map_err(|_| AraError::InvalidArgument("processing algorithm ID contains NUL"))?,
            name: CString::new(name)
                .map_err(|_| AraError::InvalidArgument("processing algorithm name contains NUL"))?,
        })
    }

    /// Returns the persistent algorithm ID.
    pub fn persistent_id(&self) -> &str {
        self.persistent_id
            .to_str()
            .expect("constructor accepted ASCII")
    }

    /// Returns the display name.
    pub fn name(&self) -> &str {
        self.name.to_str().expect("constructor accepted UTF-8")
    }
}

/// Borrowed ABI-facing algorithm properties backed by a catalog entry.
#[derive(Clone, Copy, Debug)]
pub struct ProcessingAlgorithmFfi {
    persistent_id: *const c_char,
    name: *const c_char,
}

impl ProcessingAlgorithmFfi {
    /// Returns the stable persistent-ID pointer.
    pub const fn persistent_id(self) -> *const c_char {
        self.persistent_id
    }

    /// Returns the stable display-name pointer.
    pub const fn name(self) -> *const c_char {
        self.name
    }

    /// Builds the generated ARA record whose nested pointers remain catalog-backed.
    pub fn as_ara(self) -> ara2_bridge_sys::ARAProcessingAlgorithmProperties {
        ara2_bridge_sys::ARAProcessingAlgorithmProperties {
            structSize: std::mem::size_of::<ara2_bridge_sys::ARAProcessingAlgorithmProperties>(),
            persistentID: self.persistent_id,
            name: self.name,
        }
    }
}

/// Algorithm table with heap-backed strings owned for a document-controller lifetime.
#[derive(Debug)]
pub struct ProcessingAlgorithmCatalog {
    entries: Vec<ProcessingAlgorithmProperties>,
}

impl ProcessingAlgorithmCatalog {
    /// Creates a catalog and rejects duplicate persistent IDs.
    pub fn new(entries: Vec<ProcessingAlgorithmProperties>) -> Result<Self, AraError> {
        let mut ids = BTreeSet::new();
        if entries
            .iter()
            .any(|entry| !ids.insert(entry.persistent_id().to_owned()))
        {
            return Err(AraError::InvalidArgument(
                "duplicate processing algorithm ID",
            ));
        }
        Ok(Self { entries })
    }

    /// Returns the number of algorithms as an ARA index count.
    pub fn len_i32(&self) -> Result<i32, AraError> {
        i32::try_from(self.entries.len())
            .map_err(|_| AraError::InvalidState("processing algorithm count exceeds i32"))
    }

    /// Returns whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a validated stable-index entry.
    pub fn get(&self, index: i32) -> Result<&ProcessingAlgorithmProperties, AraError> {
        let index = usize::try_from(index)
            .map_err(|_| AraError::InvalidArgument("negative processing algorithm index"))?;
        self.entries.get(index).ok_or(AraError::InvalidArgument(
            "processing algorithm index is out of bounds",
        ))
    }

    /// Returns stable pointers suitable for an ARA properties record.
    pub fn raw(&self, index: i32) -> Result<ProcessingAlgorithmFfi, AraError> {
        let entry = self.get(index)?;
        Ok(ProcessingAlgorithmFfi {
            persistent_id: entry.persistent_id.as_ptr(),
            name: entry.name.as_ptr(),
        })
    }
}

/// Plug-in capabilities against which license requests are checked.
#[derive(Clone, Debug)]
pub struct LicenseCapabilities {
    content_types: BTreeSet<i32>,
    transformations: PlaybackTransformationFlags,
}

impl LicenseCapabilities {
    /// Creates supported content and transformation capabilities.
    pub fn new(
        content_types: impl IntoIterator<Item = i32>,
        transformations: PlaybackTransformationFlags,
    ) -> Result<Self, AraError> {
        let fades = transformations & PlaybackTransformationFlags::CONTENT_FADES;
        if !fades.is_empty() && fades != PlaybackTransformationFlags::CONTENT_FADES {
            return Err(AraError::InvalidArgument(
                "supported content fades must include both borders",
            ));
        }
        Ok(Self {
            content_types: content_types.into_iter().collect(),
            transformations,
        })
    }
}

/// Validated license request whose content types and flags are a supported subset.
#[derive(Clone, Debug)]
pub struct LicenseRequest {
    run_modal_activation: bool,
    content_types: Vec<i32>,
    transformations: PlaybackTransformationFlags,
}

impl LicenseRequest {
    /// Creates a request after subset and duplicate validation.
    pub fn new(
        run_modal_activation: bool,
        content_types: impl IntoIterator<Item = i32>,
        transformations: PlaybackTransformationFlags,
        capabilities: &LicenseCapabilities,
    ) -> Result<Self, AraError> {
        let content_types: Vec<_> = content_types.into_iter().collect();
        let mut unique = BTreeSet::new();
        if content_types.iter().any(|content| !unique.insert(*content)) {
            return Err(AraError::InvalidArgument(
                "license request contains duplicate content types",
            ));
        }
        if content_types
            .iter()
            .any(|content| !capabilities.content_types.contains(content))
        {
            return Err(AraError::InvalidArgument(
                "license content request is not supported",
            ));
        }
        if !(transformations & !capabilities.transformations).is_empty() {
            return Err(AraError::InvalidArgument(
                "license transformation request is not supported",
            ));
        }
        Ok(Self {
            run_modal_activation,
            content_types,
            transformations,
        })
    }

    /// Returns whether a modal activation dialog may run.
    pub const fn run_modal_activation(&self) -> bool {
        self.run_modal_activation
    }

    /// Returns requested content types in caller order.
    pub fn content_types(&self) -> &[i32] {
        &self.content_types
    }

    /// Returns requested playback transformations.
    pub const fn transformations(&self) -> PlaybackTransformationFlags {
        self.transformations
    }
}
