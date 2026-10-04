//! Owned store and restore selections.

use crate::properties::{write_raw, zeroed_raw};
use crate::{
    AraBool, AraError, ArchiveError, AudioModificationKind, AudioSourceKind, ForeignSlice,
    ForeignStr, Handle, RegistrySession, SizedInput,
};
use ara2_bridge_sys::{
    ARAAudioModificationRef, ARAAudioSourceRef, ARABool, ARAPersistentID, ARARestoreObjectsFilter,
    ARAStoreObjectsFilter,
};
use std::collections::HashSet;
use std::ffi::CString;
use std::mem::offset_of;
use std::pin::Pin;
use std::ptr::null;

const MAX_FILTER_OBJECTS: usize = 1 << 20;
const MAX_PERSISTENT_ID_BYTES: usize = 1 << 20;

/// A null/all selection or an explicit owned filter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FilterSelection<T> {
    /// Select every matching piece of state, corresponding to a null FFI filter.
    All,
    /// Select only the entries represented by the owned filter.
    Selected(T),
}

impl<T> From<Option<T>> for FilterSelection<T> {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::All, Self::Selected)
    }
}

impl<T> FilterSelection<T> {
    /// Returns the explicit filter or `None` for the all/null selection.
    pub fn as_selected(&self) -> Option<&T> {
        match self {
            Self::All => None,
            Self::Selected(filter) => Some(filter),
        }
    }
}

/// One archive-to-current persistent-ID mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoreMapping {
    archive_id: String,
    current_id: String,
}

impl RestoreMapping {
    /// Returns the persistent ID stored in the archive.
    pub fn archive_id(&self) -> &str {
        &self.archive_id
    }

    /// Returns the matching persistent ID in the current document graph.
    pub fn current_id(&self) -> &str {
        &self.current_id
    }
}

/// Ordered categories used when applying a partial restore.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RestorePhase {
    /// Restore audio-source graph state.
    AudioSources,
    /// Restore audio-modification graph state.
    AudioModifications,
    /// Restore private document data after dependent graph state.
    DocumentData,
}

/// Owned partial-restore mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoreFilter {
    document_data: bool,
    audio_sources: Vec<RestoreMapping>,
    audio_modifications: Vec<RestoreMapping>,
}

/// Pinned ARA restore-filter record with owned persistent-ID backing.
pub struct FfiRestoreFilter {
    raw: ARARestoreObjectsFilter,
    source_archive_ids: Box<[CString]>,
    source_current_ids: Box<[CString]>,
    modification_archive_ids: Box<[CString]>,
    modification_current_ids: Box<[CString]>,
    source_archive_pointers: Box<[ARAPersistentID]>,
    source_current_pointers: Box<[ARAPersistentID]>,
    modification_archive_pointers: Box<[ARAPersistentID]>,
    modification_current_pointers: Box<[ARAPersistentID]>,
}

impl FfiRestoreFilter {
    /// Returns the raw filter pointer valid while this pinned owner is borrowed.
    pub fn as_ptr(&self) -> *const ARARestoreObjectsFilter {
        &raw const self.raw
    }
}

impl RestoreFilter {
    /// Starts a restore-filter builder.
    pub fn builder() -> RestoreFilterBuilder {
        RestoreFilterBuilder::default()
    }

    /// Returns whether private document data is selected.
    pub fn includes_document_data(&self) -> bool {
        self.document_data
    }

    /// Returns audio-source persistent-ID mappings in insertion order.
    pub fn audio_sources(&self) -> &[RestoreMapping] {
        &self.audio_sources
    }

    /// Returns audio-modification persistent-ID mappings in insertion order.
    pub fn audio_modifications(&self) -> &[RestoreMapping] {
        &self.audio_modifications
    }

    /// Returns the dependency-safe phase order for the selected categories.
    pub fn phases(&self) -> Vec<RestorePhase> {
        let mut phases = Vec::with_capacity(3);
        if !self.audio_sources.is_empty() {
            phases.push(RestorePhase::AudioSources);
        }
        if !self.audio_modifications.is_empty() {
            phases.push(RestorePhase::AudioModifications);
        }
        if self.document_data {
            phases.push(RestorePhase::DocumentData);
        }
        phases
    }

    /// Builds a pinned raw restore filter with all nested ID storage retained.
    pub fn as_ffi(&self) -> Pin<Box<FfiRestoreFilter>> {
        fn strings<'a>(values: impl Iterator<Item = &'a str>) -> Box<[CString]> {
            values
                .map(|value| CString::new(value).expect("validated persistent ID"))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        }
        let source_archive_ids = strings(self.audio_sources.iter().map(RestoreMapping::archive_id));
        let source_current_ids = strings(self.audio_sources.iter().map(RestoreMapping::current_id));
        let modification_archive_ids = strings(
            self.audio_modifications
                .iter()
                .map(RestoreMapping::archive_id),
        );
        let modification_current_ids = strings(
            self.audio_modifications
                .iter()
                .map(RestoreMapping::current_id),
        );
        let source_count = source_archive_ids.len();
        let modification_count = modification_archive_ids.len();
        // SAFETY: generated raw filter fields all accept the zero bit pattern.
        let raw = unsafe { zeroed_raw::<ARARestoreObjectsFilter>() };
        let mut output = Box::pin(FfiRestoreFilter {
            raw,
            source_archive_ids,
            source_current_ids,
            modification_archive_ids,
            modification_current_ids,
            source_archive_pointers: vec![null(); source_count].into_boxed_slice(),
            source_current_pointers: vec![null(); source_count].into_boxed_slice(),
            modification_archive_pointers: vec![null(); modification_count].into_boxed_slice(),
            modification_current_pointers: vec![null(); modification_count].into_boxed_slice(),
        });
        // SAFETY: the allocation is pinned before publishing any interior pointer. Every pointee
        // remains owned by `output` for at least as long as the raw filter can be borrowed.
        unsafe {
            let output = Pin::get_unchecked_mut(output.as_mut());
            for (pointer, value) in output
                .source_archive_pointers
                .iter_mut()
                .zip(output.source_archive_ids.iter())
            {
                *pointer = value.as_ptr();
            }
            for (pointer, value) in output
                .source_current_pointers
                .iter_mut()
                .zip(output.source_current_ids.iter())
            {
                *pointer = value.as_ptr();
            }
            for (pointer, value) in output
                .modification_archive_pointers
                .iter_mut()
                .zip(output.modification_archive_ids.iter())
            {
                *pointer = value.as_ptr();
            }
            for (pointer, value) in output
                .modification_current_pointers
                .iter_mut()
                .zip(output.modification_current_ids.iter())
            {
                *pointer = value.as_ptr();
            }
            let source_archive = slice_pointer(&output.source_archive_pointers);
            let source_current = slice_pointer(&output.source_current_pointers);
            let modification_archive = slice_pointer(&output.modification_archive_pointers);
            let modification_current = slice_pointer(&output.modification_current_pointers);
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, structSize),
                ara2_bridge_sys::layout::ARARESTORE_OBJECTS_FILTER_AUDIO_MODIFICATION_CURRENT_IDS,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, documentData),
                AraBool::from(self.document_data).into_raw(),
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, audioSourceIDsCount),
                source_count,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, audioSourceArchiveIDs),
                source_archive,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, audioSourceCurrentIDs),
                source_current,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, audioModificationIDsCount),
                modification_count,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, audioModificationArchiveIDs),
                modification_archive,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARARestoreObjectsFilter, audioModificationCurrentIDs),
                modification_current,
            );
        }
        output
    }

    /// Copies an optional ARA restore filter and all nested ID arrays.
    ///
    /// # Safety
    ///
    /// `pointer` and every represented nested array/string must satisfy their ARA call-lifetime
    /// contracts. Null selects all archived objects.
    pub unsafe fn copy_selection_from_ffi(
        pointer: *const ARARestoreObjectsFilter,
    ) -> Result<FilterSelection<Self>, AraError> {
        if pointer.is_null() {
            return Ok(FilterSelection::All);
        }
        // SAFETY: forwarded from this method's caller contract.
        let input = unsafe { SizedInput::from_ptr(pointer) }?;
        macro_rules! field {
            ($field:ident, $type:ty, $extent:ident) => {{
                // SAFETY: generated offset/type/extent identify this represented field.
                unsafe {
                    input.copy_field::<$type>(
                        offset_of!(ARARestoreObjectsFilter, $field),
                        ara2_bridge_sys::layout::$extent,
                    )
                }?
            }};
        }
        let document_data = AraBool::from_raw(field!(
            documentData,
            ARABool,
            ARARESTORE_OBJECTS_FILTER_DOCUMENT_DATA
        ))
        .get();
        let source_count = field!(
            audioSourceIDsCount,
            usize,
            ARARESTORE_OBJECTS_FILTER_AUDIO_SOURCE_IDS_COUNT
        );
        let source_archive = field!(
            audioSourceArchiveIDs,
            *const ARAPersistentID,
            ARARESTORE_OBJECTS_FILTER_AUDIO_SOURCE_ARCHIVE_IDS
        );
        let source_current = field!(
            audioSourceCurrentIDs,
            *const ARAPersistentID,
            ARARESTORE_OBJECTS_FILTER_AUDIO_SOURCE_CURRENT_IDS
        );
        let modification_count = field!(
            audioModificationIDsCount,
            usize,
            ARARESTORE_OBJECTS_FILTER_AUDIO_MODIFICATION_IDS_COUNT
        );
        let modification_archive = field!(
            audioModificationArchiveIDs,
            *const ARAPersistentID,
            ARARESTORE_OBJECTS_FILTER_AUDIO_MODIFICATION_ARCHIVE_IDS
        );
        let modification_current = field!(
            audioModificationCurrentIDs,
            *const ARAPersistentID,
            ARARESTORE_OBJECTS_FILTER_AUDIO_MODIFICATION_CURRENT_IDS
        );
        // SAFETY: the enclosing filter contract covers both represented source-ID arrays.
        let source_ids = unsafe { copy_id_mappings(source_archive, source_current, source_count) }?;
        // SAFETY: the enclosing filter contract covers both represented modification-ID arrays.
        let modification_ids = unsafe {
            copy_id_mappings(
                modification_archive,
                modification_current,
                modification_count,
            )
        }?;
        let mut builder = Self::builder().document_data(document_data);
        for (archive, current) in source_ids {
            builder = builder.audio_source(archive, current);
        }
        for (archive, current) in modification_ids {
            builder = builder.audio_modification(archive, current);
        }
        builder.build().map(FilterSelection::Selected)
    }
}

fn slice_pointer<T>(values: &[T]) -> *const T {
    if values.is_empty() {
        null()
    } else {
        values.as_ptr()
    }
}

unsafe fn copy_id_mappings(
    archives: *const ARAPersistentID,
    currents: *const ARAPersistentID,
    count: usize,
) -> Result<Vec<(String, String)>, AraError> {
    if count > MAX_FILTER_OBJECTS {
        return Err(AraError::InvalidArgument(
            "restore filter count exceeds limit",
        ));
    }
    // SAFETY: forwarded from the enclosing filter's nested-array contract.
    let archives = unsafe { ForeignSlice::copy_from_raw(archives, count) }?.into_vec();
    let currents = if currents.is_null() {
        archives.clone()
    } else {
        // SAFETY: same enclosing nested-array contract.
        unsafe { ForeignSlice::copy_from_raw(currents, count) }?.into_vec()
    };
    archives
        .into_iter()
        .zip(currents)
        .map(|(archive, current)| {
            // SAFETY: the enclosing filter contract covers every nested persistent ID.
            let archive =
                unsafe { ForeignStr::copy_persistent_id(archive, MAX_PERSISTENT_ID_BYTES) }?
                    .into_string();
            // SAFETY: same nested persistent-ID contract.
            let current =
                unsafe { ForeignStr::copy_persistent_id(current, MAX_PERSISTENT_ID_BYTES) }?
                    .into_string();
            Ok((archive, current))
        })
        .collect()
}

/// Builder for [`RestoreFilter`].
#[derive(Clone, Debug, Default)]
pub struct RestoreFilterBuilder {
    document_data: bool,
    audio_sources: Vec<(String, String)>,
    audio_modifications: Vec<(String, String)>,
}

impl RestoreFilterBuilder {
    /// Selects whether private document data is restored.
    pub fn document_data(mut self, selected: bool) -> Self {
        self.document_data = selected;
        self
    }

    /// Adds an archive-to-current audio-source persistent-ID mapping.
    pub fn audio_source(
        mut self,
        archive_id: impl Into<String>,
        current_id: impl Into<String>,
    ) -> Self {
        self.audio_sources
            .push((archive_id.into(), current_id.into()));
        self
    }

    /// Adds an archive-to-current audio-modification persistent-ID mapping.
    pub fn audio_modification(
        mut self,
        archive_id: impl Into<String>,
        current_id: impl Into<String>,
    ) -> Self {
        self.audio_modifications
            .push((archive_id.into(), current_id.into()));
        self
    }

    /// Builds and validates the restore filter.
    pub fn build(self) -> Result<RestoreFilter, AraError> {
        Ok(RestoreFilter {
            document_data: self.document_data,
            audio_sources: validate_mappings(self.audio_sources)?,
            audio_modifications: validate_mappings(self.audio_modifications)?,
        })
    }
}

fn validate_mappings(mappings: Vec<(String, String)>) -> Result<Vec<RestoreMapping>, AraError> {
    let mut archive_ids = HashSet::with_capacity(mappings.len());
    let mut current_ids = HashSet::with_capacity(mappings.len());
    mappings
        .into_iter()
        .map(|(archive_id, current_id)| {
            validate_id(&archive_id)?;
            validate_id(&current_id)?;
            if !archive_ids.insert(archive_id.clone()) || !current_ids.insert(current_id.clone()) {
                return Err(AraError::Archive(ArchiveError::InvalidFilter(
                    "restore mappings contain duplicate IDs",
                )));
            }
            Ok(RestoreMapping {
                archive_id,
                current_id,
            })
        })
        .collect()
}

fn validate_id(id: &str) -> Result<(), AraError> {
    if id.is_empty() || !id.is_ascii() || id.contains('\0') {
        return Err(AraError::Archive(ArchiveError::InvalidFilter(
            "persistent ID must be nonempty ASCII",
        )));
    }
    Ok(())
}

/// Owned partial-store selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoreFilter {
    session: RegistrySession,
    document_data: bool,
    audio_sources: Vec<Handle<AudioSourceKind>>,
    audio_modifications: Vec<Handle<AudioModificationKind>>,
}

/// Pinned ARA store-filter record with owned peer-reference arrays.
pub struct FfiStoreFilter {
    raw: ARAStoreObjectsFilter,
    audio_sources: Box<[ARAAudioSourceRef]>,
    audio_modifications: Box<[ARAAudioModificationRef]>,
}

impl FfiStoreFilter {
    /// Returns the raw filter pointer valid while this pinned owner is borrowed.
    pub fn as_ptr(&self) -> *const ARAStoreObjectsFilter {
        &raw const self.raw
    }
}

impl StoreFilter {
    /// Starts a store-filter builder for one registry session.
    pub fn builder(session: RegistrySession) -> StoreFilterBuilder {
        StoreFilterBuilder {
            session,
            ..StoreFilterBuilder::default()
        }
    }

    /// Returns whether private document data is selected.
    pub fn includes_document_data(&self) -> bool {
        self.document_data
    }

    /// Returns selected audio-source handles.
    pub fn audio_sources(&self) -> &[Handle<AudioSourceKind>] {
        &self.audio_sources
    }

    /// Returns selected audio-modification handles.
    pub fn audio_modifications(&self) -> &[Handle<AudioModificationKind>] {
        &self.audio_modifications
    }

    /// Returns the owning document-session identity.
    pub fn session(&self) -> RegistrySession {
        self.session
    }

    /// Builds a pinned raw store filter after resolving local handles to foreign peer references.
    pub fn as_ffi(
        &self,
        mut resolve_source: impl FnMut(Handle<AudioSourceKind>) -> Result<ARAAudioSourceRef, AraError>,
        mut resolve_modification: impl FnMut(
            Handle<AudioModificationKind>,
        ) -> Result<ARAAudioModificationRef, AraError>,
    ) -> Result<Pin<Box<FfiStoreFilter>>, AraError> {
        let audio_sources = self
            .audio_sources
            .iter()
            .copied()
            .map(&mut resolve_source)
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let audio_modifications = self
            .audio_modifications
            .iter()
            .copied()
            .map(&mut resolve_modification)
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        // SAFETY: generated raw filter fields all accept the zero bit pattern.
        let raw = unsafe { zeroed_raw::<ARAStoreObjectsFilter>() };
        let mut output = Box::pin(FfiStoreFilter {
            raw,
            audio_sources,
            audio_modifications,
        });
        // SAFETY: the allocation is pinned before publishing pointers into its boxed arrays.
        unsafe {
            let output = Pin::get_unchecked_mut(output.as_mut());
            let audio_sources = slice_pointer(&output.audio_sources);
            let audio_modifications = slice_pointer(&output.audio_modifications);
            write_raw(
                &mut output.raw,
                offset_of!(ARAStoreObjectsFilter, structSize),
                ara2_bridge_sys::layout::ARASTORE_OBJECTS_FILTER_AUDIO_MODIFICATION_REFS,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARAStoreObjectsFilter, documentData),
                AraBool::from(self.document_data).into_raw(),
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARAStoreObjectsFilter, audioSourceRefsCount),
                output.audio_sources.len(),
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARAStoreObjectsFilter, audioSourceRefs),
                audio_sources,
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARAStoreObjectsFilter, audioModificationRefsCount),
                output.audio_modifications.len(),
            );
            write_raw(
                &mut output.raw,
                offset_of!(ARAStoreObjectsFilter, audioModificationRefs),
                audio_modifications,
            );
        }
        Ok(output)
    }

    /// Copies an optional ARA store filter through controller-owned identity resolvers.
    ///
    /// # Safety
    ///
    /// `pointer` and every represented reference array must satisfy the ARA callback contract.
    /// Null selects the complete document graph.
    pub unsafe fn copy_selection_from_ffi(
        pointer: *const ARAStoreObjectsFilter,
        session: RegistrySession,
        mut resolve_source: impl FnMut(ARAAudioSourceRef) -> Result<Handle<AudioSourceKind>, AraError>,
        mut resolve_modification: impl FnMut(
            ARAAudioModificationRef,
        ) -> Result<Handle<AudioModificationKind>, AraError>,
    ) -> Result<FilterSelection<Self>, AraError> {
        if pointer.is_null() {
            return Ok(FilterSelection::All);
        }
        // SAFETY: forwarded from this method's caller contract.
        let input = unsafe { SizedInput::from_ptr(pointer) }?;
        macro_rules! field {
            ($field:ident, $type:ty, $extent:ident) => {{
                // SAFETY: generated offset/type/extent identify this represented field.
                unsafe {
                    input.copy_field::<$type>(
                        offset_of!(ARAStoreObjectsFilter, $field),
                        ara2_bridge_sys::layout::$extent,
                    )
                }?
            }};
        }
        let document_data = AraBool::from_raw(field!(
            documentData,
            ARABool,
            ARASTORE_OBJECTS_FILTER_DOCUMENT_DATA
        ))
        .get();
        let source_count = field!(
            audioSourceRefsCount,
            usize,
            ARASTORE_OBJECTS_FILTER_AUDIO_SOURCE_REFS_COUNT
        );
        let source_pointer = field!(
            audioSourceRefs,
            *const ARAAudioSourceRef,
            ARASTORE_OBJECTS_FILTER_AUDIO_SOURCE_REFS
        );
        let modification_count = field!(
            audioModificationRefsCount,
            usize,
            ARASTORE_OBJECTS_FILTER_AUDIO_MODIFICATION_REFS_COUNT
        );
        let modification_pointer = field!(
            audioModificationRefs,
            *const ARAAudioModificationRef,
            ARASTORE_OBJECTS_FILTER_AUDIO_MODIFICATION_REFS
        );
        if source_count > MAX_FILTER_OBJECTS || modification_count > MAX_FILTER_OBJECTS {
            return Err(AraError::InvalidArgument(
                "store filter count exceeds limit",
            ));
        }
        // SAFETY: forwarded nested reference-array contract.
        let sources = unsafe { ForeignSlice::copy_from_raw(source_pointer, source_count) }?;
        // SAFETY: same nested reference-array contract.
        let modifications =
            unsafe { ForeignSlice::copy_from_raw(modification_pointer, modification_count) }?;
        let mut builder = Self::builder(session).document_data(document_data);
        for reference in sources.as_slice() {
            builder = builder.audio_source(resolve_source(*reference)?);
        }
        for reference in modifications.as_slice() {
            builder = builder.audio_modification(resolve_modification(*reference)?);
        }
        builder.build().map(FilterSelection::Selected)
    }
}

/// Builder for [`StoreFilter`].
#[derive(Clone, Debug, Default)]
pub struct StoreFilterBuilder {
    session: RegistrySession,
    document_data: bool,
    audio_sources: Vec<Handle<AudioSourceKind>>,
    audio_modifications: Vec<Handle<AudioModificationKind>>,
}

impl StoreFilterBuilder {
    /// Selects whether private document data is stored.
    pub fn document_data(mut self, selected: bool) -> Self {
        self.document_data = selected;
        self
    }

    /// Adds an audio-source handle.
    pub fn audio_source(mut self, handle: Handle<AudioSourceKind>) -> Self {
        self.audio_sources.push(handle);
        self
    }

    /// Adds an audio-modification handle.
    pub fn audio_modification(mut self, handle: Handle<AudioModificationKind>) -> Self {
        self.audio_modifications.push(handle);
        self
    }

    /// Builds and validates session ownership and uniqueness.
    pub fn build(self) -> Result<StoreFilter, AraError> {
        validate_handles(self.session, &self.audio_sources)?;
        validate_handles(self.session, &self.audio_modifications)?;
        Ok(StoreFilter {
            session: self.session,
            document_data: self.document_data,
            audio_sources: self.audio_sources,
            audio_modifications: self.audio_modifications,
        })
    }
}

fn validate_handles<K>(session: RegistrySession, handles: &[Handle<K>]) -> Result<(), AraError> {
    let mut unique = HashSet::with_capacity(handles.len());
    for handle in handles {
        if handle.session() != session.get() {
            return Err(AraError::Archive(ArchiveError::InvalidFilter(
                "store handle belongs to another document session",
            )));
        }
        if !unique.insert(*handle) {
            return Err(AraError::Archive(ArchiveError::InvalidFilter(
                "store filter contains duplicate handles",
            )));
        }
    }
    Ok(())
}
