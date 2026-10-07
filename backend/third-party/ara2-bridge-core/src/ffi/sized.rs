//! Validation for ARA records led by a packed `structSize` field.

use crate::AraError;
use ara2_bridge_sys::{access, ARASize};
use std::marker::PhantomData;
use std::ptr::NonNull;

mod sealed {
    pub trait Sealed {}
}

/// Generated layout metadata for a versioned ARA record.
///
/// This trait is sealed because an incorrect implementation could make field validation disagree
/// with the C ABI. Implementations below only connect raw types to generated sys constants.
pub trait SizedRecord: Copy + sealed::Sealed {
    /// Minimum complete prefix accepted by the released header.
    const MIN_SIZE: usize;
    /// Complete field extents in C declaration order.
    const FIELD_EXTENTS: &'static [usize];
}

macro_rules! sized_records {
    ($(($record:ty, $minimum:path, $extents:path)),+ $(,)?) => {
        $(
            impl sealed::Sealed for $record {}
            impl SizedRecord for $record {
                const MIN_SIZE: usize = $minimum as usize;
                const FIELD_EXTENTS: &'static [usize] = $extents;
            }
        )+
    };
}

sized_records!(
    (
        ara2_bridge_sys::ARADocumentProperties,
        ara2_bridge_sys::kARADocumentPropertiesMinSize,
        ara2_bridge_sys::layout::ARADOCUMENT_PROPERTIES_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAMusicalContextProperties,
        ara2_bridge_sys::kARAMusicalContextPropertiesMinSize,
        ara2_bridge_sys::layout::ARAMUSICAL_CONTEXT_PROPERTIES_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARARegionSequenceProperties,
        ara2_bridge_sys::kARARegionSequencePropertiesMinSize,
        ara2_bridge_sys::layout::ARAREGION_SEQUENCE_PROPERTIES_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAAudioSourceProperties,
        ara2_bridge_sys::kARAAudioSourcePropertiesMinSize,
        ara2_bridge_sys::layout::ARAAUDIO_SOURCE_PROPERTIES_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAAudioModificationProperties,
        ara2_bridge_sys::kARAAudioModificationPropertiesMinSize,
        ara2_bridge_sys::layout::ARAAUDIO_MODIFICATION_PROPERTIES_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAPlaybackRegionProperties,
        ara2_bridge_sys::kARAPlaybackRegionPropertiesMinSize,
        ara2_bridge_sys::layout::ARAPLAYBACK_REGION_PROPERTIES_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAAudioAccessControllerInterface,
        ara2_bridge_sys::kARAAudioAccessControllerInterfaceMinSize,
        ara2_bridge_sys::layout::ARAAUDIO_ACCESS_CONTROLLER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAArchivingControllerInterface,
        ara2_bridge_sys::kARAArchivingControllerInterfaceMinSize,
        ara2_bridge_sys::layout::ARAARCHIVING_CONTROLLER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAContentAccessControllerInterface,
        ara2_bridge_sys::kARAContentAccessControllerInterfaceMinSize,
        ara2_bridge_sys::layout::ARACONTENT_ACCESS_CONTROLLER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAModelUpdateControllerInterface,
        ara2_bridge_sys::kARAModelUpdateControllerInterfaceMinSize,
        ara2_bridge_sys::layout::ARAMODEL_UPDATE_CONTROLLER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAPlaybackControllerInterface,
        ara2_bridge_sys::kARAPlaybackControllerInterfaceMinSize,
        ara2_bridge_sys::layout::ARAPLAYBACK_CONTROLLER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARADocumentControllerHostInstance,
        ara2_bridge_sys::kARADocumentControllerHostInstanceMinSize,
        ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_HOST_INSTANCE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARARestoreObjectsFilter,
        ara2_bridge_sys::kARARestoreObjectsFilterMinSize,
        ara2_bridge_sys::layout::ARARESTORE_OBJECTS_FILTER_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAStoreObjectsFilter,
        ara2_bridge_sys::kARAStoreObjectsFilterMinSize,
        ara2_bridge_sys::layout::ARASTORE_OBJECTS_FILTER_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAProcessingAlgorithmProperties,
        ara2_bridge_sys::kARAProcessingAlgorithmPropertiesMinSize,
        ara2_bridge_sys::layout::ARAPROCESSING_ALGORITHM_PROPERTIES_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARADocumentControllerInterface,
        ara2_bridge_sys::kARADocumentControllerInterfaceMinSize,
        ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARADocumentControllerInstance,
        ara2_bridge_sys::kARADocumentControllerInstanceMinSize,
        ara2_bridge_sys::layout::ARADOCUMENT_CONTROLLER_INSTANCE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAInterfaceConfiguration,
        ara2_bridge_sys::kARAInterfaceConfigurationMinSize,
        ara2_bridge_sys::layout::ARAINTERFACE_CONFIGURATION_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAFactory,
        ara2_bridge_sys::kARAFactoryMinSize,
        ara2_bridge_sys::layout::ARAFACTORY_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAPlaybackRendererInterface,
        ara2_bridge_sys::kARAPlaybackRendererInterfaceMinSize,
        ara2_bridge_sys::layout::ARAPLAYBACK_RENDERER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAEditorRendererInterface,
        ara2_bridge_sys::kARAEditorRendererInterfaceMinSize,
        ara2_bridge_sys::layout::ARAEDITOR_RENDERER_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAViewSelection,
        ara2_bridge_sys::kARAViewSelectionMinSize,
        ara2_bridge_sys::layout::ARAVIEW_SELECTION_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAEditorViewInterface,
        ara2_bridge_sys::kARAEditorViewInterfaceMinSize,
        ara2_bridge_sys::layout::ARAEDITOR_VIEW_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAPlugInExtensionInterface,
        ara2_bridge_sys::kARAPlugInExtensionInterfaceMinSize,
        ara2_bridge_sys::layout::ARAPLUG_IN_EXTENSION_INTERFACE_FIELD_EXTENTS
    ),
    (
        ara2_bridge_sys::ARAPlugInExtensionInstance,
        ara2_bridge_sys::kARAPlugInExtensionInstanceMinSize,
        ara2_bridge_sys::layout::ARAPLUG_IN_EXTENSION_INSTANCE_FIELD_EXTENTS
    ),
);

/// A validated view of caller-owned versioned ARA storage.
///
/// The view never creates a Rust reference to the packed record or its fields. Its lifetime tracks
/// the caller-owned storage, while individual fields are copied with unaligned generated access.
pub struct SizedInput<'a, T: SizedRecord> {
    base: NonNull<u8>,
    advertised_size: usize,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T: SizedRecord> SizedInput<'a, T> {
    /// Validates a foreign ARA record and its complete represented prefix.
    ///
    /// # Safety
    ///
    /// `pointer` must be non-null and readable for at least one `ARASize`. It must then remain
    /// readable and initialized for the complete byte extent advertised by that leading value for
    /// lifetime `'a`. The storage may be unaligned. This function cannot prove OS-level pointer
    /// readability; violating the caller precondition is undefined behavior before value validation.
    pub unsafe fn from_ptr(pointer: *const T) -> Result<Self, AraError> {
        let base = NonNull::new(pointer.cast_mut().cast::<u8>())
            .ok_or(AraError::InvalidArgument("null sized-struct pointer"))?;
        // SAFETY: the caller guarantees at least the leading `ARASize` is readable and initialized.
        let advertised_size = unsafe { access::read_field::<ARASize>(base.as_ptr(), 0) };
        if advertised_size < T::MIN_SIZE {
            return Err(AraError::Abi("struct too small"));
        }
        let known_extent = T::FIELD_EXTENTS
            .last()
            .copied()
            .ok_or(AraError::Abi("sized record has no generated field extents"))?;
        if advertised_size <= known_extent && !T::FIELD_EXTENTS.contains(&advertised_size) {
            return Err(AraError::Abi("struct ends inside a field"));
        }
        Ok(Self {
            base,
            advertised_size,
            _borrow: PhantomData,
        })
    }

    /// Returns the peer-advertised byte extent.
    pub const fn advertised_size(&self) -> usize {
        self.advertised_size
    }

    /// Returns whether a field with the given generated complete extent is present.
    pub const fn contains_extent(&self, complete_extent: usize) -> bool {
        self.advertised_size >= complete_extent
    }

    /// Copies one represented field without borrowing potentially unaligned storage.
    ///
    /// # Safety
    ///
    /// `offset`, `complete_extent`, and `F` must describe the same field of `T` using the generated
    /// sys layout metadata. Construction already carries the foreign-storage readability
    /// precondition; this method additionally relies on the field being initialized for `F`.
    pub unsafe fn copy_field<F: Copy>(
        &self,
        offset: usize,
        complete_extent: usize,
    ) -> Result<F, AraError> {
        if !self.contains_extent(complete_extent) {
            return Err(AraError::Abi("field is outside represented prefix"));
        }
        // SAFETY: construction validates the caller-owned advertised storage. The internal caller
        // supplies the generated offset, matching field type, and complete field extent.
        Ok(unsafe { access::read_field(self.base.as_ptr(), offset) })
    }
}
