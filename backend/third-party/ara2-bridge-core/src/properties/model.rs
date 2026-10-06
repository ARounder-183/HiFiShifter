//! 拥有型模型属性与稳定存储；仅宿主音频对象输入兼容UTF-8 ID，保留标准输出校验。

use super::{
    copy_optional_display, copy_host_object_id, display_string, persistent_id, write_raw, zeroed_raw,
    FfiProperties,
};
use crate::{ApiGeneration, AraBool, AraError, ForeignSlice, ModelRef, SizedInput};
use ara2_bridge_sys::{
    access, layout, ARAAudioModificationProperties, ARAAudioSourceProperties,
    ARAChannelArrangementDataType, ARAColor, ARAInt32, ARAMusicalContextProperties,
    ARAMusicalContextRef, ARAPersistentID, ARAPlaybackRegionProperties,
    ARAPlaybackTransformationFlags, ARARegionSequenceProperties, ARARegionSequenceRef,
    ARASampleCount, ARASampleRate, ARASize, ARATimeDuration, ARATimePosition, ARAUtf8String,
};
use std::ffi::CString;
use std::mem::{offset_of, size_of};
use std::pin::Pin;

/// Marker kind for audio-source handles and references.
pub enum AudioSourceKind {}
/// Marker kind for audio-modification handles and references.
pub enum AudioModificationKind {}
/// Marker kind for musical-context handles and references.
pub enum MusicalContextKind {}
/// Marker kind for region-sequence handles and references.
pub enum RegionSequenceKind {}
/// Marker kind for playback-region handles and references.
pub enum PlaybackRegionKind {}

/// Validated RGB color with components in the inclusive range `0.0..=1.0`.
#[derive(Clone, Debug)]
pub struct Color {
    raw: Box<ARAColor>,
}

impl Color {
    /// Creates a finite ARA color.
    pub fn new(red: f32, green: f32, blue: f32) -> Result<Self, AraError> {
        if ![red, green, blue]
            .into_iter()
            .all(|component| component.is_finite() && (0.0..=1.0).contains(&component))
        {
            return Err(AraError::InvalidArgument(
                "color components must be finite and between zero and one",
            ));
        }
        // SAFETY: all fields of `ARAColor` are floats and accept the zero bit pattern.
        let mut raw = unsafe { zeroed_raw::<ARAColor>() };
        // SAFETY: offsets and field types come directly from the generated raw record.
        unsafe {
            write_raw(&mut raw, offset_of!(ARAColor, r), red);
            write_raw(&mut raw, offset_of!(ARAColor, g), green);
            write_raw(&mut raw, offset_of!(ARAColor, b), blue);
        }
        Ok(Self { raw: Box::new(raw) })
    }

    /// Returns the red component.
    pub fn red(&self) -> f32 {
        // SAFETY: `self.raw` is initialized and the generated offset/type match.
        unsafe { access::read_field(self.as_ptr().cast(), offset_of!(ARAColor, r)) }
    }

    /// Returns the green component.
    pub fn green(&self) -> f32 {
        // SAFETY: `self.raw` is initialized and the generated offset/type match.
        unsafe { access::read_field(self.as_ptr().cast(), offset_of!(ARAColor, g)) }
    }

    /// Returns the blue component.
    pub fn blue(&self) -> f32 {
        // SAFETY: `self.raw` is initialized and the generated offset/type match.
        unsafe { access::read_field(self.as_ptr().cast(), offset_of!(ARAColor, b)) }
    }

    pub(crate) fn as_ptr(&self) -> *const ARAColor {
        self.raw.as_ref()
    }

    unsafe fn copy_from_ptr(pointer: *const ARAColor) -> Result<Self, AraError> {
        if pointer.is_null() {
            return Err(AraError::InvalidArgument("null color pointer"));
        }
        if (pointer as usize) % std::mem::align_of::<ARAColor>() != 0 {
            return Err(AraError::InvalidArgument("misaligned color pointer"));
        }
        let base = pointer.cast::<u8>();
        // SAFETY: the enclosing property-copy contract guarantees readable nested color storage.
        let red = unsafe { access::read_field(base, offset_of!(ARAColor, r)) };
        // SAFETY: same readable color allocation and generated field metadata.
        let green = unsafe { access::read_field(base, offset_of!(ARAColor, g)) };
        // SAFETY: same readable color allocation and generated field metadata.
        let blue = unsafe { access::read_field(base, offset_of!(ARAColor, b)) };
        Self::new(red, green, blue)
    }
}

/// Opaque channel-arrangement bytes with at least eight-byte-stable alignment.
#[derive(Clone, Debug)]
pub struct RawChannelArrangement {
    data_type: ARAChannelArrangementDataType,
    words: Box<[u64]>,
    byte_len: usize,
}

impl RawChannelArrangement {
    /// Copies raw companion arrangement bytes into stable aligned storage.
    pub fn new(data_type: ARAChannelArrangementDataType, bytes: &[u8]) -> Result<Self, AraError> {
        if data_type == 0 || bytes.is_empty() {
            return Err(AraError::InvalidArgument(
                "defined channel arrangement requires bytes",
            ));
        }
        let words_len = bytes
            .len()
            .checked_add(size_of::<u64>() - 1)
            .and_then(|length| length.checked_div(size_of::<u64>()))
            .ok_or(AraError::InvalidArgument(
                "channel arrangement extent overflow",
            ))?;
        let mut words = vec![0_u64; words_len].into_boxed_slice();
        // SAFETY: `words` exposes at least `bytes.len()` writable bytes and the regions do not
        // overlap. Both allocations remain live for the copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                words.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(Self {
            data_type,
            words,
            byte_len: bytes.len(),
        })
    }

    /// Returns the raw ARA companion data-type value.
    pub const fn data_type(&self) -> ARAChannelArrangementDataType {
        self.data_type
    }

    /// Returns the copied arrangement bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the allocation contains `byte_len` initialized bytes and remains live with self.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast(), self.byte_len) }
    }

    fn as_ptr(&self) -> *const std::os::raw::c_void {
        self.words.as_ptr().cast()
    }
}

/// Owned musical-context display properties.
#[derive(Clone, Debug)]
pub struct MusicalContextProperties {
    name: Option<CString>,
    order_index: ARAInt32,
    color: Option<Color>,
}

impl MusicalContextProperties {
    /// Creates musical-context properties.
    pub fn new(
        name: Option<&str>,
        order_index: ARAInt32,
        color: Option<Color>,
    ) -> Result<Self, AraError> {
        Ok(Self {
            name: display_string(name)?,
            order_index,
            color,
        })
    }

    /// Copies an ephemeral packed musical-context property record.
    ///
    /// # Safety
    ///
    /// The record, advertised prefix, and represented nested name/color pointers must remain
    /// readable and initialized for this call.
    pub unsafe fn copy_from_ffi(
        pointer: *const ARAMusicalContextProperties,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded caller-valid storage contract.
        let input = unsafe { SizedInput::from_ptr(pointer)? };
        let name = if input.contains_extent(layout::ARAMUSICAL_CONTEXT_PROPERTIES_NAME) {
            // SAFETY: generated offset/type/extent match the record.
            let pointer = unsafe {
                input.copy_field::<ARAUtf8String>(
                    offset_of!(ARAMusicalContextProperties, name),
                    layout::ARAMUSICAL_CONTEXT_PROPERTIES_NAME,
                )?
            };
            // SAFETY: the outer contract covers the represented nested string.
            unsafe { copy_optional_display(pointer)? }
        } else {
            None
        };
        let order_index =
            if input.contains_extent(layout::ARAMUSICAL_CONTEXT_PROPERTIES_ORDER_INDEX) {
                // SAFETY: generated offset/type/extent match the record.
                unsafe {
                    input.copy_field::<ARAInt32>(
                        offset_of!(ARAMusicalContextProperties, orderIndex),
                        layout::ARAMUSICAL_CONTEXT_PROPERTIES_ORDER_INDEX,
                    )?
                }
            } else {
                0
            };
        let color = if input.contains_extent(layout::ARAMUSICAL_CONTEXT_PROPERTIES_COLOR) {
            // SAFETY: generated offset/type/extent match the record.
            let pointer = unsafe {
                input.copy_field::<*const ARAColor>(
                    offset_of!(ARAMusicalContextProperties, color),
                    layout::ARAMUSICAL_CONTEXT_PROPERTIES_COLOR,
                )?
            };
            if pointer.is_null() {
                None
            } else {
                // SAFETY: the outer contract covers represented nested color storage.
                Some(unsafe { Color::copy_from_ptr(pointer)? })
            }
        } else {
            None
        };
        Ok(Self {
            name,
            order_index,
            color,
        })
    }

    /// Returns the optional display name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(|value| {
            value
                .to_str()
                .expect("musical-context names originate from UTF-8")
        })
    }

    /// Returns the host-defined ordering index.
    pub const fn order_index(&self) -> ARAInt32 {
        self.order_index
    }

    /// Returns the optional color.
    pub const fn color(&self) -> Option<&Color> {
        self.color.as_ref()
    }

    /// Builds a pinned generation-specific raw record.
    pub fn as_ffi(
        &self,
        generation: ApiGeneration,
    ) -> Result<Pin<Box<FfiProperties<'_, ARAMusicalContextProperties>>>, AraError> {
        ensure_generation(generation)?;
        let full = generation >= ApiGeneration::V2Draft;
        let represented = if full {
            layout::ARAMUSICAL_CONTEXT_PROPERTIES_COLOR
        } else {
            layout::ARAMUSICAL_CONTEXT_PROPERTIES_STRUCT_SIZE
        };
        // SAFETY: all raw fields accept zero.
        let mut raw = unsafe { zeroed_raw::<ARAMusicalContextProperties>() };
        // SAFETY: generated offsets and matching types.
        unsafe {
            write_raw(
                &mut raw,
                offset_of!(ARAMusicalContextProperties, structSize),
                represented as ARASize,
            );
            if full {
                write_raw(
                    &mut raw,
                    offset_of!(ARAMusicalContextProperties, name),
                    self.name
                        .as_ref()
                        .map_or(std::ptr::null(), |value| value.as_ptr()),
                );
                write_raw(
                    &mut raw,
                    offset_of!(ARAMusicalContextProperties, orderIndex),
                    self.order_index,
                );
                write_raw(
                    &mut raw,
                    offset_of!(ARAMusicalContextProperties, color),
                    self.color.as_ref().map_or(std::ptr::null(), Color::as_ptr),
                );
            }
        }
        Ok(FfiProperties::pin(raw))
    }
}

/// Owned audio-source properties.
#[derive(Clone, Debug)]
pub struct AudioSourceProperties {
    name: Option<CString>,
    persistent_id: CString,
    sample_count: ARASampleCount,
    sample_rate: ARASampleRate,
    channel_count: ARAInt32,
    merits_64_bit_samples: AraBool,
    channel_arrangement: Option<RawChannelArrangement>,
}

impl AudioSourceProperties {
    /// Creates validated audio-source properties without a companion channel arrangement.
    pub fn new(
        name: Option<&str>,
        persistent_id_value: &str,
        sample_count: ARASampleCount,
        sample_rate: ARASampleRate,
        channel_count: ARAInt32,
        merits_64_bit_samples: AraBool,
    ) -> Result<Self, AraError> {
        validate_audio_shape(sample_count, sample_rate, channel_count)?;
        Ok(Self {
            name: display_string(name)?,
            persistent_id: persistent_id(persistent_id_value)?,
            sample_count,
            sample_rate,
            channel_count,
            merits_64_bit_samples,
            channel_arrangement: None,
        })
    }

    /// Attaches copied raw companion channel-arrangement storage.
    pub fn with_channel_arrangement(mut self, arrangement: RawChannelArrangement) -> Self {
        self.channel_arrangement = Some(arrangement);
        self
    }

    /// Copies an ephemeral packed ARA audio-source record.
    ///
    /// # Safety
    ///
    /// The record, its advertised prefix, and all represented nested pointers must remain readable
    /// and initialized for this call. Non-undefined arrangements must point to the companion shape
    /// implied by their data type.
    pub unsafe fn copy_from_ffi(
        pointer: *const ARAAudioSourceProperties,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded caller-valid storage contract.
        let input = unsafe { SizedInput::from_ptr(pointer)? };
        macro_rules! field {
            ($type:ty, $name:ident, $extent:path) => {{
                // SAFETY: the offset, extent, and field type are generated from this record.
                unsafe {
                    input
                        .copy_field::<$type>(offset_of!(ARAAudioSourceProperties, $name), $extent)?
                }
            }};
        }
        let name = field!(ARAUtf8String, name, layout::ARAAUDIO_SOURCE_PROPERTIES_NAME);
        let id = field!(
            ARAPersistentID,
            persistentID,
            layout::ARAAUDIO_SOURCE_PROPERTIES_PERSISTENT_ID
        );
        let sample_count = field!(
            ARASampleCount,
            sampleCount,
            layout::ARAAUDIO_SOURCE_PROPERTIES_SAMPLE_COUNT
        );
        let sample_rate = field!(
            ARASampleRate,
            sampleRate,
            layout::ARAAUDIO_SOURCE_PROPERTIES_SAMPLE_RATE
        );
        let channel_count = field!(
            ARAInt32,
            channelCount,
            layout::ARAAUDIO_SOURCE_PROPERTIES_CHANNEL_COUNT
        );
        let merits = field!(
            ara2_bridge_sys::ARABool,
            merits64BitSamples,
            layout::ARAAUDIO_SOURCE_PROPERTIES_MERITS64_BIT_SAMPLES
        );
        validate_audio_shape(sample_count, sample_rate, channel_count)?;
        let channel_arrangement =
            if input.contains_extent(layout::ARAAUDIO_SOURCE_PROPERTIES_CHANNEL_ARRANGEMENT) {
                let data_type = field!(
                    ARAChannelArrangementDataType,
                    channelArrangementDataType,
                    layout::ARAAUDIO_SOURCE_PROPERTIES_CHANNEL_ARRANGEMENT_DATA_TYPE
                );
                let arrangement_pointer = field!(
                    *const std::os::raw::c_void,
                    channelArrangement,
                    layout::ARAAUDIO_SOURCE_PROPERTIES_CHANNEL_ARRANGEMENT
                );
                // SAFETY: the caller contract covers the nested companion storage selected by type.
                unsafe { copy_arrangement(data_type, arrangement_pointer, channel_count)? }
            } else {
                None
            };
        // SAFETY: the caller contract covers the nested ephemeral display string.
        let name = unsafe { copy_optional_display(name)? };
        // SAFETY: the caller contract covers the nested ephemeral persistent ID.
        let persistent_id = unsafe { copy_host_object_id(id)? };
        Ok(Self {
            name,
            persistent_id,
            sample_count,
            sample_rate,
            channel_count,
            merits_64_bit_samples: AraBool::from_raw(merits),
            channel_arrangement,
        })
    }

    /// Returns the optional display name.
    pub fn name(&self) -> Option<&str> {
        self.name
            .as_ref()
            .map(|value| value.to_str().expect("validated UTF-8"))
    }
    /// Returns the persistent ID.
    pub fn persistent_id(&self) -> &str {
        self.persistent_id.to_str().expect("validated UTF-8")
    }
    /// Returns the per-channel sample count.
    pub const fn sample_count(&self) -> ARASampleCount {
        self.sample_count
    }
    /// Returns the sample rate in hertz.
    pub const fn sample_rate(&self) -> ARASampleRate {
        self.sample_rate
    }
    /// Returns the channel count.
    pub const fn channel_count(&self) -> ARAInt32 {
        self.channel_count
    }
    /// Returns whether lossless access merits 64-bit samples.
    pub const fn merits_64_bit_samples(&self) -> bool {
        self.merits_64_bit_samples.get()
    }
    /// Returns the optional raw companion channel arrangement.
    pub const fn channel_arrangement(&self) -> Option<&RawChannelArrangement> {
        self.channel_arrangement.as_ref()
    }

    /// Builds a pinned generation-specific raw record.
    pub fn as_ffi(
        &self,
        generation: ApiGeneration,
    ) -> Result<Pin<Box<FfiProperties<'_, ARAAudioSourceProperties>>>, AraError> {
        ensure_generation(generation)?;
        let has_arrangement_tail = generation >= ApiGeneration::V2Final;
        let represented = if has_arrangement_tail {
            layout::ARAAUDIO_SOURCE_PROPERTIES_CHANNEL_ARRANGEMENT
        } else {
            layout::ARAAUDIO_SOURCE_PROPERTIES_MERITS64_BIT_SAMPLES
        };
        // SAFETY: all raw fields accept zero.
        let mut raw = unsafe { zeroed_raw::<ARAAudioSourceProperties>() };
        // SAFETY: generated offsets and matching raw types.
        unsafe {
            write_raw(
                &mut raw,
                offset_of!(ARAAudioSourceProperties, structSize),
                represented as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioSourceProperties, name),
                self.name
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioSourceProperties, persistentID),
                self.persistent_id.as_ptr(),
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioSourceProperties, sampleCount),
                self.sample_count,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioSourceProperties, sampleRate),
                self.sample_rate,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioSourceProperties, channelCount),
                self.channel_count,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioSourceProperties, merits64BitSamples),
                self.merits_64_bit_samples.into_raw(),
            );
            if has_arrangement_tail {
                write_raw(
                    &mut raw,
                    offset_of!(ARAAudioSourceProperties, channelArrangementDataType),
                    self.channel_arrangement
                        .as_ref()
                        .map_or(0, RawChannelArrangement::data_type),
                );
                write_raw(
                    &mut raw,
                    offset_of!(ARAAudioSourceProperties, channelArrangement),
                    self.channel_arrangement
                        .as_ref()
                        .map_or(std::ptr::null(), RawChannelArrangement::as_ptr),
                );
            }
        }
        Ok(FfiProperties::pin(raw))
    }
}

/// Owned audio-modification display and persistent identity.
#[derive(Clone, Debug)]
pub struct AudioModificationProperties {
    name: Option<CString>,
    persistent_id: CString,
}

impl AudioModificationProperties {
    /// Creates audio-modification properties.
    pub fn new(name: Option<&str>, persistent_id_value: &str) -> Result<Self, AraError> {
        Ok(Self {
            name: display_string(name)?,
            persistent_id: persistent_id(persistent_id_value)?,
        })
    }
    /// Copies an ephemeral packed audio-modification record.
    ///
    /// # Safety
    ///
    /// The record, advertised prefix, and represented nested strings must remain readable and
    /// initialized for this call.
    pub unsafe fn copy_from_ffi(
        pointer: *const ARAAudioModificationProperties,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded caller-valid storage contract.
        let input = unsafe { SizedInput::from_ptr(pointer)? };
        // SAFETY: generated field metadata matches the property record.
        let name = unsafe {
            input.copy_field::<ARAUtf8String>(
                offset_of!(ARAAudioModificationProperties, name),
                layout::ARAAUDIO_MODIFICATION_PROPERTIES_NAME,
            )?
        };
        // SAFETY: generated field metadata matches the property record.
        let id = unsafe {
            input.copy_field::<ARAPersistentID>(
                offset_of!(ARAAudioModificationProperties, persistentID),
                layout::ARAAUDIO_MODIFICATION_PROPERTIES_PERSISTENT_ID,
            )?
        };
        // SAFETY: the outer contract covers the nested display string.
        let name = unsafe { copy_optional_display(name)? };
        // SAFETY: the outer contract covers the nested persistent ID.
        let persistent_id = unsafe { copy_host_object_id(id)? };
        Ok(Self {
            name,
            persistent_id,
        })
    }
    /// Returns the optional display name.
    pub fn name(&self) -> Option<&str> {
        self.name
            .as_ref()
            .map(|v| v.to_str().expect("validated UTF-8"))
    }
    /// Returns the persistent ID.
    pub fn persistent_id(&self) -> &str {
        self.persistent_id.to_str().expect("validated UTF-8")
    }
    /// Builds a pinned raw record.
    pub fn as_ffi(&self) -> Pin<Box<FfiProperties<'_, ARAAudioModificationProperties>>> {
        // SAFETY: all fields accept zero.
        let mut raw = unsafe { zeroed_raw::<ARAAudioModificationProperties>() };
        // SAFETY: generated offsets and matching types.
        unsafe {
            write_raw(
                &mut raw,
                offset_of!(ARAAudioModificationProperties, structSize),
                layout::ARAAUDIO_MODIFICATION_PROPERTIES_PERSISTENT_ID as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioModificationProperties, name),
                self.name.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
            );
            write_raw(
                &mut raw,
                offset_of!(ARAAudioModificationProperties, persistentID),
                self.persistent_id.as_ptr(),
            );
        }
        FfiProperties::pin(raw)
    }
}

/// Owned region-sequence properties using a typed musical-context reference.
#[derive(Clone, Debug)]
pub struct RegionSequenceProperties {
    name: Option<CString>,
    order_index: ARAInt32,
    musical_context: ModelRef<MusicalContextKind>,
    color: Option<Color>,
}

impl RegionSequenceProperties {
    /// Creates region-sequence properties.
    pub fn new(
        name: Option<&str>,
        order_index: ARAInt32,
        musical_context: ModelRef<MusicalContextKind>,
        color: Option<Color>,
    ) -> Result<Self, AraError> {
        Ok(Self {
            name: display_string(name)?,
            order_index,
            musical_context,
            color,
        })
    }
    /// Copies region-sequence properties using a runtime-provided checked reference resolver.
    ///
    /// # Safety
    ///
    /// The record, advertised prefix, and represented nested pointers must remain readable and
    /// initialized for this call. `resolve_context` must reject pointers outside the owning session.
    pub unsafe fn copy_from_ffi_with_context(
        pointer: *const ARARegionSequenceProperties,
        resolve_context: impl FnOnce(
            ARAMusicalContextRef,
        ) -> Result<ModelRef<MusicalContextKind>, AraError>,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded caller-valid storage contract.
        let input = unsafe { SizedInput::from_ptr(pointer)? };
        // SAFETY: generated field metadata matches this record.
        let name = unsafe {
            input.copy_field::<ARAUtf8String>(
                offset_of!(ARARegionSequenceProperties, name),
                layout::ARAREGION_SEQUENCE_PROPERTIES_NAME,
            )?
        };
        // SAFETY: generated field metadata matches this record.
        let order_index = unsafe {
            input.copy_field::<ARAInt32>(
                offset_of!(ARARegionSequenceProperties, orderIndex),
                layout::ARAREGION_SEQUENCE_PROPERTIES_ORDER_INDEX,
            )?
        };
        // SAFETY: generated field metadata matches this record.
        let context = unsafe {
            input.copy_field::<ARAMusicalContextRef>(
                offset_of!(ARARegionSequenceProperties, musicalContextRef),
                layout::ARAREGION_SEQUENCE_PROPERTIES_MUSICAL_CONTEXT_REF,
            )?
        };
        if context.is_null() {
            return Err(AraError::InvalidArgument(
                "region sequence requires a musical context",
            ));
        }
        let color = if input.contains_extent(layout::ARAREGION_SEQUENCE_PROPERTIES_COLOR) {
            // SAFETY: generated field metadata matches this record.
            let color = unsafe {
                input.copy_field::<*const ARAColor>(
                    offset_of!(ARARegionSequenceProperties, color),
                    layout::ARAREGION_SEQUENCE_PROPERTIES_COLOR,
                )?
            };
            if color.is_null() {
                None
            } else {
                // SAFETY: the outer contract covers represented nested color storage.
                Some(unsafe { Color::copy_from_ptr(color)? })
            }
        } else {
            None
        };
        // SAFETY: the outer contract covers the nested display string.
        let name = unsafe { copy_optional_display(name)? };
        Ok(Self {
            name,
            order_index,
            musical_context: resolve_context(context)?,
            color,
        })
    }
    /// Builds a pinned raw record for an ARA2 generation.
    pub fn as_ffi(
        &self,
        generation: ApiGeneration,
    ) -> Result<Pin<Box<FfiProperties<'_, ARARegionSequenceProperties>>>, AraError> {
        ensure_generation(generation)?;
        if generation < ApiGeneration::V2Draft {
            return Err(AraError::Unsupported(
                "region sequences are unavailable in ARA1",
            ));
        }
        let represented = if self.color.is_some() {
            layout::ARAREGION_SEQUENCE_PROPERTIES_COLOR
        } else {
            layout::ARAREGION_SEQUENCE_PROPERTIES_MUSICAL_CONTEXT_REF
        };
        // SAFETY: all fields accept zero.
        let mut raw = unsafe { zeroed_raw::<ARARegionSequenceProperties>() };
        // SAFETY: generated offsets and matching types.
        unsafe {
            write_raw(
                &mut raw,
                offset_of!(ARARegionSequenceProperties, structSize),
                represented as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARARegionSequenceProperties, name),
                self.name.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
            );
            write_raw(
                &mut raw,
                offset_of!(ARARegionSequenceProperties, orderIndex),
                self.order_index,
            );
            write_raw(
                &mut raw,
                offset_of!(ARARegionSequenceProperties, musicalContextRef),
                self.musical_context
                    .as_raw()
                    .cast::<ara2_bridge_sys::ARAMusicalContextRefMarkupType>()
                    as ARAMusicalContextRef,
            );
            write_raw(
                &mut raw,
                offset_of!(ARARegionSequenceProperties, color),
                self.color.as_ref().map_or(std::ptr::null(), Color::as_ptr),
            );
        }
        Ok(FfiProperties::pin(raw))
    }

    /// Returns the optional display name.
    pub fn name(&self) -> Option<&str> {
        self.name
            .as_ref()
            .map(|value| value.to_str().expect("validated UTF-8"))
    }

    /// Returns the ordering index.
    pub const fn order_index(&self) -> ARAInt32 {
        self.order_index
    }

    /// Returns the typed musical-context reference.
    pub const fn musical_context(&self) -> ModelRef<MusicalContextKind> {
        self.musical_context
    }

    /// Returns the optional display color.
    pub const fn color(&self) -> Option<&Color> {
        self.color.as_ref()
    }
}

/// Owned playback-region mapping and generation-specific graph reference.
#[derive(Clone, Debug)]
pub struct PlaybackRegionProperties {
    transformation_flags: ARAPlaybackTransformationFlags,
    start_in_modification: ARATimePosition,
    duration_in_modification: ARATimeDuration,
    start_in_playback: ARATimePosition,
    duration_in_playback: ARATimeDuration,
    musical_context: Option<ModelRef<MusicalContextKind>>,
    region_sequence: Option<ModelRef<RegionSequenceKind>>,
    name: Option<CString>,
    color: Option<Color>,
}

impl PlaybackRegionProperties {
    /// Creates a legacy ARA1 playback-region mapping.
    #[allow(clippy::too_many_arguments)]
    pub fn for_ara1(
        flags: ARAPlaybackTransformationFlags,
        modification_start: f64,
        modification_duration: f64,
        playback_start: f64,
        playback_duration: f64,
        musical_context: ModelRef<MusicalContextKind>,
        name: Option<&str>,
        color: Option<Color>,
    ) -> Result<Self, AraError> {
        validate_playback(
            flags,
            modification_start,
            modification_duration,
            playback_start,
            playback_duration,
        )?;
        Ok(Self {
            transformation_flags: flags,
            start_in_modification: modification_start,
            duration_in_modification: modification_duration,
            start_in_playback: playback_start,
            duration_in_playback: playback_duration,
            musical_context: Some(musical_context),
            region_sequence: None,
            name: display_string(name)?,
            color,
        })
    }

    /// Creates an ARA2 playback-region mapping.
    #[allow(clippy::too_many_arguments)]
    pub fn for_ara2(
        flags: ARAPlaybackTransformationFlags,
        modification_start: f64,
        modification_duration: f64,
        playback_start: f64,
        playback_duration: f64,
        region_sequence: ModelRef<RegionSequenceKind>,
        name: Option<&str>,
        color: Option<Color>,
    ) -> Result<Self, AraError> {
        validate_playback(
            flags,
            modification_start,
            modification_duration,
            playback_start,
            playback_duration,
        )?;
        Ok(Self {
            transformation_flags: flags,
            start_in_modification: modification_start,
            duration_in_modification: modification_duration,
            start_in_playback: playback_start,
            duration_in_playback: playback_duration,
            musical_context: None,
            region_sequence: Some(region_sequence),
            name: display_string(name)?,
            color,
        })
    }

    /// Returns the required ARA 2 region-sequence reference, if this is an ARA 2 mapping.
    pub const fn region_sequence(&self) -> Option<ModelRef<RegionSequenceKind>> {
        self.region_sequence
    }

    /// Replaces the ARA 2 graph reference while preserving the playback mapping.
    ///
    /// This is primarily useful when forwarding host-owned properties to a foreign ARA peer,
    /// whose object reference must replace the host reference at the ABI boundary.
    pub fn with_region_sequence(
        mut self,
        region_sequence: ModelRef<RegionSequenceKind>,
    ) -> Result<Self, AraError> {
        if self.region_sequence.is_none() {
            return Err(AraError::InvalidArgument(
                "cannot assign a region sequence to ARA1 playback properties",
            ));
        }
        self.region_sequence = Some(region_sequence);
        Ok(self)
    }

    /// Returns the required ARA 1 musical-context reference, if this is an ARA 1 mapping.
    pub const fn musical_context(&self) -> Option<ModelRef<MusicalContextKind>> {
        self.musical_context
    }

    /// Replaces the ARA 1 graph reference while preserving the playback mapping.
    ///
    /// This is primarily useful when forwarding host-owned properties to a foreign ARA peer,
    /// whose object reference must replace the host reference at the ABI boundary.
    pub fn with_musical_context(
        mut self,
        musical_context: ModelRef<MusicalContextKind>,
    ) -> Result<Self, AraError> {
        if self.musical_context.is_none() {
            return Err(AraError::InvalidArgument(
                "cannot assign a musical context to ARA2 playback properties",
            ));
        }
        self.musical_context = Some(musical_context);
        Ok(self)
    }

    /// Returns the requested playback transformation flags, retaining future bits.
    pub const fn transformation_flags(&self) -> ARAPlaybackTransformationFlags {
        self.transformation_flags
    }

    // ── 本地补丁（HiFiShifter）──────────────────────────────────────────────
    //
    // 【为什么打了这几行】上游 0.3.0 的 `PlaybackRegionProperties` **只有
    // `transformation_flags()` 一个公开读访问器**：区间的起点/长度/名字四个字段
    // 都是私有的，也没有 getter。而 ARA→时间线映射的全部输入正是这四个数
    //（placement 与拉伸）。没有它们，插件拿到的模型只剩计数，无法建时间线。
    //
    // 这不是"用得别扭"，是**信息拿不到**：高层 `PluginModel` 的
    // `PlaybackRegions::create_playback_region` 只交出这个不透明结构。所以要么
    // 给上游补 getter，要么自己重写一整层 ARA 文档控制器委托。这里选前者。
    //
    // 补丁是**纯增量**的（只加读访问器，不动任何现有行为），因此
    // `[patch.crates-io]` 之后其余 ara2-bridge crate 的行为与本机未打补丁时一致。
    // 上游若补了同名 getter，删掉这一段即可。

    /// 播放区间在 **modification** 时间轴上的起点（秒）。
    pub const fn start_in_modification_time(&self) -> f64 {
        self.start_in_modification
    }

    /// 播放区间在 **modification** 时间轴上的长度（秒）。
    pub const fn duration_in_modification_time(&self) -> f64 {
        self.duration_in_modification
    }

    /// 播放区间在**工程**时间轴上的起点（秒）。
    pub const fn start_in_playback_time(&self) -> f64 {
        self.start_in_playback
    }

    /// 播放区间在**工程**时间轴上的长度（秒）。
    pub const fn duration_in_playback_time(&self) -> f64 {
        self.duration_in_playback
    }

    /// 宿主给这个播放区间起的名字（未命名时为 `None`）。
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().and_then(|value| value.to_str().ok())
    }

    /// Copies playback-region properties using checked runtime reference resolvers.
    ///
    /// # Safety
    ///
    /// The record, advertised prefix, and represented nested pointers must remain readable and
    /// initialized for this call. Each resolver must reject pointers outside the owning session.
    pub unsafe fn copy_from_ffi_with_refs(
        pointer: *const ARAPlaybackRegionProperties,
        generation: ApiGeneration,
        resolve_context: impl Fn(ARAMusicalContextRef) -> Result<ModelRef<MusicalContextKind>, AraError>,
        resolve_sequence: impl Fn(
            ARARegionSequenceRef,
        ) -> Result<ModelRef<RegionSequenceKind>, AraError>,
    ) -> Result<Self, AraError> {
        ensure_generation(generation)?;
        // SAFETY: forwarded caller-valid storage contract.
        let input = unsafe { SizedInput::from_ptr(pointer)? };
        macro_rules! field {
            ($type:ty, $name:ident, $extent:path) => {{
                // SAFETY: generated offset/type/extent match this record.
                unsafe {
                    input.copy_field::<$type>(
                        offset_of!(ARAPlaybackRegionProperties, $name),
                        $extent,
                    )?
                }
            }};
        }
        let flags = field!(
            ARAPlaybackTransformationFlags,
            transformationFlags,
            layout::ARAPLAYBACK_REGION_PROPERTIES_TRANSFORMATION_FLAGS
        );
        let modification_start = field!(
            ARATimePosition,
            startInModificationTime,
            layout::ARAPLAYBACK_REGION_PROPERTIES_START_IN_MODIFICATION_TIME
        );
        let modification_duration = field!(
            ARATimeDuration,
            durationInModificationTime,
            layout::ARAPLAYBACK_REGION_PROPERTIES_DURATION_IN_MODIFICATION_TIME
        );
        let playback_start = field!(
            ARATimePosition,
            startInPlaybackTime,
            layout::ARAPLAYBACK_REGION_PROPERTIES_START_IN_PLAYBACK_TIME
        );
        let playback_duration = field!(
            ARATimeDuration,
            durationInPlaybackTime,
            layout::ARAPLAYBACK_REGION_PROPERTIES_DURATION_IN_PLAYBACK_TIME
        );
        validate_playback(
            flags,
            modification_start,
            modification_duration,
            playback_start,
            playback_duration,
        )?;
        let ara2 = generation >= ApiGeneration::V2Draft;
        let (musical_context, region_sequence) = if ara2 {
            let reference = field!(
                ARARegionSequenceRef,
                regionSequenceRef,
                layout::ARAPLAYBACK_REGION_PROPERTIES_REGION_SEQUENCE_REF
            );
            if reference.is_null() {
                return Err(AraError::InvalidArgument(
                    "ARA2 playback region requires a region sequence",
                ));
            }
            (None, Some(resolve_sequence(reference)?))
        } else {
            let reference = field!(
                ARAMusicalContextRef,
                musicalContextRef,
                layout::ARAPLAYBACK_REGION_PROPERTIES_MUSICAL_CONTEXT_REF
            );
            if reference.is_null() {
                return Err(AraError::InvalidArgument(
                    "ARA1 playback region requires a musical context",
                ));
            }
            (Some(resolve_context(reference)?), None)
        };
        let name = if ara2 && input.contains_extent(layout::ARAPLAYBACK_REGION_PROPERTIES_NAME) {
            let pointer = field!(
                ARAUtf8String,
                name,
                layout::ARAPLAYBACK_REGION_PROPERTIES_NAME
            );
            // SAFETY: the outer contract covers the represented nested display string.
            unsafe { copy_optional_display(pointer)? }
        } else {
            None
        };
        let color = if ara2 && input.contains_extent(layout::ARAPLAYBACK_REGION_PROPERTIES_COLOR) {
            let pointer = field!(
                *const ARAColor,
                color,
                layout::ARAPLAYBACK_REGION_PROPERTIES_COLOR
            );
            if pointer.is_null() {
                None
            } else {
                // SAFETY: the outer contract covers represented nested color storage.
                Some(unsafe { Color::copy_from_ptr(pointer)? })
            }
        } else {
            None
        };
        Ok(Self {
            transformation_flags: flags,
            start_in_modification: modification_start,
            duration_in_modification: modification_duration,
            start_in_playback: playback_start,
            duration_in_playback: playback_duration,
            musical_context,
            region_sequence,
            name,
            color,
        })
    }

    /// Builds a pinned raw record for the selected generation.
    pub fn as_ffi(
        &self,
        generation: ApiGeneration,
    ) -> Result<Pin<Box<FfiProperties<'_, ARAPlaybackRegionProperties>>>, AraError> {
        ensure_generation(generation)?;
        let ara2 = generation >= ApiGeneration::V2Draft;
        if ara2 && self.region_sequence.is_none() {
            return Err(AraError::InvalidArgument(
                "ARA2 playback region requires a region sequence",
            ));
        }
        if !ara2 && self.musical_context.is_none() {
            return Err(AraError::InvalidArgument(
                "ARA1 playback region requires a musical context",
            ));
        }
        let represented = if !ara2 {
            layout::ARAPLAYBACK_REGION_PROPERTIES_MUSICAL_CONTEXT_REF
        } else if self.color.is_some() {
            layout::ARAPLAYBACK_REGION_PROPERTIES_COLOR
        } else if self.name.is_some() {
            layout::ARAPLAYBACK_REGION_PROPERTIES_NAME
        } else {
            layout::ARAPLAYBACK_REGION_PROPERTIES_REGION_SEQUENCE_REF
        };
        // SAFETY: all fields accept zero.
        let mut raw = unsafe { zeroed_raw::<ARAPlaybackRegionProperties>() };
        // SAFETY: generated offsets and matching types.
        unsafe {
            write_raw(
                &mut raw,
                offset_of!(ARAPlaybackRegionProperties, structSize),
                represented as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAPlaybackRegionProperties, transformationFlags),
                self.transformation_flags,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAPlaybackRegionProperties, startInModificationTime),
                self.start_in_modification,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAPlaybackRegionProperties, durationInModificationTime),
                self.duration_in_modification,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAPlaybackRegionProperties, startInPlaybackTime),
                self.start_in_playback,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAPlaybackRegionProperties, durationInPlaybackTime),
                self.duration_in_playback,
            );
            write_raw(
                &mut raw,
                offset_of!(ARAPlaybackRegionProperties, musicalContextRef),
                self.musical_context.map_or(
                    std::ptr::null_mut::<ara2_bridge_sys::ARAMusicalContextRefMarkupType>(),
                    |v| v.as_raw().cast(),
                ) as ARAMusicalContextRef,
            );
            if ara2 {
                write_raw(
                    &mut raw,
                    offset_of!(ARAPlaybackRegionProperties, regionSequenceRef),
                    self.region_sequence.map_or(
                        std::ptr::null_mut::<ara2_bridge_sys::ARARegionSequenceRefMarkupType>(),
                        |v| v.as_raw().cast(),
                    ) as ARARegionSequenceRef,
                );
                write_raw(
                    &mut raw,
                    offset_of!(ARAPlaybackRegionProperties, name),
                    self.name.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                );
                write_raw(
                    &mut raw,
                    offset_of!(ARAPlaybackRegionProperties, color),
                    self.color.as_ref().map_or(std::ptr::null(), Color::as_ptr),
                );
            }
        }
        Ok(FfiProperties::pin(raw))
    }
}

fn ensure_generation(generation: ApiGeneration) -> Result<(), AraError> {
    if generation.supported_on_target() {
        Ok(())
    } else {
        Err(AraError::Unsupported(
            "API generation is unavailable on this target",
        ))
    }
}

fn validate_audio_shape(
    sample_count: ARASampleCount,
    sample_rate: ARASampleRate,
    channel_count: ARAInt32,
) -> Result<(), AraError> {
    if sample_count < 0 {
        return Err(AraError::InvalidArgument(
            "sample count must be nonnegative",
        ));
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(AraError::InvalidArgument(
            "sample rate must be finite and positive",
        ));
    }
    if channel_count <= 0 {
        return Err(AraError::InvalidArgument("channel count must be positive"));
    }
    Ok(())
}

fn validate_playback(
    flags: ARAPlaybackTransformationFlags,
    modification_start: f64,
    modification_duration: f64,
    playback_start: f64,
    playback_duration: f64,
) -> Result<(), AraError> {
    let known = (ara2_bridge_sys::kARAPlaybackTransformationTimestretch
        | ara2_bridge_sys::kARAPlaybackTransformationTimestretchReflectingTempo
        | ara2_bridge_sys::kARAPlaybackTransformationContentBasedFadeAtTail
        | ara2_bridge_sys::kARAPlaybackTransformationContentBasedFadeAtHead)
        as ARAPlaybackTransformationFlags;
    if flags & !known != 0 {
        return Err(AraError::InvalidArgument(
            "unknown playback transformation flags",
        ));
    }
    if !modification_start.is_finite()
        || !playback_start.is_finite()
        || !modification_duration.is_finite()
        || !playback_duration.is_finite()
        || modification_duration < 0.0
        || playback_duration < 0.0
    {
        return Err(AraError::InvalidArgument(
            "playback times must be finite with nonnegative durations",
        ));
    }
    Ok(())
}

unsafe fn copy_arrangement(
    data_type: ARAChannelArrangementDataType,
    pointer: *const std::os::raw::c_void,
    channel_count: ARAInt32,
) -> Result<Option<RawChannelArrangement>, AraError> {
    if data_type == 0 {
        if !pointer.is_null() {
            return Err(AraError::InvalidArgument(
                "undefined channel arrangement must be null",
            ));
        }
        return Ok(None);
    }
    if pointer.is_null() {
        return Err(AraError::InvalidArgument(
            "defined channel arrangement is null",
        ));
    }
    let byte_count = match data_type {
        1 => size_of::<u64>(),
        3 => size_of::<i32>(),
        4 => usize::try_from(channel_count)
            .map_err(|_| AraError::InvalidArgument("channel count must be positive"))?,
        2 | 5 => {
            return Err(AraError::Unsupported(
                "channel arrangement requires companion adapter sizing",
            ))
        }
        _ => {
            return Err(AraError::InvalidArgument(
                "unknown channel arrangement data type",
            ))
        }
    };
    // SAFETY: the outer property contract guarantees the selected nested byte extent is readable.
    let bytes = unsafe { ForeignSlice::<u8>::copy_from_raw(pointer.cast(), byte_count)? };
    RawChannelArrangement::new(data_type, bytes.as_slice()).map(Some)
}
