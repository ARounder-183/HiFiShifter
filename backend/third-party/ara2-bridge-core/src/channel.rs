//! Owned, companion-independent channel-arrangement inspection.

use crate::AraError;

/// One Core Audio channel description.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreAudioChannelDescription {
    label: u32,
    flags: u32,
    coordinates: [f32; 3],
}

impl CoreAudioChannelDescription {
    /// Creates a description with finite coordinates.
    pub fn new(label: u32, flags: u32, coordinates: [f32; 3]) -> Result<Self, AraError> {
        if coordinates.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(AraError::InvalidArgument(
                "Core Audio channel coordinates must be finite",
            ));
        }
        Ok(Self {
            label,
            flags,
            coordinates,
        })
    }

    /// Returns the Core Audio channel label.
    pub const fn label(self) -> u32 {
        self.label
    }

    /// Returns the Core Audio coordinate flags.
    pub const fn flags(self) -> u32 {
        self.flags
    }

    /// Returns the three channel coordinates.
    pub const fn coordinates(self) -> [f32; 3] {
        self.coordinates
    }
}

/// Safe Core Audio channel-layout forms accepted by ARA.
#[derive(Clone, Debug, PartialEq)]
pub enum CoreAudioChannelLayout {
    /// A standard layout tag; its low 16 bits encode channel count.
    Tag(u32),
    /// Explicit channel descriptions, one per channel.
    Descriptions(Vec<CoreAudioChannelDescription>),
}

/// Opaque future arrangement bytes whose meaning is caller-validated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpaqueChannelArrangement {
    data_type: i32,
    bytes: Box<[u8]>,
}

impl OpaqueChannelArrangement {
    /// Creates a future arrangement without bridge validation.
    ///
    /// # Safety
    ///
    /// `data_type` and `bytes` must be a complete, valid representation for the companion API and
    /// channel count with which this value is used. The bridge cannot inspect or validate it.
    pub unsafe fn new_unchecked(data_type: i32, bytes: Box<[u8]>) -> Self {
        Self { data_type, bytes }
    }

    /// Returns the future ARA data-type tag.
    pub const fn data_type(&self) -> i32 {
        self.data_type
    }

    /// Returns the retained opaque bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Owned channel-arrangement payload variants.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ChannelArrangement {
    /// No arrangement data.
    Undefined,
    /// VST3 speaker bit mask.
    Vst3(u64),
    /// Core Audio layout.
    CoreAudio(CoreAudioChannelLayout),
    /// AAX stem format whose low 16 bits encode channel count.
    Aax(u32),
    /// CLAP channel identifiers, one byte per channel.
    ClapMap(Vec<u8>),
    /// CLAP ambisonic ordering and normalization.
    ClapAmbisonic {
        /// CLAP ambisonic ordering value.
        ordering: u32,
        /// CLAP ambisonic normalization value.
        normalization: u32,
    },
    /// Explicitly unsafe future representation.
    Opaque(OpaqueChannelArrangement),
}

impl ChannelArrangement {
    /// Decodes a known ARA arrangement tag from a complete caller-owned byte extent.
    pub fn from_raw(data_type: i32, bytes: &[u8], channel_count: u32) -> Result<Self, AraError> {
        match data_type {
            0 if bytes.is_empty() => Ok(Self::Undefined),
            0 => Err(AraError::InvalidArgument(
                "undefined arrangement must have no bytes",
            )),
            1 => Ok(Self::Vst3(u64::from_ne_bytes(bytes.try_into().map_err(
                |_| AraError::InvalidArgument("VST3 arrangement must be 8 bytes"),
            )?))),
            2 => decode_core_audio(bytes),
            3 => Ok(Self::Aax(u32::from_ne_bytes(bytes.try_into().map_err(
                |_| AraError::InvalidArgument("AAX arrangement must be 4 bytes"),
            )?))),
            4 if bytes.len() == channel_count as usize => Ok(Self::ClapMap(bytes.to_vec())),
            4 => Err(AraError::InvalidArgument(
                "CLAP channel map extent must equal channel count",
            )),
            5 => {
                let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
                    AraError::InvalidArgument("CLAP ambisonic info must be 8 bytes")
                })?;
                Ok(Self::ClapAmbisonic {
                    ordering: u32::from_ne_bytes(bytes[..4].try_into().expect("four bytes")),
                    normalization: u32::from_ne_bytes(bytes[4..].try_into().expect("four bytes")),
                })
            }
            _ => Err(AraError::InvalidArgument(
                "unknown channel arrangement data type",
            )),
        }
    }

    fn implied_channel_count(&self) -> Option<u32> {
        match self {
            Self::Undefined | Self::ClapAmbisonic { .. } | Self::Opaque(_) => None,
            Self::Vst3(mask) => Some(mask.count_ones()),
            Self::CoreAudio(CoreAudioChannelLayout::Tag(tag)) => Some(tag & 0xFFFF),
            Self::CoreAudio(CoreAudioChannelLayout::Descriptions(descriptions)) => {
                u32::try_from(descriptions.len()).ok()
            }
            Self::Aax(stem) => Some(stem & 0xFFFF),
            Self::ClapMap(map) => u32::try_from(map.len()).ok(),
        }
    }
}

/// A channel count paired with a validated owned arrangement.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelFormat {
    channel_count: u32,
    arrangement: ChannelArrangement,
}

impl ChannelFormat {
    /// Validates the arrangement against its explicit channel count.
    pub fn new(channel_count: u32, arrangement: ChannelArrangement) -> Result<Self, AraError> {
        if matches!(
            arrangement,
            ChannelArrangement::CoreAudio(CoreAudioChannelLayout::Tag(0 | 0x0001_0000))
        ) {
            return Err(AraError::InvalidArgument(
                "Core Audio description and bitmap tags require validated forms",
            ));
        }
        if matches!(arrangement, ChannelArrangement::Undefined) && channel_count > 2 {
            return Err(AraError::InvalidArgument(
                "more than stereo requires a channel arrangement",
            ));
        }
        if let Some(implied) = arrangement.implied_channel_count() {
            if implied != channel_count {
                return Err(AraError::InvalidArgument(
                    "arrangement channel count does not match",
                ));
            }
        }
        Ok(Self {
            channel_count,
            arrangement,
        })
    }

    /// Returns the explicit channel count.
    pub const fn channel_count(&self) -> u32 {
        self.channel_count
    }

    /// Returns the owned arrangement.
    pub const fn arrangement(&self) -> &ChannelArrangement {
        &self.arrangement
    }
}

fn decode_core_audio(bytes: &[u8]) -> Result<ChannelArrangement, AraError> {
    if bytes.len() < 12 {
        return Err(AraError::InvalidArgument(
            "Core Audio layout is shorter than its header",
        ));
    }
    let tag = u32::from_ne_bytes(bytes[..4].try_into().expect("four bytes"));
    let description_count = u32::from_ne_bytes(bytes[8..12].try_into().expect("four bytes"));
    if tag == 0x0001_0000 {
        return Err(AraError::InvalidArgument(
            "Core Audio bitmap layouts are not allowed",
        ));
    }
    if tag != 0 {
        if description_count != 0 || bytes.len() != 12 {
            return Err(AraError::InvalidArgument(
                "tagged Core Audio layout must have no descriptions",
            ));
        }
        return Ok(ChannelArrangement::CoreAudio(CoreAudioChannelLayout::Tag(
            tag,
        )));
    }
    let count = usize::try_from(description_count)
        .map_err(|_| AraError::InvalidArgument("Core Audio description count overflow"))?;
    let expected = count
        .checked_mul(20)
        .and_then(|extent| extent.checked_add(12))
        .ok_or(AraError::InvalidArgument(
            "Core Audio layout extent overflow",
        ))?;
    if bytes.len() != expected {
        return Err(AraError::InvalidArgument(
            "Core Audio description extent mismatch",
        ));
    }
    let mut descriptions = Vec::with_capacity(count);
    for raw in bytes[12..].chunks_exact(20) {
        let label = u32::from_ne_bytes(raw[..4].try_into().expect("four bytes"));
        let flags = u32::from_ne_bytes(raw[4..8].try_into().expect("four bytes"));
        let coordinates = [
            f32::from_ne_bytes(raw[8..12].try_into().expect("four bytes")),
            f32::from_ne_bytes(raw[12..16].try_into().expect("four bytes")),
            f32::from_ne_bytes(raw[16..20].try_into().expect("four bytes")),
        ];
        descriptions.push(CoreAudioChannelDescription::new(label, flags, coordinates)?);
    }
    Ok(ChannelArrangement::CoreAudio(
        CoreAudioChannelLayout::Descriptions(descriptions),
    ))
}
