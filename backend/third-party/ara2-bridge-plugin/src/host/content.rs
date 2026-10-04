//! Dispatcher-scoped host content access.

use super::{AudioAccess, HostAudioReader, SampleFormat};
use ara2_bridge_core::{
    AraError, ContentGrade, ContentKind, ContentReader, ContentReaderBackend, SizedInput,
};
use ara2_bridge_sys::*;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::offset_of;
use std::ptr::{null, NonNull};

type AvailableMusical = unsafe extern "C" fn(
    ARAContentAccessControllerHostRef,
    ARAMusicalContextHostRef,
    ARAContentType,
) -> ARABool;
type GradeMusical = unsafe extern "C" fn(
    ARAContentAccessControllerHostRef,
    ARAMusicalContextHostRef,
    ARAContentType,
) -> ARAContentGrade;
type CreateMusical = unsafe extern "C" fn(
    ARAContentAccessControllerHostRef,
    ARAMusicalContextHostRef,
    ARAContentType,
    *const ARAContentTimeRange,
) -> ARAContentReaderHostRef;
type AvailableSource = unsafe extern "C" fn(
    ARAContentAccessControllerHostRef,
    ARAAudioSourceHostRef,
    ARAContentType,
) -> ARABool;
type GradeSource = unsafe extern "C" fn(
    ARAContentAccessControllerHostRef,
    ARAAudioSourceHostRef,
    ARAContentType,
) -> ARAContentGrade;
type CreateSource = unsafe extern "C" fn(
    ARAContentAccessControllerHostRef,
    ARAAudioSourceHostRef,
    ARAContentType,
    *const ARAContentTimeRange,
) -> ARAContentReaderHostRef;
type Count =
    unsafe extern "C" fn(ARAContentAccessControllerHostRef, ARAContentReaderHostRef) -> i32;
type Data = unsafe extern "C" fn(
    ARAContentAccessControllerHostRef,
    ARAContentReaderHostRef,
    i32,
) -> *const c_void;
type Destroy = unsafe extern "C" fn(ARAContentAccessControllerHostRef, ARAContentReaderHostRef);

#[derive(Clone, Copy)]
pub(crate) struct ContentAccess<'host> {
    host_ref: ARAContentAccessControllerHostRef,
    available_musical: AvailableMusical,
    grade_musical: GradeMusical,
    create_musical: CreateMusical,
    available_source: AvailableSource,
    grade_source: GradeSource,
    create_source: CreateSource,
    count: Count,
    data: Data,
    destroy: Destroy,
    _lifetime: PhantomData<&'host ()>,
}

impl<'host> ContentAccess<'host> {
    pub(crate) unsafe fn from_raw(
        host_ref: ARAContentAccessControllerHostRef,
        interface: *const ARAContentAccessControllerInterface,
    ) -> Result<Option<Self>, AraError> {
        if interface.is_null() {
            return Ok(None);
        }
        if host_ref.is_null() {
            return Err(AraError::Abi("content host reference is null"));
        }
        // SAFETY: the caller supplies a readable optional interface for the returned lifetime.
        let input = unsafe { SizedInput::from_ptr(interface) }?;
        macro_rules! callback {
            ($field:ident, $type:ty, $extent:ident, $error:literal) => {{
                // SAFETY: the generated offset, callback type, and extent name the same field.
                unsafe {
                    input.copy_field::<Option<$type>>(
                        offset_of!(ARAContentAccessControllerInterface, $field),
                        ara2_bridge_sys::layout::$extent,
                    )
                }?
                .ok_or(AraError::Abi($error))?
            }};
        }
        Ok(Some(Self {
            host_ref,
            available_musical: callback!(
                isMusicalContextContentAvailable,
                AvailableMusical,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_IS_MUSICAL_CONTEXT_CONTENT_AVAILABLE,
                "musical content availability callback is null"
            ),
            grade_musical: callback!(
                getMusicalContextContentGrade,
                GradeMusical,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_GET_MUSICAL_CONTEXT_CONTENT_GRADE,
                "musical content grade callback is null"
            ),
            create_musical: callback!(
                createMusicalContextContentReader,
                CreateMusical,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_CREATE_MUSICAL_CONTEXT_CONTENT_READER,
                "musical content reader callback is null"
            ),
            available_source: callback!(
                isAudioSourceContentAvailable,
                AvailableSource,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_IS_AUDIO_SOURCE_CONTENT_AVAILABLE,
                "audio-source content availability callback is null"
            ),
            grade_source: callback!(
                getAudioSourceContentGrade,
                GradeSource,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_GET_AUDIO_SOURCE_CONTENT_GRADE,
                "audio-source content grade callback is null"
            ),
            create_source: callback!(
                createAudioSourceContentReader,
                CreateSource,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_CREATE_AUDIO_SOURCE_CONTENT_READER,
                "audio-source content reader callback is null"
            ),
            count: callback!(
                getContentReaderEventCount,
                Count,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_GET_CONTENT_READER_EVENT_COUNT,
                "content count callback is null"
            ),
            data: callback!(
                getContentReaderDataForEvent,
                Data,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_GET_CONTENT_READER_DATA_FOR_EVENT,
                "content data callback is null"
            ),
            destroy: callback!(
                destroyContentReader,
                Destroy,
                ARACONTENT_ACCESS_CONTROLLER_INTERFACE_DESTROY_CONTENT_READER,
                "content reader destruction callback is null"
            ),
            _lifetime: PhantomData,
        }))
    }
}

enum ScopeObject {
    AudioSource(HostAudioSourceRef),
    MusicalContext(HostMusicalContextRef),
    EndEditing,
}

/// Non-null host identity for an audio source received from a validated ARA callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostAudioSourceRef(NonNull<ARAAudioSourceHostRefMarkupType>);

impl HostAudioSourceRef {
    /// Wraps an opaque audio-source host reference received from an ARA peer.
    ///
    /// # Safety
    ///
    /// `raw` must identify a live host audio source for the enclosing document controller.
    pub unsafe fn from_raw(raw: ARAAudioSourceHostRef) -> Result<Self, AraError> {
        NonNull::new(raw).map(Self).ok_or(AraError::InvalidArgument(
            "audio-source host reference is null",
        ))
    }

    /// Returns the unchanged opaque ARA host reference.
    pub fn as_raw(self) -> ARAAudioSourceHostRef {
        self.0.as_ptr()
    }
}

/// Non-null host identity for a musical context received from a validated ARA callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostMusicalContextRef(NonNull<ARAMusicalContextHostRefMarkupType>);

impl HostMusicalContextRef {
    /// Wraps an opaque musical-context host reference received from an ARA peer.
    ///
    /// # Safety
    ///
    /// `raw` must identify a live host musical context for the enclosing document controller.
    pub unsafe fn from_raw(raw: ARAMusicalContextHostRef) -> Result<Self, AraError> {
        NonNull::new(raw).map(Self).ok_or(AraError::InvalidArgument(
            "musical-context host reference is null",
        ))
    }

    /// Returns the unchanged opaque ARA host reference.
    pub fn as_raw(self) -> ARAMusicalContextHostRef {
        self.0.as_ptr()
    }
}

/// Non-storable token granting host content access only during an eligible callback.
pub struct HostContentScope<'call, 'host> {
    content: Option<ContentAccess<'host>>,
    audio: Option<AudioAccess<'host>>,
    object: ScopeObject,
    _not_send: PhantomData<*mut &'call mut ()>,
}

impl<'call, 'host> HostContentScope<'call, 'host> {
    pub(crate) fn for_audio_source(
        content: Option<&ContentAccess<'host>>,
        audio: &AudioAccess<'host>,
        current: HostAudioSourceRef,
    ) -> Self {
        Self {
            content: content.copied(),
            audio: Some(audio.clone()),
            object: ScopeObject::AudioSource(current),
            _not_send: PhantomData,
        }
    }

    pub(crate) fn for_musical_context(
        content: Option<&ContentAccess<'host>>,
        audio: &AudioAccess<'host>,
        current: HostMusicalContextRef,
    ) -> Self {
        Self {
            content: content.copied(),
            audio: Some(audio.clone()),
            object: ScopeObject::MusicalContext(current),
            _not_send: PhantomData,
        }
    }

    pub(crate) fn end_editing(
        content: Option<&ContentAccess<'host>>,
        audio: &AudioAccess<'host>,
    ) -> Self {
        Self {
            content: content.copied(),
            audio: Some(audio.clone()),
            object: ScopeObject::EndEditing,
            _not_send: PhantomData,
        }
    }

    pub(crate) fn for_audio_source_management(
        audio: &AudioAccess<'host>,
        current: HostAudioSourceRef,
    ) -> Self {
        Self {
            content: None,
            audio: Some(audio.clone()),
            object: ScopeObject::AudioSource(current),
            _not_send: PhantomData,
        }
    }

    pub(crate) fn unavailable() -> Self {
        Self {
            content: None,
            audio: None,
            object: ScopeObject::EndEditing,
            _not_send: PhantomData,
        }
    }

    /// Returns the current audio-source host identity, when this is a source-scoped callback.
    pub fn current_audio_source(&self) -> Option<HostAudioSourceRef> {
        match self.object {
            ScopeObject::AudioSource(current) => Some(current),
            _ => None,
        }
    }

    /// Returns the current musical-context host identity, when this is a context-scoped callback.
    pub fn current_musical_context(&self) -> Option<HostMusicalContextRef> {
        match self.object {
            ScopeObject::MusicalContext(current) => Some(current),
            _ => None,
        }
    }

    /// Creates a typed reader for the current or end-editing-eligible audio source.
    pub fn audio_source_grade<K: ContentKind>(
        &self,
        source: HostAudioSourceRef,
    ) -> Result<ContentGrade, AraError> {
        if !self.allows_audio_source(source) {
            return Err(AraError::InvalidState(
                "audio source is not current in this content scope",
            ));
        }
        let access = self
            .content
            .as_ref()
            .ok_or(AraError::Unsupported("host content access is unavailable"))?;
        // SAFETY: the callback and current host identity were validated by this scope.
        Ok(ContentGrade::from_raw(unsafe {
            (access.grade_source)(access.host_ref, source.as_raw(), K::RAW_TYPE)
        }))
    }

    /// Returns the host content grade for the current or end-editing-eligible musical context.
    pub fn musical_context_grade<K: ContentKind>(
        &self,
        context: HostMusicalContextRef,
    ) -> Result<ContentGrade, AraError> {
        if !self.allows_musical_context(context) {
            return Err(AraError::InvalidState(
                "musical context is not current in this content scope",
            ));
        }
        let access = self
            .content
            .as_ref()
            .ok_or(AraError::Unsupported("host content access is unavailable"))?;
        // SAFETY: the callback and current host identity were validated by this scope.
        Ok(ContentGrade::from_raw(unsafe {
            (access.grade_musical)(access.host_ref, context.as_raw(), K::RAW_TYPE)
        }))
    }

    /// Creates a typed reader for the current or end-editing-eligible audio source.
    pub fn audio_source<K: ContentKind>(
        &self,
        source: HostAudioSourceRef,
        range: Option<ARAContentTimeRange>,
    ) -> Result<HostContentReader<'call, K>, AraError> {
        if !self.allows_audio_source(source) {
            return Err(AraError::InvalidState(
                "audio source is not current in this content scope",
            ));
        }
        let access = self
            .content
            .as_ref()
            .ok_or(AraError::Unsupported("host content access is unavailable"))?;
        // SAFETY: the validated callback is called only for the current scoped object.
        let available =
            unsafe { (access.available_source)(access.host_ref, source.as_raw(), K::RAW_TYPE) };
        if available == kARAFalse {
            return Err(AraError::Unsupported(
                "requested host content is unavailable",
            ));
        }
        let range_pointer = range
            .as_ref()
            .map_or(null(), |range| range as *const ARAContentTimeRange);
        // SAFETY: the callback and ref were validated, and the optional range lives through call.
        let reader = unsafe {
            (access.create_source)(access.host_ref, source.as_raw(), K::RAW_TYPE, range_pointer)
        };
        HostContentReader::new(access, reader)
    }

    /// Creates a typed reader for the current or end-editing-eligible musical context.
    pub fn musical_context<K: ContentKind>(
        &self,
        context: HostMusicalContextRef,
        range: Option<ARAContentTimeRange>,
    ) -> Result<HostContentReader<'call, K>, AraError> {
        if !self.allows_musical_context(context) {
            return Err(AraError::InvalidState(
                "musical context is not current in this content scope",
            ));
        }
        let access = self
            .content
            .as_ref()
            .ok_or(AraError::Unsupported("host content access is unavailable"))?;
        // SAFETY: the validated callback is called only for the current scoped object.
        let available =
            unsafe { (access.available_musical)(access.host_ref, context.as_raw(), K::RAW_TYPE) };
        if available == kARAFalse {
            return Err(AraError::Unsupported(
                "requested host content is unavailable",
            ));
        }
        let range_pointer = range
            .as_ref()
            .map_or(null(), |range| range as *const ARAContentTimeRange);
        // SAFETY: the callback and ref were validated, and the optional range lives through call.
        let reader = unsafe {
            (access.create_musical)(
                access.host_ref,
                context.as_raw(),
                K::RAW_TYPE,
                range_pointer,
            )
        };
        HostContentReader::new(access, reader)
    }

    /// Creates a long-lived 32- or 64-bit audio reader under the same source gate.
    pub fn audio_reader<S: SampleFormat>(
        &self,
        source: HostAudioSourceRef,
        channels: usize,
    ) -> Result<HostAudioReader<S>, AraError> {
        if !self.allows_audio_source(source) {
            return Err(AraError::InvalidState(
                "audio source is not current in this content scope",
            ));
        }
        self.audio
            .as_ref()
            .ok_or(AraError::Unsupported("host audio access is unavailable"))?
            .reader::<S>(source.as_raw(), channels)
    }

    fn allows_audio_source(&self, source: HostAudioSourceRef) -> bool {
        matches!(self.object, ScopeObject::EndEditing)
            || matches!(self.object, ScopeObject::AudioSource(current) if current == source)
    }

    fn allows_musical_context(&self, context: HostMusicalContextRef) -> bool {
        matches!(self.object, ScopeObject::EndEditing)
            || matches!(self.object, ScopeObject::MusicalContext(current) if current == context)
    }
}

struct Backend<'call> {
    host_ref: ARAContentAccessControllerHostRef,
    reader: ARAContentReaderHostRef,
    raw_type: ARAContentType,
    event_size: usize,
    count: Count,
    data: Data,
    destroy: Destroy,
    _scope: PhantomData<&'call mut ()>,
}

// SAFETY: construction exclusively owns a non-null peer reader; the ARA host contract supplies
// kind-matched event storage until the next reader callback, and `destroy` consumes it once.
unsafe impl ContentReaderBackend for Backend<'_> {
    fn raw_content_type(&self) -> ARAContentType {
        self.raw_type
    }

    fn event_count(&mut self) -> Result<i32, AraError> {
        // SAFETY: the validated access and exclusively owned reader remain live.
        Ok(unsafe { (self.count)(self.host_ref, self.reader) })
    }

    unsafe fn event_data(&mut self, index: i32) -> Result<(*const c_void, usize), AraError> {
        // SAFETY: the caller respects the backend invalidation contract; callback is validated.
        let pointer = unsafe { (self.data)(self.host_ref, self.reader, index) };
        if pointer.is_null() {
            return Err(AraError::Peer("host returned null content event"));
        }
        Ok((pointer, self.event_size))
    }

    fn destroy(&mut self) {
        // SAFETY: the backend exclusively owns the reader and the owner invokes this exactly once.
        unsafe { (self.destroy)(self.host_ref, self.reader) };
    }
}

/// Typed host content reader confined to its callback scope.
pub struct HostContentReader<'call, K: ContentKind> {
    inner: ContentReader<K, Backend<'call>>,
}

impl<'call, K: ContentKind> HostContentReader<'call, K> {
    fn new(access: &ContentAccess<'_>, reader: ARAContentReaderHostRef) -> Result<Self, AraError> {
        if reader.is_null() {
            return Err(AraError::Peer("host failed to create content reader"));
        }
        let backend = Backend {
            host_ref: access.host_ref,
            reader,
            raw_type: K::RAW_TYPE,
            event_size: K::RAW_EVENT_SIZE,
            count: access.count,
            data: access.data,
            destroy: access.destroy,
            _scope: PhantomData,
        };
        Ok(Self {
            inner: ContentReader::new(backend)?,
        })
    }

    /// Returns the checked event count.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether this reader contains no events.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Copies one event into aligned, owned Rust storage.
    pub fn event(&mut self, index: usize) -> Result<K::Event, AraError> {
        self.inner.event(index)
    }
}
