//! RAII content readers and lending event references.

use super::{copy_event_from_ffi, ContentKind, Notes};
use crate::AraError;
use ara2_bridge_sys::{ARAContentNote, ARAContentType};
use std::cell::Cell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::rc::Rc;

/// Low-level peer operations owned by a content reader.
///
/// # Safety
///
/// Implementations must exclusively own one successfully created ARA reader. Each successful
/// `event_data` result must match `raw_content_type`, remain readable through its returned extent
/// until the next backend method or destruction, and uphold the nested-pointer contract documented
/// by [`crate::copy_event_from_ffi`]. `destroy` must destroy the peer reader exactly once.
pub unsafe trait ContentReaderBackend {
    /// Returns the raw content type selected when the peer reader was created.
    fn raw_content_type(&self) -> ARAContentType;

    /// Returns the peer's event count.
    fn event_count(&mut self) -> Result<i32, AraError>;

    /// Returns one ephemeral event pointer and its readable extent.
    ///
    /// # Safety
    ///
    /// The caller must not retain or read the returned pointer after another backend method or
    /// `destroy` call.
    unsafe fn event_data(&mut self, index: i32) -> Result<(*const c_void, usize), AraError>;

    /// Destroys the exclusively owned peer reader.
    fn destroy(&mut self);
}

/// Uninhabited default parameter used solely for `ContentReader::<K>::new(...)` inference.
#[derive(Debug)]
pub enum NoContentReaderBackend {}

// SAFETY: the type is uninhabited, so none of the backend methods can be invoked.
unsafe impl ContentReaderBackend for NoContentReaderBackend {
    fn raw_content_type(&self) -> ARAContentType {
        match *self {}
    }

    fn event_count(&mut self) -> Result<i32, AraError> {
        match *self {}
    }

    unsafe fn event_data(&mut self, _index: i32) -> Result<(*const c_void, usize), AraError> {
        match *self {}
    }

    fn destroy(&mut self) {
        match *self {}
    }
}

/// Model-thread gate used to keep controller operations exclusive while a reader exists.
#[derive(Debug, Default)]
pub struct ContentReaderGate {
    active: Cell<bool>,
    _model_thread_only: PhantomData<Rc<()>>,
}

impl ContentReaderGate {
    /// Creates an idle gate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquires exclusive reader access until the returned lease is dropped.
    pub fn acquire(&self) -> Result<ContentReaderLease<'_>, AraError> {
        if self.active.replace(true) {
            return Err(AraError::InvalidState("a content reader is already active"));
        }
        Ok(ContentReaderLease { gate: self })
    }

    /// Rejects an operation while a reader lease is active.
    pub fn require_idle(&self) -> Result<(), AraError> {
        if self.active.get() {
            Err(AraError::InvalidState(
                "controller operation conflicts with an active content reader",
            ))
        } else {
            Ok(())
        }
    }
}

/// Exclusive reader lease obtained from [`ContentReaderGate`].
#[derive(Debug)]
pub struct ContentReaderLease<'a> {
    gate: &'a ContentReaderGate,
}

impl Drop for ContentReaderLease<'_> {
    fn drop(&mut self) {
        self.gate.active.set(false);
    }
}

#[derive(Debug)]
struct ReaderOwner<B: ContentReaderBackend> {
    backend: B,
}

impl<B: ContentReaderBackend> Drop for ReaderOwner<B> {
    fn drop(&mut self) {
        self.backend.destroy();
    }
}

/// A lending reference whose lifetime ends before another peer call can occur.
#[derive(Debug)]
pub struct EventRef<'event, K: ContentKind> {
    pointer: *const c_void,
    extent: usize,
    _event: PhantomData<&'event K::Event>,
}

impl<K: ContentKind> EventRef<'_, K> {
    /// Copies the lent event into its owned representation.
    pub fn to_owned(&self) -> K::Event {
        // SAFETY: only a validated reader call can construct this reference, and its borrow prevents
        // the next backend call until this method returns.
        unsafe { copy_event_from_ffi::<K>(K::RAW_TYPE, self.pointer, self.extent) }
            .expect("EventRef construction validates the event")
    }
}

impl EventRef<'_, Notes> {
    fn raw(&self) -> ARAContentNote {
        // SAFETY: construction validates a complete readable note extent; unaligned peers are valid.
        unsafe { self.pointer.cast::<ARAContentNote>().read_unaligned() }
    }

    /// Returns the quantized pitch number or the ARA invalid-pitch sentinel without copying.
    pub fn pitch_number(&self) -> i32 {
        self.raw().pitchNumber
    }

    /// Returns the note start position without copying.
    pub fn start_position(&self) -> f64 {
        self.raw().startPosition
    }
}

/// A typed RAII content reader.
#[derive(Debug)]
pub struct ContentReader<K: ContentKind, B: ContentReaderBackend = NoContentReaderBackend> {
    owner: ReaderOwner<B>,
    count: usize,
    next_index: usize,
    _kind: PhantomData<K>,
}

impl<K: ContentKind> ContentReader<K, NoContentReaderBackend> {
    /// Creates a typed reader from an exclusively owned backend.
    pub fn new<B: ContentReaderBackend>(backend: B) -> Result<ContentReader<K, B>, AraError> {
        let mut owner = ReaderOwner { backend };
        if owner.backend.raw_content_type() != K::RAW_TYPE {
            return Err(AraError::InvalidArgument("content reader kind mismatch"));
        }
        let count = checked_count(owner.backend.event_count()?)?;
        K::validate_count(count)?;
        Ok(ContentReader {
            owner,
            count,
            next_index: 0,
            _kind: PhantomData,
        })
    }
}

impl<K: ContentKind, B: ContentReaderBackend> ContentReader<K, B> {
    /// Returns the checked event count.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns whether the reader contains no events.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn check_index(&self, index: usize) -> Result<i32, AraError> {
        if index >= self.count {
            return Err(AraError::InvalidArgument(
                "content event index out of bounds",
            ));
        }
        i32::try_from(index).map_err(|_| AraError::InvalidArgument("content event index overflow"))
    }

    fn fetch_owned(&mut self, index: usize) -> Result<K::Event, AraError> {
        let index = self.check_index(index)?;
        // SAFETY: the unsafe backend contract supplies a readable pointer for this kind until the
        // next backend method; copying completes before such a call.
        let (pointer, extent) = unsafe { self.owner.backend.event_data(index)? };
        // SAFETY: the backend contract and selected reader kind establish the decoder preconditions.
        unsafe { copy_event_from_ffi::<K>(K::RAW_TYPE, pointer, extent) }
    }

    fn previous_event(&mut self, index: usize) -> Result<Option<K::Event>, AraError> {
        if index == 0 {
            Ok(None)
        } else {
            self.fetch_owned(index - 1).map(Some)
        }
    }

    /// Copies an event into aligned owned Rust storage.
    pub fn event(&mut self, index: usize) -> Result<K::Event, AraError> {
        let previous = self.previous_event(index)?;
        let current = self.fetch_owned(index)?;
        if let Some(previous) = previous.as_ref() {
            K::validate_pair(previous, &current)?;
        }
        Ok(current)
    }

    /// Lends an ephemeral event to a higher-ranked closure.
    pub fn with_event<R>(
        &mut self,
        index: usize,
        f: impl for<'event> FnOnce(EventRef<'event, K>) -> R,
    ) -> Result<R, AraError> {
        let previous = self.previous_event(index)?;
        let index = self.check_index(index)?;
        // SAFETY: the backend's exclusive reader contract keeps this event readable until the next
        // backend call, which cannot occur while `f` holds the higher-ranked reference.
        let (pointer, extent) = unsafe { self.owner.backend.event_data(index)? };
        // SAFETY: validate the full event and all nested data before lending the pointer to safe code.
        let current = unsafe { copy_event_from_ffi::<K>(K::RAW_TYPE, pointer, extent) }?;
        if let Some(previous) = previous.as_ref() {
            K::validate_pair(previous, &current)?;
        }
        Ok(f(EventRef {
            pointer,
            extent,
            _event: PhantomData,
        }))
    }
}

impl<K: ContentKind, B: ContentReaderBackend> Iterator for ContentReader<K, B> {
    type Item = Result<K::Event, AraError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index == self.count {
            return None;
        }
        let index = self.next_index;
        self.next_index += 1;
        Some(self.event(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.next_index;
        (remaining, Some(remaining))
    }
}

impl<K: ContentKind, B: ContentReaderBackend> ExactSizeIterator for ContentReader<K, B> {}

/// An RAII reader retaining a runtime-discovered raw content type.
#[derive(Debug)]
pub struct DynamicContentReader<B: ContentReaderBackend = NoContentReaderBackend> {
    owner: ReaderOwner<B>,
    raw_type: ARAContentType,
    count: usize,
}

impl DynamicContentReader<NoContentReaderBackend> {
    /// Creates a dynamic reader from an exclusively owned backend.
    pub fn new<B: ContentReaderBackend>(backend: B) -> Result<DynamicContentReader<B>, AraError> {
        let mut owner = ReaderOwner { backend };
        let raw_type = owner.backend.raw_content_type();
        let count = checked_count(owner.backend.event_count()?)?;
        Ok(DynamicContentReader {
            owner,
            raw_type,
            count,
        })
    }
}

impl<B: ContentReaderBackend> DynamicContentReader<B> {
    /// Returns the preserved raw content type.
    pub fn raw_content_type(&self) -> ARAContentType {
        self.raw_type
    }

    /// Returns the checked event count.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns whether the reader contains no events.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns whether this reader has exactly the requested typed content kind.
    pub fn is<K: ContentKind>(&self) -> bool {
        self.raw_type == K::RAW_TYPE && K::validate_count(self.count).is_ok()
    }

    /// Downcasts after exact content-type and count validation.
    pub fn downcast<K: ContentKind>(self) -> Result<ContentReader<K, B>, Self> {
        if !self.is::<K>() {
            return Err(self);
        }
        Ok(ContentReader {
            owner: self.owner,
            count: self.count,
            next_index: 0,
            _kind: PhantomData,
        })
    }
}

fn checked_count(count: i32) -> Result<usize, AraError> {
    usize::try_from(count).map_err(|_| AraError::InvalidArgument("negative content event count"))
}
