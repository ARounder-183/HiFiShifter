//! Structured, bounded diagnostics independent of a logging framework.

use crate::AraError;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

/// Stable identity of a document session used in diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DocumentId(u64);

impl DocumentId {
    /// Creates an identity from a runtime-owned numeric value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the runtime-owned numeric value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable identity of a host, plug-in, or controller instance used in diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InstanceId(u64);

impl InstanceId {
    /// Creates an identity from a runtime-owned numeric value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the runtime-owned numeric value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One contextualized ARA failure suitable for deferred reporting.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    error: AraError,
    message: Arc<str>,
    interface: Option<&'static str>,
    method: Option<&'static str>,
    document: Option<DocumentId>,
    instance: Option<InstanceId>,
}

impl Diagnostic {
    /// Creates a diagnostic whose message is the error's display representation.
    pub fn new(error: AraError) -> Self {
        let message = Arc::<str>::from(error.to_string());
        Self {
            error,
            message,
            interface: None,
            method: None,
            document: None,
            instance: None,
        }
    }

    /// Attaches the static ABI interface and method names where the failure occurred.
    pub fn at(mut self, interface: &'static str, method: &'static str) -> Self {
        self.interface = Some(interface);
        self.method = Some(method);
        self
    }

    /// Attaches the affected document identity.
    pub fn with_document(mut self, document: DocumentId) -> Self {
        self.document = Some(document);
        self
    }

    /// Attaches the affected runtime instance identity.
    pub fn with_instance(mut self, instance: InstanceId) -> Self {
        self.instance = Some(instance);
        self
    }

    /// Replaces the display message with owned contextual text.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Arc::<str>::from(message.into());
        self
    }

    /// Returns the typed failure category.
    pub fn error(&self) -> &AraError {
        &self.error
    }

    /// Returns the deferred display message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the ABI interface name, when known.
    pub fn interface(&self) -> Option<&'static str> {
        self.interface
    }

    /// Returns the ABI method name, when known.
    pub fn method(&self) -> Option<&'static str> {
        self.method
    }

    /// Returns the document identity, when attached.
    pub fn document(&self) -> Option<DocumentId> {
        self.document
    }

    /// Returns the runtime instance identity, when attached.
    pub fn instance(&self) -> Option<InstanceId> {
        self.instance
    }
}

/// Thread-safe destination for deferred ARA diagnostics.
pub trait DiagnosticSink: Send + Sync {
    /// Records one diagnostic without making logging a correctness dependency.
    fn record(&self, diagnostic: Diagnostic);
}

/// A bounded first-in, first-out diagnostic sink.
///
/// When full, recording a new entry evicts the oldest entry. A poisoned synchronization primitive
/// does not disable diagnostic recovery; the contained queue is recovered and remains bounded.
#[derive(Debug)]
pub struct BoundedDiagnosticSink {
    capacity: usize,
    entries: Mutex<VecDeque<Diagnostic>>,
}

impl BoundedDiagnosticSink {
    /// Default maximum number of retained diagnostics.
    pub const DEFAULT_CAPACITY: usize = 256;

    /// Creates a bounded sink.
    pub fn new(capacity: usize) -> Result<Self, AraError> {
        if capacity == 0 {
            return Err(AraError::InvalidArgument(
                "diagnostic capacity must be nonzero",
            ));
        }
        Ok(Self {
            capacity,
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
        })
    }

    /// Returns the maximum number of retained diagnostics.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns an ordered snapshot from oldest to newest.
    pub fn snapshot(&self) -> Vec<Diagnostic> {
        self.lock_entries().iter().cloned().collect()
    }

    /// Removes every retained diagnostic.
    pub fn clear(&self) {
        self.lock_entries().clear();
    }

    fn lock_entries(&self) -> MutexGuard<'_, VecDeque<Diagnostic>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for BoundedDiagnosticSink {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY).expect("default capacity is nonzero")
    }
}

impl DiagnosticSink for BoundedDiagnosticSink {
    fn record(&self, diagnostic: Diagnostic) {
        let mut entries = self.lock_entries();
        if entries.len() == self.capacity {
            entries.pop_front();
        }
        entries.push_back(diagnostic);
    }
}
