//! Allocation-free immutable realtime query data and deferred failure codes.

use crate::AraError;
use crossbeam_queue::ArrayQueue;

/// One immutable playback-region head/tail result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadTailEntry {
    key: u64,
    head: f64,
    tail: f64,
}

impl HeadTailEntry {
    /// Creates a finite entry with nonnegative head and tail durations.
    pub fn new(key: u64, head: f64, tail: f64) -> Result<Self, AraError> {
        if !head.is_finite() || !tail.is_finite() || head < 0.0 || tail < 0.0 {
            return Err(AraError::InvalidArgument(
                "head and tail must be finite and nonnegative",
            ));
        }
        Ok(Self { key, head, tail })
    }
}

/// Immutable bounded snapshot used by realtime/offline head-and-tail queries.
#[derive(Debug)]
pub struct RealtimeHeadTailView {
    entries: Box<[HeadTailEntry]>,
}

impl RealtimeHeadTailView {
    /// Builds and sorts a snapshot outside the realtime path.
    pub fn new(
        entries: impl IntoIterator<Item = HeadTailEntry>,
        capacity: usize,
    ) -> Result<Self, AraError> {
        let mut entries: Vec<_> = entries.into_iter().collect();
        if entries.len() > capacity {
            return Err(AraError::InvalidArgument(
                "realtime snapshot exceeds configured capacity",
            ));
        }
        entries.sort_unstable_by_key(|entry| entry.key);
        if entries.windows(2).any(|pair| pair[0].key == pair[1].key) {
            return Err(AraError::InvalidArgument(
                "realtime snapshot contains duplicate region",
            ));
        }
        Ok(Self {
            entries: entries.into_boxed_slice(),
        })
    }

    /// Performs an allocation-free, lock-free bounded lookup.
    pub fn query(&self, key: u64) -> Option<(f64, f64)> {
        self.entries
            .binary_search_by_key(&key, |entry| entry.key)
            .ok()
            .map(|index| {
                let entry = self.entries[index];
                (entry.head, entry.tail)
            })
    }

    /// Returns the number of entries in the immutable snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the snapshot contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Fixed-size failure categories safe to enqueue from realtime paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RealtimeFailureCode {
    /// A queried playback region is absent from the current snapshot.
    MissingRegion,
    /// The realtime operation observed an invalid lifecycle state.
    InvalidState,
    /// A foreign peer operation failed.
    PeerFailure,
}

/// Preallocated bounded lock-free queue for deferred diagnostic expansion.
#[derive(Debug)]
pub struct RealtimeFailureQueue {
    queue: ArrayQueue<RealtimeFailureCode>,
}

impl RealtimeFailureQueue {
    /// Allocates a fixed-capacity queue outside the realtime path.
    pub fn new(capacity: usize) -> Result<Self, AraError> {
        if capacity == 0 {
            return Err(AraError::InvalidArgument(
                "realtime failure capacity must be nonzero",
            ));
        }
        Ok(Self {
            queue: ArrayQueue::new(capacity),
        })
    }

    /// Enqueues a fixed-size code without allocating or blocking.
    ///
    /// Returns false when the bounded queue is full.
    pub fn report(&self, code: RealtimeFailureCode) -> bool {
        self.queue.push(code).is_ok()
    }

    /// Removes the oldest code for later model-thread expansion.
    pub fn pop(&self) -> Option<RealtimeFailureCode> {
        self.queue.pop()
    }

    /// Returns the configured queue capacity.
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }
}
