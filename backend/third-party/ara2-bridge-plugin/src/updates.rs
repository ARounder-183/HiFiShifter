//! Reliable ARA 2.3 persistent-category update coalescing and delivery.

use ara2_bridge_core::{AraError, ContentTimeRange, ContentUpdateScopes, RawHandle};
use std::sync::{Arc, Mutex, MutexGuard};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Cloneable application handle for queuing persistent model changes until the next ARA flush.
#[derive(Clone, Default)]
pub struct UpdateEmitter {
    pending: Arc<Mutex<UpdateTracker>>,
}

impl UpdateEmitter {
    /// Creates an empty shared update queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks persistent audio-source state changed.
    pub fn mark_source(
        &self,
        source: RawHandle,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        origin: UpdateOrigin,
    ) -> Result<(), AraError> {
        lock(&self.pending).mark_source(source, range, flags, origin)
    }

    /// Marks persistent audio-modification state changed.
    pub fn mark_modification(
        &self,
        modification: RawHandle,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        origin: UpdateOrigin,
    ) -> Result<(), AraError> {
        lock(&self.pending).mark_modification(modification, range, flags, origin)
    }

    /// Marks persistent playback-region state changed.
    pub fn mark_region(
        &self,
        region: RawHandle,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        origin: UpdateOrigin,
    ) -> Result<(), AraError> {
        lock(&self.pending).mark_region(region, range, flags, origin)
    }

    /// Marks private persistent document state changed.
    pub fn mark_document(&self, origin: UpdateOrigin) {
        lock(&self.pending).mark_document(origin);
    }

    /// Returns the currently queued category count.
    pub fn pending_count(&self) -> usize {
        lock(&self.pending).pending_count()
    }

    pub(crate) fn take_pending(&self) -> UpdateTracker {
        std::mem::take(&mut *lock(&self.pending))
    }
}

/// Origin of a persistent-state mutation, used to suppress host echo notifications.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateOrigin {
    /// Application or analysis code produced a new persistent change.
    Application,
    /// Host-originated property/content input already known to the host.
    Host,
    /// State was copied directly from a host-provided archive.
    Restore,
    /// Recovery, conversion, or derived-state work produced an additional real change.
    Recovery,
}

impl UpdateOrigin {
    const fn should_notify(self) -> bool {
        matches!(self, Self::Application | Self::Recovery)
    }
}

#[derive(Clone, Debug)]
struct PendingContent {
    range: Option<ContentTimeRange>,
    flags: ContentUpdateScopes,
}

/// One owned notification ready for delivery during `notifyModelUpdates`.
#[derive(Clone, Debug)]
pub enum UpdateNotification {
    /// Persistent audio-source state changed.
    AudioSource {
        /// Runtime-owned audio-source identity.
        source: RawHandle,
        /// Coalesced changed range, or `None` for the entire object.
        range: Option<ContentTimeRange>,
        /// Scopes known to remain unchanged across every coalesced mutation.
        flags: ContentUpdateScopes,
    },
    /// Persistent audio-modification state changed.
    AudioModification {
        /// Runtime-owned audio-modification identity.
        modification: RawHandle,
        /// Coalesced changed range, or `None` for the entire object.
        range: Option<ContentTimeRange>,
        /// Scopes known to remain unchanged across every coalesced mutation.
        flags: ContentUpdateScopes,
    },
    /// Persistent playback-region state changed.
    PlaybackRegion {
        /// Runtime-owned playback-region identity.
        region: RawHandle,
        /// Coalesced changed range, or `None` for the entire object.
        range: Option<ContentTimeRange>,
        /// Scopes known to remain unchanged across every coalesced mutation.
        flags: ContentUpdateScopes,
    },
    /// Private persistent document data changed.
    Document,
}

/// Pending per-object/category state flushed only from `notifyModelUpdates`.
#[derive(Default)]
pub struct UpdateTracker {
    sources: Vec<(RawHandle, PendingContent)>,
    modifications: Vec<(RawHandle, PendingContent)>,
    regions: Vec<(RawHandle, PendingContent)>,
    document: bool,
}

impl UpdateTracker {
    /// Creates an empty update tracker.
    pub const fn new() -> Self {
        Self {
            sources: Vec::new(),
            modifications: Vec::new(),
            regions: Vec::new(),
            document: false,
        }
    }

    /// Marks persistent audio-source state changed.
    pub fn mark_source(
        &mut self,
        source: RawHandle,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        origin: UpdateOrigin,
    ) -> Result<(), AraError> {
        if origin.should_notify() {
            merge(&mut self.sources, source, range, flags)?;
        }
        Ok(())
    }

    /// Marks persistent audio-modification state changed.
    pub fn mark_modification(
        &mut self,
        modification: RawHandle,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        origin: UpdateOrigin,
    ) -> Result<(), AraError> {
        if origin.should_notify() {
            merge(&mut self.modifications, modification, range, flags)?;
        }
        Ok(())
    }

    /// Marks persistent playback-region state changed.
    pub fn mark_region(
        &mut self,
        region: RawHandle,
        range: Option<ContentTimeRange>,
        flags: ContentUpdateScopes,
        origin: UpdateOrigin,
    ) -> Result<(), AraError> {
        if origin.should_notify() {
            merge(&mut self.regions, region, range, flags)?;
        }
        Ok(())
    }

    /// Marks private persistent document state changed.
    pub fn mark_document(&mut self, origin: UpdateOrigin) {
        self.document |= origin.should_notify();
    }

    /// Returns the number of pending object/category notifications.
    pub fn pending_count(&self) -> usize {
        self.sources.len()
            + self.modifications.len()
            + self.regions.len()
            + usize::from(self.document)
    }

    /// Delivers the current batch in category order while retaining reentrant changes for later.
    pub fn flush_with(&mut self, mut deliver: impl FnMut(UpdateNotification, &mut UpdateTracker)) {
        let sources = std::mem::take(&mut self.sources);
        let modifications = std::mem::take(&mut self.modifications);
        let regions = std::mem::take(&mut self.regions);
        let document = std::mem::take(&mut self.document);
        for (source, pending) in sources {
            deliver(
                UpdateNotification::AudioSource {
                    source,
                    range: pending.range,
                    flags: pending.flags,
                },
                self,
            );
        }
        for (modification, pending) in modifications {
            deliver(
                UpdateNotification::AudioModification {
                    modification,
                    range: pending.range,
                    flags: pending.flags,
                },
                self,
            );
        }
        for (region, pending) in regions {
            deliver(
                UpdateNotification::PlaybackRegion {
                    region,
                    range: pending.range,
                    flags: pending.flags,
                },
                self,
            );
        }
        if document {
            deliver(UpdateNotification::Document, self);
        }
    }
}

fn merge(
    pending: &mut Vec<(RawHandle, PendingContent)>,
    object: RawHandle,
    range: Option<ContentTimeRange>,
    flags: ContentUpdateScopes,
) -> Result<(), AraError> {
    if let Some((_, existing)) = pending.iter_mut().find(|(handle, _)| *handle == object) {
        existing.range = match (existing.range.as_ref(), range.as_ref()) {
            (Some(existing), Some(incoming)) => {
                let start = existing.start().min(incoming.start());
                let end = (existing.start() + existing.duration())
                    .max(incoming.start() + incoming.duration());
                Some(ContentTimeRange::new(start, end - start)?)
            }
            _ => None,
        };
        existing.flags &= flags;
    } else {
        pending.push((object, PendingContent { range, flags }));
    }
    Ok(())
}
