//! Analysis-job lifecycle, progress ordering, and synchronous cancellation.

use ara2_bridge_core::{AraError, RawHandle};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Copy)]
pub(crate) enum PendingAnalysisProgress {
    Updated(RawHandle, f32),
    Completed(RawHandle),
}

/// Cloneable application handle for queuing ordered analysis progress to the ARA model thread.
#[derive(Clone, Default)]
pub struct AnalysisEmitter {
    pending: Arc<Mutex<Vec<PendingAnalysisProgress>>>,
}

impl AnalysisEmitter {
    /// Creates an empty progress queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a finite in-progress fraction; ordering is validated during model-thread delivery.
    pub fn update(&self, source: RawHandle, progress: f32) -> Result<(), AraError> {
        if !progress.is_finite() || !(0.0..1.0).contains(&progress) {
            return Err(AraError::InvalidArgument(
                "analysis progress must be finite and in 0..1",
            ));
        }
        lock(&self.pending).push(PendingAnalysisProgress::Updated(source, progress));
        Ok(())
    }

    /// Queues successful analysis completion.
    pub fn complete(&self, source: RawHandle) {
        lock(&self.pending).push(PendingAnalysisProgress::Completed(source));
    }

    pub(crate) fn take_pending(&self) -> Vec<PendingAnalysisProgress> {
        std::mem::take(&mut *lock(&self.pending))
    }

    pub(crate) fn cancel(&self, source: RawHandle) {
        lock(&self.pending).retain(|event| match event {
            PendingAnalysisProgress::Updated(event_source, _)
            | PendingAnalysisProgress::Completed(event_source) => *event_source != source,
        });
    }
}

/// Ordered analysis progress emitted toward the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnalysisProgress {
    /// The job started at zero progress.
    Started,
    /// The job advanced monotonically to the contained fraction.
    Updated(f32),
    /// The job completed at one.
    Completed,
    /// The job was synchronously cancelled because access or object lifetime ended.
    Cancelled,
}

struct Job {
    content_types: BTreeSet<i32>,
    progress: f32,
}

/// Model-thread analysis registry keyed by runtime-owned audio-source identity.
#[derive(Default)]
pub struct AnalysisCoordinator {
    jobs: HashMap<RawHandle, Job>,
}

impl AnalysisCoordinator {
    /// Starts one job and rejects empty, duplicate, or already-active requests.
    pub fn start(
        &mut self,
        source: RawHandle,
        content_types: impl IntoIterator<Item = i32>,
    ) -> Result<AnalysisProgress, AraError> {
        let content_types = content_types.into_iter().collect::<Vec<_>>();
        let unique = content_types.iter().copied().collect::<BTreeSet<_>>();
        if content_types.is_empty() || unique.len() != content_types.len() {
            return Err(AraError::InvalidArgument(
                "analysis content types must be nonempty and unique",
            ));
        }
        if self.jobs.contains_key(&source) {
            return Err(AraError::InvalidState(
                "audio source analysis is already active",
            ));
        }
        self.jobs.insert(
            source,
            Job {
                content_types: unique,
                progress: 0.0,
            },
        );
        Ok(AnalysisProgress::Started)
    }

    /// Advances one active job with finite monotonic progress below completion.
    pub fn update(
        &mut self,
        source: RawHandle,
        progress: f32,
    ) -> Result<AnalysisProgress, AraError> {
        let job = self.jobs.get_mut(&source).ok_or(AraError::InvalidState(
            "audio source analysis is not active",
        ))?;
        if !progress.is_finite() || !(job.progress..1.0).contains(&progress) {
            return Err(AraError::InvalidArgument(
                "analysis progress must increase and remain below one",
            ));
        }
        job.progress = progress;
        Ok(AnalysisProgress::Updated(progress))
    }

    /// Completes and removes one active analysis job.
    pub fn complete(&mut self, source: RawHandle) -> Result<AnalysisProgress, AraError> {
        self.jobs.remove(&source).ok_or(AraError::InvalidState(
            "audio source analysis is not active",
        ))?;
        Ok(AnalysisProgress::Completed)
    }

    /// Cancels one job synchronously before sample access is revoked or the source is destroyed.
    pub fn cancel(&mut self, source: RawHandle) -> Option<AnalysisProgress> {
        self.jobs
            .remove(&source)
            .map(|_| AnalysisProgress::Cancelled)
    }

    /// Returns whether a job currently includes a content type.
    pub fn contains(&self, source: RawHandle, content_type: i32) -> bool {
        self.jobs
            .get(&source)
            .is_some_and(|job| job.content_types.contains(&content_type))
    }
}
