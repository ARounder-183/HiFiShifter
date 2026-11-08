use super::{document::DocumentSession, source::SourcePcm};
use ara2_bridge::plugin::{HostAudioReader, HostContentScope};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Condvar, Mutex, OnceLock,
};

type Task = Box<dyn FnOnce() + Send + 'static>;
fn lane() -> Result<&'static mpsc::SyncSender<Task>, String> {
    static LANE: OnceLock<Result<mpsc::SyncSender<Task>, String>> = OnceLock::new();
    LANE.get_or_init(|| {
        #[cfg(all(windows, not(test)))]
        crate::editor::resources::pin_analysis_code()?;
        let (tx, rx) = mpsc::sync_channel::<Task>(256);
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..2 {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name("hfs-source-read".into())
                .spawn(move || loop {
                    let task = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    match task {
                        Ok(task) => {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
                        }
                        Err(_) => break,
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(tx)
    })
    .as_ref()
    .map_err(Clone::clone)
}

pub(crate) struct SourceRead {
    pub version: u64,
    frames: usize,
    channels: usize,
    sample_rate: u32,
    cancel: Arc<AtomicBool>,
    done: Arc<(Mutex<bool>, Condvar)>,
    // Keep the last reader reference on the model thread until a legal SDK scope.
    _reader: Arc<Mutex<HostAudioReader<f32>>>,
}
impl SourceRead {
    pub(crate) fn start(
        host: &HostContentScope<'_, '_>,
        document: &Arc<DocumentSession>,
        id: String,
        frames: usize,
        channels: usize,
        sample_rate: u32,
        version: u64,
    ) -> Result<Option<Self>, String> {
        if frames == 0 || !(1..=2).contains(&channels) || !(8000..=384000).contains(&sample_rate) {
            return Err("invalid source geometry".into());
        }
        let source = host.current_audio_source().ok_or("missing source scope")?;
        let reader = Arc::new(Mutex::new(
            host.audio_reader::<f32>(source, channels)
                .map_err(|e| format!("{e:?}"))?,
        ));
        let cancel = Arc::new(AtomicBool::new(false));
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let task_reader = reader.clone();
        let stopped = cancel.clone();
        let completed = done.clone();
        let weak = Arc::downgrade(document);
        let task: Task = Box::new(move || {
            struct Finish(Arc<(Mutex<bool>, Condvar)>);
            impl Drop for Finish {
                fn drop(&mut self) {
                    let (lock, wake) = &*self.0;
                    *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
                    wake.notify_all();
                }
            }
            let _finished = Finish(completed);
            let task_reader = task_reader;
            let _span = crate::startup::Span::new("ara.async_source_read", 0, frames as u64);
            let result = (|| -> Result<SourcePcm, String> {
                if stopped.load(Ordering::Acquire) {
                    return Err("source read cancelled".into());
                }
                let bytes = frames
                    .checked_mul(channels)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or("PCM size overflow")?;
                let reservation = super::budget::global_budget()
                    .reserve(bytes)
                    .ok_or("PCM budget exceeded")?;
                let mut planes = Vec::with_capacity(channels);
                for _ in 0..channels {
                    let mut plane = Vec::new();
                    plane.try_reserve_exact(frames).map_err(|e| e.to_string())?;
                    plane.resize(frames, 0.0);
                    planes.push(plane);
                }
                for start in (0..frames).step_by(4096) {
                    if stopped.load(Ordering::Acquire)
                        || weak.upgrade().is_none_or(|doc| !doc.is_alive())
                    {
                        return Err("source read cancelled".into());
                    }
                    let end = (start + 4096).min(frames);
                    let mut buffers = planes
                        .iter_mut()
                        .map(|p| &mut p[start..end])
                        .collect::<Vec<_>>();
                    task_reader
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .read(start as i64, &mut buffers)
                        .map_err(|e| format!("{e:?}"))?;
                }
                if planes.iter().flatten().any(|n| !n.is_finite()) {
                    return Err("host returned nonfinite PCM".into());
                }
                Ok(SourcePcm {
                    sample_rate,
                    planes,
                    version,
                    covered: vec![],
                    _reservation: Some(reservation),
                })
            })();
            drop(task_reader);
            if let Some(document) = weak.upgrade() {
                document.publish_source_read_result(id, result, &stopped);
            }
        });
        match lane()?.try_send(task) {
            Ok(()) => Ok(Some(Self {
                version,
                frames,
                channels,
                sample_rate,
                cancel,
                done,
                _reader: reader,
            })),
            Err(mpsc::TrySendError::Full(_)) => Ok(None),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err("source read workers unavailable".into())
            }
        }
    }
    pub(crate) fn matches(
        &self,
        version: u64,
        frames: usize,
        channels: usize,
        sample_rate: u32,
    ) -> bool {
        self.version == version
            && self.frames == frames
            && self.channels == channels
            && self.sample_rate == sample_rate
    }
    pub(crate) fn finished(&self) -> bool {
        *self.done.0.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(crate) fn cancel(&self, document: &DocumentSession) {
        self.cancel.store(true, Ordering::Release);
        drop(crate::sync::MutexExt::lock_or_recover(
            &document.transaction,
        ));
    }
    pub(crate) fn wait(&self) {
        self.cancel.store(true, Ordering::Release);
        let (lock, wake) = &*self.done;
        let mut done = lock.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = wake.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ara2_bridge::{
        core::ApiGeneration,
        plugin::{HostAudioSourceRef, HostClients},
    };
    fn await_read(job: &SourceRead) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !job.finished() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    #[test]
    fn background_reader_publishes_complete_pcm_and_destroys_in_a_source_scope() {
        let mut fixture = crate::test_host::HostFixture::new(vec![vec![0.25; 100000]]);
        let host = fixture.instance();
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0u8;
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        let document = DocumentSession::new(999001);
        let job = clients
            .with_audio_source_management(source, |scope| {
                SourceRead::start(&scope, &document, "source".into(), 100000, 1, 44100, 7)
            })
            .unwrap()
            .unwrap();
        await_read(&job);
        assert_eq!(
            document.edit_sources.lock().unwrap()["source"].planes[0].last(),
            Some(&0.25)
        );
        assert_eq!(fixture.destroyed.load(Ordering::Relaxed), 0);
        clients.with_audio_source_management(source, |_| drop(job));
        assert_eq!(fixture.destroyed.load(Ordering::Relaxed), 1);
        document.close();
    }
    #[test]
    fn cancelled_reader_cannot_publish_after_the_revoke_fence() {
        let mut fixture = crate::test_host::HostFixture::new(vec![vec![0.5; 100000]]);
        let host = fixture.instance();
        let clients = unsafe { HostClients::from_raw(&host, ApiGeneration::V2Final) }.unwrap();
        let mut identity = 0u8;
        let source = unsafe { HostAudioSourceRef::from_raw((&raw mut identity).cast()) }.unwrap();
        let document = DocumentSession::new(999002);
        let job = clients
            .with_audio_source_management(source, |scope| {
                SourceRead::start(&scope, &document, "source".into(), 100000, 1, 44100, 8)
            })
            .unwrap()
            .unwrap();
        job.cancel(&document);
        document.edit_sources.lock().unwrap().clear();
        document.sources.lock().unwrap().clear();
        await_read(&job);
        assert!(!document.edit_sources.lock().unwrap().contains_key("source"));
        clients.with_audio_source_management(source, |_| drop(job));
        document.close();
    }
}
