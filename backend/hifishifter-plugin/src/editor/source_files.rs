use crate::render::source::SourcePcm;
use crate::sync::MutexExt;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex, Weak,
};
use std::thread::JoinHandle;

type Paths = HashMap<String, String>;
struct Ready {
    source: Weak<SourcePcm>,
    path: String,
}
struct Job {
    sources: Vec<(String, Weak<SourcePcm>)>,
    result: mpsc::Receiver<Result<Paths, String>>,
    cancel: Arc<AtomicBool>,
    worker: JoinHandle<()>,
}
#[derive(Default)]
struct State {
    ready: HashMap<String, Ready>,
    job: Option<Job>,
}
#[derive(Default)]
pub(super) struct SourceFiles {
    state: Mutex<State>,
}

impl SourceFiles {
    pub(super) fn pending(&self) -> bool {
        self.state.lock_or_recover().job.is_some()
    }
    pub(super) fn finished(&self) -> bool {
        self.state
            .lock_or_recover()
            .job
            .as_ref()
            .is_some_and(|job| job.worker.is_finished())
    }
    pub(super) fn prepare(
        &self,
        sources: &[(String, Arc<SourcePcm>)],
        dir: &Path,
        limit: usize,
        wake: impl FnOnce() + Send + 'static,
    ) -> Result<Option<Paths>, String> {
        let mut state = self.state.lock_or_recover();
        if state
            .job
            .as_ref()
            .is_some_and(|job| job.worker.is_finished())
        {
            let job = state.job.take().unwrap();
            let paths = job
                .result
                .recv()
                .map_err(|_| "source file worker ended".to_owned())??;
            for (id, source) in job.sources {
                if let Some(path) = paths.get(&id) {
                    state.ready.insert(
                        id,
                        Ready {
                            source,
                            path: path.clone(),
                        },
                    );
                }
            }
        }
        let ids: HashSet<_> = sources.iter().map(|(id, _)| id.as_str()).collect();
        state.ready.retain(|id, _| ids.contains(id.as_str()));
        let mut paths = HashMap::new();
        let mut missing = Vec::new();
        for (id, source) in sources {
            let ready = state.ready.get(id).filter(|entry| {
                entry
                    .source
                    .upgrade()
                    .is_some_and(|old| Arc::ptr_eq(&old, source))
            });
            if let Some(ready) = ready {
                paths.insert(id.clone(), ready.path.clone());
            } else {
                missing.push((id.clone(), source.clone()));
            }
        }
        if missing.is_empty() {
            return Ok(Some(paths));
        }
        if state.job.is_none() {
            #[cfg(all(windows, not(test)))]
            super::resources::pin_analysis_code()?;
            let weak = missing
                .iter()
                .map(|(id, pcm)| (id.clone(), Arc::downgrade(pcm)))
                .collect();
            let directory: PathBuf = dir.into();
            let cancel = Arc::new(AtomicBool::new(false));
            let stopped = cancel.clone();
            let (tx, rx) = mpsc::channel();
            let worker = std::thread::Builder::new()
                .name("hfs-source-files".into())
                .spawn(move || {
                    let _span = crate::startup::Span::new("source.files", 0, missing.len() as u64);
                    let views: Vec<_> = missing
                        .iter()
                        .map(|(id, pcm)| hifishifter_kernel::editor::host_pcm::PcmView {
                            persistent_id: id,
                            sample_rate: pcm.sample_rate,
                            planes: &pcm.planes,
                        })
                        .collect();
                    let result = hifishifter_kernel::editor::host_pcm::materialize_cancellable(
                        Default::default(),
                        &views,
                        &directory,
                        limit,
                        hifishifter_kernel::editor::host_pcm::ReversePolicy::Materialize,
                        &|| stopped.load(Ordering::Acquire),
                    )
                    .map(|(_, reverse)| reverse.into_iter().map(|(path, id)| (id, path)).collect());
                    let _ = tx.send(result);
                    wake();
                })
                .map_err(|e| e.to_string())?;
            state.job = Some(Job {
                sources: weak,
                result: rx,
                cancel,
                worker,
            });
        }
        Ok(None)
    }
    pub(super) fn cancel(&self) {
        if let Some(job) = self.state.lock_or_recover().job.take() {
            job.cancel.store(true, Ordering::Release);
        }
    }
}
impl Drop for SourceFiles {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_files_are_prepared_off_thread_and_replaced_pcm_does_not_reuse_a_path() {
        let dir = std::env::temp_dir().join(format!("hfs-source-files-{}", std::process::id()));
        let files = SourceFiles::default();
        let pcm = |value| {
            Arc::new(SourcePcm {
                sample_rate: 44100,
                planes: vec![vec![value; 10000]],
                version: 1,
                covered: vec![],
                _reservation: None,
            })
        };
        let first = vec![("source".to_owned(), pcm(0.1))];
        assert!(files
            .prepare(&first, &dir, 16 * 1024 * 1024, || {})
            .unwrap()
            .is_none());
        let await_paths = |sources: &[(String, Arc<SourcePcm>)]| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                if let Some(paths) = files
                    .prepare(sources, &dir, 16 * 1024 * 1024, || {})
                    .unwrap()
                {
                    break paths;
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        let original = await_paths(&first);
        let second = vec![("source".to_owned(), pcm(0.2))];
        assert!(files
            .prepare(&second, &dir, 16 * 1024 * 1024, || {})
            .unwrap()
            .is_none());
        assert_ne!(await_paths(&second)["source"], original["source"]);
    }
}
