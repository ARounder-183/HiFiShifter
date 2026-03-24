use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use super::io::decode_resampled_stereo;
use super::types::{AudioKey, EngineCommand, ResampledStereo};

pub(crate) struct ResourceManager {
    cache: Arc<Mutex<HashMap<AudioKey, ResampledStereo>>>,
    inflight: Arc<Mutex<HashSet<AudioKey>>>,
    request_tx: mpsc::Sender<AudioKey>,
}

impl ResourceManager {
    pub(crate) fn new(engine_tx: mpsc::Sender<EngineCommand>) -> Self {
        let cache: Arc<Mutex<HashMap<AudioKey, ResampledStereo>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let inflight: Arc<Mutex<HashSet<AudioKey>>> = Arc::new(Mutex::new(HashSet::new()));

        let (request_tx, request_rx) = mpsc::channel::<AudioKey>();

        {
            let cache_for_worker = cache.clone();
            let inflight_for_worker = inflight.clone();
            thread::spawn(move || {
                // 在进入循环前只读取一次环境变量
                let debug_commands =
                    std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");

                while let Ok(key) = request_rx.recv() {
                    let (path, out_rate) = key.clone();
                    let ok = decode_resampled_stereo(path.as_path(), out_rate)
                        .and_then(|v| {
                            if let Ok(mut m) = cache_for_worker.lock() {
                                m.insert(key.clone(), v);
                                Some(())
                            } else {
                                None
                            }
                        })
                        .is_some();

                    // 直接使用布尔值判断，消除系统调用
                    if !ok && debug_commands {
                        eprintln!(
                            "AudioEngine: ResourceManager decode failed: path={} out_rate={} ",
                            path.display(),
                            out_rate
                        );
                    }

                    if let Ok(mut s) = inflight_for_worker.lock() {
                        s.remove(&key);
                    }
                    if ok {
                        let _ = engine_tx.send(EngineCommand::AudioReady { key });
                    }
                }
            });
        }

        Self {
            cache,
            inflight,
            request_tx,
        }
    }

    pub(crate) fn cache(&self) -> &Arc<Mutex<HashMap<AudioKey, ResampledStereo>>> {
        &self.cache
    }

    pub(crate) fn get_or_request(&self, path: &Path, out_rate: u32) -> Option<ResampledStereo> {
        // 先生成 key
        let key: AudioKey = (PathBuf::from(path), out_rate);

        // 1. 优先查询内存缓存，纯内存操作
        if let Ok(m) = self.cache.lock() {
            if let Some(v) = m.get(&key) {
                return Some(v.clone());
            }
        }

        // 2. 只有在缓存未命中时，才去执行硬盘状态查询
        if !path.exists() {
            return None;
        }

        let should_enqueue = if let Ok(mut s) = self.inflight.lock() {
            if s.contains(&key) {
                false
            } else {
                s.insert(key.clone());
                true
            }
        } else {
            false
        };

        if should_enqueue {
            let _ = self.request_tx.send(key);
        }

        None
    }
}
