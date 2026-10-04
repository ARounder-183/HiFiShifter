//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。

use super::ownership::{region_owners, RegionKey};
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

/// 全PCM内容参与身份，重开时同路径不同音频不能命中旧缓存。
fn pcm_fingerprint(pcm: &super::source::SourcePcm) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(&pcm.sample_rate.to_le_bytes());
    hash.update(&(pcm.planes.len() as u64).to_le_bytes());
    for plane in &pcm.planes { for sample in plane { hash.update(&sample.to_le_bytes()); } }
    hash.finalize().to_hex().to_string()
}

/// 仅模型线程访问；音频线程将在 Task 14 消费已发布快照，不能锁此 owner。
#[derive(Default)]
pub(crate) struct ExtensionOwner {
    binding: Mutex<Option<ExtensionBinding>>,
    document: Mutex<Option<std::sync::Weak<super::document::DocumentSession>>>,
    assignments: Mutex<HashMap<i32, Vec<RegionKey>>>,
    sequences: Mutex<HashMap<i32, Vec<u64>>>,
    role: AtomicI32,
    pub snapshots: [super::snapshot::SnapshotPublisher; 2],
    pub(crate) edits: Arc<Mutex<crate::state_channel::EditState>>,
    channel: Mutex<Option<hifishifter_ara_ipc::Server>>,
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::ara_entry::HostEntry;
    use ara2_bridge::companion::vst3::ffi::{ara2_vst3_plugin_entry_bind, ARA2_VST3_OK};
    use ara2_bridge::companion::{CompanionFactory, CompanionProcessorBinding, CompanionRoles};
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;

    #[test]
    fn gui_commit_changes_assigned_pcm_and_persisted_state_restores_it() {
        use crate::render::source::SourcePcm;
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let mut timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],
            "clips":[{"id":"one","track_id":"track","name":"one","start_sec":0,"length_sec":4.0/44100.0,
                "takes":[{"id":"take","name":"one","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]},
                {"id":"other","track_id":"track","name":"other","start_sec":1,"length_sec":4.0/44100.0,
                "takes":[{"id":"take2","name":"other","source_path":"ara://pcm","source_start_sec":0,"source_end_sec":4.0/44100.0}]}],
            "bpm":120,"project_sec":2
        })).unwrap();
        for clip in &mut timeline.clips { clip.normalize_takes(); }
        *document.timeline.lock().unwrap() = Some(timeline);
        document.edit_sources.lock().unwrap().insert("ara://pcm".into(), Arc::new(SourcePcm { sample_rate:44100,
            planes:vec![vec![0.1,0.2,0.3,0.4]], version:0, _reservation:None }));
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners().lock().unwrap().register(key, document.id, 0).unwrap();
        document.clip_ids.lock().unwrap().insert(key, "one".into());
        document.regions.lock().unwrap().insert(key, crate::ara::AraPlaybackRegion {
            audio_source_persistent_id:"ara://pcm".into(), duration_in_modification_time:4.0/44100.0,
            duration_in_playback_time:4.0/44100.0, ..Default::default() });
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner.bind_to_document(document.clone(), ApiGeneration::V2Final, ExtensionRoles::all(), ExtensionRoles::EDITOR_RENDERER, None).unwrap();
        unsafe { let ext = &*raw; ((*ext.editorRendererInterface).addPlaybackRegion.unwrap())(ext.editorRendererRef, key as *mut _); }
        let snapshot = owner.handle_request(hifishifter_ara_ipc::Request::Snapshot);
        assert!(snapshot.ok, "{:?}", snapshot.error);
        let mut client = snapshot.timeline.unwrap();
        assert_eq!(client["clips"].as_array().unwrap().len(), 1);
        client["tracks"][0]["volume"] = serde_json::json!(0.5);
        let response = owner.handle_request(hifishifter_ara_ipc::Request::Commit {
            base_revision:snapshot.revision, model_revision:snapshot.model_revision, timeline:client });
        assert!(response.ok, "{:?}", response.error);
        let companion = Arc::new(ExtensionOwner::default());
        companion.bind_to_document(document.clone(), ApiGeneration::V2Final, ExtensionRoles::all(), ExtensionRoles::EDITOR_RENDERER, None).unwrap();
        assert_eq!(companion.edit_state().lock().unwrap().revision, response.revision);
        assert_eq!(companion.edit_state().lock().unwrap().tracks[0].volume, 0.5);
        let state = owner.edit_state().lock().unwrap().encode().unwrap();
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(&state).unwrap();
        let output = owner.render_edits(&document, &restored).unwrap();
        assert_eq!(output[0].left.len(), 4);
        for (actual, expected) in output[0].left.iter().zip([0.05_f32,0.1,0.15,0.2]) { assert!((*actual-expected).abs()<1e-6); }
        assert!(restored.restore(b"bad state").is_err());
        drop(model);
    }

    #[test]
    fn gui_request_commits_parameters_and_rejects_stale_document_revision() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let timeline: hifishifter_kernel::state::TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"host","order":0}],"clips":[],"bpm":120,"project_sec":0
        })).unwrap();
        *document.timeline.lock().unwrap() = Some(timeline);
        document.ready.store(true, Ordering::Release);
        let owner = Arc::new(ExtensionOwner::default());
        owner.bind_to_document(document.clone(), ApiGeneration::V2Final, ExtensionRoles::all(), ExtensionRoles::EDITOR_RENDERER, None).unwrap();
        let snapshot = owner.handle_request(hifishifter_ara_ipc::Request::Snapshot);
        assert!(snapshot.ok, "{:?}", snapshot.error);
        let mut client = snapshot.timeline.unwrap();
        client["tracks"][0]["volume"] = serde_json::json!(0.5);
        let commit = hifishifter_ara_ipc::Request::Commit { base_revision: snapshot.revision, model_revision: snapshot.model_revision, timeline: client };
        let response = owner.handle_request(commit.clone());
        assert!(response.ok, "{:?}", response.error);
        assert_eq!(response.revision, 1);
        assert!(!owner.handle_request(commit).ok);
        document.clear_renderers();
        assert!(!owner.handle_request(hifishifter_ara_ipc::Request::Snapshot).ok);
        drop(model);
    }

    /// 真实空文档绑定也必须收到销毁；不依赖 first region assignment 猜 document。
    #[test]
    fn bound_native_entry_is_tombstoned_by_actual_document_teardown() {
        let factory = Box::leak(Box::new(
            FactoryBuilder::new("org.hfs.bound", "org.hfs.bound.archive")
                .display("bound", "HiFiShifter", "https://example.invalid", "1")
                .document_controller(|| {
                    let model = crate::ara::model::ModelHandle::new();
                    let session = model.session();
                    PluginBuilder::new(model)
                        .controller_identity(move |key| session.register(key))
                        .build()
                })
                .build()
                .unwrap(),
        ));
        factory
            .entry()
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
        let host = fixture.instance();
        let properties = ARADocumentProperties {
            structSize: std::mem::size_of::<ARADocumentProperties>(),
            name: c"bound empty".as_ptr(),
        };
        // SAFETY: test host、factory 和 properties 保持到实际 controller 终止。
        let raw = unsafe {
            (factory
                .raw_copy()
                .createDocumentControllerWithDocument
                .unwrap())(&host, &properties)
        };
        assert!(!raw.is_null());
        // SAFETY: 原生工厂返回完整 packed 实例。
        let controller = unsafe { raw.read_unaligned() };
        let document = crate::render::document::DocumentSession::lookup(
            controller.documentControllerRef as usize,
        )
        .unwrap();
        // SAFETY: test factory backing 已保留到进程结束。
        let association =
            unsafe { CompanionFactory::from_raw("bound", &*factory.as_raw()) }.unwrap();
        let processor =
            CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
        let probe = processor.lifetime_probe();
        let owner = Arc::new(ExtensionOwner::default());
        let entry = HostEntry::new(processor, "bound", owner.clone()).unwrap();
        let mut extension_raw = std::ptr::null();
        // SAFETY: real C++ shim 调真实产品 bind，controller 存活且已登记。
        assert_eq!(
            unsafe {
                ara2_vst3_plugin_entry_bind(
                    entry.as_raw(),
                    controller.documentControllerRef.cast(),
                    7,
                    1,
                    1,
                    &raw mut extension_raw,
                )
            },
            ARA2_VST3_OK
        );
        assert!(!extension_raw.is_null());
        assert!(probe.controller_alive());
        let identity = Box::new(0_u8);
        let key = (&*identity as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 0)
            .unwrap();
        // SAFETY: extension 与 controller 保留；region 仅作为不透明身份。
        unsafe {
            let extension = extension_raw
                .cast::<ARAPlugInExtensionInstance>()
                .read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, key as *mut _);
        }
        assert_eq!(owner.assignments.lock().unwrap().get(&1).unwrap(), &[key]);
        // SAFETY: actual controller destructor invokes ModelHandle::destroy_document。
        unsafe {
            ((*controller.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(controller.documentControllerRef)
        };
        assert!(owner.assignments.lock().unwrap().is_empty());
        assert!(owner.assigned_regions().is_err());
        assert!(!probe.controller_alive());
        // SAFETY: native entry 仍保留 companion storage；销毁文档后回调必须被 tombstone 拒绝。
        unsafe {
            let extension = extension_raw
                .cast::<ARAPlugInExtensionInstance>()
                .read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, key as *mut _);
        }
        assert!(owner.assignments.lock().unwrap().is_empty());
        drop(entry);
        drop(owner);
        factory.entry().uninitialize().unwrap();
    }

    /// sequence 预览只展开本 sequence；显式区域重复分配不能导致双倍混音。
    #[test]
    fn editor_sequence_selection_is_expanded_and_deduplicated() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        let region_a = Box::new(0_u8);
        let region_b = Box::new(0_u8);
        let sequence = Box::new(0_u8);
        let a = (&*region_a as *const u8) as u64;
        let b = (&*region_b as *const u8) as u64;
        let seq = (&*sequence as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(a, document.id, 0)
            .unwrap();
        region_owners()
            .lock()
            .unwrap()
            .register(b, document.id, 1)
            .unwrap();
        document
            .sequence_regions
            .lock()
            .unwrap()
            .insert(seq, [a, b].into_iter().collect());
        // SAFETY: binding、session 和不透明身份在调用期间均存活。
        unsafe {
            let extension = &*raw;
            let api = &*extension.editorRendererInterface;
            api.addPlaybackRegion.unwrap()(extension.editorRendererRef, a as *mut _);
            api.addRegionSequence.unwrap()(extension.editorRendererRef, seq as *mut _);
        }
        let mut expected = vec![a, b];
        expected.sort_unstable();
        assert_eq!(owner.assigned_regions().unwrap(), expected);
        document
            .sequence_regions
            .lock()
            .unwrap()
            .get_mut(&seq)
            .unwrap()
            .remove(&b);
        region_owners().lock().unwrap().remove(b);
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        // SAFETY: lease 保留接口，remove sequence 不会移除显式 region。
        unsafe {
            let extension = &*raw;
            ((*extension.editorRendererInterface)
                .removeRegionSequence
                .unwrap())(extension.editorRendererRef, seq as *mut _);
        }
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        drop(model);
        assert!(owner.assigned_regions().is_err());
    }

    /// companion 先销毁时 controller lease 必须独立保留真实 raw storage。
    #[test]
    fn actual_document_retains_bound_storage_after_owner_is_dropped() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document,
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .unwrap();
        drop(owner);
        let mut identity = 0_u8;
        // SAFETY: actual model 的 session 独立持有控制器 lease，owner 已 drop 也不释放 storage。
        unsafe {
            let extension = raw.read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
            ((*extension.playbackRendererInterface)
                .removePlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
        }
        drop(model);
        // 模型终止后所有 owner 都已释放，此后不得再访问 raw。
    }
}

impl ExtensionOwner {
    /// 组件释放/文档关闭前结束后台服务，避免DLL卸载后线程仍执行插件代码。
    pub fn stop_channel(&self) { let channel = self.channel.lock().unwrap().take(); drop(channel); }

    /// 未绑定时暂存组件state；绑定后所有处理器读取同一文档的参数权威。
    pub fn edit_state(&self) -> Arc<Mutex<crate::state_channel::EditState>> {
        self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade)
            .map(|document| document.edits.clone()).unwrap_or_else(|| self.edits.clone())
    }

    /// 恢复/undo组件状态后刷新整张文档，各renderer保持原分配。
    pub fn refresh_document(&self) {
        let document = self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade);
        if let Some(document) = document { for owner in document.renderer_owners() { owner.prepare(); } } else { self.prepare(); }
    }

    /// 接收非实时编辑请求；具体处理只在文档事务锁内进行。
    pub(crate) fn handle_request(&self, request: hifishifter_ara_ipc::Request) -> hifishifter_ara_ipc::Response {
        use hifishifter_ara_ipc::{Request, Response, HostPcm};
        let document = self.document.lock().unwrap().as_ref().and_then(std::sync::Weak::upgrade);
        let Some(document) = document else { return Response { error: Some("document closed".into()), ..Default::default() }; };
        let _transaction = document.transaction.lock().unwrap();
        let model_revision = document.revision.load(Ordering::Acquire);
        let mut edits = document.edits.lock().unwrap();
        let outcome = (|| {
            if !document.ready.load(Ordering::Acquire) { return Err("host model not ready; refresh after editing".into()); }
            let mut timeline = self.assigned_timeline(&document)?;
            match request {
                Request::Snapshot => {
                    edits.apply(&mut timeline);
                    let available = document.edit_sources.lock().unwrap();
                    let mut sources = Vec::new();
                    for id in timeline.clips.iter().filter_map(|clip| clip.source_path.as_ref()).collect::<BTreeSet<_>>() {
                        let pcm = available.get(id).ok_or_else(|| format!("host PCM unavailable: {id}"))?;
                        sources.push(HostPcm { persistent_id: id.clone(), sample_rate: pcm.sample_rate,
                            fingerprint: pcm_fingerprint(pcm), planes: pcm.planes.clone() });
                    }
                    Ok(Response { ok: true, timeline: Some(serde_json::to_value(&timeline).map_err(|e| e.to_string())?), sources, ..Default::default() })
                }
                Request::Commit { base_revision, model_revision: base_model, timeline: client } => {
                    if base_model != model_revision { return Err("Conflict: host model changed; refresh".into()); }
                    let client = serde_json::from_value(client).map_err(|e| e.to_string())?;
                    let candidate = edits.merge(&timeline, &client, base_revision)?;
                    let mut prepared = Vec::new();
                    for owner in document.renderer_owners() {
                        let snapshots = owner.render_edits(&document, &candidate)?;
                        if !owner.snapshots.iter().zip(&snapshots).all(|(publisher, snapshot)| publisher.has_capacity(snapshot)) {
                            return Err("retired snapshot budget exhausted; reopen the instance".into());
                        }
                        prepared.push((owner, snapshots));
                    }
                    for (owner, snapshots) in prepared {
                        for (publisher, snapshot) in owner.snapshots.iter().zip(snapshots) { publisher.publish(snapshot).map_err(|e| format!("snapshot publish failed: {e:?}"))?; }
                    }
                    *edits = candidate;
                    log::info!("[ara] GUI commit ready revision={} model={model_revision}", edits.revision);
                    Ok(Response { ok: true, ..Default::default() })
                }
            }
        })();
        let mut response = outcome.unwrap_or_else(|error| Response { error: Some(error), ..Default::default() });
        response.revision = edits.revision;
        response.model_revision = model_revision;
        response
    }

    /// 使用现有离线内核，不将宿主文件路径当成音频权威。
    pub(crate) fn render_edits(&self, document: &super::document::DocumentSession, edits: &crate::state_channel::EditState)
        -> Result<Vec<super::snapshot::PlaybackSnapshot>, String> {
        use hifishifter_kernel::mixdown::{MixdownOptions, MixdownPcm, QualityPreset, render_mixdown_with_pcm};
        use super::snapshot::PlaybackSnapshot;
        if !document.ready.load(Ordering::Acquire) { return Err("host PCM/model not ready".into()); }
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        if keys.is_empty() { return Ok([44100, 48000].into_iter().map(|sample_rate| PlaybackSnapshot {
            sample_rate, origin_sample: 0, left: vec![], right: vec![], _reservation: None,
        }).collect()); }
        let mut timeline = self.assigned_timeline(document)?;
        edits.apply(&mut timeline);
        let geometry = document.regions.lock().unwrap();
        let regions = keys.iter().map(|key| geometry.get(key).cloned().ok_or("assigned region disappeared"))
            .collect::<Result<Vec<_>, _>>()?;
        drop(geometry);
        let sources = document.edit_sources.lock().unwrap().clone();
        // 先复用先导范围校验：时间拉伸/宿主内容淡化仍不能伪称支持。
        let validation = super::snapshot::mix_plain_regions(&regions, &sources, 44100).map_err(|e| format!("unsupported host region: {e:?}"))?;
        drop(validation);
        let start = timeline.clips.iter().map(|c| c.start_sec).fold(f64::INFINITY, f64::min);
        let end = timeline.clips.iter().map(|c| c.start_sec + c.length_sec).fold(0.0_f64, f64::max);
        if start < 0.0 || !start.is_finite() || !end.is_finite() { return Err("unsupported host position".into()); }
        let mut input = HashMap::new();
        for id in timeline.clips.iter().filter_map(|clip| clip.source_path.as_ref()).collect::<BTreeSet<_>>() {
            let pcm = sources.get(id).ok_or_else(|| format!("host PCM unavailable: {id}"))?;
            let channels = pcm.planes.len();
            let mut samples = Vec::with_capacity(pcm.planes[0].len() * channels);
            for frame in 0..pcm.planes[0].len() { for plane in &pcm.planes { samples.push(plane[frame]); } }
            input.insert(id.clone(), MixdownPcm { sample_rate: pcm.sample_rate, channels: channels as u16, samples: Arc::new(samples) });
        }
        let mut snapshots = Vec::new();
        for sample_rate in [44100, 48000] {
            let frames = ((end - start) * sample_rate as f64).round() as usize;
            if frames > 64 * 1024 * 1024 / 8 { return Err("snapshot span exceeds 64MiB".into()); }
            let reservation = super::budget::global_budget().reserve(frames * 8).ok_or("PCM memory budget exceeded")?;
            let options = MixdownOptions { sample_rate, start_sec: start, end_sec: Some(end),
                stretch: hifishifter_kernel::time_stretch::StretchAlgorithm::SoundTouchDll, apply_pitch_edit: true,
                output: hifishifter_kernel::encode::OutputSpec::wav_32f(), quality_preset: QualityPreset::Export,
                cancel_flag: None, progress: None, cache_stats: None };
            let (_, channels, _, samples) = render_mixdown_with_pcm(&timeline, options, &input)?;
            if channels != 2 || samples.len() != frames * 2 || samples.iter().any(|v| !v.is_finite()) { return Err("invalid kernel output".into()); }
            snapshots.push(PlaybackSnapshot { sample_rate, origin_sample: (start * sample_rate as f64).round() as i64,
                left: samples.iter().step_by(2).copied().collect(), right: samples.iter().skip(1).step_by(2).copied().collect(),
                _reservation: Some(reservation) });
        }
        Ok(snapshots)
    }

    /// 从宿主时间线筛选实际分配的区域，而不是让每个处理器混整张文档。
    fn assigned_timeline(&self, document: &super::document::DocumentSession) -> Result<hifishifter_kernel::state::TimelineState, String> {
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        let identities = document.clip_ids.lock().unwrap();
        let ids = keys.iter().map(|key| identities.get(key).cloned().ok_or("missing assigned clip identity"))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut timeline = document.timeline.lock().unwrap().clone().ok_or("host timeline unavailable")?;
        timeline.clips.retain(|clip| ids.contains(&clip.id));
        if !timeline.clips.is_empty() { timeline.tracks.retain(|track| timeline.clips.iter().any(|clip| clip.track_id == track.id)); }
        Ok(timeline)
    }

    /// 把租约关联到真实文档；关闭文档时由其同步撤销。
    pub fn bind_to_document(
        self: &Arc<Self>,
        document: Arc<super::document::DocumentSession>,
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
        companion: Option<ara2_bridge::companion::CompanionControllerBinding<'static>>,
    ) -> Result<*const ara2_bridge::sys::ARAPlugInExtensionInstance, AraError> {
        let mut current = self.binding.lock().unwrap_or_else(|p| p.into_inner());
        if current.is_some() {
            return Err(AraError::InvalidState("extension already bound"));
        }
        let owner = Arc::downgrade(self);
        let document_id = document.id;
        let observer = Arc::new(
            move |role: ExtensionRoles, keys: &[usize], sequences: &[usize]| {
                if let Some(owner) = owner.upgrade() {
                    let keys = keys.iter().map(|key| *key as RegionKey).collect::<Vec<_>>();
                    let valid = keys.is_empty()
                        || region_owners()
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .resolve(&keys)
                            .is_ok_and(|(owner, _)| owner == document_id);
                    let count = keys.len();
                    owner
                        .assignments
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(role.bits(), if valid { keys } else { Vec::new() });
                    owner.sequences.lock().unwrap().insert(
                        role.bits(),
                        sequences.iter().map(|key| *key as u64).collect(),
                    );
                    if valid {
                        log::info!(
                            "[ara] renderer assignment role={} regions={count}",
                            role.bits()
                        );
                    } else {
                        log::warn!("[ara] rejected unknown or cross-document renderer assignment");
                    }
                    owner.prepare();
                }
            },
        );
        let supported = ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER;
        let enabled = ExtensionRoles::resolve(known, assigned, supported)?;
        let owned = ExtensionBinding::new_with_renderer_observer(
            generation, known, assigned, supported, observer,
        )?;
        let raw = owned.0.as_raw();
        {
            let local = self.edits.lock().unwrap();
            let mut shared = document.edits.lock().unwrap();
            if local.revision > shared.revision { *shared = local.clone(); }
        }
        document.attach(self, owned.1, companion)?;
        *self.document.lock().unwrap() = Some(Arc::downgrade(&document));
        *current = Some(owned.0);
        self.role.store(
            if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                1
            } else {
                2
            },
            Ordering::Release,
        );
        drop(current);
        if enabled.contains(ExtensionRoles::EDITOR_RENDERER) {
            let weak = Arc::downgrade(self);
            match hifishifter_ara_ipc::Server::start("HiFiShifter / REAPER".into(), move |request| {
                weak.upgrade().map(|owner| owner.handle_request(request)).unwrap_or_else(|| hifishifter_ara_ipc::Response {
                    error: Some("plugin instance closed".into()), ..Default::default()
                })
            }) {
                Ok(server) => { log::info!("[ara] GUI channel ready instance={}", server.record().instance_id); *self.channel.lock().unwrap() = Some(server); }
                Err(error) => log::warn!("[ara] GUI channel unavailable: {error}"),
            }
        }
        Ok(raw)
    }

    /// 文档侧已撤销 lease；保留 binding 以供宿主最后几次合法移除/释放回调访问。
    pub fn document_closed(&self) {
        self.stop_channel();
        self.assignments.lock().unwrap().clear();
        self.sequences.lock().unwrap().clear();
        self.document.lock().unwrap().take();
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
    }

    /// 模型线程展开当前 renderer 的显式区域与 editor sequence，并去重，拒绝跨文档身份。
    pub fn assigned_regions(&self) -> Result<Vec<u64>, AraError> {
        let document = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or(AraError::InvalidState("document no longer available"))?;
        let role = self.role.load(Ordering::Acquire);
        let mut keys = self
            .assignments
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let sequences = self
            .sequences
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default();
        let members = document.sequence_regions.lock().unwrap();
        for sequence in sequences {
            keys.extend(
                members
                    .get(&sequence)
                    .ok_or(AraError::InvalidState("unknown assigned sequence"))?
                    .iter()
                    .copied(),
            );
        }
        drop(members);
        let keys = keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty() {
            let (owner, _) = region_owners()
                .lock()
                .unwrap()
                .resolve(&keys)
                .map_err(|_| AraError::InvalidState("invalid assigned region"))?;
            if owner != document.id {
                return Err(AraError::InvalidState(
                    "assigned region belongs to another document",
                ));
            }
        }
        Ok(keys)
    }

    /// 非实时准备两个支持的输出采样率，callback 不重采样、不等待。
    pub fn prepare(&self) {
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        let Some(document) = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
        else {
            return;
        };
        let _transaction = document.transaction.lock().unwrap();
        let edits = document.edits.lock().unwrap().clone();
        if edits.revision > 0 {
            match self.render_edits(&document, &edits) {
                Ok(snapshots) => { for (publisher, snapshot) in self.snapshots.iter().zip(snapshots) { let _ = publisher.publish(snapshot); } }
                Err(error) => log::warn!("[ara] edited snapshot unavailable: {error}"),
            }
            return;
        }
        let Ok(keys) = self.assigned_regions() else {
            return;
        };
        if keys.is_empty() {
            return;
        }
        let geometry = document.regions.lock().unwrap();
        let Some(regions) = keys
            .iter()
            .map(|key| geometry.get(key).cloned())
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        drop(geometry);
        let sources = document.sources.lock().unwrap().clone();
        for (publisher, rate) in self.snapshots.iter().zip([44100, 48000]) {
            match super::snapshot::mix_plain_regions(&regions, &sources, rate)
                .and_then(|snapshot| publisher.publish(snapshot))
            {
                Ok(()) => log::info!(
                    "[ara] snapshot ready role={} rate={rate} regions={}",
                    self.role.load(Ordering::Relaxed),
                    keys.len()
                ),
                Err(error) => log::warn!("[ara] snapshot unavailable: {error:?}"),
            }
        }
    }
}
