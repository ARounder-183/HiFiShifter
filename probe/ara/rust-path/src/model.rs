//! HiFiShifter ARA 探针 / Task 2 Step 3 —— ARA 文档模型探针。
//!
//! 这个模块只做一件事：把 ARA 宿主（REAPER）推给插件的模型事件计数并落盘，
//! 用于验证"Rust 侧真的读到了 ARA 对象"。它不参与渲染，也不属于产品代码，
//! 是 `probe/ara/` 下的一次性产物。

use ara2_bridge::core::{
    ApiGeneration, AraError, AudioModificationProperties, AudioSourceProperties,
    ContentTimeRange, ContentUpdateScopes, DocumentProperties, MusicalContextProperties,
    PlaybackRegionProperties, RegionSequenceProperties,
};
use ara2_bridge::plugin::{
    AudioModifications, AudioSources, CreateContext, DocumentLifecycle, HostContentScope,
    MusicalContexts, PlaybackRegions, RegionSequences,
};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// 逐行追加写日志。每条写一行，行首带自增序号，方便在 REAPER 崩溃后确认写到哪一步。
pub struct ProbeLog {
    path: PathBuf,
    seq: AtomicUsize,
}

impl ProbeLog {
    /// 用目标文件路径建立日志器；文件不存在时由第一次写入创建。
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            seq: AtomicUsize::new(0),
        }
    }

    /// 追加一行。失败时静默忽略：探针不因为日志写不进去而影响宿主。
    pub fn write(&self, message: &str) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(file, "[{seq:04}] {message}");
        }
    }
}

/// 探针共享状态：日志器 + 各 ARA 对象计数。
///
/// 计数用原子量放在共享状态里，因为宿主可能为同一工厂建立多个文档控制器；
/// 探针只关心"看到了多少"，所以多个实例累加即可。
pub struct ProbeState {
    /// 日志器。
    pub log: ProbeLog,
    /// 已见过的 audioSource 数量。
    pub sources: AtomicUsize,
    /// 已见过的 audioModification 数量。
    pub modifications: AtomicUsize,
    /// 已见过的 regionSequence 数量。
    pub sequences: AtomicUsize,
    /// 已见过的 playbackRegion 数量。
    pub regions: AtomicUsize,
}

impl ProbeState {
    /// 建立空计数的共享状态。
    pub fn new(path: PathBuf) -> Self {
        Self {
            log: ProbeLog::new(path),
            sources: AtomicUsize::new(0),
            modifications: AtomicUsize::new(0),
            sequences: AtomicUsize::new(0),
            regions: AtomicUsize::new(0),
        }
    }

    /// 当前计数的一行摘要。
    pub fn summary(&self) -> String {
        format!(
            "sources={} modifications={} regionSequences={} playbackRegions={}",
            self.sources.load(Ordering::SeqCst),
            self.modifications.load(Ordering::SeqCst),
            self.sequences.load(Ordering::SeqCst),
            self.regions.load(Ordering::SeqCst),
        )
    }

    /// 以固定标签写一行计数摘要。
    fn log_summary(&self, tag: &str) {
        self.log.write(&format!("{tag}: {}", self.summary()));
    }
}

/// ARA 文档模型探针本体。
#[derive(Clone)]
pub struct ProbeModel {
    /// 与宿主模型回调共享的计数与日志状态。
    pub state: Arc<ProbeState>,
}

/// 每个文档控制器的应用态（探针只需记住协商到的 ARA 版本）。
pub struct ProbeDocument {
    /// 该文档控制器协商到的 ARA API 版本。
    pub generation: ApiGeneration,
}

impl DocumentLifecycle for ProbeModel {
    type Document = ProbeDocument;

    fn create_document(
        &mut self,
        context: &CreateContext,
        _properties: DocumentProperties,
    ) -> Result<Self::Document, AraError> {
        let generation = context.generation();
        self.state
            .log
            .write(&format!("document_controller created: apiGeneration={generation:?}"));
        Ok(ProbeDocument { generation })
    }

    fn begin_editing(&mut self, _document: &mut Self::Document) -> Result<(), AraError> {
        self.state.log.write("begin_editing");
        Ok(())
    }

    fn end_editing(
        &mut self,
        _document: &mut Self::Document,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.state.log_summary("end_editing");
        Ok(())
    }

    fn destroy_document(&mut self, _document: Self::Document) {
        self.state.log_summary("document destroyed, final counts");
    }
}

impl MusicalContexts for ProbeModel {
    type MusicalContext = ();

    fn create_musical_context(
        &mut self,
        _context: &CreateContext,
        _properties: MusicalContextProperties,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<Self::MusicalContext, AraError> {
        Ok(())
    }
}

impl RegionSequences for ProbeModel {
    type RegionSequence = String;

    fn create_region_sequence(
        &mut self,
        _context: &CreateContext,
        properties: RegionSequenceProperties,
    ) -> Result<Self::RegionSequence, AraError> {
        let index = self.state.sequences.fetch_add(1, Ordering::SeqCst) + 1;
        let name = properties.name().unwrap_or("<unnamed>").to_owned();
        self.state
            .log
            .write(&format!("region_sequence #{index}: name={name}"));
        Ok(name)
    }
}

impl AudioSources for ProbeModel {
    type AudioSource = String;

    fn create_audio_source(
        &mut self,
        _context: &CreateContext,
        properties: AudioSourceProperties,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<Self::AudioSource, AraError> {
        let index = self.state.sources.fetch_add(1, Ordering::SeqCst) + 1;
        let persistent_id = properties.persistent_id();
        self.state.log.write(&format!(
            "audio_source #{index}: persistentID={persistent_id} sampleRate={} sampleCount={} channels={} merits64Bit={}",
            properties.sample_rate(),
            properties.sample_count(),
            properties.channel_count(),
            properties.merits_64_bit_samples(),
        ));
        Ok(persistent_id.to_owned())
    }

    fn update_audio_source_content(
        &mut self,
        _state: &mut Self::AudioSource,
        _range: Option<ContentTimeRange>,
        _flags: ContentUpdateScopes,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        self.state.log.write("audio_source content updated");
        Ok(())
    }

    fn enable_audio_source_samples_access(
        &mut self,
        state: &mut Self::AudioSource,
        enable: bool,
        _host: &HostContentScope<'_, '_>,
    ) -> Result<(), AraError> {
        // 这一行直接回答探针的未决问题：REAPER 是否给插件样本访问权。
        self.state
            .log
            .write(&format!("audio_source samples_access source={state} enable={enable}"));
        Ok(())
    }
}

impl AudioModifications for ProbeModel {
    type AudioModification = String;

    fn create_audio_modification(
        &mut self,
        _context: &CreateContext,
        properties: AudioModificationProperties,
    ) -> Result<Self::AudioModification, AraError> {
        let index = self.state.modifications.fetch_add(1, Ordering::SeqCst) + 1;
        let persistent_id = properties.persistent_id();
        self.state.log.write(&format!(
            "audio_modification #{index}: persistentID={persistent_id} name={:?}",
            properties.name(),
        ));
        Ok(persistent_id.to_owned())
    }

    fn clone_audio_modification(
        &mut self,
        _context: &CreateContext,
        source: &Self::AudioModification,
        properties: AudioModificationProperties,
    ) -> Result<Self::AudioModification, AraError> {
        let index = self.state.modifications.fetch_add(1, Ordering::SeqCst) + 1;
        let persistent_id = properties.persistent_id();
        self.state.log.write(&format!(
            "audio_modification #{index} (clone of {source}): persistentID={persistent_id}",
        ));
        Ok(persistent_id.to_owned())
    }
}

impl PlaybackRegions for ProbeModel {
    type PlaybackRegion = usize;

    fn create_playback_region(
        &mut self,
        _context: &CreateContext,
        _properties: PlaybackRegionProperties,
    ) -> Result<Self::PlaybackRegion, AraError> {
        let index = self.state.regions.fetch_add(1, Ordering::SeqCst) + 1;
        self.state
            .log
            .write(&format!("playback_region #{index} created"));
        Ok(index)
    }
}
