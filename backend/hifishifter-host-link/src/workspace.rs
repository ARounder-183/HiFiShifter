//! 工作区目录的读写。
//!
//! # 一次交接就是一组文件
//!
//! 生产者写 `manifest.hfslink`（最后写，它出现即表示去程完成），消费者写
//! `edits.hfsedit`（同样原子改名后可见）。两侧**写不同的文件**，所以不需要锁、
//! 不需要协商、不需要令牌 —— 这是本文档里最重要的一条纪律，
//! 任何"两侧都写同一个文件"的改动都会把并发问题重新引入。
//!
//! # 为什么音频是 WAV 而清单是 JSON
//!
//! 清单是**契约**，人可读有价值；音频是**数据**，体积最大且没有任何理由被读。
//! 把 f32 数组塞进 JSON 会让体积膨胀三到四倍，还要在一次解析里全量缓冲 ——
//! 这正是管道时代 64 MiB 上限的来源。WAV 32-bit float 在上游已实现且踩过坑
//! （`host_pcm::load_pcm_file`：**WAV 头里的采样率优先于声明值**），这里沿用同一条纪律。
//!
//! # 为什么摘要覆盖的是**采样**而不是文件字节
//!
//! 覆盖文件字节只能证明"文件没被换过"；覆盖采样能证明**解码出来的音频**就是我们写下去的那份，
//! 于是编码器/解码器的错误也会被抓住。它也带来一个实际好处：同一份音频在不同编码下
//! 摘要相同，因此**天然去重** —— 重新导出时未变化的素材不会重写。
//! 这与内核既有的 `host_pcm` 纪律一致（那边覆盖的是 `sample_rate + channels + 每个采样的位模式`）。

use crate::{
    EditManifest, LinkManifest, LinkStatus, SourceEntry, AUDIO_DIR, MANIFEST_FILE, OPEN_MARK,
    TMP_DIR,
};
use std::io::Write;
use std::path::{Path, PathBuf};

/// 一次音频写入的结果。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioRef {
    /// 相对工作区根目录的路径。
    pub file: String,
    pub blake3: String,
    pub bytes: u64,
}

/// 读回的一份音频。
#[derive(Clone, Debug)]
pub struct LoadedAudio {
    pub sample_rate: u32,
    pub planes: Vec<Vec<f32>>,
}

/// 一个工作区目录。
///
/// 它**只是路径**：不持有句柄、不缓存内容、不假设对方是否活着。
/// 这正是交接与链路的分野 —— 生产者进程退出后，这个值依然有效。
#[derive(Clone, Debug)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join(MANIFEST_FILE)
    }

    pub fn edit_path(&self) -> PathBuf {
        self.root.join(crate::EDIT_FILE)
    }

    pub fn status_path(&self) -> PathBuf {
        self.root.join(crate::STATUS_FILE)
    }

    pub fn audio_dir(&self) -> PathBuf {
        self.root.join(AUDIO_DIR)
    }

    fn temp_dir(&self) -> PathBuf {
        self.root.join(TMP_DIR)
    }

    /// 这份目录是否**看起来**是一个工作区。
    ///
    /// 【为什么只查清单】消费者要能在用户拖进来一个普通目录时说清楚"这不是工作区"，
    /// 而不是报一个下游的解析错误。`manifest.hfslink` 是去程的完成标记，也必须是它。
    pub fn looks_like_workspace(&self) -> bool {
        self.manifest_path().is_file()
    }

    /// 读清单并做结构自检。
    pub fn read_manifest(&self) -> Result<LinkManifest, String> {
        let path = self.manifest_path();
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let manifest: LinkManifest = serde_json::from_slice(&bytes)
            .map_err(|error| format!("{} is not a link manifest: {error}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// 写清单。**必须最后写** —— 它出现就是去程完成的信号。
    pub fn write_manifest(&self, manifest: &LinkManifest) -> Result<(), String> {
        manifest.validate()?;
        self.prepare()?;
        self.write_json_atomic(&self.manifest_path(), manifest)
    }

    pub fn read_edit(&self) -> Result<EditManifest, String> {
        let path = self.edit_path();
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let edit: EditManifest = serde_json::from_slice(&bytes)
            .map_err(|error| format!("{} is not an edit manifest: {error}", path.display()))?;
        edit.validate()?;
        Ok(edit)
    }

    /// 写编辑载荷（消费者侧）。原子改名后可见 —— 生产者不会读到半个文件。
    pub fn write_edit(&self, edit: &EditManifest) -> Result<(), String> {
        edit.validate()?;
        self.prepare()?;
        self.write_json_atomic(&self.edit_path(), edit)
    }

    /// 编辑载荷的最后修改时间（毫秒）；生产者靠它判断「有没有新的编辑」。
    ///
    /// 【为什么用 mtime 而不是轮询一整个目录】生产者的 actor 循环每秒醒几次，
    /// 对一个已知路径做一次 `metadata` 是最便宜的"有没有新东西"判据。
    pub fn edit_stamp_ms(&self) -> Option<u64> {
        self.stamp_ms(&self.edit_path())
    }

    pub fn manifest_stamp_ms(&self) -> Option<u64> {
        self.stamp_ms(&self.manifest_path())
    }

    fn stamp_ms(&self, path: &Path) -> Option<u64> {
        let modified = std::fs::metadata(path).ok()?.modified().ok()?;
        let since = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
        Some(since.as_millis() as u64)
    }

    pub fn read_status(&self) -> Option<LinkStatus> {
        let bytes = std::fs::read(self.status_path()).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// 写状态（生产者侧）。成功与被拒**同一份可见**。
    pub fn write_status(&self, status: &LinkStatus) -> Result<(), String> {
        self.prepare()?;
        self.write_json_atomic(&self.status_path(), status)
    }

    /// 把一份素材音频写进工作区，返回相对路径与内容摘要。
    ///
    /// 【为什么按内容命名】重新导出时未变化的素材直接命中已有文件、跳过写入。
    /// 于是「宿主里改了一个片段就重导整个工程」从 O(全部音频) 降到 O(改动)。
    ///
    /// 【为什么摘要要算上 `key`】文件名 `source-{digest}.wav` 与内核
    /// `host_pcm::materialize` 的命名**逐字节相同** —— 于是消费者可以直接把这个目录
    /// 当作物化目标，`matches_authorized_pcm` 一看名字与头就命中并跳过写出。
    /// **导入因此不再复制任何音频。**
    ///
    /// 【这条一致性为什么必须钉住】两处命名一旦分叉，每导入一次就会在用户的磁盘上
    /// 多出一份同样大小的文件（而且永不回收）。摘要的**字节顺序**因此是契约的一部分，
    /// 见 [`audio_digest`]。
    pub fn write_audio(
        &self,
        key: &str,
        sample_rate: u32,
        planes: &[Vec<f32>],
    ) -> Result<AudioRef, String> {
        if planes.is_empty() || sample_rate == 0 {
            return Err("audio needs a sample rate and at least one channel".into());
        }
        let frames = planes[0].len();
        if planes.iter().any(|plane| plane.len() != frames) {
            return Err("audio planes are ragged".into());
        }
        let digest = audio_digest(key, sample_rate, planes);
        let name = format!("source-{digest}.wav");
        let relative = format!("{AUDIO_DIR}/{name}");
        let target = self.root.join(&relative);
        self.prepare()?;
        std::fs::create_dir_all(self.audio_dir()).map_err(|error| error.to_string())?;
        // 命中已有文件就跳过：内容寻址的全部意义。
        let bytes = if target.is_file() {
            std::fs::metadata(&target).map_err(|e| e.to_string())?.len()
        } else {
            let encoded = encode_wav(sample_rate, planes)?;
            let length = encoded.len() as u64;
            self.write_bytes_atomic(&target, &encoded)?;
            length
        };
        Ok(AudioRef {
            file: relative,
            blake3: digest,
            bytes,
        })
    }

    /// 读取并**校验**一份素材音频。
    ///
    /// 【为什么必须校验】路径是可变的：宿主写完、消费者读之前，那个文件可能被别的东西覆盖
    /// （临时目录清理、用户操作）。没有摘要校验，「读到别的音频」是静默的，
    /// 而且会被当成用户的编辑存进工程 —— 那是最难排查的一类故障。
    pub fn read_audio(&self, entry: &SourceEntry) -> Result<LoadedAudio, String> {
        let file = entry
            .file
            .as_deref()
            .ok_or_else(|| format!("source {} names no file", entry.key))?;
        let path = self.resolve(file);
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let loaded = decode_wav(&bytes)
            .map_err(|error| format!("{} is not readable float WAV: {error}", path.display()))?;
        if let Some(expected) = entry.blake3.as_deref() {
            let actual = audio_digest(&entry.key, loaded.sample_rate, &loaded.planes);
            if actual != expected {
                return Err(format!(
                    "audio at {} does not match its digest; refusing to import it",
                    path.display()
                ));
            }
        }
        Ok(loaded)
    }

    /// 把清单里的相对路径解析成绝对路径；已经是绝对路径的原样返回（`extern` 用）。
    pub fn resolve(&self, file: &str) -> PathBuf {
        let as_path = Path::new(file);
        if as_path.is_absolute() {
            as_path.to_path_buf()
        } else {
            self.root.join(as_path)
        }
    }

    fn prepare(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
        std::fs::create_dir_all(self.temp_dir()).map_err(|error| error.to_string())
    }

    fn write_json_atomic<T: serde::Serialize>(&self, path: &Path, value: &T) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
        self.write_bytes_atomic(path, &bytes)
    }

    /// 先写临时文件再改名。
    ///
    /// 【为什么不能直接写目标】对方可能正在读它。半个文件对读者来说
    /// 是一个"看起来合法、解出来是半个对象"的输入 —— 而 JSON 解不开至少是**明确的失败**，
    /// 比读到半份数据好。改名在同一目录内是原子的，于是读者要么看到旧的、要么看到新的。
    fn write_bytes_atomic(&self, path: &Path, bytes: &[u8]) -> Result<(), String> {
        let temp = self.temp_dir().join(format!(
            "{}.{}.{}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("link"),
            std::process::id(),
            crate::now_ms()
        ));
        {
            let mut file = std::fs::File::create(&temp).map_err(|error| error.to_string())?;
            file.write_all(bytes).map_err(|error| error.to_string())?;
            // 落到盘再改名：改名是元数据操作，不刷新数据就改名，断电后可能留下一个空的新文件。
            file.sync_all().map_err(|error| error.to_string())?;
        }
        std::fs::rename(&temp, path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            error.to_string()
        })
    }
}

// ─── 工作区发现 ───────────────────────────────────────────────────────────────

/// 扫描一个根目录下的一级子目录，返回其中的工作区。
///
/// 【为什么是"扫描"而不是"心跳"】管道时代的发现目录里每个实例每秒钟重写一次 JSON，
/// 读到的记录还可能是崩溃进程留下的 15 秒幽灵。工作区**不需要心跳**：
/// 它是一份磁盘上的事实，"上次更新时间"由清单的 `created_unix_ms` 回答，
/// 而"宿主还在不在"由 `status.hfslink` 的 `updated_unix_ms` 回答 —— 都是**信息**，
/// 不再是**可用性的前提**。
pub fn workspaces_under(root: &Path) -> Vec<Workspace> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut found: Vec<Workspace> = entries
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| Workspace::at(entry.path()))
        .filter(Workspace::looks_like_workspace)
        .collect();
    found.sort_by_key(|workspace| std::cmp::Reverse(workspace.manifest_stamp_ms().unwrap_or(0)));
    found
}

/// 默认的工作区根目录（用户可见、可备份、可拷走）。
///
/// 【为什么不是 `%TEMP%`】管道时代的素材被解到临时目录、随会话一起删。
/// 工作区是用户的资产：它要在宿主关掉之后还在，要被 `另存为` 引用，要被拷到另一台机器。
pub fn default_root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("HIFISHIFTER_LINK_DIR") {
        return Some(PathBuf::from(root));
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
            })
    }?;
    Some(base.join("HiFiShifter/links"))
}

// ─── 打开标记（同一工作区不开两个编辑窗口）─────────────────────────────────────

/// 打开标记的内容，仅用于告诉用户「谁」已经开着它。
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpenMark {
    pub pid: u32,
    pub since_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// 打开工作区时持有的标记；`Drop` 时删除**自己写下的那一份**。
///
/// 【为什么释放前要比对内容】标记可能已被接管（原持有者被杀、标记过期后别的窗口
/// 认领了它）。无条件删除会把**接管者**的标记一起删掉 —— 于是两个窗口都以为自己握着它。
/// 只有当前内容与自己写下的完全一致时，这份标记才是自己的。
#[derive(Debug)]
pub struct WorkspaceLock {
    path: PathBuf,
    written: Vec<u8>,
}

impl WorkspaceLock {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WorkspaceLock {
    fn drop(&mut self) {
        if std::fs::read(&self.path).is_ok_and(|current| current == self.written) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl Workspace {
    /// 已经开着的标记（若存在且未过期）。
    pub fn open_mark(&self, stale_after_ms: u64) -> Option<OpenMark> {
        let mark: OpenMark =
            serde_json::from_slice(&std::fs::read(self.open_mark_path()).ok()?).ok()?;
        let age = crate::now_ms().saturating_sub(mark.since_unix_ms);
        if age > stale_after_ms {
            None
        } else {
            Some(mark)
        }
    }

    pub fn open_mark_path(&self) -> PathBuf {
        self.root.join(OPEN_MARK)
    }

    /// 取用打开标记。
    ///
    /// 【为什么用"独占创建"而不是锁文件协议】两边都是本机进程，`create_new` 就是
    /// 一个原子的"谁先到"。标记过期（进程被杀、机器断电）由**年龄**判断，
    /// 不查 pid 是否存活 —— 那需要平台相关的代码，而且"另一个用户/沙箱里的同号 pid"
    /// 会让它给出错误的答案。12 小时的默认寿命足够避开误判，也足够让用户接管。
    pub fn acquire_open_mark(
        &self,
        label: Option<&str>,
        stale_after_ms: u64,
    ) -> Result<WorkspaceLock, String> {
        self.prepare()?;
        let path = self.open_mark_path();
        for attempt in 0..2 {
            let mark = OpenMark {
                pid: std::process::id(),
                since_unix_ms: crate::now_ms(),
                label: label.map(str::to_string),
            };
            let bytes = serde_json::to_vec(&mark).map_err(|error| error.to_string())?;
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    file.write_all(&bytes).map_err(|error| error.to_string())?;
                    return Ok(WorkspaceLock {
                        path,
                        written: bytes,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if attempt > 0 {
                        return Err(format!("open mark at {} is held", path.display()));
                    }
                    // 过期就接管；没过期就如实拒绝，并把持有者告诉调用方。
                    if self.open_mark(stale_after_ms).is_some() {
                        let holder = self
                            .open_mark(stale_after_ms)
                            .map(|mark| format!("pid {}", mark.pid))
                            .unwrap_or_else(|| "another window".into());
                        return Err(format!("already open by {holder}"));
                    }
                    let _ = std::fs::remove_file(&path);
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        Err(format!("cannot take the open mark at {}", path.display()))
    }
}

// ─── WAV 编解码 ───────────────────────────────────────────────────────────────

/// 音频内容摘要：覆盖素材键、采样率、声道数与**每个采样的位模式**。
///
/// 【为什么覆盖的是采样而不是文件字节】覆盖文件字节只能证明「文件没被换过」；
/// 覆盖采样能证明**解码出来的音频**就是我们写下去的那份，于是编码器/解码器的错误
/// 也会被抓住。这与内核既有的 `host_pcm` 纪律一致（那边覆盖的也是
/// `persistent_id + 采样率 + 声道数 + 每个采样的位模式`）。
///
/// 【为什么摘要的字节顺序是契约】它的输出就是文件名。消费者的物化目标就是这个
/// 目录，靠文件名命中来跳过写出 —— 所以这里的喂给顺序必须与
/// `host_pcm::materialize_with_byte_limit` 里那段指纹计算**逐字节一致**：
/// `persistent_id`、`sample_rate`（u32 LE）、`planes.len()`（**u64** LE）、
/// 再按平面顺序每个采样（f32 LE）。改动任何一项都会让每次导入多复制一份音频。
fn audio_digest(key: &str, sample_rate: u32, planes: &[Vec<f32>]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(key.as_bytes());
    hasher.update(&sample_rate.to_le_bytes());
    hasher.update(&(planes.len() as u64).to_le_bytes());
    for plane in planes {
        for sample in plane {
            hasher.update(&sample.to_le_bytes());
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn encode_wav(sample_rate: u32, planes: &[Vec<f32>]) -> Result<Vec<u8>, String> {
    let channels = planes.len() as u16;
    if planes.len() > u16::MAX as usize {
        return Err("too many channels".into());
    }
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buffer = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buffer, spec).map_err(|e| e.to_string())?;
        let frames = planes[0].len();
        for frame in 0..frames {
            for plane in planes {
                writer
                    .write_sample(plane[frame])
                    .map_err(|e| e.to_string())?;
            }
        }
        writer.finalize().map_err(|e| e.to_string())?;
    }
    Ok(buffer.into_inner())
}

fn decode_wav(bytes: &[u8]) -> Result<LoadedAudio, String> {
    let mut reader =
        hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
        return Err(format!(
            "expected 32-bit float, found {}-bit {:?}",
            spec.bits_per_sample, spec.sample_format
        ));
    }
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err("WAV declares no channels".into());
    }
    let samples: Vec<f32> = reader
        .samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if !samples.len().is_multiple_of(channels) {
        return Err("WAV sample count is not a whole number of frames".into());
    }
    let frames = samples.len() / channels;
    let mut planes = vec![Vec::with_capacity(frames); channels];
    for frame in 0..frames {
        for (channel, plane) in planes.iter_mut().enumerate() {
            plane.push(samples[frame * channels + channel]);
        }
    }
    Ok(LoadedAudio {
        // 【为什么信头部而不是信声明值】与 `host_pcm::load_pcm_file` 同一条纪律：
        // 文件头是写这份音频的人留下的**事实**，而清单里的 `sample_rate` 是一个可能
        // 忘记同步的**声明**。两者冲突时，能对得上采样数的是头部。
        sample_rate: spec.sample_rate,
        planes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioState;
    use crate::{Admission, Revisions, SourceKind, FORMAT, FORMAT_VERSION};

    fn temp_workspace(tag: &str) -> Workspace {
        let root = std::env::temp_dir().join(format!(
            "hfs-link-{tag}-{}-{}",
            std::process::id(),
            crate::now_ms()
        ));
        Workspace::at(root)
    }

    fn manifest_for(sources: Vec<SourceEntry>) -> LinkManifest {
        LinkManifest {
            format: FORMAT.into(),
            format_version: FORMAT_VERSION,
            link_id: crate::link_id(SourceKind::Ara, "REAPER", "doc"),
            source_kind: SourceKind::Ara,
            host_label: "REAPER".into(),
            created_unix_ms: crate::now_ms(),
            producer: crate::Producer {
                implementation: "test".into(),
                pid: None,
            },
            timeline: serde_json::json!({"clips": [], "tracks": []}),
            host_geometry: serde_json::json!({"clips": [], "tracks": []}),
            sources,
            granted: vec![crate::Capability::Commit],
            revisions: Revisions { edits: 0, model: 1 },
            admission: Admission::Strict,
            losses: Vec::new(),
            diagnostics: None,
        }
    }

    /// 一份音频写进去、读回来必须逐位相同。
    #[test]
    fn audio_round_trips_bit_exactly() {
        let workspace = temp_workspace("audio");
        let planes = vec![
            vec![0.0, 0.5, -0.25, 1.0, -1.0, 1.0 / 3.0],
            vec![1.0, -0.5, 0.25, -1.0, 1.0, -1.0 / 7.0],
        ];
        let reference = workspace.write_audio("k", 48_000, &planes).unwrap();
        assert!(reference.file.starts_with("audio/source-"));
        assert!(workspace.resolve(&reference.file).is_file());

        let entry = SourceEntry {
            key: "k".into(),
            sample_rate: 48_000,
            channels: 2,
            frames: 6,
            state: AudioState::Complete,
            file: Some(reference.file.clone()),
            blake3: Some(reference.blake3.clone()),
            bytes: Some(reference.bytes),
            note: None,
        };
        let loaded = workspace.read_audio(&entry).unwrap();
        assert_eq!(loaded.sample_rate, 48_000);
        assert_eq!(loaded.planes, planes);
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// 内容寻址：同一份音频写两次是**同一个文件**，且第二次不重写。
    #[test]
    fn identical_audio_is_addressed_by_content_and_not_rewritten() {
        let workspace = temp_workspace("dedup");
        let planes = vec![vec![0.1_f32; 128]];
        let first = workspace.write_audio("k", 44_100, &planes).unwrap();
        let path = workspace.resolve(&first.file);
        let stamp = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let second = workspace.write_audio("k", 44_100, &planes).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            stamp,
            "第二次写入必须直接命中已有文件"
        );
        // 采样率不同 ⇒ 内容不同 ⇒ 另一个文件。
        let other = workspace.write_audio("k", 96_000, &planes).unwrap();
        assert_ne!(other.blake3, first.blake3);
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// **篡改过的音频必须被拒**：这是「读到别的音频被当成用户的编辑」的唯一防线。
    #[test]
    fn tampered_audio_is_rejected_by_its_digest() {
        let workspace = temp_workspace("tamper");
        let reference = workspace
            .write_audio("k", 48_000, &[vec![0.25; 64]])
            .unwrap();
        let entry = SourceEntry {
            key: "k".into(),
            sample_rate: 48_000,
            channels: 1,
            frames: 64,
            state: AudioState::Complete,
            file: Some(reference.file.clone()),
            blake3: Some(reference.blake3.clone()),
            bytes: Some(reference.bytes),
            note: None,
        };
        workspace.read_audio(&entry).unwrap();

        let path = workspace.resolve(&reference.file);
        let replacement = encode_wav(48_000, &[vec![0.75; 64]]).unwrap();
        std::fs::write(&path, replacement).unwrap();
        let error = workspace.read_audio(&entry).unwrap_err();
        assert!(error.contains("does not match its digest"), "{error}");
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// 缺 `blake3` 的条目仍可读（`extern` 场景由宿主决定校验策略），
    /// 但没有文件必须明确失败。
    #[test]
    fn an_entry_without_a_file_fails_loudly() {
        let workspace = temp_workspace("nofile");
        let entry = SourceEntry {
            key: "k".into(),
            sample_rate: 48_000,
            channels: 1,
            frames: 4,
            state: AudioState::Partial,
            file: None,
            blake3: None,
            bytes: None,
            note: None,
        };
        assert!(workspace
            .read_audio(&entry)
            .unwrap_err()
            .contains("names no file"));
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// `extern` 素材走绝对路径，能读工作区之外的文件。
    #[test]
    fn extern_audio_is_read_from_its_absolute_path() {
        let workspace = temp_workspace("extern");
        let outside = std::env::temp_dir().join(format!("hfs-outside-{}.wav", crate::now_ms()));
        let planes = vec![vec![0.2_f32; 32], vec![-0.2_f32; 32]];
        std::fs::write(&outside, encode_wav(44_100, &planes).unwrap()).unwrap();
        let entry = SourceEntry {
            key: "k".into(),
            sample_rate: 44_100,
            channels: 2,
            frames: 32,
            state: AudioState::Extern,
            file: Some(outside.to_string_lossy().into_owned()),
            blake3: Some(audio_digest("k", 44_100, &planes)),
            bytes: None,
            note: None,
        };
        assert_eq!(workspace.read_audio(&entry).unwrap().planes, planes);
        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// 清单与编辑载荷各自往返，且写完之后**原子可见**（读者看不到半个文件）。
    #[test]
    fn manifests_round_trip_through_the_workspace() {
        let workspace = temp_workspace("manifest");
        workspace.write_manifest(&manifest_for(Vec::new())).unwrap();
        assert!(workspace.looks_like_workspace());
        assert_eq!(workspace.read_manifest().unwrap().host_label, "REAPER");

        let edit = EditManifest {
            format: crate::EDIT_FORMAT.into(),
            format_version: FORMAT_VERSION,
            link_id: crate::link_id(SourceKind::Ara, "REAPER", "doc"),
            base_revisions: Revisions { edits: 0, model: 1 },
            host_geometry: serde_json::json!({"clips": []}),
            params: serde_json::json!({}),
            tracks: serde_json::json!([]),
        };
        workspace.write_edit(&edit).unwrap();
        assert!(workspace.edit_stamp_ms().is_some());
        assert_eq!(workspace.read_edit().unwrap().link_id, edit.link_id);

        // 临时目录不该留下任何东西。
        let leftovers: Vec<_> = std::fs::read_dir(workspace.root().join(TMP_DIR))
            .unwrap()
            .flatten()
            .collect();
        assert!(leftovers.is_empty(), "临时文件必须被改名走");

        let status = LinkStatus::initial(&edit.link_id, Revisions { edits: 1, model: 1 });
        workspace.write_status(&status).unwrap();
        assert!(workspace.read_status().unwrap().is_current(&edit.link_id));
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// 覆盖写必须留下**完整的新版本**，而不是新旧混合。
    #[test]
    fn rewriting_replaces_the_whole_document() {
        let workspace = temp_workspace("rewrite");
        let mut manifest = manifest_for(Vec::new());
        workspace.write_manifest(&manifest).unwrap();
        manifest.host_label = "Cubase".into();
        manifest.link_id = crate::link_id(SourceKind::Ara, "Cubase", "doc");
        workspace.write_manifest(&manifest).unwrap();
        let back = workspace.read_manifest().unwrap();
        assert_eq!(back.host_label, "Cubase");
        assert_eq!(back.link_id, manifest.link_id);
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// 发现只认**有清单**的一级子目录，且按新旧排序。
    #[test]
    fn discovery_lists_only_real_workspaces() {
        let root = std::env::temp_dir().join(format!("hfs-links-{}", crate::now_ms()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(root.join("not-a-workspace")).unwrap();
        let older = Workspace::at(root.join("older"));
        older.write_manifest(&manifest_for(Vec::new())).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let newer = Workspace::at(root.join("newer"));
        newer.write_manifest(&manifest_for(Vec::new())).unwrap();

        let found = workspaces_under(&root);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].root(), newer.root(), "最新的排在最前");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 打开标记是独占的；过期标记可被接管；`Drop` 后释放。
    #[test]
    fn the_open_mark_is_exclusive_and_expires() {
        let workspace = temp_workspace("lock");
        workspace.write_manifest(&manifest_for(Vec::new())).unwrap();
        let held = workspace.acquire_open_mark(Some("test"), 60_000).unwrap();
        assert!(workspace.open_mark(60_000).is_some());
        let error = workspace
            .acquire_open_mark(Some("test"), 60_000)
            .unwrap_err();
        assert!(error.contains("already open"), "{error}");
        // 寿命给 0 ⇒ 已有标记立刻算过期，可以接管。
        let taken = workspace.acquire_open_mark(Some("test"), 0).unwrap();
        drop(held);
        assert!(
            workspace.open_mark(60_000).is_some(),
            "接管之后标记仍在（由接管者持有）"
        );
        drop(taken);
        assert!(workspace.open_mark(60_000).is_none());
        let _ = std::fs::remove_dir_all(workspace.root());
    }

    /// 非工作区目录不会被误认。
    #[test]
    fn a_plain_directory_is_not_a_workspace() {
        let workspace = temp_workspace("plain");
        std::fs::create_dir_all(workspace.root()).unwrap();
        assert!(!workspace.looks_like_workspace());
        assert!(workspace
            .read_manifest()
            .unwrap_err()
            .contains("cannot read"));
        let _ = std::fs::remove_dir_all(workspace.root());
    }
}
