//! 宿主PCM的私有分析副本：只写调用方提供的私有目录，绝不把宿主ID当文件读取。
use crate::state::TimelineState;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
pub struct PcmView<'a> {
    pub persistent_id: &'a str,
    pub sample_rate: u32,
    pub planes: &'a [Vec<f32>],
}

/// 遇到倒放 clip 时怎么办。
///
/// 【为什么需要两种】两条调用路径对"倒放"的含义完全不同：
/// - **独立 App** 的工程文件里倒放是合法内容（导入/导出都支持），遇到它说明数据或调用方
///   有问题 —— 硬失败是对的，能立刻暴露。
/// - **ARA 插件**里的倒放是**宿主**的既有状态（REAPER 的 `SOURCE SECTION MODE` /
///   负 `PLAYRATE`），用户没做错任何事。ARA 又不提供反向 PCM，插件渲染不出正确的反向
///   内容 —— 但"一个片段倒放"不该升级成"整个插件打不开"。所以插件要**隔离**：
///   该 clip 不物化（无源、界面显示为宿主处理），其余照常。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReversePolicy {
    /// 遇到倒放即整份失败（独立 App 的既有行为）。
    Reject,
    /// 跳过倒放 clip：不物化、不报错，其余照常。
    Isolate,
    /// 照常物化倒放 clip —— 方向交给下游消费方。
    ///
    /// 【为什么插件用这一支而不是 `Isolate`】隔离是"宁可显示占位，也不拿正向 PCM
    /// 当反向内容"的安全选择，代价是倒放片段**永远渲染不出来**。当播种层已按宿主
    /// 事实把 ARA 的**镜像窗口翻回正向**（`project_host_take_facts_locked`）后，
    /// 内核的倒放消费数学（`clip_playback_window_sec` 反向取窗 + `reverse_*`）
    /// 自会产出正确的反向内容 —— 此时物化的**正向** WAV 是正确输入，隔离反而
    /// 会让它缺失。分析也依赖它（分析读正向完整源，装配期再镜像）。
    Materialize,
}

/// 在所有源/几何校验成功后生成只用于波形与分析的WAV，返回路径→宿主ID反向表。
/// 音频渲染仍从ARA授权PCM注入；不回读这些文件作为宿主供音。
pub fn materialize(
    timeline: TimelineState,
    sources: &[PcmView<'_>],
    dir: &Path,
) -> Result<(TimelineState, HashMap<String, String>), String> {
    materialize_with_byte_limit(
        timeline,
        sources,
        dir,
        64 * 1024 * 1024,
        ReversePolicy::Reject,
    )
}

/// 同进程授权源已由调用者计入硬预算；文件流式写出不复制PCM，不按时长拒绝正常长源。
/// 外部IPC入口仍走materialize的64MiB限制，不能借此扩大旧传输协议。
pub fn materialize_with_byte_limit(
    mut timeline: TimelineState,
    sources: &[PcmView<'_>],
    dir: &Path,
    byte_limit: usize,
    reverse_policy: ReversePolicy,
) -> Result<(TimelineState, HashMap<String, String>), String> {
    if byte_limit == 0 || byte_limit > 512 * 1024 * 1024 {
        return Err("invalid host PCM byte limit".into());
    }
    if !timeline.bpm.is_finite()
        || timeline.bpm <= 0.
        || !timeline.project_sec.is_finite()
        || timeline.project_sec < 0.
    {
        return Err("invalid host timeline".into());
    }
    let mut known = HashMap::new();
    let mut bytes = 0_usize;
    for pcm in sources {
        let frames = pcm.planes.first().map(Vec::len).unwrap_or(0);
        bytes = bytes
            .checked_add(
                frames
                    .checked_mul(pcm.planes.len())
                    .and_then(|n| n.checked_mul(4))
                    .ok_or("host PCM size overflow")?,
            )
            .ok_or("host PCM size overflow")?;
        if bytes > byte_limit
            || pcm.persistent_id.is_empty()
            // 【为什么不再限定 44.1/48k】与 `render/source.rs` 的读取门保持**同一条判据**。
            // 那一侧已经放开到 8k..=384k（宿主工程与素材可能是 88.2/96/176.4/192k），
            // 理由写在它的注释里：读取是离线的，下游两处都会重采样。这里若仍只认
            // 44.1/48k，就会出现"读进来了、`publish` 了，然后 `materialize` 整会话失败"
            // —— 一个 96 kHz 素材就能让插件打不开。两处判据分叉是这次的真实缺陷。
            || !(8000..=384_000).contains(&pcm.sample_rate)
            || !(1..=2).contains(&pcm.planes.len())
            || frames == 0
            || pcm
                .planes
                .iter()
                .any(|p| p.len() != frames || p.iter().any(|n| !n.is_finite()))
            || known.insert(pcm.persistent_id, pcm).is_some()
        {
            return Err("invalid or oversized host PCM".into());
        }
    }
    // 被隔离（跳过物化）的倒放 clip。第二趟改写路径时也要跳过它们 ——
    // 它们已经没有源引用，`known[...]` 会失败。
    let mut isolated: BTreeSet<String> = BTreeSet::new();
    for clip in &mut timeline.clips {
        clip.normalize_takes();
        if clip.reversed || clip.takes.iter().any(|t| t.reversed) {
            match reverse_policy {
                ReversePolicy::Reject => {
                    return Err("ARA reverse playback is unsupported".into());
                }
                ReversePolicy::Isolate => {
                    // 隔离：**不物化**。清掉源引用而不是留着它 —— 留着会让下游把它当成
                    // "有源"，于是拿正向 PCM 当反向内容渲染（那是错的，且不可见）。
                    // 清掉之后界面按"无源占位"渲染，与"宿主在处理这一条"是同一个事实。
                    clip.source_path = None;
                    clip.source_path_relative = None;
                    for take in &mut clip.takes {
                        take.source_path = None;
                        take.source_path_relative = None;
                    }
                    isolated.insert(clip.id.clone());
                    continue;
                }
                // 照常物化：走下面的常规几何校验与路径改写。
                ReversePolicy::Materialize => {}
            }
        }
        if !clip.start_sec.is_finite()
            || !clip.length_sec.is_finite()
            || clip.length_sec <= 0.
            || clip.takes.iter().any(|t| {
                !t.source_start_sec.is_finite()
                    || !t.source_end_sec.is_finite()
                    || !t.playback_rate.is_finite()
                    || t.playback_rate <= 0.
            })
        {
            return Err("invalid or unsupported host clip geometry".into());
        }
        for id in
            std::iter::once(&clip.source_path).chain(clip.takes.iter().map(|t| &t.source_path))
        {
            if !known.contains_key(id.as_deref().ok_or("missing host PCM reference")?) {
                return Err("missing host PCM".into());
            }
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut paths = HashMap::new();
    let mut reverse = HashMap::new();
    for pcm in sources {
        let mut fingerprint = blake3::Hasher::new();
        fingerprint.update(pcm.persistent_id.as_bytes());
        fingerprint.update(&pcm.sample_rate.to_le_bytes());
        fingerprint.update(&(pcm.planes.len() as u64).to_le_bytes());
        for plane in pcm.planes {
            for sample in plane {
                fingerprint.update(&sample.to_le_bytes());
            }
        }
        let path = dir.join(format!("source-{}.wav", fingerprint.finalize().to_hex()));
        if !matches_authorized_pcm(&path, pcm) {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let temporary = path.with_extension(format!(
                "{}-{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let written = (|| {
                let mut wav = hound::WavWriter::create(
                    &temporary,
                    hound::WavSpec {
                        channels: pcm.planes.len() as u16,
                        sample_rate: pcm.sample_rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )
                .map_err(|e| e.to_string())?;
                for frame in 0..pcm.planes[0].len() {
                    for plane in pcm.planes {
                        wav.write_sample(plane[frame]).map_err(|e| e.to_string())?;
                    }
                }
                wav.finalize().map_err(|e| e.to_string())?;
                std::fs::rename(&temporary, &path).map_err(|e| e.to_string())
            })();
            if written.is_err() {
                let _ = std::fs::remove_file(&temporary);
            }
            written?;
        }
        let path = path.to_string_lossy().into_owned();
        paths.insert(pcm.persistent_id, path.clone());
        reverse.insert(path, pcm.persistent_id.to_owned());
    }
    for clip in &mut timeline.clips {
        // 被隔离的倒放 clip 没有源引用，跳过 —— 它们保持无源（界面按占位渲染）。
        if isolated.contains(&clip.id) {
            continue;
        }
        clip.source_path = clip
            .source_path
            .as_deref()
            .and_then(|id| paths.get(id).cloned());
        clip.source_path_relative = None;
        for take in &mut clip.takes {
            let pcm = known[take.source_path.as_deref().ok_or("missing host PCM")?];
            take.source_path = Some(paths[pcm.persistent_id].clone());
            take.source_path_relative = None;
            take.source_sample_rate = Some(pcm.sample_rate);
            take.source_channels = Some(pcm.planes.len() as u16);
            take.duration_frames = Some(pcm.planes[0].len() as u64);
            take.duration_sec = Some(pcm.planes[0].len() as f64 / pcm.sample_rate as f64);
            take.source_file_fingerprint = None;
            take.source_file_mtime = None;
            take.source_file_size = None;
            take.waveform_preview = None;
            take.pitch_range = None;
        }
        clip.normalize_takes();
    }
    Ok((timeline, reverse))
}

/// 校验并解码宿主提供的 PCM 文件，返回 `(采样率, 平面)`。
///
/// 【为什么必须校验摘要】路径是**可变的**：宿主写完、接收端读之前，那个文件可能被
/// 别的东西覆盖（临时目录清理、用户操作）。不校验就会静默地把**别的音频**当成用户的
/// 编辑存进工程 —— 那是最难排查的一类故障，所以摘要不符时**拒收**，绝不"凑合读"。
///
/// 【为什么只认 32-bit float WAV】契约把它写死为唯一形态：一种格式 = 一种判据，
/// 不必为"某个宿主的容器"维护解码分支。宿主侧转成正向 float WAV 是一行的事，
/// 而倒放等需要宿主侧先渲染成正向（见契约 §6.9）。
pub fn load_pcm_file(path: &Path, expected_blake3: &str) -> Result<(u32, Vec<Vec<f32>>), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("open host PCM file {}: {e}", path.display()))?;
    // 流式摘要：文件可能很大，不在内存里再放一份完整副本。
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|e| format!("read host PCM file {}: {e}", path.display()))?;
    let digest = hasher.finalize().to_hex().to_string();
    if !digest.eq_ignore_ascii_case(expected_blake3.trim()) {
        return Err(format!(
            "host PCM file {} digest mismatch (expected {}, got {digest})",
            path.display(),
            expected_blake3.trim()
        ));
    }
    let mut wav = hound::WavReader::open(path)
        .map_err(|e| format!("decode host PCM file {}: {e}", path.display()))?;
    let spec = wav.spec();
    if spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
        return Err(format!(
            "host PCM file {} must be a 32-bit float WAV",
            path.display()
        ));
    }
    let channels = spec.channels as usize;
    if !(1..=2).contains(&channels) {
        return Err(format!(
            "host PCM file {} has {channels} channels (1 or 2 expected)",
            path.display()
        ));
    }
    let mut planes = vec![Vec::new(); channels];
    for (index, sample) in wav.samples::<f32>().enumerate() {
        let sample = sample.map_err(|e| format!("decode host PCM file {}: {e}", path.display()))?;
        planes[index % channels].push(sample);
    }
    Ok((spec.sample_rate, planes))
}

/// 授权 PCM 的私有分析文件是否可复用。
///
/// 【为什么不再逐样本回读比对】缓存文件名 `source-{blake3}.wav` 已经把**整份授权内容**
/// （persistent_id + 采样率 + 声道数 + 每个样本的 bit 模式）编进名字：内容一变文件名就变，
/// 旧文件根本不会被选中。于是再逐样本回读整份 WAV 只是在重复推导同一个事实，代价却是
/// 一次全量 I/O + 解码 —— 长工程上每次导入都要把整首歌再读一遍。
///
/// 【现在查什么】头（率/声道/位深/浮点格式）+ 帧数 + **末尾样本可读**：截断与格式损坏
/// 仍然会被发现并触发原子重建（`materialize` 的 tmp+rename），而载荷内部的随机位翻转
/// 不再被发现 —— 文件名即内容令牌，这条取舍是有意的。
fn matches_authorized_pcm(path: &Path, pcm: &PcmView<'_>) -> bool {
    let Ok(mut wav) = hound::WavReader::open(path) else {
        return false;
    };
    let spec = wav.spec();
    let frames = pcm.planes[0].len();
    if spec.sample_rate != pcm.sample_rate
        || spec.channels as usize != pcm.planes.len()
        || spec.bits_per_sample != 32
        || spec.sample_format != hound::SampleFormat::Float
        || wav.duration() as usize != frames
    {
        return false;
    }
    // 廉价完整性：能读到**最后一个**样本才算数据区没被截断（O(1)，不逐样本比对）。
    let last = (frames * pcm.planes.len()).saturating_sub(1);
    let last = u32::try_from(last).unwrap_or(u32::MAX);
    matches!(wav.seek(last), Ok(())) && matches!(wav.samples::<f32>().next(), Some(Ok(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 位置/名称不影响文件身份和mtime；同ID新PCM独立命名，私有文件损坏从授权源重建。
    #[test]
    fn materialized_analysis_file_reuses_content_and_repairs_corruption() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-host-pcm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let timeline:TimelineState=serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"T","order":0}],"bpm":120,"project_sec":1,
            "clips":[{"id":"clip","name":"C","track_id":"track","start_sec":0,"length_sec":4.0/44100.,
                "takes":[{"id":"take","source_path":"same-host-id","source_end_sec":4.0/44100.}]}]
        })).unwrap();
        let planes = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let view = PcmView {
            persistent_id: "same-host-id",
            sample_rate: 44100,
            planes: &planes,
        };
        let (mut first, _) = materialize(timeline.clone(), &[view], &dir).unwrap();
        let path = first.clips[0].source_path.clone().unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        first.clips[0].start_sec = 10.;
        first.clips[0].name = "Moved and renamed".into();
        first.clips[0].source_path = Some("same-host-id".into());
        first.clips[0].takes[0].source_path = Some("same-host-id".into());
        let (moved, _) = materialize(
            first,
            &[PcmView {
                persistent_id: "same-host-id",
                sample_rate: 44100,
                planes: &planes,
            }],
            &dir,
        )
        .unwrap();
        assert_eq!(moved.clips[0].source_path.as_deref(), Some(path.as_str()));
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4)
            .unwrap();
        let (repaired, _) = materialize(
            timeline.clone(),
            &[PcmView {
                persistent_id: "same-host-id",
                sample_rate: 44100,
                planes: &planes,
            }],
            &dir,
        )
        .unwrap();
        assert_eq!(
            repaired.clips[0].source_path.as_deref(),
            Some(path.as_str())
        );
        assert!(matches_authorized_pcm(
            Path::new(&path),
            &PcmView {
                persistent_id: "same-host-id",
                sample_rate: 44100,
                planes: &planes
            }
        ));
        let changed = vec![vec![0.5, 0.6, 0.7, 0.8]];
        let (next, _) = materialize(
            timeline,
            &[PcmView {
                persistent_id: "same-host-id",
                sample_rate: 44100,
                planes: &changed,
            }],
            &dir,
        )
        .unwrap();
        assert_ne!(next.clips[0].source_path.as_deref(), Some(path.as_str()));
    }

    /// 倒放的三种策略：独立 App 保持"整份拒绝"；插件在 F-4 之前"逐 clip 隔离"、
    /// F-4 之后"照常物化"（镜像窗口已在播种层翻回正向）。
    ///
    /// 【为什么这条必须存在】`Isolate` 曾让"一个片段倒放不再打死整个插件"；而 `Reject`
    /// 是独立 App 的既有行为，不能被削弱。`Materialize` 是"倒放真的能渲染"的依据。
    /// 三条一起钉住，才不会出现"为了修插件把独立 App 的校验也放开了"。
    #[test]
    fn reverse_policy_rejects_for_the_app_and_isolates_for_the_plugin() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-host-pcm-reverse-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"T","order":0}],"bpm":120,"project_sec":1,
            "clips":[
                {"id":"reversed","name":"R","track_id":"track","start_sec":0,"length_sec":4.0/44100.,
                 "takes":[{"id":"rt","source_path":"same-host-id","source_end_sec":4.0/44100.,
                           "reversed":true}]},
                {"id":"plain","name":"P","track_id":"track","start_sec":4.0/44100.,"length_sec":4.0/44100.,
                 "takes":[{"id":"pt","source_path":"same-host-id","source_end_sec":4.0/44100.}]}
            ]
        }))
        .unwrap();
        let planes = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let view = PcmView {
            persistent_id: "same-host-id",
            sample_rate: 44100,
            planes: &planes,
        };

        // 独立 App：整份拒绝（既有行为不变）。
        assert!(
            materialize_with_byte_limit(
                timeline.clone(),
                &[PcmView {
                    persistent_id: view.persistent_id,
                    sample_rate: view.sample_rate,
                    planes: view.planes,
                }],
                &dir,
                64 * 1024 * 1024,
                ReversePolicy::Reject,
            )
            .is_err(),
            "独立 App 的倒放校验不能被削弱"
        );

        // 插件（隔离策略，保留给其它调用方）：不物化倒放 clip，其余照常。
        let (isolated, _) = materialize_with_byte_limit(
            timeline.clone(),
            &[PcmView {
                persistent_id: view.persistent_id,
                sample_rate: view.sample_rate,
                planes: view.planes,
            }],
            &dir,
            64 * 1024 * 1024,
            ReversePolicy::Isolate,
        )
        .unwrap();
        let reversed = isolated
            .clips
            .iter()
            .find(|clip| clip.id == "reversed")
            .unwrap();
        assert!(reversed.reversed, "宿主事实保留，界面才能显示倒放标记");
        assert!(
            reversed.source_path.is_none(),
            "隔离 = 不物化：拿正向 PCM 当反向渲染是错的，而且不可见"
        );
        assert!(reversed.takes.iter().all(|take| take.source_path.is_none()));
        let plain = isolated
            .clips
            .iter()
            .find(|clip| clip.id == "plain")
            .unwrap();
        assert!(
            plain.source_path.is_some(),
            "其余 clip 必须照常物化 —— 隔离不是整份放弃"
        );

        // 插件（F-4 之后）：**照常物化**倒放 clip —— 方向由下游消费（镜像窗口已在播种层
        // 翻回正向，内核按反向窗口消费这份正向 PCM）。这是"倒放真的能渲染"的唯一依据。
        let (materialized, _) = materialize_with_byte_limit(
            timeline,
            &[view],
            &dir,
            64 * 1024 * 1024,
            ReversePolicy::Materialize,
        )
        .unwrap();
        let reversed = materialized
            .clips
            .iter()
            .find(|clip| clip.id == "reversed")
            .unwrap();
        assert!(reversed.reversed, "宿主事实保留");
        assert!(
            reversed.source_path.is_some(),
            "照常物化：内核按反向窗口消费这份正向 PCM"
        );
        assert!(reversed.takes.iter().all(|take| take.source_path.is_some()));
        let plain = materialized
            .clips
            .iter()
            .find(|clip| clip.id == "plain")
            .unwrap();
        assert!(plain.source_path.is_some(), "其余 clip 照常物化");
    }

    /// **不变量：`materialize` 与 `render/source.rs` 的读取门接受同一个采样率区间。**
    ///
    /// 【为什么这是回归测试】`render/source.rs` 已放开到 8k..=384k（宿主工程与素材可能是
    /// 88.2/96/176.4/192k，读取是离线的、下游会重采样），而这里此前只认 44.1/48k。
    /// 两处判据分叉的后果不是"某个片段降级"，而是"读进来 → `publish` → `materialize`
    /// 整会话失败"：**一个 96 kHz 素材就能让插件打不开**。
    #[test]
    fn materialize_accepts_the_same_sample_rates_as_the_ara_reader() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-host-pcm-rate-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let timeline: TimelineState = serde_json::from_value(serde_json::json!({
            "tracks":[{"id":"track","name":"T","order":0}],"bpm":120,"project_sec":1,
            "clips":[{"id":"clip","name":"C","track_id":"track","start_sec":0,"length_sec":4.0/96000.,
                "takes":[{"id":"take","source_path":"host-id","source_end_sec":4.0/96000.}]}]
        }))
        .unwrap();
        let planes = vec![vec![0.1, 0.2, 0.3, 0.4]];
        for sample_rate in [44_100_u32, 48_000, 88_200, 96_000, 192_000] {
            let (materialized, _) = materialize(
                timeline.clone(),
                &[PcmView {
                    persistent_id: "host-id",
                    sample_rate,
                    planes: &planes,
                }],
                &dir,
            )
            .unwrap_or_else(|error| panic!("{sample_rate} Hz 必须能物化：{error}"));
            assert!(
                materialized.clips[0].source_path.is_some(),
                "{sample_rate} Hz 的源必须拿到物化路径，否则会话整体加载失败"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 文件引用形态：摘要相符才解码，摘要不符必须**拒收**而不是读错音频。
    #[test]
    fn pcm_file_reference_verifies_the_digest_before_decoding() {
        let dir = std::env::temp_dir().join(format!(
            "hfs-host-pcm-file-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("host.wav");
        let mut wav = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        // 用二进制可精确表示的样本，逐位比较才有意义。
        let left = [0.5_f32, 0.25, 0.125, 0.0625];
        for sample in left {
            wav.write_sample(sample).unwrap();
            wav.write_sample(-sample).unwrap();
        }
        wav.finalize().unwrap();
        let digest = blake3::hash(&std::fs::read(&path).unwrap())
            .to_hex()
            .to_string();

        let (rate, planes) = load_pcm_file(&path, &digest).unwrap();
        assert_eq!(rate, 48_000);
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0], left.to_vec());
        assert_eq!(planes[1], left.iter().map(|s| -s).collect::<Vec<_>>());

        // 路径没变、内容变了 —— 这正是摘要要拦的那类故障。
        std::fs::write(&path, b"not the audio you were promised").unwrap();
        let error = load_pcm_file(&path, &digest).unwrap_err();
        assert!(error.contains("digest mismatch"), "{error}");

        // 非 float WAV：契约只认一种形态。
        let int_path = dir.join("host-int.wav");
        let mut int_wav = hound::WavWriter::create(
            &int_path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 44_100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        int_wav.write_sample(0_i16).unwrap();
        int_wav.finalize().unwrap();
        let int_digest = blake3::hash(&std::fs::read(&int_path).unwrap())
            .to_hex()
            .to_string();
        let error = load_pcm_file(&int_path, &int_digest).unwrap_err();
        assert!(error.contains("32-bit float WAV"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
