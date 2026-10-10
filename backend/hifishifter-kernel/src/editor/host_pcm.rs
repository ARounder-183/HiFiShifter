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
            || !matches!(pcm.sample_rate, 44100 | 48000)
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

/// 完整逐样本核对授权PCM后才复用私有分析文件；位置/名称变化保持mtime与原线cache。
/// 文件损坏重建采用临时文件原子替换，不允许分析worker读到半个WAV。
fn matches_authorized_pcm(path: &Path, pcm: &PcmView<'_>) -> bool {
    let Ok(mut wav) = hound::WavReader::open(path) else {
        return false;
    };
    let spec = wav.spec();
    if spec.sample_rate != pcm.sample_rate
        || spec.channels as usize != pcm.planes.len()
        || spec.bits_per_sample != 32
        || spec.sample_format != hound::SampleFormat::Float
        || wav.duration() as usize != pcm.planes[0].len()
    {
        return false;
    }
    let mut samples = wav.samples::<f32>();
    for frame in 0..pcm.planes[0].len() {
        for plane in pcm.planes {
            if !matches!(samples.next(),Some(Ok(sample)) if sample.to_bits()==plane[frame].to_bits())
            {
                return false;
            }
        }
    }
    samples.next().is_none()
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
}
