use crate::state::{SynthPipelineKind, TimelineState};
use crate::time_stretch::UserStretchAlgorithm;
use serde::{Deserialize, Serialize};
use std::path::Component;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CustomScale {
    pub id: String,
    pub name: String,
    pub notes: Vec<u8>,
}

impl CustomScale {
    pub fn normalized(&self) -> Self {
        let mut unique = std::collections::BTreeSet::new();
        for n in &self.notes {
            unique.insert(n % 12);
        }
        let mut notes: Vec<u8> = unique.into_iter().collect();
        if notes.is_empty() {
            notes = vec![0, 2, 4, 5, 7, 9, 11];
        }
        Self {
            id: if self.id.trim().is_empty() {
                "custom".to_string()
            } else {
                self.id.trim().to_string()
            },
            name: if self.name.trim().is_empty() {
                "Custom Scale".to_string()
            } else {
                self.name.trim().to_string()
            },
            notes,
        }
    }
}

// ─── 媒体注册表 ────────────────────────────────────────────────────────────────

/// 工程媒体文件注册表条目，用于追踪音频文件的路径和完整性。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaEntry {
    /// 唯一标识符。
    pub id: String,
    /// 导入时的原始绝对路径。
    pub original_path: String,
    /// 相对于工程文件的相对路径（保存时写入）。
    pub relative_path: String,
    /// 文件内容的 SHA-256 哈希，用于完整性校验。
    pub sha256: [u8; 32],
}

// ─── 合成配置 ──────────────────────────────────────────────────────────────────

/// 工程级合成配置。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SynthConfig {
    /// 工程默认合成管线，`None` 时由 Track 的 `pitch_analysis_algo` 决定。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_pipeline: Option<SynthPipelineKind>,
    /// 工程级外部时间拉伸算法覆盖；`None` 表示继承全局默认值。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stretch_algorithm_override: Option<UserStretchAlgorithm>,
    /// 工程级 HiFiGAN mel-stretch 开关覆盖；`None` 表示继承全局默认值。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hifigan_mel_stretch_override: Option<bool>,
}

impl SynthConfig {
    fn is_default(&self) -> bool {
        self.default_pipeline.is_none()
            && self.stretch_algorithm_override.is_none()
            && self.hifigan_mel_stretch_override.is_none()
    }
}

// ─── 工程文件 ──────────────────────────────────────────────────────────────────

/// 当前程序读写的最新工程文件版本号。
///
/// 打开工程时若文件版本高于该值，必须先经用户确认后才能尝试加载。
///
/// v4：`Clip.loop_enabled`（Loop / 循环源属性）。v3 及更早的工程不含该字段，
/// 打开时按"为新的音频块启用循环"设置迁移（见 open_project）。
pub const CURRENT_PROJECT_FILE_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProjectFile {
    pub version: u32,
    pub name: String,
    /// 用户笔记；为空时省略（旧版本已容忍缺省，且空内容无可丢失信息）。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes_markdown: String,
    pub timeline: TimelineState,
    /// 工程的基础音乐参数（基准音阶/拍号/网格）始终序列化。
    /// 这些参数定义工程的语义身份，不能依赖"缺省 = 默认值"的隐式规则：
    /// 一旦未来版本调整默认值，历史文件将静默改变含义，且不利于跨版本兼容与维护。
    #[serde(default = "default_base_scale")]
    pub base_scale: String,
    #[serde(default = "default_beats_per_bar")]
    pub beats_per_bar: u32,
    /// 工程基准拍号分母（v2 新增，旧工程反序列化时默认 4）。
    #[serde(default = "default_time_signature_denominator")]
    pub time_signature_denominator: u32,
    #[serde(default = "default_grid_size")]
    pub grid_size: String,
    #[serde(default)]
    pub use_custom_scale: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_scale: Option<CustomScale>,
    /// 媒体文件注册表（v2 新增，旧工程反序列化时默认为空）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media_registry: Vec<MediaEntry>,
    /// 工程级合成配置（v2 新增，旧工程反序列化时使用默认值）。
    #[serde(default, skip_serializing_if = "SynthConfig::is_default")]
    pub synth_config: SynthConfig,
}

impl ProjectFile {
    pub fn new(
        name: String,
        timeline: TimelineState,
        base_scale: String,
        beats_per_bar: u32,
        time_signature_denominator: u32,
        grid_size: String,
    ) -> Self {
        Self {
            version: CURRENT_PROJECT_FILE_VERSION,
            name,
            notes_markdown: String::new(),
            timeline,
            base_scale,
            beats_per_bar,
            time_signature_denominator,
            grid_size,
            use_custom_scale: false,
            custom_scale: None,
            media_registry: Vec::new(),
            synth_config: SynthConfig::default(),
        }
    }
}

fn default_base_scale() -> String {
    "C".to_string()
}

fn default_beats_per_bar() -> u32 {
    4
}

fn default_time_signature_denominator() -> u32 {
    4
}

fn default_grid_size() -> String {
    "1/4".to_string()
}

// ─── 序列化 / 反序列化 ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ProjectFileVersionProbe {
    #[serde(default)]
    version: Option<u32>,
}

/// 只读取工程文件头部的版本号，不解析完整时间轴。
///
/// 即使未来版本的工程文件因结构变化而无法完整反序列化，打开工程时也能
/// 先依据版本号向用户发出“可能不兼容”的警告。
pub fn read_project_file_version(bytes: &[u8]) -> Option<u32> {
    if let Ok(probe) = rmp_serde::from_slice::<ProjectFileVersionProbe>(bytes) {
        if probe.version.is_some() {
            return probe.version;
        }
    }
    serde_json::from_slice::<ProjectFileVersionProbe>(bytes)
        .ok()
        .and_then(|probe| probe.version)
}

/// 从字节流加载工程文件，自动检测格式。
///
/// 优先尝试 MessagePack 格式（v3），失败后 fallback 到 JSON（v1/v2 兼容）。
pub fn load_project_file(bytes: &[u8]) -> Result<ProjectFile, String> {
    // 先尝试 MessagePack（新格式）
    if let Ok(mut pf) = rmp_serde::from_slice::<ProjectFile>(bytes) {
        pf.timeline.migrate_legacy_common_param_curves();
        pf.timeline.restore_derived_clip_fields();
        return Ok(pf);
    }
    // fallback：JSON（兼容旧工程文件）
    serde_json::from_slice(bytes)
        .map_err(|e| format!("无法解析工程文件: {}", e))
        .map(|mut pf: ProjectFile| {
            pf.timeline.migrate_legacy_common_param_curves();
            pf.timeline.restore_derived_clip_fields();
            pf
        })
}

pub fn is_json_project_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

pub fn serialize_project_file_for_path(pf: &ProjectFile, path: &Path) -> Result<Vec<u8>, String> {
    if is_json_project_path(path) {
        // 当用户选择 .json 后缀时，按 JSON 文本保存工程。
        // 使用紧凑输出：工程文件以可移植/可读为主，不承担人工编辑的排版职责，
        // pretty-printing 会给长参数曲线文件带来成倍的缩进开销。
        return serde_json::to_vec(pf).map_err(|e| e.to_string());
    }
    rmp_serde::to_vec_named(pf).map_err(|e| e.to_string())
}

// ─── 路径处理 ──────────────────────────────────────────────────────────────────

pub fn project_name_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled")
        .to_string()
}

fn compute_relative_source_path(source_path: &Path, project_path: &Path) -> Option<String> {
    let project_dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    let base_dir_abs = if project_dir.is_absolute() {
        project_dir.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(project_dir)
    };
    let source_abs = if source_path.is_absolute() {
        source_path.to_path_buf()
    } else {
        base_dir_abs.join(source_path)
    };

    let base_components: Vec<Component<'_>> = base_dir_abs.components().collect();
    let source_components: Vec<Component<'_>> = source_abs.components().collect();

    let mut common = 0usize;
    while common < base_components.len()
        && common < source_components.len()
        && base_components[common] == source_components[common]
    {
        common += 1;
    }

    if common == 0 {
        return None;
    }

    let mut rel_parts: Vec<String> = Vec::new();

    for comp in &base_components[common..] {
        if matches!(comp, Component::Normal(_)) {
            rel_parts.push("..".to_string());
        }
    }

    for comp in &source_components[common..] {
        match comp {
            Component::Normal(part) => rel_parts.push(part.to_string_lossy().to_string()),
            Component::ParentDir => rel_parts.push("..".to_string()),
            Component::CurDir => {}
            _ => {}
        }
    }

    if rel_parts.is_empty() {
        return None;
    }

    Some(rel_parts.join("/"))
}

pub fn prepare_source_paths_for_save(mut tl: TimelineState, project_path: &Path) -> TimelineState {
    for c in tl.clips.iter_mut() {
        if let Some(sp) = c.source_path.clone() {
            let trimmed = sp.trim();
            if trimmed.is_empty() {
                c.source_path_relative = None;
            } else {
                let p = PathBuf::from(trimmed);
                if p.is_absolute() {
                    c.source_path_relative = compute_relative_source_path(&p, project_path);
                } else {
                    c.source_path_relative = Some(trimmed.replace('\\', "/"));
                }
            }
        } else {
            c.source_path_relative = None;
        }
    }
    tl
}

fn curve_is_all_zero(curve: &[f32]) -> bool {
    curve.iter().all(|value| *value == 0.0)
}

/// 保存工程文件前的专门精简处理。
///
/// 与内存中的运行时 `TimelineState` 不同，落盘数据采用如下策略：
/// - 基础/语义参数（音准曲线之外的用户配置）一律原样序列化，确保自描述性
///   与跨版本兼容（见 `Track` / `Clip` / `ProjectFile` 上的 serde 注解）。
/// - 纯缓存/派生数据置空后仍以 `null`/空值落盘（字段保持存在，兼容旧版本
///   反序列化要求字段必须存在），内容由现有分析/波形管线重新生成：
///   - `waveform_preview`：波形预览缓存（前端另有 mipmap 二进制缓存）。
///   - `pitch_edit` / `pitch_orig` 的冗余副本：未编辑时二者相同，只保留 orig；
///     已编辑时 orig 仍可能作为未编辑帧的基线，因此按需保留。
///   - `tension_orig`：从未参与渲染，属于历史遗留字段。
///   - 全零曲线与空 `extra_curves`：与反序列化后的默认值语义完全一致。
///   - `project_scale_notes`：可由 `base_scale` / `custom_scale` / tempo map 重建。
pub fn prepare_timeline_for_project_save(
    mut tl: TimelineState,
    project_path: &Path,
) -> TimelineState {
    for clip in &mut tl.clips {
        clip.waveform_preview = None;
        // 保证保存的工程中携带用于后续哈希匹配的内容指纹。
        // 已持久化或运行时刚更新的值保持不变；旧工程没有指纹时按当前
        // 磁盘文件补算一次，文件缺失则继续保持 None。
        if clip.source_file_fingerprint.is_none() {
            if let Some(source_path) = clip.source_path.as_deref().map(str::trim) {
                if !source_path.is_empty() {
                    clip.source_file_fingerprint =
                        crate::audio_utils::compute_file_fingerprint(Path::new(source_path));
                }
            }
        }
        // 注意：duration_sec / duration_frames / source_sample_rate 是基础媒体信息，
        // 始终原样序列化（不省略），以免旧版本读取时缺少必需字段。
        if let Some(curves) = clip.extra_curves.as_mut() {
            curves.retain(|_, curve| !curve.is_empty());
            if curves.is_empty() {
                clip.extra_curves = None;
            }
        }
        if let Some(params) = clip.extra_params.as_mut() {
            if params.is_empty() {
                clip.extra_params = None;
            }
        }
    }

    for params in tl.params_by_root_track.values_mut() {
        if params.pitch_edit_user_modified {
            // 用户只编辑了部分帧时，orig 仍是未编辑帧的显示/导出基线，保留；
            // 全零 orig 没有信息量，直接省略。
            if curve_is_all_zero(&params.pitch_orig) {
                params.pitch_orig.clear();
            }
            if curve_is_all_zero(&params.pitch_edit) {
                params.pitch_edit.clear();
            }
        } else {
            // 未编辑时 pitch_edit 只是 pitch_orig 的同步副本，落盘一份即可。
            params.pitch_edit.clear();
            if curve_is_all_zero(&params.pitch_orig) {
                params.pitch_orig.clear();
            }
        }

        // tension_orig 目前没有任何读取路径写入非零数据，且渲染只消费
        // extra_curves / tension_edit；落盘时始终省略。
        params.tension_orig.clear();
        if curve_is_all_zero(&params.tension_edit) {
            params.tension_edit.clear();
        }

        // 空曲线与缺失 key 语义相同（都取参数默认值）。
        params.extra_curves.retain(|_, curve| !curve.is_empty());
    }

    // 精简后没有任何用户数据的 root-track 参数记录直接删除；
    // 打开工程时会按需用 ensure_params_for_root 重新创建。
    tl.params_by_root_track
        .retain(|_, params| !params.is_empty_project_data());

    // project_scale_notes 是 base_scale / custom_scale / tempo map 的派生缓存，
    // 打开工程时会重新计算（见 open_project 中的 effective_scale_notes）。
    tl.project_scale_notes.clear();

    prepare_source_paths_for_save(tl, project_path)
}

pub fn resolve_source_paths_on_open(
    mut tl: TimelineState,
    project_path: &Path,
) -> (TimelineState, Vec<String>) {
    let dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    let mut missing_files = std::collections::BTreeSet::new();

    for c in tl.clips.iter_mut() {
        let source_path_raw = c
            .source_path
            .as_ref()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let source_path_relative_raw = c
            .source_path_relative
            .as_ref()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        let mut resolved_absolute: Option<String> = None;
        let mut missing_display_abs: Option<String> = None;

        if let Some(sp) = source_path_raw.as_ref() {
            let p = PathBuf::from(sp);
            if p.is_absolute() {
                if p.exists() {
                    resolved_absolute = Some(p.to_string_lossy().to_string());
                } else {
                    missing_display_abs = Some(p.to_string_lossy().to_string());
                }
            }
        }

        if resolved_absolute.is_none() {
            if let Some(rel) = source_path_relative_raw.as_ref() {
                let joined = dir.join(rel);
                if joined.exists() {
                    resolved_absolute = Some(joined.to_string_lossy().to_string());
                } else if missing_display_abs.is_none() {
                    missing_display_abs = Some(joined.to_string_lossy().to_string());
                }
            }
        }

        if resolved_absolute.is_none() {
            if let Some(sp) = source_path_raw.as_ref() {
                let p = PathBuf::from(sp);
                if !p.is_absolute() {
                    let joined = dir.join(p);
                    if joined.exists() {
                        resolved_absolute = Some(joined.to_string_lossy().to_string());
                        c.source_path_relative = Some(sp.clone());
                    } else if missing_display_abs.is_none() {
                        missing_display_abs = Some(joined.to_string_lossy().to_string());
                    }
                }
            }
        }

        if let Some(found) = resolved_absolute {
            c.source_path = Some(found);
            if c.source_path_relative.is_none() {
                c.source_path_relative = source_path_relative_raw;
            }
        } else if let Some(missing_abs) = missing_display_abs {
            c.source_path = Some(missing_abs.clone());
            if c.source_path_relative.is_none() {
                c.source_path_relative = source_path_relative_raw;
            }
            missing_files.insert(missing_abs);
        }
    }

    (tl, missing_files.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{PitchAnalysisAlgo, TimelineState};

    fn project_file_with_clip(tl: TimelineState) -> ProjectFile {
        ProjectFile::new(
            "test".to_string(),
            tl,
            "D".to_string(),
            3,
            8,
            "1/8".to_string(),
        )
    }

    fn timeline_with_clip_and_zero_curves() -> TimelineState {
        let mut tl = TimelineState::default();
        let root = tl.tracks[0].id.clone();
        tl.ensure_params_for_root(&root);
        let clip_id = tl.add_clip(
            Some(root.clone()),
            Some("Vocal".to_string()),
            Some(0.0),
            Some(3.0),
            Some("C:/audio/Vocal.wav".to_string()),
        );
        if let Some(clip) = tl.clips.iter_mut().find(|c| c.id == clip_id) {
            clip.waveform_preview = Some(vec![0.25f32; 4096]);
        }
        if let Some(params) = tl.params_by_root_track.get_mut(&root) {
            params.pitch_orig = vec![0.0f32; 6400];
            params.pitch_edit = vec![0.0f32; 6400];
            params.tension_orig = vec![0.0f32; 6400];
            params.tension_edit = vec![0.0f32; 6400];
        }
        tl
    }

    #[test]
    fn json_project_output_is_compact() {
        let pf = project_file_with_clip(TimelineState::default());
        let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(!text.contains('\n'), "JSON project should be compact");
        assert!(text.contains("\"version\":4"));
    }

    #[test]
    fn project_file_version_can_be_read_without_full_timeline_parse() {
        let pf = project_file_with_clip(TimelineState::default());
        let json_bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
        assert_eq!(read_project_file_version(&json_bytes), Some(4));

        let msgpack_bytes = serialize_project_file_for_path(&pf, Path::new("test.hshp")).unwrap();
        assert_eq!(read_project_file_version(&msgpack_bytes), Some(4));
    }

    #[test]
    fn prepare_project_save_strips_waveform_and_zero_curves() {
        let tl = timeline_with_clip_and_zero_curves();
        let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));

        let clip = &prepared.clips[0];
        assert!(
            clip.waveform_preview.is_none(),
            "waveform cache must not be saved"
        );
        assert!(clip.source_path_relative.is_some());

        assert!(
            prepared.params_by_root_track.is_empty(),
            "all-default root params should be omitted entirely"
        );
        assert!(prepared.project_scale_notes.is_empty());
    }

    #[test]
    fn compact_json_keeps_required_and_core_fields() {
        // 即使全部处于默认值，工程核心参数与旧版本必需的基础字段也必须始终出现。
        let tl = prepare_timeline_for_project_save(
            timeline_with_clip_and_zero_curves(),
            Path::new("test.hshp"),
        );
        let pf = project_file_with_clip(tl);
        let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();

        for key in [
            "\"base_scale\"",
            "\"beats_per_bar\"",
            "\"time_signature_denominator\"",
            "\"grid_size\"",
            "\"use_custom_scale\"",
            "\"version\"",
        ] {
            assert!(
                text.contains(key),
                "core project parameter {key} must always be serialized"
            );
        }

        // Track / Clip / TimelineState 中属于旧版本必需或基础语义的字段
        // 即使取默认值也必须存在（旧版本反序列化要求字段必须携带）。
        for key in [
            "\"parent_id\"",
            "\"muted\"",
            "\"solo\"",
            "\"volume\"",
            "\"compose_enabled\"",
            "\"pitch_analysis_algo\"",
            "\"source_path\"",
            "\"duration_sec\"",
            "\"duration_frames\"",
            "\"source_sample_rate\"",
            "\"waveform_preview\"",
            "\"pitch_range\"",
            "\"gain\"",
            "\"source_start_sec\"",
            "\"source_end_sec\"",
            "\"playback_rate\"",
            "\"reversed\"",
            "\"loop_enabled\"",
            "\"fade_in_sec\"",
            "\"fade_out_sec\"",
            "\"fade_in_curve\"",
            "\"fade_out_curve\"",
            "\"selected_track_id\"",
            "\"selected_clip_id\"",
            "\"playhead_sec\"",
        ] {
            assert!(
                text.contains(key),
                "field {key} must always be present to stay readable by older app versions"
            );
        }
    }

    #[test]
    fn zero_curve_project_serializes_tiny() {
        let tl = timeline_with_clip_and_zero_curves();
        let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
        let pf = project_file_with_clip(prepared);
        let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
        assert!(
            bytes.len() < 2048,
            "cache/default-only project should serialize to <2KB, got {}",
            bytes.len()
        );
    }

    #[test]
    fn prepare_project_save_keeps_user_edited_and_unmodified_pitch_data() {
        let mut tl = timeline_with_clip_and_zero_curves();
        let root = tl.tracks[0].id.clone();
        let params = tl.params_by_root_track.get_mut(&root).unwrap();
        params.pitch_orig = vec![60.0f32; 10];
        params.pitch_edit = vec![62.0f32; 10];
        params.pitch_edit_user_modified = true;
        params.tension_orig = vec![1.0f32; 10];
        params.tension_edit = vec![2.0f32; 10];
        params
            .extra_curves
            .insert("volume".to_string(), vec![0.5f32; 10]);

        let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
        let params = prepared
            .params_by_root_track
            .get(&root)
            .expect("root params should exist");
        assert_eq!(
            params.pitch_orig.len(),
            10,
            "edited pitch keeps orig baseline"
        );
        assert_eq!(params.pitch_edit.len(), 10);
        assert!(
            params.tension_orig.is_empty(),
            "tension_orig is a legacy cache"
        );
        assert_eq!(params.tension_edit.len(), 10);
        assert_eq!(params.extra_curves.get("volume").map(Vec::len), Some(10));

        // Unmodified track: the edit copy is redundant with orig.
        let mut unmodified = TimelineState::default();
        let unmodified_root = unmodified.tracks[0].id.clone();
        unmodified.ensure_params_for_root(&unmodified_root);
        let params = unmodified
            .params_by_root_track
            .get_mut(&unmodified_root)
            .unwrap();
        params.pitch_orig = vec![57.0f32; 8];
        params.pitch_edit = params.pitch_orig.clone();
        let prepared =
            prepare_timeline_for_project_save(unmodified, Path::new("C:/proj/unmodified.hshp"));
        let params = prepared
            .params_by_root_track
            .get(&unmodified_root)
            .expect("root params should exist");
        assert_eq!(params.pitch_orig.len(), 8);
        assert!(params.pitch_edit.is_empty());
    }

    #[test]
    fn compact_json_roundtrip_preserves_non_default_fields() {
        let mut tl = timeline_with_clip_and_zero_curves();
        let root = tl.tracks[0].id.clone();
        {
            let track = &mut tl.tracks[0];
            track.muted = true;
            track.volume = 0.5;
            track.compose_enabled = true;
            track.pitch_analysis_algo = PitchAnalysisAlgo::WorldDll;
            track.color = "#112233".to_string();
        }
        {
            let clip = &mut tl.clips[0];
            clip.gain = 0.75;
            clip.muted = true;
            clip.source_start_sec = 0.25;
            clip.playback_rate = 1.5;
            clip.reversed = true;
            clip.loop_enabled = true;
            clip.fade_in_sec = 0.1;
            clip.fade_out_sec = 0.2;
            clip.fade_in_curve = "logarithmic".to_string();
            clip.fade_out_curve = "exponential".to_string();
            clip.color = "blue".to_string();
            clip.source_file_fingerprint = Some(0x1122334455667788);
        }
        {
            let params = tl.params_by_root_track.get_mut(&root).unwrap();
            params.pitch_orig = vec![60.0f32; 12];
            params.pitch_edit = vec![63.0f32; 12];
            params.pitch_edit_user_modified = true;
            params.tension_edit = vec![1.5f32; 12];
        }

        let prepared = prepare_timeline_for_project_save(tl, Path::new("C:/proj/test.hshp"));
        let mut pf = project_file_with_clip(prepared);
        pf.use_custom_scale = true;
        pf.custom_scale = Some(CustomScale {
            id: "custom".to_string(),
            name: "Custom".to_string(),
            notes: vec![0, 2, 3, 5, 7, 9, 10],
        });
        pf.notes_markdown = "# Notes".to_string();
        pf.synth_config.default_pipeline = Some(SynthPipelineKind::WorldVocoder);

        let bytes = serialize_project_file_for_path(&pf, Path::new("test.json")).unwrap();
        let loaded = load_project_file(&bytes).expect("compact JSON must roundtrip");

        assert_eq!(loaded.name, "test");
        assert_eq!(loaded.base_scale, "D");
        assert_eq!(loaded.beats_per_bar, 3);
        assert_eq!(loaded.time_signature_denominator, 8);
        assert_eq!(loaded.grid_size, "1/8");
        assert_eq!(loaded.notes_markdown, "# Notes");
        assert!(loaded.use_custom_scale);
        assert_eq!(
            loaded.custom_scale.as_ref().map(|s| s.notes.clone()),
            Some(vec![0, 2, 3, 5, 7, 9, 10])
        );

        let track = &loaded.timeline.tracks[0];
        assert!(track.muted);
        assert!((track.volume - 0.5).abs() < f32::EPSILON);
        assert!(track.compose_enabled);
        assert_eq!(track.pitch_analysis_algo, PitchAnalysisAlgo::WorldDll);
        assert_eq!(track.color, "#112233");

        let clip = &loaded.timeline.clips[0];
        assert!((clip.gain - 0.75).abs() < f32::EPSILON);
        assert!(clip.muted);
        assert!((clip.source_start_sec - 0.25).abs() < 1e-12);
        assert!((clip.playback_rate - 1.5).abs() < f32::EPSILON);
        assert!(clip.reversed);
        assert!(clip.loop_enabled, "loop flag must roundtrip");
        assert!((clip.fade_in_sec - 0.1).abs() < 1e-12);
        assert!((clip.fade_out_sec - 0.2).abs() < 1e-12);
        assert_eq!(clip.fade_in_curve, "logarithmic");
        assert_eq!(clip.fade_out_curve, "exponential");
        assert_eq!(
            clip.source_file_fingerprint,
            Some(0x1122334455667788),
            "source fingerprint must be persisted for later hash matching"
        );
        assert!(
            clip.waveform_preview.is_none(),
            "waveform preview is stripped"
        );

        let params = loaded.timeline.params_by_root_track.get(&root).unwrap();
        assert_eq!(params.pitch_orig.len(), 12);
        assert_eq!(params.pitch_edit.len(), 12);
        assert!(params.pitch_edit_user_modified);
        assert_eq!(params.tension_edit.len(), 12);
        assert_eq!(
            loaded.synth_config.default_pipeline,
            Some(SynthPipelineKind::WorldVocoder)
        );
    }
}
