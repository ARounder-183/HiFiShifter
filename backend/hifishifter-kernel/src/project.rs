use crate::state::{SynthPipelineKind, TimelineState};
use crate::time_stretch::UserStretchAlgorithm;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
/// v4：Clip Take 结构（`takes` + `active_take_id`）、`Clip.loop_enabled`
/// （Loop / 循环源属性）与 `Clip.snap_offset_sec`。v3 及更早的扁平 Clip
/// 打开时迁移为单 Take，并按"为新的音频块启用循环"设置补齐 Loop
/// （见 open_project）。
///
/// v5：`ClipTake` 新增 `channel_mode`（声道模式，0..=4 对齐 REAPER
/// CHANMODE）与 `source_channels`（源声道数）。旧工程反序列化时
/// `channel_mode` 缺省为 0（正常）、`source_channels` 缺省为 None，
/// `finalize_timeline_for_session` 阶段对越界值规范化 —— 无需数据搬移。
///
/// v5 同期还加入了记事本字段：`notes_markdown`（正文）与 `notebook_assets`
/// （附件登记表 —— 图片与 HiFiShifter 剪贴板载荷，字节以 base64 **内嵌在
/// 工程文件里**，因此工程自包含：拷走/分享/打包都不会丢图）。两者都带
/// `serde(default)`，缺省为空，旧文件照常打开。
/// 当前工程文件格式版本。
///
/// v6：气声分离开关（`breath_enabled`）开始同时门控张力。为避免旧工程静默失去
/// 张力，打开 v5 及更早的工程时会自动置位该开关（见
/// [`crate::state::TimelineState::migrate_legacy_breath_separation`]）；
/// 该迁移**仅对 < v6 生效**，因此用户此后手动关闭开关的选择会持久保留。
pub const CURRENT_PROJECT_FILE_VERSION: u32 = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProjectFile {
    pub version: u32,
    pub name: String,
    /// 用户笔记；为空时省略（旧版本已容忍缺省，且空内容无可丢失信息）。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes_markdown: String,
    /// 记事本附件登记表：图片与剪贴板载荷，**字节（base64）内嵌在这里**。
    ///
    /// 用 `BTreeMap` 而非 `HashMap`：序列化顺序稳定，工程文件在不同次保存
    /// 之间可二进制比对。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub notebook_assets: BTreeMap<String, crate::notebook_assets::NotebookAsset>,
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
    /// 保存本工程时是否一并写出 UNDO 操作记录数据。
    ///
    /// 这是**工程级**开关（与「新建工程默认值」的全局设置相互独立）：
    /// 只影响保存，打开工程时总是尝试读取 UNDO 数据。缺省（旧文件没有该
    /// 字段）= 不写出，与全局「新建工程默认值」一致（默认关闭）。
    #[serde(default)]
    pub save_undo_history: bool,
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
            notebook_assets: BTreeMap::new(),
            timeline,
            base_scale,
            beats_per_bar,
            time_signature_denominator,
            grid_size,
            use_custom_scale: false,
            custom_scale: None,
            media_registry: Vec::new(),
            synth_config: SynthConfig::default(),
            save_undo_history: false,
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
        pf.timeline.normalize_clip_takes();
        pf.timeline.migrate_legacy_common_param_curves();
        pf.timeline.migrate_legacy_breath_separation(pf.version);
        pf.timeline.restore_derived_clip_fields();
        pf.timeline.sync_clip_takes_from_flat();
        return Ok(pf);
    }
    // fallback：JSON（兼容旧工程文件）
    serde_json::from_slice(bytes)
        .map_err(|e| format!("无法解析工程文件: {}", e))
        .map(|mut pf: ProjectFile| {
            pf.timeline.normalize_clip_takes();
            pf.timeline.migrate_legacy_common_param_curves();
            pf.timeline.migrate_legacy_breath_separation(pf.version);
            pf.timeline.restore_derived_clip_fields();
            pf.timeline.sync_clip_takes_from_flat();
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
    tl.sync_clip_takes_from_flat();
    for c in tl.clips.iter_mut() {
        for take in &mut c.takes {
            if let Some(sp) = take.source_path.clone() {
                let trimmed = sp.trim();
                if trimmed.is_empty() {
                    take.source_path_relative = None;
                } else {
                    let p = PathBuf::from(trimmed);
                    if p.is_absolute() {
                        take.source_path_relative = compute_relative_source_path(&p, project_path);
                    } else {
                        take.source_path_relative = Some(trimmed.replace('\\', "/"));
                    }
                }
            } else {
                take.source_path_relative = None;
            }
        }
        // active take 的内存投影与磁盘数据保持一致。
        let active_take = c
            .active_take_id
            .as_deref()
            .and_then(|id| c.takes.iter().find(|t| t.id == id))
            .or_else(|| c.takes.first())
            .cloned();
        if let Some(take) = active_take {
            take.apply_to_clip(c);
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
    tl.sync_clip_takes_from_flat();
    for clip in &mut tl.clips {
        for take in &mut clip.takes {
            take.waveform_preview = None;
            // 保证保存的工程中携带用于后续哈希匹配的内容指纹。
            // 已持久化或运行时刚更新的值保持不变；旧工程没有指纹时按当前
            // 磁盘文件补算一次，文件缺失则继续保持 None。
            if take.source_file_fingerprint.is_none() {
                if let Some(source_path) = take.source_path.as_deref().map(str::trim) {
                    if !source_path.is_empty() {
                        take.source_file_fingerprint =
                            crate::audio_utils::compute_file_fingerprint(Path::new(source_path));
                    }
                }
            }
        }
        let active_take = clip
            .active_take_id
            .as_deref()
            .and_then(|id| clip.takes.iter().find(|t| t.id == id))
            .or_else(|| clip.takes.first())
            .cloned();
        if let Some(take) = active_take {
            take.apply_to_clip(clip);
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
        for take in &mut c.takes {
            let source_path_raw = take
                .source_path
                .as_ref()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
            let source_path_relative_raw = take
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
                            take.source_path_relative = Some(sp.clone());
                        } else if missing_display_abs.is_none() {
                            missing_display_abs = Some(joined.to_string_lossy().to_string());
                        }
                    }
                }
            }

            if let Some(found) = resolved_absolute {
                take.source_path = Some(found);
                if take.source_path_relative.is_none() {
                    take.source_path_relative = source_path_relative_raw;
                }
            } else if let Some(missing_abs) = missing_display_abs {
                take.source_path = Some(missing_abs.clone());
                if take.source_path_relative.is_none() {
                    take.source_path_relative = source_path_relative_raw;
                }
                missing_files.insert(missing_abs);
            }
        }
        // 把 active take 的解析结果物化到内存投影。
        let active_take = c
            .active_take_id
            .as_deref()
            .and_then(|id| c.takes.iter().find(|t| t.id == id))
            .or_else(|| c.takes.first())
            .cloned();
        if let Some(take) = active_take {
            take.apply_to_clip(c);
        }
    }

    (tl, missing_files.into_iter().collect())
}

/// 把**刚反序列化**出来的 `TimelineState` 整理成可投入会话使用的形态。
///
/// 磁盘形态（工程文件 / `-UNDO` 伴生文件）与运行时形态并不等价：
/// - `Clip` 上的媒体字段（`source_path` / `duration_frames` /
///   `source_sample_rate` / 内容指纹 / 波形预览…）是 active take 的**内存
///   投影**，以 `#[serde(skip_serializing)]` 省略，权威数据只在 `takes` 里；
/// - `source_file_mtime` / `source_file_size` 等运行时元数据是 `#[serde(skip)]`；
/// - `project_scale_notes`、空参数记录等派生/可重建数据在保存时会精简掉。
///
/// 因此任何「反序列化后直接用」的路径都必须走这里统一物化：**少了
/// `normalize_clip_takes()` 这一步，恢复出来的 Clip 就只有时间位置、没有
/// 音频**（典型症状：读取 `-UNDO` 后撤销，Clip 里的音频内容被清空）。
///
/// 工程打开与操作记录恢复共用此函数，避免两条恢复路径各自演化而漂移。
pub fn finalize_timeline_for_session(
    timeline: crate::state::TimelineState,
    project_path: &Path,
    project_file_version: u32,
) -> (crate::state::TimelineState, Vec<String>) {
    let (mut tl, missing_files) = resolve_source_paths_on_open(timeline, project_path);

    // 旧工程兼容迁移（仅 v3 及更早）：source_end_sec == 0.0 曾表示"到源文件
    // 末尾"，新语义要求它是真实的结束时间，此处自动修正为 duration_sec 或
    // length_sec。v4+ 工程的 se 恒为真实坐标，不得改写。
    if project_file_version < 4 {
        for clip in &mut tl.clips {
            if clip.source_end_sec == 0.0 {
                clip.source_end_sec = clip.duration_sec.unwrap_or(clip.length_sec);
            }
        }
    }
    // v4 迁移：v3 及更早的工程 Clip 不携带 loop_enabled，按当前"为新的音频块
    // 启用循环"设置作为这些既有**音频** Clip 的 Loop 属性。
    if project_file_version < 4 {
        let default_loop = crate::config::loop_new_clips_default();
        for clip in &mut tl.clips {
            if clip.source_path.is_some() {
                clip.loop_enabled = default_loop;
            }
        }
    }
    // 非 Loop 存储窗口规范化（对**所有版本**生效）+ take 窗口自愈。
    for clip in &mut tl.clips {
        crate::state::normalize_nonloop_source_window(clip);
        crate::state::normalize_nonloop_all_take_windows(clip);
    }
    // 上面的规范化改的是 active take 内存投影：先写回 Take 权威数据。
    tl.sync_clip_takes_from_flat();
    // 再由 Take 物化回 Clip 投影（补齐 duration_frames / source_sample_rate /
    // 指纹 / 波形等被序列化省略的字段），并顺带完成旧 Fade 字段迁移。
    tl.normalize_clip_takes();
    tl.migrate_legacy_common_param_curves();
    // 张力/气声曲线存在但开关未开 ⇒ 自动置位开关（否则会静默失去效果，见该函数说明）。
    // 传入版本号以**只跑一次**：见该函数的 one-shot 说明。
    tl.migrate_legacy_breath_separation(project_file_version);
    // 归一化轨道顺序（Vec 顺序 == 显示顺序）与 Tempo Map（排序/钳制/补 0 点）。
    tl.normalize_track_vec();
    tl.normalize_tempo_map();
    // 运行时文件元数据（外部文件变更检测）：不落盘，按当前磁盘补算。
    for clip in &mut tl.clips {
        crate::state::TimelineState::populate_clip_file_metadata(clip);
    }
    tl.sync_clip_takes_from_flat();

    // 加载边界：清掉历史上被批量伪造的"用户封印"（标着用户决定、却没有记录
    // 用户选了什么）。它们会让 Take 永久免疫于折叠，而本程序写出的每个工程都
    // 是 v5，等于折叠功能对所有保存过的工程失效 —— 必须清回"未判定"重新裁决。
    //
    // 与工程版本**无关**：判定该不该扫的依据是档案本身（谁定的、选了什么），
    // 不是版本号。这也是"版本号只用于格式迁移、不用于语义推断"的落点。
    let cleared = tl.clear_untrusted_channel_seals();
    if cleared > 0 {
        log::info!("[open_project] cleared {cleared} fabricated channel seal(s) back to undecided");
    }

    (tl, missing_files)
}
