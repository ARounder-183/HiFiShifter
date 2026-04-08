use crate::state::{SynthPipelineKind, TimelineState};
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
    #[serde(default)]
    pub default_pipeline: Option<SynthPipelineKind>,
}

// ─── 工程文件 ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProjectFile {
    pub version: u32,
    pub name: String,
    pub timeline: TimelineState,
    #[serde(default = "default_base_scale")]
    pub base_scale: String,
    #[serde(default = "default_beats_per_bar")]
    pub beats_per_bar: u32,
    #[serde(default = "default_grid_size")]
    pub grid_size: String,
    #[serde(default)]
    pub use_custom_scale: bool,
    #[serde(default)]
    pub custom_scale: Option<CustomScale>,
    /// 媒体文件注册表（v2 新增，旧工程反序列化时默认为空）。
    #[serde(default)]
    pub media_registry: Vec<MediaEntry>,
    /// 工程级合成配置（v2 新增，旧工程反序列化时使用默认值）。
    #[serde(default)]
    pub synth_config: SynthConfig,
}

impl ProjectFile {
    pub fn new(
        name: String,
        timeline: TimelineState,
        base_scale: String,
        beats_per_bar: u32,
        grid_size: String,
    ) -> Self {
        Self {
            version: 2,
            name,
            timeline,
            base_scale,
            beats_per_bar,
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

fn default_grid_size() -> String {
    "1/4".to_string()
}

// ─── 序列化 / 反序列化 ─────────────────────────────────────────────────────────

/// 从字节流加载工程文件，自动检测格式。
///
/// 优先尝试 MessagePack 格式（v2），失败后 fallback 到 JSON（v1 兼容）。
pub fn load_project_file(bytes: &[u8]) -> Result<ProjectFile, String> {
    // 先尝试 MessagePack（新格式）
    if let Ok(pf) = rmp_serde::from_slice::<ProjectFile>(bytes) {
        return Ok(pf);
    }
    // fallback：JSON（兼容旧工程文件）
    serde_json::from_slice(bytes).map_err(|e| format!("无法解析工程文件: {}", e))
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
        return serde_json::to_vec_pretty(pf).map_err(|e| e.to_string());
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

pub fn resolve_source_paths_on_open(mut tl: TimelineState, project_path: &Path) -> (TimelineState, Vec<String>) {
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
