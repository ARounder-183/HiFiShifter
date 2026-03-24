use crate::state::{SynthPipelineKind, TimelineState};
use serde::{Deserialize, Serialize};
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

// ─── 路径处理 ──────────────────────────────────────────────────────────────────

pub fn project_name_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled")
        .to_string()
}

pub fn make_paths_relative(mut tl: TimelineState, project_path: &Path) -> TimelineState {
    let dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    for c in tl.clips.iter_mut() {
        if let Some(sp) = c.source_path.clone() {
            let p = PathBuf::from(&sp);
            if p.is_absolute() {
                if let Ok(stripped) = p.strip_prefix(dir) {
                    c.source_path = Some(stripped.to_string_lossy().to_string());
                }
            }
        }
    }
    tl
}

pub fn resolve_paths_relative(mut tl: TimelineState, project_path: &Path) -> TimelineState {
    let dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    for c in tl.clips.iter_mut() {
        if let Some(sp) = c.source_path.clone() {
            let p = PathBuf::from(&sp);
            if !p.is_absolute() {
                let joined = dir.join(p);
                c.source_path = Some(joined.to_string_lossy().to_string());
            }
        }
    }
    tl
}
