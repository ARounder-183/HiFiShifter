//! Tauri 命令：外部 Resampler 注册表管理（增删查、扫描）。

use crate::state::{AppState, FlagParam, ResamplerEntry};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::State;
use uuid::Uuid;

// ─── Payload 类型 ──────────────────────────────────────────────────────────────

/// 前端展示的 flag 参数条目。
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FlagParamPayload {
    pub key: String,
    pub display_name: String,
    pub min_value: f64,
    pub max_value: f64,
    pub default_value: f64,
}

impl From<&FlagParam> for FlagParamPayload {
    fn from(fp: &FlagParam) -> Self {
        Self {
            key: fp.key.clone(),
            display_name: fp.display_name.clone(),
            min_value: fp.min_value,
            max_value: fp.max_value,
            default_value: fp.default_value,
        }
    }
}

impl From<FlagParamPayload> for FlagParam {
    fn from(fp: FlagParamPayload) -> Self {
        Self {
            key: fp.key,
            display_name: fp.display_name,
            min_value: fp.min_value,
            max_value: fp.max_value,
            default_value: fp.default_value,
        }
    }
}

/// 前端展示的 resampler 条目。
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResamplerEntryPayload {
    pub id: String,
    pub display_name: String,
    pub exe_path: String,
    pub default_flags: String,
    pub flag_params: Vec<FlagParamPayload>,
    pub available: bool,
}

impl From<&ResamplerEntry> for ResamplerEntryPayload {
    fn from(e: &ResamplerEntry) -> Self {
        Self {
            id: e.id.clone(),
            display_name: e.display_name.clone(),
            exe_path: e.exe_path.to_string_lossy().to_string(),
            default_flags: e.default_flags.clone(),
            flag_params: e.flag_params.iter().map(|fp| fp.into()).collect(),
            available: e.available,
        }
    }
}

/// 注册表列表响应。
#[derive(Debug, Serialize)]
pub struct ResamplerListPayload {
    pub ok: bool,
    pub entries: Vec<ResamplerEntryPayload>,
}

/// 通用操作响应。
#[derive(Debug, Serialize)]
pub struct ResamplerOpPayload {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<ResamplerEntryPayload>,
}

// ─── 辅助函数 ──────────────────────────────────────────────────────────────────

/// 将当前注册表持久化到 config 目录。
fn persist_registry(state: &AppState) {
    if let Some(config_dir) = state.config_dir.get() {
        let registry = state.resampler_registry.lock().unwrap_or_else(|e| e.into_inner());
        crate::config::save_resampler_registry(config_dir, &registry);
    }
}

// ─── Tauri Commands ────────────────────────────────────────────────────────────

/// 列出所有已注册的外部 resampler。
pub(crate) fn list_resamplers(state: State<'_, AppState>) -> ResamplerListPayload {
    let mut registry = state.resampler_registry.lock().unwrap_or_else(|e| e.into_inner());
    // 每次列出时刷新可用性
    registry.refresh_availability();
    let entries: Vec<ResamplerEntryPayload> = registry.list().iter().map(|e| (*e).into()).collect();
    ResamplerListPayload {
        ok: true,
        entries,
    }
}

/// 添加一个外部 resampler。
///
/// 参数：
///  - `display_name`: 显示名称（如 "Moresampler"）
///  - `exe_path`: 可执行文件绝对路径
///  - `default_flags`: 默认 flags 字符串（可为空）
pub(crate) fn add_resampler(
    state: State<'_, AppState>,
    display_name: String,
    exe_path: String,
    default_flags: Option<String>,
    flag_params: Option<Vec<FlagParamPayload>>,
) -> ResamplerOpPayload {
    let path = PathBuf::from(&exe_path);

    // 检查路径是否存在
    let available = path.exists();
    if !available {
        eprintln!(
            "[resampler_registry] 警告: 添加的 resampler 路径不存在: {}",
            exe_path
        );
    }

    let id = Uuid::new_v4().to_string();
    let entry = ResamplerEntry {
        id: id.clone(),
        display_name,
        exe_path: path,
        default_flags: default_flags.unwrap_or_default(),
        flag_params: flag_params
            .unwrap_or_default()
            .into_iter()
            .map(|fp| fp.into())
            .collect(),
        available,
    };

    let payload = ResamplerEntryPayload::from(&entry);

    {
        let mut registry = state.resampler_registry.lock().unwrap_or_else(|e| e.into_inner());
        registry.register(entry);
    }

    // 持久化
    persist_registry(&state);

    ResamplerOpPayload {
        ok: true,
        error: None,
        entry: Some(payload),
    }
}

/// 移除一个已注册的外部 resampler。
pub(crate) fn remove_resampler(
    state: State<'_, AppState>,
    id: String,
) -> ResamplerOpPayload {
    let removed = {
        let mut registry = state.resampler_registry.lock().unwrap_or_else(|e| e.into_inner());
        registry.remove(&id)
    };

    if !removed {
        return ResamplerOpPayload {
            ok: false,
            error: Some(format!("未找到 ID 为 '{}' 的 resampler", id)),
            entry: None,
        };
    }

    // 持久化
    persist_registry(&state);

    ResamplerOpPayload {
        ok: true,
        error: None,
        entry: None,
    }
}

/// 更新已注册 resampler 的信息（显示名称、路径、默认 flags、flag 参数列表）。
pub(crate) fn update_resampler(
    state: State<'_, AppState>,
    id: String,
    display_name: Option<String>,
    exe_path: Option<String>,
    default_flags: Option<String>,
    flag_params: Option<Vec<FlagParamPayload>>,
) -> ResamplerOpPayload {
    let result = {
        let mut registry = state.resampler_registry.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = registry.entries.get_mut(&id) {
            if let Some(name) = display_name {
                entry.display_name = name;
            }
            if let Some(path) = exe_path {
                entry.exe_path = PathBuf::from(&path);
                entry.available = entry.exe_path.exists();
            }
            if let Some(flags) = default_flags {
                entry.default_flags = flags;
            }
            if let Some(fps) = flag_params {
                entry.flag_params = fps.into_iter().map(|fp| fp.into()).collect();
            }
            Some(ResamplerEntryPayload::from(&*entry))
        } else {
            None
        }
    };

    match result {
        Some(payload) => {
            persist_registry(&state);
            ResamplerOpPayload {
                ok: true,
                error: None,
                entry: Some(payload),
            }
        }
        None => ResamplerOpPayload {
            ok: false,
            error: Some(format!("未找到 ID 为 '{}' 的 resampler", id)),
            entry: None,
        },
    }
}

/// 扫描指定目录下的 resampler 可执行文件，自动注册新发现的。
///
/// 扫描策略：遍历目录下所有 `.exe`（Windows）/ 可执行文件，
/// 跳过已注册的（按路径去重），将新发现的自动添加。
pub(crate) fn scan_resamplers(
    state: State<'_, AppState>,
    directory: String,
) -> ResamplerListPayload {
    let dir = PathBuf::from(&directory);
    if !dir.is_dir() {
        return ResamplerListPayload {
            ok: false,
            entries: vec![],
        };
    }

    let mut newly_added = vec![];

    // 收集当前已注册的 exe 路径集合（用于去重）
    let existing_paths: std::collections::HashSet<PathBuf> = {
        let registry = state.resampler_registry.lock().unwrap_or_else(|e| e.into_inner());
        registry.entries.values().map(|e| e.exe_path.clone()).collect()
    };

    // 遍历目录（仅一级，不递归）
    if let Ok(read_dir) = std::fs::read_dir(&dir) {
        for entry in read_dir.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Windows: 仅 .exe 文件
            #[cfg(target_os = "windows")]
            {
                if path.extension().and_then(|ext| ext.to_str()) != Some("exe") {
                    continue;
                }
            }

            // 非 Windows: 检查可执行权限
            #[cfg(not(target_os = "windows"))]
            {
                use std::os::unix::fs::PermissionsExt;
                let Ok(meta) = path.metadata() else { continue };
                if meta.permissions().mode() & 0o111 == 0 {
                    continue;
                }
            }

            // 去重
            if existing_paths.contains(&path) {
                continue;
            }

            // 从文件名推断显示名称
            let display_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown Resampler")
                .to_string();

            let id = Uuid::new_v4().to_string();
            let new_entry = ResamplerEntry {
                id,
                display_name,
                exe_path: path,
                default_flags: String::new(),
                flag_params: vec![],
                available: true,
            };
            newly_added.push(new_entry);
        }
    }

    // 注册新发现的
    if !newly_added.is_empty() {
        let mut registry = state.resampler_registry.lock().unwrap_or_else(|e| e.into_inner());
        for entry in &newly_added {
            registry.register(entry.clone());
        }
        drop(registry);
        persist_registry(&state);
    }

    let entries: Vec<ResamplerEntryPayload> = newly_added.iter().map(|e| e.into()).collect();
    ResamplerListPayload {
        ok: true,
        entries,
    }
}

/// 通过文件选择对话框让用户选择 resampler exe。
///
/// 返回选择的路径（或空）。前端可使用此路径调用 `add_resampler`。
pub(crate) async fn browse_resampler_exe() -> Option<String> {
    let result = rfd::AsyncFileDialog::new()
        .set_title("选择 Resampler 可执行文件")
        .add_filter("可执行文件", &["exe"])
        .add_filter("所有文件", &["*"])
        .pick_file()
        .await;

    result.map(|f| f.path().to_string_lossy().to_string())
}
