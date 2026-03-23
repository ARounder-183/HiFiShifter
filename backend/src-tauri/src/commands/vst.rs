//! VST 插件相关的 Tauri 命令实现。
//!
//! 提供插件扫描、加载/卸载、FX 链管理、编辑器 GUI 窗口等操作的后端逻辑。
//! 由 `commands.rs` 的 `#[tauri::command]` 门面层调用。

use crate::state::AppState;
use serde::{Deserialize, Serialize};

/// 前端可用的 VST 插件描述信息（序列化传输用）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VstPluginInfoPayload {
    pub uid: String,
    pub name: String,
    pub vendor: String,
    pub format: String,
    pub path: String,
    pub category: String,
    pub is_instrument: bool,
    pub num_inputs: u32,
    pub num_outputs: u32,
}

/// FX 链中单个插件的前端视图。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VstChainSlotPayload {
    pub index: usize,
    pub plugin_uid: String,
    pub plugin_name: String,
    pub plugin_path: String,
    pub format: String,
    pub bypassed: bool,
}

// ─── 插件扫描 ──────────────────────────────────────────────────────────────

/// 触发 VST 插件扫描并返回当前已知的扫描结果。
///
/// 扫描在后台线程中异步执行，本命令立即返回当前注册表中的结果。
/// 前端可在收到返回后定时轮询 `vst_list_plugins` 获取更新后的列表。
pub(super) fn vst_scan_plugins(state: &AppState) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let registry = state.vst_registry.clone();
        crate::vst_host::scanner::scan_plugins_async(registry);

        // 返回当前已有的结果（后台扫描完成后前端可再次查询）
        let descs = state.vst_registry.list_all();
        let plugins: Vec<VstPluginInfoPayload> = descs
            .into_iter()
            .map(|d| VstPluginInfoPayload {
                uid: d.uid,
                name: d.name,
                vendor: d.vendor,
                format: match d.format {
                    crate::vst_host::VstFormat::Vst2 => "vst2".to_string(),
                    crate::vst_host::VstFormat::Vst3 => "vst3".to_string(),
                },
                path: d.path.to_string_lossy().to_string(),
                category: d.category,
                is_instrument: d.is_instrument,
                num_inputs: d.num_inputs,
                num_outputs: d.num_outputs,
            })
            .collect();

        serde_json::json!({
            "ok": true,
            "scanning": state.vst_registry.scan_in_progress.load(std::sync::atomic::Ordering::Relaxed),
            "plugins": plugins,
        })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = state;
        serde_json::json!({
            "ok": false,
            "error": "VST feature is not enabled",
            "plugins": [],
        })
    }
}

/// 获取已扫描的插件列表（不触发新扫描）。
pub(super) fn vst_list_plugins(state: &AppState) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let descs = state.vst_registry.list_all();
        let plugins: Vec<VstPluginInfoPayload> = descs
            .into_iter()
            .map(|d| VstPluginInfoPayload {
                uid: d.uid,
                name: d.name,
                vendor: d.vendor,
                format: match d.format {
                    crate::vst_host::VstFormat::Vst2 => "vst2".to_string(),
                    crate::vst_host::VstFormat::Vst3 => "vst3".to_string(),
                },
                path: d.path.to_string_lossy().to_string(),
                category: d.category,
                is_instrument: d.is_instrument,
                num_inputs: d.num_inputs,
                num_outputs: d.num_outputs,
            })
            .collect();

        serde_json::json!({
            "ok": true,
            "plugins": plugins,
        })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = state;
        serde_json::json!({
            "ok": false,
            "error": "VST feature is not enabled",
            "plugins": [],
        })
    }
}

// ─── FX 链管理 ─────────────────────────────────────────────────────────────

/// 获取指定轨道的 VST FX 链。
pub(super) fn vst_get_track_chain(
    state: &AppState,
    track_id: &str,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let track = tl.tracks.iter().find(|t| t.id == track_id);

        let Some(track) = track else {
            return serde_json::json!({ "ok": false, "error": "Track not found", "slots": [] });
        };

        let slots: Vec<VstChainSlotPayload> = track
            .vst_chain
            .plugins
            .iter()
            .enumerate()
            .map(|(i, p)| VstChainSlotPayload {
                index: i,
                plugin_uid: p.plugin_uid.clone(),
                plugin_name: p.plugin_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                plugin_path: p.plugin_path.to_string_lossy().to_string(),
                format: match p.format {
                    crate::vst_host::VstFormat::Vst2 => "vst2".to_string(),
                    crate::vst_host::VstFormat::Vst3 => "vst3".to_string(),
                },
                bypassed: p.bypassed,
            })
            .collect();

        serde_json::json!({
            "ok": true,
            "trackId": track_id,
            "slots": slots,
        })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, track_id);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled", "slots": [] })
    }
}

/// 向轨道 FX 链添加插件。
pub(super) fn vst_add_to_chain(
    state: &AppState,
    track_id: &str,
    plugin_uid: &str,
    index: Option<usize>,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        // 从注册表查找插件描述符
        let desc = state.vst_registry.find_descriptor(plugin_uid);
        let Some(desc) = desc else {
            return serde_json::json!({ "ok": false, "error": "Plugin not found in registry" });
        };

        let plugin_state = crate::vst_host::VstPluginState {
            plugin_uid: desc.uid.clone(),
            plugin_path: desc.path.clone(),
            format: desc.format,
            chunk_data: None,
            bypassed: false,
            params: std::collections::HashMap::new(),
        };

        let snapshot = {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            let Some(track) = tl.tracks.iter_mut().find(|t| t.id == track_id) else {
                return serde_json::json!({ "ok": false, "error": "Track not found" });
            };

            let idx = index.unwrap_or(track.vst_chain.plugins.len());
            let idx = idx.min(track.vst_chain.plugins.len());
            track.vst_chain.plugins.insert(idx, plugin_state);

            let snap = tl.clone();
            state.checkpoint_timeline(&snap);
            snap
        };

        state.audio_engine.update_timeline(snapshot);

        serde_json::json!({ "ok": true })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, track_id, plugin_uid, index);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled" })
    }
}

/// 从轨道 FX 链移除插件。
pub(super) fn vst_remove_from_chain(
    state: &AppState,
    track_id: &str,
    index: usize,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let snapshot = {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            let Some(track) = tl.tracks.iter_mut().find(|t| t.id == track_id) else {
                return serde_json::json!({ "ok": false, "error": "Track not found" });
            };

            if index >= track.vst_chain.plugins.len() {
                return serde_json::json!({ "ok": false, "error": "Index out of range" });
            }
            track.vst_chain.plugins.remove(index);

            let snap = tl.clone();
            state.checkpoint_timeline(&snap);
            snap
        };

        state.audio_engine.update_timeline(snapshot);

        serde_json::json!({ "ok": true })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, track_id, index);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled" })
    }
}

/// 设置 FX 链中某个插件的 bypass 状态。
pub(super) fn vst_set_bypass(
    state: &AppState,
    track_id: &str,
    index: usize,
    bypassed: bool,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let snapshot = {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            let Some(track) = tl.tracks.iter_mut().find(|t| t.id == track_id) else {
                return serde_json::json!({ "ok": false, "error": "Track not found" });
            };

            if let Some(plugin) = track.vst_chain.plugins.get_mut(index) {
                plugin.bypassed = bypassed;
            } else {
                return serde_json::json!({ "ok": false, "error": "Index out of range" });
            }

            let snap = tl.clone();
            state.checkpoint_timeline(&snap);
            snap
        };

        state.audio_engine.update_timeline(snapshot);

        serde_json::json!({ "ok": true })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, track_id, index, bypassed);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled" })
    }
}

/// 在 FX 链内重新排序插件。
pub(super) fn vst_reorder_chain(
    state: &AppState,
    track_id: &str,
    from_index: usize,
    to_index: usize,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let snapshot = {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            let Some(track) = tl.tracks.iter_mut().find(|t| t.id == track_id) else {
                return serde_json::json!({ "ok": false, "error": "Track not found" });
            };

            let len = track.vst_chain.plugins.len();
            if from_index >= len || to_index >= len {
                return serde_json::json!({ "ok": false, "error": "Index out of range" });
            }

            let item = track.vst_chain.plugins.remove(from_index);
            track.vst_chain.plugins.insert(to_index, item);

            let snap = tl.clone();
            state.checkpoint_timeline(&snap);
            snap
        };

        state.audio_engine.update_timeline(snapshot);

        serde_json::json!({ "ok": true })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, track_id, from_index, to_index);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled" })
    }
}

// ─── 编辑器 GUI ─────────────────────────────────────────────────────────────

/// 打开 VST 插件编辑器窗口。
///
/// 从轨道 FX 链配置中获取目标插件信息，在 `VstPluginRegistry.instances` 中
/// 查找或新建插件实例，然后调用 `gui::open_editor_window()` 创建原生窗口。
/// 编辑器窗口运行在独立线程中，关闭窗口时自动释放资源。
pub(super) fn vst_open_editor(
    state: &AppState,
    track_id: &str,
    index: usize,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        use crate::vst_host::{gui, plugin_host};

        // 从 timeline 获取目标插件的状态信息
        let plugin_state = {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            let Some(track) = tl.tracks.iter().find(|t| t.id == track_id) else {
                return serde_json::json!({ "ok": false, "error": "Track not found" });
            };
            let Some(ps) = track.vst_chain.plugins.get(index) else {
                return serde_json::json!({ "ok": false, "error": "Plugin index out of range" });
            };
            ps.clone()
        };

        // 生成实例 ID（track_id + 插件索引）
        let instance_id = format!("{}:{}:{}", track_id, index, plugin_state.plugin_uid);

        // 从注册表获取或创建插件实例
        let instance = {
            let mut instances = state
                .vst_registry
                .instances
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            if let Some(existing) = instances.get(&instance_id) {
                existing.clone()
            } else {
                // 需要加载新实例
                let path = &plugin_state.plugin_path;
                if !path.exists() {
                    return serde_json::json!({
                        "ok": false,
                        "error": format!("Plugin file not found: {}", path.display())
                    });
                }

                let load_result = match plugin_state.format {
                    crate::vst_host::VstFormat::Vst2 => {
                        plugin_host::load_vst2(path, 44100.0, 512)
                    }
                    crate::vst_host::VstFormat::Vst3 => {
                        plugin_host::load_vst3(path, 44100.0, 512)
                    }
                };

                let inst = match load_result {
                    Ok(i) => i,
                    Err(e) => {
                        return serde_json::json!({
                            "ok": false,
                            "error": format!("Failed to load plugin: {}", e)
                        });
                    }
                };

                // 恢复 chunk 数据
                if let Some(ref chunk) = plugin_state.chunk_data {
                    let mut locked = inst.lock().unwrap_or_else(|e| e.into_inner());
                    if let Err(e) = locked.set_chunk(chunk) {
                        eprintln!(
                            "[vst::open_editor] Failed to restore chunk: {}",
                            e
                        );
                    }
                }

                instances.insert(instance_id.clone(), inst.clone());
                inst
            }
        };

        // 获取插件名称作为窗口标题
        let plugin_name = {
            let inst = instance.lock().unwrap_or_else(|e| e.into_inner());
            inst.name.clone()
        };
        let window_title = format!("{} - Track {}", plugin_name, track_id);

        // 创建编辑器窗口
        match gui::open_editor_window(&instance, &window_title) {
            Ok(window) => {
                eprintln!(
                    "[vst::open_editor] Editor opened for {}[{}]: {}",
                    track_id, index, plugin_name
                );
                serde_json::json!({
                    "ok": true,
                    "pluginName": plugin_name,
                    "width": window.width,
                    "height": window.height,
                })
            }
            Err(e) => {
                serde_json::json!({
                    "ok": false,
                    "error": format!("Failed to open editor window: {}", e)
                })
            }
        }
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, track_id, index);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled" })
    }
}

/// 添加自定义 VST 扫描路径。
///
/// 添加到内存注册表并持久化到 `config_dir/vst_scan_paths.json`。
/// 重启应用后自动恢复。
pub(super) fn vst_add_scan_path(
    state: &AppState,
    path: &str,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let path_buf = std::path::PathBuf::from(path);
        if !path_buf.exists() || !path_buf.is_dir() {
            return serde_json::json!({ "ok": false, "error": "Directory does not exist" });
        }

        let mut paths = state
            .vst_registry
            .custom_scan_paths
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if !paths.contains(&path_buf) {
            paths.push(path_buf);
        }

        // 持久化到配置文件
        if let Some(config_dir) = state.config_dir.get() {
            crate::vst_host::scanner::save_custom_scan_paths(config_dir, &paths);
        }

        serde_json::json!({ "ok": true })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, path);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled" })
    }
}

/// 获取当前自定义 VST 扫描路径列表。
pub(super) fn vst_list_scan_paths(state: &AppState) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let paths = state
            .vst_registry
            .custom_scan_paths
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let str_paths: Vec<String> = paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
        serde_json::json!({ "ok": true, "paths": str_paths })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = state;
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled", "paths": [] })
    }
}

/// 移除自定义 VST 扫描路径。
///
/// 从内存注册表中移除并持久化到 `config_dir/vst_scan_paths.json`。
pub(super) fn vst_remove_scan_path(
    state: &AppState,
    path: &str,
) -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        let path_buf = std::path::PathBuf::from(path);
        let mut paths = state
            .vst_registry
            .custom_scan_paths
            .write()
            .unwrap_or_else(|e| e.into_inner());
        paths.retain(|p| p != &path_buf);

        // 持久化到配置文件
        if let Some(config_dir) = state.config_dir.get() {
            crate::vst_host::scanner::save_custom_scan_paths(config_dir, &paths);
        }

        serde_json::json!({ "ok": true })
    }

    #[cfg(not(feature = "vst"))]
    {
        let _ = (state, path);
        serde_json::json!({ "ok": false, "error": "VST feature is not enabled" })
    }
}

/// 获取 VST 功能是否可用。
pub(super) fn vst_get_status() -> serde_json::Value {
    #[cfg(feature = "vst")]
    {
        serde_json::json!({
            "ok": true,
            "available": true,
            "formats": ["vst2", "vst3"],
        })
    }

    #[cfg(not(feature = "vst"))]
    {
        serde_json::json!({
            "ok": true,
            "available": false,
            "formats": [],
        })
    }
}
