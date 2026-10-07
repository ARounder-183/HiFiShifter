// 命令层共用工具函数
use crate::state::AppState;
use serde::Serialize;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PlaybackRenderingStateEvent {
    pub(crate) active: bool,
    pub(crate) progress: Option<f64>,
    pub(crate) target: Option<String>,
    /// 本轮渲染 pass 的标识。
    ///
    /// 【为什么需要它】进度在**同一轮内**单调不减（见 `renderer::progress` 的
    /// 单调闸门），但**跨轮允许重新起始** —— 打开大工程时音高分析逐批解锁，
    /// 每批各跑一轮，每轮都从 0% 涨到 100% 才是正确反馈。前端只看到
    /// `active`/`progress` 时无法区分"新一轮开始"与"同一轮回退"，
    /// 于是要么误杀真实的重新起始，要么把回退显示给用户。带上 pass 序号后，
    /// 前端按"pass 变了 ⇒ 允许重新起始"判定，不需要任何超时启发式。
    pub(crate) pass: Option<u64>,
}

pub(crate) fn guard_json_command(
    name: &str,
    f: impl FnOnce() -> serde_json::Value,
) -> serde_json::Value {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            log::error!("command panicked: {name}");
            serde_json::json!({"ok": false, "error": format!("panic in command: {name}")})
        }
    }
}

/// 波形命令的 panic 守卫。
///
/// 【为什么保留而未删除】它是 `guard_command` 的同款守卫，只是返回值类型不同
/// （`WaveformPeaksSegmentPayload` 而不是 `serde_json::Value`）。目前**没有调用点**
/// —— 也就是说波形命令实际上没有这层遏制。这是"写了安全网但没接线"，删除它等于
/// 把这个事实埋掉；保留 + 本注解把它留在明面上，等一次专门的判断（接线或明确删除）。
#[allow(
    dead_code,
    reason = "已实现但尚未接线的 panic 守卫；保留以暴露该缺口，见上方说明"
)]
pub(crate) fn guard_waveform_command(
    name: &str,
    f: impl FnOnce() -> super::waveform::WaveformPeaksSegmentPayload,
) -> super::waveform::WaveformPeaksSegmentPayload {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            log::error!("command panicked: {name}");
            super::waveform::WaveformPeaksSegmentPayload {
                ok: false,
                min: vec![],
                max: vec![],
            }
        }
    }
}

pub(crate) fn ok_bool() -> serde_json::Value {
    serde_json::json!({ "ok": true })
}

/// 在系统文件管理器中定位一批路径：存在的文件被**多选高亮**，全是目录时打开
/// 第一个目录。
///
/// 【为什么走 Rust 而不是前端 `@tauri-apps/plugin-opener`】`opener:default` 实际
/// 只授予 `open-url` / `reveal-item-in-dir` / `default-urls`，**不含 `open_path`**；
/// 而 Rust 侧的 `OpenerExt` 调用不过 ACL。本仓既有 4 处同款做法
/// （`diagnostics.rs` 的日志目录与 zip、`cache.rs` 的缓存目录与导出产物）。
///
/// 【为什么先过滤不存在的项】`reveal_items_in_dir` 内部会 `canonicalize`，
/// 任何一个缺失路径都会让整批 reveal 失败 —— 分轨导出时个别目标被跳过 / 写入失败
/// 就会连累其余文件。
pub(crate) fn reveal_paths_in_file_manager(
    app: &tauri::AppHandle,
    paths: Vec<String>,
    empty_error: &str,
) -> serde_json::Value {
    use tauri_plugin_opener::OpenerExt;

    let mut files: Vec<PathBuf> = Vec::new();
    let mut dirs: Vec<PathBuf> = Vec::new();
    for raw in paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            dirs.push(path);
        }
    }

    if !files.is_empty() {
        return match app.opener().reveal_items_in_dir(files.iter()) {
            Ok(()) => serde_json::json!({ "ok": true, "count": files.len() }),
            Err(e) => {
                // 多选失败（个别文件管理器不支持）→ 至少打开首个文件的所在目录。
                if let Some(parent) = files[0].parent() {
                    let _ = app
                        .opener()
                        .open_path(parent.to_string_lossy(), None::<&str>);
                }
                serde_json::json!({
                    "ok": false,
                    "error": format!("reveal files failed: {e}"),
                })
            }
        };
    }

    if let Some(dir) = dirs.first() {
        return match app.opener().open_path(dir.to_string_lossy(), None::<&str>) {
            Ok(()) => serde_json::json!({ "ok": true, "path": dir.to_string_lossy() }),
            Err(e) => serde_json::json!({
                "ok": false,
                "error": format!("open folder failed: {e}"),
            }),
        };
    }

    serde_json::json!({ "ok": false, "error": empty_error })
}

/// 用系统默认程序打开一个路径（文件或目录）。
///
/// 与 [`reveal_paths_in_file_manager`] 同一 ACL 理由：必须走 Rust 侧 `OpenerExt`。
pub(crate) fn open_path_with_default_app(app: &tauri::AppHandle, path: &str) -> serde_json::Value {
    use tauri_plugin_opener::OpenerExt;

    let trimmed = path.trim();
    if trimmed.is_empty() {
        return serde_json::json!({ "ok": false, "error": "empty path" });
    }
    match app.opener().open_path(trimmed, None::<&str>) {
        Ok(()) => serde_json::json!({ "ok": true }),
        Err(e) => serde_json::json!({
            "ok": false,
            "error": format!("open path failed: {e}"),
        }),
    }
}

pub(crate) fn ensure_temp_dir() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("hifishifter");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub(crate) fn new_temp_wav_path(prefix: &str) -> Result<PathBuf, String> {
    let dir = ensure_temp_dir().map_err(|e| e.to_string())?;
    Ok(dir.join(format!("{}_{}.wav", prefix, Uuid::new_v4().simple())))
}

pub(crate) fn render_timeline_to_wav(
    state: &AppState,
    output_path: &Path,
    start_sec: f64,
    end_sec: Option<f64>,
) -> Result<crate::mixdown::MixdownResult, String> {
    let timeline = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    crate::mixdown::render_mixdown_to_file(
        &timeline,
        output_path,
        crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec,
            end_sec,
            stretch: crate::time_stretch::resolved_external_stretch_algorithm(),
            apply_pitch_edit: true,
            // 临时渲染固定 32-bit float WAV（内部用途，不受导出设置影响）。
            output: crate::encode::OutputSpec::wav_32f(),
            quality_preset: crate::mixdown::QualityPreset::Export,
            cancel_flag: None,
            progress: None,
            cache_stats: None,
        },
    )
}
