//! 诊断支持命令：日志文件夹访问、诊断包导出、前端错误回传。
//!
//! - `open_log_folder`：Help 菜单「打开日志文件夹」。
//! - `pick_diagnostics_output_path` / `export_diagnostics`：Help 菜单
//!   「导出诊断信息」。对话框在同步命令（主线程）里弹出，打包与基准测试
//!   在阻塞线程池执行，避免冻结 UI。
//! - `log_frontend_error`：前端把 invoke 失败与全局异常回传到后端日志，
//!   使前后端日志落在同一文件、同一时间轴上。

use std::io::Write as _;
use std::path::{Path, PathBuf};

use tauri::{Manager, State};
use tauri_plugin_opener::OpenerExt;

use crate::logging;
use crate::state::AppState;

/// 单个日志文件纳入诊断包的大小上限（防止异常膨胀的日志拖垮导出）。
const DIAGNOSTICS_LOG_SIZE_CAP: u64 = 32 * 1024 * 1024;
/// 前端回传的错误详情长度上限。
const FRONTEND_DETAIL_MAX_CHARS: usize = 4000;

// ===================== open_log_folder =====================

/// 打开日志所在文件夹。文件日志被显式关闭（`--log-file=-`）时回退到
/// Tauri 的 `app_log_dir`，保证菜单入口始终可用。
pub(super) fn open_log_folder(app: tauri::AppHandle) -> serde_json::Value {
    let dir: Option<PathBuf> = logging::log_dir()
        .map(Path::to_path_buf)
        .or_else(|| app.path().app_log_dir().ok());
    let Some(dir) = dir else {
        return serde_json::json!({ "ok": false, "error": "log directory unavailable" });
    };

    if let Err(e) = std::fs::create_dir_all(&dir) {
        return serde_json::json!({ "ok": false, "error": format!("create log dir failed: {e}") });
    }

    match app.opener().open_path(dir.to_string_lossy(), None::<&str>) {
        Ok(()) => serde_json::json!({ "ok": true, "path": dir.to_string_lossy() }),
        Err(e) => serde_json::json!({ "ok": false, "error": format!("open folder failed: {e}") }),
    }
}

// ===================== export_diagnostics =====================

/// 「导出诊断信息」第一步：选择输出路径（同步命令，运行在主线程，rfd 才可用）。
pub(super) fn pick_diagnostics_output_path() -> serde_json::Value {
    let default_name = format!(
        "HiFiShifter-diagnostics-{}.zip",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );
    let picked = rfd::FileDialog::new()
        .set_title("Export diagnostics")
        .set_file_name(&default_name)
        .add_filter("Zip archive", &["zip"])
        .save_file();
    match picked {
        Some(path) => serde_json::json!({ "ok": true, "path": path.to_string_lossy() }),
        None => serde_json::json!({ "ok": false, "canceled": true }),
    }
}

/// 「导出诊断信息」第二步：打包系统信息 + 全部日志文件，随后运行推理设备
/// 基准测试并把结果追加进 zip。
///
/// 顺序刻意安排为「先落盘基础包，再跑基准测试」：基准测试会初始化 ORT
/// 会话，在 GPU/驱动异常的环境下存在硬崩溃可能 —— 即使崩溃，包含日志的
/// 基础包也已经在磁盘上，用户仍可提交。
pub(super) fn export_diagnostics(
    state: State<'_, AppState>,
    output_path: String,
) -> serde_json::Value {
    let out = PathBuf::from(&output_path);
    log::info!("[diagnostics] exporting diagnostics package to {}", out.display());

    let result = write_base_zip(&out, &build_system_info(&state)).and_then(|()| append_benchmark(&out));

    match result {
        Ok(()) => {
            // 定位失败不影响导出结果本身。
            if let Err(e) = reveal_in_file_manager(&out) {
                log::warn!("[diagnostics] could not reveal exported file: {e}");
            }
            log::info!("[diagnostics] export finished: {}", out.display());
            serde_json::json!({ "ok": true, "path": output_path })
        }
        Err(error) => {
            log::error!("[diagnostics] export failed: {error}");
            serde_json::json!({ "ok": false, "error": error })
        }
    }
}

/// 系统信息 + 运行时状态（刻意不含 timeline 等工程内容，避免把用户工程
/// 数据带进诊断包）。
fn build_system_info(state: &State<'_, AppState>) -> serde_json::Value {
    let rt = state.runtime_info();
    serde_json::json!({
        "app": {
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
        },
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "generatedAt": chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
        "runtime": {
            "device": rt.device,
            "modelLoaded": rt.model_loaded,
            "audioLoaded": rt.audio_loaded,
            "isPlaying": rt.is_playing,
            "playbackTarget": rt.playback_target,
        },
        "gpuDevices": super::onnx_status::get_gpu_devices(),
        "dmlAdapters": super::onnx_status::get_dml_adapters(),
    })
}

fn write_base_zip(out: &Path, system_info: &serde_json::Value) -> Result<(), String> {
    let file = std::fs::File::create(out).map_err(|e| format!("create zip failed: {e}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("system_info.json", options.clone())
        .map_err(|e| format!("zip add system_info failed: {e}"))?;
    let info = serde_json::to_string_pretty(system_info)
        .map_err(|e| format!("serialize system_info failed: {e}"))?;
    zip.write_all(info.as_bytes())
        .map_err(|e| format!("write system_info failed: {e}"))?;

    for path in logging::log_files() {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if size > DIAGNOSTICS_LOG_SIZE_CAP {
            log::warn!("[diagnostics] skipping oversized log file {name} ({size} bytes)");
            continue;
        }
        match std::fs::read(&path) {
            Ok(bytes) => {
                zip.start_file(format!("logs/{name}"), options.clone())
                    .map_err(|e| format!("zip add {name} failed: {e}"))?;
                zip.write_all(&bytes)
                    .map_err(|e| format!("write {name} failed: {e}"))?;
            }
            Err(e) => log::warn!("[diagnostics] failed to read log file {name}: {e}"),
        }
    }

    zip.finish().map_err(|e| format!("finish zip failed: {e}"))?;
    Ok(())
}

/// 运行推理设备基准测试并把结果追加进 zip；失败也写入错误信息（这本身就是
/// 有价值的诊断数据）。
fn append_benchmark(zip_path: &Path) -> Result<(), String> {
    log::info!("[diagnostics] running inference-device benchmark (this may take a while)...");
    let benchmark = match super::onnx_status::run_vocoder_benchmark() {
        Ok(results) => serde_json::to_string_pretty(&results)
            .map_err(|e| format!("serialize benchmark failed: {e}"))?,
        Err(error) => serde_json::json!({ "error": error }).to_string(),
    };

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(zip_path)
        .map_err(|e| format!("reopen zip failed: {e}"))?;
    let mut zip = zip::ZipWriter::new_append(file)
        .map_err(|e| format!("append zip failed: {e}"))?;
    let options =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("benchmark.json", options.clone())
        .map_err(|e| format!("zip add benchmark failed: {e}"))?;
    zip.write_all(benchmark.as_bytes())
        .map_err(|e| format!("write benchmark failed: {e}"))?;
    zip.finish().map_err(|e| format!("finish zip failed: {e}"))?;
    Ok(())
}

fn reveal_in_file_manager(path: &Path) -> Result<(), String> {
    // 在系统文件管理器中定位导出的 zip；失败不影响导出结果本身。
    tauri_plugin_opener::reveal_item_in_dir(path).map_err(|e| format!("reveal failed: {e}"))
}

// ===================== log_frontend_error =====================

/// 前端错误回传入口：写入统一日志文件，便于和后端日志按时间轴对齐。
pub(super) fn log_frontend_error(message: String, detail: Option<String>) -> serde_json::Value {
    let detail = detail
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty());
    match detail {
        Some(detail) => {
            let detail = if detail.chars().count() > FRONTEND_DETAIL_MAX_CHARS {
                let mut truncated: String =
                    detail.chars().take(FRONTEND_DETAIL_MAX_CHARS).collect();
                truncated.push_str("…(truncated)");
                truncated
            } else {
                detail
            };
            log::error!("[frontend] {message} | detail: {detail}");
        }
        None => log::error!("[frontend] {message}"),
    }
    serde_json::json!({ "ok": true })
}
