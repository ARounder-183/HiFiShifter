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
/// 前端回传的错误正文长度上限。
///
/// 【为什么正文也要上限】`message` 由渲染进程给出、长度完全不受控：一个循环里抛出
/// 的长错误足以把日志写成 GB 级（诊断包随后被同一个上限拖大）。详情早就有上限，
/// 正文此前没有。
const FRONTEND_MESSAGE_MAX_CHARS: usize = 512;

// ===================== open_log_folder =====================

/// 打开日志所在文件夹。
///
/// 【目录来源必须与 logger 同源】优先用 `logging::log_dir()`（本次会话实际写入
/// 的那个目录），未启用文件日志时退到 `logfile::log_dir()`（同一套路径解析）。
///
/// **不用 Tauri 的 `app_log_dir` 作为首选兜底**：那个目录本应用从来不写 ——
/// 桌面是 `%LOCALAPPDATA%\<identifier>\logs`、Android 是 `<sandbox>/logs`，
/// 而真正的日志在 `<sandbox>/files/HiFiShifter/logs`。把用户指过去，他会打开
/// 一个空文件夹，然后合理地认为"日志丢了"。（它只在两处都解析不出时兜底，
/// 那种情况下应用连配置目录都没有，是更严重的环境问题。）
pub(super) fn open_log_folder(app: tauri::AppHandle) -> serde_json::Value {
    let dir: Option<PathBuf> = logging::log_dir()
        .map(Path::to_path_buf)
        .or_else(hifishifter_kernel::logfile::log_dir)
        .or_else(|| app.path().app_log_dir().ok());
    let Some(dir) = dir else {
        return serde_json::json!({ "ok": false, "error": "log directory unavailable" });
    };

    if let Err(e) = std::fs::create_dir_all(&dir) {
        return serde_json::json!({ "ok": false, "error": format!("create log dir failed: {e}") });
    }

    /*
     * ── 移动端：没有"用文件管理器打开这个目录"这个动作 ────────────────────
     *
     * 用户在这里真正需要的是**路径**（复制给我们），而不是一个打不开的目录。
     * Android 上 `open_path` 对应用私有目录只会失败：`files/` 是沙箱内路径，
     * 系统文件应用看不到它，而 Android 11 起连 `Android/data` 都被隐藏。
     * 直接尝试打开的代价是用户拿到一句"open folder failed"，而**路径本身
     * 没有出现在任何地方** —— 那正是他最需要的信息。
     *
     * 想真正把日志带出去，走「导出诊断信息」：它写进 `Android/media/<pkg>`，
     * 文件管理器与 USB 都可见（见 `platform::paths::export_dir`）。
     */
    if crate::platform::is_mobile() {
        let file = logging::log_file().map(|p| p.to_string_lossy().into_owned());
        // 【为什么要报 fileEnabled】目录存在不等于里面有日志：用户可能把
        // 「日志详细程度」选成了「无」，此时 `init_logging` 直接返回、一个字节
        // 都不写。只给路径会让他在空文件夹里找半天，而真实原因是"没开日志"。
        let enabled = hifishifter_kernel::log_policy::policy().file_enabled;
        log::info!(
            "[diagnostics] log folder requested on mobile (not openable): {} (file_enabled={enabled})",
            dir.display()
        );
        return serde_json::json!({
            "ok": true,
            "path": dir.to_string_lossy(),
            "file": file,
            // `opened: false` = "没有打开文件管理器，只是把位置告诉你"。
            // 前端据此把路径显示成可复制的文本（见 `logFolderNotice`）。
            "opened": false,
            "fileEnabled": enabled,
            "loggingHealthy": hifishifter_kernel::log_policy::logging_healthy(),
            "lastError": hifishifter_kernel::log_policy::last_error(),
        });
    }

    match app.opener().open_path(dir.to_string_lossy(), None::<&str>) {
        Ok(()) => serde_json::json!({ "ok": true, "path": dir.to_string_lossy() }),
        Err(e) => serde_json::json!({ "ok": false, "error": format!("open folder failed: {e}") }),
    }
}

// ===================== export_diagnostics =====================

/// 「导出诊断信息」第一步：选择输出路径（同步命令，运行在主线程，rfd 才可用）。
///
/// 对话框标题由前端按当前语言传入（与 `open_audio_dialog_for_source` 的
/// `dialog_title` 同一约定）；空串 / 缺省时回退系统默认标题。
pub(super) fn pick_diagnostics_output_path(
    app: &tauri::AppHandle,
    dialog_title: Option<String>,
) -> serde_json::Value {
    let default_name = format!(
        "HiFiShifter-diagnostics-{}.zip",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );
    // 类型名与扩展名来自内核词表（hifishifter_kernel::file_types），与插件侧
    // 同一张表 —— 两处各写一份时，改了一边就会让同一个导出在 App 与插件里
    // 出现两种下拉框。
    let file_type = &hifishifter_kernel::file_types::DIAGNOSTICS_ZIP;
    let mut dialog = crate::platform::dialog::FileDialog::new()
        .set_file_name(&default_name)
        .add_filter(file_type.name(), &file_type.extension_refs());
    let title = dialog_title.as_deref().map(str::trim).unwrap_or("");
    if !title.is_empty() {
        dialog = dialog.set_title(title);
    }
    let picked = dialog.save_file(app);
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
    frontend_settings: Option<serde_json::Value>,
) -> serde_json::Value {
    let out = PathBuf::from(&output_path);
    log::info!(
        "[diagnostics] exporting diagnostics package to {}",
        out.display()
    );

    let _ = crate::logging::flush_logs();
    let settings = build_settings_info(&state, frontend_settings);
    let result = write_base_zip(&out, &build_system_info(&state), &settings);
    // 【基准失败不得让整包导出失败】旧实现把 `append_benchmark` 串在 `and_then` 上：
    // 基准一报错（ORT 会话建不起来、模型缺失、驱动异常），整个 `export_diagnostics`
    // 返回 `ok:false`，**用户连日志包都拿不到** —— 而日志恰恰是排查那个错误最需要
    // 的东西。现在基准单独失败，错误写进包里（`benchmark.json` 的 `error` 字段），
    // 基础包照常返回成功。
    if result.is_ok() {
        if let Err(error) = append_benchmark(&out) {
            log::warn!("[diagnostics] benchmark failed (bundle kept): {error}");
        }
    }

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
        // 真实渲染耗时（最近 256 次后台渲染的 p50/p95）：零成本、贴近体感，
        // 比合成基准更能回答「这台机器慢不慢」。
        "render_stats": super::render_stats::snapshot(),
        "app": {
            "name": env!("CARGO_PKG_NAME"),
            "version": crate::build_info::version(),
        },
        "git": {
            "commit": crate::build_info::commit_full(),
            "commitShort": crate::build_info::commit_short(),
            "dirty": crate::build_info::dirty(),
            "repoUrl": crate::build_info::repo_url(),
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
        "models": build_models_info(state),
        "dmlAdapters": super::onnx_status::get_dml_adapters(),
        "memory": build_memory_info(),
        "logVerbosity": hifishifter_kernel::log_policy::current().as_str(),
        "sessionId": hifishifter_kernel::log_policy::session_id(),
    })
}

/// 模型库的现场：目录、APK、已登记项、实际文件。
///
/// 【为什么必须进诊断包】"音高画了没反应"最常见的根因是声码器权重不在设备上
/// （没进包、物化失败、被系统的"清除存储"删掉）。此前诊断包只有
/// `runtime.modelLoaded` 一个布尔 —— 它只能说明"没加载"，说不出**为什么**。
/// 这里把"目录在哪、从哪个 APK 解出来的、登记了哪些、实际有哪些文件多大"
/// 一次列清：这三条互相独立，缺哪条都能一眼定位。
fn build_models_info(state: &State<'_, AppState>) -> serde_json::Value {
    use hifishifter_kernel as kernel;

    let (root, root_error) = match state.app_handle.get() {
        Some(app) => match crate::platform::resources::models_root(app) {
            Ok(dir) => (Some(dir), None),
            Err(error) => (None, Some(error)),
        },
        None => (None, Some("app handle unavailable".to_string())),
    };

    let mut files = Vec::new();
    if let Some(dir) = &root {
        collect_model_files(dir, dir, 0, &mut files);
    }

    serde_json::json!({
        "root": root.as_ref().map(|p| p.display().to_string()),
        "rootError": root_error,
        "apk": crate::platform::resources::apk_path_for_diagnostics()
            .map(|p| p.display().to_string()),
        "registered": {
            "fcpe": kernel::fcpe_onnx_path().map(|p| p.display().to_string()),
            "nsfHifigan": kernel::nsf_hifigan_model_dir().map(|p| p.display().to_string()),
            "hnsep": kernel::hnsep_model_dir().map(|p| p.display().to_string()),
            "game": kernel::game_model_dir().map(|p| p.display().to_string()),
        },
        "files": files,
    })
}

/// 递归列出模型目录（深度 ≤3、最多 64 条）：相对路径 + 字节数。
///
/// 【为什么有上限】模型目录正常只有十几个文件；上限防的是用户把整个模型树
/// 塞进 `models-user/`（那会把诊断包撑到几百 MB，而它要能贴进 issue）。
fn collect_model_files(root: &Path, dir: &Path, depth: usize, out: &mut Vec<serde_json::Value>) {
    const MAX_ENTRIES: usize = 64;
    if depth > 3 || out.len() >= MAX_ENTRIES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= MAX_ENTRIES {
            break;
        }
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_dir() {
            collect_model_files(root, &path, depth + 1, out);
        } else if meta.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            out.push(serde_json::json!({ "path": relative, "bytes": meta.len() }));
        }
    }
}

/// 内存相关诊断：进程级缓存的驻留规模。
///
/// 【为什么放进诊断包】"导入长音频后内存不降"这类问题只有现场数据能定位 ——
/// 让用户导出诊断包时顺手带上各缓存的条目数与字节数，比事后试图复现便宜得多。
/// 关注点：`clipPitchCache.totalBytes` 是否远大于 `largestEntryBytes`（条目泄漏），
/// 以及 `inflight` 在工程切换后是否归零（分析线程未退出）。
fn build_memory_info() -> serde_json::Value {
    let pitch = crate::pitch_clip::pitch_cache_memory_stats();
    serde_json::json!({
        "clipPitchCache": {
            "entries": pitch.entries,
            "totalBytes": pitch.total_bytes,
            "largestEntryBytes": pitch.largest_entry_bytes,
            "inflight": pitch.inflight,
            "entryLimit": crate::pitch_clip::clip_pitch_cache_entry_limit(),
        },
    })
}

/// 用户设置快照：后端持久化配置（UI / 导出 / 自动备份 / 录音 / 窗口）+
/// 前端 `localStorage` 设置（外观 / 自定义主题 / 快捷键 / 布局偏好等，由面板侧
/// 收集后传入）。刻意不含最近工程列表（`recent`）等工程数据，与
/// `build_system_info` 的隐私取向保持一致。
fn build_settings_info(
    state: &State<'_, AppState>,
    frontend_settings: Option<serde_json::Value>,
) -> serde_json::Value {
    let backend = match state.config_dir.get() {
        Some(dir) => serde_json::json!({
            "ui": crate::config::load_ui_settings(dir),
            "export": crate::config::load_export_settings(dir),
            "autoBackup": crate::config::load_auto_backup_settings(dir),
            "recording": crate::config::load_recording_settings(dir),
            "window": crate::config::load_window_state(dir),
        }),
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "backend": backend,
        "frontend": frontend_settings.unwrap_or(serde_json::Value::Null),
    })
}

fn write_base_zip(
    out: &Path,
    system_info: &serde_json::Value,
    settings: &serde_json::Value,
) -> Result<(), String> {
    let file = std::fs::File::create(out).map_err(|e| format!("create zip failed: {e}"))?;
    let mut zip = zip::ZipWriter::new(file);

    zip.start_file("system_info.json", crate::zip_util::options_now())
        .map_err(|e| format!("zip add system_info failed: {e}"))?;
    let info = serde_json::to_string_pretty(system_info)
        .map_err(|e| format!("serialize system_info failed: {e}"))?;
    zip.write_all(info.as_bytes())
        .map_err(|e| format!("write system_info failed: {e}"))?;

    // 生效的日志策略与变更历史：排障时能判断"日志里为什么没有某类行"。
    zip.start_file("log_policy.json", crate::zip_util::options_now())
        .map_err(|e| format!("zip add log_policy failed: {e}"))?;
    let policy_json =
        serde_json::to_string_pretty(&hifishifter_kernel::log_policy::diagnostics_json())
            .map_err(|e| format!("serialize log_policy failed: {e}"))?;
    zip.write_all(policy_json.as_bytes())
        .map_err(|e| format!("write log_policy failed: {e}"))?;

    zip.start_file("policy_changes.jsonl", crate::zip_util::options_now())
        .map_err(|e| format!("zip add policy_changes failed: {e}"))?;
    zip.write_all(hifishifter_kernel::log_policy::diagnostics_changes_jsonl().as_bytes())
        .map_err(|e| format!("write policy_changes failed: {e}"))?;

    zip.start_file("settings.json", crate::zip_util::options_now())
        .map_err(|e| format!("zip add settings failed: {e}"))?;
    let settings_json = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("serialize settings failed: {e}"))?;
    zip.write_all(settings_json.as_bytes())
        .map_err(|e| format!("write settings failed: {e}"))?;

    for path in logging::log_files() {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if size > DIAGNOSTICS_LOG_SIZE_CAP {
            log::warn!("[diagnostics] skipping oversized log file {name} ({size} bytes)");
            continue;
        }
        // 流式写入（避免整文件读入内存），并保留日志文件的修改时间。
        match std::fs::File::open(&path) {
            Ok(mut src) => {
                zip.start_file(
                    format!("logs/{name}"),
                    crate::zip_util::options_for_source(&path),
                )
                .map_err(|e| format!("zip add {name} failed: {e}"))?;
                std::io::copy(&mut src, &mut zip)
                    .map_err(|e| format!("write {name} failed: {e}"))?;
            }
            Err(e) => log::warn!("[diagnostics] failed to read log file {name}: {e}"),
        }
    }

    zip.finish()
        .map_err(|e| format!("finish zip failed: {e}"))?;
    Ok(())
}

/// 移动端的 benchmark.json：一次短跑 + 环境快照。
///
/// 字段与桌面版**同名同义**（`availableProviders` / `ortBuildInfo` / `cpuMedianMs`
/// 等），下游解析器因此不需要为两种形态各写一条分支；差异只有两条：
/// `skipped: true`（说明「没有跑全量」）与 `smoke`（说明「短跑的结果」）。
///
/// 【为什么连 `skipped` 都要写】诊断包是**给人看的**。一个只有 `smoke` 而没有
/// 全量字段的包，看起来像「导出被截断了」；写明 `skipped` + `reason` 之后，
/// 它变成一条**结论**：这台设备上没有可比较的第二条推理通路。
fn build_mobile_benchmark_json() -> String {
    // 环境快照复用既有的 ONNX 诊断出口（不新开一条读取通路）：它已经处理了
    // 「onnx feature 未编译」「模型未加载」等情形，并且与 `onnx_status` 命令
    // 返回的是同一份事实。
    let info = crate::nsf_hifigan_onnx::diagnose_onnx_availability();
    let providers = info.providers.unwrap_or_default();
    let ort_build_info = info.onnx_version.unwrap_or_else(|| "unknown".to_string());
    // 128 帧 ≈ 1.5 秒音频：足以建会话、跑通一次真实推理，又能在 1 秒内出结果。
    const SMOKE_FRAMES: usize = 128;
    let smoke = match crate::nsf_hifigan_onnx::run_benchmark_smoke(SMOKE_FRAMES, 1) {
        Ok(elapsed_ms) => serde_json::json!({
            "ok": true,
            "frames": SMOKE_FRAMES,
            "runs": 1,
            "elapsedMs": elapsed_ms,
        }),
        Err(error) => serde_json::json!({
            "ok": false,
            "frames": SMOKE_FRAMES,
            "runs": 1,
            "error": error,
        }),
    };
    serde_json::json!({
        "skipped": true,
        "reason": "mobile_single_provider",
        "note": "Android/iOS expose a single execution provider; the full 5-run benchmark cannot change any decision and costs ~80s.",
        "smoke": smoke,
        "availableProviders": providers,
        "ortBuildInfo": ort_build_info,
        "gpuAvailable": false,
        "dmlAvailable": false,
        "benchmarkSamples": 0,
    })
    .to_string()
}

/// 运行推理设备基准测试并把结果追加进 zip；失败也写入错误信息（这本身就是
/// 有价值的诊断数据）。
fn append_benchmark(zip_path: &Path) -> Result<(), String> {
    // 【移动端不跑全量基准】见 `run_benchmark_smoke` 的文件头：Android / iOS 只
    // 暴露一个执行提供者，全量基准的决策空间在那里恒为空集，代价却是 80 秒。
    // 换成一次短跑 + 环境快照：既证明「这个包里的模型在这台机器上真能推理」，
    // 又不会让用户以为导出卡死了。
    let benchmark = if crate::platform::is_mobile() {
        build_mobile_benchmark_json()
    } else {
        log::info!("[diagnostics] running inference-device benchmark (this may take a while)...");
        match super::onnx_status::run_vocoder_benchmark() {
            Ok(results) => serde_json::to_string_pretty(&results)
                .map_err(|e| format!("serialize benchmark failed: {e}"))?,
            Err(error) => serde_json::json!({ "error": error }).to_string(),
        }
    };

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(zip_path)
        .map_err(|e| format!("reopen zip failed: {e}"))?;
    let mut zip =
        zip::ZipWriter::new_append(file).map_err(|e| format!("append zip failed: {e}"))?;
    zip.start_file("benchmark.json", crate::zip_util::options_now())
        .map_err(|e| format!("zip add benchmark failed: {e}"))?;
    zip.write_all(benchmark.as_bytes())
        .map_err(|e| format!("write benchmark failed: {e}"))?;
    zip.finish()
        .map_err(|e| format!("finish zip failed: {e}"))?;
    Ok(())
}

fn reveal_in_file_manager(path: &Path) -> Result<(), String> {
    // 在系统文件管理器中定位导出的 zip；失败不影响导出结果本身。
    //
    // 【移动端不尝试】`reveal_item_in_dir` 在 Android/iOS 上是空实现（见
    // `platform::share` 的文件头）：调用"成功"但屏幕上什么都不会发生，属于
    // 本仓明令禁止的静默失效。导出的落点本身已经是用户可见的位置
    // （Android `Android/media/<pkg>/HiFiShifter`、iOS 文件 App），前端会把
    // 路径直接显示出来 —— 那才是移动端的"交付"。
    if crate::platform::is_mobile() {
        log::info!("[diagnostics] exported to {}", path.display());
        return Ok(());
    }
    tauri_plugin_opener::reveal_item_in_dir(path).map_err(|e| format!("reveal failed: {e}"))
}

// ===================== log_frontend_error =====================

// 单行化与按字符截断已提升到 `kernel::logfile::log_field`：App 与插件共用同一份
// CWE-117 防护，且长度上限由用户选择的档位决定。
use hifishifter_kernel::logfile::log_field;

/// 每个进程回放的前端行上限（见 `frontend_replay_budget`）。
const FRONTEND_REPLAY_MAX_LINES: u64 = 500;

static FRONTEND_REPLAY_LINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FRONTEND_REPLAY_SUPPRESSED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 回放预算：`true` = 这一行照常写；`false` = 已超预算，只可能补一条汇总。
///
/// 【为什么封顶】见调用点。汇总行在**第一条**被压下的行上就出现（用户不会以为日志
/// 还完整），之后每 `FRONTEND_REPLAY_MAX_LINES` 条再补一次 —— 既不静默，也不刷屏。
fn frontend_replay_budget() -> bool {
    use std::sync::atomic::Ordering;
    let written = FRONTEND_REPLAY_LINES.fetch_add(1, Ordering::Relaxed);
    if written < FRONTEND_REPLAY_MAX_LINES {
        return true;
    }
    let suppressed = FRONTEND_REPLAY_SUPPRESSED.fetch_add(1, Ordering::Relaxed) + 1;
    if suppressed == 1 || suppressed.is_multiple_of(FRONTEND_REPLAY_MAX_LINES) {
        log::warn!(
            target: "frontend",
            "[frontend] {suppressed} line(s) suppressed (replay budget {FRONTEND_REPLAY_MAX_LINES} per session)"
        );
    }
    false
}

/// 前端错误回传入口：写入统一日志文件，便于和后端日志按时间轴对齐。
pub(super) fn log_frontend_error(
    message: String,
    detail: Option<String>,
    level: Option<String>,
) -> serde_json::Value {
    let policy = hifishifter_kernel::log_policy::policy();
    let requested = level.as_deref().unwrap_or("error");
    let accepted = match requested {
        "warn" => policy.frontend_accept.warn,
        "info" => policy.frontend_accept.info,
        "debug" => policy.frontend_accept.debug,
        _ => policy.frontend_accept.error,
    };
    if !accepted || !policy.file_enabled {
        // 「无」档严格零产出：命令本身是只读的，不写日志。
        return serde_json::json!({ "ok": true, "suppressed": true });
    }
    // 【剥掉前端自带的前缀】前端回传的是它自己格式化过的整行，后端再经 `log` 门面
    // 写一次就会出现两个时间戳、两个级别、两个 target（现场占 App 日志的 13.5%）。
    // 见 `logfile::strip_prefix`：剥掉的只是**我们自己写出来的那种前缀**。
    let message = log_field(
        hifishifter_kernel::logfile::strip_prefix(message.trim()),
        FRONTEND_MESSAGE_MAX_CHARS,
    );
    let detail = detail
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty())
        .map(|d| log_field(&d, policy.detail_cap_chars.min(FRONTEND_DETAIL_MAX_CHARS)));
    let line = match detail {
        Some(detail) => format!("[frontend] {message} | detail: {detail}"),
        None => format!("[frontend] {message}"),
    };
    // 【回放有预算】前端一旦陷入"报错 → 触发上报 → 再报错"的循环，日志会被它写满。
    // 每个进程最多回放 [`FRONTEND_REPLAY_MAX_LINES`] 行，之后只补汇总行 —— 不静默丢弃，
    // 也不让一行错误变成一万行日志。
    if !frontend_replay_budget() {
        return serde_json::json!({ "ok": true, "suppressed": true });
    }
    // 【target 用 `frontend`】此前落在 `backend_lib::commands::diagnostics` 下，按 target
    // 统计时前端行被算成后端行（现场 1 448 行）。
    let level = match requested {
        "warn" => log::Level::Warn,
        "info" => log::Level::Info,
        "debug" => log::Level::Debug,
        _ => log::Level::Error,
    };
    log::log!(target: "frontend", level, "{line}");
    serde_json::json!({ "ok": true })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **不变量：一条前端错误永远只占一行日志。**
    #[test]
    fn a_log_field_never_spans_more_than_one_line() {
        let field = log_field("boom\n[frontend] forged entry\r\nnext\u{7}", 512);
        assert!(!field.contains('\n'), "{field}");
        assert!(!field.contains('\r'), "{field}");
        assert!(!field.contains('\u{7}'), "{field}");
        assert!(field.contains("\\n[frontend] forged entry"), "{field}");
    }

    /// **不变量：截断按字符算，不按字节算（中文不会被截成半个）。**
    #[test]
    fn truncation_counts_characters_not_bytes() {
        let long = "安".repeat(FRONTEND_MESSAGE_MAX_CHARS + 10);
        let field = log_field(&long, FRONTEND_MESSAGE_MAX_CHARS);
        assert!(field.ends_with("…(truncated)"), "{field}");
        assert_eq!(field.matches('安').count(), FRONTEND_MESSAGE_MAX_CHARS);
    }
}
