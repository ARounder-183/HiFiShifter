//! VST 插件扫描器。
//!
//! 搜索系统标准 VST 目录和用户自定义路径，
//! 发现 VST2 `.dll` 和 VST3 `.vst3` 插件并提取元数据。
//!
//! VST2 扫描时会尝试短暂加载 DLL 获取真实元数据（名称/厂商/通道数等），
//! 加载失败则回退到文件名推断。VST3 扫描同样通过 COM 接口提取元数据。

use std::path::{Path, PathBuf};

use super::{VstFormat, VstPluginDescriptor, VstPluginRegistry};

// ─── 系统 VST 路径 ──────────────────────────────────────────────────────────

/// 获取系统默认的 VST2 扫描路径。
pub fn default_vst2_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    #[cfg(target_os = "windows")]
    {
        // Windows 标准路径
        if let Ok(pf) = std::env::var("ProgramFiles") {
            paths.push(PathBuf::from(&pf).join("VSTPlugins"));
            paths.push(PathBuf::from(&pf).join("Steinberg").join("VSTPlugins"));
            paths.push(PathBuf::from(&pf).join("Common Files").join("VST2"));
        }
        if let Ok(pf86) = std::env::var("ProgramFiles(x86)") {
            paths.push(PathBuf::from(&pf86).join("VSTPlugins"));
            paths.push(PathBuf::from(&pf86).join("Steinberg").join("VSTPlugins"));
        }
        // 注册表路径（VstPluginsPath）
        if let Ok(vst_path) = std::env::var("VST_PATH") {
            for p in vst_path.split(';') {
                paths.push(PathBuf::from(p));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        paths.push(PathBuf::from("/Library/Audio/Plug-Ins/VST"));
        if let Ok(home) = std::env::var("HOME") {
            paths.push(PathBuf::from(&home).join("Library/Audio/Plug-Ins/VST"));
        }
    }

    #[cfg(target_os = "linux")]
    {
        paths.push(PathBuf::from("/usr/lib/vst"));
        paths.push(PathBuf::from("/usr/local/lib/vst"));
        if let Ok(home) = std::env::var("HOME") {
            paths.push(PathBuf::from(&home).join(".vst"));
        }
        if let Ok(vst_path) = std::env::var("VST_PATH") {
            for p in vst_path.split(':') {
                paths.push(PathBuf::from(p));
            }
        }
    }

    paths
}

/// 获取系统默认的 VST3 扫描路径。
pub fn default_vst3_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    #[cfg(target_os = "windows")]
    {
        if let Ok(pf) = std::env::var("ProgramFiles") {
            paths.push(PathBuf::from(&pf).join("Common Files").join("VST3"));
        }
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            paths.push(PathBuf::from(&local).join("Programs").join("Common").join("VST3"));
        }
    }

    #[cfg(target_os = "macos")]
    {
        paths.push(PathBuf::from("/Library/Audio/Plug-Ins/VST3"));
        if let Ok(home) = std::env::var("HOME") {
            paths.push(PathBuf::from(&home).join("Library/Audio/Plug-Ins/VST3"));
        }
    }

    #[cfg(target_os = "linux")]
    {
        paths.push(PathBuf::from("/usr/lib/vst3"));
        paths.push(PathBuf::from("/usr/local/lib/vst3"));
        if let Ok(home) = std::env::var("HOME") {
            paths.push(PathBuf::from(&home).join(".vst3"));
        }
    }

    paths
}

// ─── 扫描逻辑 ──────────────────────────────────────────────────────────────

/// 尝试短暂加载 VST2 DLL 获取真实元数据。
///
/// 加载后立即获取 info 并 suspend + drop，不保留实例。
/// 失败时返回 None，调用方回退到文件名推断。
#[cfg(feature = "vst")]
fn probe_vst2_metadata(path: &Path) -> Option<VstPluginDescriptor> {
    use super::plugin_instance::SimpleVst2Host;
    use std::sync::{Arc, Mutex};
    use vst2::host::PluginLoader;
    use vst2::plugin::Plugin;

    let host = Arc::new(Mutex::new(SimpleVst2Host));

    // 使用 catch_unwind 防止有问题的 DLL 导致整个扫描崩溃
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut loader = PluginLoader::load(path, host.clone()).ok()?;
        let mut plugin = loader.instance().ok()?;
        let info = plugin.get_info();

        // 确定类别
        let category = match info.category {
            vst2::plugin::Category::Effect => "Effect",
            vst2::plugin::Category::Synth => "Synth",
            vst2::plugin::Category::Analysis => "Analysis",
            vst2::plugin::Category::Mastering => "Mastering",
            vst2::plugin::Category::Spacializer => "Spacializer",
            vst2::plugin::Category::RoomFx => "RoomFx",
            vst2::plugin::Category::Generator => "Generator",
            _ => "Effect",
        }
        .to_string();

        let is_instrument = matches!(
            info.category,
            vst2::plugin::Category::Synth | vst2::plugin::Category::Generator
        );

        let uid = generate_uid(path);
        let name = if info.name.is_empty() {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown")
                .to_string()
        } else {
            info.name.clone()
        };

        let desc = VstPluginDescriptor {
            uid,
            name,
            vendor: info.vendor.clone(),
            format: VstFormat::Vst2,
            path: path.to_path_buf(),
            category,
            is_instrument,
            version: format!("{}", info.version),
            num_inputs: info.inputs as u32,
            num_outputs: info.outputs as u32,
        };

        // 清理：suspend 插件再 drop
        plugin.suspend();
        drop(plugin);

        Some(desc)
    }));

    match result {
        Ok(opt) => opt,
        Err(_) => {
            eprintln!(
                "[vst_host::scanner] Plugin panicked during probe, skipping: {}",
                path.display()
            );
            None
        }
    }
}

/// 最大递归扫描深度，防止在深层目录树中耗费过多时间。
const MAX_SCAN_DEPTH: u32 = 3;

/// 尝试通过 COM 接口从 VST3 插件提取真实元数据。
///
/// 加载 DLL → 获取 IPluginFactory → 读取类信息 → 释放。
/// 失败时返回 None，调用方回退到文件名推断。
#[cfg(feature = "vst")]
fn probe_vst3_metadata(path: &Path) -> Option<VstPluginDescriptor> {
    use super::plugin_host;

    // 解析 bundle 路径为实际 DLL 路径
    let module_path = match plugin_host::resolve_vst3_module_path(path) {
        Ok(p) => p,
        Err(_) => return None,
    };

    let probe = super::vst3_com::probe_vst3_metadata(&module_path)?;

    let uid = generate_uid(path);
    let name = if probe.name.is_empty() {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string()
    } else {
        probe.name
    };

    Some(VstPluginDescriptor {
        uid,
        name,
        vendor: probe.vendor,
        format: VstFormat::Vst3,
        path: path.to_path_buf(),
        category: probe.category,
        is_instrument: probe.is_instrument,
        version: String::new(),
        num_inputs: probe.num_inputs,
        num_outputs: probe.num_outputs,
    })
}

/// 扫描指定目录中的 VST2 插件。
///
/// 对每个发现的 `.dll`/`.vst`/`.so` 文件，尝试加载获取真实元数据。
/// 加载失败时回退到文件名推断，确保扫描不会因单个插件而中断。
/// `depth` 参数控制递归深度，超过 `MAX_SCAN_DEPTH` 时停止递归。
fn scan_vst2_directory(dir: &Path) -> Vec<VstPluginDescriptor> {
    scan_vst2_directory_impl(dir, 0)
}

fn scan_vst2_directory_impl(dir: &Path, depth: u32) -> Vec<VstPluginDescriptor> {
    let mut results = Vec::new();

    if !dir.exists() || !dir.is_dir() {
        return results;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return results,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // 递归搜索子目录（受深度限制）
            if depth < MAX_SCAN_DEPTH {
                results.extend(scan_vst2_directory_impl(&path, depth + 1));
            }
            continue;
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        #[cfg(target_os = "windows")]
        let is_vst2 = ext.eq_ignore_ascii_case("dll");

        #[cfg(target_os = "macos")]
        let is_vst2 = ext.eq_ignore_ascii_case("vst");

        #[cfg(target_os = "linux")]
        let is_vst2 = ext.eq_ignore_ascii_case("so");

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        let is_vst2 = false;

        if !is_vst2 {
            continue;
        }

        // 尝试加载获取真实元数据
        #[cfg(feature = "vst")]
        {
            if let Some(desc) = probe_vst2_metadata(&path) {
                results.push(desc);
                continue;
            }
        }

        // 回退：仅使用文件名信息
        let uid = generate_uid(&path);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        results.push(VstPluginDescriptor {
            uid,
            name,
            vendor: String::new(),
            format: VstFormat::Vst2,
            path: path.clone(),
            category: "Effect".to_string(),
            is_instrument: false,
            version: String::new(),
            num_inputs: 2,
            num_outputs: 2,
        });
    }

    results
}

/// 扫描指定目录中的 VST3 插件。
fn scan_vst3_directory(dir: &Path) -> Vec<VstPluginDescriptor> {
    scan_vst3_directory_impl(dir, 0)
}

fn scan_vst3_directory_impl(dir: &Path, depth: u32) -> Vec<VstPluginDescriptor> {
    let mut results = Vec::new();

    if !dir.exists() || !dir.is_dir() {
        return results;
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return results,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        let is_vst3 = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("vst3"))
            .unwrap_or(false);

        if !is_vst3 {
            if path.is_dir() && depth < MAX_SCAN_DEPTH {
                // 递归搜索子目录（受深度限制）
                results.extend(scan_vst3_directory_impl(&path, depth + 1));
            }
            continue;
        }

        // 尝试通过 COM 接口提取真实元数据
        #[cfg(feature = "vst")]
        {
            if let Some(desc) = probe_vst3_metadata(&path) {
                results.push(desc);
                continue;
            }
        }

        // 回退：仅使用文件名信息
        let uid = generate_uid(&path);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        results.push(VstPluginDescriptor {
            uid,
            name,
            vendor: String::new(),
            format: VstFormat::Vst3,
            path: path.clone(),
            category: "Effect".to_string(),
            is_instrument: false,
            version: String::new(),
            num_inputs: 2,
            num_outputs: 2,
        });
    }

    results
}

/// 生成插件 UID（基于路径的 BLAKE3 哈希）。
fn generate_uid(path: &Path) -> String {
    let input = path.to_string_lossy();
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex()[..16].to_string()
}

// ─── 公共 API ──────────────────────────────────────────────────────────────

/// 执行完整的 VST 插件扫描。
///
/// 搜索系统默认路径和注册表中的自定义路径，
/// 返回扫描到的所有插件描述符列表。
pub fn scan_all_plugins(custom_paths: &[PathBuf]) -> Vec<VstPluginDescriptor> {
    let mut all = Vec::new();

    // VST2 目录
    for dir in default_vst2_paths() {
        all.extend(scan_vst2_directory(&dir));
    }

    // VST3 目录
    for dir in default_vst3_paths() {
        all.extend(scan_vst3_directory(&dir));
    }

    // 自定义路径（同时搜索 VST2 和 VST3）
    for dir in custom_paths {
        all.extend(scan_vst2_directory(dir));
        all.extend(scan_vst3_directory(dir));
    }

    // 去重（按 UID）
    let mut seen = std::collections::HashSet::new();
    all.retain(|d| seen.insert(d.uid.clone()));

    // 按名称排序
    all.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    all
}

/// 在后台线程中异步执行插件扫描，并更新注册表。
///
/// 使用 `std::thread::spawn` 在独立线程中执行扫描，不阻塞 Tauri 命令线程。
/// 扫描完成后通过 `scan_in_progress` 原子标志通知调用方。
/// 如果已有扫描任务在运行，本次调用将被忽略。
pub fn scan_plugins_async(registry: std::sync::Arc<VstPluginRegistry>) {
    use std::sync::atomic::Ordering;

    if registry
        .scan_in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        // 已经有扫描在运行
        return;
    }

    let reg = registry.clone();
    std::thread::Builder::new()
        .name("vst-plugin-scan".to_string())
        .spawn(move || {
            eprintln!("[vst_host::scanner] Background scan started...");

            let custom_paths = reg
                .custom_scan_paths
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();

            let result = scan_all_plugins(&custom_paths);
            let count = result.len();

            let mut descs = reg
                .descriptors
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *descs = result;
            drop(descs);

            reg.scan_in_progress
                .store(false, Ordering::SeqCst);

            eprintln!(
                "[vst_host::scanner] Background scan complete: {} plugins found",
                count
            );
        })
        .unwrap_or_else(|e| {
            eprintln!("[vst_host::scanner] Failed to spawn scan thread: {}", e);
            registry
                .scan_in_progress
                .store(false, std::sync::atomic::Ordering::SeqCst);
        });
}

/// 同步扫描并更新注册表（用于需要立即获取结果的场景）。
///
/// 阻塞当前线程直到扫描完成。适用于初始化或手动触发后需要立即使用结果的场景。
pub fn scan_plugins_sync(registry: &VstPluginRegistry) {
    use std::sync::atomic::Ordering;

    if registry
        .scan_in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }

    let custom_paths = registry
        .custom_scan_paths
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    let result = scan_all_plugins(&custom_paths);

    let mut descs = registry
        .descriptors
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *descs = result;

    registry
        .scan_in_progress
        .store(false, Ordering::SeqCst);

    eprintln!(
        "[vst_host::scanner] Sync scan complete: {} plugins found",
        descs.len()
    );
}

// ─── 自定义扫描路径持久化 ───────────────────────────────────────────────────

/// 从配置目录加载用户自定义的 VST 扫描路径。
///
/// 路径保存在 `config_dir/vst_scan_paths.json` 中。
/// 文件不存在或解析失败时返回空列表。
pub fn load_custom_scan_paths(config_dir: &Path) -> Vec<PathBuf> {
    let file = config_dir.join("vst_scan_paths.json");
    let Ok(data) = std::fs::read_to_string(&file) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<String>>(&data)
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .filter(|p| p.exists() && p.is_dir())
        .collect()
}

/// 将用户自定义的 VST 扫描路径保存到配置目录。
///
/// 持久化为 `config_dir/vst_scan_paths.json`。
/// 写入失败时静默忽略（仅打印日志）。
pub fn save_custom_scan_paths(config_dir: &Path, paths: &[PathBuf]) {
    let file = config_dir.join("vst_scan_paths.json");
    let str_paths: Vec<String> = paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
    match serde_json::to_string_pretty(&str_paths) {
        Ok(data) => {
            if let Err(e) = std::fs::write(&file, data) {
                eprintln!(
                    "[vst_host::scanner] Failed to save custom scan paths: {}",
                    e
                );
            }
        }
        Err(e) => {
            eprintln!(
                "[vst_host::scanner] Failed to serialize custom scan paths: {}",
                e
            );
        }
    }
}
