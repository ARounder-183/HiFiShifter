//! 随包资源（模型）的定位与物化。
//!
//! 【为什么不能直接用 `resource_dir()`】它在桌面与 iOS 上是一个真实的文件系统路径，
//! 但在 **Android 上不是**：tauri 的 `PathPlugin.kt` 里 `getResourcesDir` 直接返回
//! 常量 `"asset://localhost/"`（一个 WebView 的自定义协议 URL）。拿它去 `join("models")`
//! 得到的是 `asset://localhost/models` —— 一个 `Path` 对象，`exists()` 恒为 false，
//! 内核的 `model_store::resolve_and_register` 会静默报"模型不可用"。
//!
//! 因此 Android 上必须先**物化**：把 APK 里 `assets/` 下的资源按需解出来，落到
//! `app_data_dir()/models`，此后内核按普通路径读。
//!
//! ## 为什么用 zip 直读而不是 JNI 的 `AAssetManager`
//!
//! APK 就是一个 zip。`zip` 已经是本 crate 的依赖（工程文件归档在用），因此物化不需要
//! 新增任何依赖、也不需要 JNI 上下文（后者在 `tao` 里已初始化，但取它的 API 在
//! `ndk-context` 的多个版本间改过签名 —— 一个纯 Rust 的 zip 读取没有这个问题）。
//!
//! APK 路径的解析见 [`android_apk_path`]：优先环境变量（调试 / 测试用），其次从
//! `/proc/self/maps` 里认出 `base.apk`。

// `Path` 只在 Android 的物化路径上用到（`materialize_from_apk`），因此那条 import
// 跟着 cfg 走 —— 否则桌面构建会多出一条 "unused import" 警告，而警告噪声会掩盖真问题。
#[cfg(target_os = "android")]
use std::path::Path;
use std::path::PathBuf;

use tauri::Manager;

/// 模型目录。
///
/// 调用方把它交给 `hifishifter_kernel::model_store::resolve_and_register`。
pub fn models_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if !crate::platform::is_mobile() {
        // 桌面：与重构前逐字一致。
        let res_dir = app
            .path()
            .resource_dir()
            .map_err(|error| format!("resource dir unresolved: {error}"))?;
        return Ok(res_dir.join("models"));
    }

    #[cfg(target_os = "ios")]
    {
        // iOS 的 `resource_dir()` 是 .app bundle 里的真实目录，资源就是文件 ——
        // 不需要物化，直接读。
        let res_dir = app
            .path()
            .resource_dir()
            .map_err(|error| format!("resource dir unresolved: {error}"))?;
        return Ok(res_dir.join("models"));
    }

    #[cfg(target_os = "android")]
    {
        let base = app
            .path()
            .app_data_dir()
            .map_err(|error| format!("app data dir unresolved: {error}"))?;
        let target = base.join("models");
        materialize_from_apk(&target)?;
        return Ok(target);
    }

    #[allow(unreachable_code)]
    Err("models root unresolved on this surface".to_string())
}

/// 物化 APK 里的 `assets/models/**` 到 `target`。
///
/// 【幂等判据是"逐文件长度相等"】Android 的系统清理不会动 `files/`，但用户在设置里
/// "清除存储"会。用长度比较而不是"目录存在"：一个被截断的模型文件比没有更糟 ——
/// ORT 会在建会话时抛一个与根因无关的解析错误。
#[cfg(target_os = "android")]
fn materialize_from_apk(target: &Path) -> Result<(), String> {
    let Some(apk) = android_apk_path() else {
        return Err(
            "android APK path not found; cannot materialize bundled models (set HIFISHIFTER_ANDROID_APK to override)"
                .to_string(),
        );
    };
    let file = std::fs::File::open(&apk)
        .map_err(|error| format!("open apk {}: {error}", apk.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|error| format!("read apk as zip: {error}"))?;

    std::fs::create_dir_all(target)
        .map_err(|error| format!("create {}: {error}", target.display()))?;

    let mut copied = 0usize;
    for index in 0..archive.len() {
        let mut entry = match archive.by_index(index) {
            Ok(entry) => entry,
            Err(error) => {
                log::warn!("[resources] apk entry #{index} unreadable: {error}");
                continue;
            }
        };
        let Some(relative) = entry.name().strip_prefix("assets/models/") else {
            continue;
        };
        if relative.is_empty() || relative.ends_with('/') {
            continue;
        }
        let dest = target.join(relative);
        let expected = entry.size();
        if std::fs::metadata(&dest).map(|meta| meta.len()).ok() == Some(expected) {
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        // 写 `.part` 再 rename：中途被杀掉时不会留下一个长度错误的"完整"文件。
        let partial = dest.with_extension("part");
        {
            let mut out = std::fs::File::create(&partial)
                .map_err(|error| format!("create {}: {error}", partial.display()))?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|error| format!("extract {relative}: {error}"))?;
        }
        std::fs::rename(&partial, &dest).map_err(|error| format!("commit {relative}: {error}"))?;
        copied += 1;
    }
    if copied > 0 {
        log::info!(
            "[resources] materialized {copied} model file(s) into {}",
            target.display()
        );
    } else {
        log::warn!("[resources] no bundled models found in {}", apk.display());
    }
    Ok(())
}

/// 定位 APK。
///
/// 【两条来源】
/// 1. `HIFISHIFTER_ANDROID_APK`：调试与自动化测试用（真机上是 `pm path <pkg>` 的输出）。
/// 2. `/proc/self/maps`：应用自己的 native 库必然出现在映射表里，其路径形如
///    `/data/app/~~<hash>/<pkg>-<n>/lib/arm64/libbackend_lib.so`，同一目录下的
///    `base.apk` 就是安装包。这是无依赖的启发式；换不到时返回 `None` 由上层报错，
///    绝不猜一个可能错的路径。
#[cfg(target_os = "android")]
fn android_apk_path() -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("HIFISHIFTER_ANDROID_APK") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Some(path);
        }
    }
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    for line in maps.lines() {
        let Some(candidate) = line.split_whitespace().last() else {
            continue;
        };
        if !candidate.contains("base.apk") {
            continue;
        }
        // maps 里的路径是 `<apk>!/lib/...`（未压缩的 .so）或 `<apk>`（已解压的
        // .so 所在目录的同级）。两种情况都只要 `base.apk` 之前那一段。
        let apk = candidate.split("!/").next().unwrap_or(candidate);
        let path = PathBuf::from(apk);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}
