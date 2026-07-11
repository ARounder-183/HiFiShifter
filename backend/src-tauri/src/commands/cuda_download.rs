/// CUDA Runtime 下载命令。
///
/// 从 NVIDIA 官方 redist CDN 下载 ONNX Runtime CUDA EP 所需的运行时 DLL，
/// 解压后放到 exe 同目录，使 ONNX Runtime 能在运行时通过 LoadLibrary 找到它们。
/// 下载过程通过 Tauri 事件 "cuda-download-progress" 向 UI 报告进度。
use serde::Serialize;
use std::io::{Read, Write};
use std::path::PathBuf;

/// 一个待下载的 CUDA redist 包
struct CudaPackage {
    /// 显示名（用于日志/进度）
    name: &'static str,
    /// NVIDIA redist URL
    url: &'static str,
    /// ZIP 内需要提取的 DLL 文件名
    dlls: &'static [&'static str],
}

const CUDA_PACKAGES: &[CudaPackage] = &[
    CudaPackage {
        name: "CUDA Runtime",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.6.77-archive.zip",
        dlls: &["cudart64_12.dll"],
    },
    CudaPackage {
        name: "cuBLAS",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cublas/windows-x86_64/cuda_cublas-windows-x86_64-12.6.4.1-archive.zip",
        dlls: &["cublas64_12.dll", "cublasLt64_12.dll"],
    },
    CudaPackage {
        name: "cuDNN",
        url: "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0_cuda12-archive.zip",
        dlls: &[
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_adv64_9.dll",
        ],
    },
];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CudaDownloadProgress {
    /// 当前包名
    package: String,
    /// 0-based 当前包索引
    package_index: usize,
    /// 总包数
    package_count: usize,
    /// 当前包已下载字节数
    downloaded_bytes: u64,
    /// 当前包总字节数（可能为 0 如果无法获取）
    total_bytes: u64,
    /// "downloading" | "extracting" | "installing" | "done" | "error"
    stage: String,
    /// 仅 stage="error" 时有值
    error: Option<String>,
}

fn send_progress(
    app: &tauri::AppHandle,
    pkg_idx: usize,
    pkg_count: usize,
    pkg_name: &str,
    stage: &str,
    downloaded: u64,
    total: u64,
    error: Option<&str>,
) {
    let payload = CudaDownloadProgress {
        package: pkg_name.to_string(),
        package_index: pkg_idx,
        package_count: pkg_count,
        downloaded_bytes: downloaded,
        total_bytes: total,
        stage: stage.to_string(),
        error: error.map(|s| s.to_string()),
    };
    let _ = app.emit("cuda-download-progress", payload);
}

/// 下载单个 URL 到内存，通过回调报告进度。
fn download_to_memory(
    url: &str,
    on_progress: impl Fn(u64, u64),
) -> Result<Vec<u8>, String> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = resp.status();
    if status != 200 {
        return Err(format!("HTTP {status}"));
    }

    let total = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; if total > 0 { total as usize } else { 64 * 1024 }];
    let mut downloaded = 0u64;
    let mut chunk = [0u8; 65536];

    loop {
        let n = reader
            .read(&mut chunk)
            .map_err(|e| format!("download read error: {e}"))?;
        if n == 0 {
            break;
        }
        if (downloaded + n as u64) > buf.len() as u64 {
            buf.resize(buf.len() + 64 * 1024, 0);
        }
        buf[downloaded as usize..downloaded as usize + n].copy_from_slice(&chunk[..n]);
        downloaded += n as u64;
        on_progress(downloaded, total);
    }

    buf.truncate(downloaded as usize);
    Ok(buf)
}

/// 从 ZIP 字节流中搜索指定 DLL 文件。
fn extract_dlls_from_zip(
    zip_data: &[u8],
    dll_names: &[&str],
    dest_dir: &std::path::Path,
) -> Result<usize, String> {
    let cursor = std::io::Cursor::new(zip_data);
    let mut archive =
        zip::ZipArchive::new(cursor).map_err(|e| format!("open zip failed: {e}"))?;

    let mut extracted = 0usize;
    for dll_name in dll_names {
        // 在 ZIP 中递归搜索（NVIDIA redist ZIP 内部结构不固定）
        let mut found = false;
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| format!("zip entry error: {e}"))?;
            let entry_name = file.name().to_string();
            if entry_name.ends_with(dll_name) && !file.is_dir() {
                let dest_path = dest_dir.join(dll_name);
                let mut dest =
                    std::fs::File::create(&dest_path)
                        .map_err(|e| format!("create {dll_name}: {e}"))?;
                let mut buf = Vec::with_capacity(file.size() as usize);
                file.read_to_end(&mut buf)
                    .map_err(|e| format!("read {dll_name} from zip: {e}"))?;
                dest.write_all(&buf)
                    .map_err(|e| format!("write {dll_name}: {e}"))?;
                extracted += 1;
                found = true;
                break;
            }
        }
        if !found {
            eprintln!(
                "[cuda-download] DLL '{dll_name}' not found in zip archive"
            );
        }
    }
    Ok(extracted)
}

/// Tauri 命令：下载 CUDA Runtime DLL 到 exe 同目录。
///
/// 下载过程中通过 "cuda-download-progress" 事件推送进度。
/// 前端可监听此事件显示进度条。
#[tauri::command(rename_all = "camelCase")]
pub fn download_cuda_runtime(app: tauri::AppHandle) -> Result<(), String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .ok_or_else(|| "cannot determine exe directory".to_string())?;

    let pkg_count = CUDA_PACKAGES.len();

    for (idx, pkg) in CUDA_PACKAGES.iter().enumerate() {
        // 先检查是否已有这个包的所有 DLL
        let all_present = pkg.dlls.iter().all(|dll| exe_dir.join(dll).is_file());
        if all_present {
            send_progress(&app, idx, pkg_count, pkg.name, "done", 1, 1, None);
            continue;
        }

        send_progress(&app, idx, pkg_count, pkg.name, "downloading", 0, 0, None);

        let zip_data = download_to_memory(pkg.url, |downloaded, total| {
            send_progress(&app, idx, pkg_count, pkg.name, "downloading", downloaded, total, None);
        })?;

        send_progress(&app, idx, pkg_count, pkg.name, "extracting", 0, 0, None);

        let count = extract_dlls_from_zip(&zip_data, pkg.dlls, &exe_dir)?;

        if count == 0 && !pkg.dlls.is_empty() {
            send_progress(
                &app,
                idx,
                pkg_count,
                pkg.name,
                "error",
                0,
                0,
                Some("no DLLs extracted from zip"),
            );
        } else {
            send_progress(&app, idx, pkg_count, pkg.name, "done", 1, 1, None);
        }
    }

    Ok(())
}

/// 获取 CUDA runtime DLL 的目标下载大小估算（用于 UI 展示）。
/// 返回值是所有包的总大小（字节），若无法获取则返回 None。
pub fn cuda_runtime_estimated_size() -> Option<u64> {
    // 基于已知的 NVIDIA redist 包大小估算（实际可能因版本不同而变化）
    Some(1_100_000_000) // ~1.1 GB
}
