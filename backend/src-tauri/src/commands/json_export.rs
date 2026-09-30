//! 通用「导出 JSON」：原生保存对话框 + 写文件 + 在文件管理器中定位。
//!
//! 【为什么必须走后端】Tauri 的 WebView（wry）默认拦截页面发起的下载，
//! Blob + `<a download>` 的浏览器方案在壳内**静默失败** —— 用户点「导出」什么都
//! 不会发生。凡是要"存成文件"的导出都必须走本模块（布局、外观主题各一处；外观
//! 主题此前正是漏走了这一步，所以导出完全不起作用）。
//!
//! 【为什么是一条通用命令而不是每处一条】它做的事与业务无关（选路径 → 写字节 →
//! 定位），唯一随业务变化的只有"标题 / 默认文件名"。各写一份的代价已经出现过一次：
//! 外观主题那条路径没跟着布局一起搬家，于是它单独坏掉了。
//!
//! 【为什么不用异步/线程池】JSON 是几 KB 的纯文本：对话框与写文件都在同步命令
//! （主线程）里完成即可，不值得为它拆成两步 IPC（对比诊断包的先选路径、再在
//! 线程池里打包 + 跑基准测试）。

/// 弹出原生保存对话框，把前端序列化好的 JSON 写入所选路径。
///
/// 返回 `{ ok: true, canceled: true }`（用户取消）、`{ ok: true, path }`（成功）
/// 或 `{ ok: false, error }`（失败）—— 前端只需判断 `ok` / `canceled`。
///
/// @param json 前端序列化好的 JSON 文本。
/// @param default_file_name 保存对话框的默认文件名（含扩展名）。
/// @param title 保存对话框标题。
/// @param log_tag 日志前缀（如 `layout` / `theme`），便于在日志里区分来源。
pub(super) fn export_json_file(
    json: String,
    default_file_name: &str,
    title: &str,
    log_tag: &str,
) -> serde_json::Value {
    let Some(path) = rfd::FileDialog::new()
        .set_title(title)
        .add_filter("JSON", &["json"])
        .set_file_name(default_file_name)
        .save_file()
    else {
        return serde_json::json!({ "ok": true, "canceled": true });
    };

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                return serde_json::json!({
                    "ok": false,
                    "error": format!("create directory failed: {error}")
                });
            }
        }
    }

    if let Err(error) = std::fs::write(&path, json.as_bytes()) {
        log::error!("[{log_tag}] export failed: {error}");
        return serde_json::json!({ "ok": false, "error": format!("write failed: {error}") });
    }

    // 在文件管理器中定位导出的文件；失败不影响导出结果本身。
    if let Err(error) = tauri_plugin_opener::reveal_item_in_dir(&path) {
        log::warn!("[{log_tag}] could not reveal the exported file: {error}");
    }
    log::info!("[{log_tag}] export finished: {}", path.display());
    serde_json::json!({ "ok": true, "path": path.display().to_string() })
}
