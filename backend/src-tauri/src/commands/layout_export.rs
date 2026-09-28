//! 布局导出命令：视图 → 布局 → 「导出布局...」。
//!
//! 【为什么不走浏览器的 `<a download>`】Tauri 的 WebView（wry）默认拦截页面
//! 发起的下载，Blob + download 属性的方案在壳内**静默失败** —— 用户点「导出
//! 布局」什么都不会发生。导出走**原生保存对话框 + 后端写文件**，与「导出诊断
//! 信息」同一模式。导入则不需要这一步：`<input type=file>` 的文件选择在
//! WebView 内可用。
//!
//! JSON 是几 KB 的纯文本：对话框与写文件都在同步命令（主线程）里完成，
//! 不值得为它拆成两步 IPC 或动用阻塞线程池（对比诊断包的先选路径、再在
//! 线程池里打包 + 跑基准测试）。

/// 弹出原生保存对话框（默认名 `hifishifter-layout.json`），把前端序列化好的
/// 布局 JSON 写入所选路径。
pub(super) fn export_layout_json(json: String) -> serde_json::Value {
    let Some(path) = rfd::FileDialog::new()
        .set_title("Export layout")
        .add_filter("JSON", &["json"])
        .set_file_name("hifishifter-layout.json")
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
        log::error!("[layout] export failed: {error}");
        return serde_json::json!({ "ok": false, "error": format!("write failed: {error}") });
    }

    // 在文件管理器中定位导出的文件；失败不影响导出结果本身。
    if let Err(error) = tauri_plugin_opener::reveal_item_in_dir(&path) {
        log::warn!("[layout] could not reveal exported layout: {error}");
    }
    log::info!("[layout] export finished: {}", path.display());
    serde_json::json!({ "ok": true, "path": path.display().to_string() })
}
