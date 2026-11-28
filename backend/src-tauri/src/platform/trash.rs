//! 删除语义门面：回收站 / 永久删除。
//!
//! 【为什么需要门面】桌面上"删掉用户素材"必须进系统回收站 —— 素材不在本应用的
//! 撤销栈里，永久删除不可逆。移动端**没有可编程的回收站**：Android 的 `MediaStore`
//! 回收站只覆盖相册/视频，iOS 根本没有面向第三方应用的回收站 API。因此移动端的删除
//! 只能是永久删除，界面必须**先说清楚**（这是产品语义的差异，不是实现偷懒）。

use std::path::Path;

/// 把一个路径移入系统回收站；移动端退化为永久删除。
///
/// 【调用方必须已经做过用户确认】本函数不做二次确认 —— 确认属于界面，不属于门面。
pub fn delete(path: &Path) -> std::io::Result<()> {
    #[cfg(desktop)]
    let result = {
        // 与重构前逐字一致：`trash` crate 负责三个平台的回收站集成
        // （Windows `IFileOperation` / macOS `NSFileManager` / Linux XDG trash）。
        trash::delete(path).map_err(|error| std::io::Error::other(error.to_string()))
    };
    #[cfg(not(desktop))]
    let result = {
        log::warn!(
            "[trash] no system trash on this surface; permanently deleting {}",
            path.display()
        );
        if path.is_dir() {
            std::fs::remove_dir_all(path)
        } else {
            std::fs::remove_file(path)
        }
    };
    result
}

/// 本形态是否存在"可以撤销的删除"。
///
/// 前端据此决定文案：桌面说"移入回收站"，移动端说"永久删除"。
pub fn is_recoverable() -> bool {
    !crate::platform::is_mobile()
}
