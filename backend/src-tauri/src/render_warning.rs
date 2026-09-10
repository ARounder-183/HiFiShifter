/*
 * render_warning.rs - 渲染/推理层"用户可见告警"的单一出口。
 *
 * 主要内容：
 * - `install`：在 app setup 时登记 `AppHandle`（engine / vocoder / 导出路径都
 *   不在命令上下文里，拿不到 `State`，因此需要进程级句柄）。
 * - `warn`：发出 `render_warning` 事件给前端，同时写日志。
 *
 * 为什么需要它（见 P0-5 / A6 / A12）：
 * 渲染链上有几处失败此前是**静默**的 —— 解码失败的片段会永远缺席（用户只
 * 听到"莫名无声"），GPU 执行提供者（DirectML/CoreML）因驱动异常被禁用后
 * 用户只会觉得"突然变慢了"。这两类都不是崩溃，日志里有，但用户看不到，
 * 于是排查成本极高。这里把它们统一成前端可见的告警。
 *
 * 节流：
 * 同一个 `(kind, message)` 在 `THROTTLE_WINDOW` 内只投递一次、只记一条日志。
 * 解码失败发生在每个 clip 的每次渲染尝试上，不节流会瞬间刷屏。
 */

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use tauri::Emitter as _;

/// 同一告警的最小重复投递间隔。
const THROTTLE_WINDOW: Duration = Duration::from_secs(10);

/// 节流表的条目上限（超出后清空重建，避免键爆炸）。
const THROTTLE_MAX_ENTRIES: usize = 256;

/// 告警类别（前端可用于区分展示方式）。
pub(crate) const KIND_DECODE_FAILED: &str = "decode_failed";
pub(crate) const KIND_GPU_DISABLED: &str = "gpu_disabled";
pub(crate) const KIND_CLIP_RENDER_FAILED: &str = "clip_render_failed";

static APP_HANDLE: OnceLock<tauri::AppHandle> = OnceLock::new();

fn throttle_table() -> &'static Mutex<HashMap<String, Instant>> {
    static TABLE: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 在 app setup 阶段登记句柄。重复调用以首次为准。
pub(crate) fn install(handle: tauri::AppHandle) {
    let _ = APP_HANDLE.set(handle);
}

/// 是否应该投递（按 `(kind, message)` 节流）。
fn should_deliver(key: &str) -> bool {
    let Ok(mut table) = throttle_table().lock() else {
        return true;
    };
    let now = Instant::now();
    match table.get(key) {
        Some(last) if now.duration_since(*last) < THROTTLE_WINDOW => false,
        _ => {
            if table.len() >= THROTTLE_MAX_ENTRIES {
                table.clear();
            }
            table.insert(key.to_string(), now);
            true
        }
    }
}

/// 发出一个渲染告警。
///
/// `kind` 见本模块常量；`message` 是给用户看的短句（英文，前端按需本地化或
/// 原样展示）；`detail` 是可选的诊断细节（片段 id / 路径 / 底层错误）。
pub(crate) fn warn(kind: &str, message: &str, detail: Option<&str>) {
    let throttle_key = match detail {
        Some(d) => format!("{kind}|{message}|{d}"),
        None => format!("{kind}|{message}"),
    };
    if !should_deliver(&throttle_key) {
        return;
    }

    // 日志始终写（日志层另有自己的限流），事件按上面的节流投递。
    match detail {
        Some(d) => log::warn!("[render_warning] kind={kind} message={message} detail={d}"),
        None => log::warn!("[render_warning] kind={kind} message={message}"),
    }

    let Some(handle) = APP_HANDLE.get() else {
        // 尚未 install（例如单元测试 / CLI 基准路径）：只写日志即可。
        return;
    };

    #[derive(serde::Serialize, Clone)]
    struct Payload<'a> {
        kind: &'a str,
        message: &'a str,
        detail: Option<&'a str>,
    }

    if let Err(e) = handle.emit("render_warning", Payload { kind, message, detail }) {
        log::warn!("[render_warning] emit failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_warning_is_throttled_within_the_window() {
        // 节流表是进程级的；用唯一的 key 避免与其它测试互相影响。
        let msg = "throttle-probe";
        assert!(should_deliver(msg), "first delivery must pass");
        assert!(!should_deliver(msg), "second delivery inside the window must be suppressed");
    }

    #[test]
    fn distinct_warnings_are_not_throttled_against_each_other() {
        assert!(should_deliver("probe-a"));
        assert!(should_deliver("probe-b"));
    }

    #[test]
    fn warn_without_install_does_not_panic() {
        // 未 install 句柄时（测试/CLI 路径）必须安全降级为仅记日志。
        warn(KIND_DECODE_FAILED, "probe message", Some("probe detail"));
        warn(KIND_GPU_DISABLED, "probe gpu", None);
    }
}
