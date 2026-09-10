// 渲染取消令牌（RenderCancelToken）
//
// 职责：为"一轮渲染"提供取消信号，避免渲染路径直接读写裸的全局标志而
// 漏掉纪元推进。本模块只提供"取消信号"这一基础设施，不涉及渲染、缓存或
// 前端事件；`commands/playback.rs` 是它当前唯一的使用者。
//
// ── 模型 ──────────────────────────────────────────────────────────────────
// 前后台渲染已统一为同一套 pass 机制（是否启用后台预渲染只决定渲染的
// 触发时机），因此只有一种令牌：跟踪全局 `BG_RENDER_CANCEL` 的原始值。
//
// 纪元（CANCEL_EPOCH）仍然必要：单靠一个布尔标志无法回答"这次取消是在
// 我这轮开始之前还是之后发出的"。后台渲染的清理与重启逻辑并不能保证标志
// 一定被复位 —— 例如取消请求与渲染线程收尾相撞时（`was_active=true` 但
// 线程已跑完清理），标志就会残留。当前统一模型下 `start_background_render`
// 会在开头复位标志，纪元主要作为"新轮次忽略历史残留"的结构性保障保留。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// 全局"取消请求"纪元：每产生一次新的取消请求就递增。
static CANCEL_EPOCH: AtomicU64 = AtomicU64::new(0);

/// 发出一次新的全局取消请求：置位标志并推进纪元。
///
/// ★ 所有希望中断"当前正在跑的渲染"的调用点都必须走这个函数，而不是直接
/// `BG_RENDER_CANCEL.store(true, ..)`：不推进纪元的新请求会被之后启动的
/// 渲染轮次当作历史残留而忽略掉。
pub(crate) fn request_global_cancel() {
    CANCEL_EPOCH.fetch_add(1, Ordering::AcqRel);
    crate::commands::playback::BG_RENDER_CANCEL.store(true, Ordering::Release);
}

/// 一轮渲染使用的取消令牌：跟踪全局 `BG_RENDER_CANCEL` 的原始值。
///
/// 由 `playback.rs` 在启动渲染时构造，再以 `&RenderCancelToken` 传入
/// `render_single_clip`，供其在各个耗时阶段之间做检查点判断。
pub(crate) enum RenderCancelToken {
    Background,
}

impl RenderCancelToken {
    /// 后台预渲染使用的令牌：绑定全局标志，以便时间线编辑能中断它。
    pub(crate) fn background() -> Self {
        Self::Background
    }

    /// 本轮渲染是否已被请求取消。
    ///
    /// 内存序取 `Relaxed`：取消标志只用于"尽快退出"的软中断，不参与任何
    /// 数据同步，因此不需要与临界区建立 happens-before 关系。
    pub(crate) fn is_cancelled(&self) -> bool {
        match self {
            Self::Background => {
                crate::commands::playback::BG_RENDER_CANCEL.load(Ordering::Relaxed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 涉及全局 `BG_RENDER_CANCEL` / `CANCEL_EPOCH` 的用例必须串行执行：
    /// cargo test 默认并行跑用例，两个用例同时读写同一组全局状态会互相干扰。
    static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

    /// 复位全局取消状态，返回刚刚过去的纪元值。
    fn reset_global_cancel() -> u64 {
        crate::commands::playback::BG_RENDER_CANCEL.store(false, Ordering::Release);
        CANCEL_EPOCH.load(Ordering::Acquire)
    }

    /// 渲染令牌跟踪全局标志的原始值；纪元推进使新轮次不受历史残留影响。
    #[test]
    fn background_token_tracks_global_flag() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_global_cancel();

        let token = RenderCancelToken::background();
        assert!(!token.is_cancelled());

        request_global_cancel();
        assert!(token.is_cancelled());

        reset_global_cancel();
    }
}
