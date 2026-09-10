// 渲染取消令牌（RenderCancelToken）
//
// 职责：为"一轮渲染"提供取消信号，避免渲染路径直接读写裸的全局标志。
// 本模块只提供"取消信号"这一基础设施，不涉及渲染、缓存或前端事件；
// `commands/playback.rs` 是它当前唯一的使用者。
//
// ── 模型 ──────────────────────────────────────────────────────────────────
// 前后台渲染已统一为同一套 pass 机制（是否启用后台预渲染只决定渲染的触发
// 时机），因此只有一种令牌：跟踪全局 `BG_RENDER_CANCEL`。
//
// ── "新轮次忽略历史残留"由谁保证 ────────────────────────────────────────────
// 由 `playback.rs` 的**代数守卫** `BG_RENDER_GENERATION` 保证：每轮渲染启动时
// 递增并记住自己的代数，退出路径只有在代数仍匹配时才允许清理全局标志 ——
// "取消旧渲染后立刻开新一轮、旧线程却把新状态清掉"这类交错因此不会发生。
//
// 本模块曾维护一个 `CANCEL_EPOCH` 计数器，并在注释里把它说成该保护的来源。
// 核实后：它**从未被任何生产代码读取**（只有本模块的测试读写），因此不提供
// 任何保护，真正起作用的是上面那个代数守卫。为避免"看起来有守卫"误导后续
// 修改（例如误以为必须推进纪元才生效，或随手加一个读点从而改变行为），该
// 计数器已删除，注释改为描述真实机制。
//
// ★ 仍然要求所有取消请求统一走 `request_global_cancel()`，而不要裸写
//   `BG_RENDER_CANCEL.store(true, ..)`：理由不是"推进纪元"，而是让"置位"只有
//   一个入口、便于审计（当前两个调用点都遵循此约定）。

use std::sync::atomic::Ordering;

/// 发出一次全局取消请求：置位 `BG_RENDER_CANCEL`。
///
/// ★ 所有希望中断"当前正在跑的渲染"的调用点都应走这个函数，而不是裸写
/// `BG_RENDER_CANCEL.store(true, ..)`：统一入口便于审计"置位与清除成对"这一
/// 生命周期契约。需要区分"取消的是哪一轮"时，用的是 `playback.rs` 的
/// `BG_RENDER_GENERATION` 代数守卫，而不是本函数。
pub(crate) fn request_global_cancel() {
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
    use std::sync::Mutex;

    /// 涉及全局 `BG_RENDER_CANCEL` 的用例必须串行执行：cargo test 默认并行跑
    /// 用例，两个用例同时读写同一全局状态会互相干扰。
    static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

    /// 复位全局取消状态。
    fn reset_global_cancel() {
        crate::commands::playback::BG_RENDER_CANCEL.store(false, Ordering::Release);
    }

    /// 渲染令牌跟踪全局标志的原始值。
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
