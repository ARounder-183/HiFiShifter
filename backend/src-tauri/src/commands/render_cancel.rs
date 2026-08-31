// 渲染取消令牌（RenderCancelToken）
//
// 职责：为"一轮渲染"提供彼此隔离的取消信号，避免不同渲染路径共用同一个
// 全局标志而互相污染。本模块只提供"取消信号"这一基础设施，不涉及渲染、
// 缓存或前端事件；`commands/playback.rs` 是它当前唯一的使用者。
//
// ── 为什么要独立成模块 ────────────────────────────────────────────────────
// `render_single_clip` 原先直接读取全局 `BG_RENDER_CANCEL`。该标志的
// 生命周期是不对称的：
//
//   * 置位：`cancel_background_render()` **无条件**置位，而它在
//     `new_project` / `open_project` 中必被调用（前端 thunk 也会各调一次），
//     因此即便当时没有任何渲染在跑（日志里表现为 `was_active=false`），
//     标志也会被置为 true。
//   * 清除：只在"后台预渲染"的启动与收尾路径上发生
//     （`start_background_render` 开头、各条退出分支）。
//
// 于是当"后台预渲染"未启用时 —— `request_background_render` 会在 disabled
// 分支直接返回、根本不清理标志 —— 打开工程后该标志便永久保持为 true。
// 之后每次前台 `play_original` 预渲染，都会在解码完成后的第一个检查点返回
// `bg_render_cancelled`，整轮渲染在几十毫秒内"失败"，播放随即降级为原声。
//
// 短片段之所以看起来正常，是因为它们命中了渲染缓存
// （`playback.rs` 中 `base_entry.is_none()` 才会调 `render_single_clip`），
// 根本不会走到取消检查；只有真正需要合成的片段（通常就是长音频）才会暴露。
//
// ── 现在的模型 ────────────────────────────────────────────────────────────
//   * 后台预渲染：继续跟踪全局 `BG_RENDER_CANCEL` 的原始值。它不需要纪元保护，
//     因为 `start_background_render` 在开头就会复位该标志。
//   * 前台播放预渲染：改用本轮私有的 `Arc<AtomicBool>`；对全局标志则通过
//     `CANCEL_EPOCH` 做"这次取消是否针对本轮"的判定，历史状态无法影响它。
//   * 工程切换/新建：由 `cancel_background_render` 显式调用
//     `cancel_all_foreground()` 通知在跑的前台轮次。
//
// 之所以还要看全局标志（而不是只看私有标志）：时间线编辑会在后台渲染运行期间
// 置位全局标志，此时并发的前台轮次同样应当立即作废。纪元机制让我们既能保留
// 这个语义，又不会被残留的标志误伤。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// 全局"取消请求"纪元：每产生一次新的取消请求就递增。
///
/// 单靠一个布尔标志无法回答"这次取消是在我这轮开始之前还是之后发出的"。
/// 后台渲染的清理与重启逻辑并不能保证标志一定被复位 —— 例如取消请求与渲染
/// 线程收尾相撞时（`was_active=true` 但线程已跑完清理），标志就会残留。
/// 纪元让每一轮渲染只需记录开始时的值即可准确判定，从而使"标志残留"
/// 这一整类问题在结构上不可能发生。
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

/// 全局取消标志为真，且该请求发生在本轮开始（纪元为 `start_epoch`）之后。
///
/// 内存序：先读标志再读纪元。即便某次读取错过了刚发生的置位，标志是
/// "粘滞"的，下一个检查点必然再次观察到，不会漏掉取消。
fn global_cancel_requested_since(start_epoch: u64) -> bool {
    crate::commands::playback::BG_RENDER_CANCEL.load(Ordering::Relaxed)
        && CANCEL_EPOCH.load(Ordering::Acquire) > start_epoch
}

/// 一轮渲染使用的取消令牌。
///
/// 由 `playback.rs` 在启动渲染时构造，再以 `&RenderCancelToken` 传入
/// `render_single_clip`，供其在各个耗时阶段之间做检查点判断。
pub(crate) enum RenderCancelToken {
    /// 后台预渲染：直接跟踪全局 `BG_RENDER_CANCEL` 的原始值。
    Background,

    /// 前台播放预渲染：本轮私有的取消标志 + 本轮开始时的纪元。
    Owned {
        flag: Arc<AtomicBool>,
        start_epoch: u64,
    },
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
            Self::Owned { flag, start_epoch } => {
                flag.load(Ordering::Relaxed) || global_cancel_requested_since(*start_epoch)
            }
        }
    }
}

/// 进行中的前台渲染轮次的取消标志登记表。
///
/// 存 `Weak` 而非 `Arc`：万一某轮渲染线程 panic 且未走到 `Drop`，
/// 条目也会随 `Arc` 释放而变为不可升级，并在下一次清理时被回收，
/// 不会让登记表无限增长。
static FOREGROUND_CANCELS: Mutex<Vec<Weak<AtomicBool>>> = Mutex::new(Vec::new());

/// 一轮前台渲染的取消控制块（RAII）。
///
/// 构造即登记、`Drop` 即注销，因此渲染线程的任何提前 `return` 或 panic
/// 都不会残留登记表条目。控制块必须活到该轮渲染结束。
pub(crate) struct ForegroundRenderCancel {
    flag: Arc<AtomicBool>,
    /// 本轮开始时的全局取消纪元，用于忽略本轮之前的历史取消请求。
    start_epoch: u64,
}

impl ForegroundRenderCancel {
    /// 登记一轮新的前台渲染，返回其控制块。
    pub(crate) fn register() -> Self {
        // ★ 纪元必须在登记之前读取：若顺序颠倒，与本函数并发发生的一次
        // 取消请求可能被漏判为"历史残留"。
        let start_epoch = CANCEL_EPOCH.load(Ordering::Acquire);
        let flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut registry) = FOREGROUND_CANCELS.lock() {
            Self::purge_dead(&mut registry);
            registry.push(Arc::downgrade(&flag));
        }
        Self { flag, start_epoch }
    }

    /// 生成本轮渲染使用的取消令牌。
    ///
    /// 令牌只是 `Arc` 的一个克隆，可以放心地在循环中反复生成。
    pub(crate) fn token(&self) -> RenderCancelToken {
        RenderCancelToken::Owned {
            flag: Arc::clone(&self.flag),
            start_epoch: self.start_epoch,
        }
    }

    /// 从登记表中移除本控制块对应的条目。
    ///
    /// ★ 不能只靠 `strong_count() == 0` 判断：`Drop::drop` 执行时 `self.flag`
    /// 尚未释放，它自己的 `Weak` 仍是"存活"的，仅按引用计数过滤会把自己留下，
    /// 导致登记表只增不减。因此这里按指针身份精确移除，并顺带清理失效条目。
    fn unregister(&self) {
        if let Ok(mut registry) = FOREGROUND_CANCELS.lock() {
            let self_ptr = Arc::as_ptr(&self.flag);
            registry.retain(|weak| {
                // 先按引用计数剔除已结束的轮次，再排除自己。
                weak.strong_count() > 0 && !std::ptr::eq(weak.as_ptr(), self_ptr)
            });
        }
    }

    /// 丢弃登记表中已失效（对应轮次已结束）的条目。
    fn purge_dead(registry: &mut Vec<Weak<AtomicBool>>) {
        registry.retain(|weak| weak.strong_count() > 0);
    }
}

impl Drop for ForegroundRenderCancel {
    fn drop(&mut self) {
        self.unregister();
    }
}

/// 向所有进行中的前台渲染发出取消请求。
///
/// 由 `playback::cancel_background_render` 在工程切换 / 新建时调用，
/// 使旧工程的前台预渲染也能及时退出。
pub(crate) fn cancel_all_foreground() {
    if let Ok(registry) = FOREGROUND_CANCELS.lock() {
        for weak in registry.iter() {
            if let Some(flag) = weak.upgrade() {
                flag.store(true, Ordering::Release);
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

    /// 回归用例：前台渲染令牌不能被全局标志的**历史**状态影响
    /// —— 这正是"打开工程后长音频渲染不出来"的根因。
    #[test]
    fn foreground_token_ignores_stale_global_cancel() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_global_cancel();

        // 模拟"打开工程时调用了取消、但后台预渲染未开启"留下的粘滞标志。
        request_global_cancel();

        let pass = ForegroundRenderCancel::register();
        assert!(
            !pass.token().is_cancelled(),
            "本轮开始之前的取消请求不应影响新开的前台渲染轮次"
        );

        // 而本轮开始之后发出的取消请求必须生效。
        request_global_cancel();
        assert!(
            pass.token().is_cancelled(),
            "本轮开始之后的全局取消请求应立即生效"
        );

        reset_global_cancel();
    }

    /// 工程切换时的显式通知必须能立即中断前台渲染轮次。
    #[test]
    fn cancel_all_foreground_reaches_live_pass() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_global_cancel();

        let pass = ForegroundRenderCancel::register();
        assert!(!pass.token().is_cancelled());

        cancel_all_foreground();
        assert!(
            pass.token().is_cancelled(),
            "cancel_all_foreground 后前台渲染轮次应立即判定为已取消"
        );

        reset_global_cancel();
    }

    /// 后台渲染令牌仍须跟踪全局标志的原始值。
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

    /// 控制块析构后应从登记表中移除，避免登记表只增不减。
    #[test]
    fn dropped_pass_is_removed_from_registry() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_global_cancel();

        let live = ForegroundRenderCancel::register();
        let dropped = ForegroundRenderCancel::register();
        drop(dropped);

        cancel_all_foreground();
        assert!(live.token().is_cancelled());

        let registry = FOREGROUND_CANCELS.lock().unwrap();
        assert_eq!(registry.len(), 1, "已析构的控制块应从登记表中移除");
    }
}
