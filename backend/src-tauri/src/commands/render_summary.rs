//! 一次"工程加载"期间的渲染复用汇总。
//!
//! # 为什么需要单独一层汇总
//! 打开工程是**逐步加载**的：音高分析逐批完成 → 逐批解锁可渲染的 clip → 触发一轮
//! 又一轮后台渲染 pass。实测一个 465 片段的工程在打开过程中依次跑出
//! `6 clips / 35 clips / 137 clips / … / 465 clips` 十余轮 pass。
//!
//! 于是"逐轮上报"的数字对用户毫无意义：第 1 轮是 `6/6`、第 3 轮是 `5/35`、
//! 最后一轮才是 `156/465`。分母是**该轮处理的 clip 数**而不是工程规模，用户看到的
//! 是一串互相矛盾、且都不代表最终复用率的中途值。
//!
//! 本模块把口径固定成**工程级**：
//! - 分母 `project_total` = 本工程需要渲染的 clip 总数（不受音高分析进度影响）；
//! - 分子按 **clip 去重**累计整个加载过程的复用情况，跨 pass 重启保持。
//!
//! # 与 `render_cache` 会话统计的区别
//! `render_cache` 的 `session_*` 计数是**进程级**的（跨工程累加），用于设置面板的
//! 长期观察；本模块是**加载级**的，在打开/新建工程时重置，只服务于"这次打开省了
//! 多少"这一条提示。

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// 单个 clip 在一次加载中的结算结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipOutcome {
    /// 从磁盘渲染缓存复用（跨会话节省）。
    DiskHit,
    /// 本次真正重新合成。
    Rendered,
    /// 合成失败。
    Failed,
}

#[derive(Debug, Default)]
struct LoadSummary {
    /// 本工程需要渲染的 clip 总数。
    project_total: usize,
    /// `clip_id → 首次结论`。按 clip 去重：加载过程中的 pass 重启、以及编辑导致的
    /// 失效重渲，都不应让同一个 clip 被重复计数。
    resolved: HashMap<String, ClipOutcome>,
    disk_hits: u64,
    rendered: u64,
    failed: u64,
    /// 已计入 `rendered` 的合成耗时总和（毫秒），用于估算节省时间。
    render_ms: f64,
    /// 落盘准入的累计结果（跨 pass 相加）。
    persisted: u64,
    skipped: u64,
}

static SUMMARY: OnceLock<Mutex<LoadSummary>> = OnceLock::new();

fn summary() -> &'static Mutex<LoadSummary> {
    SUMMARY.get_or_init(|| Mutex::new(LoadSummary::default()))
}

/// 打开 / 新建工程时重置（由 `cancel_background_render` 调用 —— 它在两条路径上
/// 都无条件执行）。
pub fn reset() {
    *summary().lock().unwrap_or_else(|e| e.into_inner()) = LoadSummary::default();
}

/// 记录本工程需要渲染的 clip 总数。
///
/// 每轮 pass 都会上报同一个值（它只取决于时间线上需要 pitch edit 的 clip 集合，
/// 与音高分析进度无关）；取最大值兜底，避免某轮收集异常把分母改小。
pub fn set_project_total(total: usize) {
    let mut s = summary().lock().unwrap_or_else(|e| e.into_inner());
    s.project_total = s.project_total.max(total);
}

/// 记录一个 clip 的结算结果。
///
/// 同一个 clip 只计一次。唯一允许改写结论的情形是"先失败、后成功"——那说明重试
/// 生效了，如实记成已渲染。
pub fn note(clip_id: &str, outcome: ClipOutcome, render_elapsed: Duration) {
    let mut s = summary().lock().unwrap_or_else(|e| e.into_inner());
    match s.resolved.get(clip_id).copied() {
        Some(previous) => {
            if previous == ClipOutcome::Failed && outcome == ClipOutcome::Rendered {
                s.failed = s.failed.saturating_sub(1);
                s.rendered += 1;
                s.render_ms += render_elapsed.as_secs_f64() * 1000.0;
                s.resolved.insert(clip_id.to_string(), outcome);
            }
        }
        None => {
            match outcome {
                ClipOutcome::DiskHit => s.disk_hits += 1,
                ClipOutcome::Rendered => {
                    s.rendered += 1;
                    s.render_ms += render_elapsed.as_secs_f64() * 1000.0;
                }
                ClipOutcome::Failed => s.failed += 1,
            }
            s.resolved.insert(clip_id.to_string(), outcome);
        }
    }
}

/// 累计一轮 pass 的落盘准入结果。
pub fn add_admission(persisted: u64, skipped: u64) {
    let mut s = summary().lock().unwrap_or_else(|e| e.into_inner());
    s.persisted = s.persisted.saturating_add(persisted);
    s.skipped = s.skipped.saturating_add(skipped);
}

/// 上报给前端的工程级汇总快照。
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderLoadSummary {
    /// 本工程需要渲染的 clip 总数（分母）。
    pub project_total: u64,
    /// 从磁盘缓存复用的 clip 数（分子）。
    pub disk_hits: u64,
    /// 本次重新合成的 clip 数。
    pub rendered: u64,
    /// 合成失败的 clip 数。
    pub failed: u64,
    /// 通过落盘准入的条目数（累计）。
    pub persisted: u64,
    /// 被拒绝落盘的条目数（累计）。
    pub skipped: u64,
    /// 估算节省的合成时间（毫秒）。
    pub saved_ms: u64,
}

/// 取快照。
pub fn snapshot() -> RenderLoadSummary {
    let s = summary().lock().unwrap_or_else(|e| e.into_inner());
    // 用"本轮真实合成的平均耗时"估算复用省下的时间：全命中时无样本，估 0。
    let avg_render_ms = if s.rendered > 0 {
        s.render_ms / s.rendered as f64
    } else {
        0.0
    };
    RenderLoadSummary {
        project_total: s.project_total as u64,
        disk_hits: s.disk_hits,
        rendered: s.rendered,
        failed: s.failed,
        persisted: s.persisted,
        skipped: s.skipped,
        saved_ms: (avg_render_ms * s.disk_hits as f64).round() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试之间共享同一份全局状态，必须串行。
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn with_fresh_summary<T>(f: impl FnOnce() -> T) -> T {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let out = f();
        reset();
        out
    }

    /// 同一个 clip 跨多轮 pass 只计一次 —— 这正是"逐轮上报"会算错的地方。
    #[test]
    fn repeated_resolutions_of_one_clip_count_once() {
        with_fresh_summary(|| {
            set_project_total(465);
            note("clip-a", ClipOutcome::DiskHit, Duration::ZERO);
            note("clip-a", ClipOutcome::DiskHit, Duration::ZERO);
            note("clip-a", ClipOutcome::Rendered, Duration::from_millis(50));

            let snap = snapshot();
            assert_eq!(snap.disk_hits, 1);
            assert_eq!(snap.rendered, 0, "首次结论是命中，后续重算不得改判");
            assert_eq!(snap.project_total, 465);
        });
    }

    /// 先失败后成功要如实改判（重试生效）。
    #[test]
    fn a_later_success_upgrades_a_previous_failure() {
        with_fresh_summary(|| {
            note("clip-a", ClipOutcome::Failed, Duration::ZERO);
            assert_eq!(snapshot().failed, 1);

            note("clip-a", ClipOutcome::Rendered, Duration::from_millis(200));
            let snap = snapshot();
            assert_eq!(snap.failed, 0);
            assert_eq!(snap.rendered, 1);
        });
    }

    /// 分母取最大值：某轮收集异常报小了也不能把工程规模改小。
    #[test]
    fn project_total_never_shrinks() {
        with_fresh_summary(|| {
            set_project_total(465);
            set_project_total(6);
            assert_eq!(snapshot().project_total, 465);
        });
    }

    /// 节省时间用"真实合成的平均耗时 × 复用条数"估算；没有合成样本时为 0。
    #[test]
    fn saved_ms_uses_the_average_real_render_cost() {
        with_fresh_summary(|| {
            // 无合成样本 → 无从估算。
            note("clip-a", ClipOutcome::DiskHit, Duration::ZERO);
            assert_eq!(snapshot().saved_ms, 0);

            // 两条各 100 ms 的合成 → 均摊 100 ms；复用 1 条 → 省 100 ms。
            note("clip-b", ClipOutcome::Rendered, Duration::from_millis(100));
            note("clip-c", ClipOutcome::Rendered, Duration::from_millis(100));
            assert_eq!(snapshot().saved_ms, 100);
        });
    }

    /// 重置必须清空全部累计值（工程切换后不得串味）。
    #[test]
    fn reset_clears_everything() {
        with_fresh_summary(|| {
            set_project_total(465);
            note("clip-a", ClipOutcome::DiskHit, Duration::ZERO);
            note("clip-b", ClipOutcome::Rendered, Duration::from_millis(10));
            add_admission(2, 3);

            reset();
            let snap = snapshot();
            assert_eq!(snap.project_total, 0);
            assert_eq!(snap.disk_hits, 0);
            assert_eq!(snap.rendered, 0);
            assert_eq!(snap.persisted, 0);
            assert_eq!(snap.skipped, 0);
        });
    }

    /// 落盘准入计数跨 pass 相加。
    #[test]
    fn admission_counts_accumulate_across_passes() {
        with_fresh_summary(|| {
            add_admission(2, 3);
            add_admission(5, 0);
            let snap = snapshot();
            assert_eq!(snap.persisted, 7);
            assert_eq!(snap.skipped, 3);
        });
    }
}
