//! 渲染进度追踪器：一轮渲染 pass 的跨处理器共享进度通道。
//!
//! 本模块把进度拆成两个正交来源，对所有渲染器一视同仁：
//! - **clip 级**（权威）：渲染循环每完成一个 clip（命中缓存或新渲染入库）
//!   调用 [`advance_clip`]，进度 = 已完成 clip 数 / 本轮 clip 总数；
//! - **clip 内**（细化）：处理器在单个 clip 的渲染过程中上报 `local ∈ [0,1]`，
//!   进度 = (已完成 + local) / 总数。NSF-HiGAN 按 mel chunk、WORLD 按 6s
//!   合成块各自上报。
//!
//! # 声道扇出：为什么 clip 内进度必须先"折叠"再上报
//!
//! 真立体声（Normal/Swap + 双声道源）时，处理器链会被**逐声道扇出**调用两次
//! （见 `pitch_editing::maybe_apply_pitch_edit_to_clip_segment` 的说明）。而处理器
//! 的上报点（mel chunk / WORLD 合成块）分母只含**本声道**的块数 —— 它不知道
//! 自己是两个单元之一。
//!
//! 若把这种"单元内进度"直接当成"clip 内进度"，整体进度就会走两遍：
//!
//! ```text
//! 真立体声（2 单元），假设本轮共 2 个 clip：
//!   第 1 声道 local 0 → 1  ⇒ 进度 0%   → 50%
//!   第 2 声道 local 0 → 1  ⇒ 进度 50%  → 100%   ← 又一遍
//!   done += 1（整个 clip 完成）
//! ```
//!
//! 用户看到的就是"进度条跑两遍，第二遍跑完才算完"。修复办法是让**调用方**
//!（知道扇出倍数的那一层）用 [`ClipUnitGuard`] 声明"当前处在第几个单元、共几个
//! 单元"，处理器侧改调 [`report_unit_progress_current`]，由本模块把单元内进度
//! 折叠到 clip 内 `[0,1]`。
//!
//! 【为什么折叠公式不写在声码器里】声码器拿不到扇出倍数（`ClipProcessContext`
//! 只有 `channel_index`，没有总数）。让声码器写 `/ 2` 就是把"调用方的扇出倍数"
//! 这个属于调用方的知识泄漏进声码器 —— 一旦将来出现 3 单元路径（A/B 变体对比）
//! 就是第二份会漂移的实现。
//!
//! # 单调性：进度只能前进
//!
//! 所有出口都经过 [`gate_monotonic`]：同一轮 pass 内进度**单调不减**。进度回退
//! 对用户是纯粹的故障信号（"渲染到 80% 又回到 0"），而回退的来源（取消后重启、
//! 旧代线程迟到、多单元折叠顺序）都在后端，前端无法可靠区分"新一轮"与"同一轮
//! 回退"。闸门因此设在这里 —— 唯一出口，无法绕过。

/// 进度回调：参数为整体进度（0.0 ~ 1.0）。
pub type ProgressCallback = Box<dyn Fn(f64) + Send + Sync>;

static CALLBACK: OnceLock<Mutex<Option<ProgressCallback>>> = OnceLock::new();
/// 本轮渲染 pass 的 clip 总数（含命中缓存的 clip）。
static TOTAL_CLIPS: OnceLock<std::sync::atomic::AtomicUsize> = OnceLock::new();
/// 已完成（处理出结果）的 clip 数：命中缓存或新渲染入库各计 1。
static DONE_CLIPS: OnceLock<std::sync::atomic::AtomicUsize> = OnceLock::new();

/// 本轮已发出的最大进度（f64 位模式），见 [`gate_monotonic`]。
static EMITTED_MAX_BITS: OnceLock<std::sync::atomic::AtomicU64> = OnceLock::new();

/// `f64::NEG_INFINITY` 的位模式。
///
/// 【为什么用字面量而不是 `f64::NEG_INFINITY.to_bits()`】`to_bits` 在较旧的编译器
/// 上不是 `const fn`，而这里需要 `static` 初始化。写位模式省掉一个仅为初始化
/// 服务的 MSRV 约束。
///
/// 【为什么初值是 −∞ 而不是 0】闸门规则是"不高于已发最大值即丢弃"，初值取 0 会让
/// **第一次上报 0.0 被误拦**（0.0 不大于 0.0）—— 于是"刚开始渲染"那一帧的 0%
/// 事件永远发不出去。
const NEG_INFINITY_BITS: u64 = 0xFFF0_0000_0000_0000;

use std::cell::Cell;
use std::sync::{Mutex, OnceLock};

fn callback_slot() -> &'static Mutex<Option<ProgressCallback>> {
    CALLBACK.get_or_init(|| Mutex::new(None))
}

fn total_slot() -> &'static std::sync::atomic::AtomicUsize {
    TOTAL_CLIPS.get_or_init(|| std::sync::atomic::AtomicUsize::new(0))
}

fn done_slot() -> &'static std::sync::atomic::AtomicUsize {
    DONE_CLIPS.get_or_init(|| std::sync::atomic::AtomicUsize::new(0))
}

fn emitted_max_slot() -> &'static std::sync::atomic::AtomicU64 {
    EMITTED_MAX_BITS.get_or_init(|| std::sync::atomic::AtomicU64::new(NEG_INFINITY_BITS))
}

// ─── 单元作用域（声道扇出）────────────────────────────────────────────────────

thread_local! {
    /// 当前线程所处的"clip 内单元"：`(单元下标, 单元总数)`。
    ///
    /// 用 `Cell` 保存/恢复而不是 `Vec` 栈：`ClipUnitGuard` 构造时把旧值换出来、
    /// `Drop` 时换回去，天然支持嵌套且零分配。渲染是同步调用链
    ///（`process()` → 声码器 → 上报），同一线程内不会出现并发交错。
    static CURRENT_UNIT: Cell<Option<(u32, u32)>> = const { Cell::new(None) };
}

/// 「当前 clip 的第 i 个单元，共 n 个」的作用域守卫。
///
/// 由**知道扇出倍数的那一层**（`pitch_editing` 的声道扇出循环）在每个
/// `processor.process()` 之前构造，`Drop` 时自动恢复外层值：
///
/// ```ignore
/// for channel in 0..fanout_channels {
///     let _unit = ClipUnitGuard::enter(channel, fanout_channels);
///     let out = processor.process(&ctx)?;   // 内部上报会落到本单元的时间片
///     outs.push(out);
/// }
/// ```
///
/// 【为什么用守卫而不是 begin/end 两个函数】单元作用域与 `process()` 调用在
/// **同一段代码**里成对出现，语法上不可能忘记复位；`?` 提前返回与 panic
///（渲染循环另有 `catch_unwind` 兜底）都会执行 `Drop`。
pub struct ClipUnitGuard {
    previous: Option<(u32, u32)>,
}

impl ClipUnitGuard {
    /// 进入第 `unit_index` 个单元（共 `units` 个）。
    ///
    /// `units` 归一为 ≥1、`unit_index` 夹到 `[0, units-1]`：非法下标不应产生大于 1
    /// 的进度（那会让进度条溢出），也不应静默变成"第 0 个单元"（那会让最后一个
    /// 单元的进度被重复计入）。
    pub fn enter(unit_index: usize, units: usize) -> Self {
        let units = units.clamp(1, u32::MAX as usize) as u32;
        let index = unit_index.min(units as usize - 1) as u32;
        let previous = CURRENT_UNIT.with(|cell| cell.replace(Some((index, units))));
        Self { previous }
    }
}

impl Drop for ClipUnitGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        CURRENT_UNIT.with(|cell| cell.set(previous));
    }
}

/// 把一个 clip 内部的单元进度折叠为 clip 级 `[0,1]`。
///
/// `units == 1` 时逐值等于输入 —— 等效单声道（`fanout_channels == 1`）与旧实现
/// 完全一致，单声道工作流零行为变化。
///
/// 【为什么这里不处理 NaN】非有限值不是进度值，由 [`report_unit_progress_current`]
/// 在入口直接丢弃（见那里的说明）。`clamp` 对 NaN 返回 NaN，因此即便有漏网的
/// NaN 进来也会被下游的 [`gate_monotonic`] 拦下 —— 三层防御，但只有入口那一层
/// 是"语义"，另两层是兜底。
fn fold_unit_progress(unit_index: usize, units: usize, local: f64) -> f64 {
    let local = local.clamp(0.0, 1.0);
    let units = units.max(1);
    if units == 1 {
        return local;
    }
    let index = unit_index.min(units - 1);
    let span = 1.0 / units as f64;
    ((index as f64) + local) * span
}

// ─── 单调闸门 ─────────────────────────────────────────────────────────────────

/// 单调闸门：只放行**严格高于**本轮已发最大值的进度。
///
/// 返回 `None` 表示本次上报没有新信息（与已发最大值相同或更低），调用方应跳过
/// 事件发送 —— 既省一次 IPC，也让"进度回退"在源头消失。
///
/// 【为什么用 CAS 而不是"先读后写"】多单元 / 多阶段的上报来自不同代码路径，
/// 两次 `load`/`store` 之间可能交错，导致高水位被较低的值覆盖，闸门随即失效。
fn gate_monotonic(raw: f64) -> Option<f64> {
    use std::sync::atomic::Ordering;
    if !raw.is_finite() {
        return None;
    }
    let value = raw.clamp(0.0, 1.0);
    let slot = emitted_max_slot();
    let mut prev_bits = slot.load(Ordering::Relaxed);
    loop {
        let prev = f64::from_bits(prev_bits);
        if value <= prev {
            return None;
        }
        match slot.compare_exchange_weak(
            prev_bits,
            value.to_bits(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Some(value),
            Err(actual) => prev_bits = actual,
        }
    }
}

// ─── 对外 API ─────────────────────────────────────────────────────────────────

/// 设置/清除进度回调。每轮渲染 pass 启动时注册，收尾时清除。
pub fn set_callback(cb: Option<ProgressCallback>) {
    *callback_slot().lock().unwrap_or_else(|e| e.into_inner()) = cb;
}

/// 开始新一轮 pass：记录 clip 总数、把已完成计数与单调高水位一并归零。
///
/// 【为什么高水位也要归零】新的一轮是从 0% 重新开始的一次独立渲染；保留上一轮的
/// 100% 会让整轮新进度被闸门全部丢弃 —— 状态栏停在"渲染中 100%"直到本轮结束。
/// 跨轮回落因此是**允许**的，而它也正是"又开始了新一批"的正确反馈（前端另有
/// `pass` 序号用于识别这一跃迁，见 `App.tsx` 的监听器）。
pub fn reset(total_clips: usize) {
    use std::sync::atomic::Ordering;
    total_slot().store(total_clips, Ordering::Relaxed);
    done_slot().store(0, Ordering::Relaxed);
    emitted_max_slot().store(NEG_INFINITY_BITS, Ordering::Relaxed);
}

/// 渲染循环在每个 clip 边界调用：已完成计数 +1。
///
/// 只推进计数，不发射事件 —— 是否可见（进度条是否已点亮）由渲染循环的
/// `rendering_started` 门控决定，缓存全命中的 pass 不应闪进度条。
pub fn advance_clip() {
    done_slot().fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// 当前进度（已完成 clip 数 / 总数）；总数未知时返回 0。
pub fn current_fraction() -> f64 {
    let total = total_slot().load(std::sync::atomic::Ordering::Relaxed);
    if total == 0 {
        return 0.0;
    }
    let done = done_slot().load(std::sync::atomic::Ordering::Relaxed);
    (done as f64 / total as f64).clamp(0.0, 1.0)
}

/// clip 级进度，经单调闸门过滤。
///
/// 返回 `None` 表示本轮进度没有前进（例如 `advance_clip` 后数值与上一条事件
/// 相同），调用方**应跳过事件发送**。
///
/// 【为什么 clip 级也要过闸门】闸门是"已发最大值"的唯一记账处。若 clip 级上报
/// 绕过它，clip 内上报就会以陈旧的低水位为基准 —— 于是 `advance_clip` 之后紧接
/// 着的单元上报会把进度**发回**到 clip 内部的比例，正是要消除的那种回落。
pub fn gated_fraction() -> Option<f64> {
    gate_monotonic(current_fraction())
}

/// 处理器在**当前单元内**上报进度（`local ∈ [0,1]`）。
///
/// 声道扇出时，调用方（`pitch_editing` 的扇出循环）须已用 [`ClipUnitGuard`]
/// 声明当前单元，本函数把单元内进度折叠成 clip 级 `[0,1]`；未声明时按"单单元"
/// 处理 —— 等效单声道、导出路径等无扇出场景因此不必额外做任何事。
///
/// 【非有限值直接丢弃，而不是当作 0】NaN / ±∞ 不是"进度为 0"的意思，而是上报方
/// 算错了。把它们当 0 会让进度条**倒退到本 clip 的起点**（正是本次要修的症状），
/// 而丢弃的代价只是少一条进度事件 —— 下一条正常上报会补上。
pub fn report_unit_progress_current(local: f64) {
    if !local.is_finite() {
        return;
    }
    let (index, units) = CURRENT_UNIT.with(|cell| cell.get()).unwrap_or((0, 1));
    emit_clip_local(fold_unit_progress(index as usize, units as usize, local));
}

/// 上报入口的唯一实现：算整体进度 → 过闸门 → 回调。
///
/// 无回调时（导出路径、非渲染 pass 上下文）是空操作，开销只有一次存在性检查。
fn emit_clip_local(clip_local: f64) -> Option<f64> {
    // 回调不存在时**不记账**：闸门的高水位属于"已发给前端的进度"，没有前端可发
    // 时推进它只会让本轮首次真实上报被误拦。
    let slot = CALLBACK.get()?;
    let guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    let cb = guard.as_ref()?;

    let total = total_slot().load(std::sync::atomic::Ordering::Relaxed);
    let done = done_slot().load(std::sync::atomic::Ordering::Relaxed);
    let progress = if total > 0 {
        ((done as f64 + clip_local.clamp(0.0, 1.0)) / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let gated = gate_monotonic(progress)?;
    cb(gated);
    Some(gated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    /// 全局进度状态的用例必须串行执行，避免并行用例互相污染计数器。
    static GLOBAL_LOCK: StdMutex<()> = StdMutex::new(());

    fn reset_for_test(total: usize) {
        set_callback(None);
        reset(total);
    }

    /// 注册收集器并返回累积到的事件序列。
    fn collect_progress() -> Arc<StdMutex<Vec<f64>>> {
        let seen: Arc<StdMutex<Vec<f64>>> = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        set_callback(Some(Box::new(move |p| sink.lock().unwrap().push(p))));
        seen
    }

    fn events(seen: &Arc<StdMutex<Vec<f64>>>) -> Vec<f64> {
        seen.lock().unwrap().clone()
    }

    #[test]
    fn clip_progress_is_fraction_of_total() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(4);

        assert_eq!(current_fraction(), 0.0);
        advance_clip();
        assert_eq!(current_fraction(), 0.25);
        advance_clip();
        advance_clip();
        assert_eq!(current_fraction(), 0.75);

        set_callback(None);
    }

    #[test]
    fn local_progress_combines_done_and_local() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(5);

        let seen = collect_progress();

        advance_clip(); // 第 1 个 clip 完成（done=1）
        report_unit_progress_current(0.0); // 第 2 个 clip 刚开始 → (1+0)/5
        report_unit_progress_current(0.5);
        report_unit_progress_current(1.0); // 第 2 个 clip 收尾 → (1+1)/5

        assert_eq!(events(&seen), vec![0.2, 0.3, 0.4]);

        set_callback(None);
    }

    #[test]
    fn local_progress_clamps_out_of_range_input() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(2);

        let seen = collect_progress();

        report_unit_progress_current(-1.0); // 钳到 0 → (0+0)/2
        report_unit_progress_current(2.0); // 钳到 1 → (0+1)/2
        assert_eq!(events(&seen), vec![0.0, 0.5]);

        set_callback(None);
    }

    #[test]
    fn without_callback_report_is_noop_and_zero_total_is_safe() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(0);

        // total=0：不 panic，进度恒为 0。
        report_unit_progress_current(0.5);
        assert_eq!(current_fraction(), 0.0);

        // 无回调：report 不 panic（导出路径没有注册回调）。
        reset_for_test(3);
        report_unit_progress_current(0.5);
        set_callback(None);
    }

    /// 单元折叠的算术契约：单元内 `[0,1]` 映射到 clip 内的对应时间片。
    #[test]
    fn unit_folding_maps_each_unit_onto_its_slice() {
        // 单单元：逐值等于输入（等效单声道与旧实现一致）。
        assert_eq!(fold_unit_progress(0, 1, 0.0), 0.0);
        assert_eq!(fold_unit_progress(0, 1, 0.37), 0.37);
        assert_eq!(fold_unit_progress(0, 1, 1.0), 1.0);

        // 双单元（真立体声）：两个声道各占一半，且首尾相接。
        assert_eq!(fold_unit_progress(0, 2, 0.0), 0.0);
        assert_eq!(fold_unit_progress(0, 2, 1.0), 0.5);
        assert_eq!(fold_unit_progress(1, 2, 0.0), 0.5);
        assert_eq!(fold_unit_progress(1, 2, 1.0), 1.0);

        // 三单元：均匀切片。
        assert!((fold_unit_progress(1, 3, 0.5) - 0.5).abs() < 1e-12);
        assert!((fold_unit_progress(2, 3, 1.0) - 1.0).abs() < 1e-12);
    }

    /// 越界输入不产生溢出 / 静默错位：整体必须始终落在 `[0,1]`。
    #[test]
    fn unit_folding_clamps_degenerate_inputs() {
        // 单元下标越界 → 夹到末单元（不是第 0 单元：否则末单元进度被重复计入）。
        assert_eq!(fold_unit_progress(9, 2, 1.0), 1.0);
        assert_eq!(fold_unit_progress(9, 2, 0.0), 0.5);
        // units = 0 → 归一为 1。
        assert_eq!(fold_unit_progress(0, 0, 0.4), 0.4);
        // local 越界 → 钳到 [0,1]。
        assert_eq!(fold_unit_progress(0, 2, -5.0), 0.0);
        assert_eq!(fold_unit_progress(1, 2, 5.0), 1.0);
        assert_eq!(fold_unit_progress(1, 2, f64::NEG_INFINITY), 0.5);
        assert_eq!(fold_unit_progress(1, 2, f64::INFINITY), 1.0);
        // NaN 穿透本函数（`clamp` 对 NaN 返回 NaN），由上报入口与闸门拦下。
        assert!(fold_unit_progress(1, 2, f64::NAN).is_nan());
    }

    /// 回归锁：**真立体声的双声道上报必须拼出一条单调的 0→1**。
    ///
    /// 这正是用户报告的缺陷 —— 修复前两个声道各自 0→1，进度条跑两遍
    ///（0%→50%→100%，再 50%→100%）。
    #[test]
    fn stereo_channels_yield_one_monotonic_sweep() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(1);

        let seen = collect_progress();

        for channel in 0..2usize {
            let _unit = ClipUnitGuard::enter(channel, 2);
            for step in 0..=4 {
                report_unit_progress_current(step as f64 / 4.0);
            }
        }
        advance_clip();
        let _ = gated_fraction();

        let values = events(&seen);
        // 两个声道各 5 个采样点，但**声道交界处**的值（0.5）两边都会报一次，
        // 被严格递增的闸门合并成一条 —— 于是 10 次上报得到 9 条事件。
        // 这正是"两个声道之间不应出现停顿或重复"的体现。
        assert_eq!(values.len(), 9, "交界处的重复值应被闸门合并，实际 {values:?}");
        assert_eq!(*values.first().unwrap(), 0.0, "应从 0% 开始");
        assert_eq!(*values.last().unwrap(), 1.0, "最终必须到达 100%");
        for pair in values.windows(2) {
            assert!(
                pair[1] > pair[0],
                "同一轮 pass 内进度必须严格递增，实际出现 {:?} → {:?}（全序列 {:?}）",
                pair[0],
                pair[1],
                values
            );
        }

        set_callback(None);
    }

    /// 等效单声道的行为必须与旧实现逐值一致（零回归）。
    #[test]
    fn mono_clip_matches_legacy_report_path() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(1);

        let seen = collect_progress();
        let _unit = ClipUnitGuard::enter(0, 1);
        report_unit_progress_current(0.0);
        report_unit_progress_current(0.25);
        report_unit_progress_current(0.5);

        assert_eq!(events(&seen), vec![0.0, 0.25, 0.5]);
        set_callback(None);
    }

    /// 守卫是栈式的：内层退出后外层单元恢复，嵌套不会串味。
    #[test]
    fn unit_guard_restores_previous_scope() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(1);

        let seen = collect_progress();
        {
            let _outer = ClipUnitGuard::enter(0, 2);
            report_unit_progress_current(1.0); // 0.5
            {
                let _inner = ClipUnitGuard::enter(1, 2);
                report_unit_progress_current(0.0); // 0.5（与上一条相等 → 闸门丢弃）
            }
            // 回到外层：仍按 (0, 2) 折叠 → 0.5 → 丢弃。
            report_unit_progress_current(1.0);
        }
        // 守卫全部退出后回到"单单元"：0.75 直接就是 0.75。
        report_unit_progress_current(0.75);

        assert_eq!(events(&seen), vec![0.5, 0.75]);
        set_callback(None);
    }

    /// 单调闸门：同轮内的回落值被丢弃，新 pass 允许重新起始。
    #[test]
    fn monotonic_gate_drops_rollback_and_resets_per_pass() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(1);

        let seen = collect_progress();
        report_unit_progress_current(0.8);
        report_unit_progress_current(0.3); // 回落 → 丢弃
        report_unit_progress_current(0.8); // 相等 → 丢弃
        report_unit_progress_current(0.9);
        assert_eq!(events(&seen), vec![0.8, 0.9]);

        // 新一轮 pass：高水位归零，进度允许从低位重新开始。
        reset(1);
        report_unit_progress_current(0.1);
        assert_eq!(events(&seen), vec![0.8, 0.9, 0.1]);

        set_callback(None);
    }

    /// 非有限进度一律丢弃：既不产生事件，也不污染高水位。
    ///
    /// 【为什么这条重要】NaN 一旦进了闸门，`value <= prev` 恒为 false，闸门会
    /// 被"永久放行"；而把 NaN 当成 0 又会让进度条倒退到本 clip 起点 ——
    /// 正是本次要修的症状。两种坏结果都靠"入口直接丢弃"避免。
    #[test]
    fn non_finite_progress_is_dropped_without_poisoning_the_gate() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(1);

        let seen = collect_progress();
        report_unit_progress_current(f64::NAN);
        report_unit_progress_current(f64::INFINITY);
        report_unit_progress_current(f64::NEG_INFINITY);
        assert_eq!(events(&seen), Vec::<f64>::new(), "非有限值不应产生任何事件");

        // 闸门高水位未被污染：后续有限值按正常规则放行 / 拦截。
        report_unit_progress_current(0.25);
        report_unit_progress_current(0.1); // 回落 → 丢弃
        assert_eq!(events(&seen), vec![0.25]);

        set_callback(None);
    }

    /// 无回调时不推进高水位：本轮首次真实上报不能被误拦。
    #[test]
    fn reports_without_callback_do_not_advance_the_gate() {
        let _guard = GLOBAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test(1);

        set_callback(None);
        report_unit_progress_current(1.0); // 无回调：不记账

        let seen = collect_progress();
        report_unit_progress_current(0.25);
        assert_eq!(events(&seen), vec![0.25]);

        set_callback(None);
    }
}
