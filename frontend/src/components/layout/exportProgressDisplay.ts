/**
 * 导出进度条的显示推进（纯函数 + 常量）
 *
 * 【为什么单独成模块】显示推进的正确性此前依赖"事件节奏"：缓动计时器建在一个依赖
 * 事件派生值的 effect 里，事件一密（后端节流后约 50ms 一次）计时器就被反复重建，
 * 180ms 的周期永远够不到，显示值因而冻结。把推进规则抽成纯函数后：计时器只负责
 * **按固定节拍**调用它，规则本身可单测，也不再与事件频率耦合。
 *
 * 【两条保证】
 * 1. **正确性优先**：只要拿到过真实进度，显示值就单调跟随它（事件到达时立即推进，
 *    计时器只做事件之间的插值）；即便计时器因任何原因停摆，显示也不会偏离真实进度。
 * 2. **单调不减**：显示值永不回退（进度回退在导出语义里没有意义，回退会造成视觉抖动）。
 *
 * 【与其他模块的关系】
 * - 上游：`ExportAudioDialog` 的事件回调写入目标值（ref），计时器按
 *   {@link DISPLAY_TICK_MS} 调用 {@link nextDisplayProgress}。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 */

/** 显示推进节拍（毫秒）。≈8Hz：视觉足够平滑，且与后端约 50ms 的上报节流不同频。 */
export const DISPLAY_TICK_MS = 120;

/** 真实进度目标的收敛速度（每秒最多前进的百分点）。 */
export const CONVERGE_PERCENT_PER_SEC = 80;

/** 尚无真实进度时，"假进度"的爬升速度（每秒百分点）——刻意远慢于真实收敛。 */
export const FAKE_PROGRESS_PERCENT_PER_SEC = 8;

/**
 * "假进度"上限。
 *
 * 【为什么必须有上限】首个真实事件到达前，界面需要一个"在动"的观感；但这个自行
 * 爬升的值**绝不能**高于随后到达的真实进度，否则就要么显示停在真实位置之后（只升
 * 不降），要么被迫回退。封顶 90 留出足够余量：真实进度一旦到来就立即接管。
 */
export const FAKE_PROGRESS_CAP = 90;

/** 导出进度状态的形状（`ExportAudioDialog` 的事件派生状态；用于跳过无变化的更新）。 */
export interface ExportProgressSnapshot {
    readonly active: boolean;
    readonly mode: string | null;
    readonly progress: number | null;
    readonly current: number | null;
    readonly total: number | null;
}

/** 把后端进度（`0..1`）换算为百分比；非法值返回 `null`。 */
export function realProgressPercent(value: number | null): number | null {
    if (value === null || !Number.isFinite(value)) return null;
    return Math.max(0, Math.min(1, value)) * 100;
}

/**
 * 两份进度快照的字段是否逐一相等。
 *
 * 【用途】导出期间事件很密（约 50ms 一次），若每次都新建对象交给 `setState`，整个
 * 导出对话框都会重渲染。相等时返回 `true` 让调用方复用旧对象（React 会跳过提交）。
 *
 * @param a 旧快照。
 * @param b 新快照。
 * @returns 所有字段相等时为 `true`。
 */
export function isSameExportProgress<T extends ExportProgressSnapshot>(a: T, b: T): boolean {
    return (
        a.active === b.active &&
        a.mode === b.mode &&
        a.progress === b.progress &&
        a.current === b.current &&
        a.total === b.total
    );
}

/** {@link nextDisplayProgress} 入参。 */
export interface NextDisplayProgressArgs {
    /** 当前显示值（0..100）。 */
    readonly current: number;
    /** 后端最近一次真实进度（0..100）；`null` = 尚未收到任何真实进度。 */
    readonly target: number | null;
    /** 本帧距上一帧的时长（毫秒）。 */
    readonly tickMs: number;
}

/**
 * 计算显示进度的下一个取值。
 *
 * 规则（按序）：
 * 1. `current` 先归一化到 `[0, 100]`（防御非法值）；
 * 2. **有真实目标**：已到/超过目标 → 保持（不回退）；否则按
 *    {@link CONVERGE_PERCENT_PER_SEC} 逼近，**不越过目标**；
 * 3. **无真实目标**：按 {@link FAKE_PROGRESS_PERCENT_PER_SEC} 缓慢爬升，但不超过
 *    {@link FAKE_PROGRESS_CAP}；
 * 4. 结果恒不小于 `current`（单调不减）。
 *
 * @param args 见 {@link NextDisplayProgressArgs}。
 * @returns 下一显示值（0..100）。
 */
export function nextDisplayProgress(args: NextDisplayProgressArgs): number {
    const current = Number.isFinite(args.current) ? Math.max(0, Math.min(100, args.current)) : 0;
    const tickMs = Number.isFinite(args.tickMs) && args.tickMs > 0 ? args.tickMs : 0;

    const target = args.target;
    if (target !== null && Number.isFinite(target)) {
        const clampedTarget = Math.max(0, Math.min(100, target));
        // 已到目标（或目标回退）：保持，绝不回退。
        if (current >= clampedTarget) return current;
        const step = (CONVERGE_PERCENT_PER_SEC * tickMs) / 1000;
        return Math.min(clampedTarget, current + step);
    }

    // 尚无真实进度：缓慢爬升，且绝不越过封顶（真实进度一到立即接管）。
    if (current >= FAKE_PROGRESS_CAP) return current;
    const step = (FAKE_PROGRESS_PERCENT_PER_SEC * tickMs) / 1000;
    return Math.min(FAKE_PROGRESS_CAP, current + step);
}
