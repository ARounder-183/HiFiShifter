/**
 * 时间轴内核 · 轨道头「窗口化滚动位置」的量化提交。
 *
 * 【主要内容】
 * 给出轨道头（`TrackList`）行窗口化所用的 `scrollTop` **提交步长**，以及"本帧是否
 * 值得提交"的判定。
 *
 * 【作用：为什么需要它——这是纵向拖动残留顿挫的根因】
 * 轨道头容器在内核模式下每帧都被镜像回写，其原生 `scroll` 事件因此**每帧都触发**。
 * 事件处理里直接 `setListScrollTop(el.scrollTop)` 会让 `TrackList` 每帧重渲染一次。
 *
 * 实测（1920×1200，80 步拖竖向滚动条）：
 * - React 提交 **130 次**（约每帧一次）；
 * - 阻断该事件后提交降到 **1 次**，帧时间最大值从 **49.9ms → 16.8ms**、>25ms 的
 *   长帧从 1 个降到 0 个。
 *
 * 也就是说：滚动**位置**已经是准的（无回退），残留的是"每帧重渲染"造成的顿挫——
 * 用户感受为「改善很多但还有一点吸附感」。
 *
 * 【为什么可以量化：窗口化自带 overscan 余量】
 * 该 state 的唯一消费者是 `computeVisibleTrackWindow`（见 `TrackList`），它按
 * `overscanRows` 在可视区间上下各多渲染若干行。因此渲染所用的 `scrollTop` 允许
 * **滞后**，只要滞后不超过 overscan 预算，窗口就仍然覆盖真正可见的行。
 *
 * 推导（`rh` = 行高，`K` = overscanRows，阈值 `step`，滞后 `δ`）：
 * 以 `S' = S − δ` 计算窗口时，下边界为
 * `floor(S'/rh) + ceil(vh/rh) + K ≥ floor(S/rh) − ceil(δ/rh) + vh/rh + K`，
 * 要覆盖真正可见的最后一行 `floor((S + vh)/rh)`，只需 `K ≥ ceil(δ/rh)`，
 * 即 **`δ ≤ K × rh`**。上边界恒成立（滞后只会让窗口起点更高，不会漏顶部）。
 *
 * 【为什么取安全预算的一半】
 * 上述是**临界**条件（`δ = K × rh` 时刚好卡住边界）。取一半留出 2× 余量，
 * 使边界行不会因为 rAF 时序抖动而闪缺。
 *
 * 【与其他模块的关系】
 * - 上游：`TrackList` 的原生 `onScroll` 每帧拿到新的 `el.scrollTop`。
 * - 消费者：判定结果决定是否 `setListScrollTop`（驱动行窗口化）。
 * - 同源先例：`TimelinePanel` 的 `commitTimelineScrollTop` 用的是同一套
 *   「rAF + 步长」思路（那里的 overscan 更大，故步长取 2 行）。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 */

/** 量化步长的入参。 */
export interface ScrollCommitStepArgs {
    /** 行高（CSS px）。 */
    readonly rowHeight: number;
    /** 窗口化的 overscan 行数（`computeVisibleTrackWindow` 的 `overscanRows`）。 */
    readonly overscanRows: number;
}

/**
 * 解析量化提交步长（CSS px）。
 *
 * 流程：安全预算 `overscanRows × rowHeight` → 取一半 → 下限钳到 1px。
 *
 * 特殊说明 1：**下限 1px 是必需的**。步长为 0 会让"是否跨过阈值"恒为真，
 * 每帧都提交，量化完全失效（且可能形成无限 rAF 循环）。
 *
 * 特殊说明 2：`rowHeight` / `overscanRows` 非法（非有限值、负数）时按下限处理，
 * 不抛异常——窗口化是渲染路径，不能因为上游量测异常而中断。
 *
 * @param args 行高与 overscan 行数。
 * @returns 提交步长（CSS px，≥ 1）。
 */
export function resolveScrollCommitStepPx(args: ScrollCommitStepArgs): number {
    const rowHeight = Number.isFinite(args.rowHeight) ? Math.max(0, args.rowHeight) : 0;
    const overscanRows = Number.isFinite(args.overscanRows) ? Math.max(0, args.overscanRows) : 0;
    // 安全预算的一半（见文件头推导）；下限 1px 保证阈值判定始终有效。
    return Math.max(1, (rowHeight * overscanRows) / 2);
}

/** 提交判定入参。 */
export interface ShouldCommitScrollArgs {
    /** 上一次真正提交给 React 的位置。 */
    readonly committedPx: number;
    /** 本帧的最新位置。 */
    readonly nextPx: number;
    /** 提交步长（来自 `resolveScrollCommitStepPx`）。 */
    readonly stepPx: number;
}

/**
 * 判定本帧是否应当把新位置提交给 React。
 *
 * 流程：任一入参非有限 → 不提交（保持既有渲染，避免 NaN 传染）；否则比较
 * 与上次提交值的差是否达到步长。
 *
 * 特殊说明 1：判据是「与**上次提交值**比较」，不是"与上一帧比较"。后者会让每帧
 * 的小位移都被判定为未达阈值而**永远不提交**（窗口化彻底停更），前者则保证滞后
 * 累积到步长即提交。
 *
 * 特殊说明 2：本函数**不做钳制**，也不关心方向——向后滚动同样适用（推导对称）。
 *
 * @param args 见 `ShouldCommitScrollArgs`。
 * @returns 应当提交时为 true。
 */
export function shouldCommitScroll(args: ShouldCommitScrollArgs): boolean {
    const step = args.stepPx;
    const committed = args.committedPx;
    const next = args.nextPx;
    // 显式守卫：语义上是"入参非法 → 不提交"，读代码时不必依赖 NaN 的比较语义。
    //
    // 特殊说明：**本守卫在行为上是冗余的**——`Math.abs(NaN - x) >= y` 恒为 false，
    // 去掉它结果完全相同（变异验证：删掉后 9 项用例全部照旧通过）。保留它是把意图
    // 写进代码；对应单测另配了"有限值应当提交"的对照，用来区分"守卫生效"与
    // "恒返回 false"——这两种情形都会让 NaN 用例通过。
    if (!Number.isFinite(step) || !Number.isFinite(committed) || !Number.isFinite(next)) {
        return false;
    }
    return Math.abs(next - committed) >= Math.max(0, step);
}
