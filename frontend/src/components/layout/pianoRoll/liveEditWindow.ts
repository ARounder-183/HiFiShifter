/**
 * live 覆盖窗口的**纯写入逻辑**（从 `useLiveParamEditing` 抽出）。
 *
 * 【为什么独立成模块】这两段逻辑在指针频率的热路径上执行（手绘工具会把一个
 * `pointermove` 的 `coalescedEvents` 全部展开逐点处理），且它们的**下标语义**
 * 必须与旧的"全窗口扫描"实现逐值等价 —— 否则画出来的曲线会与手指位置错位。
 * vitest 跑在 node 环境、不允许 import `.tsx`（见 mainCanvasSignature.ts 的
 * 说明），故把纯逻辑放在这里以便直接单测（见 liveEditWindow.test.ts：用
 * 旧的全扫实现作为参照逐值对拍）。
 *
 * 【从 O(窗口) 到 O(受影响)】两条旧实现都在**整份参数窗口**上做线性扫描：
 *
 * - 写入：遍历全部下标、只用 `f ∈ [minF, maxF]` 判定跳过；
 * - 擦除：`覆盖 = null` 后重新 `slice()` 整份窗口（直线 / 颤音工具每帧一次）。
 *
 * 窗口典型 ~6400 个元素（`usePianoRollData` 的 `viewFrames` 上限是 200000），
 * 而每次真正改动的点只有笔刷宽度那么多。窗口内的帧号是**等差**的
 * （`frame(i) = startFrame + i · stride`），因此区间端点可直接反解，只需遍历
 * 受影响的那些下标。
 */

import type { StrokeMode } from "./types";

/** 窗口内一段连续的下标区间（闭区间）。 */
export interface IndexRange {
    lo: number;
    hi: number;
}

export interface LiveEditWindowWriteArgs {
    /** 将被**原地改写**的曲线副本（与键中窗口对齐）。 */
    edit: number[];
    /** 已提交曲线（`mode === "restore"` 时的还原源）。 */
    orig: readonly number[];
    /** 窗口首帧（`ParamViewSegment.startFrame`）。 */
    startFrame: number;
    /** 窗口帧步长（`ParamViewSegment.stride`）。 */
    stride: number;
    /** 待写入的稠密值；`null` 表示只做还原（`mode === "restore"`）。 */
    dense: number[] | null;
    /** `dense[0]` 对应的绝对帧号。 */
    denseStartFrame: number;
    /** 受影响帧范围（闭区间，绝对帧号）。 */
    minF: number;
    maxF: number;
    mode: StrokeMode;
    /**
     * 写入前的**值域钳制**（缺省恒等）。
     *
     * 【为什么必须有】live 覆盖是"用户看到的曲线"，而提交值会被后端按参数值域
     * 钳制。若预览不钳、提交被钳，拖拽时参数线因画布裁切看着"停在顶端"、波形却
     * 按超出值放大，松手后又跳回来。传入与后端同构的钳制函数
     * （`paramRanges::clampParamWriteValue`）即可让两条路径逐值一致。
     */
    clampValue?: (value: number) => number;
}

/**
 * 把一段稠密值写进 live 覆盖窗口（原地改写 `edit`）。
 *
 * @returns 实际遍历过的下标区间；窗口为空时为 `null`。调用方据此累计"本次
 *   覆盖写过的区间并集"，供 {@link restoreLiveEditRange} 精确回滚。
 */
export function writeDenseIntoLiveWindow(args: LiveEditWindowWriteArgs): IndexRange | null {
    const { edit, orig, startFrame, stride, dense, denseStartFrame, minF, maxF, mode, clampValue } =
        args;
    const length = edit.length;
    if (length === 0) return null;

    // 帧号 → 下标的反解：`i ∈ [ceil((minF - start)/stride), floor((maxF - start)/stride)]`。
    // 非正步长无法反解（契约完备性分支；生产路径 `stride` 恒 ≥ 1，见
    // usePianoRollData），此时退回全窗口遍历，区间外的帧由循环内的 `f` 边界
    // 判定跳过 —— 与旧实现逐值等价。
    let first = 0;
    let last = length - 1;
    if (stride > 0) {
        const lo = Math.ceil((minF - startFrame) / stride);
        const hi = Math.floor((maxF - startFrame) / stride);
        first = Math.max(0, lo);
        last = Math.min(last, hi);
    }
    if (first > last) {
        // 窗口与受影响帧区间完全不相交：无需遍历，但仍要报出区间（空区间由
        // 调用方按 lo > hi 忽略）。
        return { lo: first, hi: last };
    }

    for (let i = first; i <= last; i += 1) {
        const f = startFrame + i * stride;
        if (f < minF || f > maxF) continue;
        if (mode === "restore") {
            // 还原源来自后端（已在其值域内），仍过一遍钳制以保持"任何进入 live
            // 覆盖的值都已在值域内"这条不变量；对已钳值恒等。
            const restored = orig[i] ?? edit[i];
            edit[i] = clampValue !== undefined ? clampValue(restored) : restored;
        } else if (dense) {
            const j = f - denseStartFrame;
            if (j >= 0 && j < dense.length) {
                const next = dense[j] ?? edit[i];
                edit[i] = clampValue !== undefined ? clampValue(next) : next;
            }
        }
    }
    return { lo: first, hi: last };
}

/**
 * 把一段下标区间还原为**已提交曲线**的值（原地改写 `edit`）。
 *
 * 直线 / 颤音工具的预览每帧按当前端点重算整段，必须先擦掉上一帧写过的点，
 * 否则端点往回拖时旧预览会残留在图上。旧实现是丢弃整份覆盖再整份拷贝
 * （每帧一次 O(窗口) 拷贝 + 一次分配）；这里只还原上一帧真正写过的区间。
 *
 * @returns 是否真的改动了数据（用于决定要不要推进版本号）。
 */
export function restoreLiveEditRange(args: {
    edit: number[];
    committed: readonly number[];
    range: IndexRange;
}): boolean {
    const { edit, committed, range } = args;
    const from = Math.max(0, range.lo);
    const to = Math.min(range.hi, edit.length - 1);
    if (from > to) return false;
    let changed = false;
    for (let i = from; i <= to; i += 1) {
        const value = committed[i];
        if (value !== undefined && edit[i] !== value) {
            edit[i] = value;
            changed = true;
        }
    }
    return changed;
}
