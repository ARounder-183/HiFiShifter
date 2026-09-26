/**
 * live 覆盖窗口的**纯逻辑**（从 `useLiveParamEditing` 抽出）：写入、回滚、
 * 换窗口时的**重锚**，以及窗口键的解析。
 *
 * 【为什么独立成模块】这些逻辑在指针频率的热路径上执行（手绘工具会把一个
 * `pointermove` 的 `coalescedEvents` 全部展开逐点处理），且它们的**下标语义**
 * 必须与旧的"全窗口扫描"实现逐值等价 —— 否则画出来的曲线会与手指位置错位。
 * vitest 跑在 node 环境、不允许 import `.tsx`（见 mainCanvasSignature.ts 的
 * 说明），故把纯逻辑放在这里以便直接单测（见 liveEditWindow.test.ts：用
 * 旧的全扫实现作为参照逐值对拍；重锚见 liveEditReanchor.test.ts）。
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

/**
 * live 覆盖窗口键的字段。
 *
 * 键形如 `v2|{trackId}|{paramId}|{startFrame}|{frameCount}|{stride}`（由
 * `usePianoRollData.computeVisibleRequest` 生成）。`scope` 是"这条覆盖属于哪个
 * 参数的哪个窗口族"，与 `useLoudnessCurves.parseLiveOverrideKey` 的字段切分
 * 口径一致（那边按 `paramId` 归属参数、供幅度映射采样；这边按 `scope` 判定
 * "换窗口是否合法"）。
 */
export interface LiveEditWindowKey {
    /** `v2|{trackId}|{paramId}` —— 作用域；变了就是换参数 / 换轨。 */
    scope: string;
    /** 窗口首帧。 */
    startFrame: number;
    /** 窗口帧步长。 */
    stride: number;
}

/**
 * 解析 live 覆盖的窗口键。
 *
 * 【为什么要 `scope`】窗口键同时承载"属于谁"（轨道 + 参数）与"窗口在哪"
 * （起帧 / 帧数 / 步长）两组信息，而它们的**变更语义完全相反**：
 *
 * - 作用域变了（切参数 / 切轨）→ 合法换窗口，正在画的笔画本就该丢弃；
 * - 只有窗口参数变了（取数回包换掉窗口 / `dyn` 帧周期变化 / 视口移动）→
 *   正在画的笔画必须**保留**（见 {@link reanchorLiveEditWindow}）。
 *
 * 把两者混为一谈（只看 key 是否相等）正是"画音量 / 动态时轨迹凭空消失"的
 * 机制之一（见 docs/plans/2026-09-26-volume-dyn-drag-trail-fix.md §1.1）。
 */
export function parseLiveEditWindowKey(key: string): LiveEditWindowKey {
    const parts = key.split("|");
    return {
        // [v2, trackId, paramId, startFrame, frameCount, stride]
        scope: `${parts[0] ?? ""}|${parts[1] ?? ""}|${parts[2] ?? ""}`,
        startFrame: Number(parts[3]) || 0,
        stride: Number(parts[5]) || 1,
    };
}

export interface LiveEditReanchorArgs {
    /** 旧覆盖的当前值（与**旧**窗口对齐；已原地更新过）。 */
    edit: readonly number[];
    /** 旧覆盖建立时的已提交曲线（同旧窗口；用来区分"用户画的帧"与"未动的帧"）。 */
    base: readonly number[];
    /** 旧窗口首帧。 */
    startFrame: number;
    /** 旧窗口帧步长。 */
    stride: number;
    /** **新**窗口的已提交曲线（新基准）。 */
    nextEdit: readonly number[];
    /** 新窗口首帧。 */
    nextStartFrame: number;
    /** 新窗口帧步长。 */
    nextStride: number;
}

export interface ReanchoredLiveEditWindow {
    /** 与**新**窗口对齐的值：新基准 + 用户画过的帧。 */
    values: number[];
    /** 用户画过的值在新窗口中的下标区间（供 {@link restoreLiveEditRange} 回滚）；无则 null。 */
    drawnRange: IndexRange | null;
}

/**
 * 把一份 live 覆盖**重锚**到新窗口：保留用户画过的帧，其余取新基准。
 *
 * 【为什么不能直接 `nextEdit.slice()`】旧实现换窗口时整份丢弃覆盖层，
 * 于是"取数回包换掉 `paramView`（键变了或 `pv.edit` 换了引用）"会让正在画的
 * 笔画瞬间消失 —— 用户报告的现象是"按下那一点画出来了、之后轨迹一点都没有"
 * （见 docs/plans/2026-09-26-volume-dyn-drag-trail-fix.md §1.1）。重锚把已画的
 * 值按**绝对帧号**搬进新窗口，笔画就不再依赖"窗口恰好没被换掉"这个前提。
 *
 * 【为什么只搬"与旧基准不同"的帧】覆盖层 = 旧基准 + 用户改动。只搬差值，
 * 未动的帧就自动跟随新基准 —— 于是外部改动（撤销 / 别的编辑）在用户没画的
 * 帧上照常显现，不会被一份陈旧的整窗口快照盖住。
 *
 * 【帧号是唯一坐标】新旧窗口的步长可以不同（快照降采样），故按绝对帧号反解
 * 下标；两个网格对不上的帧（步长非整数 / 新窗口落在旧窗口之外）不搬，取新基准。
 *
 * @returns 新窗口下的值，以及用户画过的下标区间（供回滚）。
 */
export function reanchorLiveEditWindow(args: LiveEditReanchorArgs): ReanchoredLiveEditWindow {
    const { edit, base, startFrame, stride, nextEdit, nextStartFrame, nextStride } = args;
    const values = nextEdit.slice();
    // 旧覆盖里"用户画过"的帧数：`base` 与 `edit` 同窗口同长度，取小值防御
    // 极端情况下的错位（例如旧基准短于覆盖）。
    const oldLength = Math.min(edit.length, base.length);
    let lo = -1;
    let hi = -1;
    for (let i = 0; i < values.length; i += 1) {
        const frame = nextStartFrame + i * nextStride;
        const j = frameToIndex(frame, startFrame, stride, oldLength);
        if (j === null) continue;
        if (edit[j] === base[j]) continue;
        values[i] = edit[j] as number;
        if (lo < 0) lo = i;
        hi = i;
    }
    return { values, drawnRange: lo < 0 ? null : { lo, hi } };
}

/**
 * 绝对帧号 → 窗口下标（反解 `frame(i) = startFrame + i · stride`）。
 *
 * @returns 下标；帧不落在窗口的采样栅格上（步长非整数、越界、步长退化）时为 null。
 */
function frameToIndex(
    frame: number,
    startFrame: number,
    stride: number,
    length: number,
): number | null {
    if (!(stride > 0)) return null;
    const index = Math.round((frame - startFrame) / stride);
    if (index < 0 || index >= length) return null;
    // 非整数步长下 `Math.round` 可能落到"看起来最接近"的槽位，必须复核帧号真的
    // 相等 —— 宁可少搬一帧，也不能把值放到错误的帧上。
    if (startFrame + index * stride !== frame) return null;
    return index;
}

/**
 * live 覆盖与新窗口不匹配时的**可观测**告警（仅开发构建）。
 *
 * 【为什么必须有】这处不匹配此前是**静默 return**：不写、不报错，于是
 * "画音量 / 动态时轨迹不画"只能靠肉眼复现、无法从日志定位
 * （见 docs/plans/2026-09-26-volume-dyn-drag-trail-fix.md §1.1 / 阶段 3）。
 * 重锚（{@link reanchorLiveEditWindow}）之后这条分支应当不可达；保留它作为
 * **不变量哨兵** —— 一旦触发，说明重锚被绕过，开发期立刻可见。
 */
export function warnLiveEditWindowMismatch(expectedKey: string, actualKey: string | null): void {
    if (!isDevBuild()) return;
    console.warn(
        "[liveEdit] 覆盖层与当前参数窗口不匹配：本次笔画不会绘制。" +
            "（重锚逻辑被绕过？见 liveEditWindow.reanchorLiveEditWindow）" +
            ` expected=${expectedKey} actual=${actualKey ?? "<none>"}`,
    );
}

/**
 * 是否开发构建。
 *
 * 包一层 try：`import.meta.env` 由 Vite / vitest 注入，在纯 node 环境里读取会
 * 抛错 —— 单测可能直接 import 本模块（见 liveEditReanchor.test.ts）。
 */
function isDevBuild(): boolean {
    try {
        return import.meta.env.DEV === true;
    } catch {
        return false;
    }
}
