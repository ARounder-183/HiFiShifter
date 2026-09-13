/**
 * paramSelection.ts — 参数编辑器（钢琴卷帘）多选区模型。
 *
 * 选区从「单个 {aBeat, bBeat}」升级为「有序、互不相交的 beat 区间列表」。
 * 本模块只放与 UI 无关的纯函数，用单测固定下面这些不变式：
 *
 *   1. 段按 startBeat 升序排列；
 *   2. 任意两段互不相交；
 *   3. 相邻（端点相接或重叠）的段一律合并 —— 这是「每段独立计算统计量」
 *      语义成立的前提：若相邻两段并存，某段的边缘淡化会跨进另一段的起点。
 *   4. `startBeat <= endBeat`，非有限值不参与。
 *
 * 归一化**不夹负值**：拖拽到 0 左侧时选区随数据一起越界（渲染自然裁掉），
 * 与改造前的单选区行为一致 —— 若在此夹到 0，选区宽度会在越界期间被永久
 * 吃掉。帧域换算才夹 `startFrame >= 0`（与旧算式一致），创建手势的
 * `[0, 工程时长]` 夹取仍由调用方负责。
 *
 * 帧域换算刻意保留两套既有取整约定，避免改语义：
 *   - `beatRangesToFrameRanges`    —— (startFrame, frameCount)，对应旧
 *     `handleEditOp` 的 `floor(起点)` + `ceil(时长)`；
 *   - `beatRangesToInclusiveSpans` —— [startFrame, endFrame]（闭区间），
 *     对应旧拖拽/拉伸/形变的 `floor(起点)` + `ceil(终点)`。
 * 两者都在帧域再次合并重叠/相接的窗口（beat 域不足一帧的缝隙在帧域会消失）。
 *
 * 「无选区」沿用既有的 null 语义（而不是空数组）：所有既有
 * `if (!sel) return` 守卫因此不需要改写。
 */

/** 单段选区（beat 单位，`startBeat <= endBeat`）。 */
export type BeatRange = { startBeat: number; endBeat: number };

/** 多选区：归一化的段列表（升序、互不相交、相邻已合并、非空）。 */
export type ParamSelection = BeatRange[];

/** 帧域区间：与后端 `get/set_param_frames` 的 (startFrame, frameCount) 契约一致。 */
export type FrameRange = { startFrame: number; frameCount: number };

/** 帧域闭区间（含两端），对齐旧拖拽/拉伸路径的端点约定。 */
export type FrameSpan = { startFrame: number; endFrame: number };

/** 与旧 `handleEditOp` 的 `clamp(..., 1, 200_000)` 保持一致的一次操作帧数上限。 */
export const MAX_SELECTION_FRAMES = 200_000;

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** 构造归一的单段（自动排序两端点）。 */
export function makeBeatRange(aBeat: number, bBeat: number): BeatRange {
    return { startBeat: Math.min(aBeat, bBeat), endBeat: Math.max(aBeat, bBeat) };
}

/**
 * 归一化：过滤非有限值、排序、合并重叠或相接的段。
 * 无有效段时返回 null（与「无选区」统一）。
 */
export function normalizeSelection(
    ranges: readonly BeatRange[] | null | undefined,
): ParamSelection | null {
    if (!ranges || ranges.length === 0) return null;

    const cleaned: BeatRange[] = [];
    for (const range of ranges) {
        if (!Number.isFinite(range.startBeat) || !Number.isFinite(range.endBeat)) continue;
        cleaned.push(makeBeatRange(range.startBeat, range.endBeat));
    }
    if (cleaned.length === 0) return null;

    cleaned.sort((a, b) => a.startBeat - b.startBeat);

    const merged: BeatRange[] = [];
    for (const range of cleaned) {
        const last = merged[merged.length - 1];
        // 相接（last.endBeat === range.startBeat）也合并：相邻段并存会让
        // 「每段独立」的边界淡化互相侵入。
        if (last && range.startBeat <= last.endBeat) {
            if (range.endBeat > last.endBeat) last.endBeat = range.endBeat;
            continue;
        }
        merged.push({ ...range });
    }
    return merged;
}

/** 单段选区（拖拽框选出的那一段）；无有效段时为 null。 */
export function selectionFromBeatRange(aBeat: number, bBeat: number): ParamSelection | null {
    return normalizeSelection([makeBeatRange(aBeat, bBeat)]);
}

/** 追加一段（并集，重叠/相接自动合并）。 */
export function addBeatRange(
    selection: ParamSelection | null,
    aBeat: number,
    bBeat: number,
): ParamSelection | null {
    const next = [
        ...(selection ?? []),
        makeBeatRange(aBeat, bBeat),
    ];
    return normalizeSelection(next);
}

/** 移除包含 `beat` 的那一段（多选修饰键点击已有段 = 切换取消）。 */
export function removeRangeAtBeat(
    selection: ParamSelection | null,
    beat: number,
): ParamSelection | null {
    if (!selection) return null;
    return normalizeSelection(
        selection.filter((range) => !(beat >= range.startBeat && beat <= range.endBeat)),
    );
}

/**
 * 选区是否**完整覆盖** [aBeat, bBeat]。
 *
 * 用于「切换式」手势的判定：覆盖了才挖掉，否则并入 —— 这样同一个手势
 * 连按两次能回到原状（心形自反），而不是第二次把一个片段切碎。
 * 段升序且互不相交，因此一次线性扫描即可：一旦出现空洞即未覆盖。
 */
export function selectionCoversRange(
    selection: ParamSelection | null,
    aBeat: number,
    bBeat: number,
): boolean {
    if (!selection || selection.length === 0) return false;
    const a = Math.min(aBeat, bBeat);
    const b = Math.max(aBeat, bBeat);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return false;

    let coveredUntil = a;
    for (const range of selection) {
        if (range.endBeat < coveredUntil) continue;
        if (range.startBeat > coveredUntil) return false; // 空洞
        if (range.endBeat > coveredUntil) coveredUntil = range.endBeat;
        if (coveredUntil >= b) return true;
    }
    return coveredUntil >= b;
}

/**
 * 区间相减：从选区中挖掉 [aBeat, bBeat]。
 *
 * 结果仍归一化 —— 挖掉中间一小段会把原来的段切成两段，这正是「取消某个
 * 片段」应有的形态（断层即数据，不合并）。无重叠时原样返回（新数组）。
 */
export function subtractBeatRange(
    selection: ParamSelection | null,
    aBeat: number,
    bBeat: number,
): ParamSelection | null {
    if (!selection || selection.length === 0) return null;
    const a = Math.min(aBeat, bBeat);
    const b = Math.max(aBeat, bBeat);
    if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return normalizeSelection(selection);

    const out: BeatRange[] = [];
    for (const range of selection) {
        // 无重叠（含仅端点相接：相接不构成重叠）
        if (range.endBeat <= a || range.startBeat >= b) {
            out.push({ startBeat: range.startBeat, endBeat: range.endBeat });
            continue;
        }
        if (range.startBeat < a) out.push({ startBeat: range.startBeat, endBeat: a });
        if (range.endBeat > b) out.push({ startBeat: b, endBeat: range.endBeat });
    }
    return normalizeSelection(out);
}

/**
 * 切换：已完整覆盖 [aBeat, bBeat] 则挖掉该区间，否则并入。
 *
 * 时间轴的「修饰键 + 双击音频块」手势即此语义：同一个块再点一次即撤销。
 */
export function toggleBeatRange(
    selection: ParamSelection | null,
    aBeat: number,
    bBeat: number,
): ParamSelection | null {
    return selectionCoversRange(selection, aBeat, bBeat)
        ? subtractBeatRange(selection, aBeat, bBeat)
        : addBeatRange(selection, aBeat, bBeat);
}

/** `beat` 落在第几段（-1 = 不在任何段内）。 */
export function rangeIndexAtBeat(selection: ParamSelection | null, beat: number): number {    if (!selection || !Number.isFinite(beat)) return -1;
    for (let i = 0; i < selection.length; i += 1) {
        const range = selection[i];
        if (beat >= range.startBeat && beat <= range.endBeat) return i;
    }
    return -1;
}

export function selectionContainsBeat(selection: ParamSelection | null, beat: number): boolean {
    return rangeIndexAtBeat(selection, beat) >= 0;
}

export function selectionTotalBeats(selection: ParamSelection | null): number {
    if (!selection) return 0;
    let total = 0;
    for (const range of selection) total += range.endBeat - range.startBeat;
    return total;
}

/** 包围区间（首段起点 → 末段终点）。用于「只有一个时间窗」的旧接口。 */
export function selectionBoundingRange(selection: ParamSelection | null): BeatRange | null {
    if (!selection || selection.length === 0) return null;
    return {
        startBeat: selection[0].startBeat,
        endBeat: selection[selection.length - 1].endBeat,
    };
}

/** 整体平移（选区拖动时随数据一起移动）；结果仍归一化（可能合并出新重叠）。 */
export function shiftSelectionRanges(
    selection: ParamSelection | null,
    beatDelta: number,
): ParamSelection | null {
    if (!selection) return null;
    if (!Number.isFinite(beatDelta) || beatDelta === 0) return selection;
    return normalizeSelection(
        selection.map((range) => ({
            startBeat: range.startBeat + beatDelta,
            endBeat: range.endBeat + beatDelta,
        })),
    );
}

/**
 * 帧域合并：重叠或相接的窗口合并成一个（beat 域不足一帧的缝隙在帧域会消失）。
 *
 * 也供多段编辑计划复用：若干段的「原位 ∪ 落地位 ± 边缘淡化」窗口合并后，
 * 一次写入即可覆盖全部受影响帧，且缝隙帧写入的是读回的基准值（语义不变）。
 */
export function mergeFrameWindows(windows: readonly FrameSpan[]): FrameSpan[] {
    if (windows.length === 0) return [];
    const sorted = [...windows].sort((a, b) => a.startFrame - b.startFrame);
    const merged: FrameSpan[] = [];
    for (const window of sorted) {
        const last = merged[merged.length - 1];
        if (last && window.startFrame <= last.endFrame + 1) {
            if (window.endFrame > last.endFrame) last.endFrame = window.endFrame;
            continue;
        }
        merged.push({ startFrame: window.startFrame, endFrame: window.endFrame });
    }
    return merged;
}

/** beat 段 → 帧闭区间（floor 起点 / ceil 终点，含两端）。 */
export function beatRangesToInclusiveSpans(
    selection: ParamSelection | null,
    secPerBeat: number,
    framePeriodMs: number,
): FrameSpan[] {
    if (!selection || selection.length === 0) return [];
    const secPerBeatSafe = Math.max(1e-9, Number(secPerBeat) || 0);
    const fp = Math.max(1e-6, Number(framePeriodMs) || 5);
    return mergeFrameWindows(
        selection.map((range) => {
            const startFrame = Math.max(
                0,
                Math.floor((range.startBeat * secPerBeatSafe * 1000) / fp),
            );
            const endFrame = Math.max(
                startFrame,
                Math.ceil((range.endBeat * secPerBeatSafe * 1000) / fp),
            );
            return { startFrame, endFrame };
        }),
    );
}

/**
 * beat 段 → (startFrame, frameCount)。
 *
 * 逐段沿用旧 `handleEditOp` 的算式（起点 floor、帧数 ceil(时长) 且下钳 1），
 * 保证单段选区下的行为逐帧不变；随后在帧域合并重叠/相接窗口。
 */
export function beatRangesToFrameRanges(
    selection: ParamSelection | null,
    secPerBeat: number,
    framePeriodMs: number,
): FrameRange[] {
    if (!selection || selection.length === 0) return [];
    const secPerBeatSafe = Math.max(1e-9, Number(secPerBeat) || 0);
    const fp = Math.max(1e-6, Number(framePeriodMs) || 5);
    return mergeFrameWindows(
        selection.map((range) => {
            const startFrame = Math.max(
                0,
                Math.floor((range.startBeat * secPerBeatSafe * 1000) / fp),
            );
            const durationSec = Math.max(0, (range.endBeat - range.startBeat) * secPerBeatSafe);
            const frameCount = clamp(Math.ceil((durationSec * 1000) / fp), 1, MAX_SELECTION_FRAMES);
            return { startFrame, endFrame: startFrame + frameCount - 1 };
        }),
    ).map(({ startFrame, endFrame }) => ({
        startFrame,
        frameCount: Math.min(MAX_SELECTION_FRAMES, endFrame - startFrame + 1),
    }));
}
