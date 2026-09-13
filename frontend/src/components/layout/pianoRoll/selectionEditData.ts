// 底部参数面板「选区拖动编辑」的数据层。
//
// 为什么需要这个模块
// ------------------
// 参数曲线不是全量加载的：`usePianoRollData` 每次只向后端拉一个「可见 ±1 视口」的
// 窗口，并且在低缩放时会按画布宽度做降采样（自适应 stride），否则长音频会拉回
// 几十万个采样点。这个窗口（`paramView`，下文简称 pv）**只用于显示**。
//
// 一旦把降采样后的 pv 直接写回后端，全分辨率曲线就被覆盖了 —— 这是不可接受的
// 数据失真。所以编辑必须走另一条路：
//
//   显示：pv（可能降采样）      —— 只画，不回写
//   编辑：本模块取全分辨率数据  —— 变换后逐帧回写，不失真
//
// 同时「全选」的选区可以覆盖整个工程，一次性拉全量会阻塞主线程。这里统一按
// `CHUNK_FRAMES` 分块收发，块之间让出事件循环，保证再长的工程也不会卡死。
//
// 与 usePianoRollInteractions 的关系：交互 hook 负责手势与状态，本模块只负责
// 「取数 / 回写」两件事，不持有任何 UI 状态，便于单独测试。

import { paramsApi } from "../../../services/api";
import type { ParamName, ParamViewSegment } from "./types";
import { applyEdgeBlend, type EdgeShape } from "./paramSmoothing";
import { mergeFrameWindows, type FrameSpan } from "./paramSelection";

/** 单次 IPC 收发的帧数上限。约 32k 帧 ≈ 190 ms @ fp 5.8ms，单次处理不会掉帧。 */
const CHUNK_FRAMES = 32_768;

/** 让出事件循环，使长任务期间界面仍能响应（滚动、绘制、取消操作）。 */
function yieldToUi(): Promise<void> {
    return new Promise<void>((resolve) => {
        if (typeof requestAnimationFrame === "function") {
            requestAnimationFrame(() => resolve());
        } else {
            setTimeout(resolve, 0);
        }
    });
}

/** 逐帧（stride = 1）的曲线片段：`values[k]` 对应第 `startFrame + k` 帧。 */
export type FrameCurve = {
    startFrame: number;
    values: number[];
};

/**
 * 判断 pv 是否已以 stride=1 完整覆盖 `[startFrame, endFrame]`。
 *
 * 只有 stride 严格等于 1 才算覆盖：stride > 1 说明 pv 是降采样数据，拿去编辑
 * 会造成失真，必须重新向后端取全分辨率。
 */
export function pvCoversFullRes(
    pv: ParamViewSegment | null,
    startFrame: number,
    endFrame: number,
): boolean {
    if (!pv || pv.edit.length === 0) return false;
    if (Math.max(1, Math.floor(pv.stride)) !== 1) return false;
    const pvEnd = pv.startFrame + pv.edit.length - 1;
    return pv.startFrame <= startFrame && pvEnd >= endFrame;
}

/**
 * 从 pv 读出 `[startFrame, endFrame]` 的值，越界部分补 0。
 *
 * 自动处理 pv 的 stride（pv 可能被降采样，此时相邻帧会取到同一个样本）。
 * 仅用于即时预览：pv 是显示数据，不能作为回写后端的数据源。
 */
export function readPvRange(pv: ParamViewSegment, startFrame: number, endFrame: number): number[] {
    const step = Math.max(1, Math.floor(pv.stride));
    const out = new Array<number>(Math.max(0, endFrame - startFrame + 1));
    for (let k = 0; k < out.length; k += 1) {
        const idx = Math.round((startFrame + k - pv.startFrame) / step);
        out[k] = idx >= 0 && idx < pv.edit.length ? pv.edit[idx] : 0;
    }
    return out;
}

/**
 * 取 `[startFrame, endFrame]` 的全分辨率（stride=1）曲线。
 *
 * - pv 已以 stride=1 覆盖该范围 → 直接切片返回，不产生任何网络请求。
 *   这是最常见的路径（选区在已加载窗口内），行为与改动前完全一致。
 * - 否则分块向后端拉取，块之间让出事件循环。
 *
 * @param onProgress 可选进度回调，参数为已完成的帧数与总帧数。
 */
export async function fetchFullResCurve(args: {
    trackId: string;
    param: ParamName;
    startFrame: number;
    endFrame: number;
    paramView?: ParamViewSegment | null;
    onProgress?: (doneFrames: number, totalFrames: number) => void;
}): Promise<FrameCurve> {
    const { trackId, param, paramView, onProgress } = args;
    const startFrame = Math.max(0, Math.floor(args.startFrame));
    const endFrame = Math.max(startFrame, Math.floor(args.endFrame));
    const totalFrames = endFrame - startFrame + 1;

    // 快路径：pv 已以 stride=1 覆盖，直接切片。
    const pvFastPath = paramView ?? null;
    if (pvCoversFullRes(pvFastPath, startFrame, endFrame)) {
        const pv = pvFastPath as ParamViewSegment;
        const offset = startFrame - pv.startFrame;
        onProgress?.(totalFrames, totalFrames);
        return { startFrame, values: pv.edit.slice(offset, offset + totalFrames) };
    }

    const values = new Array<number>(totalFrames);
    let done = 0;
    for (let chunkStart = startFrame; chunkStart <= endFrame; chunkStart += CHUNK_FRAMES) {
        const chunkEnd = Math.min(endFrame, chunkStart + CHUNK_FRAMES - 1);
        const count = chunkEnd - chunkStart + 1;
        const res = await paramsApi.getParamFrames(trackId, param, chunkStart, count, 1);
        const src = res?.ok ? res.edit : undefined;
        for (let i = 0; i < count; i += 1) {
            values[chunkStart - startFrame + i] = src ? (src[i] ?? 0) : 0;
        }
        done += count;
        onProgress?.(done, totalFrames);
        await yieldToUi();
    }
    return { startFrame, values };
}

/**
 * 计算选区拖动会触及的帧范围（含边缘平滑向两侧扩展的上下文）。
 *
 * 提交前需要先知道这个范围，才能把对应区间的全分辨率数据拉下来作为基底。
 */
export function selectionDragRange(args: {
    origStartFrame: number;
    origValuesLength: number;
    frameDelta: number;
    extraEdgeFrames: number;
}): { startFrame: number; endFrame: number } {
    const { origStartFrame, origValuesLength, frameDelta, extraEdgeFrames } = args;
    if (origValuesLength <= 0) {
        return { startFrame: origStartFrame, endFrame: origStartFrame };
    }
    const newStartFrame = origStartFrame + frameDelta;
    const origEndFrame = origStartFrame + (origValuesLength - 1);
    const newEndFrame = newStartFrame + (origValuesLength - 1);
    const overallMin = Math.max(0, Math.min(origStartFrame, newStartFrame));
    const overallMax = Math.max(origEndFrame, newEndFrame);
    const edge = Math.max(0, Math.floor(extraEdgeFrames));
    return {
        startFrame: Math.max(0, overallMin - edge),
        endFrame: overallMax + edge,
    };
}

/** buildSelectionDragDense 的边缘淡化参数（halfSpan 已换算为 dense 索引数）。 */
export type SelectionDragEdgeBlend = {
    halfSpanFrames: number;
    shape?: EdgeShape;
    /** pitch 等哨兵参数的“可编辑值”判定（v!==0）；缺省全部可编辑。 */
    isEditable?: (v: number) => boolean;
};

/**
 * 构造选区拖动后的 dense 数组（**逐帧索引**：`values[k]` 对应第 `startFrame + k` 帧）。
 *
 * 预览（pointermove）与提交（pointerup）共用本函数，唯一差别是 `sourceAt`
 * 的数据来源：
 *   - 预览：从 pv 取（pv 可能被降采样，仅用于即时反馈，绝不回写）
 *   - 提交：从后端取全分辨率（保证写回不失真）
 * 两者共用同一份变换，因此用户拖动时看到的曲线与最终写入的数据完全一致。
 *
 * @param sourceAt        取某帧「当前值」，用于填充选区外的上下文
 * @param origValues      选区的全分辨率原始值（逐帧）
 * @param origStartFrame  origValues[0] 对应的帧号
 * @param frameDelta      X 方向帧偏移（纯上下拖动时为 0）
 * @param extraEdgeFrames 边缘平滑需要向两侧额外扩展的帧数
 * @param transform       逐帧变换：(原始值, 落地帧) => 新值
 * @param edgeBlend       边缘淡化参数（delta 空间交叉淡化；缺省不淡化）
 */
export function buildSelectionDragDense(args: {
    sourceAt: (frame: number) => number;
    origValues: number[];
    origStartFrame: number;
    frameDelta: number;
    extraEdgeFrames: number;
    transform: (origValue: number, frame: number) => number;
    edgeBlend?: SelectionDragEdgeBlend;
}): { startFrame: number; endFrame: number; values: number[] } {
    const {
        sourceAt,
        origValues,
        origStartFrame,
        frameDelta,
        extraEdgeFrames,
        transform,
        edgeBlend,
    } = args;

    const selLen = origValues.length;
    const { startFrame, endFrame } = selectionDragRange({
        origStartFrame,
        origValuesLength: selLen,
        frameDelta,
        extraEdgeFrames,
    });
    if (selLen <= 0) {
        return { startFrame, endFrame, values: [] };
    }

    const len = endFrame - startFrame + 1;
    const values = new Array<number>(len);
    for (let k = 0; k < len; k += 1) {
        values[k] = sourceAt(startFrame + k);
    }
    const before = values.slice();

    const newStartFrame = origStartFrame + frameDelta;
    for (let i = 0; i < selLen; i += 1) {
        const targetFrame = newStartFrame + i;
        const idx = targetFrame - startFrame;
        if (idx >= 0 && idx < len) {
            values[idx] = transform(origValues[i], targetFrame);
        }
    }

    if (edgeBlend && edgeBlend.halfSpanFrames > 0) {
        const movedStartIdx = newStartFrame - startFrame;
        applyEdgeBlend({
            dense: values,
            base: before,
            editedStartIdx: movedStartIdx,
            editedLen: selLen,
            halfSpanFrames: edgeBlend.halfSpanFrames,
            shape: edgeBlend.shape,
            isEditable: edgeBlend.isEditable,
        });
    }

    return { startFrame, endFrame, values };
}

/**
 * 把 pv 步距采样的 dense 数组展开为逐帧（stride=1）数组。
 *
 * 拉伸边缘拖拽的预览数据是 pv 步距采样（`dense[k]` ↔ `startFrame + k×stride`）。
 * 提交必须逐帧回写：按渲染同款的**线性插值**展开 —— 否则把 stride 间隔采样
 * 当连续帧写入会造成时间压缩，并覆盖未选帧（stride=1 时原样返回，零开销）。
 */
export function expandStrideSampledDense(dense: number[], stride: number): number[] {
    const step = Math.max(1, Math.floor(stride));
    if (step === 1 || dense.length === 0) {
        return dense;
    }
    const out = new Array<number>((dense.length - 1) * step + 1);
    for (let i = 0; i < dense.length; i += 1) {
        out[i * step] = dense[i];
    }
    for (let i = 0; i < dense.length - 1; i += 1) {
        const a = dense[i];
        const b = dense[i + 1];
        for (let f = 1; f < step; f += 1) {
            out[i * step + f] = a + ((b - a) * f) / step;
        }
    }
    return out;
}

/**
 * 把逐帧数组回写到后端，按 `CHUNK_FRAMES` 分块，块之间让出事件循环。
 *
 * 撤销点处理：后端 `set_param_frames` 的 `checkpoint` 是「写入前先快照时间线」，
 * 因此只在**第一块**置 true —— 撤销一次即可回退整段编辑，而不是只回退最后一块。
 *
 * @param values 逐帧数组，`values[k]` 对应第 `startFrame + k` 帧（stride=1）。
 */
export async function uploadFullResCurve(args: {
    trackId: string;
    param: ParamName;
    startFrame: number;
    values: number[];
    onProgress?: (doneFrames: number, totalFrames: number) => void;
}): Promise<void> {
    const { trackId, param, values, onProgress } = args;
    const startFrame = Math.max(0, Math.floor(args.startFrame));
    if (values.length === 0) return;

    const totalFrames = values.length;
    let isFirstChunk = true;
    let done = 0;

    for (let offset = 0; offset < totalFrames; offset += CHUNK_FRAMES) {
        const count = Math.min(CHUNK_FRAMES, totalFrames - offset);
        const chunk = values.slice(offset, offset + count);
        await paramsApi.setParamFrames(
            trackId,
            param,
            startFrame + offset,
            chunk,
            isFirstChunk, // 仅首块打撤销点
        );
        isFirstChunk = false;
        done += count;
        onProgress?.(done, totalFrames);
        await yieldToUi();
    }
}

/** 多段写入片段：`values[k]` 对应第 `startFrame + k` 帧（stride=1）。 */
export type FullResWriteSegment = { startFrame: number; values: number[] };

/**
 * 多段逐帧回写：把若干互不相连的片段写回后端。
 *
 * 撤销点纪律与 uploadFullResCurve 一致并向「多次调用」推广：**整个批次只在
 * 第一个实际写入的块上打一次撤销点**，因此多选区的一次编辑（例如跨两段的
 * 移调、粘贴）撤销一次即可整体回退。
 */
export async function uploadFullResCurveSegments(args: {
    trackId: string;
    param: ParamName;
    segments: readonly FullResWriteSegment[];
    onProgress?: (doneFrames: number, totalFrames: number) => void;
}): Promise<void> {
    const { trackId, param, onProgress } = args;
    const segments = args.segments.filter((segment) => segment.values.length > 0);
    if (segments.length === 0) return;

    const totalFrames = segments.reduce((sum, segment) => sum + segment.values.length, 0);
    let isFirstChunk = true;
    let done = 0;

    for (const segment of segments) {
        const startFrame = Math.max(0, Math.floor(segment.startFrame));
        const values = segment.values;
        for (let offset = 0; offset < values.length; offset += CHUNK_FRAMES) {
            const count = Math.min(CHUNK_FRAMES, values.length - offset);
            await paramsApi.setParamFrames(
                trackId,
                param,
                startFrame + offset,
                values.slice(offset, offset + count),
                isFirstChunk, // 整批仅首个实际写入打撤销点
            );
            isFirstChunk = false;
            done += count;
            onProgress?.(done, totalFrames);
            await yieldToUi();
        }
    }
}

/** 多段变换计划产出的写入片段（dense 逐帧索引；含边缘淡化的扩展帧）。 */
export type MultiRangeEditPiece = FrameSpan & { values: number[] };

/**
 * 多段编辑的写入窗口：每段「原位 ∪ 落地位」± 该段边缘淡化，合并重叠窗口。
 *
 * 提交路径需要**先知道窗口才能取数**（取回的窗口即 sourceAt 的来源），
 * 因此把这一步单独导出；buildMultiRangeEditPlan 内部用的是同一实现，
 * 二者结果必然一致。
 */
export function planSelectionEditWindows(args: {
    ranges: readonly FrameSpan[];
    frameDelta?: number;
    edgeHalfSpanAt?: (rangeIndex: number) => number;
}): FrameSpan[] {
    const frameDelta = Math.trunc(Number(args.frameDelta) || 0);
    const windows: FrameSpan[] = [];
    for (let i = 0; i < args.ranges.length; i += 1) {
        const range = args.ranges[i];
        const edge = Math.max(0, Math.ceil(Number(args.edgeHalfSpanAt?.(i) ?? 0) || 0));
        const landedStart = range.startFrame + frameDelta;
        const landedEnd = range.endFrame + frameDelta;
        const startFrame = Math.max(0, Math.min(range.startFrame, landedStart) - edge);
        const endFrame = Math.max(range.endFrame, landedEnd, startFrame) + edge;
        windows.push({ startFrame, endFrame });
    }
    return mergeFrameWindows(windows);
}

/**
 * 多段「选区变换」计划：对每条选区段独立施加变换，输出合并后的写入片段。
 *
 * 预览与提交共用本函数（唯一差别是 `sourceAt` / `valuesAt` 的数据来源：
 * 预览读 pv、提交读全分辨率），因此**用户看到的就是最终写回的**。
 *
 * 语义要点：
 *   - 每段独立：`valuesAt(i)` 是该段自己的（可已变换的）逐帧值，
 *     `edgeHalfSpanAt(i)` 是该段自己的边缘淡化半宽 —— 断层两侧互不影响；
 *   - 段可 X 向位移（`frameDelta`）：写入窗口 = 原位 ∪ 落地位 ± 边缘淡化，
 *     合并重叠窗口后一次写出，缝隙帧写回基准值（等价于未改动）；
 *   - 落地位越界（< 0 或超出）的帧自然被裁剪。
 */
export function buildMultiRangeEditPlan(args: {
    /** 原始选区（闭区间，升序、互不相交） */
    ranges: readonly FrameSpan[];
    /** 每段落地帧偏移（纯 Y 向变换传 0） */
    frameDelta?: number;
    /** 取第 i 段「被变换的源值」（逐帧，长度 = 段长）；返回空则跳过该段 */
    valuesAt: (rangeIndex: number) => readonly number[] | null | undefined;
    /** 取某帧的基准值（填充写入窗口内未被任何段覆盖的帧） */
    sourceAt: (frame: number) => number;
    /** 逐帧变换：(段号, 段内下标, 源值, 落地帧) => 新值；缺省恒等 */
    transformAt?: (
        rangeIndex: number,
        index: number,
        sourceValue: number,
        targetFrame: number,
    ) => number;
    /** 第 i 段的边缘淡化半宽（帧）；缺省不淡化 */
    edgeHalfSpanAt?: (rangeIndex: number) => number;
    /** pitch 等哨兵参数的「可编辑值」判定；缺省全部可编辑 */
    isEditable?: (v: number) => boolean;
    shape?: EdgeShape;
}): MultiRangeEditPiece[] {
    const { ranges, sourceAt, transformAt, edgeHalfSpanAt, isEditable, shape } = args;
    const frameDelta = Math.trunc(Number(args.frameDelta) || 0);
    if (ranges.length === 0) return [];

    const edgeAt = (rangeIndex: number) =>
        Math.max(0, Math.ceil(Number(edgeHalfSpanAt?.(rangeIndex) ?? 0) || 0));

    // 1) 写入窗口（与提交路径取数用的窗口同源）。
    const merged = planSelectionEditWindows({ ranges, frameDelta, edgeHalfSpanAt });

    // 2) 用基准值铺满窗口，并留存「编辑前」副本供 delta 空间交叉淡化使用。
    const pieces: Array<MultiRangeEditPiece & { before: number[] }> = merged.map((window) => {
        const len = window.endFrame - window.startFrame + 1;
        const values = new Array<number>(len);
        for (let k = 0; k < len; k += 1) {
            values[k] = sourceAt(window.startFrame + k);
        }
        return { startFrame: window.startFrame, endFrame: window.endFrame, values, before: values.slice() };
    });

    // 3) 逐段写入（升序；X 位移后段重叠时后写覆盖先写，确定性可复现）。
    for (let i = 0; i < ranges.length; i += 1) {
        const source = args.valuesAt(i);
        if (!source || source.length === 0) continue;
        const landedStart = ranges[i].startFrame + frameDelta;
        const landedEnd = landedStart + source.length - 1;
        for (const piece of pieces) {
            if (piece.endFrame < landedStart || piece.startFrame > landedEnd) continue;
            const from = Math.max(landedStart, piece.startFrame);
            const to = Math.min(landedEnd, piece.endFrame);
            for (let frame = from; frame <= to; frame += 1) {
                const index = frame - landedStart;
                const sourceValue = Number(source[index]) || 0;
                piece.values[frame - piece.startFrame] = transformAt
                    ? transformAt(i, index, sourceValue, frame)
                    : sourceValue;
            }
        }
    }

    // 4) 逐段边缘淡化（base 为编辑前基准值；与单段路径同一 applyEdgeBlend）。
    for (let i = 0; i < ranges.length; i += 1) {
        const halfSpan = edgeAt(i);
        const source = args.valuesAt(i);
        if (halfSpan <= 0 || !source || source.length === 0) continue;
        const landedStart = ranges[i].startFrame + frameDelta;
        for (const piece of pieces) {
            const editedStartIdx = Math.max(0, landedStart - piece.startFrame);
            const editedEndIdx = Math.min(
                piece.values.length - 1,
                landedStart + source.length - 1 - piece.startFrame,
            );
            if (editedEndIdx < editedStartIdx) continue;
            applyEdgeBlend({
                dense: piece.values,
                base: piece.before,
                editedStartIdx,
                editedLen: editedEndIdx - editedStartIdx + 1,
                halfSpanFrames: halfSpan,
                shape,
                isEditable,
            });
        }
    }

    return pieces.map(({ startFrame, endFrame, values }) => ({ startFrame, endFrame, values }));
}

