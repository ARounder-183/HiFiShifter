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
 */
export function buildSelectionDragDense(args: {
    sourceAt: (frame: number) => number;
    origValues: number[];
    origStartFrame: number;
    frameDelta: number;
    extraEdgeFrames: number;
    transform: (origValue: number, frame: number) => number;
    computeChangeFactor?: (
        before: number[],
        after: number[],
        editedStartIdx: number,
        editedLen: number,
    ) => number;
    applyEdgeSmoothing?: (
        dense: number[],
        editedStartIdx: number,
        editedLen: number,
        changeFactor: number,
    ) => void;
}): { startFrame: number; endFrame: number; values: number[] } {
    const {
        sourceAt,
        origValues,
        origStartFrame,
        frameDelta,
        extraEdgeFrames,
        transform,
        computeChangeFactor,
        applyEdgeSmoothing,
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

    if (computeChangeFactor && applyEdgeSmoothing) {
        const movedStartIdx = newStartFrame - startFrame;
        const factor = computeChangeFactor(before, values, movedStartIdx, selLen);
        applyEdgeSmoothing(values, movedStartIdx, selLen, factor);
    }

    return { startFrame, endFrame, values };
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
