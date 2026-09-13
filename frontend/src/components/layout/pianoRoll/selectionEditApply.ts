// 选区编辑（移调/设音高/shift± 等对话框 op）的「取数 → 编辑 → 边缘淡化 → 回写」
// 编排模块。
//
// 从 PianoRollPanel.tsx 的内联闭包抽出（原实现约 140 行，与
// usePianoRollInteractions.applyEdgeSmoothingToDense 逐行重复）。边缘淡化统一走
// paramSmoothing.applyEdgeBlend（delta 空间交叉淡化）。
//
// 关键语义（相对旧实现的差异，均为方案既定修复）：
//   1. 过渡带宽度以毫秒定标（每侧 ≤ strength% × 60ms，上限选区/4），
//      不再随选区长度线性放大；
//   2. 删除 changeFactor 启发式（旧实现用"编辑前边界跳变"压制淡化，
//      素材越不连续淡化越弱，与目标背道而驰）；
//   3. 未浊哨兵帧（pitch=0）绝不写非零值（旧实现的 bug）；
//   4. 选区外写入量 = w × 本次编辑的延拓 delta（可组合，重复编辑不累积失真）。

import { paramsApi } from "../../../services/api";
import type { ParamFramesPayload } from "../../../types/api";
import type { ParamName } from "./types";
import type { FrameRange } from "./paramSelection";
import { applyEdgeBlend, edgeHalfSpanFramesForSelection, type EdgeShape } from "./paramSmoothing";

/** 编辑延拓描述：选区外的"编辑意图"如何延展。 */
export type SelectionEditExtension =
    | {
          /** 编辑曲线延拓：任意帧的编辑后期望值（setPitch：常量目标）。 */
          kind: "editedAt";
          editedAt: (frame: number, baseValue: number) => number;
      }
    | {
          /** 编辑 delta 延拓：任意帧的编辑增量（常数移调：常量 delta）。 */
          kind: "deltaAt";
          deltaAt: (frame: number, baseValue: number) => number;
      };

export type ApplySelectionEditArgs = {
    trackId: string;
    param: ParamName;
    /** 选区起始帧（绝对帧号）。 */
    startFrame: number;
    /** 选区帧数。 */
    frameCount: number;
    /** 帧周期（毫秒），用于过渡带的毫秒定标。 */
    framePeriodMs: number;
    /** 平滑度（0-100）。 */
    smoothnessPercent: number;
    /** 选区内编辑：(当前选区值) => 新选区值（与旧实现的 editSelection 语义一致）。 */
    editSelection: (currentSelectionVals: number[]) => number[];
    /** 编辑延拓描述；缺省 = 边界帧实际 delta 常数延拓。 */
    extension?: SelectionEditExtension;
    /** 权重剖面形状（缺省 smoothstep）。 */
    shape?: EdgeShape;
    /** 哨兵判定（pitch: v!==0）；返回 false 的帧不做淡化。 */
    isEditable?: (v: number) => boolean;
    /**
     * 是否在本段写入前打撤销点（缺省 true）。
     * 多段批量编辑由 applySelectionEditOverRanges 只在首段置 true，
     * 保证一次操作一个撤销点。
     */
    checkpoint?: boolean;
};

/**
 * 执行一次带边缘淡化的选区编辑。返回是否实际写回。
 * 撤销语义与旧实现一致：整段（含边缘淡化区）一个 checkpoint。
 */
export async function applySelectionEditWithEdgeSmoothing(
    args: ApplySelectionEditArgs,
): Promise<boolean> {
    const {
        trackId,
        param,
        startFrame,
        frameCount,
        framePeriodMs,
        smoothnessPercent,
        editSelection,
        extension,
        shape,
        isEditable,
    } = args;
    if (frameCount <= 0) return false;
    const smoothness = Math.min(100, Math.max(0, Number(smoothnessPercent) || 0));

    const halfSpanFrames =
        smoothness > 0
            ? edgeHalfSpanFramesForSelection({
                  strengthPercent: smoothness,
                  framePeriodMs,
                  editedLen: frameCount,
              })
            : 0;
    const extend = halfSpanFrames > 0 ? Math.ceil(halfSpanFrames) : 0;

    const extStart = Math.max(0, startFrame - extend);
    const extCount = frameCount + (startFrame - extStart) + extend;
    const selOffset = startFrame - extStart;

    const res = await paramsApi.getParamFrames(trackId, param, extStart, extCount, 1);
    if (!res?.ok) return false;

    const payload = res as ParamFramesPayload;
    const beforeDense = (payload.edit ?? []).map((v) => Number(v) || 0);
    if (beforeDense.length <= 0) return false;

    const selEnd = Math.min(beforeDense.length - 1, selOffset + frameCount - 1);
    if (selOffset < 0 || selOffset >= beforeDense.length || selEnd < selOffset) {
        return false;
    }
    const actualSelLen = selEnd - selOffset + 1;
    const currentSel = beforeDense.slice(selOffset, selOffset + actualSelLen);
    const nextSel = editSelection(currentSel);

    const editedDense = beforeDense.slice();
    for (let i = 0; i < actualSelLen; i += 1) {
        editedDense[selOffset + i] = Number(nextSel[i] ?? currentSel[i] ?? 0) || 0;
    }

    if (halfSpanFrames > 0) {
        // 延拓回调以绝对帧号表达（dense 索引 = 绝对帧号 − extStart）
        const toFrame = (idx: number) => extStart + idx;
        applyEdgeBlend({
            dense: editedDense,
            base: beforeDense,
            editedStartIdx: selOffset,
            editedLen: actualSelLen,
            halfSpanFrames: halfSpanFrames,
            shape,
            isEditable,
            editedAt:
                extension?.kind === "editedAt"
                    ? (idx, baseValue) => extension.editedAt(toFrame(idx), baseValue)
                    : undefined,
            editedDeltaAt:
                extension?.kind === "deltaAt"
                    ? (idx, baseValue) => extension.deltaAt(toFrame(idx), baseValue)
                    : undefined,
        });
    }

    await paramsApi.setParamFrames(trackId, param, extStart, editedDense, args.checkpoint ?? true);
    return true;
}

/**
 * 多选区版本的选区编辑：**逐段独立**执行（每段等价于一个独立的旧式选区，
 * 断层两侧互不影响），整个批次只打一个撤销点。
 *
 * 「每段独立」是已确认语义：平均化取各段自己的均值、平滑化取各段自己的
 * 高斯上下文与边界、量化取各段自己的基准，段间统计量不混合。
 *
 * @returns 是否至少有一段实际写回（全段取数失败时为 false）。
 */
export async function applySelectionEditOverRanges(
    args: Omit<ApplySelectionEditArgs, "startFrame" | "frameCount" | "checkpoint"> & {
        ranges: readonly FrameRange[];
    },
): Promise<boolean> {
    const { ranges, ...rest } = args;
    let wrote = false;
    for (const range of ranges) {
        const ok = await applySelectionEditWithEdgeSmoothing({
            ...rest,
            startFrame: range.startFrame,
            frameCount: range.frameCount,
            // 首段写入前打撤销点；此前若某段取数失败（未写回），撤销点顺延到
            // 第一个真正写入的段，不会出现「整批无撤销点」。
            checkpoint: !wrote,
        });
        if (ok) wrote = true;
    }
    return wrote;
}
