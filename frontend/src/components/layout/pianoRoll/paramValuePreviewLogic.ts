/**
 * Popup parameter preview helpers for piano roll interactions.
 *
 * These helpers keep preview formatting logic consistent with interaction logic:
 * - select-tool drag preview quantization
 * - draw-tool preview quantization
 * - right-drag morph percentage preview
 */

import {
    scaleStepDeltaBetween,
    snapToScale,
    snapToSemitone,
    transposePitchByScaleSteps,
} from "../../../utils/musicalScales";
import type { ScaleLike } from "../../../utils/musicalScales";
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    snapChildPitchOffsetValue,
} from "./childPitchOffsetParams";

function isPitchSnapTargetParam(param: string): boolean {
    return (
        param === "pitch" ||
        isChildPitchOffsetCentsParam(param) ||
        isChildPitchOffsetDegreesParam(param)
    );
}

export function getSelectDragPreviewValue(args: {
    editParam: string;
    startValue: number;
    currentValue: number;
    fineScale: number;
    effectiveSnap: boolean;
    pitchSnapUnit?: "semitone" | "scale";
    projectScale?: ScaleLike;
}): number {
    const {
        editParam,
        startValue,
        currentValue,
        fineScale,
        effectiveSnap,
        pitchSnapUnit,
        projectScale,
    } = args;

    // 动态不再特殊：拖拽对所有参数都是**值域内线性偏移**（见
    // paramRanges.shiftValueForDrag），且位移量就取自指针在值域纵轴上的位移。
    // 于是"起点处的值被同样偏移后的结果"恰好等于指针当前所在的值 ——
    // 弹窗直接显示 `currentValue` 即与实际拖拽结果一致（下方通用分支已如此）。
    //
    // 【曾经的写法】这里原来对 dyn 单独做乘性换算（`startValue × 2^Δ`），与
    // 拖拽的乘性法则配套。那个组合的表现是"动态值越小、同样的指针位移带来的
    // 变化越小"，用户观感为"不跟手"；改成线性后两侧同时简化。
    if (!effectiveSnap || !isPitchSnapTargetParam(editParam)) {
        return currentValue;
    }

    const rawValueDelta = (currentValue - startValue) * fineScale;

    if (editParam === "pitch") {
        if (pitchSnapUnit === "scale" && projectScale) {
            const stepDelta = scaleStepDeltaBetween(startValue, currentValue, projectScale);
            return transposePitchByScaleSteps(startValue, stepDelta, projectScale);
        }
        return startValue + Math.round(rawValueDelta);
    }

    if (isChildPitchOffsetCentsParam(editParam)) {
        return startValue + Math.round(rawValueDelta / 100) * 100;
    }

    if (isChildPitchOffsetDegreesParam(editParam)) {
        return startValue + Math.round(rawValueDelta);
    }

    return currentValue;
}

export function getDrawPreviewValue(args: {
    editParam: string;
    rawValue: number;
    effectiveSnap: boolean;
    pitchSnapUnit?: "semitone" | "scale";
    projectScale?: ScaleLike;
    pitchSnapToleranceCents?: number;
}): number {
    const {
        editParam,
        rawValue,
        effectiveSnap,
        pitchSnapUnit,
        projectScale,
        pitchSnapToleranceCents,
    } = args;

    if (!effectiveSnap || !isPitchSnapTargetParam(editParam)) {
        return rawValue;
    }

    if (isChildPitchOffsetCentsParam(editParam) || isChildPitchOffsetDegreesParam(editParam)) {
        return snapChildPitchOffsetValue(editParam, rawValue);
    }

    if (editParam !== "pitch") {
        return rawValue;
    }

    const snapped =
        pitchSnapUnit === "scale" && projectScale
            ? snapToScale(rawValue, projectScale)
            : snapToSemitone(rawValue);
    const toleranceSemitone = Math.max(0, Number(pitchSnapToleranceCents ?? 0) / 100);
    if (Math.abs(rawValue - snapped) <= toleranceSemitone) {
        return rawValue;
    }
    return snapped + (rawValue - snapped > 0 ? 1 : -1) * toleranceSemitone;
}

export function formatRightDragMorphPercent(dragDelta: number): string {
    const percent = Math.round(dragDelta * 2);
    if (percent > 0) {
        return `+${percent}%`;
    }
    return `${percent}%`;
}
