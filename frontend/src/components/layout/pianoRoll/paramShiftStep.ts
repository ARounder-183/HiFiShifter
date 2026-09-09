import type { ProcessorParamDescriptor } from "../../../types/api.js";
import { childPitchOffsetShiftStep } from "./childPitchOffsetParams.js";

/**
 * 参数线平移的变化幅度档位：
 * - "normal" — 默认步长（与既有 "=" / "-" / "]" / "[" 行为一致）；
 * - "coarse" — 大幅变体（Shift 修饰）：以"语义单位"为尺度的大步进；
 * - "fine"   — 微调变体（Ctrl 修饰）：细粒度步进。
 */
export type ParamShiftMagnitude = "normal" | "coarse" | "fine";

/** 从事件/消息载荷中解析幅度档位（未知值回退为 normal）。 */
export function parseParamShiftMagnitude(value: unknown): ParamShiftMagnitude {
    if (value === "coarse" || value === "fine") {
        return value;
    }
    return "normal";
}

function normalizeStep(step: number): number {
    if (!Number.isFinite(step) || step <= 0) {
        return 0.05;
    }
    if (step >= 1) {
        return Number(step.toFixed(2));
    }
    if (step >= 0.1) {
        return Number(step.toFixed(2));
    }
    return Number(step.toFixed(3));
}

export function getParamShiftStep(
    paramId: string,
    descriptor?: ProcessorParamDescriptor,
    magnitude: ParamShiftMagnitude = "normal",
): number {
    const childStep = childPitchOffsetShiftStep(paramId, magnitude);
    if (childStep != null) {
        return childStep;
    }

    if (paramId === "pitch") {
        // pitch 帧值的单位是半音：
        // - 默认 ±1 半音（100 音分）；
        // - 微调 ±1 音分（0.01 半音）；
        // - 大幅 ±12 半音（1200 音分 = 一个八度）。
        switch (magnitude) {
            case "fine":
                return 0.01;
            case "coarse":
                return 12;
            default:
                return 1;
        }
    }

    if (descriptor?.kind.type === "automation_curve") {
        const range = Math.abs(descriptor.kind.max_value - descriptor.kind.min_value);
        // 连续量参数以自身量程为尺度：默认 2.5% 量程，微调 0.25%，
        // 大幅 12.5%（约为默认步长的 5 倍）。
        switch (magnitude) {
            case "fine":
                return normalizeStep(range / 400);
            case "coarse":
                return normalizeStep(range / 8);
            default:
                return normalizeStep(range / 40);
        }
    }

    // 无描述符的未知参数：与连续量参数同比例（默认 0.05 → 微调 0.005，
    // 大幅 0.25）。
    switch (magnitude) {
        case "fine":
            return 0.005;
        case "coarse":
            return 0.25;
        default:
            return 0.05;
    }
}
