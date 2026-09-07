/**
 * Child-track pitch-offset parameter helpers for PianoRoll.
 *
 * This module centralizes:
 * - synthetic param IDs used by the parameter editor,
 * - child-track param ID parsing,
 * - axis ranges and snap steps,
 * - degree display formatting helpers.
 */

import { degreeInputToScaleSteps, scaleStepsToDegreeDisplay } from "../../../utils/musicalScales";

export const CHILD_PITCH_OFFSET_CENTS_PREFIX = "child_pitch_offset_cents@";
export const CHILD_PITCH_OFFSET_DEGREES_PREFIX = "child_pitch_offset_degrees@";
export const CHILD_FORMANT_OFFSET_CENTS_PREFIX = "child_formant_offset_cents@";

export const CHILD_PITCH_OFFSET_CENTS_RANGE = {
    min: -2400,
    max: 2400,
} as const;

export const CHILD_PITCH_OFFSET_DEGREES_RANGE = {
    // Internal scale-step range. UI displays this as degree labels [-15, 15].
    min: -14,
    max: 14,
} as const;

export const CHILD_FORMANT_OFFSET_CENTS_RANGE = {
    min: -2400,
    max: 2400,
} as const;

export function buildChildPitchOffsetCentsParam(trackId: string): string {
    return `${CHILD_PITCH_OFFSET_CENTS_PREFIX}${trackId}`;
}

export function buildChildPitchOffsetDegreesParam(trackId: string): string {
    return `${CHILD_PITCH_OFFSET_DEGREES_PREFIX}${trackId}`;
}

export function buildChildFormantOffsetCentsParam(trackId: string): string {
    return `${CHILD_FORMANT_OFFSET_CENTS_PREFIX}${trackId}`;
}

export function isChildFormantOffsetCentsParam(param: string): boolean {
    return param.startsWith(CHILD_FORMANT_OFFSET_CENTS_PREFIX);
}

export function isChildPitchOffsetCentsParam(param: string): boolean {
    return param.startsWith(CHILD_PITCH_OFFSET_CENTS_PREFIX);
}

export function isChildPitchOffsetDegreesParam(param: string): boolean {
    return param.startsWith(CHILD_PITCH_OFFSET_DEGREES_PREFIX);
}

export function isChildPitchOffsetParam(param: string): boolean {
    return (
        isChildPitchOffsetCentsParam(param) ||
        isChildPitchOffsetDegreesParam(param) ||
        isChildFormantOffsetCentsParam(param)
    );
}

export function parseChildPitchOffsetParam(
    param: string,
): { mode: "cents" | "degrees" | "formant"; trackId: string } | null {
    if (isChildPitchOffsetCentsParam(param)) {
        return {
            mode: "cents",
            trackId: param.slice(CHILD_PITCH_OFFSET_CENTS_PREFIX.length),
        };
    }
    if (isChildPitchOffsetDegreesParam(param)) {
        return {
            mode: "degrees",
            trackId: param.slice(CHILD_PITCH_OFFSET_DEGREES_PREFIX.length),
        };
    }
    if (isChildFormantOffsetCentsParam(param)) {
        return {
            mode: "formant",
            trackId: param.slice(CHILD_FORMANT_OFFSET_CENTS_PREFIX.length),
        };
    }
    return null;
}

export function snapChildPitchOffsetValue(param: string, value: number): number {
    if (!Number.isFinite(value)) return 0;
    if (isChildPitchOffsetCentsParam(param)) {
        return Math.round(value / 100) * 100;
    }
    if (isChildPitchOffsetDegreesParam(param)) {
        return Math.round(value);
    }
    if (isChildFormantOffsetCentsParam(param)) {
        return Math.round(value / 50) * 50;
    }
    return value;
}

export type ParamShiftMagnitude = "normal" | "coarse" | "fine";

export function childPitchOffsetShiftStep(
    param: string,
    magnitude: ParamShiftMagnitude = "normal",
): number | null {
    if (isChildPitchOffsetCentsParam(param)) {
        // 音高偏移以音分为单位：默认 ±100（一个半音），微调 ±1 音分，
        // 大幅 ±1200（一个八度）。
        switch (magnitude) {
            case "fine":
                return 1;
            case "coarse":
                return 1200;
            default:
                return 100;
        }
    }
    if (isChildPitchOffsetDegreesParam(param)) {
        // 度数（音级）已是 granularity 最小的音乐单位：微调与默认相同
        // （±1 音级）；大幅 ±7 ≈ 七声音阶的一个八度（与音级量程 ±14 =
        // ±2 个八度一致）。
        switch (magnitude) {
            case "coarse":
                return 7;
            default:
                return 1;
        }
    }
    if (isChildFormantOffsetCentsParam(param)) {
        // 共振峰偏移同样以音分为单位：默认 ±50（与 UI 吸附步长一致），
        // 微调 ±1 音分，大幅 ±1200（一个八度）。
        switch (magnitude) {
            case "fine":
                return 1;
            case "coarse":
                return 1200;
            default:
                return 50;
        }
    }
    return null;
}

export function childPitchOffsetValueToDisplay(param: string, value: number): number {
    if (!Number.isFinite(value)) return 0;
    if (isChildPitchOffsetDegreesParam(param)) {
        return scaleStepsToDegreeDisplay(value);
    }
    return value;
}

export function childPitchOffsetDisplayToInternal(
    mode: "cents" | "degrees" | "formant",
    value: number,
): number {
    if (!Number.isFinite(value)) return 0;
    if (mode === "degrees") {
        return degreeInputToScaleSteps(value);
    }
    if (mode === "formant") {
        return Math.round(value / 50) * 50;
    }
    return value;
}
