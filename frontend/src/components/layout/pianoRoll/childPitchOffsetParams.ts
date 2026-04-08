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

export const CHILD_PITCH_OFFSET_CENTS_RANGE = {
    min: -2400,
    max: 2400,
} as const;

export const CHILD_PITCH_OFFSET_DEGREES_RANGE = {
    // Internal scale-step range. UI displays this as degree labels [-15, 15].
    min: -14,
    max: 14,
} as const;

export function buildChildPitchOffsetCentsParam(trackId: string): string {
    return `${CHILD_PITCH_OFFSET_CENTS_PREFIX}${trackId}`;
}

export function buildChildPitchOffsetDegreesParam(trackId: string): string {
    return `${CHILD_PITCH_OFFSET_DEGREES_PREFIX}${trackId}`;
}

export function isChildPitchOffsetCentsParam(param: string): boolean {
    return param.startsWith(CHILD_PITCH_OFFSET_CENTS_PREFIX);
}

export function isChildPitchOffsetDegreesParam(param: string): boolean {
    return param.startsWith(CHILD_PITCH_OFFSET_DEGREES_PREFIX);
}

export function isChildPitchOffsetParam(param: string): boolean {
    return isChildPitchOffsetCentsParam(param) || isChildPitchOffsetDegreesParam(param);
}

export function parseChildPitchOffsetParam(
    param: string,
): { mode: "cents" | "degrees"; trackId: string } | null {
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
    return value;
}

export function childPitchOffsetShiftStep(param: string): number | null {
    if (isChildPitchOffsetCentsParam(param)) return 100;
    if (isChildPitchOffsetDegreesParam(param)) return 1;
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
    mode: "cents" | "degrees",
    value: number,
): number {
    if (!Number.isFinite(value)) return 0;
    if (mode === "degrees") {
        return degreeInputToScaleSteps(value);
    }
    return value;
}
