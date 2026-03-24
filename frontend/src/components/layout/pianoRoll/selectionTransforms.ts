import type { ParamName } from "./types";

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

function isPitchParam(editParam: ParamName): boolean {
    return editParam === "pitch";
}

function isEditableValue(editParam: ParamName, value: number): boolean {
    return Number.isFinite(value) && (!isPitchParam(editParam) || value !== 0);
}

export function computeSelectionMean(
    values: number[],
    editParam: ParamName,
): number {
    let sum = 0;
    let count = 0;
    for (const value of values) {
        if (!isEditableValue(editParam, value)) {
            continue;
        }
        sum += value;
        count += 1;
    }
    return count > 0 ? sum / count : 0;
}

export function averageSelectionValues(
    values: number[],
    editParam: ParamName,
    strengthPercent: number,
): number[] {
    const strength = clamp((Number(strengthPercent) || 0) / 100, 0, 1);
    if (strength <= 0) {
        return values.slice();
    }
    const mean = computeSelectionMean(values, editParam);
    return values.map((value) => {
        if (!isEditableValue(editParam, value)) {
            return value;
        }
        return value + (mean - value) * strength;
    });
}

export function scaleSelectionDeviation(
    values: number[],
    editParam: ParamName,
    scale: number,
): number[] {
    const mean = computeSelectionMean(values, editParam);
    return values.map((value) => {
        if (!isEditableValue(editParam, value)) {
            return value;
        }
        return mean + (value - mean) * scale;
    });
}

export function smoothSelectionValues(
    values: number[],
    editParam: ParamName,
    strengthUnits: number,
): number[] {
    const units = Math.max(0, Number(strengthUnits) || 0);
    if (units <= 0 || values.length === 0) {
        return values.slice();
    }

    const blend = clamp(units, 0, 1);
    const radius = Math.max(1, Math.round(units * 50));
    const passes = Math.max(1, Math.round(units * 3));
    const skipMask = values.map((value) => !isEditableValue(editParam, value));
    let buffer = values.slice();

    for (let pass = 0; pass < passes; pass += 1) {
        const next = new Array<number>(buffer.length);
        for (let index = 0; index < buffer.length; index += 1) {
            if (skipMask[index]) {
                next[index] = values[index] ?? 0;
                continue;
            }
            const low = Math.max(0, index - radius);
            const high = Math.min(buffer.length - 1, index + radius);
            let sum = 0;
            let count = 0;
            for (let cursor = low; cursor <= high; cursor += 1) {
                if (skipMask[cursor]) {
                    continue;
                }
                sum += buffer[cursor] ?? 0;
                count += 1;
            }
            next[index] = count > 0 ? sum / count : buffer[index] ?? 0;
        }
        buffer = next;
    }

    return values.map((value, index) => {
        if (skipMask[index]) {
            return value;
        }
        return value + ((buffer[index] ?? value) - value) * blend;
    });
}

export function rightDragDownSmoothStrength(
    dragDelta: number,
): number {
    return Math.max(0, -dragDelta / 50);
}

export function transformSelectionByRightDrag(
    values: number[],
    editParam: ParamName,
    dragDelta: number,
): number[] {
    if (dragDelta >= 0) {
        const deviationPercent = dragDelta * 2;
        const scale = Math.max(0, 1 + deviationPercent / 100);
        return scaleSelectionDeviation(values, editParam, scale);
    }
    return smoothSelectionValues(
        values,
        editParam,
        rightDragDownSmoothStrength(dragDelta),
    );
}