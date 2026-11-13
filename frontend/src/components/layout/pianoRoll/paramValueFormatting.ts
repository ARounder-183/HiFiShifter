import { formatDbReadout, supportsParamAxisUnit, type ParamAxisUnit } from "./paramAxisUnits";
import { formatPanValue, isPanParam } from "./paramPan";
import {
    childPitchOffsetValueToDisplay,
    isChildPitchOffsetDegreesParam,
} from "./childPitchOffsetParams";
import { NOTE_NAMES_SHARP } from "./panel/panelHelpers";
import type { ParamName } from "./types";

/** 参数曲线读数的唯一格式化入口（轴浮窗、曲线浮窗与拖拽预览共用）。 */
export function formatParamValuePreviewValue(args: {
    readonly value: number;
    readonly editParam: ParamName;
    readonly axisUnit: ParamAxisUnit;
    readonly tf: (key: string) => string;
}): string {
    const { value, editParam, axisUnit, tf } = args;
    if (!Number.isFinite(value)) return "";
    if (isPanParam(editParam)) {
        return formatPanValue(value, {
            left: tf("pan_left"),
            center: tf("pan_center"),
            right: tf("pan_right"),
        });
    }
    if (editParam === "pitch") {
        const rounded = Math.round(value);
        const pitchClass = ((rounded % 12) + 12) % 12;
        const octave = Math.floor(rounded / 12) - 1;
        const noteName = `${NOTE_NAMES_SHARP[pitchClass]}${octave}`;
        const cents = Math.round((value - rounded) * 100);
        const signedCents = cents >= 0 ? `+${cents}` : `${cents}`;
        return `${noteName}${signedCents}`;
    }
    if (isChildPitchOffsetDegreesParam(editParam)) {
        const display = childPitchOffsetValueToDisplay(editParam, value);
        if (Math.abs(display) >= 100) return display.toFixed(1);
        if (Math.abs(display) >= 10) return display.toFixed(2);
        return display.toFixed(3);
    }
    if (axisUnit === "db" && supportsParamAxisUnit(editParam)) {
        return formatDbReadout(value);
    }
    if (Math.abs(value) >= 100) return value.toFixed(1);
    if (Math.abs(value) >= 10) return value.toFixed(2);
    return value.toFixed(3);
}
