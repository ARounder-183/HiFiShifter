/**
 * 直线/颤音拖拽时的调参映射与步进计算。
 *
 * 【本文件负责什么】把"滚轮 / 方向键 / 侧键 / 预设切换"这几个输入，映射成
 * 拖拽工作副本（`VibratoDragWorking`）上的深度、速率与预设变化。纯函数，
 * 不碰 React、不碰画布 —— 手感规则因此可以单测。
 *
 * 【工作副本的语义】拖拽永远在活动预设之上维护一份**工作副本**：
 * - 起手时从活动预设取深度与速率；`lastVibrato*` 有值时优先用它续上；
 * - 拖拽中滚轮 / 方向键改的是副本，**不改预设本身**；
 * - 切换预设时副本整体换成新预设 —— 用户换的是"音色"，把上一个预设的调整
 *   叠加到新预设上会得到一个既不是 A 也不是 B 的东西。
 *
 * 预设本身只能在预设编辑器里改。拖拽悄悄改写预设是这类工具最恼人的错法：
 * 用户下次用同一个预设时，音色已经和上次不一样了，且无从察觉。
 */

import type { Keybinding } from "../../../features/keybindings/types";
import { matchesKeybindingAllowingFineModifier } from "../../../features/keybindings/keybindingMatch";
import { isNoneBinding } from "../../../features/keybindings/keybindingsSlice";
import { buildVibratoCurve } from "../../../features/vibrato/vibratoCurve";
import {
    depthFamilyOf,
    PITCH_PARAM_ID,
    type VibratoParamRange,
} from "../../../features/vibrato/vibratoDepth";
import type { VibratoPreset } from "../../../features/vibrato/vibratoTypes";
import type { ParamName } from "./types";

export type VibratoAdjustTarget = "depth" | "rate";
export type VibratoAdjustDirection = 1 | -1;

export type VibratoDragKeyboardBindings = {
    amplitudeIncrease: Keybinding;
    amplitudeDecrease: Keybinding;
    frequencyIncrease: Keybinding;
    frequencyDecrease: Keybinding;
};

export type VibratoDragKeyboardAdjustment = {
    target: VibratoAdjustTarget;
    direction: VibratoAdjustDirection;
};

/** 拖拽中的工作副本。 */
export interface VibratoDragWorking {
    /** 当前预设（切换预设时整体替换）。 */
    preset: VibratoPreset;
    /** 本次拖拽的深度（cents）：起手自预设，可被滚轮 / 方向键改动。 */
    depthCents: number;
    /** 本次拖拽的速率（Hz）。 */
    rateHz: number;
    /** 本次拖拽是否被用户调整过（决定是否回写 `lastVibrato*`）。 */
    adjusted: boolean;
}

/**
 * 建立拖拽工作副本。
 *
 * @param preset 活动预设。
 * @param lastDepthCents 上次拖拽后的深度；`null` 表示用预设自带值。
 * @param lastRateHz 上次拖拽后的速率；`null` 表示用预设自带值。
 */
export function createDragWorking(
    preset: VibratoPreset,
    lastDepthCents: number | null,
    lastRateHz: number | null,
): VibratoDragWorking {
    return {
        preset,
        depthCents: Number.isFinite(lastDepthCents)
            ? (lastDepthCents as number)
            : preset.depthCents,
        rateHz: Number.isFinite(lastRateHz) ? (lastRateHz as number) : preset.rateHz,
        adjusted: false,
    };
}

/** 切换到另一个预设：深度与速率整体换成新预设的值。 */
export function switchDragPreset(next: VibratoPreset): VibratoDragWorking {
    return {
        preset: next,
        depthCents: next.depthCents,
        rateHz: next.rateHz,
        adjusted: false,
    };
}

/**
 * 拖拽调参每一步改变多少深度（cents）。
 *
 * 【为什么以分计】深度在预设里就是 cents，拖拽时用户对"幅度"的直觉也是分
 * （"再多 20 分"）。各族取值：
 * - 音高：24 分 / 格 —— 与历史实现 `rangeSpan / 200`（48 半音 / 200 = 0.24
 *   半音）完全一致，手感不变；
 * - cents 类参数：值域的 1/200，下限 1 分；
 * - 乘性增益（`dyn` / `volume` / `breath_gain`）：1 分 = 1% / 格；
 * - 其余原始值域：半量程的 1/200，换算回分恒为 0.5。
 */
export function depthStepCentsFor(param: ParamName, range?: VibratoParamRange): number {
    if (param === PITCH_PARAM_ID) return 24;
    const family = depthFamilyOf(param);
    // 乘性增益的显示单位就是百分比，1 分 = 1%。
    if (family === "ratio") return 1;
    const span = Number(range ? range.max - range.min : 0);
    if (family === "cents") {
        return Math.max(1, (Number.isFinite(span) && span > 0 ? span : 4800) / 200);
    }
    return 0.5;
}

export function resolveVibratoDragKeyboardAdjustment(
    event: KeyboardEvent,
    bindings: VibratoDragKeyboardBindings,
    fineAdjustKb?: Keybinding,
): VibratoDragKeyboardAdjustment | null {
    if (bindings.amplitudeIncrease.modifierOnly || bindings.amplitudeDecrease.modifierOnly) {
        return null;
    }
    if (bindings.frequencyIncrease.modifierOnly || bindings.frequencyDecrease.modifierOnly) {
        return null;
    }

    if (matchesKeybindingAllowingFineModifier(event, bindings.amplitudeIncrease, fineAdjustKb)) {
        return { target: "depth", direction: 1 };
    }
    if (matchesKeybindingAllowingFineModifier(event, bindings.amplitudeDecrease, fineAdjustKb)) {
        return { target: "depth", direction: -1 };
    }
    if (matchesKeybindingAllowingFineModifier(event, bindings.frequencyIncrease, fineAdjustKb)) {
        return { target: "rate", direction: 1 };
    }
    if (matchesKeybindingAllowingFineModifier(event, bindings.frequencyDecrease, fineAdjustKb)) {
        return { target: "rate", direction: -1 };
    }

    return null;
}

/**
 * 判断事件是否命中"上一个 / 下一个预设"的绑定。
 *
 * 与深度 / 速率的解析分开：预设切换是**离散跳转**而不是连续调参，误判的代价
 * 是"拖着拖着音色变了"，所以单独一条路径、单独测。
 *
 * @returns `1` = 下一个，`-1` = 上一个，`null` = 未命中。
 */
export function resolveVibratoPresetSwitch(
    event: KeyboardEvent,
    prevKb: Keybinding,
    nextKb: Keybinding,
    fineAdjustKb?: Keybinding,
): 1 | -1 | null {
    const prevDisabled = isNoneBinding(prevKb);
    const nextDisabled = isNoneBinding(nextKb);
    if (prevDisabled && nextDisabled) return null;
    if (!nextDisabled && matchesKeybindingAllowingFineModifier(event, nextKb, fineAdjustKb)) {
        return 1;
    }
    if (!prevDisabled && matchesKeybindingAllowingFineModifier(event, prevKb, fineAdjustKb)) {
        return -1;
    }
    return null;
}

/**
 * 计算一次调参后的深度 / 速率。
 *
 * 深度是加性的（分），速率是几何的（等比缩放）—— 速率用加性会在低速端过于
 * 敏感：4 Hz 加 1 Hz 是 +25%，12 Hz 加 1 Hz 只有 +8%。
 *
 * 深度下限为 0（= 直线预设，用来画直线）。速率钳在预设的合法区间内。
 */
export function computeVibratoDragAdjustment(input: {
    editParam: ParamName;
    currentParamRange?: VibratoParamRange;
    depthCents: number;
    rateHz: number;
    target: VibratoAdjustTarget;
    direction: VibratoAdjustDirection;
    steps: number;
    fineScale: number;
}): { depthCents: number; rateHz: number } {
    const safeSteps = Math.max(1, Math.round(Math.abs(input.steps)));
    const safeFineScale =
        Number.isFinite(input.fineScale) && input.fineScale > 0 ? input.fineScale : 1;

    let depthCents = input.depthCents;
    let rateHz = input.rateHz;

    if (input.target === "depth") {
        const step = depthStepCentsFor(input.editParam, input.currentParamRange);
        depthCents = Math.max(0, depthCents + input.direction * step * safeSteps * safeFineScale);
    } else {
        const ratio = Math.pow(1 + 0.1 * safeFineScale, safeSteps);
        rateHz = input.direction > 0 ? rateHz * ratio : rateHz / ratio;
        rateHz = Math.max(0.1, Math.min(20, rateHz));
    }

    return { depthCents, rateHz };
}

/**
 * 按工作副本渲染一整段颤音曲线。
 *
 * 工作副本的深度 / 速率覆盖预设自身的值，其余参数（波形、包络、基线…）
 * 一律来自预设。
 */
export function buildDragVibratoCurve(input: {
    working: VibratoDragWorking;
    startFrame: number;
    startValue: number;
    endFrame: number;
    endValue: number;
    param: ParamName;
    framePeriodMs: number;
    range?: VibratoParamRange;
    /** 逐帧吸附：作用于合成后的值（保留既有的量化画线行为）。 */
    snapFinalValue?: (value: number, frame: number) => number;
    original?: ArrayLike<number>;
    seed?: number;
}): { minF: number; maxF: number; dense: number[] } {
    const { working } = input;
    return buildVibratoCurve({
        startFrame: input.startFrame,
        startValue: input.startValue,
        endFrame: input.endFrame,
        endValue: input.endValue,
        original: input.original,
        preset: {
            ...working.preset,
            depthCents: working.depthCents,
            rateHz: working.rateHz,
            // 拖拽期间一律按 Hz：按周期数会在拖拽过程中不断把固定个数重新
            // 均分到一直在变的选区长度上，听感是"越拖越乱"。
            rateMode: "hz",
        },
        param: input.param,
        framePeriodMs: input.framePeriodMs,
        range: input.range,
        snapFinalValue: input.snapFinalValue,
        seed: input.seed,
    });
}
