/**
 * 直线/颤音拖拽时的调参映射与步进计算。
 *
 * 【本文件负责什么】把"滚轮 / 方向键 / 侧键 / 预设切换"这几个输入，映射成
 * 拖拽工作副本（`VibratoDragWorking`）上的深度、速率与预设变化。纯函数，
 * 不碰 React、不碰画布 —— 手感规则因此可以单测。
 *
 * 【工作副本的语义】拖拽永远在活动预设之上维护一份**工作副本**：
 * - 起手时从活动预设取深度与速率；
 * - 拖拽中滚轮 / 方向键改的是副本，**不改预设本身**，也不跨手势保留；
 * - 切换预设时副本整体换成新预设 —— 用户换的是"音色"，把上一个预设的调整
 *   叠加到新预设上会得到一个既不是 A 也不是 B 的东西。
 *
 * 预设本身只能在预设编辑器里改。拖拽悄悄改写预设是这类工具最恼人的错法：
 * 用户下次用同一个预设时，音色已经和上次不一样了，且无从察觉。
 */

import type { Keybinding } from "../../../features/keybindings/types";
import { matchesKeybindingAllowingFineModifier } from "../../../features/keybindings/keybindingMatch";
import { isModifierActive, isNoneBinding } from "../../../features/keybindings/keybindingsSlice";
import { buildVibratoCurve } from "../../../features/vibrato/vibratoCurve";
import {
    clampDepthCentsForParam,
    depthStepCentsFor,
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
    /**
     * 本次拖拽是否调过深度 / 速率。
     *
     * 【为什么分两个标记】两者各自独立：只调了振幅时切换预设，应当只把振幅带过去、
     * 速率仍取新预设自带的值。合成一个 `adjusted` 会让"没碰过的那个"也被一起继承。
     */
    depthAdjusted: boolean;
    rateAdjusted: boolean;
}

/**
 * 建立拖拽工作副本：深度与速率一律取预设自带值。
 *
 * 【为什么没有"续上一次"】预设切换是**持久化**的（切换即写活动预设），拖拽中的
 * 滚轮 / 方向键微调只属于本次手势 —— 若把上一次的调整跨预设地带进下一次起手，
 * "换了个音色深度却没变"的困惑就回来了。要保留调整，去管理器里改预设。
 *
 * 【为什么在这里就钳深度】预设是跨参数共用的，它的深度以 cents 存储，落到窄
 * 量程的参数上（声像 ±1、共振峰 ±500）可能远超该参数能表达的幅度 —— 不钳的话
 * 拖出来的曲线会被写入口钳平，顶部变成一条直线。钳制按**当前参数**的满摆幅
 * 走（见 `fullSwingCentsFor`），因此音高上的大深度不受影响。
 */
export function createDragWorking(
    preset: VibratoPreset,
    param: ParamName,
    range?: VibratoParamRange,
): VibratoDragWorking {
    return {
        preset,
        depthCents: clampDepthCentsForParam(preset.depthCents, param, range),
        rateHz: preset.rateHz,
        depthAdjusted: false,
        rateAdjusted: false,
    };
}

/**
 * 切换到另一个预设。
 *
 * 【切换时的深度 / 速率归属】用户切换预设时，波形、包络、摆放方式这些"音色"
 * 一律换成新预设的；但**本次手势调过的**那一个量要跟着走 —— 用户调完幅度再换
 * 预设，期待的是"换个音色、幅度不变"。两个量各自独立判断：
 * - 调过深度 → 沿用工作副本的深度；否则取新预设自带的深度。
 * - 调过速率 → 沿用工作副本的速率；否则取新预设自带的速率。
 *
 * 无论取自哪一边，深度都按当前参数的满摆幅钳一次（理由同 `createDragWorking`）。
 */
export function switchDragPreset(
    previous: VibratoDragWorking,
    next: VibratoPreset,
    param: ParamName,
    range?: VibratoParamRange,
): VibratoDragWorking {
    return {
        preset: next,
        depthCents: clampDepthCentsForParam(
            previous.depthAdjusted ? previous.depthCents : next.depthCents,
            param,
            range,
        ),
        rateHz: previous.rateAdjusted ? previous.rateHz : next.rateHz,
        depthAdjusted: previous.depthAdjusted,
        rateAdjusted: previous.rateAdjusted,
    };
}

/**
 * 重置本次手势的振幅：回到**预设自带**的深度，并清掉"调过振幅"的记录。
 *
 * 【重置成什么】回到预设自带值，而不是归零 —— 归零是「重置到直线」的语义
 * （它换的是预设本身）。这里撤销的是"我对幅度的微调"，撤销之后本次手势的振幅
 * 就等于预设的振幅，记录自然也不必再留着。
 */
export function resetVibratoDragDepth(
    working: VibratoDragWorking,
    param: ParamName,
    range?: VibratoParamRange,
): VibratoDragWorking {
    return {
        ...working,
        depthCents: clampDepthCentsForParam(working.preset.depthCents, param, range),
        depthAdjusted: false,
    };
}

/** 重置本次手势的速率：回到预设自带的速率，并清掉"调过速率"的记录。 */
export function resetVibratoDragRate(working: VibratoDragWorking): VibratoDragWorking {
    return { ...working, rateHz: working.preset.rateHz, rateAdjusted: false };
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
 * 每格走多少由 `depthStepCentsFor` 按**参数类型**决定（满摆幅的 1/50），
 * 结果再按同一满摆幅钳住 —— 于是"一格""到顶"在每个参数上都是同一量级的响应。
 * 深度可为负：负值等于把波形整体反相（起点先往下摆），与预设深度的合法区间一致。
 * 速率钳在预设的合法区间内。
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
        depthCents = clampDepthCentsForParam(
            depthCents + input.direction * step * safeSteps * safeFineScale,
            input.editParam,
            input.currentParamRange,
        );
    } else {
        const ratio = Math.pow(1 + 0.1 * safeFineScale, safeSteps);
        rateHz = input.direction > 0 ? rateHz * ratio : rateHz / ratio;
        rateHz = Math.max(0.1, Math.min(20, rateHz));
    }

    return { depthCents, rateHz };
}

/** 鼠标侧键在 `MouseEvent.button` 上的取值。 */
export const SIDE_BUTTON_BACK = 3;
export const SIDE_BUTTON_FORWARD = 4;

/** 鼠标侧键在 `MouseEvent.buttons` 位掩码上的位。 */
export const SIDE_BUTTON_BACK_MASK = 8;
export const SIDE_BUTTON_FORWARD_MASK = 16;

/**
 * 把鼠标侧键映射成"上一个 / 下一个预设"。
 *
 * 【为什么单独抽出来】侧键的 `button`（索引 3 / 4）与 `buttons`（位掩码 8 / 16）
 * 是两套编号 —— 这正是 `penInput.ts` 里"橡皮端是位 32 不是位 2"那条注释警告的
 * 同一类陷阱，写错一位就变成"按了没反应"或"按了乱跳"，而后者不会报错。
 *
 * 约定与键盘一致：**前进键 = 下一个**（`+1`），后退键 = 上一个（`-1`）。
 *
 * @returns `1` = 下一个，`-1` = 上一个，`null` = 不是侧键。
 */
export function resolveVibratoSideButton(button: number): 1 | -1 | null {
    if (button === SIDE_BUTTON_FORWARD) return 1;
    if (button === SIDE_BUTTON_BACK) return -1;
    return null;
}

/**
 * 拖拽期间可参与"双键重置"的绑定槽。
 *
 * 用槽名而不是键名：用户可以把任一动作改绑到任意键，判定必须按**绑定**走。
 */
export type VibratoDragResetSlot =
    | "presetPrev"
    | "presetNext"
    | "amplitudeIncrease"
    | "amplitudeDecrease"
    | "frequencyIncrease"
    | "frequencyDecrease";

export interface VibratoDragResetBindings {
    presetPrev: Keybinding;
    presetNext: Keybinding;
    amplitudeIncrease: Keybinding;
    amplitudeDecrease: Keybinding;
    frequencyIncrease: Keybinding;
    frequencyDecrease: Keybinding;
}

/**
 * 事件命中了哪些绑定槽。
 *
 * 返回数组而不是单个：两个动作可以被绑到同一个键上，此时一次按键同时命中两槽，
 * "双键重置"会立刻成立 —— 这是该配置下唯一说得通的解释。`modifierOnly` 与
 * "无"绑定一律不参与（它们不是可以"按住"的实体键）。
 */
export function matchedVibratoResetSlots(
    event: KeyboardEvent,
    bindings: VibratoDragResetBindings,
    fineAdjustKb?: Keybinding,
): VibratoDragResetSlot[] {
    const slots: VibratoDragResetSlot[] = [];
    const test = (slot: VibratoDragResetSlot, kb: Keybinding) => {
        if (isNoneBinding(kb) || kb.modifierOnly) return;
        if (matchesKeybindingAllowingFineModifier(event, kb, fineAdjustKb)) slots.push(slot);
    };
    test("presetPrev", bindings.presetPrev);
    test("presetNext", bindings.presetNext);
    test("amplitudeIncrease", bindings.amplitudeIncrease);
    test("amplitudeDecrease", bindings.amplitudeDecrease);
    test("frequencyIncrease", bindings.frequencyIncrease);
    test("frequencyDecrease", bindings.frequencyDecrease);
    return slots;
}

/** 双键同时按下时的重置意图。 */
export type VibratoDragResetIntent = "straight" | "depth" | "rate";

/**
 * 一对绑定同时按下 → 重置。
 *
 * 优先级：预设切换对 > 振幅对 > 频率对。三对全按满（六个键）时按这个顺序取一个，
 * 不会三件事一起做 —— 同时重置预设、振幅与速率既难解释，也没有实际需求。
 */
export function resolveVibratoPairReset(
    held: ReadonlySet<VibratoDragResetSlot>,
): VibratoDragResetIntent | null {
    if (held.has("presetPrev") && held.has("presetNext")) return "straight";
    if (held.has("amplitudeIncrease") && held.has("amplitudeDecrease")) return "depth";
    if (held.has("frequencyIncrease") && held.has("frequencyDecrease")) return "rate";
    return null;
}

/**
 * 中键在拖拽中按下时，判定当前滚轮处于哪一路调整 —— 决定"重置振幅"还是"重置频率"。
 *
 * 与滚轮的判定同源：频率修饰键生效时滚轮调速率，否则调振幅。中键拿不到滚轮的
 * 横向分量（触摸板的横向手势不参与），因此只看修饰键状态。
 */
export function resolveVibratoMiddleClickReset(
    event: { altKey: boolean; ctrlKey: boolean; shiftKey: boolean; metaKey?: boolean },
    frequencyAdjustKb: Keybinding,
): "depth" | "rate" {
    const rateRequested =
        isNoneBinding(frequencyAdjustKb) || isModifierActive(frequencyAdjustKb, event);
    return rateRequested ? "rate" : "depth";
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
