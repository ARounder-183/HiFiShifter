/**
 * 出厂颤音预设。
 *
 * 【命名原则】按**听感**命名而不是按波形命名 —— 用户找的是"演歌那种颤音"，
 * 不是"梯形波"。波形只是实现细节，在预设编辑器里可见可改。
 *
 * 【只读】系统预设的 `builtin` 为 true，编辑器里禁用一切字段，只能
 * 「复制为自定义」后再改。这样既保证出厂预设永远可复原，又不会让"改坏了"
 * 变成不可逆。
 *
 * 【i18n】名称走显式 `Record` 映射（`BUILTIN_VIBRATO_PRESET_NAME_KEYS`），
 * 不做模板拼接 —— `keyReferenceIntegrity.test.ts` 要求 `labelKey` 的字面量
 * 必须真实存在于词条中，`` `vibrato_preset_${id}` `` 这类拼接过不了校验。
 */

import type { MessageKey } from "../../i18n/messages";
import { sanitizeVibratoPreset } from "./vibratoPresets";
import type { VibratoPreset, VibratoPresetInput } from "./vibratoTypes";

/** 系统预设的稳定 id（去前缀后的键）。 */
export type BuiltinVibratoId =
    | "natural"
    | "soft"
    | "deep"
    | "fine"
    | "fast"
    | "trill"
    | "enka"
    | "opera"
    | "breath"
    | "drift"
    | "synth"
    | "straight";

/** 系统预设名称的词条键。显式映射，禁止模板拼接。 */
export const BUILTIN_VIBRATO_PRESET_NAME_KEYS: Record<BuiltinVibratoId, MessageKey> = {
    natural: "vibrato_preset_natural",
    soft: "vibrato_preset_soft",
    deep: "vibrato_preset_deep",
    fine: "vibrato_preset_fine",
    fast: "vibrato_preset_fast",
    trill: "vibrato_preset_trill",
    enka: "vibrato_preset_enka",
    opera: "vibrato_preset_opera",
    breath: "vibrato_preset_breath",
    drift: "vibrato_preset_drift",
    synth: "vibrato_preset_synth",
    straight: "vibrato_preset_straight",
};

/** 系统预设的系统提示文案键（一行说明"适合什么"）。 */
export const BUILTIN_VIBRATO_PRESET_DESC_KEYS: Record<BuiltinVibratoId, MessageKey> = {
    natural: "vibrato_preset_natural_desc",
    soft: "vibrato_preset_soft_desc",
    deep: "vibrato_preset_deep_desc",
    fine: "vibrato_preset_fine_desc",
    fast: "vibrato_preset_fast_desc",
    trill: "vibrato_preset_trill_desc",
    enka: "vibrato_preset_enka_desc",
    opera: "vibrato_preset_opera_desc",
    breath: "vibrato_preset_breath_desc",
    drift: "vibrato_preset_drift_desc",
    synth: "vibrato_preset_synth_desc",
    straight: "vibrato_preset_straight_desc",
};

const SINE = { kind: "shape", shape: "sine", skew: 0.5 } as const;

/**
 * 出厂预设的参数表。
 *
 * 只写与默认值不同的字段（`sanitizeVibratoPreset` 补齐其余），这样加一个
 * 出厂预设只需要几行，且默认值变更会统一传播。
 */
const BUILTIN_SPECS: Record<BuiltinVibratoId, VibratoPresetInput> = {
    // 通用默认：正弦 + 短渐入，幅度和速率都取人声最常见的中心值。
    natural: {
        cycle: { ...SINE },
        depthCents: 30,
        rateHz: 5.5,
        attackMs: 90,
        releaseMs: 90,
        irregularity: 8,
    },
    // 长渐入 + 小幅度：适合抒情段的收尾，颤音"轻轻浮上来"。
    soft: {
        cycle: { ...SINE },
        depthCents: 18,
        rateHz: 5.0,
        attackMs: 180,
        releaseMs: 140,
        irregularity: 5,
    },
    // 大幅度 + 渐强：起点只给 40%，到句尾推到满幅。
    deep: {
        cycle: { ...SINE },
        depthCents: 55,
        rateHz: 4.5,
        attackMs: 140,
        releaseMs: 140,
        depthRamp: { start: 0.4, end: 1 },
        irregularity: 8,
    },
    // 三角波 + 高速小幅度：紧而细的"抖音"，线性渐入更干脆。
    fine: {
        cycle: { kind: "shape", shape: "triangle", skew: 0.5 },
        depthCents: 12,
        rateHz: 7.5,
        attackMs: 60,
        attackCurve: "linear",
        releaseMs: 60,
        releaseCurve: "linear",
        irregularity: 6,
    },
    // 流行尾音：起得快、收得略慢。
    fast: {
        cycle: { ...SINE },
        depthCents: 25,
        rateHz: 8.5,
        attackMs: 50,
        attackCurve: "linear",
        releaseMs: 80,
        irregularity: 5,
    },
    // 颤指：两个音高之间切换，深度拉到全音，几乎不要抖动。
    trill: {
        cycle: { kind: "shape", shape: "trill", skew: 0.5 },
        depthCents: 100,
        rateHz: 6.0,
        attackMs: 120,
        releaseMs: 100,
        irregularity: 0,
        alignCycles: true,
    },
    // 演歌：慢起 + 强渐强 + 末端加速，收在整数周期上。
    enka: {
        cycle: { kind: "shape", shape: "trapezoid", skew: 0.5 },
        depthCents: 70,
        rateHz: 4.0,
        rateRampEnd: 1.2,
        attackMs: 260,
        releaseMs: 180,
        depthRamp: { start: 0.35, end: 1 },
        irregularity: 10,
        alignCycles: true,
    },
    // 歌剧：匀速全幅，稳定而宽阔。
    opera: {
        cycle: { ...SINE },
        depthCents: 45,
        rateHz: 6.0,
        attackMs: 100,
        releaseMs: 120,
        irregularity: 4,
        alignCycles: true,
    },
    // 气息 / 气声音量：乘性调制（静音保持静音），在已有曲线上叠加。
    breath: {
        cycle: { ...SINE },
        depthCents: 28,
        rateHz: 5.5,
        attackMs: 60,
        attackCurve: "linear",
        releaseMs: 80,
        irregularity: 10,
        baseline: "existing",
        blend: 100,
    },
    // 摇曳：慢速 + 高不规则度，做出"不稳"的人味。
    drift: {
        cycle: { ...SINE },
        depthCents: 22,
        rateHz: 3.5,
        attackMs: 150,
        releaseMs: 150,
        irregularity: 45,
    },
    // 电音：上升锯齿 + 高速，机械感来自零抖动与整数周期。
    synth: {
        cycle: { kind: "shape", shape: "sawUp", skew: 0.8 },
        depthCents: 40,
        rateHz: 9.0,
        attackMs: 20,
        attackCurve: "linear",
        releaseMs: 40,
        releaseCurve: "linear",
        irregularity: 0,
        alignCycles: true,
    },
    // 直线：深度 0。直线/颤音工具因此共用一条代码路径 ——
    // 「直线」就是这个工具在零颤音时的样子。
    straight: {
        cycle: { ...SINE },
        depthCents: 0,
        rateHz: 5.5,
        attackMs: 0,
        releaseMs: 0,
        irregularity: 0,
        baseline: "line",
    },
};

/**
 * 出厂预设的默认顺序。
 *
 * 【为什么「直线」排首位】直线/颤音工具共用一条代码路径，「直线」就是这个工具在
 * 零颤音时的样子。把它放在列表最前，切到该工具后默认看到的就是最常用的那个
 * —— 先画直线、需要时再往上加颤音，比先落在一个颤音预设上更贴近实际用法。
 */
export const BUILTIN_VIBRATO_ORDER: readonly BuiltinVibratoId[] = [
    "straight",
    "natural",
    "soft",
    "deep",
    "fine",
    "fast",
    "trill",
    "enka",
    "opera",
    "breath",
    "drift",
    "synth",
];

/** 出厂预设的 id（`builtin.<key>`）。 */
export function builtinVibratoPresetId(id: BuiltinVibratoId): string {
    return `builtin.${id}`;
}

/** 全部出厂预设。每次调用返回新对象，调用方可安全改写。 */
export function buildSystemVibratoPresets(): VibratoPreset[] {
    return BUILTIN_VIBRATO_ORDER.map((key) =>
        sanitizeVibratoPreset({
            ...BUILTIN_SPECS[key],
            id: builtinVibratoPresetId(key),
            builtin: true,
        }),
    );
}

/**
 * 出厂预设的稳定列表（模块加载时构建一次）。
 *
 * 逐项冻结：系统预设是只读的，冻结让"不小心就地改写"变成一条明确的报错，
 * 而不是让出厂音色被静默污染。要改必须先 `duplicateVibratoPreset`。
 */
export const SYSTEM_VIBRATO_PRESETS: readonly VibratoPreset[] = Object.freeze(
    buildSystemVibratoPresets().map(
        (preset) =>
            Object.freeze({
                ...preset,
                cycle:
                    preset.cycle.kind === "table"
                        ? Object.freeze({
                              kind: "table" as const,
                              table: Object.freeze([...preset.cycle.table]),
                          })
                        : Object.freeze({ ...preset.cycle }),
                depthRamp: Object.freeze({ ...preset.depthRamp }),
            }) as VibratoPreset,
    ),
);

/** 默认活动预设 = 自然。 */
export const DEFAULT_ACTIVE_VIBRATO_PRESET_ID = builtinVibratoPresetId("natural");
