/**
 * 颤音预设的规整与默认值。
 *
 * 【定位】与 `utils/customScales.ts` 同构：后端只做透传存储，字段语义、
 * 取值合法性与默认值全部在前端收口。手改过 / 旧版本的 `app_config.json`
 * 必须能在这里被无异常地拉回合法状态。
 */

import { CYCLE_TABLE_DEFAULT_LEN, normalizeCycleTable, wrap01 } from "./vibratoCycle";
import type {
    BaselineMode,
    CycleSource,
    EnvelopeCurve,
    VibratoPreset,
    VibratoPresetInput,
    VibratoRateMode,
    WaveShape,
} from "./vibratoTypes";

/** 新建用户预设的 id 前缀。 */
export const CUSTOM_VIBRATO_PREFIX = "custom_";
/** 系统预设的 id 前缀。 */
export const BUILTIN_VIBRATO_PREFIX = "builtin.";

const WAVE_SHAPES: readonly WaveShape[] = [
    "sine",
    "triangle",
    "sawUp",
    "sawDown",
    "square",
    "trapezoid",
    "trill",
];
const BASELINE_MODES: readonly BaselineMode[] = [
    "line",
    "holdStart",
    "holdEnd",
    "average",
    "existing",
];
const ENVELOPE_CURVES: readonly EnvelopeCurve[] = ["linear", "exp", "s"];
const RATE_MODES: readonly VibratoRateMode[] = ["hz", "cycles"];

/** 各字段的合法区间。浏览器手改配置只能落进这些范围。 */
export const VIBRATO_LIMITS = {
    depthCents: { min: 0, max: 1200 },
    rateHz: { min: 0.1, max: 20 },
    cycles: { min: 0.5, max: 128 },
    rateRampEnd: { min: 0.25, max: 4 },
    attackMs: { min: 0, max: 10000 },
    releaseMs: { min: 0, max: 10000 },
    depthRamp: { min: 0, max: 2 },
    irregularity: { min: 0, max: 100 },
    biasCents: { min: -200, max: 200 },
    blend: { min: 0, max: 100 },
} as const;

/** 预设的出厂默认值（`sanitizeVibratoPreset` 的兜底来源）。 */
export const DEFAULT_VIBRATO_PRESET: VibratoPreset = {
    id: "",
    name: "",
    builtin: false,
    cycle: { kind: "shape", shape: "sine", skew: 0.5 },
    depthCents: 30,
    rateMode: "hz",
    rateHz: 5.5,
    cycles: 6,
    rateRampEnd: 1,
    startPhaseDeg: 0,
    attackMs: 90,
    attackCurve: "exp",
    releaseMs: 90,
    releaseCurve: "exp",
    depthRamp: { start: 1, end: 1 },
    alignCycles: false,
    irregularity: 0,
    biasCents: 0,
    baseline: "line",
    blend: 100,
};

/** 新建用户预设的 id。 */
export function createVibratoPresetId(): string {
    return `${CUSTOM_VIBRATO_PREFIX}${Math.random().toString(36).slice(2, 10)}`;
}

/** 判定是否为系统预设（按 id 而非 `builtin` 字段，避免字段被手改后失配）。 */
export function isBuiltinVibratoPresetId(id: string): boolean {
    return id.startsWith(BUILTIN_VIBRATO_PREFIX);
}

function clampNumber(raw: unknown, min: number, max: number, fallback: number): number {
    const value = Number(raw);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, value));
}

function pickEnum<T extends string>(raw: unknown, allowed: readonly T[], fallback: T): T {
    return typeof raw === "string" && (allowed as readonly string[]).includes(raw)
        ? (raw as T)
        : fallback;
}

/** 规整周期波形来源。 */
export function sanitizeCycleSource(raw: unknown): CycleSource {
    const source = raw as Partial<CycleSource> | undefined;
    if (source && source.kind === "table") {
        const table = (source as { table?: unknown }).table;
        const normalized = normalizeCycleTable(table);
        return { kind: "table", table: normalized.length ? normalized : normalizeCycleTable(null) };
    }
    if (source && source.kind === "shape") {
        return {
            kind: "shape",
            shape: pickEnum((source as { shape?: unknown }).shape, WAVE_SHAPES, "sine"),
            skew: clampNumber((source as { skew?: unknown }).skew, 0.02, 0.98, 0.5),
        };
    }
    return { ...DEFAULT_VIBRATO_PRESET.cycle };
}

/**
 * 把任意输入拉回一个合法的颤音预设。
 *
 * 【`id` 的处理】空 id 会生成新 id；但 `builtin` 前缀不可通过此函数伪造 ——
 * 调用方传入 `builtin: true` 时保留 id，用于系统预设表自身。
 */
export function sanitizeVibratoPreset(input: VibratoPresetInput | null | undefined): VibratoPreset {
    const raw = (input ?? {}) as VibratoPresetInput;
    const id = String(raw.id ?? "").trim() || createVibratoPresetId();
    const builtin = isBuiltinVibratoPresetId(id);
    const depthRamp = (raw.depthRamp ?? {}) as { start?: unknown; end?: unknown };

    return {
        id,
        name: String(raw.name ?? "").trim(),
        builtin,
        cycle: sanitizeCycleSource(raw.cycle),
        depthCents: clampNumber(
            raw.depthCents,
            VIBRATO_LIMITS.depthCents.min,
            VIBRATO_LIMITS.depthCents.max,
            DEFAULT_VIBRATO_PRESET.depthCents,
        ),
        rateMode: pickEnum(raw.rateMode, RATE_MODES, DEFAULT_VIBRATO_PRESET.rateMode),
        rateHz: clampNumber(
            raw.rateHz,
            VIBRATO_LIMITS.rateHz.min,
            VIBRATO_LIMITS.rateHz.max,
            DEFAULT_VIBRATO_PRESET.rateHz,
        ),
        cycles: clampNumber(
            raw.cycles,
            VIBRATO_LIMITS.cycles.min,
            VIBRATO_LIMITS.cycles.max,
            DEFAULT_VIBRATO_PRESET.cycles,
        ),
        rateRampEnd: clampNumber(
            raw.rateRampEnd,
            VIBRATO_LIMITS.rateRampEnd.min,
            VIBRATO_LIMITS.rateRampEnd.max,
            DEFAULT_VIBRATO_PRESET.rateRampEnd,
        ),
        // 相位按圈取模，360 与 0 等价。
        startPhaseDeg: wrap01(clampNumber(raw.startPhaseDeg, -360, 720, 0) / 360) * 360,
        attackMs: clampNumber(
            raw.attackMs,
            VIBRATO_LIMITS.attackMs.min,
            VIBRATO_LIMITS.attackMs.max,
            DEFAULT_VIBRATO_PRESET.attackMs,
        ),
        attackCurve: pickEnum(raw.attackCurve, ENVELOPE_CURVES, DEFAULT_VIBRATO_PRESET.attackCurve),
        releaseMs: clampNumber(
            raw.releaseMs,
            VIBRATO_LIMITS.releaseMs.min,
            VIBRATO_LIMITS.releaseMs.max,
            DEFAULT_VIBRATO_PRESET.releaseMs,
        ),
        releaseCurve: pickEnum(
            raw.releaseCurve,
            ENVELOPE_CURVES,
            DEFAULT_VIBRATO_PRESET.releaseCurve,
        ),
        depthRamp: {
            start: clampNumber(
                depthRamp.start,
                VIBRATO_LIMITS.depthRamp.min,
                VIBRATO_LIMITS.depthRamp.max,
                DEFAULT_VIBRATO_PRESET.depthRamp.start,
            ),
            end: clampNumber(
                depthRamp.end,
                VIBRATO_LIMITS.depthRamp.min,
                VIBRATO_LIMITS.depthRamp.max,
                DEFAULT_VIBRATO_PRESET.depthRamp.end,
            ),
        },
        alignCycles: Boolean(raw.alignCycles),
        irregularity: clampNumber(
            raw.irregularity,
            VIBRATO_LIMITS.irregularity.min,
            VIBRATO_LIMITS.irregularity.max,
            DEFAULT_VIBRATO_PRESET.irregularity,
        ),
        biasCents: clampNumber(
            raw.biasCents,
            VIBRATO_LIMITS.biasCents.min,
            VIBRATO_LIMITS.biasCents.max,
            DEFAULT_VIBRATO_PRESET.biasCents,
        ),
        baseline: pickEnum(raw.baseline, BASELINE_MODES, DEFAULT_VIBRATO_PRESET.baseline),
        blend: clampNumber(
            raw.blend,
            VIBRATO_LIMITS.blend.min,
            VIBRATO_LIMITS.blend.max,
            DEFAULT_VIBRATO_PRESET.blend,
        ),
    };
}

/**
 * 由既有预设派生一份**用户预设**（系统预设只读，要改必须先复制）。
 *
 * @param source 来源预设。
 * @param name 新名称；省略时在原名后追加「副本」。
 */
export function duplicateVibratoPreset(source: VibratoPreset, name?: string): VibratoPreset {
    const base = source.name.trim();
    return {
        ...source,
        id: createVibratoPresetId(),
        builtin: false,
        name: (name ?? (base ? `${base} 2` : "")).trim(),
        cycle:
            source.cycle.kind === "table"
                ? { kind: "table", table: [...source.cycle.table] }
                : { ...source.cycle },
        depthRamp: { ...source.depthRamp },
    };
}

/**
 * 列表去重：同 id 保留后者（后写覆盖先写），并剔除互相重复的 id。
 * 手改配置可能造出重复 id，重复项会让"当前预设"指向不明确。
 */
export function dedupeVibratoPresets(list: readonly VibratoPreset[]): VibratoPreset[] {
    const byId = new Map<string, VibratoPreset>();
    for (const preset of list) byId.set(preset.id, preset);
    return [...byId.values()];
}

/** 纯色表的周期表长度（供提取路径构造新预设）。 */
export { CYCLE_TABLE_DEFAULT_LEN };
