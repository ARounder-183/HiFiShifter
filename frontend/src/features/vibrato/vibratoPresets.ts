/**
 * 颤音预设的规整与默认值。
 *
 * 【定位】与 `utils/customScales.ts` 同构：后端只做透传存储，字段语义、
 * 取值合法性与默认值全部在前端收口。手改过 / 旧版本的 `app_config.json`
 * 必须能在这里被无异常地拉回合法状态。
 */

import { CYCLE_TABLE_DEFAULT_LEN, normalizeCycleTable, wrap01 } from "./vibratoCycle";
import { vibratoSeedForPreset } from "./vibratoSeed";
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

/**
 * 用户自定义预设的数量上限。
 *
 * 【为什么要封顶】预设列表随设置整体读写，也在预设菜单里完整铺开。无上限既会
 * 让菜单比屏幕高，也让"环绕切换预设"失去意义。上限收在前端 —— 后端只做透传
 * 存储，不重复定义第二份业务规则。
 */
export const MAX_VIBRATO_PRESETS = 256;

const WAVE_SHAPES: readonly WaveShape[] = [
    "sine",
    "triangle",
    "sawUp",
    "sawDown",
    "square",
    "trapezoid",
    "trill",
];
/**
 * 「摆放方式」的合法取值（颤音围绕哪条曲线摆）。
 *
 * 【为什么留在这里而不是预设里】它是**添加颤音时的一个参数**，与具体预设无关
 * （见 `BaselineMode` 的说明）。放这里是因为"合法取值集合"与其它字段的值域是同一类
 * 东西 —— 设置的读取侧要用它做校验。
 */
export const BASELINE_MODES: readonly BaselineMode[] = [
    "line",
    "holdStart",
    "holdEnd",
    "average",
    "existing",
];
/**
 * 「摆放方式」的默认值：**起点 → 终点**。
 *
 * 【为什么是它】它等于抽取成设置**之前**的既有行为（预设的 `baseline` 一直兜底为
 * `line`）—— 改设置默认值等于替所有用户改一次行为，不该顺手做。用户想要"保留歌手
 * 原本的音高运动"时，在窗口右上角换成「保持现有曲线」即可，且会被记住。
 */
export const DEFAULT_VIBRATO_BASELINE: BaselineMode = "line";
const ENVELOPE_CURVES: readonly EnvelopeCurve[] = ["linear", "exp", "s"];
const RATE_MODES: readonly VibratoRateMode[] = ["hz", "cycles"];

/** 各字段的合法区间。浏览器手改配置只能落进这些范围。 */
export const VIBRATO_LIMITS = {
    // 深度允许为负：负值等于把波形整体反相（起点先往下摆），拖拽调幅与预设编辑
    // 都要能表达它。量程对称，绝对值上限不变。
    depthCents: { min: -1200, max: 1200 },
    rateHz: { min: 0.1, max: 20 },
    cycles: { min: 0.5, max: 128 },
    rateRampEnd: { min: 0.25, max: 4 },
    attackMs: { min: 0, max: 10000 },
    releaseMs: { min: 0, max: 10000 },
    depthRamp: { min: 0, max: 2 },
    irregularity: { min: 0, max: 100 },
    seed: { min: 0, max: 99999 },
    biasCents: { min: -200, max: 200 },
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
    seed: 0,
    biasCents: 0,
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
        // 旧数据没有 seed：回落到按 id 派生，与移除字段前的图案一致。
        seed: clampNumber(
            raw.seed,
            VIBRATO_LIMITS.seed.min,
            VIBRATO_LIMITS.seed.max,
            vibratoSeedForPreset({ id }),
        ),
        biasCents: clampNumber(
            raw.biasCents,
            VIBRATO_LIMITS.biasCents.min,
            VIBRATO_LIMITS.biasCents.max,
            DEFAULT_VIBRATO_PRESET.biasCents,
        ),
    };
}

/** 名称末尾的「 N」副本编号。 */
const DUPLICATE_SUFFIX_PATTERN = /\s+\d+$/;

/**
 * 求一个"副本名"：在来源名后面追加编号，并避开已占用的名字。
 *
 * 【为什么不能直接拼 " 2"】连点两次会得到 `名字 2 2`、再点变 `名字 2 2 2` —— 编号
 * 越叠越长，读起来像是名字本身的一部分。这里先把来源名里**已有的编号**剥掉，再从
 * 2 开始找第一个没被占用的编号：`名字` → `名字 2` → `名字 3`。
 *
 * @param displayName 来源预设**显示用的**名字（系统预设要走词条，见调用方）。
 * @param takenNames 现有全部预设的显示名 —— 编号要避开它们，而不是只看用户段。
 */
export function nextDuplicatePresetName(
    displayName: string,
    takenNames: readonly string[],
): string {
    const trimmed = displayName.trim();
    // 反复剥掉末尾编号：`名字 2 2` 也要回到 `名字`，否则这个坏名字会被当成基底继续叠。
    let base = trimmed;
    let previous = "";
    while (base !== previous) {
        previous = base;
        base = base.replace(DUPLICATE_SUFFIX_PATTERN, "").trim();
    }
    if (!base) return "";
    const taken = new Set(takenNames.map((name) => name.trim()));
    let index = 2;
    while (taken.has(`${base} ${index}`)) index += 1;
    return `${base} ${index}`;
}

/**
 * 由既有预设派生一份**用户预设**（系统预设只读，要改必须先复制）。
 *
 * @param source 来源预设。
 * @param name 新名称；省略时在原名后追加 `2`。调用方通常先用
 *   `nextDuplicatePresetName` 算好（它会避开已占用的编号，且能处理系统预设的显示名）。
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
