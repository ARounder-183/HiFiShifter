/**
 * 颤音预设的类型定义。
 *
 * 【核心抽象】任何颤音波形都归一化为「单位周期上的采样函数」
 * （`CycleSource`）—— 参数式（形状 + 偏斜）与采样式（手绘折叠出的周期表）
 * 共用同一条下游管线：渲染、包络、应用、拖拽预览都不需要区分二者。
 *
 * 单位约定：
 * - 深度一律以 **cents** 为规范单位存储，落到具体参数时按参数族换算
 *   （见 `vibratoDepth.ts`）。这样同一个预设套到音高、动态、共振峰上
 *   都有可解释的含义。
 * - 速率有「按 Hz」与「按整段周期数」两种模式：前者给固定听感（拖多长
 *   都是同一个音色），后者给固定个数（"这一段打 6 个颤音"）。
 */

/** 参数式波形的形状。`skew` 的含义随形状变化，见 `vibratoCycle.ts`。 */
export type WaveShape =
    | "sine"
    | "triangle"
    | "sawUp"
    | "sawDown"
    | "square"
    | "trapezoid"
    | "trill";

/**
 * 归一化周期波形：相位 `u ∈ [0,1)` → 波形值 `∈ [-1,1]`。
 *
 * - `shape`：由形状 + 偏斜生成，可无损存成两个数字
 * - `table`：一个周期的采样表（手绘提取 / 用户捏出来的），首尾相接
 */
export type CycleSource =
    | { kind: "shape"; shape: WaveShape; skew: number }
    | { kind: "table"; table: number[] };

/** 深度换算的族别。决定「同一个 cents 值」在参数上是什么量。 */
export type DepthFamily = "cents" | "ratio" | "raw";

/** 包络段的形状。 */
export type EnvelopeCurve = "linear" | "exp" | "s";

/**
 * 摆放方式：颤音围绕哪条曲线上下摆动。
 *
 * 【为什么不是预设的字段】它回答的是"这一次要把颤音**挂到**什么上面"，与"颤音长
 * 什么样"（波形 / 深度 / 速率 / 包络）是两件事：
 *
 * 1. 用户的心智是先定摆放方式、再挑预设 —— 换预设不该把摆放方式也换掉（曾经是
 *    预设字段，切一次预设就得重选一次）；
 * 2. 它只在「添加颤音」里起作用：预设库、拖拽工具、预设文件都与它无关。
 *
 * 因此它存在**设置**里（本机记忆，见 `session.vibratoBaseline`），随窗口一起打开。
 */
export type BaselineMode =
    /** 起点 → 终点线性插值（拖拽的既有行为）。 */
    | "line"
    /** 起点值水平保持。 */
    | "holdStart"
    /** 终点值水平保持。 */
    | "holdEnd"
    /** 全程取均值。 */
    | "average"
    /** 在已有曲线上做调制（保留原曲线的音高运动）。 */
    | "existing";

/** 速率模式。 */
export type VibratoRateMode =
    /** 频率以 Hz 计，听感与选区长度无关。 */
    | "hz"
    /** 整段固定周期个数，与选区长度无关。 */
    | "cycles";

/** 颤音预设。 */
export interface VibratoPreset {
    /** `builtin.<name>` 为系统预设；`custom_<rand>` 为用户预设。 */
    id: string;
    /** 用户可见名。系统预设的名称走 i18n（`labelKey`），此字段仅作兜底。 */
    name: string;
    /** 系统预设只读：要改必须先「复制为自定义」。 */
    builtin: boolean;

    /** 归一化周期波形。 */
    cycle: CycleSource;

    /** 深度（cents，规范单位）。 */
    depthCents: number;

    rateMode: VibratoRateMode;
    /** `rateMode === "hz"` 时的频率。 */
    rateHz: number;
    /** `rateMode === "cycles"` 时的整段周期数（可含小数）。 */
    cycles: number;
    /** 末端速率比：1 = 匀速；1.25 = 到末端快 25%。 */
    rateRampEnd: number;
    /** 起始相位（度）：0 = 形状的自然起点。 */
    startPhaseDeg: number;

    attackMs: number;
    attackCurve: EnvelopeCurve;
    releaseMs: number;
    releaseCurve: EnvelopeCurve;
    /** 渐强：起点 / 终点的深度倍率。 */
    depthRamp: { start: number; end: number };

    /** 收尾对齐整数周期：让音符干净地落回基准线。 */
    alignCycles: boolean;

    /** 不规则度（自然度）0..100：确定性的相位与深度抖动。 */
    irregularity: number;
    /**
     * 不规则度的抖动图案种子（0..99999）。
     *
     * 【为什么成为字段】它曾由预设 id 派生 —— 用户只能接受预设"摇成什么样"，
     * 无法换一种随机味。成为字段后，管理器里的骰子按钮就能换图案，而同一预设的
     * 每次渲染 / 试听 / 提交仍逐帧一致（种子是确定的，不是随机数）。
     */
    seed: number;
    /** 基线偏移（cents）：整体偏高 / 偏低。 */
    biasCents: number;
}

/** 预设的最小必要字段（用于默认值合并）。 */
export type VibratoPresetInput = Partial<VibratoPreset>;
