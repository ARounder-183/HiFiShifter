/**
 * 颤音预设编辑器的纯逻辑。
 *
 * 抽出来的理由与 `customScaleDialogLogic.ts` 相同：命名映射、单位换算与预览
 * 采样都是可单测的纯函数，混在组件里就只能靠肉眼验证。
 *
 * 【为什么标签映射写成显式 Record】`keyReferenceIntegrity.test.ts` 要求
 * `labelKey` 之类的字面量必须真实存在于词条中，`` `vibrato_shape_${shape}` ``
 * 这类模板拼接过不了校验；显式表也顺带让"新增一个波形要补哪些词条"一目了然。
 */

import type { MessageKey } from "../../../i18n/messages";
import {
    BUILTIN_VIBRATO_PRESET_DESC_KEYS,
    BUILTIN_VIBRATO_PRESET_NAME_KEYS,
    type BuiltinVibratoId,
} from "../../../features/vibrato/systemPresets";
import { isBuiltinVibratoPresetId } from "../../../features/vibrato/vibratoPresets";
import { sampleCycle } from "../../../features/vibrato/vibratoCycle";
import { buildVibratoCurve, DEFAULT_FRAME_PERIOD_MS } from "../../../features/vibrato/vibratoCurve";
import {
    depthStepUnitFor,
    depthToDisplay,
    displayToDepth,
    paramUnitToDepth,
    type VibratoParamRange,
} from "../../../features/vibrato/vibratoDepth";
import {
    bridgeShortGaps,
    planVibratoTarget,
    refineNoteFrames,
    suppressNonTargetFrames,
} from "../../../features/vibrato/vibratoPitch";
import type {
    BaselineMode,
    CycleSource,
    EnvelopeCurve,
    VibratoPreset,
    VibratoRateMode,
    WaveShape,
} from "../../../features/vibrato/vibratoTypes";

/** 翻译函数（`t` 的签名子集）。 */
export type Translate = (key: MessageKey) => string;

/** 波形形状的显示名。 */
export const WAVE_SHAPE_KEYS: Record<WaveShape, MessageKey> = {
    sine: "vibrato_shape_sine",
    triangle: "vibrato_shape_triangle",
    sawUp: "vibrato_shape_saw_up",
    sawDown: "vibrato_shape_saw_down",
    square: "vibrato_shape_square",
    trapezoid: "vibrato_shape_trapezoid",
    trill: "vibrato_shape_trill",
};

/** 包络曲线的显示名。 */
export const ENVELOPE_CURVE_KEYS: Record<EnvelopeCurve, MessageKey> = {
    linear: "vibrato_curve_linear",
    exp: "vibrato_curve_exp",
    s: "vibrato_curve_s",
};

/** 基线模式的显示名。 */
export const BASELINE_MODE_KEYS: Record<BaselineMode, MessageKey> = {
    line: "vibrato_baseline_line",
    holdStart: "vibrato_baseline_hold_start",
    holdEnd: "vibrato_baseline_hold_end",
    average: "vibrato_baseline_average",
    existing: "vibrato_baseline_existing",
};

/** 速率模式的显示名。 */
export const RATE_MODE_KEYS: Record<VibratoRateMode, MessageKey> = {
    hz: "vibrato_rate_mode_hz",
    cycles: "vibrato_rate_mode_cycles",
};

/** 波形形状的枚举顺序（与下拉框一致）。 */
export const WAVE_SHAPE_ORDER: readonly WaveShape[] = [
    "sine",
    "triangle",
    "sawUp",
    "sawDown",
    "square",
    "trapezoid",
    "trill",
];

/** 基线模式的枚举顺序。 */
export const BASELINE_MODE_ORDER: readonly BaselineMode[] = [
    "line",
    "holdStart",
    "holdEnd",
    "average",
    "existing",
];

/** 包络曲线的枚举顺序。 */
export const ENVELOPE_CURVE_ORDER: readonly EnvelopeCurve[] = ["linear", "exp", "s"];

/** 预设的显示名：系统预设走词条，用户预设用用户输入的名字。 */
export function vibratoPresetLabel(preset: VibratoPreset, t: Translate): string {
    const builtinKey = builtinIdOf(preset.id);
    if (builtinKey) return t(BUILTIN_VIBRATO_PRESET_NAME_KEYS[builtinKey]);
    return preset.name.trim();
}

/** 预设的一行说明；用户预设没有说明，返回 `undefined`。 */
export function vibratoPresetDescription(preset: VibratoPreset, t: Translate): string | undefined {
    const builtinKey = builtinIdOf(preset.id);
    return builtinKey ? t(BUILTIN_VIBRATO_PRESET_DESC_KEYS[builtinKey]) : undefined;
}

/** `builtin.<key>` → `<key>`；不是系统预设或键未知时返回 `undefined`。 */
export function builtinIdOf(id: string): BuiltinVibratoId | undefined {
    if (!isBuiltinVibratoPresetId(id)) return undefined;
    const key = id.slice(id.indexOf(".") + 1);
    return key in BUILTIN_VIBRATO_PRESET_NAME_KEYS ? (key as BuiltinVibratoId) : undefined;
}

/**
 * 波形的"形状"标签。
 *
 * 【为什么采样表不能一律叫「来自选区」】表波形有两个来源：从选区提取，以及用户手绘
 * —— 后者占了编辑器里的主要操作，把它说成"来自选区"是错的。这里改成按**形状**说话：
 * 采样表先与各个参数式形状比对，足够像就报那个形状名（"正弦"），都不像才承认它是
 * 手绘出来的自定义波形。这样无论来源如何，标签描述的都是用户真正看到的东西。
 */
export function cycleShapeLabelKey(cycle: CycleSource): MessageKey {
    if (cycle.kind === "shape") return WAVE_SHAPE_KEYS[cycle.shape];

    const samples = new Array<number>(SHAPE_MATCH_SAMPLES);
    for (let i = 0; i < SHAPE_MATCH_SAMPLES; i += 1) {
        samples[i] = sampleCycle(cycle, i / SHAPE_MATCH_SAMPLES);
    }
    let bestShape: WaveShape | null = null;
    let bestScore = 0;
    for (const shape of WAVE_SHAPE_ORDER) {
        const reference = new Array<number>(SHAPE_MATCH_SAMPLES);
        for (let i = 0; i < SHAPE_MATCH_SAMPLES; i += 1) {
            reference[i] = sampleCycle(
                { kind: "shape", shape, skew: 0.5 },
                i / SHAPE_MATCH_SAMPLES,
            );
        }
        const score = correlation(samples, reference);
        if (bestShape === null || score > bestScore) {
            bestShape = shape;
            bestScore = score;
        }
    }
    return bestShape !== null && bestScore >= SHAPE_MATCH_MIN_CORRELATION
        ? WAVE_SHAPE_KEYS[bestShape]
        : "vibrato_cycle_drawn";
}

/** 相关系数（`-1..1`）；任一侧无变化时返回 0（无法判断，不算匹配）。 */
function correlation(a: readonly number[], b: readonly number[]): number {
    const n = Math.min(a.length, b.length);
    if (n < 2) return 0;
    let meanA = 0;
    let meanB = 0;
    for (let i = 0; i < n; i += 1) {
        meanA += a[i];
        meanB += b[i];
    }
    meanA /= n;
    meanB /= n;
    let dot = 0;
    let varA = 0;
    let varB = 0;
    for (let i = 0; i < n; i += 1) {
        const x = a[i] - meanA;
        const y = b[i] - meanB;
        dot += x * y;
        varA += x * x;
        varB += y * y;
    }
    const denom = Math.sqrt(varA * varB);
    return denom > 1e-12 ? dot / denom : 0;
}

/**
 * 预设的一行摘要（列表行 / 选择器 / HUD 共用）。
 *
 * 形如 `正弦 · 30 分 · 5.5 Hz`；按周期数模式时把频率换成周期数。
 */
export function vibratoPresetSummary(preset: VibratoPreset, t: Translate): string {
    const rate =
        preset.rateMode === "cycles"
            ? `${formatNumber(preset.cycles)} ${t("vibrato_unit_cycles")}`
            : `${formatNumber(preset.rateHz)} ${t("vibrato_unit_hz")}`;
    const parts = [
        t(cycleShapeLabelKey(preset.cycle)),
        `${formatNumber(preset.depthCents)} ${t("vibrato_unit_cents")}`,
        rate,
    ];
    return parts.join(" · ");
}

/** 数值的紧凑显示：去掉无意义的小数尾巴。 */
export function formatNumber(value: number): string {
    if (!Number.isFinite(value)) return "0";
    return String(Math.round(value * 100) / 100);
}

/**
 * 把 `depthCents` 换算成当前参数在**编辑器 / HUD 上显示的单位**。
 *
 * 【与曲线换算不同】曲线把音高按半音存，编辑器按分显示 —— 用户在音高上应当
 * 看到 `30`（分）而不是 `0.3`（半音）。用 `depthToDisplay` 而不是
 * `depthToParamUnit`，否则音高上会差 100 倍。
 */
export function depthForParam(
    depthCents: number,
    param: string,
    range?: { min: number; max: number },
): number {
    return depthToDisplay(depthCents, param, range);
}

/** {@link depthForParam} 的逆运算，用于把编辑器里输入的数值写回预设。 */
export function depthToCents(
    value: number,
    param: string,
    range?: { min: number; max: number },
): number {
    return displayToDepth(value, param, range);
}

/**
 * 深度显示值的**单位标签**（`null` = 原始值域参数，不带单位后缀）。
 *
 * 【为什么必须按参数取】`depthForParam` 已经把深度换算成参数原生单位，
 * 标签必须跟着变：在 `dyn` 上显示 "30 cents" 是把百分比说成了音分。单位
 * 由 `depthStepUnitFor` 决定（它同时决定编辑器里滚轮一格走多少），两者
 * 从同一个来源取，不会各自漂移。
 */
export function depthUnitLabelKey(param: string): MessageKey | null {
    switch (depthStepUnitFor(param)) {
        case "cents":
            return "vibrato_unit_cents";
        case "percent":
            return "vibrato_unit_percent";
        case "scaleDegree":
            return "vibrato_unit_degree";
        default:
            return null;
    }
}

/** 预设编辑器波形预览的几何。 */
export interface VibratoPreviewGeometry {
    /** 采样点数（横向）。 */
    frameCount: number;
    /** 名义帧周期（仅用于把"整段时长"算出来，与真实工程无关）。 */
    framePeriodMs: number;
}

/** 预览默认窗口：约 1.6 秒，足以看清渐入、渐强与若干个周期。 */
export const PREVIEW_DEFAULT: VibratoPreviewGeometry = {
    frameCount: 320,
    framePeriodMs: DEFAULT_FRAME_PERIOD_MS,
};

/** 预览采样结果，单位一律是 **cents**（与参数无关，便于画布定标）。 */
export interface VibratoPreviewSamples {
    /**
     * 逐点波形（cents）。
     *
     * 管理器预览里它是**预设波形本身**（围绕 0）；"套用到选区"预览里它是**套用后的
     * 参数线**（相对下面那个共同中心）—— 两种用法都只做单位换算，不另写公式。
     */
    wave: number[];
    /** 逐点包络上界（cents，恒非负）—— 渐入 / 渐强 / 渐出入画就靠它。 */
    envelope: number[];
    /**
     * 纵轴半幅（cents）：画出来的东西的绝对值上界。
     *
     * 【为什么不在这里保底为 1】它是**读数**（"±N 分"）的来源，直线预设就该报 0 ——
     * 曾经在这里保底 1，于是完全平直的颤音线一直显示"±1 分"。需要非零尺度的地方
     * （`previewScaleCents` / `fitPreviewRangeCents`）各自兜底，不必让读数替它们背锅。
     */
    peakCents: number;
    /**
     * 背景参考线（"套用到选区"预览里的原参数线；断口为 NaN）。
     *
     * 与 `wave` **共用同一套纵轴**（同一个原点、同一套比例尺），因此按「适应」换比例尺
     * 时两条线一起缩放。管理器预览不提供它 —— 那里只画波形本身。
     */
    contour?: number[];
    /**
     * 包络带的**中心线**（省略 = 围绕 0）。
     *
     * 【为什么需要】带子表示"颤音摆动的幅度"，而摆动是围绕**它所在的那条基线**的，
     * 不是围绕图上的零点。少了这条中心线，`baseline: "line"` 这类预设的带子会画在
     * 一条与波形无关的横线上。
     */
    envelopeCenter?: number[];
}

/**
 * 生成预设的预览采样。
 *
 * 走的是与拖拽、落盘**完全相同**的 `buildVibratoCurve`（参数取 `pitch`，基线
 * 恒为 0，因此输出值 × 100 即 cents）—— 预览画的就是用户真正会得到的波形，
 * 而不是另写一份"看起来像"的公式。
 */
export function buildVibratoPreview(
    preset: VibratoPreset,
    geometry: VibratoPreviewGeometry = PREVIEW_DEFAULT,
): VibratoPreviewSamples {
    const frameCount = Math.max(2, Math.round(geometry.frameCount));
    const result = buildVibratoCurve({
        startFrame: 0,
        startValue: 0,
        endFrame: frameCount - 1,
        endValue: 0,
        // 强制 `line` 基线：预览要展示波形本身，`existing` 在没有原曲线时
        // 会退化成端点插值，画出来与用户看到的不一致。
        preset: { ...preset, baseline: "line", blend: 100 },
        param: "pitch",
        framePeriodMs: geometry.framePeriodMs,
        collectEnvelope: true,
    });

    const wave = result.dense.map((value) => value * 100);
    const envelope = (result.envelope ?? new Array(frameCount).fill(0)).slice();
    // 真实峰值：直线预设就是 0（读数据此显示 "±0 分"）。
    let peak = 0;
    for (const value of envelope) peak = Math.max(peak, Math.abs(value));
    for (const value of wave) peak = Math.max(peak, Math.abs(value));
    return { wave, envelope, peakCents: peak };
}

/** "套用到选区"预览的采样结果：颤音偏移 + 包络 + 原/后参数线轮廓。 */
export interface VibratoAppliedPreview extends VibratoPreviewSamples {
    /** 原参数线（背景，弱化虚线；断口为 NaN）。 */
    contour: number[];
    /** 包络带的中心线（断口为 NaN）。 */
    envelopeCenter: number[];
    /**
     * 颤音本身的峰值幅度（cents）—— 读数用它。
     *
     * 与 `peakCents` 的区别：那个是"画出来的全部"的峰值（纵轴拟合用），会被素材自身的
     * 音高起伏撑大；读数要的是"这个预设会摆多少"，只跟包络有关。
     */
    vibratoPeakCents: number;
    /**
     * **可信帧**掩码（与 `contour` / `wave` 等长）。
     *
     * 【为什么单独交出来】两条线**照画**音符帧（断口只留给真正的气口），但试听是
     * 合成人声 —— 跟踪器在音符内部给出的异常帧（八度跳、从无声区爬上来的过渡段）
     * 按原值合成是几十赫兹的超低频，听感上是一声闷响，那不是这段素材的音高。
     * 试听据此把不可信的帧按最近的音高持续（见 `buildContourAuditionPair`），
     * 而画面上它们照旧可见（用户要看得见"这里有点怪"）。
     */
    stableFrames: boolean[];
}

/**
 * 生成"套用到选区"的预览采样。
 *
 * 【与管理器预览的区别】管理器预览拿 `buildVibratoPreview`（无真实数据、基线强制
 * `line`），回答"这个预设的波形长什么样"；这里喂入选区的**真实帧值**，走同一条
 * `buildVibratoCurve`，回答"套到这段上，颤音会怎么走"。
 *
 * 【只画"相对基线的偏移"，不画绝对音高】这是纵轴定标的关键，也是这个弹窗唯一
 * 该回答的问题：
 *
 * - 画绝对音高时，纵轴必须容下整条音高轮廓。而轮廓与颤音可以差**两个数量级** ——
 *   选一整句必然包含句首句尾的气口与跟踪器从无声区缓升出来的过渡帧（它们在值域
 *   之内，因此属于音符段），一句 0→60 的缓升就把峰值撑到几千分，40 分的颤音只剩
 *   1% 的画布高度。
 * - 相对基线作图后，`结果 − 基线` = 偏置 + 颤音，幅度上界即包络 —— 纵轴**只跟
 *   颤音自己的幅度走**，与素材轮廓多宽、基线模式是哪种都无关，切换预设也不会
 *   重新缩放，因此可以横向比较不同预设。
 * - 相对基线的**原曲线**（早先版本里那条虚线参考）刻意不再提供：它就是那个被
 *   撑大的量，画出来只会把纵轴重新拉走。素材的绝对轮廓在钢琴卷帘里本来就看得见。
 *
 * @returns 没有可调制的音符段（音高全未检测 / 没有够长的音符）或原值不足两点时
 *          返回 `null`，由调用方显示占位提示。
 */
export function buildAppliedPreview(args: {
    preset: VibratoPreset;
    original: readonly number[];
    param: string;
    framePeriodMs: number;
    range?: VibratoParamRange;
}): VibratoAppliedPreview | null {
    const values = args.original.map((value) => (Number.isFinite(value) ? Number(value) : 0));
    if (values.length < 2) return null;

    const framePeriodMs =
        Number.isFinite(args.framePeriodMs) && args.framePeriodMs > 0
            ? args.framePeriodMs
            : DEFAULT_FRAME_PERIOD_MS;

    /*
     * 锚点与"哪些帧可调制"一次规划出来，整段未检测（或没有够长的音符段）时没有
     * 可预览的对象。
     *
     * pitch 的 0 是"未检测到音高"的哨兵，而浊清边界上还有**低而非零**的过渡帧
     * （跟踪器给的 20~40 Hz 低估）—— 两类都不是音符。落盘侧调同一对函数，因此这里
     * 画的就是"应用"真正会写出来的东西，预览与结果不会各说各话。
     */
    const plan = planVibratoTarget(args.param, values, framePeriodMs);
    if (!plan) return null;

    const result = buildVibratoCurve({
        startFrame: 0,
        startValue: plan.anchors.startValue,
        endFrame: values.length - 1,
        endValue: plan.anchors.endValue,
        original: values,
        preset: args.preset,
        param: args.param,
        framePeriodMs,
        range: args.range,
        collectEnvelope: true,
        // 基线要一并拿回来：下面按"相对基线的偏移"作图（见下）。
        collectBaseline: true,
    });
    // 非音符帧写回哨兵：它们不受颤音影响（"不改这一帧"，与落盘侧同一语义）。
    suppressNonTargetFrames(result.dense, plan);

    const toCents = (value: number) => paramUnitToDepth(value, args.param, args.range);

    /*
     * 两条线画在**音符帧**上（`plan.modulatable`），断口只在"这里没有音高"处出现。
     *
     * 【为什么不是 refine 之后的稳定帧】refine 回答的是"这些帧稳不稳、值不值得
     * 拿去拟合"，不是"这里有没有音高"（见 `planVibratoTarget` 的说明）。用它当画图
     * 掩码，轮廓就会在音符**内部**断开 —— 用户看到的是"在不应该断的地方断了"
     * （跟踪器给一段连奏 / 一个八度跳时残差很大，那段就被剔掉）。
     *
     * 画音符帧还有一个更硬的理由：**落盘掩码就是它**。预览画的必须就是"应用会写出来
     * 的东西"，否则预览与结果各说各话。
     */
    const noteFrames = plan.modulatable;
    /*
     * 稳定帧只用于**统计**：共同中心取它，免得一段滑音把中心拽偏。
     */
    const stableFrames = refineNoteFrames(
        values.map((value, index) => (noteFrames[index] ? value : Number.NaN)),
        framePeriodMs,
    );

    /*
     * 共同中心：稳定帧上原参数线的均值。
     *
     * 【为什么两条线都要相对它】它们必须**共用一套纵轴**：按「适应」换比例尺时两条线
     * 一起缩放，而不是只有一条动。共用的前提是同一个原点 —— 轮廓是绝对音高（几千分）、
     * 颤音只有几十分，各自减掉**同一个**中心之后，两者才落在同一根标尺上，"新参数线
     * 与原参数线差多少"才读得出来。
     */
    const stableCents: number[] = [];
    for (let i = 0; i < values.length; i += 1) {
        if (stableFrames[i]) stableCents.push(toCents(values[i]));
    }
    const centerCents =
        stableCents.length > 0
            ? stableCents.reduce((sum, value) => sum + value, 0) / stableCents.length
            : 0;
    const deviationAt = (source: readonly number[], index: number) =>
        noteFrames[index] ? toCents(source[index]) - centerCents : Number.NaN;
    const baselineSource = result.baseline ?? values;

    // 原参数线（背景）与套用后的参数线（前景）—— 同一原点、同一标尺。
    const contour = bridgeShortGaps(
        values.map((_, index) => deviationAt(values, index)),
        framePeriodMs,
    );
    const wave = bridgeShortGaps(
        result.dense.map((_, index) => deviationAt(result.dense, index)),
        framePeriodMs,
    );
    /*
     * 不受颤音影响的帧画成**断口**（NaN，画布抬笔）：它既不是一个音高，也不是
     * "停在中心"。让两种含义在图上可区分，用户才不会把"这里没数据 / 这里不加颤音"
     * 读成"这里被拉平了"。
     */
    const envelope = (result.envelope ?? new Array(values.length).fill(0)).map((value, index) =>
        noteFrames[index] ? Math.abs(value) : Number.NaN,
    );
    /** 带子的中心：颤音所围绕的那条基线（与两条线同一原点）。 */
    const envelopeCenter = bridgeShortGaps(
        baselineSource.map((_, index) =>
            noteFrames[index] ? toCents(baselineSource[index]) - centerCents : Number.NaN,
        ),
        framePeriodMs,
    );

    /*
     * 纵轴按**稳定帧上画出来的东西**拟合：两条线 + 包络带的两缘。
     *
     * 【为什么定标只认稳定帧，而画图认音符帧】两者回答的问题不同：
     * - **画图**问"这里有没有音高"—— 有就画，断口只留给真正的气口；
     * - **定标**问"这段素材的音高大致在哪个范围"—— 跟踪器的八度跳、从无声区爬上来的
     *   过渡段会把量程撑到上千分，40 分的颤音随即被压成一条直线（这正是修过几轮的
     *   "看不清颤音"）。
     *
     * 于是极端帧**照画**（线不断），但不参与定标（线会跑出画框）。宁可让用户看到
     * "这里有一下很怪"，也不要让整条颤音被它压扁 —— 读数 `vibratoPeakCents` 另行给出
     * 颤音自身的幅度，不受这个取舍影响。
     */
    let peak = 0;
    let vibratoPeak = 0;
    for (let i = 0; i < values.length; i += 1) {
        if (Number.isFinite(envelope[i])) {
            vibratoPeak = Math.max(vibratoPeak, envelope[i]);
        }
        if (!stableFrames[i]) continue;
        for (const value of [contour[i], wave[i]]) {
            if (Number.isFinite(value)) peak = Math.max(peak, Math.abs(value));
        }
        if (Number.isFinite(envelope[i]) && Number.isFinite(envelopeCenter[i])) {
            peak = Math.max(peak, Math.abs(envelopeCenter[i]) + envelope[i]);
        }
    }
    return {
        wave,
        envelope,
        peakCents: peak,
        contour,
        envelopeCenter,
        vibratoPeakCents: vibratoPeak,
        stableFrames,
    };
}

/**
 * 纵轴定标：按预设自身的幅度取整到一个"好看"的档位。
 *
 * 【为什么不按参数值域定标】音高的值域是几十个半音，30 cents 的颤音按那个
 * 尺度画出来就是一条直线。按预设自身幅度定标，5 分与 100 分的预设都看得清
 * 形状，真实幅度由读数负责表达。
 *
 * 【注意】这是**一次性拟合**用的（见 `fitPreviewRangeCents`），不是每帧跟着
 * 当前值走的自适应标尺 —— 后者会让波形永远填满画布，深度变化只表现为"抖动"，
 * 用户无从判断幅度大小。
 */
export function previewScaleCents(peakCents: number): number {
    const safe = Math.max(1, Number.isFinite(peakCents) ? peakCents : 1);
    return Math.ceil(safe * 1.15);
}

/**
 * 预览纵轴的"好看"档位（半幅，cents）。
 *
 * 取值成阶梯而不是连续值：同一档位下不同预设的波形高度可以直接互相比较；跨档位
 * 时轴上的刻度标签会跟着变，读数不会失真。
 */
export const PREVIEW_RANGE_LADDER: readonly number[] = [
    5, 10, 20, 25, 50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 5000, 8000,
];

/**
 * 由一段波形的峰值拟合预览纵轴半幅。
 *
 * 只比峰值大 15% 再向上取到最近的档位，于是波形通常占据画布的六成上下 ——
 * 既看得清形状，又留得下"再深一点"的余地。**只在打开 / 换预设 / 点「适应」时
 * 调用**：编辑期间标尺保持不动，波形高度才等于深度，用户才能直观判断大小。
 */
export function fitPreviewRangeCents(peakCents: number): number {
    const safe = Math.abs(Number.isFinite(peakCents) ? peakCents : 0);
    const target = safe * 1.15;
    for (const step of PREVIEW_RANGE_LADDER) {
        if (target <= step) return step;
    }
    return PREVIEW_RANGE_LADDER[PREVIEW_RANGE_LADDER.length - 1];
}

/** 缩略图专用采样数：64 点足够表达形状，path 缓存也便宜。 */
export const GLYPH_FRAME_COUNT = 64;

/** 形状比对用的采样数（摘要标签）。 */
const SHAPE_MATCH_SAMPLES = 64;

/**
 * 采样表被判为某个参数式形状所需的最低相关系数。
 *
 * 取高阈值是故意的：宁可说"手绘"，也不要把一条捏出来的怪曲线硬说成正弦 —— 标签
 * 的价值在于可信，而不是"总能给出一个形状名"。
 */
const SHAPE_MATCH_MIN_CORRELATION = 0.95;

/**
 * 由预设求缩略图的折线点（`x0,y0 x1,y1 …`，y 向下为正）。
 *
 * 抽成纯函数以便单测：归一化方式（按自身峰值）、首末点、空输入的兜底都在这里。
 * 定标与 `previewScaleCents` 同理 —— 选择场景里"形状可辨"优先于"深度可比"。
 */
export function glyphPath(preset: VibratoPreset, width: number, height: number): string {
    const { wave } = buildVibratoPreview(preset, {
        frameCount: GLYPH_FRAME_COUNT,
        framePeriodMs: 5,
    });
    let peak = 1e-9;
    for (const value of wave) peak = Math.max(peak, Math.abs(value));

    const mid = height / 2;
    const reach = height / 2 - 1;
    const step = width / Math.max(1, wave.length - 1);
    const points: string[] = [];
    for (let i = 0; i < wave.length; i += 1) {
        const x = i * step;
        const y = mid - (wave[i] / peak) * reach;
        points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return points.join(" ");
}
