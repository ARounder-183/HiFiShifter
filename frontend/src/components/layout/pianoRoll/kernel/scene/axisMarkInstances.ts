/**
 * 参数编辑器内核 · 数值轴刻度实例构建（纯函数）
 *
 * 【主要内容】
 * 把左侧数值轴（非音高参数）的**刻度线**计算为视口坐标下的矩形实例，以及每个
 * 刻度对应的**标签文本**（文本本身由字形管线绘制，见 `pianoRollGlyphs`）。
 *
 * 【作用】
 * 数值轴刻度是静态图层：只随参数与值域视口变化，播放与横向滚动都不影响它。
 * 阶段 2 把它搬上 GL 后，播放帧只需重绘播放头。
 *
 * 【与 render.ts 的对应关系】
 * - cents 分支         → `render.ts:507-562`
 * - formant 分支       → `render.ts:563-586`
 * - degrees 分支       → `render.ts:587-613`
 * - 回退（nice step）  → `render.ts:614-628`
 *
 * 【为什么要抽出"步长选择"】原实现把候选步长表与选择循环内联在四个分支里，
 * 且 **GL 侧必须做出完全相同的选择**——否则两条路径的刻度密度会不同。把选择
 * 逻辑收敛到本模块，Canvas2D 与 GL 共用同一份，杜绝分叉。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在几何变化时调用；`render.ts` 复用 `resolveAxisStep`。
 * - 下游：实例缓冲（`writeFlatInstance`）→ WebGL2；标签文本 → 字形管线。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React。
 */

import {
    childPitchOffsetValueToDisplay,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
} from "../../childPitchOffsetParams";

/** 数值轴刻度的种类。 */
export type AxisKind = "cents" | "formantCents" | "degrees" | "fallback";

/** 一条刻度线实例 + 它对应的标签文本。 */
export interface AxisMarkInstance {
    /** 是否强刻度（决定线宽与标签字重）。 */
    readonly isStrong: boolean;
    /** 刻度对应的**内部**参数值（未经显示变换）。 */
    readonly value: number;
    /** 标签文本（已按显示规则格式化）。 */
    readonly label: string;
    /** 刻度线的矩形实例（视口坐标）。 */
    readonly line: {
        readonly x: number;
        readonly y: number;
        readonly w: number;
        readonly h: number;
    };
}

/** 刻度线构建参数。 */
export interface AxisMarkArgs {
    /** 刻度种类。 */
    readonly kind: AxisKind;
    /** 当前值域视口（中心与跨度）。 */
    readonly view: { readonly center: number; readonly span: number };
    /** 视口高度（CSS px）。 */
    readonly heightPx: number;
    /** 轴宽（CSS px）——刻度线横跨整个轴列。 */
    readonly axisWidthPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /** 值 → 视口 y 的投影。 */
    readonly valueToY: (value: number, heightPx: number) => number;
    /**
     * 参数名（用于标签的显示换算）。
     *
     * 特殊说明：度数参数的内部值是 degree-step 单位，标签必须经
     * `childPitchOffsetValueToDisplay` 换算；缺省时标签直接用内部值。
     */
    readonly paramName?: string;
}

/**
 * 为数值轴选择"好看"的刻度步长。
 *
 * 与 `render.ts:64-77` 的 `niceAxisStep` 逐字一致（1 / 2 / 5 / 10 档）。
 *
 * @param range 值域跨度。
 * @param targetCount 期望的刻度数量。
 * @returns 步长。
 */
export function niceAxisStep(range: number, targetCount: number): number {
    const roughStep = range / targetCount;
    const mag = Math.pow(10, Math.floor(Math.log10(roughStep)));
    const normalized = roughStep / mag;
    let nice: number;
    if (normalized < 1.5) nice = 1;
    else if (normalized < 3.5) nice = 2;
    else if (normalized < 7.5) nice = 5;
    else nice = 10;
    return nice * mag;
}

/**
 * 选择各类参数的主刻度步长与强刻度间隔。
 *
 * 流程：按种类取候选步长表 → 选第一个使刻度数落在 [5, 12] 的候选 → cents 额外
 * 做一次"跨度过大时回退到更粗步长"的修正（与 `render.ts:519-533` 一致）。
 *
 * 特殊说明 1：候选步长表是**降序**的（从粗到细），因此"第一个满足数量约束"的
 * 就是可用的最粗步长——这正是原实现的意图。
 *
 * 特殊说明 2：degrees 的强刻度间隔是常量 7（内部 degree-step 单位），不是从
 * 步长推导的；cents / formant 的强刻度间隔分别是 1200 / 600。
 *
 * @param kind 刻度种类。
 * @param range 值域跨度（vMax − vMin）。
 * @returns 主步长与强刻度间隔。
 */
export function resolveAxisStep(
    kind: AxisKind,
    range: number,
): { step: number; strongMod: number } {
    if (kind === "degrees") {
        // degrees 的度数经 childPitchOffsetValueToDisplay 换算，强刻度每 7 个内部单位。
        const candidates = [14, 7, 3, 1];
        let chosen = candidates[candidates.length - 1];
        for (const c of candidates) {
            const count = Math.ceil(range / c) + 1;
            if (count >= 5 && count <= 12) {
                chosen = c;
                break;
            }
        }
        return { step: chosen, strongMod: 7 };
    }

    const candidates = [1200, 600, 300, 200, 100, 50, 25, 10, 5, 1];
    let chosen = candidates[candidates.length - 1];
    for (const c of candidates) {
        const count = Math.ceil(range / c) + 1;
        if (count >= 5 && count <= 12) {
            chosen = c;
            break;
        }
    }

    if (kind === "cents") {
        // 退化修正：跨度过大时刻度会过密，回退到更粗的"好看"步长。
        const approxCount = range / chosen;
        if (approxCount > 12) {
            const niceStep = niceAxisStep(range, 8);
            if (niceStep > chosen) {
                chosen = niceStep;
            } else {
                const largerCandidate = candidates.find((c) => c > chosen);
                if (largerCandidate !== undefined) chosen = largerCandidate;
            }
        }
        return { step: chosen, strongMod: 1200 };
    }
    if (kind === "formantCents") {
        return { step: chosen, strongMod: 600 };
    }
    // 回退分支**不用**候选表：`render.ts:614-628` 用的是 `niceAxisStep(span, 4)`。
    // 用候选表会让小跨度下的步长差几个数量级（实测 span=1e-6 时 1 个刻度 vs 5 个）。
    return { step: niceAxisStep(range, 4), strongMod: Number.POSITIVE_INFINITY };
}

/**
 * 格式化刻度数值，避免浮点噪声。
 *
 * 与 `render.ts:79-86` 的 `formatAxisMark` 一致：先按参数的显示规则换算
 * （度数需要换算），再取 4 位有效数字并去掉尾随零。
 *
 * 特殊说明：`toPrecision(4)` 对极大 / 极小值会输出**指数记法**（如 `1e-7`），
 * 因此字形图集必须覆盖 `e`、`+`、`-` 与数字——不能假设标签只有数字和小数点。
 *
 * @param value 内部参数值。
 * @param param 参数名（用于度数换算）；缺省时不做换算。
 * @returns 标签文本。
 */
export function formatAxisMarkLabel(value: number, param?: string): string {
    const displayValue = param != null ? childPitchOffsetValueToDisplay(param, value) : value;
    return parseFloat(displayValue.toPrecision(4)).toString();
}

/**
 * 判定参数对应的刻度种类。
 *
 * @param param 参数名。
 * @returns 刻度种类；无法识别时回退到 `"fallback"`（nice step）。
 */
export function resolveAxisKind(param: string): AxisKind {
    if (isChildPitchOffsetCentsParam(param)) return "cents";
    if (isChildFormantOffsetCentsParam(param)) return "formantCents";
    if (isChildPitchOffsetDegreesParam(param)) return "degrees";
    return "fallback";
}

/**
 * 构建数值轴刻度实例。
 *
 * 流程：钳制 span → 由 `resolveAxisStep` 取步长 → 求可见区间内的刻度值 →
 * 逐刻度算 y 与标签 → 产出刻度线矩形（strong 用 1.25、weak 用 1.0 的线宽，
 * 与 `render.ts:557` 一致）。
 *
 * 特殊说明 1：刻度线的 y 是 `valueToY(...) + 0.5`、线宽 1 或 1.25（`render.ts:559`）。
 * 这是 dpr=1 时代的整数对齐写法（加 0.5 个 **CSS** 像素，不是半个设备像素），在
 * 分数 DPR 下并非严格的物理像素对齐——但它就是现有行为，必须原样保留。
 *
 * 特殊说明 2：degrees 分支会**额外**补一条 0 刻度（`render.ts:611-613`），因为
 * 步长可能不落在 0 上。补的那条只画标签、不画线（原实现的 `fillText` 没有配对
 * 的 `stroke`），因此本模块返回的实例 `isZeroOnly` 为真时调用方不应画线。
 *
 * @param args 构建参数。
 * @returns 刻度实例（按值升序）；参数非法时返回空数组。
 */
export function buildAxisMarkInstances(
    args: AxisMarkArgs,
): (AxisMarkInstance & { readonly lineOnly: boolean })[] {
    const { kind, view, heightPx, axisWidthPx, dpr, valueToY, paramName } = args;
    if (!Number.isFinite(view.span) || !Number.isFinite(view.center)) return [];
    if (!Number.isFinite(heightPx) || !Number.isFinite(axisWidthPx)) return [];
    if (!Number.isFinite(dpr) || dpr <= 0) return [];

    const span = Math.max(1e-6, view.span);
    const vMin = view.center - span / 2;
    const vMax = view.center + span / 2;
    const { step, strongMod } = resolveAxisStep(kind, vMax - vMin);
    if (!Number.isFinite(step) || step <= 0) return [];

    const firstMark = Math.ceil(vMin / step) * step;
    // 与 render.ts 一致：上界放宽 1% 步长，避免浮点误差吃掉最后一个刻度。
    const limit = vMax + step * 0.01;

    /** 线的描边中心换算为上缘（stroke 以中心对齐，矩形以上缘对齐）。 */
    const toLineRect = (y: number, isStrong: boolean) => {
        const width = isStrong ? 1.25 : 1;
        const center = y + 0.5;
        return { x: 0, y: center - width / 2, w: axisWidthPx, h: width };
    };

    const out: (AxisMarkInstance & { readonly lineOnly: boolean })[] = [];
    for (let v = firstMark; v <= limit; v += step) {
        const y = valueToY(v, heightPx);
        const isStrong = Number.isFinite(strongMod) ? Math.round(v) % strongMod === 0 : false;
        out.push({
            isStrong,
            value: v,
            label: formatAxisMarkLabel(v, paramName),
            line: toLineRect(y, isStrong),
            lineOnly: false,
        });
    }

    // degrees：**无条件**补一条只画标签的 0 刻度。
    //
    // 【为什么不去重】`render.ts:611-613` 是无条件 `fillText` 的，而 degrees 的步长
    // 是整数、刻度序列是步长的整数倍，因此**只要值域视口包含 0，0 就已经是普通
    // 刻度**——此时原实现会在同一位置画两次标签，两次 alpha 叠加后明显更深
    // （0.55 -> 0.7975）。这是常见路径（degrees 的视口中心默认就在 0 附近），
    // 不是边角情形。迁移必须逐字复刻，否则 0 标签会变浅。
    //
    // `lineOnly` 为真表示"只有标签、没有配对的分隔线"——这正是两次绘制中多出来的
    // 那一次（它没有 `stroke`）。
    if (kind === "degrees") {
        out.push({
            isStrong: false,
            value: 0,
            label: formatAxisMarkLabel(0, paramName),
            line: toLineRect(valueToY(0, heightPx), false),
            lineOnly: true,
        });
    }
    return out;
}
