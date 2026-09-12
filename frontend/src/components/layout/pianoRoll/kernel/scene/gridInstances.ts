/**
 * 参数编辑器内核 · 网格实例构建（纯函数）
 *
 * 【主要内容】
 * 把参数编辑器的横向网格线（音高半音线 / 各类非音高参数的刻度线）计算为
 * **内容坐标**下的矩形实例，供 WebGL2 实例化渲染直接上传。
 *
 * 【作用】
 * 阶段 2 要把静态图层搬上 GL，前提是几何能脱离 Canvas 与 DOM 独立计算、独立测试。
 * 本模块只做算术：值域 → 视口 y → 设备像素对齐 → 实例矩形。它与 `render.ts`
 * 的 Canvas2D 路径**必须逐值等价**（包括两种不同的半像素取向），否则迁移后网格
 * 会整体偏移半个设备像素——表现为线条"发虚"，很难归因。
 *
 * 【与 render.ts 的对应关系】
 * - 音高分支   → `render.ts:672-740`
 * - cents 分支 → `render.ts:741-766`
 * - degrees 分支 → `render.ts:767-793`
 * - formant 分支 → `render.ts:794-821`
 *
 * 【为什么要显式防御 NaN / Infinity】原实现的 span 由调用方保证有限，因此直接进
 * `for` 循环。本模块跑在渲染热路径上，一旦上界为 `Infinity` 会**死循环卡死面板**，
 * 故对非有限 span 直接返回空数组。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在几何变化时调用。
 * - 下游：实例缓冲（`writeFlatInstance`）→ WebGL2。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React，可在 node 环境单测。
 */

import type { Rgba } from "../../../timeline/kernel/scene/instanceTypes";

/** 非音高参数的网格种类（决定步进与强线间隔）。 */
export type ValueGridKind = "cents" | "degrees" | "formantCents";

/**
 * 一条横向网格线的实例几何。
 *
 * 特殊说明（易错点）：横线的 `w` 是**横向范围**（横跨整个视口），`h` 是**线厚**。
 * 两者不可互换——写反会画出一条通高的竖条，而不是一条线。
 */
export interface GridInstance {
    /** 左缘 x（内容坐标 CSS px）。横线从视口左缘开始，故恒为 0。 */
    readonly x: number;
    /** 上缘 y（内容坐标 CSS px，已按该层的半像素取向对齐）。 */
    readonly y: number;
    /** 横向范围（CSS px，= 视口宽）。 */
    readonly w: number;
    /** 线厚（CSS px，弱线 1/dpr、强线 2/dpr）。 */
    readonly h: number;
    /** 颜色。 */
    readonly rgba: Rgba;
    /** 该行对应的参数值（供比例高亮与调试）。 */
    readonly value: number;
}

/** 音高网格构建参数。 */
export interface PitchGridArgs {
    /** 当前音高视口（值域中心与跨度）。 */
    readonly view: { readonly center: number; readonly span: number };
    /** 绝对值域下界（MIDI）。 */
    readonly absMin: number;
    /** 绝对值域上界（MIDI）。 */
    readonly absMax: number;
    /** 视口高度（CSS px）。 */
    readonly heightPx: number;
    /** 视口宽度（CSS px）——横线的横向范围。 */
    readonly viewportWidthPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /** 值 → 视口 y 的投影（与 Canvas2D 路径同一个函数，保证等价）。 */
    readonly valueToY: (value: number, heightPx: number) => number;
    /** C 音（pc === 0）的线色。 */
    readonly colorC: Rgba;
    /** 其余半音的线色。 */
    readonly colorOther: Rgba;
}

/** 非音高参数网格构建参数。 */
export interface ValueGridArgs {
    /** 参数种类（决定步进与强线间隔）。 */
    readonly kind: ValueGridKind;
    /** 当前视口（值域中心与跨度）。 */
    readonly view: { readonly center: number; readonly span: number };
    /** 视口高度（CSS px）。 */
    readonly heightPx: number;
    /** 视口宽度（CSS px）。 */
    readonly viewportWidthPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /** 值 → 视口 y 的投影。 */
    readonly valueToY: (value: number, heightPx: number) => number;
    /** 强线颜色。 */
    readonly strongRgba: Rgba;
    /** 弱线颜色。 */
    readonly weakRgba: Rgba;
}

/** 各参数的步进与强线间隔（与 render.ts 的分支逐一对应）。 */
const VALUE_GRID_SPEC: Record<ValueGridKind, { step: number; strongMod: number }> = {
    cents: { step: 100, strongMod: 1200 },
    degrees: { step: 1, strongMod: 7 },
    formantCents: { step: 50, strongMod: 600 },
};

/**
 * 弱线 / 普通线的设备像素对齐：取整到物理像素后**再加半个物理像素**。
 *
 * 与 `render.ts:440` 的 `hairlineY` 逐字一致（注意它加的是 0.5 个**设备**像素，
 * 即 `0.5/dpr` CSS 像素——不是 0.5 个 CSS 像素）。
 *
 * @param cssY 视口 y（CSS px）。
 * @param dpr 设备像素比。
 * @returns 对齐后的 y（CSS px）。
 */
function hairlineY(cssY: number, dpr: number): number {
    return (Math.round(cssY * dpr) + 0.5) / dpr;
}

/**
 * 强线的设备像素对齐：只取整到物理像素，**不加**半像素。
 *
 * 与 `render.ts:752` 一致：强线宽 2 个物理像素，偶数宽度无需半像素偏移。
 * 两种取向并存是既有行为，迁移时必须原样保留（统一它们会改变像素）。
 *
 * @param cssY 视口 y（CSS px）。
 * @param dpr 设备像素比。
 * @returns 对齐后的 y（CSS px）。
 */
function snapY(cssY: number, dpr: number): number {
    return Math.round(cssY * dpr) / dpr;
}

/**
 * 构建音高网格线实例（每个整数半音一条）。
 *
 * 流程：钳制 span 与值域 → 求可见半音区间 `[floor(min), ceil(max)]`（**含端点**）
 * → 逐半音算 y → pc === 0 用 C 线色，其余用普通线色。
 *
 * 特殊说明 1：区间含端点是刻意的（`render.ts:698` 用 `<=`），与键盘列使用的
 * `< endMidi` 存在一处不对称；这属于既有行为，不在迁移中"顺手修正"。
 *
 * 特殊说明 2：所有线都用弱线取向（`hairlineY`）——音高分支没有强线概念，
 * C 线只是**换色**、不换宽度与取向。
 *
 * @param args 构建参数。
 * @returns 网格线实例（按半音升序）；参数非法时返回空数组。
 */
export function buildPitchGridInstances(args: PitchGridArgs): GridInstance[] {
    const { view, absMin, absMax, heightPx, viewportWidthPx, dpr, valueToY } = args;
    if (!Number.isFinite(view.span) || !Number.isFinite(view.center)) return [];
    if (!Number.isFinite(heightPx) || !Number.isFinite(dpr) || dpr <= 0) return [];

    const range = absMax - absMin;
    if (!Number.isFinite(range) || range <= 0) return [];

    const span = Math.min(Math.max(view.span, 1e-6), range);
    const min = Math.min(Math.max(view.center - span / 2, absMin), absMax - span);
    const max = min + span;
    const startMidi = Math.min(Math.max(Math.floor(min), absMin), absMax);
    const endMidi = Math.min(Math.max(Math.ceil(max), absMin), absMax);

    const thickness = 1 / dpr;
    const items: GridInstance[] = [];
    for (let midi = startMidi; midi <= endMidi; midi += 1) {
        const y = hairlineY(valueToY(midi + 0.5, heightPx), dpr);
        const pc = ((midi % 12) + 12) % 12;
        items.push({
            x: 0,
            y,
            w: viewportWidthPx,
            h: thickness,
            rgba: pc === 0 ? args.colorC : args.colorOther,
            value: midi,
        });
    }
    return items;
}

/**
 * 构建非音高参数的刻度线实例。
 *
 * 流程：按 `kind` 取步进与强线间隔 → 求可见值区间内的步进点 → 强线用 `snapY`
 * 与 2 倍线厚，弱线用 `hairlineY` 与 1 倍线厚。
 *
 * 特殊说明：`span` 按 `render.ts:743` 的 `Math.max(1e-6, …)` 取下限，因此
 * `span = 0` 会退化为**单行**（域为 `[center, center]`），而不是空数组。
 * 非有限 span 直接返回空数组（见文件头：防死循环）。
 *
 * @param args 构建参数。
 * @returns 刻度线实例（按值升序）；参数非法时返回空数组。
 */
export function buildValueGridInstances(args: ValueGridArgs): GridInstance[] {
    const { kind, view, heightPx, viewportWidthPx, dpr, valueToY } = args;
    if (!Number.isFinite(view.span) || !Number.isFinite(view.center)) return [];
    if (!Number.isFinite(heightPx) || !Number.isFinite(dpr) || dpr <= 0) return [];

    const spec = VALUE_GRID_SPEC[kind];
    const span = Math.max(1e-6, view.span);
    const vMin = view.center - span / 2;
    const vMax = view.center + span / 2;
    const start = Math.ceil(vMin / spec.step) * spec.step;

    const weakThickness = 1 / dpr;
    const strongThickness = 2 / dpr;
    const items: GridInstance[] = [];
    for (let v = start; v <= vMax + spec.step * 0.01; v += spec.step) {
        const isStrong = Math.round(v) % spec.strongMod === 0;
        const y = isStrong
            ? snapY(valueToY(v, heightPx), dpr)
            : hairlineY(valueToY(v, heightPx), dpr);
        items.push({
            x: 0,
            y,
            w: viewportWidthPx,
            h: isStrong ? strongThickness : weakThickness,
            rgba: isStrong ? args.strongRgba : args.weakRgba,
            value: v,
        });
    }
    return items;
}
