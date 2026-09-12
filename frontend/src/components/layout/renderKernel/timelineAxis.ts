/**
 * 时间线统一坐标投影（TimelineAxis）
 *
 * 【主要内容】
 * 定义时间线的唯一坐标投影对象 `TimelineAxis`，以及在该投影下全部的时间↔像素
 * 换算函数。整个前端只允许在本文件内出现「时间 × pxPerSec」这一乘法。
 *
 * 【作用】
 * 为网格、标尺、clip 体画布、波形、播放头提供**同一份**缩放与位置基准，消除
 * 各图层各自换算造成的错位。历史问题是：波形走「先除后乘」`(t − s/p)·p`、
 * 其余图层走「先乘后减」`t·p − s`，二者在 IEEE754 下不等价；且各层的取整与
 * 宽度下限各不相同。详见
 * `docs/plans/2026-08-29-timeline-unified-axis-design.md`。
 *
 * 【与其他模块的关系】
 * - 生产方：`TimelinePanel` / `PianoRollPanel` 在渲染期用当前的 pxPerSec、
 *   scrollLeftPx、scrollTopPx、viewportWidthPx、dpr 构造 axis 向下传递；
 *   `timelineViewportBus` / `pianoRollViewportBus` 的快照在滚动期覆盖
 *   scrollLeftPx / pxPerSec 后重建 axis。
 * - 消费方：`runtime/timelineCanvasModel.ts`（clip 几何）、
 *   `waveform/sceneBuilder.ts`（波形几何）、`TimelineSurface.tsx`（播放头），
 *   后续阶段扩展至 `BackgroundGrid` / `TimeRuler` 与 Piano Roll 主画布。
 *
 * 【强制约束（评审检查项）】
 * 1. 任何图层不得直接读取 pxPerSec / scrollLeft 做乘法或减法，一律走本模块。
 * 2. 禁止用 `scrollLeft / pxPerSec` 还原成秒——这是历史错位根因之一。
 *    需要秒区间做可见性裁剪时，用 `viewportStartSec()` / `viewportEndSec()`。
 * 3. 像素位置与线宽必须经 `snapPx()` / `strokePx()` 对齐到设备像素栅格。
 * 4. axis 只接受「绘制坐标」；原生 DOM 坐标只在量测边界转换。
 */

/** 时间轴投影：描述「一秒画多少像素」以及「视口左上角落在内容何处」。 */
export interface TimelineAxis {
    /** 水平缩放：每秒对应的 CSS 像素数。必须 > 0。 */
    readonly pxPerSec: number;
    /**
     * 水平滚动位置（**绘制坐标**，非 DOM 原生 scrollLeft）。
     * 表示内容坐标原点相对视口左缘的偏移。
     */
    readonly scrollLeftPx: number;
    /** 竖直滚动位置（绘制坐标），波形与 clip 画布按此平移竖直内容。 */
    readonly scrollTopPx: number;
    /** 视口可见宽度（CSS 像素）。 */
    readonly viewportWidthPx: number;
    /** 设备像素比，供 snapPx / strokePx 做栅格对齐。 */
    readonly dpr: number;
}

/**
 * 可视元素的最小宽度（CSS 像素）。
 *
 * 宽度低于此值的 clip 在屏幕上仍要保留一条可见的体，否则拖拽与命中测试将
 * 无落点。clip 体画布与波形必须共用同一常量，否则极小 clip 处两者宽度会
 * 分叉（取 1 与历史 `Math.max(1, …)` 行为一致）。
 */
export const MIN_FEATURE_WIDTH_PX = 1;

/** pxPerSec 的下限，避免 0 或负值导致除零与反向坐标。 */
const MIN_PX_PER_SEC = 1e-9;

/**
 * 构造一个 axis。
 *
 * 流程：对 pxPerSec 做下限保护后冻结所有字段。
 *
 * @param init.pxPerSec 每秒像素数，必填，内部会夹到 >= MIN_PX_PER_SEC。
 * @param init.scrollLeftPx 水平绘制坐标偏移，缺省 0。
 * @param init.scrollTopPx 竖直绘制坐标偏移，缺省 0。
 * @param init.viewportWidthPx 视口宽度，缺省 1（避免除零）。
 * @param init.dpr 设备像素比，缺省 1。
 * @returns 冻结的 axis；字段非法时回退到安全值而不是抛错，
 *          保证渲染热路径不会因脏数据中断。
 */
export function createTimelineAxis(
    init: Partial<TimelineAxis> & { pxPerSec: number },
): TimelineAxis {
    const pxPerSec = Number.isFinite(init.pxPerSec)
        ? Math.max(MIN_PX_PER_SEC, init.pxPerSec)
        : MIN_PX_PER_SEC;
    const scrollLeftPx = Number.isFinite(init.scrollLeftPx) ? (init.scrollLeftPx as number) : 0;
    const scrollTopPx = Number.isFinite(init.scrollTopPx) ? (init.scrollTopPx as number) : 0;
    const viewportWidthPx = Number.isFinite(init.viewportWidthPx)
        ? Math.max(1, init.viewportWidthPx as number)
        : 1;
    const dpr = Number.isFinite(init.dpr) && (init.dpr as number) > 0 ? (init.dpr as number) : 1;
    return Object.freeze({
        pxPerSec,
        scrollLeftPx,
        scrollTopPx,
        viewportWidthPx,
        dpr,
    });
}

/**
 * 在已有 axis 上派生新 axis（结构共享，未变更字段保持原值）。
 *
 * 作用：滚动/缩放热路径上只需覆盖发生变化的字段即可得到新 axis；
 * 全等比较（`axisEquals`）可据此跳过无变化的重绘。
 *
 * @param base 基准 axis。
 * @param patch 需要覆盖的字段；`undefined` 表示沿用 base。
 * @returns 新 axis；若所有字段均等价则**返回 base 本身**，便于引用比较去重。
 */
export function withAxis(base: TimelineAxis, patch: Partial<TimelineAxis>): TimelineAxis {
    const next = createTimelineAxis({
        pxPerSec: patch.pxPerSec ?? base.pxPerSec,
        scrollLeftPx: patch.scrollLeftPx ?? base.scrollLeftPx,
        scrollTopPx: patch.scrollTopPx ?? base.scrollTopPx,
        viewportWidthPx: patch.viewportWidthPx ?? base.viewportWidthPx,
        dpr: patch.dpr ?? base.dpr,
    });
    return axisEquals(base, next) ? base : next;
}

/**
 * 判断两个 axis 是否完全等价（用于重绘去重）。
 *
 * @returns 五个字段逐个相等时为 true。
 */
export function axisEquals(a: TimelineAxis, b: TimelineAxis): boolean {
    return (
        a.pxPerSec === b.pxPerSec &&
        a.scrollLeftPx === b.scrollLeftPx &&
        a.scrollTopPx === b.scrollTopPx &&
        a.viewportWidthPx === b.viewportWidthPx &&
        a.dpr === b.dpr
    );
}

/**
 * 时间 → 内容坐标 x（CSS 像素）。
 *
 * 这是全工程**唯一**允许出现「秒 × pxPerSec」的地方：内容坐标系的原点是
 * 工程 0 秒，不随滚动变化，DOM 内容层、clip 体画布、网格、标尺都使用它。
 *
 * @param axis 当前投影。
 * @param sec 工程时间（秒）。
 * @returns 内容坐标下的 x（CSS 像素）。
 */
export function secToContentPx(axis: TimelineAxis, sec: number): number {
    if (!Number.isFinite(sec)) return 0;
    return sec * axis.pxPerSec;
}

/**
 * 内容坐标 x → 时间。
 *
 * @param axis 当前投影。
 * @param px 内容坐标 x（CSS 像素）。
 * @returns 工程时间（秒）。
 */
export function contentPxToSec(axis: TimelineAxis, px: number): number {
    if (!Number.isFinite(px)) return 0;
    return px / axis.pxPerSec;
}

/**
 * 时间 → 视口坐标 x（CSS 像素）。
 *
 * 视口坐标系以 sticky 画布左上角为原点，= 内容坐标减去 scrollLeftPx。
 * 波形几何与播放头使用此坐标系。
 *
 * @param axis 当前投影。
 * @param sec 工程时间（秒）。
 * @returns 视口坐标下的 x（CSS 像素）。
 */
export function secToViewportPx(axis: TimelineAxis, sec: number): number {
    return secToContentPx(axis, sec) - axis.scrollLeftPx;
}

/**
 * 视口坐标 x → 时间。
 *
 * @param axis 当前投影。
 * @param px 视口坐标 x（CSS 像素）。
 * @returns 工程时间（秒）。
 */
export function viewportPxToSec(axis: TimelineAxis, px: number): number {
    return (px + axis.scrollLeftPx) / axis.pxPerSec;
}

/**
 * 时长 → 可视元素宽度（CSS 像素），**带最小宽度下限**。
 *
 * 特殊说明：clip 体画布与波形必须共用本函数，否则极小 clip 处两者宽度分叉
 * ——这是历史问题之一（clip 有 `Math.max(1, …)`、波形没有）。
 *
 * @param axis 当前投影。
 * @param durationSec 时长（秒），负值按 0 处理。
 * @returns 宽度（CSS 像素），恒 >= MIN_FEATURE_WIDTH_PX。
 */
export function durationToWidthPx(axis: TimelineAxis, durationSec: number): number {
    const sec = Number.isFinite(durationSec) ? Math.max(0, durationSec) : 0;
    return Math.max(MIN_FEATURE_WIDTH_PX, sec * axis.pxPerSec);
}

/**
 * 时长 → 纯跨度（CSS 像素），**不带最小宽度下限**。
 *
 * 用途：淡入淡出长度、吸附偏移这类「可以为 0」的跨度。若误用
 * `durationToWidthPx`，0 长度的淡入会被撑成 1 像素的可见角标。
 *
 * @param axis 当前投影。
 * @param durationSec 时长（秒），负值按 0 处理。
 * @returns 跨度（CSS 像素），可为 0。
 */
export function secToSpanPx(axis: TimelineAxis, durationSec: number): number {
    const sec = Number.isFinite(durationSec) ? Math.max(0, durationSec) : 0;
    return sec * axis.pxPerSec;
}

/**
 * 把像素坐标吸附到设备像素栅格。
 *
 * 作用：1px 细线若落在非整数设备像素上会被抗锯齿摊成两条灰线（发虚）。
 * 所有描边位置必须先过本函数。
 *
 * @param axis 当前投影（取 dpr）。
 * @param px CSS 像素坐标。
 * @returns 对齐后的 CSS 像素坐标（设备像素上为整数）。
 */
export function snapPx(axis: TimelineAxis, px: number): number {
    if (!Number.isFinite(px)) return 0;
    return Math.round(px * axis.dpr) / axis.dpr;
}

/**
 * 计算描边线的居中坐标。
 *
 * 规则：奇数宽度的线要额外偏移半个设备像素，使线体正好覆盖整数个设备像素，
 * 否则 1px 线会跨在两个物理像素上变成 2px 灰线。
 *
 * @param axis 当前投影（取 dpr）。
 * @param px 线的目标位置（CSS 像素）。
 * @param widthPx 线宽（CSS 像素）。
 * @returns 描边应使用的坐标（CSS 像素）。
 */
export function strokePx(axis: TimelineAxis, px: number, widthPx: number): number {
    const snapped = snapPx(axis, px);
    const oddWidth = Math.max(0, Math.round(widthPx * axis.dpr)) % 2 === 1;
    return oddWidth ? snapped + 0.5 / axis.dpr : snapped;
}

/**
 * 视口左缘对应的时间（秒）。
 *
 * 特殊说明：**仅用于可见性裁剪与数据窗口选择**，不得再乘回 pxPerSec 做
 * 像素换算（那会退化成「先除后乘」，重新引入不等价）。
 *
 * @param axis 当前投影。
 * @returns 视口左缘时间（秒）。
 */
export function viewportStartSec(axis: TimelineAxis): number {
    return axis.scrollLeftPx / axis.pxPerSec;
}

/**
 * 视口右缘对应的时间（秒）。用途与限制同 `viewportStartSec`。
 *
 * @param axis 当前投影。
 * @returns 视口右缘时间（秒），恒 >= 起始时间。
 */
export function viewportEndSec(axis: TimelineAxis): number {
    return Math.max(viewportStartSec(axis), viewportEndSecUnclamped(axis));
}

function viewportEndSecUnclamped(axis: TimelineAxis): number {
    return (axis.scrollLeftPx + axis.viewportWidthPx) / axis.pxPerSec;
}
