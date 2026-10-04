/**
 * 波形几何窗口与渲染原点的计算（纯函数）。
 *
 * 【主要内容】由「滚动位置 + 画布尺寸 + dpr + 行数据覆盖范围」算出：
 * - 几何要构建的内容坐标窗口 `[windowStart, windowEnd] × [windowTop, windowBottom]`；
 * - 渲染原点 `(originXPx, originYPx)` = 视口左上角在该窗口局部坐标中的位置。
 *
 * 【核心不变式：原点必须落在设备像素网格上】
 * 顶点是**窗口局部坐标**，着色器/`setTransform` 把屏幕位置算作
 * `screen = local − snap(origin, dpr)`（渲染器统一吸附原点，见
 * `surfaceRenderer.snappedOriginPx`）。而理想的屏幕位置是
 * `contentX − scrollLeft`。代入 `local = contentX − windowStart`：
 * ```
 * screen = contentX − scrollLeft + [origin − snap(origin, dpr)]
 *                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 残差
 * ```
 * 残差只取决于 `origin` 的小数相位。**它必须恒为 0**，否则整幅波形相对网格 /
 * Clip / 参数曲线平移最多半个物理像素。
 *
 * 历史缺陷（用户报告的"拖动窗口宽度时波形水平抖动"）：
 * - 水平方向 `origin = scrollLeft − windowStart = marginPx`（余量），而余量取
 *   `round(width × 0.25)` —— **整数 CSS 像素**。dpr 为 1.25 / 1.5 这类分数时
 *   `margin × dpr` 不是整数，吸附残差在 ±0.5 物理像素间循环；余量随视口宽度
 *   变化，于是拖动宽度时波形反复跳变（实测 800px 宽度扫描内跳变 200 次）。
 * - 竖直方向 `origin = scrollTop − windowTop`，`windowTop` 取首行 top（整数内容
 *   坐标），同样在分数 dpr 下产生残差（表现为恒定亚像素错位，行高变化时抖动）。
 *
 * 因此本模块把**两个方向的原点都构造为设备像素对齐**：水平取整数个物理像素的
 * 余量；竖直先吸附 `scrollTop − firstRowTop`，再由它反推窗口起点（起点因此可能
 * 偏离首行 top 半个物理像素以内 —— 它只是几何的坐标基准，该偏移与原点严格抵消，
 * 屏幕位置逐值不变）。
 *
 * 【与其他模块的关系】
 * - 上游：`WaveformSurface.draw()`；
 * - 下游：`sceneBuilder`（消费窗口）、`surfaceRenderer`（消费原点）、
 *   `geometryCache`（把窗口记为锚点，用于复用判定）。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React，可在 node 环境单测。
 */

import { snapToDevicePx } from "../utils/devicePixelLine";

/**
 * 水平余量的下限（CSS px 的**意图值**）。
 *
 * 余量的作用是让平移在窗口内完成、不必重建几何。下限保证一次快速平移不会立刻
 * 越界；上限防止窗口过大而白白多建几何。
 */
export const WAVEFORM_MARGIN_MIN_PX = 128;
/** 水平余量的上限（CSS px 的意图值）。 */
export const WAVEFORM_MARGIN_MAX_PX = 512;
/** 余量占视口宽的比例（在上下限之间取值）。 */
const WAVEFORM_MARGIN_RATIO = 0.25;

export interface WaveformWindowArgs {
    /** 视口水平滚动位置（内容坐标，绘制坐标口径）。 */
    readonly scrollLeftPx: number;
    /** 视口竖直滚动位置（内容坐标）。 */
    readonly scrollTopPx: number;
    /** 画布宽（CSS px，已按设备像素吸附）。 */
    readonly widthPx: number;
    /** 画布高（CSS px）。 */
    readonly heightPx: number;
    /** 设备像素比。 */
    readonly dpr: number;
    /**
     * 是否给水平方向留余量。
     *
     * WebGL 路径留（平移只需更新 uniform，加宽窗口零额外成本）；Canvas2D 回退
     * 不留（它没有顶点缓冲，平移要重放 path，加宽窗口纯属浪费）。
     */
    readonly horizontalOverscan: boolean;
    /** 行数据里最小的行顶（内容坐标）；无行时为 `+Infinity`。 */
    readonly firstRowTopPx: number;
    /** 几何实际覆盖的底端（内容坐标）；无行时为 `-Infinity`。 */
    readonly geometryBottomPx: number;
}

/** 几何窗口（内容坐标）与渲染原点（窗口局部坐标）。 */
export interface WaveformWindow {
    readonly windowStartPx: number;
    readonly windowEndPx: number;
    readonly windowTopPx: number;
    readonly windowBottomPx: number;
    /** 渲染原点 x = 视口左缘在窗口局部坐标中的位置。**恒为整数个物理像素。** */
    readonly originXPx: number;
    /** 渲染原点 y = 视口上缘在窗口局部坐标中的位置。**恒为整数个物理像素。** */
    readonly originYPx: number;
}

/**
 * 把水平余量取为**整数个物理像素**。
 *
 * 先在设备像素空间做上下限钳制再换回 CSS px —— 若先按 CSS 取整再钳制到整数
 * 边界（128 / 512），分数 dpr 下会重新变成非设备对齐的值，残差随之回归。
 *
 * 注意 dpr = 1 时结果与旧的 `round(width × 0.25)` 钳制**逐值相同**，行为不变。
 */
function deviceAlignedMarginPx(widthPx: number, dpr: number): number {
    const intentPx = Math.min(
        WAVEFORM_MARGIN_MAX_PX,
        Math.max(WAVEFORM_MARGIN_MIN_PX, Math.round(widthPx * WAVEFORM_MARGIN_RATIO)),
    );
    const minDevicePx = Math.round(WAVEFORM_MARGIN_MIN_PX * dpr);
    const maxDevicePx = Math.round(WAVEFORM_MARGIN_MAX_PX * dpr);
    const devicePx = Math.min(maxDevicePx, Math.max(minDevicePx, Math.round(intentPx * dpr)));
    return devicePx / dpr;
}

/**
 * 计算几何窗口与渲染原点。
 *
 * @param args 见 `WaveformWindowArgs`。
 * @returns 窗口（内容坐标）与原点（窗口局部坐标，设备像素对齐）。
 */
export function computeWaveformWindow(args: WaveformWindowArgs): WaveformWindow {
    const dpr = Number.isFinite(args.dpr) && args.dpr > 0 ? args.dpr : 1;

    const marginPx = args.horizontalOverscan ? deviceAlignedMarginPx(args.widthPx, dpr) : 0;
    const windowStartPx = args.scrollLeftPx - marginPx;
    const windowEndPx = args.scrollLeftPx + args.widthPx + marginPx;

    // 竖直：先吸附「滚动位置 − 首行 top」得到原点，再由原点反推窗口起点，
    // 使 `windowTopPx` 与 `originYPx` 严格满足 `originYPx = scrollTop − windowTop`。
    const rawWindowTopPx = Number.isFinite(args.firstRowTopPx)
        ? args.firstRowTopPx
        : args.scrollTopPx;
    const originYPx = snapToDevicePx(args.scrollTopPx - rawWindowTopPx, dpr);
    const windowTopPx = args.scrollTopPx - originYPx;

    // 竖直窗口 = 行数据实际覆盖的内容范围。**不能**用 `heightPx` 推：它是画布
    // 高度，与行数据覆盖无关（上游轨道窗口化自带 overscan）。无行时才退回视口高。
    const windowBottomPx = Number.isFinite(args.geometryBottomPx)
        ? args.geometryBottomPx
        : args.scrollTopPx + args.heightPx;

    return {
        windowStartPx,
        windowEndPx,
        windowTopPx,
        windowBottomPx,
        originXPx: marginPx,
        originYPx,
    };
}
