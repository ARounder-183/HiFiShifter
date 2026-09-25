/**
 * 波形几何缓存的**复用判定**（纯函数）。
 *
 * 【主要内容】给出「上一帧构建的几何能否直接平移复用（`repaint()`）」的判定，
 * 以及该判定所依赖的两份数据：几何的**构建锚点**与本次绘制的**查询**。
 *
 * 【作用：为什么这段逻辑必须独立成模块】它是波形性能的命门——判定为真时，
 * 一帧的成本是「一次 uniform 更新 + `drawArrays`」，与 clip 数、像素列数
 * **完全无关**；判定为假时则要重建场景与几何（实测 400 clip / 全览 ≈ 10.8 ms）。
 * 这样一段决定 10 ms 级差异的逻辑，不能藏在 React 组件的 `useCallback` 里
 * 靠肉眼推演——必须能被单测钉住。
 *
 * 【与其他模块的关系】
 * - 上游：`WaveformSurface.draw()` 用当前 props 组装 `WaveformReuseQuery`。
 * - 生产：`buildWaveformScene` + `buildWaveformGeometry` 之后由
 *   `WaveformSurface` 记录 `WaveformGeometryAnchor`。
 * - 依赖：`sceneBuilder` 的行类型与 `geometry` 的幅度映射类型（仅类型）。
 *
 * 【几何坐标系回顾（判定成立的前提）】顶点是**窗口局部坐标** =
 * 内容坐标 − 构建窗口左上角；屏幕位置 = 局部坐标 − 视口原点。因此只要视口
 * 仍落在构建窗口内，平移就只是改一次 `u_viewOrigin`，几何一行都不用重算。
 */

import type { WaveformAmplitudeMap } from "./geometry.ts";
import type { WaveformSceneRow } from "./sceneBuilder.ts";

/** 几何的**构建锚点**：记录「GPU / Canvas2D 上这份几何是按什么条件构建的」。 */
export interface WaveformGeometryAnchor {
    /** 构建时的水平缩放。 */
    pxPerSec: number;
    /** 构建时的视口宽（CSS px）。 */
    widthPx: number;
    /** 构建时的视口高（CSS px）。 */
    heightPx: number;
    /** 构建时的设备像素比：包络列按设备像素网格枚举，dpr 变了几何必须重建。 */
    dpr: number;
    /** 构建时的行数据。按**引用**比较（React 侧 memo 的产物，引用不变即内容不变）。 */
    rows: readonly WaveformSceneRow[];
    /** 构建时的描边色。 */
    color: string;
    /** 构建时的渲染后端。换后端时 GPU 侧缓冲已重置，几何必须重建。 */
    rendererKind: "webgl2" | "canvas2d";
    /**
     * 构建时使用的幅度映射。按**引用**比较：调用方每换一个面板语义就必须
     * 传入新引用（参数编辑器用 useMemo 绑定 editParam 实现）。
     */
    amplitudeMap: WaveformAmplitudeMap | undefined;
    /** 构建时的幅度映射修订号（见 `readAmplitudeRevision`）。 */
    amplitudeRevision: number;
    /** 构建窗口的内容坐标左边界（含余量）。 */
    windowStartPx: number;
    /** 构建窗口的内容坐标右边界（含余量）。 */
    windowEndPx: number;
    /** 构建窗口的内容坐标上边界（行覆盖范围顶端）。 */
    windowTopPx: number;
    /**
     * 几何实际覆盖的内容坐标底端（各行波形带底边的最大值，调用方写入时
     * **不带任何余量**）：竖直复用要求视口底边不超过它。
     */
    windowBottomPx: number;
}

/** 本次绘制的查询：当前视口与「几何覆盖的 clip 集合是否可信」。 */
export interface WaveformReuseQuery {
    pxPerSec: number;
    widthPx: number;
    heightPx: number;
    dpr: number;
    rows: readonly WaveformSceneRow[];
    color: string;
    rendererKind: "webgl2" | "canvas2d";
    amplitudeMap: WaveformAmplitudeMap | undefined;
    amplitudeRevision: number;
    /** 当前视口左缘（内容坐标，CSS px）。 */
    scrollLeftPx: number;
    /** 当前视口上缘（内容坐标，CSS px）。 */
    scrollTopPx: number;
    /**
     * 调用方对 `rows` 的**水平完整性**承诺，见
     * {@link WaveformSurfaceProps.rowsCoverViewport}。
     */
    rowsCoverViewport: boolean;
}

/**
 * 判定能否复用已构建的几何（即走 `repaint()`）。
 *
 * 【为什么需要 `rowsCoverViewport` 而不是「有没有视口总线」】
 * 复用成立的隐含前提是「几何覆盖的 clip 集合 ⊇ 当前视口内的 clip 集合」。
 * 几何是按**构建窗口**裁剪过的：视口若越出窗口，窗口外的 clip 本来就没被画过，
 * 平移旧几何只会让它们「消失」。这是**窗口包含判定**负责的部分（下面第 2 组条件）。
 *
 * 但还有一层更隐蔽的失效：`rows` 本身可能**不含**视口内的 clip。参数编辑器就是
 * 这种情况——它的 `props.rows` 由 **256px 量化提交**的 `scrollLeft` 算出
 * （`SCROLL_COMMIT_STEP_PX`），滞后内核真值最多 255px：向左滚时新 clip 已进视口，
 * 而 `rows` 还是旧的，几何里根本没有它；此时「视口仍在旧窗口内」成立，若复用就会
 * 把缺了该 clip 的几何原样平移上去——用户看到「左侧进入的 clip 波形消失，随着拖动
 * 又恢复，不松手往回拖又消失」。
 *
 * 曾经的处理是**按驱动方式一刀切**：只要由视口总线驱动就恒不复用。这在时间轴上
 * 是错的——时间轴的 `rows` **不做水平窗口化**（`TimelineKernelView` 把每条可见
 * 轨道的全部 clip 都放进 map，见 `waveformClipsByTrackId`），因此它天然满足
 * 「几何 ⊇ 视口」的前提，却被那一条 `source === null` 永久否决，每个滚动帧都付
 * 一次全量重建。
 *
 * 正确的切分维度是**数据本身是否水平窗口化**，而不是「谁在驱动绘制」。于是把
 * 这个事实显式化为调用方的承诺 `rowsCoverViewport`：
 * - `true`  = 「我的 rows 水平上不窗口化，视口移到哪儿所需的 clip 都在里面」；
 * - `false` = 保守（每次重建），适用于水平窗口化的调用方。
 *
 * 【两组条件的分工】
 * 1. **锚点全等**：缩放 / 尺寸 / dpr / rows 引用 / 颜色 / 后端 / 幅度映射与其
 *    修订号，任一变化都必须重建（这些字段共同决定了顶点数据本身）。
 * 2. **视口仍落在构建窗口内**：水平方向要求视口完整落在
 *    `[windowStartPx, windowEndPx]`；竖直方向要求视口完整落在
 *    `[windowTopPx, windowBottomPx]` —— `windowBottomPx` 是几何实际覆盖的
 *    底端（各行波形带底边的最大值）：底边越出它的部分没有任何几何可画，
 *    平移复用会在视口底部留出一条空白（快速竖直平移时肉眼可见）。
 *
 * 只有 `rowsCoverViewport` 为真、且上面两组条件全部成立时才允许复用。
 *
 * @param anchor 上一帧构建的几何锚点；`null` 表示尚无几何。
 * @param query 本次绘制的视口与数据。
 * @returns 可以只平移复用（`repaint()`）时为 `true`。
 */
export function canReuseGeometry(
    anchor: WaveformGeometryAnchor | null,
    query: WaveformReuseQuery,
): boolean {
    if (anchor === null) return false;
    // ── 第 0 组：rows 必须对水平视口完整（见上方说明）──────────────
    if (!query.rowsCoverViewport) return false;
    // ── 第 1 组：锚点全等 ────────────────────────────────────────
    if (
        anchor.pxPerSec !== query.pxPerSec ||
        anchor.widthPx !== query.widthPx ||
        anchor.heightPx !== query.heightPx ||
        anchor.dpr !== query.dpr ||
        anchor.rows !== query.rows ||
        anchor.color !== query.color ||
        anchor.rendererKind !== query.rendererKind ||
        anchor.amplitudeMap !== query.amplitudeMap ||
        anchor.amplitudeRevision !== query.amplitudeRevision
    ) {
        return false;
    }
    // ── 第 2 组：视口仍落在构建窗口内 ─────────────────────────────
    // 水平：视口必须完整落在已构建的窗口内（两侧各 `marginPx` 可平移）。
    if (query.scrollLeftPx < anchor.windowStartPx) return false;
    if (query.scrollLeftPx + query.widthPx > anchor.windowEndPx) return false;
    // 竖直：视口必须**完整**落在几何实际覆盖的行范围内 —— 顶边不得高于行
    // 覆盖顶端，底边不得越出几何底端（windowBottomPx = 各行波形带底边的
    // 最大值）。几何只画行波形带，越出底端的部分没有任何几何可画，平移
    // 复用会让视口底部出现一条空白（快速竖直平移时可感）；旧判定只查顶边
    // 并默认「overscan 会盖住底边」，但行覆盖本身就是几何覆盖的边界，该
    // 默认不成立。
    if (query.scrollTopPx < anchor.windowTopPx) return false;
    if (query.scrollTopPx + query.heightPx > anchor.windowBottomPx) return false;
    return true;
}
