/**
 * 时间轴渲染内核 · GL 光栅化参数
 *
 * 【主要内容】
 * 由「CSS 尺寸 + devicePixelRatio」算出画布物理尺寸与绘制坐标系尺寸：
 * 物理尺寸 = `round(css × dpr)`（画布真实像素），绘制坐标系尺寸 = `物理 / dpr`。
 *
 * 【作用】
 * 与既有 `runtime/canvasRaster.rasterize` 保持同一契约（物理尺寸取整到整数），
 * 但把「绘制坐标系尺寸」显式回算为 `物理 / dpr`：WebGL 的 `u_resolution` 必须用
 * 这个值，才能保证顶点坐标（CSS px）与物理像素 1:1 对应。若直接用传入的 CSS 尺寸，
 * 在分数 DPR 下会出现半个物理像素的累积偏移（网格线相位漂移）。
 *
 * 同一个回算值也用于写回 `canvas.style.width`（见 `cssWidthPx` 的说明）：
 * 布局盒必须与 backing store 严格成 `dpr` 倍关系，否则合成器重采样会让整块
 * 画面发虚。
 *
 * 【与其他模块的关系】
 * - 上游：宿主视图在 `ResizeObserver` 回调里调用；
 * - 下游：`glContext` 用它设置 `canvas.width/height` 与 `gl.viewport`，
 *   program 用它设置 `u_resolution`。
 * - 独立性：纯函数，不依赖 DOM / WebGL，可直接单测。
 */

/** 光栅化目标：画布物理尺寸与绘制坐标系尺寸。 */
export interface GlRasterTarget {
    /** 画布物理宽（= `canvas.width`）。 */
    readonly physicalWidthPx: number;
    /** 画布物理高（= `canvas.height`）。 */
    readonly physicalHeightPx: number;
    /**
     * 绘制坐标系宽（= 物理宽 / dpr），供 `u_resolution`。
     *
     * 特殊说明：这是**回算值**，不等于入参的 CSS 宽度（非整数尺寸下会差最多
     * `0.5 / dpr`）。它同时是**唯一**应当写回 `canvas.style.width` 的值：
     * `cssWidthPx * dpr` 严格等于 `physicalWidthPx`，布局盒与 backing store
     * 一一对应，合成器缩放比恒为 1。若改用入参的原始 CSS 宽，布局盒会与
     * backing store 差最多半个物理像素，合成器做非整数倍重采样 —— 表现为
     * "窗口拖到某些宽度画面发虚"，非整数 dpr 下按宽度奇偶交替。
     */
    readonly cssWidthPx: number;
    /** 绘制坐标系高（= 物理高 / dpr），供 `u_resolution`；与 cssWidthPx 同为回算值。 */
    readonly cssHeightPx: number;
    /** 生效的 DPR（非法输入已回退为 1）。 */
    readonly dpr: number;
}

/**
 * 进程级的 DPR 上限。
 *
 * 【为什么是模块级而不是每个上下文的参数】全应用有 6 个 WebGL2 上下文（时间轴主画布、
 * 标尺、叠加层、钢琴卷帘的三个）。逐个改调用点意味着"新增一个上下文就漏一次"，
 * 而漏掉的那一个在移动端就是 36 MB —— 症状是"切到某个面板后应用被杀"，且与那个
 * 面板看起来毫无关系。
 *
 * 一个进程级上限，所有上下文自动生效。设置它的地方只有一处（`mount.tsx`，按后端
 * 剖面），因此不存在两个上下文用了不同上限的可能。
 */
let globalMaxDpr = Number.POSITIVE_INFINITY;

/**
 * 设置进程级 DPR 上限。
 *
 * @param limit 上限；`Infinity` 表示不限制（桌面默认）。
 */
export function setGlRasterMaxDpr(limit: number): void {
    globalMaxDpr = Number.isFinite(limit) && limit > 0 ? limit : Number.POSITIVE_INFINITY;
}

/** 当前生效的进程级 DPR 上限（诊断 / 测试用）。 */
export function glRasterMaxDpr(): number {
    return globalMaxDpr;
}

/**
 * 计算光栅化目标。
 *
 * 流程：归一化 DPR（非法 → 1）与 CSS 尺寸（非法 → 1，且不小于 1）→ 物理尺寸取整
 * → 绘制坐标系尺寸由物理尺寸回算。
 *
 * 特殊说明：物理尺寸恒 >= 1——`canvas.width = 0` 会让 WebGL 上下文报错并丢弃后续
 * 绘制调用，宿主尚未量出尺寸时也不能让画布进入非法状态。
 *
 * 【为什么要有 `maxDpr`】移动端旗舰机的 DPR 是 3。390×844 的画布在 DPR 3 下是
 * 1170×2532 ≈ 9.0 Mpx，单个 RGBA 帧缓冲 36 MB —— 而时间轴与钢琴卷帘**各自**持有
 * 多个上下文（实测全应用 6 个）。iOS 给单进程的内存上限在 1~1.5 GB，几张这样的
 * 帧缓冲加上 ORT 会话就会触发 jetsam（无异常、无日志、进程直接消失）。
 *
 * 视觉上，DPR 从 3 降到 2 的差别在 6 英寸屏上几乎不可见（波形与曲线都是低频内容），
 * 而像素量直接降到 44%。这是一笔非常划算的交换。
 *
 * @param cssWidthPx 宿主 CSS 宽（px）。
 * @param cssHeightPx 宿主 CSS 高（px）。
 * @param dpr 设备像素比。
 * @param maxDpr DPR 上限；`Infinity`（默认）= 不限制，桌面行为逐字节不变。
 * @returns 光栅化目标；`cssWidthPx` / `cssHeightPx` 为**回算值**（物理 / dpr）。
 */
export function resolveGlRasterTarget(
    cssWidthPx: number,
    cssHeightPx: number,
    dpr: number,
    maxDpr: number = Number.POSITIVE_INFINITY,
): GlRasterTarget {
    const rawDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    // 上限本身也要归一化：调用方可能从设置里读到一个坏值（0 / NaN / 负数）。
    // 那种情况下退回"不限制"而不是把画布压成 1×1 —— 画面模糊比画面消失好。
    const explicitLimit = Number.isFinite(maxDpr) && maxDpr > 0 ? maxDpr : Number.POSITIVE_INFINITY;
    const safeDpr = Math.min(rawDpr, explicitLimit, globalMaxDpr);
    const cssWidth = Number.isFinite(cssWidthPx) ? Math.max(1, cssWidthPx) : 1;
    const cssHeight = Number.isFinite(cssHeightPx) ? Math.max(1, cssHeightPx) : 1;
    const physicalWidthPx = Math.max(1, Math.round(cssWidth * safeDpr));
    const physicalHeightPx = Math.max(1, Math.round(cssHeight * safeDpr));
    return {
        physicalWidthPx,
        physicalHeightPx,
        cssWidthPx: physicalWidthPx / safeDpr,
        cssHeightPx: physicalHeightPx / safeDpr,
        dpr: safeDpr,
    };
}
