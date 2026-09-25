/**
 * 拖拽锚点换算：把「视口 + 指针屏幕位置」换算成**时间**，并保持抓取偏移恒定。
 *
 * 【要解决的问题（一）】拖拽的落点 = 对象基准 + 指针位移。位移曾经这样算：
 * `(当前内容像素 − 按下时内容像素) / 当前缩放`。问题在于**内容像素 = 秒 × 缩放**，
 * 缩放一变，同一个屏幕位置的内容像素就变了 —— 于是滚轮缩放期间"按下时内容像素"是
 * 旧尺子上的读数，除以新缩放得到的秒数整体偏移，量级约 `指针秒数 × (1 − p₀/p₁)`
 * （放大到 2 倍、指针在 60 秒处 ≈ 偏 30 秒）。用户报告为"拖拽过程中滚动/缩放导致
 * 严重偏移"。
 *
 * 【解法（一）】起点与当前点都换算成**时间**再作差（`dragDeltaSec`）：时间不随缩放
 * 变化，滚动也被两边共同计入而抵消。
 *
 * 【要解决的问题（二）】但"抓取偏移"也随之被固化成了**时间**。用户抓住的若是对象的
 * **非零偏移处**（抓住 clip 中段；抓住 Tempo 变化点旗帜 —— 旗帜左缘对齐变化点、
 * 标签在其右侧几十像素），那段时间在当前缩放下代表的**像素**会随缩放伸缩：放大
 * 2 倍，40px 的抓取偏移就变成 80px，对象与光标越拖越远（用户报告："水平缩放时这个
 * 偏右的距离会越来越偏"）。
 *
 * 【解法（二）】`anchoredDeltaSec`：抓取偏移按**屏幕像素**恒定 —— 按下时记下
 * "指针 − 锚点"的像素距离，之后每次都用**当前**缩放把它折回时间。于是"抓在哪儿就
 * 一直在哪儿"在**像素**意义上成立（与拖动窗口标题栏一致，这才是用户对拖拽的预期）。
 *
 * 纯函数、无 React 依赖 —— 见同名测试（遍历多种缩放/滚动组合钉住"不偏移"契约）。
 */

/**
 * 指针所在的**内容时间**（秒）。
 *
 * @param args.scrollLeftPx 当前水平滚动量（**绘制坐标**，含小数）。
 * @param args.pxPerSec 当前水平缩放（每秒像素）。
 * @param args.clientX 指针屏幕 x（CSS px）。
 * @param args.rectLeft 视口内容区左缘的屏幕 x（CSS px）。
 */
export function pointerSecAt(args: {
    scrollLeftPx: number;
    pxPerSec: number;
    clientX: number;
    rectLeft: number;
}): number {
    const pxPerSec = args.pxPerSec > 0 && Number.isFinite(args.pxPerSec) ? args.pxPerSec : 1;
    return (args.scrollLeftPx + (args.clientX - args.rectLeft)) / pxPerSec;
}

/**
 * 指针相对手势起点的位移（**秒**）。
 *
 * @param startPointerSec 按下时指针所在的时间（`pointerSecAt` 的按下时读数）。
 * @param pointerSec 当前指针所在的时间。
 */
export function dragDeltaSec(startPointerSec: number, pointerSec: number): number {
    if (!Number.isFinite(startPointerSec) || !Number.isFinite(pointerSec)) return 0;
    return pointerSec - startPointerSec;
}

/**
 * 位移的"内容像素"表达（按**当前**缩放折算）。
 *
 * 下游的几何纯函数（`resolveDragDelta` / `resolveTrimEdge` / `resolveFadeDrag`）按
 * `deltaContentXPx / pxPerSec` 还原秒数，因此这里乘回当前缩放即可保持它们的语义不变。
 */
export function deltaSecToContentPx(deltaSec: number, pxPerSec: number): number {
    const safe = pxPerSec > 0 && Number.isFinite(pxPerSec) ? pxPerSec : 1;
    return deltaSec * safe;
}

/** 非负有限数（缩放的安全除数）。 */
function safeScale(pxPerSec: number): number {
    return pxPerSec > 0 && Number.isFinite(pxPerSec) ? pxPerSec : 1;
}

/**
 * 手势位移（**秒**）：时间差 + 抓取偏移修正（锚点跟随光标，偏移按**屏幕像素**恒定）。
 *
 * 语义等价于"锚点的新位置 = 指针时间 − 抓取偏移（按当前缩放折算）"，只是以
 * **位移**的形式给出，好让调用方沿用 `对象起始值 + delta` 的既有写法。
 * 缩放在拖拽中不变时，结果与 `dragDeltaSec` **逐位相同**（修正项恒为 0）。
 *
 * @param args.anchorSec 被拖对象在该轴上的**锚点**（跟随光标的那个基准位置，秒）。
 * @param args.startPointerSec 按下时指针所在的时间。
 * @param args.startPxPerSec 按下时的缩放（把抓取偏移折算成像素用）。
 * @param args.pointerSec 当前指针所在的时间。
 * @param args.pxPerSec 当前缩放。
 * @returns 应当施加到锚点上的位移（秒）。
 */
export function anchoredDeltaSec(args: {
    anchorSec: number;
    startPointerSec: number;
    startPxPerSec: number;
    pointerSec: number;
    pxPerSec: number;
}): number {
    const { anchorSec, startPointerSec, pointerSec } = args;
    if (
        !Number.isFinite(anchorSec) ||
        !Number.isFinite(startPointerSec) ||
        !Number.isFinite(pointerSec)
    ) {
        return dragDeltaSec(startPointerSec, pointerSec);
    }
    // 抓取偏移（屏幕像素）：按下时"指针 − 锚点"的距离。与缩放无关，是本次手势的常量。
    const grabOffsetPx = (startPointerSec - anchorSec) * safeScale(args.startPxPerSec);
    return pointerSec - grabOffsetPx / safeScale(args.pxPerSec) - anchorSec;
}
