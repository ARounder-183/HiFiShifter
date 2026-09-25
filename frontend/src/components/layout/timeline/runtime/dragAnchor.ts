/**
 * 拖拽锚点换算：把「视口 + 指针屏幕位置」换算成**时间**。
 *
 * 【要解决的问题】拖拽的落点 = 对象基准 + 指针位移。位移曾经这样算：
 * `(当前内容像素 − 按下时内容像素) / 当前缩放`。问题在于**内容像素 = 秒 × 缩放**，
 * 缩放一变，同一个屏幕位置的内容像素就变了 —— 于是滚轮缩放期间"按下时内容像素"是
 * 旧尺子上的读数，除以新缩放得到的秒数整体偏移，量级约 `指针秒数 × (1 − p₀/p₁)`
 * （放大到 2 倍、指针在 60 秒处 ≈ 偏 30 秒）。用户报告为"拖拽过程中滚动/缩放导致
 * 严重偏移"。
 *
 * 【解法】起点与当前点都换算成**时间**再作差：时间不随缩放变化，滚动也被两边共同
 * 计入而抵消。落点因此始终与光标保持同一个**时间偏移**（即"抓在哪儿就一直在哪儿"），
 * 无论中途怎么滚动或缩放。
 *
 * 纯函数、无 React 依赖 —— 见同名测试（遍历多种缩放/滚动组合钉住不偏移这一契约）。
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
