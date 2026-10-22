/**
 * 录完之后把视口移到刚录好的区间。
 *
 * 【为什么必须有这一步】Capture 的片段落在**宿主的绝对工程时间**上：用户从 1:23 开始
 * 录，片段就在 1:23，而视口停在 0:00 —— 用户看到的是"时间轴是空的"，于是以为录制失败。
 * 后端把录制区间报出来（`status.span` → store 的 `recordedSpanSec`），这里负责把视口
 * 移过去。这一条单独就能消掉"时间轴没有出现我录的内容"这个观感。
 *
 * 【为什么只缩小不放大】缩放是用户的视图选择。录制区间装不下时才缩小到装得下；装得下
 * 就保持当前缩放，只滚动（并把区间居中）。
 *
 * 【为什么要吃掉初始票据】控件先于时间轴挂载，store 里可能已经有一次"录制完成"的票据。
 * 不吃掉它，打开工程就会莫名其妙地跳一次视口。
 */
import { useEffect, useRef } from "react";
import type React from "react";

import type { TimelineViewportAccess } from "./timelineViewportAccess";
import { getCaptureSnapshot, subscribeCapture } from "../../../../features/capture/captureStore";

/** 视口两侧各留的边距（视口宽度的比例）。 */
const MARGIN_RATIO = 0.06;
/** 区间下限（秒）：极短的录制也要给出一段可辨认的视野。 */
const MIN_SPAN_SEC = 0.5;

export function useCaptureReveal(
    viewportRef: React.MutableRefObject<TimelineViewportAccess | null>,
): void {
    const seen = useRef(getCaptureSnapshot().recordedToken);
    useEffect(() => {
        return subscribeCapture(() => {
            const snapshot = getCaptureSnapshot();
            if (snapshot.recordedToken === seen.current) return;
            seen.current = snapshot.recordedToken;
            const span = snapshot.recordedSpanSec;
            if (!span) return;
            revealSpan(viewportRef.current, span);
        });
    }, [viewportRef]);
}

/**
 * 把工程秒区间带进视野。
 *
 * 【为什么导出】它是这条通路唯一有几何计算的地方（缩放与居中的取舍），必须能被单测。
 */
export function revealSpan(viewport: TimelineViewportAccess | null, span: [number, number]): void {
    // 【为什么只在内核模式下做】旧模式的缩放要先提交 React state 再写 scrollLeft
    // （见 `TimelineViewportAccess.setZoomAndScroll` 的说明），一次原子写入在那里不成立。
    if (!viewport || !viewport.isKernel()) return;
    const pxPerSec = viewport.getPxPerSec();
    const viewportWidth = viewport.getViewportWidth();
    if (!(pxPerSec > 0) || !(viewportWidth > 0)) return;
    const start = Math.max(0, Math.min(span[0], span[1]));
    const end = Math.max(span[0], span[1]);
    const length = Math.max(end - start, MIN_SPAN_SEC);
    const margin = viewportWidth * MARGIN_RATIO;
    const fit = (viewportWidth - margin * 2) / length;
    // 只缩小不放大：`fit > pxPerSec` 时保持用户当前的缩放。
    const target = Math.min(pxPerSec, fit);
    const center = ((start + end) / 2) * target;
    viewport.setZoomAndScroll(target, center - viewportWidth / 2);
}
