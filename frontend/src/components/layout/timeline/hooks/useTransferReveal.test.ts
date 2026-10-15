/**
 * 录完之后视口定位的几何契约。
 *
 * 【为什么必须有测试】"录完时间轴上什么都没有"是用户报障的原话之一，而真正的原因往往
 * 不是没录上，而是片段落在**宿主的绝对工程时间**上（用户从 1:23 开始录，片段就在 1:23）
 * 而视口停在 0:00。这一段几何是修好它的唯一地方，缩放的取舍（只缩小不放大）必须钉住。
 */
// @vitest-environment jsdom
import { expect, test, vi } from "vitest";

import { revealSpan } from "./useTransferReveal";
import type { TimelineViewportAccess } from "./timelineViewportAccess";

function viewport(args: { pxPerSec: number; viewportWidth?: number }) {
    const setZoomAndScroll = vi.fn((pxPerSec: number, scrollLeft: number) => ({
        pxPerSec,
        scrollLeft,
    }));
    const access: TimelineViewportAccess = {
        isKernel: () => true,
        getRect: () => null,
        getScrollLeft: () => 0,
        getScrollTop: () => 0,
        getViewportWidth: () => args.viewportWidth ?? 1000,
        getPxPerSec: () => args.pxPerSec,
        getViewportHeight: () => 400,
        setScrollLeft: (px) => px,
        setScrollTop: (px) => px,
        setZoomAndScroll,
    };
    return { access, setZoomAndScroll };
}

test("区间装不下时缩小到装得下，并把区间居中", () => {
    const { access, setZoomAndScroll } = viewport({ pxPerSec: 100 });
    revealSpan(access, [10, 20]);
    // 视口 1000、边距 6% ⇒ 可用 880；区间 10 秒 ⇒ 88 px/s（比当前 100 小 ⇒ 缩小）。
    // 区间中心 15 s ⇒ 内容坐标 1320；视口中心 ⇒ scrollLeft = 1320 - 500 = 820。
    expect(setZoomAndScroll).toHaveBeenCalledWith(88, 820);
});

test("区间装得下时**只滚动、不放大**（缩放是用户的视图选择）", () => {
    const { access, setZoomAndScroll } = viewport({ pxPerSec: 10 });
    revealSpan(access, [100, 140]);
    // 40 秒区间在 10 px/s 下只需 400 px，远小于 880 ⇒ 保持 10。
    // 中心 120 s ⇒ 1200；scrollLeft = 1200 - 500 = 700。
    expect(setZoomAndScroll).toHaveBeenCalledWith(10, 700);
});

test("极短的录制也给出一段可辨认的视野（下限 0.5 秒）", () => {
    const { access, setZoomAndScroll } = viewport({ pxPerSec: 100 });
    revealSpan(access, [5, 5]);
    // 区间按 0.5 s 计 ⇒ fit = 880/0.5 = 1760（> 100 ⇒ 保持 100）。
    // 中心 5 s ⇒ 500；scrollLeft = 0。
    expect(setZoomAndScroll).toHaveBeenCalledWith(100, 0);
});

test("区间反向或起点为负都被规整，不产生负滚动", () => {
    const { access, setZoomAndScroll } = viewport({ pxPerSec: 100 });
    revealSpan(access, [20, 10]);
    const [pxPerSec, scrollLeft] = setZoomAndScroll.mock.calls[0];
    expect(pxPerSec).toBeGreaterThan(0);
    expect(scrollLeft).toBeGreaterThanOrEqual(0);
});

test("非内核模式（旧滚动容器）不做原子写入 —— 那里它不成立", () => {
    const { access, setZoomAndScroll } = viewport({ pxPerSec: 100 });
    revealSpan({ ...access, isKernel: () => false }, [10, 20]);
    expect(setZoomAndScroll).not.toHaveBeenCalled();
});

test("视口不可用时不炸（挂载前 / 卸载后）", () => {
    expect(() => revealSpan(null, [0, 1])).not.toThrow();
});
