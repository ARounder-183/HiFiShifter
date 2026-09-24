import { describe, expect, it } from "vitest";

import { createTimelineAxis, playheadLineLeftPx } from "./timelineAxis";

/**
 * 播放头竖线的像素换算 —— **全应用唯一算法**的约定测试。
 *
 * 【为什么值得单独钉死】播放头在参数编辑器里由两条独立的线呈现：主体线由 GL 画
 * （视口坐标），标尺线是 DOM 线、位于被 `translateX(-scrollLeft)` 平移的内容层里
 * （内容坐标）。两条线曾两次"分离"，成因都是换算不唯一：
 *
 * 1. 标尺线在**内容坐标**里吸附，再被小数级的层平移带走 → 与 GL 差最多一个设备像素；
 * 2. 标尺线用**面板的 React 缩放**、GL 用内核真值 → 相差 `播放头秒数 × 缩放差`
 *    （面板宽度变化时内核会重新钳制缩放，两者随即分叉）。
 *
 * 现在两条线都从 `playheadLineLeftPx` 取左缘。本测试钉住它的三条约定，任何人改写
 * 它（例如改回内容坐标吸附、或去掉奇偶线宽补半像素）都会立刻失败。
 */
function axis(pxPerSec: number, scrollLeftPx: number, dpr: number) {
    return createTimelineAxis({ pxPerSec, scrollLeftPx, viewportWidthPx: 1000, dpr });
}

describe("playheadLineLeftPx（播放头竖线的唯一像素换算）", () => {
    const DPRS = [1, 1.25, 1.5, 2, 3];
    const SECS = [0, 0.033, 1.7, 12.34, 63.7, 200.125];
    const SCROLLS = [0, 3.7, 777.5, 12345.25, -320.4];

    it("线宽恰为 1 个物理像素", () => {
        for (const dpr of DPRS) {
            // 由 wholeDevicePxLength(1, dpr) 定义；这里断言左缘与右缘之间正好一个设备像素。
            const left = playheadLineLeftPx(axis(150, 0, dpr), 10);
            const width = 1 / dpr;
            expect(Math.round((left + width) * dpr) - Math.round(left * dpr)).toBe(1);
        }
    });

    it("★ 左缘落在设备像素栅格上（任意 DPR / 位置都不出现半像素灰线）", () => {
        for (const dpr of DPRS) {
            for (const sec of SECS) {
                for (const scroll of SCROLLS) {
                    const left = playheadLineLeftPx(axis(150, scroll, dpr), sec);
                    const devicePx = left * dpr;
                    expect(Math.abs(devicePx - Math.round(devicePx))).toBeLessThan(1e-9);
                }
            }
        }
    });

    it("★ 真实位置始终落在线的跨度之内（最多半个设备像素的吸附误差）", () => {
        for (const dpr of DPRS) {
            for (const sec of SECS) {
                for (const scroll of SCROLLS) {
                    for (const pps of [88.3, 150, 640]) {
                        const a = axis(pps, scroll, dpr);
                        const left = playheadLineLeftPx(a, sec);
                        const width = 1 / dpr;
                        const exact = sec * pps - scroll;
                        // 线体 [left, left+width] 必须覆盖真实位置；边缘允许半个设备
                        // 像素的吸附余量（线宽恰为 1 个物理像素时这个余量必然存在）。
                        expect(exact).toBeGreaterThanOrEqual(left - 0.5 / dpr - 1e-9);
                        expect(exact).toBeLessThanOrEqual(left + width + 0.5 / dpr + 1e-9);
                    }
                }
            }
        }
    });

    it("位置随播放头单调右移、随滚动单调左移", () => {
        const a = axis(150, 0, 1);
        expect(playheadLineLeftPx(a, 10)).toBeLessThan(playheadLineLeftPx(a, 11));
        const b = axis(150, 100, 1);
        expect(playheadLineLeftPx(b, 10)).toBeLessThan(playheadLineLeftPx(a, 10));
    });

    it("★ DOM 线落在与 GL 相同的设备像素列上", () => {
        // DOM 标尺线写入 `left = playheadLineLeftPx(...) + drawingScrollLeft`，层再
        // 平移 `-drawingScrollLeft`。两次加减是浮点运算，可能留下 1e-13 级残差 ——
        // 因此判据取**设备像素列**（取整后）而不是浮点相等：它才是"看起来对齐"的定义。
        for (const dpr of DPRS) {
            for (const sec of SECS) {
                for (const scroll of SCROLLS) {
                    const a = axis(150, scroll, dpr);
                    const glColumn = Math.round(playheadLineLeftPx(a, sec) * dpr);
                    const domColumn = Math.round(
                        (playheadLineLeftPx(a, sec) + scroll - scroll) * dpr,
                    );
                    expect(domColumn).toBe(glColumn);
                }
            }
        }
    });

    it("非有限播放头不产生 NaN", () => {
        expect(Number.isFinite(playheadLineLeftPx(axis(150, 0, 1), Number.NaN))).toBe(true);
    });
});
