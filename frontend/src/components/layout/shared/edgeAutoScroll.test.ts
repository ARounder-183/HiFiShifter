/**
 * 拖拽边缘自动滚屏几何（../edgeAutoScroll）行为自检。
 *
 * 【本测试守护什么】
 * 1. 带宽内线性加速、带外为 0；方向符号左负右正（与 `scrollLeft` 增大方向一致）；
 * 2. **比例上限 1.5**：指针拖出视口后不再继续加速；
 * 3. **帧率无关**：步长与本帧时长成正比 —— 60Hz / 120Hz / 30Hz 下同样的墙钟时间
 *    必须滚过同样的距离。这条同时锁住"事件频率无关"：高轮询率鼠标不再把
 *    "每事件 N 像素"放大成每秒几千像素（旧实现"每帧 18px"正是按事件计步的）；
 * 4. 非法 / 退化输入返回 0 或回退 1/60 秒，绝不返回 NaN 污染 `scrollLeft`；
 * 5. `edgeScrollMaxLeftPx` 不为负、非有限输入返回 0。
 *
 * 【与其他模块的关系】覆盖 `shared/edgeAutoScroll.ts`。时间轴内核
 *（`timeline/kernel/input/dragAutoScroll.ts`）与参数编辑器内核
 *（`pianoRoll/kernel/dragArithmetic.ts` 的调用方）共用这一份几何，因此这里的
 * 用例是**两个面板共同**的回归锁。
 */

import { describe, expect, it } from "vitest";

import {
    EDGE_SCROLL_BAND_PX,
    EDGE_SCROLL_MAX_RATIO,
    edgeScrollMaxLeftPx,
    resolveEdgeScrollDeltaPx,
} from "./edgeAutoScroll";

/** 视口：left=100、right=1100（宽 1000）。 */
const VIEW = { leftPx: 100, rightPx: 1100 };
const FRAME_60HZ = 1000 / 60;

/** 用例统一用时间轴的速度；速度本身是入参，见模块头说明。 */
const SPEED = 720;

/** 以 60Hz 帧时长调用，返回本帧步长。 */
function step(clientX: number, frameMs = FRAME_60HZ): number {
    return resolveEdgeScrollDeltaPx({ clientX, ...VIEW, frameMs, maxSpeedPxPerSec: SPEED });
}

describe("resolveEdgeScrollDeltaPx", () => {
    it("视口中央不滚屏", () => {
        expect(step(600)).toBe(0);
        expect(step(100 + EDGE_SCROLL_BAND_PX)).toBe(0);
        expect(step(1100 - EDGE_SCROLL_BAND_PX)).toBe(0);
    });

    it("左缘为负、右缘为正（方向与 scrollLeft 一致）", () => {
        expect(step(100 + EDGE_SCROLL_BAND_PX - 4)).toBeLessThan(0);
        expect(step(1100 - EDGE_SCROLL_BAND_PX + 4)).toBeGreaterThan(0);
    });

    it("带宽内越靠边越快（线性加速）", () => {
        const near = step(1100 - EDGE_SCROLL_BAND_PX + 4);
        const mid = step(1100 - EDGE_SCROLL_BAND_PX / 2);
        const edge = step(1100 - 1);
        expect(mid).toBeGreaterThan(near);
        expect(edge).toBeGreaterThan(mid);
    });

    it("左右对称", () => {
        expect(step(1100 - EDGE_SCROLL_BAND_PX / 2)).toBeCloseTo(
            -step(100 + EDGE_SCROLL_BAND_PX / 2),
            9,
        );
    });

    it("指针拖出视口后不再继续加速（比例上限 1.5，在带外 16px 处饱和）", () => {
        // 比例 1.5 ⇒ 距边缘 1.5 × 带宽 = 48px 处饱和（右缘 1100 时即 clientX ≥ 1148）。
        const saturated = step(1100 + EDGE_SCROLL_BAND_PX);
        expect(step(1100 + 5000)).toBeCloseTo(saturated, 6);
        // 饱和点之内的边缘（clientX = 1100，比例 1.0）仍小于饱和值。
        expect(step(1100)).toBeLessThan(saturated);
    });

    it("步长与帧时长成正比（帧率无关）", () => {
        const at60 = step(1100 - 1, 1000 / 60);
        const at120 = step(1100 - 1, 1000 / 120);
        const at30 = step(1100 - 1, 1000 / 30);
        expect(at120).toBeCloseTo(at60 / 2, 6);
        expect(at30).toBeCloseTo(at60 * 2, 6);
        // 同样的墙钟时间应滚过同样的距离。
        expect(at120 * 2).toBeCloseTo(at60, 6);
    });

    it("最大速度受入参约束（饱和后 = 1.5 倍速度 × 1 秒）", () => {
        const saturated = step(1100 + 5000, 1000);
        // 饱和比例 1.5、帧时长 1000ms（被夹到上限 100ms）⇒ 0.1 秒的量。
        expect(saturated).toBeCloseTo(SPEED * EDGE_SCROLL_MAX_RATIO * 0.1, 6);
    });

    it("速度是入参：同一位置按比例缩放，公式不绑死任何面板的手感", () => {
        const slow = resolveEdgeScrollDeltaPx({
            clientX: 1100 - 1,
            ...VIEW,
            frameMs: FRAME_60HZ,
            maxSpeedPxPerSec: 720,
        });
        const fast = resolveEdgeScrollDeltaPx({
            clientX: 1100 - 1,
            ...VIEW,
            frameMs: FRAME_60HZ,
            maxSpeedPxPerSec: 1080,
        });
        expect(fast / slow).toBeCloseTo(1080 / 720, 9);
    });

    it("速度非法 / 非正 → 0（不滚比乱滚安全）", () => {
        for (const bad of [0, -100, Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(
                resolveEdgeScrollDeltaPx({
                    clientX: 1100 - 1,
                    ...VIEW,
                    frameMs: FRAME_60HZ,
                    maxSpeedPxPerSec: bad,
                }),
            ).toBe(0);
        }
    });

    it("帧时长上限 100ms：挂起后恢复不会一次性跳过很长距离", () => {
        const longFrame = step(1100 + 5000, 5000);
        expect(longFrame).toBeCloseTo(step(1100 + 5000, 100), 6);
    });

    it("非法帧时长回退 1/60 秒，而不是停止滚屏", () => {
        const good = step(1100 - 1, FRAME_60HZ);
        expect(step(1100 - 1, Number.NaN)).toBeCloseTo(good, 6);
        expect(step(1100 - 1, 0)).toBeCloseTo(good, 6);
        expect(step(1100 - 1, -5)).toBeCloseTo(good, 6);
    });

    it("非有限指针坐标 / 退化视口返回 0", () => {
        expect(step(Number.NaN)).toBe(0);
        expect(step(Number.POSITIVE_INFINITY)).toBe(0);
        const degenerate = { ...VIEW, frameMs: FRAME_60HZ, maxSpeedPxPerSec: SPEED };
        expect(
            resolveEdgeScrollDeltaPx({ ...degenerate, clientX: 0, leftPx: 100, rightPx: 100 }),
        ).toBe(0);
        expect(
            resolveEdgeScrollDeltaPx({ ...degenerate, clientX: 0, leftPx: 200, rightPx: 100 }),
        ).toBe(0);
    });

    it("带宽常量在正常视口宽度下不会互相重叠", () => {
        // 两侧带宽之和必须远小于常见视口宽度，否则中部全是"边缘"。
        expect(EDGE_SCROLL_BAND_PX * 2).toBeLessThan(300);
    });
});

describe("edgeScrollMaxLeftPx", () => {
    /** 1000 帧 × 5ms/帧 = 5s，缩放 100px/s ⇒ 内容宽 500px。 */
    const BASE = {
        pxPerSec: 100,
        framePeriodMs: 5,
        maxFrame: 1000,
        viewportWidthPx: 300,
        nativeOffsetPx: 0,
    };

    it("★ 上界 = 内容宽（不减视口），再按『绘制 = 原生 − 偏移』投影", () => {
        // 不减视口宽：两个内核都把上界定义成内容宽（原生容器比视口宽出一整屏，
        // 视口只是窗口）。曾经这里减去视口宽，于是自动滚屏的右界比内核真值小了
        // 一整个视口 —— 指针停在右缘时每帧写上界、内核回写更大值，视图往复闪现。
        expect(edgeScrollMaxLeftPx(BASE)).toBeCloseTo(500, 9);
        // 偏移按「原生 = 绘制 + 偏移」投影：原生上界 = 内容宽 ⇒ 绘制上界 = 内容宽 − 偏移。
        // 写成 + 偏移会超出内核真值一个偏移量，触发同一类往复。
        expect(edgeScrollMaxLeftPx({ ...BASE, nativeOffsetPx: 200 })).toBeCloseTo(300, 9);
        expect(edgeScrollMaxLeftPx({ ...BASE, nativeOffsetPx: -200 })).toBeCloseTo(700, 9);
    });

    it("★ 上界与视口宽无关（视口只是窗口，不改变可滚范围）", () => {
        // 只在"视口仍窄于内容"的前提下成立；视口宽到装下全部内容时归 0（见下一条）。
        for (const vw of [100, 300, 499]) {
            expect(edgeScrollMaxLeftPx({ ...BASE, viewportWidthPx: vw })).toBeCloseTo(500, 9);
        }
    });

    it("内容比视口窄 → 不可滚（0，而不是负数）", () => {
        // 内容 500px、视口 900px：没有可滚余地。
        expect(edgeScrollMaxLeftPx({ ...BASE, viewportWidthPx: 900 })).toBe(0);
        expect(edgeScrollMaxLeftPx({ ...BASE, maxFrame: 10 })).toBe(0);
        expect(edgeScrollMaxLeftPx({ ...BASE, pxPerSec: 0 })).toBe(0);
    });

    it("负偏移不会把上界压成负数（只会把绘制域整体右移）", () => {
        // 偏移为负 ⇒ 绘制上界 = 内容宽 − (负) = 更大，绝不为负。
        expect(edgeScrollMaxLeftPx({ ...BASE, nativeOffsetPx: -5000 })).toBeGreaterThanOrEqual(0);
        expect(edgeScrollMaxLeftPx({ ...BASE, nativeOffsetPx: -200 })).toBeCloseTo(700, 9);
    });

    it("任一输入非有限 → 0（宁可停住，也不把视口写到未定义位置）", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(edgeScrollMaxLeftPx({ ...BASE, pxPerSec: bad })).toBe(0);
            expect(edgeScrollMaxLeftPx({ ...BASE, framePeriodMs: bad })).toBe(0);
            expect(edgeScrollMaxLeftPx({ ...BASE, maxFrame: bad })).toBe(0);
            expect(edgeScrollMaxLeftPx({ ...BASE, viewportWidthPx: bad })).toBe(0);
            expect(edgeScrollMaxLeftPx({ ...BASE, nativeOffsetPx: bad })).toBe(0);
        }
    });
});
