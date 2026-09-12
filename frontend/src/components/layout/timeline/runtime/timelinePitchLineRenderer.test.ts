import { describe, expect, it } from "vitest";

import type { ClipInfo } from "../../../../features/session/sessionTypes";
import { createTimelineAxis, type TimelineAxis } from "./timelineAxis.js";
import { drawTrackPitchLines } from "./timelinePitchLineRenderer";

/**
 * 记录型假 2D 上下文：只实现 drawTrackPitchLines 用到的命令，
 * 并把路径点按当前 translate 偏移记录下来供断言（内容绝对坐标系）。
 */
function makeFakeCtx() {
    const points: Array<{ x: number; y: number }> = [];
    const rects: Array<{ x: number; y: number; w: number; h: number }> = [];
    let offsetX = 0;
    let offsetY = 0;
    const offsetStack: Array<[number, number]> = [];
    const ctx = {
        save: () => offsetStack.push([offsetX, offsetY]),
        restore: () => {
            const top = offsetStack.pop();
            if (top) {
                offsetX = top[0];
                offsetY = top[1];
            }
        },
        beginPath: () => {},
        clip: () => {},
        translate: (dx: number, dy: number) => {
            offsetX += dx;
            offsetY += dy;
        },
        moveTo: (x: number, y: number) => points.push({ x: x + offsetX, y: y + offsetY }),
        lineTo: (x: number, y: number) => points.push({ x: x + offsetX, y: y + offsetY }),
        rect: (x: number, y: number, w: number, h: number) =>
            rects.push({ x: x + offsetX, y: y + offsetY, w, h }),
        stroke: () => {},
        closePath: () => {},
        fill: () => {},
        strokeStyle: "",
        fillStyle: "",
        lineWidth: 0,
        lineJoin: "",
        lineCap: "",
        globalAlpha: 1,
    };
    return { ctx: ctx as unknown as CanvasRenderingContext2D, points, rects };
}

function makePitchClip(overrides: Partial<ClipInfo>): ClipInfo {
    return {
        id: "clip-1",
        trackId: "track-1",
        name: "pitch",
        startSec: 2,
        lengthSec: 1,
        gain: 1,
        playbackRate: 1,
        muted: false,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 0,
        fadeOutShape: 0,
        fadeInDir: 0,
        fadeOutDir: 0,
        color: "cyan",
        sourceStartSec: 0,
        sourceEndSec: 1,
        reversed: false,
        loopEnabled: false,
        midiNoteCount: 1,
        midiNoteData: [{ startSec: 0, endSec: 1, note: 60, velocity: 100, channel: 0 }],
        ...overrides,
    } as ClipInfo;
}

const CURVES = {};
const RANGES = {};

describe("drawTrackPitchLines", () => {
    it("按内容绝对坐标绘制折线：x = 秒 × pxPerSec，与 scrollLeft 无关", () => {
        const axis: TimelineAxis = createTimelineAxis({
            pxPerSec: 100,
            scrollLeftPx: 0,
            viewportWidthPx: 500,
        });
        const { ctx, points } = makeFakeCtx();
        drawTrackPitchLines({
            ctx,
            axis,
            clips: [makePitchClip({})],
            rowTopPx: 0,
            rowHeight: 80,
            clipPitchCurves: CURVES,
            clipPitchRanges: RANGES,
        });
        // 波形区：top = 18（CLIP_HEADER_HEIGHT），高 = 80 − 2 − 18 = 60。
        expect(points.length).toBeGreaterThan(10);
        expect(points[0].x).toBeCloseTo(200, 6); // clip 起点在内容 x = 2s × 100
        const last = points[points.length - 1];
        // 最后一帧是 5ms 窗口起点：t = 2 + 199×0.005 = 2.995s。
        expect(last.x).toBeCloseTo(299.5, 6);
        // y 全部落在行波形区（含 10% padding 收缩）内。
        for (const p of points) {
            expect(p.y).toBeGreaterThanOrEqual(18 + 6 - 1e-6);
            expect(p.y).toBeLessThanOrEqual(18 + 60 - 6 + 1e-6);
        }
    });

    it("clip 与视口部分相交时只绘制可见段（内容坐标不随滚动平移）", () => {
        // 视口 [250, 750] px：clip 内容区 [200, 300]，可见段为 [250, 300]。
        const axis: TimelineAxis = createTimelineAxis({
            pxPerSec: 100,
            scrollLeftPx: 250,
            viewportWidthPx: 500,
        });
        const { ctx, points } = makeFakeCtx();
        drawTrackPitchLines({
            ctx,
            axis,
            clips: [makePitchClip({})],
            rowTopPx: 0,
            rowHeight: 80,
            clipPitchCurves: CURVES,
            clipPitchRanges: RANGES,
        });
        expect(points[0].x).toBeCloseTo(250, 6);
        for (const p of points) {
            expect(p.x).toBeGreaterThanOrEqual(250 - 1e-6);
            expect(p.x).toBeLessThanOrEqual(300 + 1e-6);
        }
    });

    it("完全在视口外的 clip 不产生任何路径命令", () => {
        const axis: TimelineAxis = createTimelineAxis({
            pxPerSec: 100,
            scrollLeftPx: 25000,
            viewportWidthPx: 500,
        });
        const { ctx, points, rects } = makeFakeCtx();
        drawTrackPitchLines({
            ctx,
            axis,
            clips: [makePitchClip({})],
            rowTopPx: 0,
            rowHeight: 80,
            clipPitchCurves: CURVES,
            clipPitchRanges: RANGES,
        });
        expect(points).toHaveLength(0);
        expect(rects).toHaveLength(0);
    });

    it("Loop（循环源）clip 按内容时长 D 在回绕点绘制倒三角标记", () => {
        // 纯音高参考块：内容时长 = 音符最大结束时间 1s ⇒ 周期 1s；
        // lengthSec=3 ⇒ 回绕点在局部 t=1s、2s（内容 x=300、400）。
        const axis: TimelineAxis = createTimelineAxis({
            pxPerSec: 100,
            scrollLeftPx: 0,
            viewportWidthPx: 500,
        });
        const { ctx, points } = makeFakeCtx();
        drawTrackPitchLines({
            ctx,
            axis,
            clips: [makePitchClip({ lengthSec: 3, loopEnabled: true })],
            rowTopPx: 0,
            rowHeight: 80,
            clipPitchCurves: CURVES,
            clipPitchRanges: RANGES,
        });
        // 倒三角顶边贴波形区顶部：内容 y = areaTop(18) + 0.5。
        // size = min(7, max(4.5, 60×0.16)) = 7 ⇒ 半宽 = 7×0.62 = 4.34。
        const markerTops = points.filter((p) => Math.abs(p.y - 18.5) < 1e-6);
        const markerXs = markerTops.map((p) => p.x).sort((a, b) => a - b);
        // 两个回绕点（内容 x=300、400）各贡献左右两个顶点。
        expect(markerXs.map((x) => Math.round(x * 100) / 100)).toEqual([
            295.66, 304.34, 395.66, 404.34,
        ]);
    });
});
