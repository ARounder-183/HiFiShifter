/**
 * 淡变曲线自适应细分采样（`sampleFadeCurveSegments`）的回归测试。
 *
 * 【要锁住什么】采样器从"每轮线性扫描最坏段 + splice"改成"二叉最大堆"后，
 * 取段的**平局次序**必须与朴素实现逐点一致——否则上限 1200 点时先拆哪一段会变，
 * 产出的折线点集随之变化（视觉上曲线形状/接角可能改变）。这里用一份**独立的
 * 朴素参考实现**逐点比对，覆盖直线 / 指数 / S 族与两种方向。
 *
 * 【参考实现为什么照抄偏差公式】它复刻的就是被替换掉的那段算法（同样的
 * `fadeGainSigned` 与中点+四分点偏差度量），这样比较的是"选段策略"而非"曲线数学"。
 */

import { describe, expect, it } from "vitest";

import { fadeGainSigned } from "../reaperFade";
import { sampleFadeCurveSegments, type FadeCurveSampleSegment } from "./timelineCanvasRenderer";

interface SamplerArgs {
    leftPx: number;
    topPx: number;
    widthPx: number;
    heightPx: number;
    shape: number;
    dir: number;
    mode: "in" | "out";
}

const MAX_POINTS = 1200;
const TOLERANCE_PX = 0.6;

/** 朴素参考实现：每轮线性扫描偏差最大的段（原算法，未做堆优化）。 */
function naiveSample(args: SamplerArgs): FadeCurveSampleSegment[] {
    const gainAt = (t: number): number => fadeGainSigned(args.shape, args.dir, args.mode, t);
    const xAt = (t: number): number => args.leftPx + t * args.widthPx;
    const yAt = (t: number): number => args.topPx + args.heightPx * (1 - gainAt(t));

    interface Seg extends FadeCurveSampleSegment {
        dev: number;
        tm: number;
        xm: number;
        ym: number;
    }

    const evaluate = (t0: number, t1: number, x0: number, y0: number, x1: number, y1: number) => {
        const tm = (t0 + t1) / 2;
        const tq0 = t0 + (t1 - t0) * 0.25;
        const tq1 = t0 + (t1 - t0) * 0.75;
        const xm = xAt(tm);
        const ym = yAt(tm);
        const xq0 = xAt(tq0);
        const yq0 = yAt(tq0);
        const xq1 = xAt(tq1);
        const yq1 = yAt(tq1);
        const devMid = Math.hypot(xm - (x0 + x1) / 2, ym - (y0 + y1) / 2);
        const devQ0 = Math.hypot(xq0 - (x0 + (x1 - x0) * 0.25), yq0 - (y0 + (y1 - y0) * 0.25));
        const devQ1 = Math.hypot(xq1 - (x0 + (x1 - x0) * 0.75), yq1 - (y0 + (y1 - y0) * 0.75));
        return { tm, xm, ym, dev: Math.max(devMid, devQ0, devQ1) };
    };

    const segments: Seg[] = [];
    const push = (t0: number, t1: number, x0: number, y0: number, x1: number, y1: number): void => {
        const { tm, xm, ym, dev } = evaluate(t0, t1, x0, y0, x1, y1);
        segments.push({ t0, t1, x0, y0, x1, y1, dev, tm, xm, ym });
    };

    push(0, 1, args.leftPx, yAt(0), args.leftPx + args.widthPx, yAt(1));
    while (segments.length < MAX_POINTS) {
        let worstIndex = -1;
        let worstDev = TOLERANCE_PX;
        for (let i = 0; i < segments.length; i += 1) {
            if (segments[i].dev > worstDev) {
                worstDev = segments[i].dev;
                worstIndex = i;
            }
        }
        if (worstIndex < 0) break;
        const seg = segments[worstIndex];
        segments.splice(worstIndex, 1);
        push(seg.t0, seg.tm, seg.x0, seg.y0, seg.xm, seg.ym);
        push(seg.tm, seg.t1, seg.xm, seg.ym, seg.x1, seg.y1);
    }
    segments.sort((a, b) => a.t0 - b.t0);
    return segments;
}

/** 只保留公开字段，便于跨实现比较（堆版内部还带 dev/tm/xm/ym/seq）。 */
function project(segments: readonly FadeCurveSampleSegment[]): FadeCurveSampleSegment[] {
    return segments.map(({ t0, t1, x0, y0, x1, y1 }) => ({ t0, t1, x0, y0, x1, y1 }));
}

const SHAPES = [0, 1, 2, 3, 4, 5, 6, 1.1];
const DIRS = [-1, -0.5, -0.33, 0, 0.33, 0.5, 1];
const MODES: Array<"in" | "out"> = ["in", "out"];

describe("sampleFadeCurveSegments — 与朴素扫描逐点一致", () => {
    it("常规尺寸下覆盖全部形状 / 方向 / 模式", () => {
        for (const shape of SHAPES) {
            for (const dir of DIRS) {
                for (const mode of MODES) {
                    const args: SamplerArgs = {
                        leftPx: 37.5,
                        topPx: 12.25,
                        widthPx: 240,
                        heightPx: 80,
                        shape,
                        dir,
                        mode,
                    };
                    expect(project(sampleFadeCurveSegments(args))).toEqual(
                        project(naiveSample(args)),
                    );
                }
            }
        }
    });

    it("极端缩放（触及 MAX_POINTS 上限）下仍逐点一致", () => {
        // 巨大像素尺度 + 陡峭曲线：偏差远大于容差，必然顶到 1200 点上限。
        const args: SamplerArgs = {
            leftPx: 0,
            topPx: 0,
            widthPx: 10_000_000,
            heightPx: 10_000_000,
            shape: 4,
            dir: 1,
            mode: "in",
        };
        const fast = sampleFadeCurveSegments(args);
        const slow = naiveSample(args);
        expect(fast.length).toBe(MAX_POINTS);
        expect(project(fast)).toEqual(project(slow));
    });
});

describe("sampleFadeCurveSegments — 不变量", () => {
    it("端点精确落在左下 / 右上（或反向）边角，且按 t0 升序", () => {
        for (const mode of MODES) {
            const args: SamplerArgs = {
                leftPx: 10,
                topPx: 20,
                widthPx: 300,
                heightPx: 90,
                shape: 6,
                dir: 0.5,
                mode,
            };
            const segments = sampleFadeCurveSegments(args);
            expect(segments.length).toBeGreaterThan(1);
            expect(segments.length).toBeLessThanOrEqual(MAX_POINTS);
            expect(segments[0].x0).toBe(10);
            expect(segments[segments.length - 1].x1).toBe(310);
            for (let i = 1; i < segments.length; i += 1) {
                expect(segments[i].t0).toBeGreaterThanOrEqual(segments[i - 1].t0);
            }
        }
    });
});
