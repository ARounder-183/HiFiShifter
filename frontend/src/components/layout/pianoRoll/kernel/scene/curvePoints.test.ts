/**
 * 曲线采样点投影（./curvePoints）行为自检。
 *
 * 【本测试守护什么】
 * 1. **可见性语义**：只返回视口内的点（`tSec < 起点` 跳过、`tSec > 终点` 立即结束），
 *    这是虚线相位正确的前提——Canvas2D 的相位从子路径起点算，而子路径起点是
 *    首个可见点；
 * 2. **pitch 的 +0.5 偏移**：MIDI 值 N 画在 N 键中心；其他参数不加偏移；
 * 3. **帧 → 秒 → 像素**的换算与 `framesToTime` / `secToViewportPx` 完全一致
 *    （不做第二套换算）；
 * 4. 非有限值被跳过（NaN 会让整条折线的顶点缓冲失效）。
 *
 * 【末尾的逐值等价测试为什么必要】前面几组只证明"实现符合我对 drawCurveTimed 的
 * 理解"。末尾原地复刻 `render.ts:162-194` 的循环并逐点比对——这是阶段 2 抓到
 * 12 个真实缺陷的同一手法（口径差异只有逐值比对才能发现）。
 */
import { describe, expect, it } from "vitest";

import {
    createTimelineAxis,
    secToViewportPx,
    viewportEndSec,
    viewportStartSec,
} from "../../../renderKernel/timelineAxis";
import { framesToTime } from "../../utils";
import {
    projectClipboardPreviewPoints,
    projectCurvePoints,
    projectDetectedCurvePoints,
} from "./curvePoints";

/** 造一个轴（与面板一致：dpr 只影响描边对齐，不影响投影）。 */
function makeAxis(pxPerSec = 150, scrollLeftPx = 0, viewportWidthPx = 1000) {
    return createTimelineAxis({ pxPerSec, scrollLeftPx, viewportWidthPx, dpr: 1 });
}

/** 线性值投影桩：值 0 在底部、100 在顶部。 */
const valueToY = (h: number) => (v: number) => h * (1 - v / 100);

describe("projectCurvePoints", () => {
    it("只返回视口内的点（左跳过、右立即结束）", () => {
        // 5ms 步长、startFrame 0：每点 0.005s
        // 视口 [0, 1000px] / 150pxPerSec = [0, 6.667s] -> 可见 1334 点
        const values = Array.from({ length: 2000 }, () => 50);
        const points = projectCurvePoints({
            values,
            param: "pitch",
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            axis: makeAxis(),
            valueToY: valueToY(400),
        });
        expect(points.length).toBeGreaterThan(0);
        // 所有点都落在视口 x 范围内（允许边缘的亚像素余量）
        for (const p of points) {
            expect(p.x).toBeGreaterThanOrEqual(-1e-6);
            expect(p.x).toBeLessThanOrEqual(1000 + 1e-6);
        }
        // 点数应约等于 6.667s / 0.005s = 1334
        expect(points.length).toBeGreaterThan(1300);
        expect(points.length).toBeLessThan(1340);
    });

    it("滚动后重新锚定可见段：两个视口的首点都贴在视口左缘", () => {
        // 曲线总长 2000 × 5ms = 10s；滚动量必须小于它，否则视口落在曲线之后
        // （那种情况返回空数组是**正确**行为，见下一个用例）。
        const values = Array.from({ length: 2000 }, () => 50);
        const base = {
            values,
            param: "pitch" as const,
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            valueToY: valueToY(400),
        };
        const atZero = projectCurvePoints({ ...base, axis: makeAxis(150, 0) });
        // 滚动 300px = 2s，仍在曲线内（10s）
        const scrolled = projectCurvePoints({ ...base, axis: makeAxis(150, 300) });
        expect(atZero.length).toBeGreaterThan(0);
        expect(scrolled.length).toBeGreaterThan(0);
        // 两条都从视口左缘附近开始绘制（x≈0），即"重新锚定"而非沿用曲线起点
        expect(atZero[0].x).toBeCloseTo(0, 6);
        expect(scrolled[0].x).toBeCloseTo(0, 6);
        // 视口宽 1000px = 6.667s，两次都完整落在 10s 曲线内，
        // 因此可见点数相同（都是整整一个视口宽）——这本身就是"重新取窗口"的证据：
        // 若实现返回整条曲线，两者点数会都等于总采样数 2000。
        expect(atZero.length).toBe(scrolled.length);
        expect(atZero.length).toBeLessThan(values.length);
        // 滚动后的首点按内容坐标回推应约等于 2s 处，
        // 证明取的是**新的可见段**而非曲线起点。
        const firstContentX = scrolled[0].x + 300;
        expect(firstContentX).toBeCloseTo(150 * 2, 0);
    });

    it("pitch 加 0.5 偏移，其他参数不加", () => {
        const values = [60, 61];
        const pitch = projectCurvePoints({
            values,
            param: "pitch",
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            axis: makeAxis(),
            valueToY: (v) => v,
        });
        const other = projectCurvePoints({
            values,
            param: "some_automation",
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            axis: makeAxis(),
            valueToY: (v) => v,
        });
        expect(pitch.map((p) => p.y)).toEqual([60.5, 61.5]);
        expect(other.map((p) => p.y)).toEqual([60, 61]);
    });

    it("x 换算与 secToViewportPx 完全一致（不另起一套公式）", () => {
        const axis = makeAxis(150, 300);
        const values = Array.from({ length: 400 }, () => 50);
        const points = projectCurvePoints({
            values,
            param: "pitch",
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            axis,
            valueToY: valueToY(400),
        });
        const startSec = viewportStartSec(axis);
        // 逐点核对：第 k 个可见点对应的帧 = 首个 >= startSec 的采样帧
        const fp = 0.005;
        const firstIndex = Math.ceil(startSec / fp - 1e-9);
        for (let k = 0; k < points.length; k += 1) {
            const expectedX = secToViewportPx(axis, framesToTime(firstIndex + k, 0.005));
            expect(points[k].x).toBeCloseTo(expectedX, 6);
        }
    });

    it("采样点数不足 2 时返回空数组", () => {
        const base = {
            param: "pitch" as const,
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            axis: makeAxis(),
            valueToY: valueToY(400),
        };
        expect(projectCurvePoints({ ...base, values: [] })).toEqual([]);
        expect(projectCurvePoints({ ...base, values: [50] })).toEqual([]);
    });

    it("视口完全在曲线之后时返回空数组（无可见点）", () => {
        // 曲线只有 10 点（0.05s），视口从 100s 开始
        const points = projectCurvePoints({
            values: Array.from({ length: 10 }, () => 50),
            param: "pitch",
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            axis: makeAxis(150, 150 * 100),
            valueToY: valueToY(400),
        });
        expect(points).toEqual([]);
    });

    it("非有限值被跳过（不产出 NaN 点，否则整层几何失效）", () => {
        const points = projectCurvePoints({
            values: [60, Number.NaN, 62, Number.POSITIVE_INFINITY, 64],
            param: "pitch",
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            axis: makeAxis(),
            valueToY: (v) => v,
        });
        expect(points.length).toBeGreaterThan(0);
        for (const p of points) {
            expect(Number.isFinite(p.x)).toBe(true);
            expect(Number.isFinite(p.y)).toBe(true);
        }
        // 有效值应保留
        expect(points.map((p) => p.y)).toContain(60.5);
        expect(points.map((p) => p.y)).toContain(64.5);
    });

    it("stride > 1 时按步长取样", () => {
        const values = Array.from({ length: 100 }, (_, i) => i);
        const points = projectCurvePoints({
            values,
            param: "other",
            startFrame: 0,
            stride: 4,
            framePeriodMs: 5,
            axis: makeAxis(),
            valueToY: (v) => v,
        });
        // 每点间隔 4 帧 = 0.02s -> 每点间隔 3px（150px/s）
        expect(points.length).toBeGreaterThan(1);
        const dx = points[1].x - points[0].x;
        expect(dx).toBeCloseTo(150 * 0.02, 6);
    });
});

/**
 * 与 `render.ts` 的 `drawCurveTimed` 循环**逐点等价**。
 *
 * 原地复刻原实现的循环（不改动原文件），在参数组合空间上比对可见点序列的
 * 长度与每个点的 (x, y)。
 */
describe("与 drawCurveTimed 循环逐点等价", () => {
    /** 复刻 render.ts:162-194 的循环。 */
    function legacyPoints(
        values: number[],
        param: string,
        startFrame: number,
        stride: number,
        framePeriodMs: number,
        axis: ReturnType<typeof makeAxis>,
        project: (v: number) => number,
    ): { x: number; y: number }[] {
        const fp = Math.max(1e-6, framePeriodMs);
        const step = Math.max(1, Math.floor(stride));
        const visibleStartSec = viewportStartSec(axis);
        const visibleEndSec = viewportEndSec(axis);
        const out: { x: number; y: number }[] = [];
        let started = false;
        for (let i = 0; i < values.length; i += 1) {
            const frame = startFrame + i * step;
            const tSec = framesToTime(frame, fp);
            if (tSec > visibleEndSec) break;
            if (tSec < visibleStartSec) {
                started = false;
                continue;
            }
            const x = secToViewportPx(axis, tSec);
            const rawValue = values[i] ?? 0;
            const mappedValue = param === "pitch" ? rawValue + 0.5 : rawValue;
            const y = project(mappedValue);
            // 原实现只区分 moveTo / lineTo，两者都产出顶点；
            // 这里用 started 记录"是否为子路径首点"，等价于我们的索引 0。
            started = true;
            void started;
            out.push({ x, y });
        }
        return out;
    }

    it("多组参数下点数与逐点坐标一致", () => {
        let compared = 0;
        for (const param of ["pitch", "child_pitch_offset_cents@t1", "other"]) {
            for (const pxPerSec of [40, 150, 600]) {
                for (const scrollLeftPx of [0, 250, 3333]) {
                    for (const stride of [1, 3]) {
                        for (const framePeriodMs of [5, 10]) {
                            const values = Array.from(
                                { length: 900 },
                                (_, i) => 50 + 20 * Math.sin(i / 17),
                            );
                            const axis = makeAxis(pxPerSec, scrollLeftPx, 1000);
                            const project = (v: number) => 400 * (1 - v / 100);
                            const legacy = legacyPoints(
                                values,
                                param,
                                stride * 7,
                                stride,
                                framePeriodMs,
                                axis,
                                project,
                            );
                            const built = projectCurvePoints({
                                values,
                                param,
                                startFrame: stride * 7,
                                stride,
                                framePeriodMs,
                                axis,
                                valueToY: project,
                            });
                            expect(
                                built.length,
                                `param=${param} pps=${pxPerSec} scroll=${scrollLeftPx} stride=${stride} fp=${framePeriodMs}`,
                            ).toBe(legacy.length);
                            for (let i = 0; i < built.length; i += 1) {
                                expect(built[i].x).toBeCloseTo(legacy[i].x, 9);
                                expect(built[i].y).toBeCloseTo(legacy[i].y, 9);
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
        // 组合空间足够大，避免"看起来通过"其实是空循环
        expect(compared).toBeGreaterThan(10000);
    });
});

/**
 * 剪贴板预览的**专属**投影（不走 `drawCurveTimed`）。
 *
 * 【本组守护什么】它与普通曲线的**时间基准不同**：普通曲线从自身起点
 * （`startFrame + i * stride`）推算，预览从**选区起点**按原始帧距排列。
 * 若误用同一函数，预览会整体平移 `selStartSec − curveStartSec`——表现为
 * "粘贴后预览曲线跳到别处"。
 */
describe("projectClipboardPreviewPoints", () => {
    it("从选区起点按原始帧距排列（忽略 startFrame / stride）", () => {
        const points = projectClipboardPreviewPoints({
            values: [50, 51, 52],
            param: "other",
            framePeriodMs: 5, // 每点 0.005s
            selStartSec: 10, // 从选区起点 10s 开始
            selEndSec: 20,
            axis: makeAxis(150, 0),
            valueToY: (v) => v,
        });
        // 三点分别落在 10s / 10.005s / 10.01s -> x = 1500 / 1500.75 / 1501.5
        expect(points.length).toBe(3);
        expect(points[0].x).toBeCloseTo(150 * 10, 6);
        expect(points[1].x).toBeCloseTo(150 * 10.005, 6);
        expect(points[2].x).toBeCloseTo(150 * 10.01, 6);
    });

    it("超过选区终点即停止（不压缩、不延伸）", () => {
        // 10 点 × 5ms = 0.05s；选区只有 0.02s -> 只保留 10s..10.02s 内的点
        const points = projectClipboardPreviewPoints({
            values: Array.from({ length: 10 }, (_, i) => 50 + i),
            param: "other",
            framePeriodMs: 5,
            selStartSec: 10,
            selEndSec: 10.02,
            axis: makeAxis(150, 0),
            valueToY: (v) => v,
        });
        // 10.000, 10.005, 10.010, 10.015, 10.020 -> 5 点（<= 终点）
        expect(points.length).toBe(5);
        expect(points[points.length - 1].x).toBeCloseTo(150 * 10.02, 6);
    });

    it("pitch 参数同样加 0.5 偏移", () => {
        const points = projectClipboardPreviewPoints({
            values: [60, 61],
            param: "pitch",
            framePeriodMs: 5,
            selStartSec: 0,
            selEndSec: 1,
            axis: makeAxis(150, 0),
            valueToY: (v) => v,
        });
        expect(points.map((p) => p.y)).toEqual([60.5, 61.5]);
    });

    it("不做视口裁剪（依赖选区裁剪，返回点可落在视口外）", () => {
        // 选区在视口右侧之外；原实现不裁剪，只靠 ctx.clip 收口
        const points = projectClipboardPreviewPoints({
            values: [50, 51],
            param: "other",
            framePeriodMs: 5,
            selStartSec: 100, // 100s * 150 = 15000px，远在 1000px 视口外
            selEndSec: 200,
            axis: makeAxis(150, 0),
            valueToY: (v) => v,
        });
        expect(points.length).toBe(2);
        expect(points[0].x).toBeGreaterThan(1000);
    });

    it("采样不足 2 点或选区非法时返回空数组", () => {
        const base = {
            param: "other",
            framePeriodMs: 5,
            selStartSec: 0,
            selEndSec: 1,
            axis: makeAxis(),
            valueToY: (v: number) => v,
        };
        expect(projectClipboardPreviewPoints({ ...base, values: [] })).toEqual([]);
        expect(projectClipboardPreviewPoints({ ...base, values: [50] })).toEqual([]);
        // 选区反向 / 零宽
        expect(projectClipboardPreviewPoints({ ...base, values: [50, 51], selEndSec: 0 })).toEqual(
            [],
        );
    });
});

/**
 * 剪贴板预览循环的**逐点等价**（复刻 `render.ts:1150-1170`）。
 */
describe("与剪贴板预览循环逐点等价", () => {
    it("多组参数下点数与逐点坐标一致", () => {
        let compared = 0;
        for (const param of ["pitch", "other"]) {
            for (const pxPerSec of [40, 150, 600]) {
                for (const scrollLeftPx of [0, 400]) {
                    for (const framePeriodMs of [5, 10]) {
                        for (const selLenSec of [0.02, 0.2, 5]) {
                            const values = Array.from(
                                { length: 400 },
                                (_, i) => 50 + 15 * Math.sin(i / 13),
                            );
                            const axis = makeAxis(pxPerSec, scrollLeftPx, 1000);
                            const project = (v: number) => 400 * (1 - v / 100);
                            const selStartSec = 3;
                            const selEndSec = selStartSec + selLenSec;

                            // 复刻原实现的循环
                            const legacy: { x: number; y: number }[] = [];
                            const fp = Math.max(1e-6, framePeriodMs);
                            for (let i = 0; i < values.length; i += 1) {
                                const tSec = selStartSec + (i * fp) / 1000;
                                if (tSec > selEndSec) break;
                                const x = secToViewportPx(axis, tSec);
                                const mapped = param === "pitch" ? values[i] + 0.5 : values[i];
                                legacy.push({ x, y: project(mapped) });
                            }

                            const built = projectClipboardPreviewPoints({
                                values,
                                param,
                                framePeriodMs,
                                selStartSec,
                                selEndSec,
                                axis,
                                valueToY: project,
                            });
                            expect(
                                built.length,
                                `param=${param} pps=${pxPerSec} fp=${framePeriodMs} len=${selLenSec}`,
                            ).toBe(legacy.length);
                            for (let i = 0; i < built.length; i += 1) {
                                expect(built[i].x).toBeCloseTo(legacy[i].x, 9);
                                expect(built[i].y).toBeCloseTo(legacy[i].y, 9);
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
        expect(compared).toBeGreaterThan(5000);
    });
});

/**
 * 检测曲线（`clipPitchCurves`）的专用投影。
 *
 * 【为什么不能复用 `projectCurvePoints`】旧实现里检测曲线有**自己的循环**
 * （`render.ts:972-1010`），与 `drawCurveTimed` 有两处实质差异：
 *
 * 1. **时间基准不同**：检测曲线是 `curveStartSec + i*fp/1000`（曲线自己带绝对
 *    起始秒），而 `drawCurveTimed` 是 `framesToTime(startFrame + i*stride, fp)`。
 *    把 `startFrame: 0` 传给 `projectCurvePoints` 会**丢掉 `curveStartSec`**，
 *    曲线整体平移到时间轴原点。
 * 2. **无声帧必须跳过**：`midi <= 0` 表示该帧无音高。旧实现 `continue` 跳过它
 *    （保持前后点的连续性），若照画会得到一条**贯穿到底部的垂直尖刺**——这是
 *    GL 迁移后实际出现的缺陷（用户截图可见粉/紫曲线的密集竖线）。
 *
 * 【为什么不做成 `projectCurvePoints` 的开关】两者的入参语义不同（一个收
 * `startFrame`/`stride`，一个收 `curveStartSec` 且无 stride），合成一个函数会让
 * 每个调用点都要传一堆无关参数，且 `midi <= 0` 的语义只对检测曲线成立
 * （`paramView` 的 edit 曲线里 0 是合法值，不能跳）。
 */
describe("projectDetectedCurvePoints", () => {
    const axis = createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 1000 });
    const valueToY = (v: number) => 100 - v;

    it("按 curveStartSec 锚定时间（不是从 0 开始）", () => {
        // curveStartSec = 5s：第 0 个采样点应落在 x = 500（5s × 100px/s）。
        const points = projectDetectedCurvePoints({
            midiCurve: [60, 62, 64],
            curveStartSec: 5,
            framePeriodMs: 10,
            axis,
            valueToY,
        });
        expect(points.length).toBe(3);
        expect(points[0].x).toBeCloseTo(500, 6);
        // 帧距 10ms → 每点 1px
        expect(points[1].x).toBeCloseTo(501, 6);
        expect(points[2].x).toBeCloseTo(502, 6);
    });

    it("无声帧（midi <= 0）被跳过，不产生贯穿底部的尖刺", () => {
        const points = projectDetectedCurvePoints({
            midiCurve: [60, 0, 62, 0, 0, 64],
            curveStartSec: 0,
            framePeriodMs: 10,
            axis,
            valueToY,
        });
        // 只保留 3 个有声帧
        expect(points.length).toBe(3);
        // 被跳过的点不应以任何形式出现（尤其不能是 valueToY(0+0.5) 那个低点）
        const silentY = valueToY(0.5);
        for (const p of points) {
            expect(p.y).not.toBeCloseTo(silentY, 6);
        }
        // 且 x 连续跳过了无声帧的位置
        expect(points.map((p) => Math.round(p.x))).toEqual([0, 2, 5]);
    });

    it("负值同样视为无声帧（<=0 而不是 ===0）", () => {
        const points = projectDetectedCurvePoints({
            midiCurve: [60, -1, 62],
            curveStartSec: 0,
            framePeriodMs: 10,
            axis,
            valueToY,
        });
        expect(points.length).toBe(2);
    });

    it("非有限值被跳过（与旧实现的 isFinite 检查一致）", () => {
        const points = projectDetectedCurvePoints({
            midiCurve: [60, Number.NaN, 62, Number.POSITIVE_INFINITY, 64],
            curveStartSec: 0,
            framePeriodMs: 10,
            axis,
            valueToY,
        });
        expect(points.length).toBe(3);
    });

    it("pitch 加 0.5 偏移（与绘制同源）", () => {
        const points = projectDetectedCurvePoints({
            midiCurve: [60, 62],
            curveStartSec: 0,
            framePeriodMs: 10,
            axis,
            valueToY,
        });
        expect(points[0].y).toBeCloseTo(valueToY(60.5), 6);
        expect(points[1].y).toBeCloseTo(valueToY(62.5), 6);
    });

    it("超出右缘立即 break（采样时间单调递增）", () => {
        // 视口 1000px / 100px每秒 = 10s；帧距 100ms → 每点 10px。
        // 右缘判定是 `x > viewportWidthPx + 10`，即 x > 1010 → 第 102 个点停。
        const many = new Array(300).fill(60);
        const points = projectDetectedCurvePoints({
            midiCurve: many,
            curveStartSec: 0,
            framePeriodMs: 100,
            axis,
            valueToY,
        });
        expect(points[points.length - 1].x).toBeLessThanOrEqual(1010);
        expect(points.length).toBeLessThan(300);
    });

    it("左侧不可见部分被跳过，首点从左缘附近开始", () => {
        const scrolled = createTimelineAxis({
            pxPerSec: 100,
            scrollLeftPx: 500,
            viewportWidthPx: 1000,
        });
        const points = projectDetectedCurvePoints({
            midiCurve: new Array(200).fill(60),
            curveStartSec: 0,
            framePeriodMs: 100, // 每点 10px
            axis: scrolled,
            valueToY,
        });
        // 视口从 5s 开始（scrollLeft=500 / 100px每秒）；每点 10px。
        // 左缘判定是 `x < -10` 才跳过（**开区间**），所以第 49 个点（x 恰为 -10，
        // 浮点误差下 -9.999…）会被保留 —— 这与旧实现逐字一致，不是 off-by-one。
        expect(points[0].x).toBeGreaterThanOrEqual(-10);
        expect(points[0].x).toBeLessThan(0);
        // 关键是它紧贴左缘（而不是把整条曲线的点都留在数组里）
        expect(points[0].x).toBeCloseTo(-10, 6);
    });

    it("全为无声帧时返回空数组（不画任何东西）", () => {
        const points = projectDetectedCurvePoints({
            midiCurve: [0, 0, 0],
            curveStartSec: 0,
            framePeriodMs: 10,
            axis,
            valueToY,
        });
        expect(points).toEqual([]);
    });

    it("采样不足 2 点或非有限入参时返回空数组", () => {
        expect(
            projectDetectedCurvePoints({
                midiCurve: [60],
                curveStartSec: 0,
                framePeriodMs: 10,
                axis,
                valueToY,
            }),
        ).toEqual([]);
        expect(
            projectDetectedCurvePoints({
                midiCurve: [60, 62],
                curveStartSec: Number.NaN,
                framePeriodMs: 10,
                axis,
                valueToY,
            }),
        ).toEqual([]);
    });
});
