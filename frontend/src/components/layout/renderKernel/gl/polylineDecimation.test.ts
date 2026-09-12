/**
 * 折线抽稀（./polylineDecimation）行为自检。
 *
 * 【本测试守护什么】
 * 1. **包络保真**：每个设备像素列内，原始点的 y 最小/最大值必须都保留在结果里。
 *    这是"保持精度"的核心——抽稀只能丢掉**同一像素上重复覆盖**的点，绝不能丢掉
 *    曲线折返的极值，否则 pitch 曲线的八度跳变会被削平。
 * 2. **弧长保真**：结果携带每个点在**原始**序列中的累积弧长。虚线相位按弧长推进，
 *    若改用抽稀后的折线重新累加，弧长会变短、虚线图案随之漂移。
 * 3. **端点保真**：整条曲线的首末点必须保留，否则曲线两端会被截短。
 * 4. **预算与安全侧**：未超过设备像素可分辨的点数时**零改动**返回（连数组都不复制）；
 *    入参非法或输入非单调时同样原样返回，绝不产出更差的结果。
 *
 * 【为什么按"设备像素列"而不是"点数"抽稀】
 * 最小缩放（4 px/s）下视口 1864 CSS px 覆盖 466 秒，而曲线按 200 点/秒采样：
 * 可见点约 93,200 个，落进 3,728 个物理像素列——**每列 25 个点**。同列的点渲染在
 * 同一个像素上，视觉完全冗余。按列抽稀把成本从"随点数"变成"随像素列"，
 * 这是根因修复；点数阈值只是用来在不需要时完全跳过这层。
 *
 * 【测试设计说明：为什么断言"每列的极值都在"而不是"点数等于某值"】
 * 断言点数会锁死实现细节（首末点、单点列、去重都会影响计数），改一处就假失败。
 * 断言极值保留才是**真正的契约**：它直接对应"视觉上没有丢信息"。
 */
import { describe, expect, it } from "vitest";

import type { PolylinePoint } from "./polylineGeometry";
import { decimatePolylinePoints } from "./polylineDecimation";

/** 按 `fn` 造 `n` 个横向铺满 `widthPx` 的点。 */
function makePoints(n: number, widthPx: number, fn: (t: number) => number): PolylinePoint[] {
    const out: PolylinePoint[] = [];
    for (let i = 0; i < n; i += 1) {
        const t = n === 1 ? 0 : i / (n - 1);
        out.push({ x: t * widthPx, y: fn(t) });
    }
    return out;
}

/** 原始序列的累积弧长。 */
function cumulativeAlong(points: readonly PolylinePoint[]): number[] {
    const out: number[] = [0];
    for (let i = 1; i < points.length; i += 1) {
        out.push(
            out[i - 1] +
                Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y),
        );
    }
    return out;
}

/** 把点归入设备像素列。 */
function columnOf(x: number, dpr: number, cols: number): number {
    return Math.min(cols - 1, Math.max(0, Math.floor(x * dpr)));
}

describe("decimatePolylinePoints", () => {
    it("点数未超过设备像素可分辨预算时零改动返回（不复制数组）", () => {
        const points = makePoints(10, 100, (t) => Math.sin(t * 6));
        const result = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });

        expect(result.decimated).toBe(false);
        // 同一引用：常见路径（放大到中高缩放）不该付任何拷贝成本。
        expect(result.points).toBe(points);
        // 未抽稀时不需要覆盖弧长，调用方按原样累加即可。
        expect(result.along).toBeNull();
    });

    it("点数远超预算时抽稀，且每设备像素列至多 2 个点", () => {
        const points = makePoints(10000, 100, (t) => Math.sin(t * 50));
        const result = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });

        expect(result.decimated).toBe(true);
        // 200 个设备列 × 2 点，再加首末两点（可能落在已计数的列里，故用上界）。
        expect(result.points.length).toBeLessThanOrEqual(200 * 2 + 2);
        expect(result.points.length).toBeLessThan(points.length);
    });

    it("包络保真：每个设备像素列的 y 极值都保留在结果中", () => {
        const dpr = 2;
        const widthPx = 100;
        const cols = widthPx * dpr;
        // 高频振荡：同列内反复折返，最容易在抽稀时被削平。
        const points = makePoints(12000, widthPx, (t) => Math.sin(t * 400) * 10 + Math.sin(t * 37));
        const result = decimatePolylinePoints({ points, dpr, viewportWidthPx: widthPx });

        // 原始：每列的 min / max
        const minByCol = new Map<number, number>();
        const maxByCol = new Map<number, number>();
        for (const p of points) {
            const c = columnOf(p.x, dpr, cols);
            const lo = minByCol.get(c);
            const hi = maxByCol.get(c);
            if (lo === undefined || p.y < lo) minByCol.set(c, p.y);
            if (hi === undefined || p.y > hi) maxByCol.set(c, p.y);
        }
        // 结果中的 y 集合
        const kept = new Set(result.points.map((p) => p.y));
        let missing = 0;
        for (const [c, lo] of minByCol) {
            if (!kept.has(lo)) missing += 1;
            const hi = maxByCol.get(c);
            if (hi !== undefined && !kept.has(hi)) missing += 1;
        }
        expect(missing).toBe(0);
    });

    it("顺序保真：结果的 x 单调不减（折线不能自交或倒退）", () => {
        const points = makePoints(9000, 100, (t) => Math.sin(t * 220));
        const result = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });

        for (let i = 1; i < result.points.length; i += 1) {
            expect(result.points[i].x).toBeGreaterThanOrEqual(result.points[i - 1].x);
        }
    });

    it("弧长保真：along 是该点在原始序列中的累积弧长，而非抽稀后重算", () => {
        const points = makePoints(8000, 100, (t) => Math.sin(t * 90) * 5);
        const result = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });
        expect(result.decimated).toBe(true);

        const original = cumulativeAlong(points);
        // 按坐标反查下标（结果点是从原始点原样复制来的）。
        const indexOf = new Map<string, number>();
        points.forEach((p, i) => indexOf.set(`${p.x},${p.y}`, i));

        let checked = 0;
        for (let i = 0; i < result.points.length; i += 1) {
            const p = result.points[i];
            const src = indexOf.get(`${p.x},${p.y}`);
            expect(src).toBeDefined();
            expect(result.along?.[i]).toBeCloseTo(original[src as number], 9);
            checked += 1;
        }
        expect(checked).toBe(result.points.length);
        // 关键：弧长严格单调不减，虚线相位才能正确推进。
        for (let i = 1; i < (result.along?.length ?? 0); i += 1) {
            expect(result.along?.[i]).toBeGreaterThanOrEqual(result.along?.[i - 1] as number);
        }
    });

    it("端点保真：整条曲线的首末点必须保留", () => {
        const points = makePoints(12000, 100, (t) => Math.sin(t * 300) * 3);
        const result = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });

        const first = result.points[0];
        const last = result.points[result.points.length - 1];
        expect({ x: first.x, y: first.y }).toEqual({ x: points[0].x, y: points[0].y });
        const srcLast = points[points.length - 1];
        expect({ x: last.x, y: last.y }).toEqual({ x: srcLast.x, y: srcLast.y });
    });

    it("dpr 决定列宽：dpr 越大保留的点越多（分辨率更高）", () => {
        const points = makePoints(20000, 100, (t) => Math.sin(t * 600) * 4);
        const at1 = decimatePolylinePoints({ points, dpr: 1, viewportWidthPx: 100 });
        const at2 = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });
        const at4 = decimatePolylinePoints({ points, dpr: 4, viewportWidthPx: 100 });

        expect(at1.points.length).toBeLessThan(at2.points.length);
        expect(at2.points.length).toBeLessThan(at4.points.length);
    });

    it("抽稀后不可能比原序列更大：预算边界上原样返回", () => {
        // 恰好超过预算 1 个点：抽稀的固定开销（端点）会让结果更大，
        // 此时必须放弃抽稀，否则"优化"反而变慢。
        const cols = 100 * 2;
        const points = makePoints(cols * 2 + 1, 100, (t) => Math.sin(t * 900) * 2);
        const result = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });
        expect(result.points.length).toBeLessThanOrEqual(points.length);
    });

    it("入参非法时原样返回", () => {
        const points = makePoints(5000, 100, () => 0);
        for (const bad of [
            { dpr: 0, viewportWidthPx: 100 },
            { dpr: -1, viewportWidthPx: 100 },
            { dpr: Number.NaN, viewportWidthPx: 100 },
            { dpr: 2, viewportWidthPx: 0 },
            { dpr: 2, viewportWidthPx: -100 },
            { dpr: 2, viewportWidthPx: Number.POSITIVE_INFINITY },
        ]) {
            const result = decimatePolylinePoints({ points, ...bad });
            expect(result.decimated).toBe(false);
            expect(result.points).toBe(points);
        }
    });

    it("点数不足 2 或存在非有限值时原样返回", () => {
        const single = [{ x: 0, y: 0 }];
        expect(
            decimatePolylinePoints({ points: single, dpr: 2, viewportWidthPx: 100 }).decimated,
        ).toBe(false);
        expect(
            decimatePolylinePoints({ points: [], dpr: 2, viewportWidthPx: 100 }).decimated,
        ).toBe(false);

        // 非有限值：抽稀的分列依赖 x 可比，出现 NaN 时按安全侧整条放弃。
        const withNaN = makePoints(5000, 100, (t) => t).map((p, i) =>
            i === 2000 ? { x: Number.NaN, y: p.y } : p,
        );
        const result = decimatePolylinePoints({ points: withNaN, dpr: 2, viewportWidthPx: 100 });
        expect(result.decimated).toBe(false);
        expect(result.points).toBe(withNaN);
    });

    it("x 非单调时原样返回（分列算法依赖 x 单调不减）", () => {
        // 构造一段回退：抽稀若按"当前列"归并会把回退段错误地并入前一列。
        const points: PolylinePoint[] = [];
        for (let i = 0; i < 6000; i += 1) {
            const t = i / 6000;
            points.push({ x: t < 0.5 ? t * 200 : 100 - (t - 0.5) * 200, y: Math.sin(t * 300) });
        }
        const result = decimatePolylinePoints({ points, dpr: 2, viewportWidthPx: 100 });
        expect(result.decimated).toBe(false);
        expect(result.points).toBe(points);
    });
});
