/**
 * 折线几何构建（./polylineGeometry）行为自检。
 *
 * 【主要内容】
 * 1. 每条线段产出 2 个三角形（6 个顶点），顶点携带**到中心线的有符号距离**
 *    （`across`）与沿线的累积弧长（`along`），而不是被展开到精确外沿——后续要靠
 *    片元着色器做亚像素抗锯齿与虚线相位；
 * 2. 拐角用 **miter 连接**（Canvas2D 默认 `lineJoin: "miter"`，本工程从未设置过
 *    `lineJoin`）；miter 比例超限时必须退化为 bevel，否则尖角会甩出长刺；
 * 3. 几何外沿 = `半线宽 + aaPadPx`：不预留 AA 余量时最外圈没有像素可承载覆盖率
 *    衰减，线会比 Canvas2D 更细；
 * 4. 少于 2 个点、非有限坐标、零/负线宽都必须返回空而不是抛错或产出 NaN
 *    （NaN 会让整个顶点缓冲失效、整层消失）。
 *
 * 【测试设计说明：miter 断言为什么看**位置**而不是 `across`】
 * 距离编码下，miter 尖点的 `across` 仍是 `±半线宽`——因为尖点落在两条偏移线的
 * 交点上，而两条偏移线到各自中心线的垂直距离都是 `半线宽`。所以「顶点被推远了」
 * 这件事只体现在**位置**上。断言 `across` 会在两种连接方式下都通过，等于没测。
 *
 * 曾经还写错过用例本身：用 (0,0)-(100,0.5)-(200,0) 当"尖角"，它的转角实际只有
 * 0.57°、miter 比例 ≈1.000，**永远走不到 bevel 分支**。因此用例一律按**转角**
 * 构造，并用 `miter比例 = 1/cos(转角/2)` 这个解析值来挑选 limit。
 */
import { describe, expect, it } from "vitest";

import { buildPolylineVertices, POLYLINE_FLOATS_PER_VERTEX } from "./polylineGeometry";

/** 取出所有顶点的 `across` 分量。 */
function acrossOf(buf: Float32Array): number[] {
    const out: number[] = [];
    for (let i = 0; i < buf.length; i += POLYLINE_FLOATS_PER_VERTEX) {
        out.push(buf[i + 3]);
    }
    return out;
}

/** 取出所有顶点的 `along` 分量。 */
function alongOf(buf: Float32Array): number[] {
    const out: number[] = [];
    for (let i = 0; i < buf.length; i += POLYLINE_FLOATS_PER_VERTEX) {
        out.push(buf[i + 2]);
    }
    return out;
}

/**
 * 判定缓冲中是否存在位于 `(x, y)` 的顶点（容差 `tol`）。
 *
 * 【为什么直接找顶点而不是测"最远距离"】测最远距离会被**同一段另一端的顶点**
 * 干扰（段长 10 ≫ 半线宽 5），于是 bevel 与 miter 都得到 10，断言失效；
 * 收窄邻域又会把 150° 转角下 19.3px 远的尖点排除掉，产生假失败。
 * miter 的解析位置是已知的（见下），直接按位置查找既无歧义也无阈值调参。
 */
function hasVertexAt(buf: Float32Array, x: number, y: number, tol = 1e-6): boolean {
    for (let i = 0; i < buf.length; i += POLYLINE_FLOATS_PER_VERTEX) {
        if (Math.abs(buf[i] - x) <= tol && Math.abs(buf[i + 1] - y) <= tol) return true;
    }
    return false;
}

/**
 * 解析计算 miter 尖点相对连接点的偏移。
 *
 * 第一段方向取 `+x`（左法线 `(0,1)`），第二段方向由转角给出；
 * `miter向量 = (n1 + n2) · 2 · offset / |n1 + n2|²`。
 *
 * @param turnDeg 转角（度）。
 * @param offset 半线宽 + AA 余量（CSS px）。
 * @returns 尖点相对连接点的位移与 miter 比例。
 */
function analyticMiter(turnDeg: number, offset: number) {
    const n1 = { x: 0, y: 1 };
    const a = (turnDeg * Math.PI) / 180;
    const n2 = { x: -Math.sin(a), y: Math.cos(a) };
    const sx = n1.x + n2.x;
    const sy = n1.y + n2.y;
    const m2 = sx * sx + sy * sy;
    const mx = (sx * 2 * offset) / m2;
    const my = (sy * 2 * offset) / m2;
    return { dx: mx, dy: my, ratio: Math.hypot(mx, my) / offset };
}

/** 由转角反推 miter 比例：`1 / cos(θ/2)`。 */
function miterRatio(turnDeg: number): number {
    return 1 / Math.cos((turnDeg * Math.PI) / 180 / 2);
}

describe("buildPolylineVertices", () => {
    it("单段产出 2 个三角形（6 顶点）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        expect(out.length).toBe(6 * POLYLINE_FLOATS_PER_VERTEX);
    });

    it("几何外沿 = 半线宽 + AA 余量（aaPadPx 可显式关闭）", () => {
        // aaPadPx = 0：纯几何语义，外沿恰好是半线宽
        const bare = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 4,
            miterLimit: 10,
            aaPadPx: 0,
        });
        const bareAcross = acrossOf(bare);
        expect(Math.min(...bareAcross)).toBeCloseTo(-2, 9);
        expect(Math.max(...bareAcross)).toBeCloseTo(2, 9);

        // 默认：外扩到 半线宽 + pad，给片元的覆盖率衰减留出采样空间。
        // 不这样做，最外圈像素拿不到部分覆盖，线会比 Canvas2D 细。
        const padded = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 4,
            miterLimit: 10,
        });
        const paddedAcross = acrossOf(padded);
        expect(Math.max(...paddedAcross)).toBeCloseTo(2.5, 9); // 2 + 0.5
        expect(Math.min(...paddedAcross)).toBeCloseTo(-2.5, 9);

        // 注意：`across` 记录的是**几何**距离（含 pad），而着色器的覆盖率阈值取
        // lineWidth/2。因此 `u_halfWidth` 必须作为独立 uniform 传入，
        // 不能从几何的 max|across| 反推。
    });

    it("沿线的累积弧长写入 along（供虚线相位使用）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 3, y: 4 },
                { x: 3, y: 14 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        const along = alongOf(out);
        // 第一段长 5（3-4-5 直角三角形），第二段长 10 -> 总 15
        expect(Math.min(...along)).toBeCloseTo(0, 9);
        expect(Math.max(...along)).toBeCloseTo(15, 9);
    });

    it("miter 未超限时走 miter（顶点被推远），超限时退化为 bevel", () => {
        // 用例按**转角**构造，并用解析比例挑 limit：
        //   90°  -> 1/cos45 = 1.414
        //   150° -> 1/cos75 = 3.864
        expect(miterRatio(90)).toBeCloseTo(1.414, 3);
        expect(miterRatio(150)).toBeCloseTo(3.864, 3);

        const rightAngle = [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 10, y: 10 },
        ];
        const turn = (150 * Math.PI) / 180;
        const sharpAngle = [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 10 + Math.cos(turn) * 10, y: Math.sin(turn) * 10 },
        ];
        // 用 aaPadPx: 0 隔离**纯几何**行为，避免 AA 余量干扰断言
        const lineWidth = 10;
        const halfWidth = lineWidth / 2;
        const offset = halfWidth; // pad = 0 时几何外沿即半线宽

        // 90° + limit 2（1.414 < 2）：走 miter，尖点应出现在解析位置
        const mitered = buildPolylineVertices({
            points: rightAngle,
            lineWidth: 10,
            miterLimit: 2,
            aaPadPx: 0,
        });
        const a90 = analyticMiter(90, offset);
        expect(hasVertexAt(mitered, 10 + a90.dx, 0 + a90.dy)).toBe(true);

        // 150° + limit 2（3.864 > 2）：退化 bevel，**不得**出现尖点
        const beveled = buildPolylineVertices({
            points: sharpAngle,
            lineWidth: 10,
            miterLimit: 2,
            aaPadPx: 0,
        });
        const a150 = analyticMiter(150, offset);
        expect(hasVertexAt(beveled, 10 + a150.dx, 0 + a150.dy)).toBe(false);

        // 同样 150° 但放宽 limit（10 > 3.864）：尖点应出现
        const loose = buildPolylineVertices({
            points: sharpAngle,
            lineWidth: 10,
            miterLimit: 10,
            aaPadPx: 0,
        });
        expect(hasVertexAt(loose, 10 + a150.dx, 0 + a150.dy)).toBe(true);

        // 两种连接的顶点数差异也应体现出来（bevel 少一组尖角三角形）
        expect(loose.length).toBeGreaterThan(beveled.length);
    });

    it("少于 2 个点返回空数组", () => {
        expect(buildPolylineVertices({ points: [], lineWidth: 2, miterLimit: 10 }).length).toBe(0);
        expect(
            buildPolylineVertices({
                points: [{ x: 1, y: 1 }],
                lineWidth: 2,
                miterLimit: 10,
            }).length,
        ).toBe(0);
    });

    it("非有限坐标 / 非法线宽返回空数组（防 NaN 污染顶点缓冲）", () => {
        const bad = [
            {
                points: [
                    { x: 0, y: 0 },
                    { x: Number.NaN, y: 1 },
                ],
                lineWidth: 2,
                miterLimit: 10,
            },
            {
                points: [
                    { x: 0, y: 0 },
                    { x: 1, y: Number.POSITIVE_INFINITY },
                ],
                lineWidth: 2,
                miterLimit: 10,
            },
            {
                points: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                ],
                lineWidth: 0,
                miterLimit: 10,
            },
            {
                points: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                ],
                lineWidth: -3,
                miterLimit: 10,
            },
        ];
        for (const args of bad) {
            expect(buildPolylineVertices(args).length).toBe(0);
        }
    });

    it("重合点不产生 NaN（相邻点相同时跳过该段）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 5, y: 5 },
                { x: 5, y: 5 },
                { x: 15, y: 5 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        expect(out.length).toBeGreaterThan(0);
        for (let i = 0; i < out.length; i += 1) {
            expect(Number.isFinite(out[i])).toBe(true);
        }
    });

    it("全部点重合时返回空数组（没有可绘制的段）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 5, y: 5 },
                { x: 5, y: 5 },
                { x: 5, y: 5 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        expect(out.length).toBe(0);
    });
});
