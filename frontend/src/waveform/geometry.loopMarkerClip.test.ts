/**
 * ★ 回归：回绕 / 媒体边界标记（▽）必须被 **Clip 边缘裁断**。
 *
 * ## 缺陷形态
 *
 * 标记是固定半宽的实心 ▽（半宽 ≈ `size × 0.62`，`size` 由行高决定），位置由
 * "回绕点相对 Clip 起点的偏移"算出。贴着 Clip 边缘的那个标记因此有一半画在
 * Clip **外面**（循环节紧贴左缘时尤其明显），看起来像悬空的一块。
 *
 * ## 本用例钉住什么
 *
 * 几何层按 `clipLeftPx` / `clipRightPx` 把每个扫描行裁到本体范围内：
 * - 贴着边缘的标记，顶点 x 不得越过该边缘；
 * - 完全落在本体之外的标记不产生任何顶点（不画退化线段）；
 * - 中间的标记不受影响（对照，证明裁剪只在边缘生效）。
 *
 * 同时保留一条**反证**：把边界放宽后，同一个标记确实会探出边缘 —— 否则本组
 * 用例可能因为"三角形本来就没探出"而假绿。
 */
import { describe, expect, it } from "vitest";

import { buildWaveformGeometry } from "./geometry";
import type { WaveformScene } from "./sceneBuilder.ts";

function markerScene(args: {
    xPx: number;
    clipLeftPx: number;
    clipRightPx: number;
}): WaveformScene {
    return {
        segments: [],
        markers: [
            {
                clipId: "c",
                timelineSec: 1,
                xPx: args.xPx,
                yPx: 0,
                heightPx: 100,
                kind: "loop",
                clipLeftPx: args.clipLeftPx,
                clipRightPx: args.clipRightPx,
            },
        ],
    } as WaveformScene;
}

/** 标记顶点的 x 集合（场景里没有段 ⇒ 全部顶点都来自标记）。 */
function markerXs(scene: WaveformScene): number[] {
    const geometry = buildWaveformGeometry({
        scene,
        color: "#ffffff",
        getPeaks: () => null,
    });
    const xs: number[] = [];
    for (let v = 0; v < geometry.vertices.length; v += 6) {
        xs.push(geometry.vertices[v] ?? 0);
    }
    return xs;
}

describe("★ 回绕标记被 Clip 边缘裁断", () => {
    it("贴着左缘的标记：顶点不得越过左缘", () => {
        const xs = markerXs(markerScene({ xPx: 0, clipLeftPx: 0, clipRightPx: 100 }));
        expect(xs.length).toBeGreaterThan(0);
        expect(Math.min(...xs)).toBeGreaterThanOrEqual(0);
    });

    it("贴着右缘的标记：顶点不得越过右缘", () => {
        const xs = markerXs(markerScene({ xPx: 100, clipLeftPx: 0, clipRightPx: 100 }));
        expect(xs.length).toBeGreaterThan(0);
        expect(Math.max(...xs)).toBeLessThanOrEqual(100);
    });

    it("★ 反证：边界放宽后同一个标记确实会探出（裁剪真的在起作用）", () => {
        const xs = markerXs(
            markerScene({ xPx: 0, clipLeftPx: -1000, clipRightPx: 1000 }),
        );
        // 半宽 ≈ 4.34 ⇒ 底行左端约为 −3.84，确实在 Clip 左缘之外。
        expect(Math.min(...xs)).toBeLessThan(0);
    });

    it("完全落在本体之外的标记 ⇒ 不产生顶点", () => {
        const xs = markerXs(markerScene({ xPx: 500, clipLeftPx: 0, clipRightPx: 100 }));
        expect(xs).toEqual([]);
    });

    it("对照：位于本体中部的标记不被裁（形状不变）", () => {
        const xs = markerXs(markerScene({ xPx: 50, clipLeftPx: 0, clipRightPx: 100 }));
        // 底行（i=0）x = 50.5 ± 4.34 ⇒ 两侧都完整保留。
        expect(Math.min(...xs)).toBeCloseTo(50.5 - 7 * 0.62, 5);
        expect(Math.max(...xs)).toBeCloseTo(50.5 + 7 * 0.62, 5);
    });
});
