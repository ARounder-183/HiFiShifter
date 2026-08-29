/**
 * 画布 DPR 光栅化契约自检。
 *
 * 【主要内容】验证 `rasterize()` 的物理尺寸取整规则，以及它给出的
 * `u_resolution` 能让 WebGL 的 NDC 映射与 Canvas2D 的 `setTransform(dpr,…)`
 * 严格等价。
 *
 * 【作用】这是波形（WebGL2）与 clip 体（Canvas2D）缩放比一致的核心守护：
 * 历史上两者取整规则不同（round vs floor），且 WebGL 的 u_resolution 传的是
 * CSS 尺寸，导致波形实际缩放比为 `round(w*dpr)/w` 而非 dpr。
 *
 * 【与其他模块的关系】仅覆盖 `canvasRaster.ts`，不依赖真实 DOM（画布用最小
 * 替身，只验证纯计算部分）。
 */

import { test } from "vitest";

import { rasterize } from "./canvasRaster.js";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (actual !== expected) {
        throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
    }
}

function assertTrue(condition: boolean, label: string): void {
    if (!condition) throw new Error(`${label}: expected true`);
}

/** 最小画布替身：rasterize 只读写 width/height/style。 */
function fakeCanvas(): HTMLCanvasElement {
    return {
        width: 0,
        height: 0,
        style: { width: "", height: "" },
    } as unknown as HTMLCanvasElement;
}

test("components/layout/timeline/runtime/canvasRaster.test.ts scripted checks", async () => {
    // ── 1. 物理尺寸取整规则：所有画布一律 Math.round(css * dpr) ──────
    {
        for (const dpr of [1, 1.25, 1.5, 2, 3]) {
            for (const css of [1, 7, 100, 333.4, 1000.5, 1500.49]) {
                const canvas = fakeCanvas();
                const target = rasterize(canvas, css, css / 2, dpr);
                assertEqual(
                    target.physicalWidth,
                    Math.max(1, Math.round(css * dpr)),
                    `physicalWidth (dpr=${dpr}, css=${css})`,
                );
                assertEqual(canvas.width, target.physicalWidth, `canvas.width (dpr=${dpr})`);
                assertEqual(canvas.style.width, `${css}px`, `style.width (dpr=${dpr})`);
            }
        }
    }

    // ── 2. 契约核心：u_resolution 必须让 CSS 坐标严格映射到 css*dpr 物理像素 ──
    // NDC: clip = pos / resolution * 2 - 1 → 物理像素 = pos / resolution * physical
    // 代入 resolution = physical / dpr 得物理像素 = pos * dpr，与 Canvas2D 一致。
    {
        for (const dpr of [1, 1.25, 1.5, 2, 3]) {
            for (const css of [7, 333.4, 1000.5, 1500.49]) {
                const target = rasterize(fakeCanvas(), css, 100, dpr);
                // 用 CSS 坐标 css（即画布右缘）验证映射
                const physicalAtRightEdge = (css / target.resolutionWidth) * target.physicalWidth;
                assertTrue(
                    Math.abs(physicalAtRightEdge - css * dpr) < 1e-6,
                    `resolution maps to css*dpr (dpr=${dpr}, css=${css}): ` +
                        `${physicalAtRightEdge} vs ${css * dpr}`,
                );
            }
        }
    }

    // ── 3. 传 CSS 尺寸当 u_resolution 会产生的偏差（回归守护的反例） ──
    // 旧实现正是这么做的：当 css*dpr 非整数时缩放比会偏离 dpr。
    {
        const dpr = 1.5;
        const css = 1000.5;
        const target = rasterize(fakeCanvas(), css, 100, dpr);
        const wrongPhysicalAtRightEdge = (css / target.cssWidthPx) * target.physicalWidth;
        assertTrue(
            Math.abs(wrongPhysicalAtRightEdge - css * dpr) > 1e-9,
            "using CSS size as u_resolution must be detectable as wrong",
        );
    }

    // ── 4. 非法输入兜底：不得产生 0 / NaN 尺寸 ──────────────────────
    {
        for (const [cssW, cssH, dpr] of [
            [0, 0, 1],
            [-100, -50, 2],
            [Number.NaN, Number.NaN, Number.NaN],
            [100, 100, 0],
            [100, 100, -2],
        ]) {
            const target = rasterize(fakeCanvas(), cssW, cssH, dpr);
            assertTrue(target.physicalWidth >= 1, `physicalWidth guard (${cssW},${cssH},${dpr})`);
            assertTrue(target.physicalHeight >= 1, `physicalHeight guard (${cssW},${cssH},${dpr})`);
            assertTrue(
                Number.isFinite(target.resolutionWidth) && target.resolutionWidth > 0,
                `resolutionWidth finite (${cssW},${cssH},${dpr})`,
            );
        }
    }

    // ── 5. 幂等：同参数重复调用结果一致，且不会把样式写成 NaN ────────
    {
        const canvas = fakeCanvas();
        const first = rasterize(canvas, 800.5, 600.25, 2);
        const second = rasterize(canvas, 800.5, 600.25, 2);
        assertEqual(first.physicalWidth, second.physicalWidth, "idempotent width");
        assertEqual(first.physicalHeight, second.physicalHeight, "idempotent height");
        assertEqual(first.resolutionWidth, second.resolutionWidth, "idempotent resolution");
        assertTrue(!canvas.style.width.includes("NaN"), "style has no NaN");
    }
});
