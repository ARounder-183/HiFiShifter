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

import { clearCanvasPhysical, fitCssPxToDevicePx, rasterize } from "./canvasRaster.js";

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

/**
 * 假画布：`style.width` / `style.height` 是访问器，记录**写入次数**，并把写入值
 * 规范化到 6 位有效数字 —— 复现 Chromium 对内联长度的序列化行为。
 *
 * 【为什么需要它】回读 `style.width` 与期望值比较时，这个规范化会让
 * `physical / dpr`（分数 dpr 下是无限小数）永远"看起来变了"，导致每帧重写样式。
 * 有了可计数的写入，该缺陷才可被单测捕获。
 */
function countingCanvas(): { canvas: HTMLCanvasElement; writes: () => number } {
    const backing: Record<string, string> = { width: "", height: "" };
    let count = 0;
    const style = {} as CSSStyleDeclaration;
    for (const key of ["width", "height"] as const) {
        Object.defineProperty(style, key, {
            configurable: true,
            get: () => backing[key],
            set: (value: string) => {
                count += 1;
                const num = Number.parseFloat(value);
                backing[key] = Number.isFinite(num) ? `${Number(num.toPrecision(6))}px` : value;
            },
        });
    }
    return {
        canvas: { width: 0, height: 0, style } as unknown as HTMLCanvasElement,
        writes: () => count,
    };
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
            }
        }
    }

    // ── 1b. 核心不变式：style × dpr 严格等于 canvas.width ─────────────
    // 物理尺寸取整后必须把 CSS 尺寸**回算**为 physical/dpr。沿用未取整的原始
    // CSS 尺寸会让浏览器把 physical 个像素铺到 css*dpr 个像素的布局盒上，
    // 合成器做非整数倍重采样 → 整块画面发虚（用户报告过：窗口拖到某些宽度
    // 波形糊、另一些宽度清晰，非整数 dpr 下按宽度奇偶交替）。
    {
        for (const dpr of [1, 1.25, 1.5, 1.75, 2, 2.5, 3]) {
            for (const css of [7, 333.4, 999, 1000, 1000.5, 1001, 1500.49]) {
                const canvas = fakeCanvas();
                const target = rasterize(canvas, css, 100, dpr);
                const styleWidth = Number.parseFloat(canvas.style.width);
                assertTrue(Number.isFinite(styleWidth), `style.width finite (dpr=${dpr})`);
                assertTrue(
                    Math.abs(styleWidth * dpr - canvas.width) < 1e-6,
                    `style × dpr === canvas.width (dpr=${dpr}, css=${css}): ` +
                        `${styleWidth * dpr} vs ${canvas.width}`,
                );
                // 合成器缩放比 = 布局盒设备像素 / backing store，必须恒为 1。
                assertTrue(
                    Math.abs((styleWidth * dpr) / target.physicalWidth - 1) < 1e-9,
                    `compositor scale === 1 (dpr=${dpr}, css=${css})`,
                );
            }
        }
    }

    // ── 1c. 回归守护：旧实现（style = 原始 CSS 尺寸）必须可被识别为错误 ──
    // 这些组合下 css*dpr 非整数，旧写法的合成器缩放比必然偏离 1。
    {
        for (const [css, dpr] of [
            [1001, 1.5],
            [1001, 1.25],
            [800.6, 1.25],
            [1000.4, 1.5],
        ]) {
            const physical = Math.max(1, Math.round(css * dpr));
            const legacyScale = (css * dpr) / physical;
            assertTrue(
                Math.abs(legacyScale - 1) > 1e-9,
                `legacy CSS-size write-back must be detectable as wrong (css=${css}, dpr=${dpr})`,
            );
            // 新实现必须把它归零。
            const canvas = fakeCanvas();
            const target = rasterize(canvas, css, 100, dpr);
            const styleWidth = Number.parseFloat(canvas.style.width);
            assertTrue(
                Math.abs((styleWidth * dpr) / target.physicalWidth - 1) < 1e-9,
                `new write-back fixes it (css=${css}, dpr=${dpr})`,
            );
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

    // ── 3. 绘制坐标系尺寸必须回算（不等于入参 CSS 尺寸） ──────────────
    // 取整一旦发生，绘制坐标系宽就不再等于请求的 CSS 宽。它必须等于
    // physical/dpr，否则顶点坐标映射到物理像素时会带半像素偏差。
    {
        const dpr = 1.5;
        const css = 1000.5;
        const target = rasterize(fakeCanvas(), css, 100, dpr);
        // round(1000.5 × 1.5) = round(1500.75) = 1501 → 1501 / 1.5 = 1000.6667
        assertEqual(target.physicalWidth, 1501, "physical width");
        assertTrue(
            Math.abs(target.cssWidthPx - target.physicalWidth / dpr) < 1e-9,
            "cssWidthPx must be the back-calculated physical/dpr",
        );
        assertTrue(
            Math.abs(target.cssWidthPx - css) > 1e-9,
            "back-calculated width must differ from the requested CSS width here",
        );
        // 用回算值当 u_resolution：CSS 坐标 cssWidthPx（画布右缘）必须精确映射到
        // physicalWidth 个物理像素。
        const physicalAtRightEdge =
            (target.cssWidthPx / target.resolutionWidth) * target.physicalWidth;
        assertTrue(
            Math.abs(physicalAtRightEdge - target.physicalWidth) < 1e-6,
            "resolution maps draw-space width exactly onto the backing store",
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

    // ── 6. 清屏契约：必须覆盖整个物理 backing store ──────────────────
    // round 向上取整（css*dpr 带 0.5 尾数）时，CSS 尺寸清屏只覆盖 css*dpr 行，
    // 底部 0~0.5 物理行永远不被清除 → 贴底绘制内容形成永久残影。
    // clearCanvasPhysical 必须在单位变换下按 physical 尺寸清除。
    {
        const requestedCss = 101; // 101 × 1.5 = 151.5 → 152（向上取整，留下 0.5 行尾差）
        const target = rasterize(fakeCanvas(), requestedCss, requestedCss, 1.5);
        assertTrue(
            target.physicalHeight > requestedCss * 1.5,
            "chosen size must have a residue tail row (round-up)",
        );

        const calls: Array<readonly unknown[]> = [];
        const ctx = {
            save(): void {
                calls.push(["save"]);
            },
            restore(): void {
                calls.push(["restore"]);
            },
            setTransform(...args: number[]): void {
                calls.push(["setTransform", ...args]);
            },
            clearRect(...args: number[]): void {
                calls.push(["clearRect", ...args]);
            },
        } as unknown as CanvasRenderingContext2D;

        clearCanvasPhysical(ctx, target);
        const clear = calls.find((c) => c[0] === "clearRect");
        assertTrue(Boolean(clear), "clearRect called");
        assertEqual(clear?.[3], target.physicalWidth, "clear width = physicalWidth");
        assertEqual(clear?.[4], target.physicalHeight, "clear height = physicalHeight");

        // 清屏前必须复位为单位变换（否则 physical 尺寸再乘 dpr 会越界清除；
        // 反方向，CSS 尺寸下的 dpr 变换则清不到尾行）。
        const identityIndex = calls.findIndex(
            (c) =>
                c[0] === "setTransform" &&
                c[1] === 1 &&
                c[2] === 0 &&
                c[3] === 0 &&
                c[4] === 1 &&
                c[5] === 0 &&
                c[6] === 0,
        );
        const clearIndex = calls.findIndex((c) => c[0] === "clearRect");
        assertTrue(identityIndex >= 0, "transform reset to identity before clear");
        assertTrue(clearIndex > identityIndex, "clear happens under identity transform");

        // 回归守护：按「请求的 CSS 尺寸」清屏必须能被识别为错误（清不到尾行）。
        const cssClearActive = calls.some(
            (c) => c[0] === "clearRect" && (c[3] === requestedCss || c[4] === requestedCss),
        );
        assertTrue(!cssClearActive, "must not clear by CSS dimensions (leaves the tail row dirty)");
    }

    // ── 7. CSS 尺寸回写必须幂等：分数 dpr 下不得每帧重写样式 ──────────
    // 1001 × 1.5 = 1501.5 → 1502 物理像素；1502 / 1.5 = 1001.3333…（无限小数），
    // 浏览器回读会规范化成 "1001.33px"。若拿回读值与期望值比较，容差再小也会
    // 判定"变了"，于是每帧重写 `style.width` 并触发样式重算。
    {
        const { canvas, writes } = countingCanvas();
        const first = rasterize(canvas, 1001, 100, 1.5);
        assertEqual(writes(), 2, "first rasterize writes both edges exactly once");
        assertTrue(
            Math.abs(first.cssWidthPx * 1.5 - canvas.width) < 1e-9,
            "written style still maps 1:1 onto the backing store",
        );

        // 重复光栅化（例如每帧 repaint）不得再写样式。
        for (let i = 0; i < 5; i += 1) rasterize(canvas, 1001, 100, 1.5);
        assertEqual(writes(), 2, "repeat rasterize must not rewrite the CSS size");

        // 尺寸真变了才写，且**只写变化的那一条边**（高度未变 → 不写）。
        rasterize(canvas, 1002, 100, 1.5);
        assertEqual(writes(), 3, "a changed width writes once, unchanged height does not");
    }

    // ── 8. fitCssPxToDevicePx：吸附到整数物理像素，且与 rasterize 逐值一致 ──
    {
        for (const dpr of [1, 1.25, 1.5, 1.75, 2, 3]) {
            for (const css of [1, 100, 333.4, 1000.5, 1001]) {
                const snapped = fitCssPxToDevicePx(css, dpr);
                assertTrue(
                    Math.abs(snapped * dpr - Math.round(snapped * dpr)) < 1e-9,
                    `snapped × dpr is a whole device pixel (css=${css}, dpr=${dpr})`,
                );
                assertTrue(
                    Math.abs(snapped - rasterize(fakeCanvas(), css, 1, dpr).cssWidthPx) < 1e-9,
                    `agrees with rasterize (css=${css}, dpr=${dpr})`,
                );
            }
        }
        // 非法 / 退化输入回退到至少 1 个物理像素对应的 CSS 尺寸。
        assertEqual(fitCssPxToDevicePx(Number.NaN, 2), 1, "NaN css falls back");
        assertEqual(fitCssPxToDevicePx(0, 2), 1, "zero css falls back");
        assertTrue(Number.isFinite(fitCssPxToDevicePx(100, 0)), "dpr 0 falls back");
    }
});
