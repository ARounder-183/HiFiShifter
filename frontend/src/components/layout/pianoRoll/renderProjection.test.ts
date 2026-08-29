/**
 * Piano Roll 曲线 x 换算与统一投影的等价性快照。
 *
 * 【主要内容】对参数编辑器的曲线 x 公式
 * `timeToPixel(t, scrollLeft / pxPerSec, w / pxPerSec, w)` 与统一投影
 * `secToViewportPx(axis, t)` 做大规模随机比对。
 *
 * 【作用】P3 把 `render.ts` 的三套 x 公式收敛到 `TimelineAxis` 之前的**安全网**。
 * 二者在代数上恒等：
 *
 *   ((t − scrollLeft/p) / (w/p)) · w  ≡  (t − scrollLeft/p) · p  ≡  t·p − scrollLeft
 *
 * 但旧式「先除后乘」会在中间步骤引入 IEEE754 舍入差，无法靠推导保证位级一致。
 * 本测试用随机参数量化该差异：若某组参数下偏差超出容差，说明替换会引入可见的
 * 曲线位移，必须先定位再动手，而不是直接替换。
 *
 * 这也是设计文档「P3 前先对 render.ts 的曲线 x 换算做快照测试」要求的产物。
 *
 * 【与其他模块的关系】
 * - 被测：`pianoRoll/utils.ts` 的 `timeToPixel`（旧公式）。
 * - 基准：`timeline/runtime/timelineAxis.ts` 的 `secToViewportPx`（新投影）。
 */

import { test } from "vitest";

import { timeToPixel } from "./utils.js";
import { createTimelineAxis, secToViewportPx } from "../timeline/runtime/timelineAxis.js";

/** 相对容差：像素坐标可达 1e7 量级，纯绝对容差在大缩放下会被机器精度淹没。 */
const RELATIVE_TOLERANCE = 1e-9;

test("components/layout/pianoRoll/renderProjection.test.ts scripted checks", async () => {
    let checks = 0;

    function assertNear(actual: number, expected: number, label: string): void {
        checks += 1;
        const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
        if (Math.abs(actual - expected) > RELATIVE_TOLERANCE * scale) {
            throw new Error(
                `${label}: expected ${expected}, received ${actual} ` +
                    `(diff ${Math.abs(actual - expected)}, tolerance ${RELATIVE_TOLERANCE * scale})`,
            );
        }
    }

    // 确定性 PRNG（mulberry32）：属性测试必须可复现，失败时能定位到具体迭代。
    let seed = 0x2f6e2b1;
    const rand = (): number => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    /** [lo, hi] 区间内的随机实数（含若干刻意取到端点的值）。 */
    const between = (lo: number, hi: number): number => lo + rand() * (hi - lo);

    for (let i = 0; i < 20000; i += 1) {
        const pxPerSec = between(1, 2000);
        const scrollLeftPx = between(0, 1e6);
        const canvasWidthPx = between(1, 4000);
        const tSec = between(0, 1e4);

        // ── 旧公式：`render.ts` 当前构造可见区间后交给 timeToPixel ──
        const visibleStartSec = scrollLeftPx / Math.max(1e-9, pxPerSec);
        const visibleDurSec = canvasWidthPx / Math.max(1e-9, pxPerSec);
        const legacyX = timeToPixel(tSec, visibleStartSec, visibleDurSec, canvasWidthPx);

        // ── 新投影：统一 axis ──
        const axis = createTimelineAxis({
            pxPerSec,
            scrollLeftPx,
            viewportWidthPx: canvasWidthPx,
        });
        const axisX = secToViewportPx(axis, tSec);

        assertNear(
            axisX,
            legacyX,
            `iteration ${i}: pxPerSec=${pxPerSec} scrollLeft=${scrollLeftPx} ` +
                `w=${canvasWidthPx} t=${tSec}`,
        );
    }

    // ── 边界：极小缩放、零滚动、零时刻 ──────────────────────────────
    // 这些是旧公式 `Math.max(1e-9, …)` 兜底真正生效的场合，必须一并等价。
    const edgeCases: Array<[number, number, number, number]> = [
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1e-9, 0, 1000, 5],
        [2000, 0, 4000, 0],
        [2000, 1e6, 4000, 1e4],
        [0.5, 12345.678, 800, 987.654],
    ];
    for (const [pxPerSec, scrollLeftPx, canvasWidthPx, tSec] of edgeCases) {
        const visibleStartSec = scrollLeftPx / Math.max(1e-9, pxPerSec);
        const visibleDurSec = canvasWidthPx / Math.max(1e-9, pxPerSec);
        const legacyX = timeToPixel(tSec, visibleStartSec, visibleDurSec, canvasWidthPx);
        const axis = createTimelineAxis({ pxPerSec, scrollLeftPx, viewportWidthPx: canvasWidthPx });
        assertNear(
            secToViewportPx(axis, tSec),
            legacyX,
            `edge: pxPerSec=${pxPerSec} scrollLeft=${scrollLeftPx} w=${canvasWidthPx} t=${tSec}`,
        );
    }

    void checks;
});
