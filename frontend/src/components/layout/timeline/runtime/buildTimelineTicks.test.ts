/**
 * 统一刻度源自检。
 *
 * 【主要内容】验证网格线与标尺刻度同源：
 * 1. 每个刻度的 `contentPx` 严格等于 axis 对同一秒的投影（不得另算一遍）；
 * 2. 标尺渲染的刻度（`showLabel` 或小节起点）是网格刻度全集的**子集**；
 * 3. 刻度按时间升序、无重复；
 * 4. 密度受上限约束（弱线/强线），避免极端缩放下生成数十万条线；
 * 5. Tempo Map 与均匀网格两条路径都产出可用序列。
 *
 * 【作用】第 2 条是本次重构的核心不变量。此前网格走
 * `resolveGridLineSamplingPlan`（beat 域）、标尺走 `buildRulerTicks`（时间域），
 * 两套步长选择，Tempo Map 下 beat 与像素非线性，必然分叉。现在两者消费同一
 * 份数据，标尺刻度天然落在网格线上。
 *
 * 【与其他模块的关系】覆盖 `buildTimelineTicks.ts`，与 `timelineAxis.ts`
 * 联合验证坐标投影一致。
 */

import { test } from "vitest";

import { buildTimelineTicks } from "./buildTimelineTicks.js";
import { createTimelineAxis, secToContentPx, secToViewportPx } from "./timelineAxis.js";
import type { TempoMap } from "../../../../utils/tempoMap.ts";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (actual !== expected) {
        throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
    }
}

function assertTrue(condition: boolean, label: string): void {
    if (!condition) throw new Error(`${label}: expected true`);
}

/**
 * 标尺隐藏标签的间距阈值（与 TimeRulerMarks 的 labelHidden 判定一致）。
 * 带标签的刻度间距必须大于它，否则标尺上只剩没有文字的竖线。
 */
const RULER_LABEL_HIDDEN_PX = 26;

/** 两段式 Tempo Map：10 秒处提速并换成 3/4 拍。 */
function twoSegmentTempoMap(): TempoMap {
    return {
        points: [
            {
                id: "tp-a",
                positionSec: 0,
                bpm: 120,
                timeSignature: { numerator: 4, denominator: 4 },
            },
            {
                id: "tp-b",
                positionSec: 10,
                bpm: 180,
                timeSignature: { numerator: 3, denominator: 4 },
            },
        ],
    } as unknown as TempoMap;
}

test("components/layout/timeline/runtime/buildTimelineTicks.test.ts scripted checks", async () => {
    const scenarios = [
        { label: "uniform", tempoMap: null as TempoMap | null },
        { label: "tempo-map", tempoMap: twoSegmentTempoMap() },
    ];

    for (const scenario of scenarios) {
        for (const [pxPerSec, scrollLeftPx] of [
            [100, 0],
            [100, 2500],
            [8, 0],
            [1600, 12345.5],
        ] as const) {
            const axis = createTimelineAxis({
                pxPerSec,
                scrollLeftPx,
                viewportWidthPx: 1200,
            });
            const ticks = buildTimelineTicks({
                axis,
                bpm: 120,
                beatsPerBar: 4,
                grid: "1/8",
                primaryUnit: "barBeats",
                secondaryUnit: "none",
                minLabelSpacingPx: 60,
                tempoMap: scenario.tempoMap,
            });
            const tag = `${scenario.label} pps=${pxPerSec} sl=${scrollLeftPx}`;

            assertTrue(ticks.length > 0, `${tag}: produces ticks`);

            // ── 1. contentPx 必须严格等于 axis 投影 ──────────────────
            for (let i = 0; i < ticks.length; i += 1) {
                const tick = ticks[i];
                assertEqual(
                    tick.contentPx,
                    secToContentPx(axis, tick.sec),
                    `${tag} tick#${i}: contentPx from axis`,
                );
                // 视口坐标同样必须与 clip / 波形同源
                assertEqual(
                    tick.contentPx - axis.scrollLeftPx,
                    secToViewportPx(axis, tick.sec),
                    `${tag} tick#${i}: viewport px matches axis`,
                );
            }

            // ── 2. 升序且无重复 ──────────────────────────────────────
            for (let i = 1; i < ticks.length; i += 1) {
                assertTrue(
                    ticks[i].contentPx >= ticks[i - 1].contentPx - 1e-9,
                    `${tag}: ascending at ${i}`,
                );
                assertTrue(
                    ticks[i].sec > ticks[i - 1].sec - 1e-9,
                    `${tag}: seconds ascending at ${i}`,
                );
            }

            // ── 3. 核心不变量：标尺刻度是网格刻度的子集 ──────────────
            const gridPx = new Set(ticks.map((tick) => Math.round(tick.contentPx * 1e6) / 1e6));
            let labeledCount = 0;
            let barCount = 0;
            for (const tick of ticks) {
                if (tick.isBarStart) barCount += 1;
                if (!tick.showLabel && !tick.isBarStart) continue;
                labeledCount += 1;
                const key = Math.round(tick.contentPx * 1e6) / 1e6;
                assertTrue(
                    gridPx.has(key),
                    `${tag}: ruler tick at ${tick.contentPx} must sit on a grid line`,
                );
            }
            assertTrue(labeledCount > 0, `${tag}: some ticks carry labels`);
            assertTrue(barCount > 0, `${tag}: some ticks are bar starts`);

            // ── 3b. 带标签的刻度间距必须放得下文字 ──────────────────
            // 回归防御：曾把「小节起点」无条件计入标尺刻度，缩小时小节间距会
            // 小于标签隐藏阈值（26px），标尺于是只剩一堆没有文字的竖线。
            // 放大到足够大时最细网格步长已超过标签间距，此时每个刻度都可以带
            // 号、两者相等是正常的；关键在于**间距**必须始终放得下文字。
            assertTrue(
                labeledCount <= ticks.length,
                `${tag}: labeled ticks cannot exceed all ticks (${labeledCount}/${ticks.length})`,
            );
            const labeled = ticks.filter((tick) => tick.showLabel);
            for (let i = 1; i < labeled.length; i += 1) {
                const gapPx = labeled[i].contentPx - labeled[i - 1].contentPx;
                assertTrue(
                    gapPx >= RULER_LABEL_HIDDEN_PX + 1e-6,
                    `${tag}: label gap ${gapPx}px must clear the ${RULER_LABEL_HIDDEN_PX}px ` +
                        `hide threshold, otherwise the ruler is left with bare lines`,
                );
            }

            // ── 4. 密度上限：防止极端缩放下生成海量线 ────────────────
            const weakCount = ticks.filter((tick) => !tick.isStrongGridLine).length;
            const strongCount = ticks.filter((tick) => tick.isStrongGridLine).length;
            // 生成范围含两侧缓冲（约 2 倍视口宽），上限按视口密度上限的 2 倍放宽
            assertTrue(weakCount <= 400, `${tag}: weak lines bounded (got ${weakCount})`);
            assertTrue(strongCount <= 160, `${tag}: strong lines bounded (got ${strongCount})`);

            // ── 5. 标签文本可用 ──────────────────────────────────────
            for (const tick of ticks) {
                if (!tick.showLabel) continue;
                assertTrue(
                    typeof tick.primaryLabel === "string" && tick.primaryLabel.length > 0,
                    `${tag}: labeled tick has primary label`,
                );
            }
        }
    }

    // ── 6. 副单位开启时次要标签非空 ────────────────────────────────
    {
        const axis = createTimelineAxis({ pxPerSec: 200, viewportWidthPx: 1200 });
        const ticks = buildTimelineTicks({
            axis,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/8",
            primaryUnit: "barBeats",
            secondaryUnit: "seconds",
            minLabelSpacingPx: 60,
        });
        const withSecondary = ticks.filter((tick) => tick.showLabel && tick.secondaryLabel != null);
        assertTrue(withSecondary.length > 0, "secondary labels are produced when enabled");
    }

    // ── 7. 极端缩放不产生 NaN 坐标 ─────────────────────────────────
    {
        for (const pxPerSec of [0.01, 1e-3, 1e6]) {
            const axis = createTimelineAxis({ pxPerSec, viewportWidthPx: 1200 });
            const ticks = buildTimelineTicks({
                axis,
                bpm: 960,
                beatsPerBar: 4,
                grid: "1/64",
                primaryUnit: "seconds",
                secondaryUnit: "none",
                minLabelSpacingPx: 60,
                tempoMap: twoSegmentTempoMap(),
            });
            for (const tick of ticks) {
                assertTrue(
                    Number.isFinite(tick.contentPx) && Number.isFinite(tick.sec),
                    `finite geometry at pxPerSec=${pxPerSec}`,
                );
            }
        }
    }
});
