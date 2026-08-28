import { test } from "vitest";

import {
    beatFromSec,
    buildRulerTicks,
    formatBarBeatsLabel,
    formatBarDivisionsLabel,
    formatClockLabel,
    formatCursorTime,
    formatSecondsCursor,
    formatSecondsRuler,
    formatTempoBarBeatsLabel,
    formatTempoBarDivisionsLabel,
    gridDivisionsPerBar,
    rulerStepCandidates,
    selectRulerStep,
    secFromBeat,
} from "./timeFormat.ts";

test("components/layout/timeline/timeFormat.test.ts scripted checks", async () => {
    let checks = 0;

    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        checks += 1;
        if (actual !== expected) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    function assertNear(actual: number, expected: number, label: string): void {
        checks += 1;
        if (Math.abs(actual - expected) > 1e-9) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    // ── 小节.节拍.小单位 ─────────────────────────────────────────────
    assertEqual(formatBarBeatsLabel(0, 4), "1.1", "bar start");
    assertEqual(formatBarBeatsLabel(2, 4), "1.3", "half note");
    assertEqual(formatBarBeatsLabel(4, 4), "2.1", "second bar");
    assertEqual(formatBarBeatsLabel(0, 4, "cursor"), "1.1.000", "cursor bar start aligned");
    assertEqual(formatBarBeatsLabel(2, 4, "cursor"), "1.3.000", "cursor beat boundary aligned");
    assertEqual(formatBarBeatsLabel(0.5, 4), "1.1.500", "eighth note exact");
    assertEqual(formatBarBeatsLabel(0.25, 4), "1.1.250", "sixteenth note exact");
    assertEqual(formatBarBeatsLabel(1.5, 4), "1.2.500", "beat + eighth");
    assertEqual(formatBarBeatsLabel(1 / 3, 4), "1.1.333..", "triplet inexact marker");
    assertEqual(formatBarBeatsLabel(0.0625, 4), "1.1.062..", "64th truncated with marker");
    assertEqual(
        formatBarBeatsLabel(1.1166666666, 4),
        "1.2.116..",
        "subdivision truncates to three digits",
    );
    assertEqual(
        formatBarBeatsLabel(0.875, 4, "cursor"),
        "1.1.875",
        "cursor keeps exact thousandths",
    );
    assertEqual(formatBarBeatsLabel(0.0625, 4, "cursor"), "1.1.062", "cursor 64th truncation");
    assertEqual(
        formatBarBeatsLabel(1.1166666666, 4, "cursor"),
        "1.2.116",
        "cursor truncates to three digits",
    );
    assertEqual(formatBarBeatsLabel(2.5, 3, "cursor"), "1.3.500", "custom beats per bar");
    assertEqual(formatBarBeatsLabel(3.9999999999, 4), "2.1", "ruler carry to next bar");
    assertEqual(formatBarBeatsLabel(3.9999999, 4), "1.4.999..", "ruler truncates near-bar value");
    assertEqual(
        formatBarBeatsLabel(3.9996, 4, "cursor"),
        "1.4.999",
        "cursor truncates near-bar value",
    );

    // ── 小节.切分 ────────────────────────────────────────────────────
    assertEqual(
        formatBarDivisionsLabel(0, { bpm: 120, beatsPerBar: 4, grid: "1/32" }),
        "1.1/32",
        "division bar start",
    );
    assertEqual(
        formatBarDivisionsLabel(0.125 * 16, { bpm: 120, beatsPerBar: 4, grid: "1/32" }),
        "1.17/32",
        "division 17 of 32",
    );
    assertEqual(
        formatBarDivisionsLabel(1 / 3, { bpm: 120, beatsPerBar: 4, grid: "1/8t" }),
        "1.2/12",
        "triplet division",
    );
    assertEqual(
        formatBarDivisionsLabel(1 / 6, { bpm: 120, beatsPerBar: 4, grid: "1/16t" }),
        "1.2/24",
        "16th triplet division",
    );
    assertNear(gridDivisionsPerBar("1/4d", 4), 8 / 3, "dotted grid fractional divisions");
    assertEqual(
        formatBarDivisionsLabel(1.5, { bpm: 120, beatsPerBar: 4, grid: "1/4d" }),
        "1.2/2.6667",
        "dotted grid label",
    );
    assertEqual(
        formatBarDivisionsLabel(1.4, { bpm: 120, beatsPerBar: 4, grid: "1/4" }),
        "1.2/4",
        "division index truncates instead of rounding",
    );

    // ── 秒 ──────────────────────────────────────────────────────────
    assertEqual(formatSecondsRuler(2), "2", "seconds exact integer");
    assertEqual(formatSecondsRuler(1.23), "1.23", "seconds exact decimal");
    assertEqual(formatSecondsRuler(0.96875), "0.9687..", "seconds truncated inexact");
    assertEqual(formatSecondsRuler(0.9375), "0.9375", "seconds exact four places");
    assertEqual(formatSecondsCursor(0.96875), "0.969", "cursor seconds round");
    assertEqual(formatSecondsCursor(2), "2.000", "cursor seconds integer aligned");
    assertEqual(formatSecondsCursor(1.23), "1.230", "cursor seconds trailing zeros kept");

    // ── 时:分:秒.毫秒 ────────────────────────────────────────────────
    assertEqual(formatClockLabel(0), "0:0.000", "clock zero");
    assertEqual(formatClockLabel(283.75), "4:43.750", "clock omit hour");
    assertEqual(formatClockLabel(3883.75), "1:4:43.750", "clock with hour");
    assertEqual(formatClockLabel(59.9999), "0:59.999", "clock truncates ms");

    // ── 播放光标组合格式 ─────────────────────────────────────────────
    {
        const ctx = { bpm: 120, beatsPerBar: 4, grid: "1/32" };
        const both = formatCursorTime("barBeats", "clock", 0.875, ctx);
        assertEqual(both.combined, "1.2.750 / 0:0.875", "cursor combined");
        const onlyPrimary = formatCursorTime("barBeats", "none", 0.875, ctx);
        assertEqual(onlyPrimary.combined, "1.2.750", "cursor none secondary");
        const sameUnit = formatCursorTime("seconds", "seconds", 0.875, ctx);
        assertEqual(sameUnit.combined, "0.875", "cursor same unit hides secondary");
    }

    // ── 自适应刻度 ───────────────────────────────────────────────────
    assertEqual(
        selectRulerStep({ pxPerBeat: 75, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        2,
        "default zoom half note",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 120, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        1,
        "quarter note",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 240, grid: "1/8", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        0.5,
        "eighth note",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 480, grid: "1/16", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        0.25,
        "sixteenth note",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 960, grid: "1/32", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        0.125,
        "32nd note",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 2000, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        1,
        "grid caps precision at 1/4",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 2000, grid: "1/32", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        0.125,
        "grid caps precision at 1/32",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 2000, grid: "1/64", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        0.0625,
        "64th grid precision",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 2000, grid: "1/8t", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        1 / 3,
        "triplet grid precision",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 2000, grid: "1/16t", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        1 / 6,
        "16th triplet grid precision",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 40, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        4,
        "zoom out keeps bar when it still fits",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 8, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        16,
        "zoom out labels every 4 bars",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 4, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        32,
        "zoom out labels every 8 bars",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 2, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        64,
        "zoom out labels every 16 bars",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 75, grid: "1/4", beatsPerBar: 3, minLabelSpacingPx: 90 }),
        3,
        "3/4 zoom out labels every bar",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 120, grid: "1/4", beatsPerBar: 3, minLabelSpacingPx: 90 }),
        1,
        "3/4 zoom in labels every beat",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 240, grid: "1/8", beatsPerBar: 3, minLabelSpacingPx: 90 }),
        0.5,
        "3/4 eighth-note labels",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 100, grid: "1/4", beatsPerBar: 3, minLabelSpacingPx: 90 }),
        1,
        "3/4 never picks half-bar step",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 8, grid: "1/4", beatsPerBar: 3, minLabelSpacingPx: 90 }),
        12,
        "3/4 far zoom out labels every 4 bars",
    );
    assertEqual(
        selectRulerStep({ pxPerBeat: 0.25, grid: "1/4", beatsPerBar: 4, minLabelSpacingPx: 90 }),
        512,
        "zoom out keeps doubling beyond fixed ladder",
    );

    assertEqual(rulerStepCandidates("1/8t", 4)[0], 1 / 3, "triplet candidates include grid step");
    assertEqual(
        rulerStepCandidates("1/4d", 4).includes(1.5),
        false,
        "dotted candidates exclude non-bar-aligning step",
    );
    assertEqual(
        rulerStepCandidates("1/4d", 4).includes(2),
        true,
        "dotted candidates keep bar-aligning steps",
    );

    // ── 刻度生成 ─────────────────────────────────────────────────────
    {
        const ticks = buildRulerTicks({
            pxPerSec: 480,
            scrollLeft: 0,
            viewportWidth: 1000,
            projectSec: 8,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/8",
            primaryUnit: "barBeats",
            secondaryUnit: "clock",
            minLabelSpacingPx: 90,
        });
        assertEqual(ticks.length > 0, true, "ticks generated");
        assertEqual(ticks[0].beat, 0, "first tick at zero");
        assertEqual(ticks[0].primaryLabel, "1.1", "first tick bar label");
        assertEqual(ticks[0].secondaryLabel, "0:0.000", "first tick secondary label");
        const beatTick = ticks.find((t) => Math.abs(t.beat - 0.5) < 1e-9);
        assertEqual(Boolean(beatTick), true, "half beat tick exists");
        assertEqual(beatTick?.primaryLabel, "1.1.500", "half beat label");
        const barTick = ticks.find((t) => Math.abs(t.beat - 4) < 1e-9);
        assertEqual(barTick?.isBarStart, true, "bar start flagged");
        assertEqual(barTick?.primaryLabel, "2.1", "second bar label");
        // 副单位与主单位相同或“不使用”时，不生成副标签
        const single = buildRulerTicks({
            pxPerSec: 480,
            scrollLeft: 0,
            viewportWidth: 1000,
            projectSec: 8,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/8",
            primaryUnit: "barBeats",
            secondaryUnit: "barBeats",
            minLabelSpacingPx: 90,
        });
        assertEqual(single[0].secondaryLabel, null, "same unit hides secondary");

        // 缩小到小节间隔不足时，不再给每个小节补标签，而是每 2/4/8… 小节一个
        const sparse = buildRulerTicks({
            pxPerSec: 10,
            scrollLeft: 0,
            viewportWidth: 1000,
            projectSec: 50,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/4",
            primaryUnit: "barBeats",
            secondaryUnit: "none",
            minLabelSpacingPx: 90,
        });
        assertEqual(sparse[0].primaryLabel, "1.1", "sparse first label");
        assertEqual(
            sparse.map((t) => t.primaryLabel).join(","),
            "1.1,9.1,17.1,25.1",
            "sparse labels every 8 bars",
        );
        assertEqual(
            sparse.every((t) => t.isBarStart),
            true,
            "sparse ticks all on bar starts",
        );
        assertEqual(
            sparse.some((t) => t.primaryLabel === "2.1"),
            false,
            "no crowded per-bar labels",
        );

        // 每小节 3 拍时，标签必须均匀：要么每小节一个，要么每拍一个，绝不出现 1.3/2.2 交替。
        const triple = buildRulerTicks({
            pxPerSec: 150,
            scrollLeft: 0,
            viewportWidth: 1000,
            projectSec: 8,
            bpm: 120,
            beatsPerBar: 3,
            grid: "1/4",
            primaryUnit: "barBeats",
            secondaryUnit: "none",
            minLabelSpacingPx: 90,
        });
        assertEqual(
            triple.map((t) => t.primaryLabel).join(","),
            "1.1,2.1,3.1,4.1,5.1,6.1",
            "3/4 labels every bar uniformly",
        );

        const tripleBeats = buildRulerTicks({
            pxPerSec: 240,
            scrollLeft: 0,
            viewportWidth: 1000,
            projectSec: 8,
            bpm: 120,
            beatsPerBar: 3,
            grid: "1/4",
            primaryUnit: "barBeats",
            secondaryUnit: "none",
            minLabelSpacingPx: 90,
        });
        assertEqual(
            tripleBeats
                .slice(0, 7)
                .map((t) => t.primaryLabel)
                .join(","),
            "1.1,1.2,1.3,2.1,2.2,2.3,3.1",
            "3/4 labels every beat uniformly",
        );
    }

    assertNear(beatFromSec(0.5, 120), 1, "beatFromSec");
    assertNear(secFromBeat(1, 120), 0.5, "secFromBeat");

    // ── Tempo Map 标签进位（浮点误差不得产生 "4.2.1000"） ──────────
    assertEqual(
        formatTempoBarBeatsLabel({ bar: 4, beat: 2, sub: 0.9999999999 }, "ruler", 4),
        "4.3",
        "tempo label carries near-beat sub",
    );
    assertEqual(
        formatTempoBarBeatsLabel({ bar: 4, beat: 2, sub: 1 }, "ruler", 4),
        "4.3",
        "tempo label carries sub=1",
    );
    assertEqual(
        formatTempoBarBeatsLabel({ bar: 4, beat: 4, sub: 0.9999999999 }, "ruler", 4),
        "5.1",
        "tempo label carries across bar",
    );
    assertEqual(
        formatTempoBarBeatsLabel({ bar: 4, beat: 4, sub: 0.9999999999 }, "cursor", 4),
        "5.1.000",
        "tempo cursor carries across bar",
    );

    // ── Tempo Map 标尺：变化点后重新对齐（不得沿用旧均匀网格位置） ──
    {
        // 空工程 120bpm 4/4，在 7.25s（旧标尺 4.3.500）插入变化点（同 BPM）。
        const map = {
            points: [
                {
                    id: "p0",
                    positionSec: 0,
                    bpm: 120,
                    timeSignature: { numerator: 4, denominator: 4 },
                    scale: null,
                },
                {
                    id: "p1",
                    positionSec: 7.25,
                    bpm: 120,
                    timeSignature: { numerator: 4, denominator: 4 },
                    scale: null,
                },
            ],
        };
        const ticks = buildRulerTicks({
            pxPerSec: 480,
            scrollLeft: 0,
            viewportWidth: 4000,
            projectSec: 10,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/4",
            primaryUnit: "barBeats",
            secondaryUnit: "none",
            minLabelSpacingPx: 90,
            tempoMap: map,
        });
        const secs = ticks.map((t) => t.sec);
        for (const bad of [7.5, 8.0, 8.5]) {
            if (secs.some((s) => Math.abs(s - bad) < 1e-9)) {
                throw new Error(`tempo ruler: stale old-grid tick at ${bad}`);
            }
        }
        const at725 = ticks.find((t) => Math.abs(t.sec - 7.25) < 1e-9);
        assertEqual(at725?.primaryLabel, "5.1", "tempo ruler: segment start label");
        assertEqual(at725?.isBarStart, true, "tempo ruler: segment start is bar");
        const at775 = ticks.find((t) => Math.abs(t.sec - 7.75) < 1e-9);
        assertEqual(at775?.primaryLabel, "5.2", "tempo ruler: next beat at 7.75");
        assertEqual(at775?.isBarStart, false, "tempo ruler: beat tick not bar");
        const at925 = ticks.find((t) => Math.abs(t.sec - 9.25) < 1e-9);
        assertEqual(at925?.primaryLabel, "6.1", "tempo ruler: next bar at 9.25");
        assertEqual(at925?.isBarStart, true, "tempo ruler: bar tick flagged");
    }

    // ── Tempo Map 标尺：跟随之前的拍号继续按前段拍号分小节 ──
    {
        // 0s 起 3/4（每小节 1.5s @120），3s 处变化点“跟随之前的拍号” → 继续 3/4 对齐。
        const map = {
            points: [
                {
                    id: "p0",
                    positionSec: 0,
                    bpm: 120,
                    timeSignature: { numerator: 3, denominator: 4 },
                    scale: null,
                },
                { id: "p1", positionSec: 3, bpm: 120, timeSignature: null, scale: null },
            ],
        };
        const ticks = buildRulerTicks({
            pxPerSec: 480,
            scrollLeft: 0,
            viewportWidth: 4000,
            projectSec: 6,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/4",
            primaryUnit: "barBeats",
            secondaryUnit: "none",
            minLabelSpacingPx: 90,
            tempoMap: map,
        });
        const at45 = ticks.find((t) => Math.abs(t.sec - 4.5) < 1e-9);
        assertEqual(at45?.primaryLabel, "4.1", "tempo ruler follow-sig: 3/4 bar continues");
        assertEqual(at45?.isBarStart, true, "tempo ruler follow-sig: bar tick flagged");
    }

    // ── 回归：分数每小节拍数（3/8 = 1.5 拍/小节）──
    {
        // 切分数不得取整：3/8 小节在 1/8 网格下是 3 个切分，而不是 round(1.5)/0.5 = 4。
        assertNear(gridDivisionsPerBar("1/8", 1.5), 3, "fractional bpb: 3/8 with 1/8 grid → 3");
        assertNear(gridDivisionsPerBar("1/16", 1.5), 6, "fractional bpb: 3/8 with 1/16 grid → 6");
        assertNear(gridDivisionsPerBar("1/4", 4), 4, "integer bpb unchanged");
        // 切分标签：3/8 小节的第 2 拍（最后一个八分音符）→ 第 3 个切分 / 共 3 个。
        assertEqual(
            formatTempoBarDivisionsLabel({ bar: 1, beat: 2, sub: 0 }, 1.5, "1/8"),
            "1.3/3",
            "fractional bpb: division label 3/3",
        );
        // 小节.拍 标签按原始 bpb 进位：1.5 拍/小节中拍内余量接近 1 时（拍 2 末尾）
        // 必须进位到下一小节（取整为 2 也会进位，但这是对原始 bpb 路径的回归保护）。
        assertEqual(
            formatTempoBarBeatsLabel({ bar: 1, beat: 2, sub: 0.9999999995 }, "ruler", 1.5),
            "2.1",
            "fractional bpb: near-boundary sub carries to next bar",
        );
    }
});
