import { test } from "vitest";

import {
    buildTimelineClipVisualStyle,
    computeTimelineFadeShadeRange,
    formatPlaybackRateLabel,
    parsePlaybackRateInput,
} from "./timelineCanvasStyle.js";

test("components/layout/timeline/runtime/timelineCanvasStyle.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const style = buildTimelineClipVisualStyle({
        widthPx: 160,
        trackColor: "#ff7a00",
        selected: false,
        muted: false,
        gain: 1,
        playbackRate: 1,
        name: "Lead Vocal Very Long Name For Playback Rate Header",
    });
    const compactStyle = buildTimelineClipVisualStyle({
        widthPx: 96,
        trackColor: "#ff7a00",
        selected: false,
        muted: false,
        gain: 1,
        playbackRate: 1,
        name: "Lead Vocal Very Long Name For Playback Rate Header",
    });
    const selectedStyle = buildTimelineClipVisualStyle({
        widthPx: 160,
        trackColor: "#ff7a00",
        selected: true,
        muted: false,
        gain: 1,
        playbackRate: 1,
        name: "Lead Vocal Very Long Name For Playback Rate Header",
    });
    const stretchedStyle = buildTimelineClipVisualStyle({
        widthPx: 160,
        trackColor: "#ff7a00",
        selected: false,
        muted: false,
        gain: 1,
        playbackRate: 1.25,
        name: "Lead Vocal Very Long Name For Playback Rate Header",
    });

    assertEqual(style.showGainKnob, true, "gain knob visible");
    assertEqual(style.showGainLabel, true, "gain label visible");
    assertEqual(style.showName, true, "name visible");
    assertEqual(style.showMuteBadge, true, "mute badge visible");
    assertEqual(style.headerFill.startsWith("rgba("), true, "header uses mixed rgba color");
    assertEqual(style.bodyFill.startsWith("rgba("), true, "body uses mixed rgba color");
    assertEqual(style.displayName.length > 0, true, "name display is produced");
    assertEqual(style.muteBadgeLabel, "M", "mute badge uses M label");
    assertEqual(style.formantBadgeLabel, "F", "formant badge uses F label");
    assertEqual(style.gainKnobAngleDeg, 0, "unity gain knob stays centered");
    assertEqual(style.playbackRateLabel, "x1", "playback rate label is formatted (unity)");
    assertEqual(
        stretchedStyle.playbackRateLabel,
        "x1.25",
        "playback rate label reflects the stretched rate",
    );
    assertEqual(style.showPlaybackRate, true, "playback rate shows on sufficiently wide clips");
    assertEqual(
        compactStyle.showPlaybackRate,
        false,
        "playback rate hides before overlapping controls",
    );
    assertEqual(style.muteBadgeFill.startsWith("rgba("), true, "mute badge fill is resolved");
    assertEqual(
        style.gainKnobIndicator.startsWith("rgba("),
        true,
        "gain knob indicator is resolved",
    );
    assertEqual(style.leadingControlsWidth, 80, "leading controls reserve prevents title overlap");
    assertEqual(style.muteBadgeWidth, 20, "mute badge is enlarged");
    assertEqual(style.formantBadgeWidth, 20, "formant badge matches mute width");
    assertEqual(style.gainKnobRadius, 7, "gain knob is enlarged");
    assertEqual(style.gainKnobCenterOffsetX, 15, "gain knob sits at the far left of the header");
    // 选中 = 整块提亮（header + body 一起），不再保持默认色。
    assertEqual(
        selectedStyle.headerFill === style.headerFill,
        false,
        "selected header is brightened (selection expressed by lightness, not border)",
    );
    // 选中 = 白色 2px 描边 + 色块提亮；未选中 = 淡收边 1px。
    assertEqual(
        selectedStyle.borderStroke,
        "rgba(255, 255, 255, 0.6)",
        "selected clip uses a subdued white 2px stroke",
    );
    assertEqual(selectedStyle.borderLineWidth, 2, "selected border is 2px");
    assertEqual(style.borderLineWidth, 1, "unselected border is 1px");
    {
        const parseLum = (fill: string): number => {
            const m = fill.match(/rgba\((\d+), (\d+), (\d+),/);
            if (!m) throw new Error(`unparseable fill: ${fill}`);
            return (Number(m[1]) * 0.299 + Number(m[2]) * 0.587 + Number(m[3]) * 0.114) / 255;
        };
        const selectedLum = parseLum(selectedStyle.bodyFill);
        const normalLum = parseLum(style.bodyFill);
        if (selectedLum <= normalLum) {
            throw new Error(
                `selected clip must be brighter than normal (selected=${selectedLum.toFixed(3)}, normal=${normalLum.toFixed(3)})`,
            );
        }
    }
    assertEqual(selectedStyle.textFill, style.textFill, "selected text keeps default visual");

    assertEqual(
        computeTimelineFadeShadeRange({
            widthPx: 200,
            fadeInPx: 40,
            fadeOutPx: 30,
        }),
        {
            startPx: 40,
            endPx: 170,
        },
        "shade range sits outside fade areas",
    );

    // ── 色块归一化：极端轨道色也必须落在安全亮度区间 ────────────────────────
    // 整块用色的前提：无论用户挑了多刺眼/多暗的轨道色，Clip 色块的感知亮度
    // 都被 HSL 归一化收敛到**明亮带**——深色文字/深色波形在色块上永远有对比，
    // 色块对深色轨道背景也永远有明度差（亮块 + 深前景是本方案的核心）。
    {
        const parseLuminance = (fill: string): number => {
            const m = fill.match(/rgba\((\d+), (\d+), (\d+),/);
            if (!m) throw new Error(`unparseable fill: ${fill}`);
            const [, rs, gs, bs] = m;
            return (Number(rs) * 0.299 + Number(gs) * 0.587 + Number(bs) * 0.114) / 255;
        };
        for (const color of ["#ff0000", "#00ff00", "#0000ff", "#ffffff", "#000000", "#808080"]) {
            const extreme = buildTimelineClipVisualStyle({
                widthPx: 160,
                trackColor: color,
                selected: false,
                muted: false,
                gain: 1,
                playbackRate: 1,
                name: "x",
            });
            const lum = parseLuminance(extreme.bodyFill);
            if (lum < 0.35 || lum > 0.65) {
                throw new Error(
                    `trackColor ${color}: clip block luminance ${lum.toFixed(3)} outside the REAPER band [0.35, 0.65]`,
                );
            }
        }
    }

    // ── formatPlaybackRateLabel ───────────────────────────────────────────────
    assertEqual(formatPlaybackRateLabel(1), "x1", "unity rate has no fractional part");
    assertEqual(
        formatPlaybackRateLabel(1.5),
        "x1.5",
        "single decimal preserved without trailing 0",
    );
    assertEqual(formatPlaybackRateLabel(1.23), "x1.23", "two decimals preserved");
    assertEqual(formatPlaybackRateLabel(0.85), "x0.85", "rates below 1 keep both decimals");
    assertEqual(formatPlaybackRateLabel(2), "x2", "integer rates collapse to bare number");
    assertEqual(formatPlaybackRateLabel(0), "x1", "non-positive rates fall back to x1");
    assertEqual(formatPlaybackRateLabel(NaN), "x1", "non-finite rates fall back to x1");
});


test("parsePlaybackRateInput accepts plain, prefixed and percent forms", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const a = JSON.stringify(actual);
        const e = JSON.stringify(expected);
        if (a !== e) throw new Error(`${label}: expected ${e}, received ${a}`);
    }
    const check = (raw: string, expected: number | null, label: string) =>
        assertEqual(parsePlaybackRateInput(raw), expected, label);
    check("1.5", 1.5, "plain decimal");
    check(" 2 ", 2, "whitespace trimmed");
    check("x1.5", 1.5, "x prefix");
    check("×0.5", 0.5, "fullwidth multiply prefix");
    check("1.5x", 1.5, "trailing x");
    check("150%", 1.5, "percent form");
    check("50%", 0.5, "percent below 100");
});

test("parsePlaybackRateInput rejects invalid input", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const a = JSON.stringify(actual);
        const e = JSON.stringify(expected);
        if (a !== e) throw new Error(`${label}: expected ${e}, received ${a}`);
    }
    const check = (raw: string, label: string) =>
        assertEqual(parsePlaybackRateInput(raw), null, label);
    check("", "empty string");
    check("abc", "non numeric");
    check("-1", "negative");
    check("0", "zero");
    check("NaN", "NaN literal");
});
