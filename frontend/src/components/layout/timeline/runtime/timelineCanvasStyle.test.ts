import { test } from "vitest";

import {
    buildTimelineClipVisualStyle,
    computeTimelineFadeShadeRange,
    formatPlaybackRateLabel,
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
    assertEqual(selectedStyle.headerFill, style.headerFill, "selected header keeps default visual");
    assertEqual(selectedStyle.bodyFill, style.bodyFill, "selected body keeps default visual");
    // Selected 边框优先读 CSS 变量 --qt-clip-selected-border；无 DOM 环境
    // （Node 测试）下回退为轨道色的全不透明变体 —— 与非选中的 0.74 有意区分。
    assertEqual(
        selectedStyle.borderStroke,
        "rgba(218, 129, 47, 1)",
        "selected border falls back to full-alpha track color without CSS variables",
    );
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

    console.log("timelineCanvasStyle checks passed");
});
