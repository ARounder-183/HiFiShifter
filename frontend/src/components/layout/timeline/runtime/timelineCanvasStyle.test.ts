/**
 * timelineCanvasStyle.test.ts - buildTimelineClipVisualStyle 单元测试。
 *
 * 主要内容：
 * - 控件可见性（gain knob / mute / formant / name / playbackRate）
 * - 文字 / badge 颜色字段产出
 * - 选中态与非选中态视觉差异（2026-06-30 Logic 风格重做后新增断言）
 * - formatPlaybackRateLabel 独立单元
 *
 * 与其他模块的关系：
 * - 通过 npx tsx 运行，作为 timeline canvas 视觉回归的最小防线。
 */
import {
    buildTimelineClipVisualStyle,
    computeTimelineFadeShadeRange,
    formatPlaybackRateLabel,
} from "./timelineCanvasStyle.js";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    const actualJson = JSON.stringify(actual);
    const expectedJson = JSON.stringify(expected);
    if (actualJson !== expectedJson) {
        throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
    }
}

function assertNotEqual(actual: unknown, expected: unknown, label: string): void {
    const actualJson = JSON.stringify(actual);
    const expectedJson = JSON.stringify(expected);
    if (actualJson === expectedJson) {
        throw new Error(`${label}: expected NOT to equal ${expectedJson}, but it did`);
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
assertEqual(style.showPlaybackRate, true, "playback rate shows on sufficiently wide clips");
assertEqual(
    compactStyle.showPlaybackRate,
    false,
    "playback rate hides before overlapping controls",
);
assertEqual(style.muteBadgeFill.startsWith("rgba("), true, "mute badge fill is resolved");
assertEqual(style.gainKnobIndicator.startsWith("rgba("), true, "gain knob indicator is resolved");
assertEqual(style.leadingControlsWidth, 80, "leading controls reserve prevents title overlap");
assertEqual(style.muteBadgeWidth, 20, "mute badge is enlarged");
assertEqual(style.formantBadgeWidth, 20, "formant badge matches mute width");
assertEqual(style.gainKnobRadius, 7, "gain knob is enlarged");
assertEqual(style.gainKnobCenterOffsetX, 15, "gain knob sits at the far left of the header");

// ── Logic Pro 风格新增断言 ─────────────────────────────────────────────
// accent bar：始终输出且宽度固定为 3px，承担色相识别。
assertEqual(style.accentBarWidthPx, 3, "accent bar reserves 3px width");
assertEqual(style.accentBarFill.startsWith("rgba("), true, "accent bar uses rgba color");
// 选中态视觉必须有区分（这是用户可感知差异的最低要求）：
assertNotEqual(
    selectedStyle.bodyFill,
    style.bodyFill,
    "selected body brightens to indicate selection",
);
assertNotEqual(
    selectedStyle.borderStroke,
    style.borderStroke,
    "selected border switches to saturated accent",
);
// accent bar 颜色不依赖选中态变化（保持识别一致性）：
assertEqual(
    selectedStyle.accentBarFill,
    style.accentBarFill,
    "accent bar color is selection-independent",
);
// header 分隔线必须输出，且采用半透明白系而非硬黑：
assertEqual(
    style.headerSeparatorFill.startsWith("rgba("),
    true,
    "header separator fill is resolved",
);
assertEqual(
    style.headerSeparatorFill.includes("255, 255, 255"),
    true,
    "header separator uses translucent white instead of hard black",
);

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
assertEqual(formatPlaybackRateLabel(1.5), "x1.5", "single decimal preserved without trailing 0");
assertEqual(formatPlaybackRateLabel(1.23), "x1.23", "two decimals preserved");
assertEqual(formatPlaybackRateLabel(0.85), "x0.85", "rates below 1 keep both decimals");
assertEqual(formatPlaybackRateLabel(2), "x2", "integer rates collapse to bare number");
assertEqual(formatPlaybackRateLabel(0), "x1", "non-positive rates fall back to x1");
assertEqual(formatPlaybackRateLabel(NaN), "x1", "non-finite rates fall back to x1");

console.log("timelineCanvasStyle checks passed");
