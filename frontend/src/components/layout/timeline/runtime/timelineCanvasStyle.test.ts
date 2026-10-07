import { test } from "vitest";

import {
    buildTimelineClipVisualStyle,
    computeTimelineFadeShadeRange,
    formatPlaybackRateLabel,
    parsePlaybackRateInput,
    timelineLaneBackgroundCss,
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

test("buildTimelineClipVisualStyle exposes label pixel widths", () => {
    function assert(condition: boolean, label: string): void {
        if (!condition) throw new Error(label);
    }
    const base = {
        widthPx: 300,
        trackColor: "#ff7a00",
        selected: false,
        muted: false,
        gain: 0,
        playbackRate: 1,
        name: "Take 1",
    };

    // 命中测试（clipHeaderControls）需要标签像素宽度：必须由样式解析统一给出，
    // 否则命中端会另写一份测量，与绘制端随时间漂移。
    const style = buildTimelineClipVisualStyle(base);
    assert(typeof style.gainLabelWidth === "number", "gainLabelWidth 应为数字");
    assert(style.gainLabelWidth > 0, "gainLabelWidth 应大于 0");
    assert(typeof style.rateLabelWidth === "number", "rateLabelWidth 应为数字");
    assert(style.rateLabelWidth > 0, "速率 = 1 时 rateLabelWidth 仍应大于 0");

    // 宽度随文本变长而增大（确认测量真的用了当前标签文案）。
    const longer = buildTimelineClipVisualStyle({ ...base, gain: -11.5 });
    assert(longer.gainLabelWidth > style.gainLabelWidth, "增益文案变长时 gainLabelWidth 应增大");

    // 窄 clip 隐藏标签时宽度归零（与 trailingReservePx 的既有语义一致）。
    const narrow = buildTimelineClipVisualStyle({ ...base, widthPx: 40 });
    assert(narrow.gainLabelWidth === 0, "标签不可见时 gainLabelWidth 应为 0");
});

/** rgba(...) 与泳道底色合成后的 RGB。 */
function compositedRgb(fill: string, darkMode: boolean): { r: number; g: number; b: number } {
    const fillMatch = fill.match(/^rgba\((\d+), (\d+), (\d+), ([\d.]+)\)$/);
    if (!fillMatch) throw new Error(`unparseable fill: ${fill}`);
    const [, rs, gs, bs, as] = fillMatch;
    const alpha = Number(as);
    const bgMatch = timelineLaneBackgroundCss(darkMode).match(/rgb\((\d+), (\d+), (\d+)\)/);
    if (!bgMatch) throw new Error(`unparseable lane background for darkMode=${darkMode}`);
    const mix = (channel: string, index: number) =>
        Number(channel) * alpha + Number(bgMatch[index]) * (1 - alpha);
    return { r: mix(rs, 1), g: mix(gs, 2), b: mix(bs, 3) };
}

/** 感知亮度（Rec.601），与样式模块同口径。 */
function perceivedLuminance(rgb: { r: number; g: number; b: number }): number {
    return (rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114) / 255;
}

/** WCAG 2.x 相对亮度（通道先线性化），用于对比度比。 */
function relativeLuminance(rgb: { r: number; g: number; b: number }): number {
    const linear = (channel: number): number => {
        const c = channel / 255;
        return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * linear(rgb.r) + 0.7152 * linear(rgb.g) + 0.0722 * linear(rgb.b);
}

test("the clip header reads as a separate control bar, not part of the audio surface", () => {
    // Issue 141 第二点：顶部控件条与下方音频体此前只差一档 HSL 明度 + 0.03–0.04
    // alpha，合成后明度差只有 0.047–0.064，读起来像"同一块表面的高光"。而它实际
    // 是控件条，且**不接受渐变手势** —— 用户因此以为顶部也能拉渐变，非常反直觉。
    //
    // 门禁在**合成之后**比：色块是半透明的，用户看到的是与泳道底色合成的结果，
    // 比较未合成的 rgb 等于没测他真正看到的东西。
    const MIN_HEADER_BODY_STEP = 0.09;
    const TRACK_COLORS = [
        "#ff7a00",
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#ffffff",
        "#000000",
        "#808080",
        "#74787e",
    ];
    // 阈值取自实测：当前最坏组合（浅色主题 + 纯黑轨道色）为 0.107，留约 19% 裕量。
    for (const darkMode of [false, true]) {
        for (const trackColor of TRACK_COLORS) {
            const style = buildTimelineClipVisualStyle({
                widthPx: 160,
                trackColor,
                selected: false,
                muted: false,
                gain: 1,
                playbackRate: 1,
                name: "x",
                darkMode,
            });
            const step = Math.abs(
                perceivedLuminance(compositedRgb(style.headerFill, darkMode)) -
                    perceivedLuminance(compositedRgb(style.bodyFill, darkMode)),
            );
            if (step < MIN_HEADER_BODY_STEP) {
                throw new Error(
                    `darkMode=${darkMode} trackColor=${trackColor}: header/body composited step ${step.toFixed(4)} < ${MIN_HEADER_BODY_STEP} — the control bar reads as part of the audio surface (Issue 141)`,
                );
            }
        }
    }

    // 名称与右侧标签都画在 header 上：新色调不得把它们变得难读。
    // 旧实现的明度位移方向**指向文字色**，文字对比反而下降；现在两个主题都
    // 背离文字色，因此这里钉住 WCAG AA 的 4.5:1。
    for (const darkMode of [false, true]) {
        const style = buildTimelineClipVisualStyle({
            widthPx: 160,
            trackColor: "#74787e",
            selected: false,
            muted: false,
            gain: 1,
            playbackRate: 1,
            name: "x",
            darkMode,
        });
        const header = relativeLuminance(compositedRgb(style.headerFill, darkMode));
        const text = relativeLuminance(compositedRgb(style.textFill, darkMode));
        const ratio = (Math.max(header, text) + 0.05) / (Math.min(header, text) + 0.05);
        if (ratio < 4.5) {
            throw new Error(
                `darkMode=${darkMode}: header text contrast ${ratio.toFixed(2)}:1 is below WCAG AA (4.5:1)`,
            );
        }
    }
});
