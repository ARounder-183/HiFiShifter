// 宿主淡化浮标回归：原始新轴优先，不用旧形状图标冒充校准完成。
import { expect, test } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement, Fragment } from "react";
import { buildSingleFadeInfoText, buildSingleFadeInfoContent } from "./fadeTooltipText";

test("host tooltip reports both raw axes and HFS visual/audio ownership", () => {
    const args = {
        isOut: false,
        shape: 6,
        dir: 0.8,
        lengthSec: 0.75,
        t: (key: string) =>
            (
                ({
                    common_label_value: "{label}: {value}",
                    fade_info_side_type_label: "{side} {type}",
                    fade_info_host_curve_note:
                        "HiFiShifter schematic curve; REAPER renders the audio",
                }) as Record<string, string>
            )[key] ?? key,
        formatCtx: {
            primaryTimeUnit: "seconds" as const,
            secondaryTimeUnit: "none" as const,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/4",
        },
        hostFades: {
            curve_mode: "reaper_new" as const,
            in_curvature: -0.2,
            out_curvature: 0,
            in_s: 0.65,
            out_s: 0,
        },
    };
    const text = buildSingleFadeInfoText(args);
    expect(text).toContain("REAPER c=-0.20 S=0.65");
    expect(text).toContain("HiFiShifter schematic curve");
    expect(text).not.toContain("fade_shape");
    // 文案必须来自词表：查不到时 lookup 会原样返回键名，那种静默失败在这里被拦下。
    expect(text).not.toContain("fade_info_host_curve_note");
    const rich = renderToStaticMarkup(
        createElement(Fragment, null, buildSingleFadeInfoContent(args)),
    );
    expect(rich).toContain("S=0.65");
    expect(rich).not.toContain("<svg");
});

test("unknown host axes are labelled through the catalog, not a hardcoded marker", () => {
    const args = {
        isOut: false,
        shape: 6,
        dir: 0.8,
        lengthSec: 0.75,
        t: (key: string) =>
            (
                ({
                    common_label_value: "{label}: {value}",
                    fade_info_side_type_label: "{side} {type}",
                    fade_info_host_curve_note:
                        "HiFiShifter schematic curve; REAPER renders the audio",
                    fade_info_host_unknown: "REAPER curve unknown",
                }) as Record<string, string>
            )[key] ?? key,
        formatCtx: {
            primaryTimeUnit: "seconds" as const,
            secondaryTimeUnit: "none" as const,
            bpm: 120,
            beatsPerBar: 4,
            grid: "1/4",
        },
        hostFades: {
            curve_mode: "unknown" as const,
            in_curvature: 0,
            out_curvature: 0,
            in_s: 0,
            out_s: 0,
        },
    };
    const text = buildSingleFadeInfoText(args);
    expect(text).toContain("REAPER curve unknown");
    expect(text).not.toContain("fade_info_host_unknown");
});
