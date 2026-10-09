// 插件（新轴宿主）里的淡变浮标必须与独立 App 说同一套话：形状 + 长度 + 曲率。
// 曾经的"REAPER c=… S=…"与"HiFiShifter 示意曲线；声音由 REAPER 控制"是给实现者看的
// 读数，对用户没有任何可行动的信息（用户反馈）。
import { expect, test } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement, Fragment } from "react";
import { buildSingleFadeInfoText, buildSingleFadeInfoContent } from "./fadeTooltipText";

const t = (key: string) =>
    (
        ({
            common_label_value: "{label}: {value}",
            common_value_sep: ": ",
            fade_info_side_type_label: "{side} {type}",
            common_length: "Length",
            common_curvature: "Curvature",
            fade_type_label: "Type",
            fade_in: "Fade In",
            fade_out: "Fade Out",
            fade_shape_linear: "Linear",
            fade_shape_fast_start: "Fast Start",
            fade_shape_slow_start_end: "Slow Start/End",
        }) as Record<string, string>
    )[key] ?? key;

const formatCtx = {
    primaryTimeUnit: "seconds" as const,
    secondaryTimeUnit: "none" as const,
    bpm: 120,
    beatsPerBar: 4,
    grid: "1/4",
};

const hostFades = (over: Partial<Record<string, unknown>> = {}) => ({
    curve_mode: "reaper_new" as const,
    in_curvature: -0.2,
    out_curvature: 0,
    in_s: 0.65,
    out_s: 0,
    ...over,
});

test("the plugin tooltip reads like the standalone one: shape, length, curvature", () => {
    const text = buildSingleFadeInfoText({
        isOut: false,
        shape: -1, // 表外坐标时宿主把 C_FADEINSHAPE 读成 -1
        dir: -0.2,
        lengthSec: 0.75,
        t,
        formatCtx,
        hostFades: hostFades(),
    });
    // 没有任何宿主术语，也没有"示意曲线 / 声音由宿主控制"这类免责声明。
    expect(text).not.toContain("c=");
    expect(text).not.toContain("S=");
    expect(text).not.toContain("REAPER");
    expect(text).not.toContain("HiFiShifter");
    // 曲率行就是滑杆持有的那个数。
    expect(text).toContain("Curvature: -0.20");
    expect(text).toContain("Length: ");
    // |S| 过半 ⇒ 画布画的是 S 族占优的混合，类型行就报 S 族（不假称某个具体预设）。
    expect(text).toContain("Slow Start/End");
});

test("axes landing exactly on a preset name that preset, and the rich form keeps the icon", () => {
    const args = {
        isOut: false,
        shape: -1,
        dir: 0.5,
        lengthSec: 0.75,
        t,
        formatCtx,
        hostFades: hostFades({ in_curvature: 0.5, in_s: 0 }),
    };
    expect(buildSingleFadeInfoText(args)).toContain("Fast Start");
    const rich = renderToStaticMarkup(
        createElement(Fragment, null, buildSingleFadeInfoContent(args)),
    );
    // 类型行与独立 App 一样：文字标签 + 内联曲线图标（富内容版用图标代替形状名）。
    expect(rich).toContain("<svg");
    expect(rich).toContain("Fade In Type");
    expect(rich).not.toContain("S=");
});

test("a host whose axis version is unknown still shows the standalone three lines", () => {
    const text = buildSingleFadeInfoText({
        isOut: false,
        shape: 1,
        dir: 0.3,
        lengthSec: 0.75,
        t,
        formatCtx,
        hostFades: hostFades({ curve_mode: "unknown" }),
    });
    expect(text).toContain("Fast Start");
    expect(text).toContain("Curvature: +0.30");
    expect(text).not.toContain("fade_info_host_unknown");
});

test("a legacy host reads its own shape and curvature unchanged", () => {
    const text = buildSingleFadeInfoText({
        isOut: true,
        shape: 2,
        dir: 0.4,
        lengthSec: 1,
        t,
        formatCtx,
        hostFades: hostFades({ curve_mode: "legacy" }),
    });
    expect(text).toContain("Fade Out");
    expect(text).toContain("Curvature: +0.40");
    expect(text).not.toContain("fade_shape_linear");
});
