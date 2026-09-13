/**
 * 参数编辑器配色表 · 主题方向性与同源约束单测。
 *
 * 【本测试要钉住的核心不变量（都是"看起来只是换个色值"、实则破坏语义的约定）】
 *
 * 1. **黑键行背景带在两套主题下都必须"压暗"**：钢琴背景的意义是复刻键盘的黑白交替，
 *    而键盘列的黑键恒比白键暗。若深色主题改成提亮，黑键行就比白键行亮——方向与
 *    键盘列**相反**（用户报告的现象："深色模式下背景的黑键和白键的深色区域是相反的"）。
 *    本用例对两套主题各自把带子合成到底色上，断言"合成了带子的行更暗"。
 *
 * 2. **带子必须看得见，但不能比最弱的网格线更抢眼**：实测过的两次失败形态分别是
 *    - 压暗 alpha 太小（深色底接近黑，`rgba(0,0,0,0.08)` 只有 Δ2，肉眼不可见）；
 *    - 提亮方向错误（见上）。
 *    因此断言 `Δ带 ≥ 可见下限` 且 `Δ带 ≤ Δ最弱网格线`：前者挡住"白做"，后者保证
 *    "网格线仍清晰可见"这条验收标准。
 *
 * 3. **播放头颜色必须与时间轴标尺同源**：参数编辑器复用时间轴那套标尺组件，其播放头
 *    取的是 `--qt-playhead`（DOM/CSS 变量，用户可在外观设置里改）。画布侧的播放头若
 *    自带一套硬编码色（旧实现是 `[0,0,0,0.2]` 的兜底），两者就会**颜色不一致**
 *    （用户报告："播放线在底下和上方标尺的颜色不一致"）。因此两套主题都必须给出
 *    **同一个 CSS 变量写法**，而不是各自的字面色值。
 */
import { describe, expect, it } from "vitest";

import { PLAYHEAD_COLOR_TOKEN, resolvePianoRollColors } from "./colors";
import { getBuiltinThemeColors } from "../../../theme/defaultThemes";

/** 解析 `#rrggbb` 为 0..255 的三通道。 */
function parseHex(hex: string): [number, number, number] {
    const value = hex.replace("#", "");
    return [
        parseInt(value.slice(0, 2), 16),
        parseInt(value.slice(2, 4), 16),
        parseInt(value.slice(4, 6), 16),
    ];
}

/** 解析 `rgba(r,g,b,a)` / `rgb(r,g,b)` 为通道 + alpha。 */
function parseCssColor(css: string): [number, number, number, number] {
    const match = /rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)/.exec(
        css,
    );
    if (!match) throw new Error(`不是可解析的颜色：${css}`);
    return [
        Number(match[1]),
        Number(match[2]),
        Number(match[3]),
        match[4] == null ? 1 : Number(match[4]),
    ];
}

/** 把半透明色按 alpha 合成到不透明底色上，返回每通道结果。 */
function composite(
    over: readonly [number, number, number, number],
    under: readonly [number, number, number],
): [number, number, number] {
    const a = over[3];
    return [
        over[0] * a + under[0] * (1 - a),
        over[1] * a + under[1] * (1 - a),
        over[2] * a + under[2] * (1 - a),
    ];
}

/** 感知亮度（与人眼对 RGB 的敏感度一致的常用权重）。 */
function luminance(rgb: readonly [number, number, number]): number {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/** 把某套主题的配色 + 画布底色准备好，供方向性断言使用。 */
function themeFixture(isDark: boolean) {
    const colors = resolvePianoRollColors(isDark);
    const graphBg = parseHex(getBuiltinThemeColors(isDark ? "dark" : "light")["qt-graph-bg"]);
    const bandOver = parseCssColor(colors.blackKeyRowBand);
    const weakLineOver = parseCssColor(colors.pitchGridOther);
    return {
        colors,
        graphBg,
        baseLum: luminance(graphBg),
        bandLum: luminance(composite(bandOver, graphBg)),
        weakLineLum: luminance(composite(weakLineOver, graphBg)),
    };
}

describe("钢琴背景带（黑键行）的明暗方向", () => {
    it("浅色主题：黑键行比白键行暗（与键盘列同向）", () => {
        const f = themeFixture(false);
        expect(f.bandLum).toBeLessThan(f.baseLum);
    });

    it("深色主题：黑键行同样比白键行暗——不得提亮", () => {
        const f = themeFixture(true);
        // 旧实现（深色用 rgba(255,255,255,0.04)）恰好在这里失败：bandLum > baseLum。
        expect(f.bandLum).toBeLessThan(f.baseLum);
    });

    it("两套主题的带子都看得见（Δ ≥ 6）", () => {
        for (const isDark of [false, true]) {
            const f = themeFixture(isDark);
            expect(f.baseLum - f.bandLum, `isDark=${isDark}`).toBeGreaterThanOrEqual(6);
        }
    });

    it("两套主题的带子都不比最弱的网格线更抢眼（网格线仍是最高对比）", () => {
        for (const isDark of [false, true]) {
            const f = themeFixture(isDark);
            const bandDelta = Math.abs(f.baseLum - f.bandLum);
            const lineDelta = Math.abs(f.weakLineLum - f.baseLum);
            // 允许相等（浅色主题两者都是 Δ14.4），但不得反超。
            expect(bandDelta, `isDark=${isDark}`).toBeLessThanOrEqual(lineDelta + 0.5);
        }
    });
});

describe("播放头颜色同源（画布 ↔ 标尺）", () => {
    it("两套主题都使用 --qt-playhead 变量本身，而不是各自的字面色值", () => {
        expect(resolvePianoRollColors(false).playheadLine).toBe(PLAYHEAD_COLOR_TOKEN);
        expect(resolvePianoRollColors(true).playheadLine).toBe(PLAYHEAD_COLOR_TOKEN);
    });

    it("token 指向时间轴标尺所用的同一个变量（TimeRuler 的 bg-qt-playhead）", () => {
        expect(PLAYHEAD_COLOR_TOKEN).toBe("var(--qt-playhead)");
        expect(getBuiltinThemeColors("dark")["qt-playhead"]).toBeDefined();
        expect(getBuiltinThemeColors("light")["qt-playhead"]).toBeDefined();
    });
});
