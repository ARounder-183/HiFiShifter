/**
 * 淡入淡出 Tooltip 的内容拼装回归。
 *
 * ## 被钉住的两件事
 *
 * 1. **悬停输出与改动前逐字一致**（黄金值）—— 拖拽增量是叠加在既有三行上的，
 *    不能顺手改掉原有读数。
 * 2. **拖拽增量**：长度行与曲率行各自按需追加 `[±增量]`。长度是**零基点时长**
 *    （与吸附偏移同一口径），曲率是两位小数；两者都必须在"显示不出来的位移"
 *    上闭嘴（不出现 `[+0.00]` / `[+0.000]`）—— 否则用户会以为功能坏了。
 *
 * 纯文本版与富内容版共用同一套行组合（`lengthLine` / `dirLine`），因此这里对
 * 文本版做逐字断言，并对富内容版**逐行对拍**（剥掉首行的图标节点）—— 两者
 * 一旦分叉就会失败。
 */
import { isValidElement, type ReactNode } from "react";
import { describe, expect, it } from "vitest";

import {
    buildCrossfadeGripInfoContent,
    buildCrossfadeGripInfoText,
    buildSingleFadeInfoContent,
    buildSingleFadeInfoText,
    type FadeLabelLookup,
} from "./fadeTooltipText";
import type { FadeLengthFormatContext } from "./timeFormat";

/** 120 BPM / 4 拍一小节 ⇒ 1 拍 = 0.5s、1 小节 = 2s。 */
function ctx(o: Partial<FadeLengthFormatContext> = {}): FadeLengthFormatContext {
    return {
        primaryTimeUnit: "barBeats",
        secondaryTimeUnit: "none",
        bpm: 120,
        beatsPerBar: 4,
        grid: "1/4",
        ...o,
    };
}

/** 词典替身（en-US 的形态：`fade_type_label` 带前导空格）。 */
const fakeT: FadeLabelLookup = (key) =>
    ({
        fade_in: "Fade in",
        fade_out: "Fade out",
        fade_type_label: " type",
        common_length: "Length",
        common_curvature: "Curvature",
        fade_shape_linear: "Linear",
    })[key] ?? key;

/** 递归取出节点里的文本（图标元素没有文本子节点，自然贡献空串）。 */
function textOf(node: ReactNode): string {
    if (typeof node === "string" || typeof node === "number") return String(node);
    if (Array.isArray(node)) return node.map(textOf).join("");
    if (isValidElement(node)) {
        return textOf((node.props as { children?: ReactNode }).children);
    }
    return "";
}

/** 富内容 → 每行的纯文本（嵌套的块数组展平；间隔节点成为空行）。 */
function contentRows(node: ReactNode): string[] {
    const items = Array.isArray(node) ? node : [node];
    return items.flatMap((item) => (Array.isArray(item) ? item.map(textOf) : [textOf(item)]));
}

describe("单侧淡变块", () => {
    const base = { isOut: false, shape: 0, dir: 0.35, lengthSec: 0.5, t: fakeT };

    it("★ 悬停（无增量）：三行黄金值，与改动前逐字一致", () => {
        expect(buildSingleFadeInfoText({ ...base, formatCtx: ctx() })).toBe(
            ["Fade in type：Linear", "Length：0.1.000", "Curvature：+0.35"].join("\n"),
        );
    });

    it("★ 长度拖拽：长度行带 `[±位移]`，曲率行不带", () => {
        const text = buildSingleFadeInfoText({
            ...base,
            formatCtx: ctx(),
            delta: { lengthSec: 0.25 },
        });
        expect(text.split("\n")).toEqual([
            "Fade in type：Linear",
            "Length：0.1.000 [+0.0.500]",
            "Curvature：+0.35",
        ]);
    });

    it("★ 曲率拖拽：曲率行带 `[±增量]`，长度行不带", () => {
        const text = buildSingleFadeInfoText({
            ...base,
            formatCtx: ctx(),
            delta: { dir: -0.2 },
        });
        expect(text.split("\n")).toEqual([
            "Fade in type：Linear",
            "Length：0.1.000",
            "Curvature：+0.35 [-0.20]",
        ]);
    });

    it("位移与曲率同时给 ⇒ 两行都带", () => {
        const text = buildSingleFadeInfoText({
            ...base,
            formatCtx: ctx(),
            delta: { lengthSec: -0.5, dir: 0.05 },
        });
        expect(text.split("\n")[1]).toBe("Length：0.1.000 [-0.1.000]");
        expect(text.split("\n")[2]).toBe("Curvature：+0.35 [+0.05]");
    });

    it("★ 位移带主/副单位（与吸附偏移同一口径：副单位各展一份）", () => {
        const text = buildSingleFadeInfoText({
            ...base,
            formatCtx: ctx({ secondaryTimeUnit: "seconds" }),
            delta: { lengthSec: 0.25 },
        });
        expect(text.split("\n")[1]).toBe("Length：0.1.000 / 0.500 [+0.0.500 / 0.250]");
    });

    it("★ 显示不出来的位移不出现方括号（不出现 `[+0.000]` / `[+0.00]`）", () => {
        for (const formatCtx of [ctx(), ctx({ secondaryTimeUnit: "seconds" })]) {
            for (const delta of [{ lengthSec: 0 }, { lengthSec: 1e-9 }, { dir: 0 }, { dir: 0.004 }]) {
                const text = buildSingleFadeInfoText({ ...base, formatCtx, delta });
                expect(text, `${JSON.stringify(delta)} ${JSON.stringify(formatCtx)}`).not.toContain(
                    "[",
                );
            }
        }
    });

    it("未传 delta（悬停）与传空对象等价", () => {
        expect(buildSingleFadeInfoText({ ...base, formatCtx: ctx(), delta: {} })).toBe(
            buildSingleFadeInfoText({ ...base, formatCtx: ctx() }),
        );
    });

    it("★ 富内容版与纯文本版逐行一致（同一套行组合，不得分叉）", () => {
        const args = {
            ...base,
            formatCtx: ctx({ secondaryTimeUnit: "seconds" }),
            delta: { lengthSec: 0.25, dir: -0.2 },
        };
        const rows = contentRows(buildSingleFadeInfoContent(args));
        const textRows = buildSingleFadeInfoText(args).split("\n");
        // 首行含内联图标节点，剥掉图标后应逐字相同。
        expect(rows[0]).toBe("Fade in type：");
        expect(rows.slice(1)).toEqual(textRows.slice(1));
    });
});

describe("交叉点抓手（双列）", () => {
    const earlier = { shape: 0, dir: 0.1, lengthSec: 0.5 };
    const later = { shape: 0, dir: -0.2, lengthSec: 0.75 };

    it("★ 悬停：两块之间空一行，且各自只给当前值", () => {
        expect(
            buildCrossfadeGripInfoText({ earlier, later, formatCtx: ctx(), t: fakeT }),
        ).toBe(
            [
                "Fade out type：Linear",
                "Length：0.1.000",
                "Curvature：+0.10",
                "",
                "Fade in type：Linear",
                "Length：0.1.500",
                "Curvature：-0.20",
            ].join("\n"),
        );
    });

    it("★ 反向模式：两侧各自的长度位移互不干扰", () => {
        const text = buildCrossfadeGripInfoText({
            earlier: { ...earlier, delta: { lengthSec: 0.25 } },
            later: { ...later, delta: { lengthSec: -0.5 } },
            formatCtx: ctx(),
            t: fakeT,
        });
        const rows = text.split("\n");
        expect(rows[1]).toBe("Length：0.1.000 [+0.0.500]");
        expect(rows[5]).toBe("Length：0.1.500 [-0.1.000]");
    });

    it("★ 曲率拖拽：两侧各自的曲率增量", () => {
        const text = buildCrossfadeGripInfoText({
            earlier: { ...earlier, dir: 0.3, delta: { dir: 0.2 } },
            later: { ...later, dir: 0.4, delta: { dir: 0.6 } },
            formatCtx: ctx(),
            t: fakeT,
        });
        const rows = text.split("\n");
        expect(rows[2]).toBe("Curvature：+0.30 [+0.20]");
        expect(rows[6]).toBe("Curvature：+0.40 [+0.60]");
    });

    it("富内容版：两块 + 一个间隔空行", () => {
        const node = buildCrossfadeGripInfoContent({
            earlier: { ...earlier, delta: { lengthSec: 0.25 } },
            later,
            formatCtx: ctx(),
            t: fakeT,
        });
        const rows = contentRows(node);
        // 3 + 1 间隔 + 3 = 7 行。
        expect(rows).toHaveLength(7);
        expect(rows[1]).toBe("Length：0.1.000 [+0.0.500]");
        expect(rows[3]).toBe("");
        expect(rows[5]).toBe("Length：0.1.500");
    });
});
