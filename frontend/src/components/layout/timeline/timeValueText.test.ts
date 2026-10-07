/**
 * 时间量 Tooltip 的**主/副单位**组合回归。
 *
 * ## 被钉住的核心
 *
 * 「把某个时间量按主/副时间单位显示」此前在四个地方各写了一遍，本模块是唯一落点。
 * 而**时长**与**时刻**是两种不同的时间量，格式化口径不同（见 `timeValueText` 文件头）：
 *
 * - 时长是**零基点**的（淡化长度、吸附偏移）：绝对位置 3.3.000 处的 0.5s 偏移
 *   显示为 `0.1.000`，而不是 `3.3.000`；
 * - 时刻是**绝对位置**（吸附偏移的位置、播放头、Tempo Map 变化点），且**感知
 *   Tempo Map**。
 *
 * 两者混用会产出"看起来对、实则错"的文本（例如把 0.5s 的偏移显示成小节 3），
 * 且只在特定主单位下暴露 —— 因此这里逐条钉住。
 */
import { describe, expect, it } from "vitest";

import { formatCursorUnit } from "./timeFormat";
import {
    buildSnapOffsetInfoText,
    formatDurationText,
    formatPositionText,
    formatSignedDurationText,
    hasSecondaryUnit,
    type TimeValueFormatContext,
} from "./timeValueText";
import type { TempoMap } from "../../../utils/tempoMap";

/** 120 BPM / 4 拍一小节 ⇒ 1 拍 = 0.5s、1 小节 = 2s。 */
function ctx(o: Partial<TimeValueFormatContext> = {}): TimeValueFormatContext {
    return {
        primaryTimeUnit: "barBeats",
        secondaryTimeUnit: "none",
        bpm: 120,
        beatsPerBar: 4,
        grid: "1/4",
        ...o,
    };
}

/** 词典替身：把 `{name}` 全部替换掉（与 `I18nProvider.tVars` 的 `formatTemplate` 同语义）。 */
const fakeT = (key: string, vars: Record<string, string>): string => {
    const templates: Record<string, string> = {
        clip_snap_offset: "吸附偏移",
        clip_snap_offset_value: "吸附偏移：{offset}\n位置：{position}",
        clip_snap_offset_value_drag: "吸附偏移：{offset} [{delta}]\n位置：{position} [{delta}]",
    };
    return (templates[key] ?? key).replace(/\{(\w+)\}/g, (m, name: string) =>
        Object.prototype.hasOwnProperty.call(vars, name) ? vars[name] : m,
    );
};

describe("副时间单位是否参与展示", () => {
    it("未启用（none）⇒ 不展示", () => {
        expect(hasSecondaryUnit(ctx({ secondaryTimeUnit: "none" }))).toBe(false);
    });

    it("与主单位相同 ⇒ 不展示（否则两列完全重复）", () => {
        expect(hasSecondaryUnit(ctx({ secondaryTimeUnit: "barBeats" }))).toBe(false);
    });

    it("与主单位不同 ⇒ 展示", () => {
        expect(hasSecondaryUnit(ctx({ secondaryTimeUnit: "seconds" }))).toBe(true);
    });

    it("三态体现在文本上：`{主}` / `{主} / {主}` / `{主} / {副}`", () => {
        expect(formatDurationText(0.5, ctx({ secondaryTimeUnit: "none" }))).toBe("0.1.000");
        expect(formatDurationText(0.5, ctx({ secondaryTimeUnit: "barBeats" }))).toBe("0.1.000");
        expect(formatDurationText(0.5, ctx({ secondaryTimeUnit: "seconds" }))).toBe(
            "0.1.000 / 0.500",
        );
    });
});

describe("时长（零基点）与时刻（绝对）口径不同", () => {
    it("★ 同一 0.5s：时长显示 0.1.000（零基点），时刻显示 3.3.000（绝对）", () => {
        const c = ctx();
        // 时刻：绝对位置 5.0s = 10 拍 = 第 3 小节第 3 拍。
        expect(formatPositionText(5, c)).toBe("3.3.000");
        // 时长：0.5s = 1 拍，但**从小节 0 起计**。
        expect(formatDurationText(0.5, c)).toBe("0.1.000");
        expect(formatDurationText(0.5, c)).not.toBe(formatPositionText(0.5, c));
    });

    it("★ 吸附偏移的位置行随 Clip 起点平移，偏移行不动", () => {
        const c = ctx();
        const atStart = buildSnapOffsetInfoText({
            offsetSec: 0.5,
            positionSec: 2.5,
            deltaSec: null,
            formatCtx: c,
            t: fakeT,
        });
        const shifted = buildSnapOffsetInfoText({
            offsetSec: 0.5,
            positionSec: 6.5,
            deltaSec: null,
            formatCtx: c,
            t: fakeT,
        });
        // 偏移行恒为 0.1.000；位置行从小节 2 走到小节 4。
        expect(atStart.split("\n")[0]).toBe("吸附偏移：0.1.000");
        expect(shifted.split("\n")[0]).toBe("吸附偏移：0.1.000");
        expect(atStart.split("\n")[1]).toBe("位置：2.2.000");
        expect(shifted.split("\n")[1]).toBe("位置：4.2.000");
    });

    it("时长格式化与淡化 ToolTip 的既有口径逐值一致（黄金值回归）", () => {
        // 这些字面量是改动前的实测输出（120 BPM / 4 拍一小节）—— 收口到本模块时
        // 必须逐字不变。副单位存在时按 `{主} / {副}` 拼接。
        const expected: Array<[number, string, string]> = [
            // [秒, 仅主单位, 主 / 副（seconds）]
            [0, "0.0.000", "0.0.000 / 0.000"],
            [0.25, "0.0.500", "0.0.500 / 0.250"],
            [0.5, "0.1.000", "0.1.000 / 0.500"],
            [1.75, "0.3.500", "0.3.500 / 1.750"],
            [3, "1.2.000", "1.2.000 / 3.000"],
        ];
        for (const [sec, primaryOnly, withSecondary] of expected) {
            expect(formatDurationText(sec, ctx({ secondaryTimeUnit: "none" })), `sec=${sec}`).toBe(
                primaryOnly,
            );
            expect(
                formatDurationText(sec, ctx({ secondaryTimeUnit: "seconds" })),
                `sec=${sec} +seconds`,
            ).toBe(withSecondary);
        }
    });

    it("时刻格式化与 formatCursorTime 的既有口径逐值一致（回归）", () => {
        for (const secondary of ["none", "seconds"] as const) {
            for (const sec of [0, 0.5, 2, 5, 9.25]) {
                const c = ctx({ secondaryTimeUnit: secondary });
                const expected =
                    secondary === "none"
                        ? formatCursorUnit("barBeats", sec, c)
                        : `${formatCursorUnit("barBeats", sec, c)} / ${formatCursorUnit("seconds", sec, c)}`;
                expect(formatPositionText(sec, c), `sec=${sec}`).toBe(expected);
            }
        }
    });

    it("★ 时刻感知 Tempo Map：同一秒在第二段（60 BPM）下给出不同的标签", () => {
        const tempoMap: TempoMap = {
            points: [
                {
                    id: "a",
                    positionSec: 0,
                    bpm: 120,
                    timeSignature: { numerator: 4, denominator: 4 },
                    scale: null,
                },
                { id: "b", positionSec: 4, bpm: 60, timeSignature: null, scale: null },
            ],
        };
        const plain = ctx({ primaryTimeUnit: "barBeats" });
        const mapped = ctx({ primaryTimeUnit: "barBeats", tempoMap });
        // 段 1（0..4s，120BPM）= 2 小节；4s 起每小节 4s。
        // 6s 落在段 2 的第 1 小节内：120BPM 外推会算成第 4 小节。
        const withMap = formatPositionText(6, mapped);
        const withoutMap = formatPositionText(6, plain);
        expect(withMap).not.toBe(withoutMap);
        expect(withMap).toBe(formatCursorUnit("barBeats", 6, mapped));
    });

    it("时长**不**感知 Tempo Map（零基点的静态折算，与既有约定一致）", () => {
        const tempoMap: TempoMap = {
            points: [
                {
                    id: "a",
                    positionSec: 0,
                    bpm: 120,
                    timeSignature: { numerator: 4, denominator: 4 },
                    scale: null,
                },
                { id: "b", positionSec: 4, bpm: 60, timeSignature: null, scale: null },
            ],
        };
        expect(formatDurationText(0.5, ctx({ tempoMap }))).toBe(formatDurationText(0.5, ctx()));
    });
});

describe("带符号的位移量", () => {
    it("正负号前置，幅度走同一套主/副单位格式化", () => {
        expect(formatSignedDurationText(0.5, ctx())).toBe("+0.1.000");
        expect(formatSignedDurationText(-0.5, ctx())).toBe("-0.1.000");
        expect(formatSignedDurationText(0.5, ctx({ secondaryTimeUnit: "seconds" }))).toBe(
            "+0.1.000 / 0.500",
        );
    });

    it("非有限值按 0 处理（不产出 `NaN`）", () => {
        expect(formatSignedDurationText(Number.NaN, ctx())).toBe("+0.0.000");
    });
});

describe("吸附偏移 Tooltip 的三种形态", () => {
    it("★ 未拖拽且偏移为 0 ⇒ 只有标签（位置恒等于 Clip 起点，无信息量）", () => {
        expect(
            buildSnapOffsetInfoText({
                offsetSec: 0,
                positionSec: 4,
                deltaSec: null,
                formatCtx: ctx(),
                t: fakeT,
            }),
        ).toBe("吸附偏移");
    });

    it("★ 未拖拽且偏移非 0 ⇒ 两行，且偏移行是时长口径", () => {
        const text = buildSnapOffsetInfoText({
            offsetSec: 0.5,
            positionSec: 4.5,
            deltaSec: null,
            formatCtx: ctx(),
            t: fakeT,
        });
        expect(text.split("\n")).toEqual(["吸附偏移：0.1.000", "位置：3.2.000"]);
    });

    it("★ 拖拽中 ⇒ 两行各追加 `[±位移]`", () => {
        const text = buildSnapOffsetInfoText({
            offsetSec: 0.75,
            positionSec: 4.75,
            deltaSec: 0.25,
            formatCtx: ctx(),
            t: fakeT,
        });
        expect(text.split("\n")).toEqual([
            "吸附偏移：0.1.500 [+0.0.500]",
            "位置：3.2.500 [+0.0.500]",
        ]);
    });

    it("★ 拖拽中偏移被拖回 0 ⇒ 仍展示两行（此时正是最需要看到 0 与位移的时候）", () => {
        const text = buildSnapOffsetInfoText({
            offsetSec: 0,
            positionSec: 4,
            deltaSec: -0.5,
            formatCtx: ctx(),
            t: fakeT,
        });
        expect(text).toContain("[");
        expect(text.split("\n")).toHaveLength(2);
    });

    it("★ 位移为 0（未越阈值 / 首帧）⇒ 不出现方括号", () => {
        for (const delta of [0, 1e-9]) {
            const text = buildSnapOffsetInfoText({
                offsetSec: 0.5,
                positionSec: 4.5,
                deltaSec: delta,
                formatCtx: ctx(),
                t: fakeT,
            });
            expect(text, `delta=${delta}`).not.toContain("[");
        }
    });

    it("负偏移按 0 处理（不出现负数时长）", () => {
        expect(
            buildSnapOffsetInfoText({
                offsetSec: -1,
                positionSec: 4,
                deltaSec: null,
                formatCtx: ctx(),
                t: fakeT,
            }),
        ).toBe("吸附偏移");
    });
});
