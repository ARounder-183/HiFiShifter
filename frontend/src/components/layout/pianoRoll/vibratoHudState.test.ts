import { describe, expect, test } from "vitest";

import { nextVibratoHudAnchor, vibratoHudState, type VibratoHudSource } from "./vibratoHudState";

const source = (overrides: Partial<VibratoHudSource> = {}): VibratoHudSource => ({
    presetId: "builtin.natural",
    depthCents: 30,
    rateHz: 5.5,
    depthAdjusted: false,
    rateAdjusted: false,
    ...overrides,
});

describe("nextVibratoHudAnchor", () => {
    test("带坐标就换", () => {
        expect(nextVibratoHudAnchor({ clientX: 1, clientY: 2 }, 10, 20)).toEqual({
            clientX: 10,
            clientY: 20,
        });
    });

    /*
     * ★ 核心规则：**省略坐标即沿用上一次**。
     *
     * 键盘事件没有指针坐标，而深度 / 速率也能由方向键改。若"没有坐标就不上报"，
     * 键盘调参要等下一次指针移动才刷新读数 —— 用户看到的是气泡里的数字比曲线慢
     * 一拍。这条规则让键盘路径也能立即上报。
     */
    test("省略坐标就沿用上一次（键盘调参没有指针坐标）", () => {
        const previous = { clientX: 300, clientY: 400 };
        expect(nextVibratoHudAnchor(previous)).toEqual(previous);
        expect(nextVibratoHudAnchor(previous, 300, 400)).toEqual(previous);
    });

    test("从未拿到过坐标时保持 null（不去猜一个位置）", () => {
        expect(nextVibratoHudAnchor(null)).toBeNull();
    });

    test("只给一半坐标视为省略（不产生半个锚点）", () => {
        const previous = { clientX: 300, clientY: 400 };
        expect(nextVibratoHudAnchor(previous, 500)).toEqual(previous);
        expect(nextVibratoHudAnchor(previous, undefined, 600)).toEqual(previous);
    });
});

describe("vibratoHudState", () => {
    test("没有锚点就不上报（宁可不动，也不要甩到 (0,0)）", () => {
        expect(vibratoHudState(source(), null)).toBeNull();
    });

    test("载荷带工作副本的深度 / 速率与锚点", () => {
        expect(
            vibratoHudState(source({ depthCents: 42, rateHz: 6.5 }), { clientX: 7, clientY: 8 }),
        ).toEqual({
            presetId: "builtin.natural",
            depthCents: 42,
            rateHz: 6.5,
            adjusted: false,
            clientX: 7,
            clientY: 8,
        });
    });

    test("「已调整」是聚合标记：深度或速率任一项调过就点亮", () => {
        const anchor = { clientX: 0, clientY: 0 };
        expect(vibratoHudState(source({ depthAdjusted: true }), anchor)?.adjusted).toBe(true);
        expect(vibratoHudState(source({ rateAdjusted: true }), anchor)?.adjusted).toBe(true);
        expect(
            vibratoHudState(source({ depthAdjusted: true, rateAdjusted: true }), anchor)?.adjusted,
        ).toBe(true);
        expect(vibratoHudState(source(), anchor)?.adjusted).toBe(false);
    });
});
