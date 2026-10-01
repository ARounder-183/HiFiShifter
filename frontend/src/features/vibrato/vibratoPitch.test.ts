import { describe, expect, test } from "vitest";

import {
    isUnsetValue,
    resolveVibratoAnchors,
    suppressUnsetValues,
    usesUnsetValue,
} from "./vibratoPitch";

/*
 * 音高哨兵适配的纯逻辑。
 *
 * 【为什么值得单独钉】这一层错了不会抛错，只会**静默改坏曲线** —— 锚点取到哨兵 0，
 * `baseline: "line"` 就把整段真实音高拉成 0；少了一次哨兵还原，就在气口上凭空造出
 * 一个音。两者都要靠断言才看得见。
 */

describe("usesUnsetValue / isUnsetValue", () => {
    test("只有音高用 0 当「无数据」哨兵", () => {
        expect(usesUnsetValue("pitch")).toBe(true);
        for (const param of ["dyn", "volume", "tension", "breath_gain"]) {
            expect(usesUnsetValue(param)).toBe(false);
        }
    });

    test("音高：0 与非有限值都算未检测，真实音高不算", () => {
        expect(isUnsetValue("pitch", 0)).toBe(true);
        expect(isUnsetValue("pitch", Number.NaN)).toBe(true);
        expect(isUnsetValue("pitch", Number.POSITIVE_INFINITY)).toBe(true);
        expect(isUnsetValue("pitch", 1)).toBe(false);
        expect(isUnsetValue("pitch", 60)).toBe(false);
        expect(isUnsetValue("pitch", -3)).toBe(false);
    });

    test("其他参数：0 是合法值，不是哨兵", () => {
        expect(isUnsetValue("volume", 0)).toBe(false);
        expect(isUnsetValue("dyn", 0)).toBe(false);
    });
});

describe("resolveVibratoAnchors", () => {
    test("非哨兵参数：与既有行为一致，原样取首末帧", () => {
        expect(resolveVibratoAnchors("volume", [0, 1, 2])).toEqual({
            startValue: 0,
            endValue: 2,
        });
    });

    test("音高：跳过首尾的未检测帧，取已检测段的端点", () => {
        // 头部气口两帧、尾部气口一帧。
        expect(resolveVibratoAnchors("pitch", [0, 0, 60, 62, 0])).toEqual({
            startValue: 60,
            endValue: 62,
        });
    });

    test("音高：中间的未检测帧不影响锚点（只有首末已检测帧说了算）", () => {
        expect(resolveVibratoAnchors("pitch", [0, 60, 0, 0, 62, 0])).toEqual({
            startValue: 60,
            endValue: 62,
        });
    });

    test("音高：只有一个已检测帧时首末同值（不做无中生有的插值）", () => {
        expect(resolveVibratoAnchors("pitch", [0, 0, 61, 0, 0])).toEqual({
            startValue: 61,
            endValue: 61,
        });
    });

    test("音高：整段未检测（含 NaN 混排）返回 null —— 没有可调制的对象", () => {
        expect(resolveVibratoAnchors("pitch", [0, 0, 0])).toBeNull();
        expect(resolveVibratoAnchors("pitch", [Number.NaN, 0, Number.NaN])).toBeNull();
    });

    test("空数组返回 null（非哨兵参数也一样）", () => {
        expect(resolveVibratoAnchors("pitch", [])).toBeNull();
        expect(resolveVibratoAnchors("volume", [])).toBeNull();
    });
});

describe("suppressUnsetValues", () => {
    test("音高：未检测帧还原成 0，已检测帧原样保留", () => {
        const result = [60, 61, 62, 63, 64, 65];
        suppressUnsetValues("pitch", result, [0, 60, 61, 62, 63, 0]);
        expect(result).toEqual([0, 61, 62, 63, 64, 0]);
    });

    test("音高：非有限的原值同样还原成 0（不让 NaN 漏进写回值）", () => {
        const result = [10, 20];
        suppressUnsetValues("pitch", result, [Number.NaN, 60]);
        expect(result).toEqual([0, 20]);
    });

    test("非哨兵参数：一个字节都不改（0 是合法值）", () => {
        const result = [0, 5, 7];
        suppressUnsetValues("volume", result, [0, 0, 0]);
        expect(result).toEqual([0, 5, 7]);
    });

    test("幂等：已还原过的结果再还原一次不变", () => {
        const once = [0, 61, 62, 0];
        const twice = once.slice();
        suppressUnsetValues("pitch", twice, [0, 60, 62, 0]);
        expect(twice).toEqual(once);
    });

    test("长度不一致时按较短者对齐，不越界", () => {
        const result = [1, 2, 3];
        suppressUnsetValues("pitch", result, [0, 60]);
        expect(result).toEqual([0, 2, 3]);
    });
});
