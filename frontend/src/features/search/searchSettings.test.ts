import { describe, expect, it } from "vitest";

import {
    DEFAULT_SEARCH_SETTINGS,
    effectiveSearchMode,
    normalizeSearchSettings,
    searchModePatch,
    searchOptionsPayload,
    type SearchSettings,
} from "./searchSettings";

describe("normalizeSearchSettings", () => {
    it("缺省 / 非法输入回落到默认（转写全开、模式 smart、命中说明关闭）", () => {
        for (const input of [undefined, null, 42, "nope", {}]) {
            const out = normalizeSearchSettings(input);
            expect(out.translit).toBe(true);
            expect(out.mode).toBe("smart");
            expect(out.heteronym).toBe(true);
            expect(out.japaneseLongVowel).toBe(true);
            expect(out.koreanChoseong).toBe(true);
            expect(out.showMatchReason).toBe(false);
        }
    });

    it("非法模式值收敛到 smart，而不是让整份设置作废", () => {
        expect(normalizeSearchSettings({ mode: "offf" }).mode).toBe("smart");
        expect(normalizeSearchSettings({ mode: 7 }).mode).toBe("smart");
    });

    it("合法字段原样保留，未提供的字段取默认", () => {
        const out = normalizeSearchSettings({ mode: "fuzzy", heteronym: false });
        expect(out.mode).toBe("fuzzy");
        expect(out.heteronym).toBe(false);
        expect(out.japaneseLongVowel).toBe(true);
    });

    it("与默认值常量一致", () => {
        expect(normalizeSearchSettings(undefined)).toEqual(DEFAULT_SEARCH_SETTINGS);
    });
});

describe("effectiveSearchMode", () => {
    it("总开关关闭时一律是 off（不看 mode 存的是什么）", () => {
        expect(
            effectiveSearchMode({ ...DEFAULT_SEARCH_SETTINGS, translit: false, mode: "fuzzy" }),
        ).toBe("off");
    });

    it("总开关打开时取 mode", () => {
        expect(effectiveSearchMode({ ...DEFAULT_SEARCH_SETTINGS, mode: "fuzzy" })).toBe("fuzzy");
    });
});

describe("searchModePatch", () => {
    it("选「关闭」只关总开关，保留宽严（再打开时回到上次的模式）", () => {
        const patch = searchModePatch("off");
        expect(patch).toEqual({ translit: false });
        // 关键：补丁里没有 mode —— 用户上次选的「模糊」不会被清成默认。
        expect("mode" in patch).toBe(false);
    });

    it("选「智能」/「模糊」同时打开总开关", () => {
        expect(searchModePatch("smart")).toEqual({ translit: true, mode: "smart" });
        expect(searchModePatch("fuzzy")).toEqual({ translit: true, mode: "fuzzy" });
    });

    it("关掉再打开能回到原来的模式", () => {
        let settings: SearchSettings = { ...DEFAULT_SEARCH_SETTINGS, mode: "fuzzy" };
        settings = { ...settings, ...searchModePatch("off") };
        expect(effectiveSearchMode(settings)).toBe("off");
        settings = { ...settings, ...searchModePatch("fuzzy") };
        expect(effectiveSearchMode(settings)).toBe("fuzzy");
    });
});

describe("searchOptionsPayload", () => {
    it("把总开关折叠进 mode 后再下发", () => {
        const off = searchOptionsPayload({ ...DEFAULT_SEARCH_SETTINGS, translit: false });
        expect(off.mode).toBe("off");
        // 子开关原样带着：总开关关掉不代表用户放弃了他对这些项的偏好。
        expect(off.heteronym).toBe(true);
    });

    it("开启时下发 mode 与三个子开关", () => {
        const payload = searchOptionsPayload({
            ...DEFAULT_SEARCH_SETTINGS,
            mode: "fuzzy",
            koreanChoseong: false,
        });
        expect(payload.mode).toBe("fuzzy");
        expect(payload.koreanChoseong).toBe(false);
    });
});
