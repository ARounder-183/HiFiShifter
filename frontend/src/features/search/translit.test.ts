/**
 * 前端匹配器的单元测试。
 *
 * 【为什么用「手工构造的形态」而不是调后端】本文件的被测对象是**匹配规则**，
 * 不是转写规则。转写由后端保证（`search::translit` 有自己的测试），这里直接喂
 * 入形态，才能把「档位、长度门槛、判定顺序」独立钉死。
 *
 * 用例与 Rust 侧 `search::matcher::tests` 一一对应：两处对同一个查询给出不同
 * 命中集合是最难排查的一类不一致，共享同一组向量是最便宜的护栏。
 */
import { describe, expect, it } from "vitest";

import {
    SCORE_FUZZY,
    SCORE_FULL_PREFIX,
    SCORE_FULL_SUBSTRING,
    SCORE_INITIALS_EXACT,
    SCORE_INITIALS_PREFIX,
    SCORE_INITIALS_SUBSEQ,
    SCORE_LITERAL,
    buildQuery,
    fallbackTranslit,
    foldText,
    isSubsequence,
    matchTranslit,
    type TranslitForms,
} from "./translit";
import type { SearchMode } from "./searchSettings";

/** 构造一份「后端会给出的形态」。默认只给 latin（等价于纯 ASCII 文本）。 */
function forms(partial: Partial<TranslitForms> & { latin: string }): TranslitForms {
    return {
        compact: partial.compact ?? partial.latin,
        initials: partial.initials ?? "",
        variants: partial.variants ?? [],
        ...partial,
    };
}

function hit(text: TranslitForms, query: string, mode: SearchMode = "smart") {
    return matchTranslit(text, buildQuery(query), mode);
}

describe("foldText — 归一化", () => {
    it("全角与连字归一", () => {
        expect(foldText("ＶＯＣＡＬ．ｗａｖ")).toBe("vocal.wav");
        expect(foldText("ﬁle")).toBe("file");
    });

    it("变音符号被剥掉", () => {
        expect(foldText("Étude")).toBe("etude");
        expect(foldText("Ünïcödé")).toBe("unicode");
    });
});

describe("fallbackTranslit — 后端不可用时的降级形态", () => {
    it("只做折叠与去分隔符，不产生初声", () => {
        const out = fallbackTranslit("Vocal_Take01.WAV");
        expect(out.latin).toBe("vocal_take01.wav");
        expect(out.compact).toBe("vocaltake01wav");
        expect(out.initials).toBe("");
    });
});

describe("matchTranslit — 档位", () => {
    it("字面子串最高", () => {
        const info = hit(forms({ latin: "vocal01.wav" }), "vocal");
        expect(info?.kind).toBe("literal");
        expect(info?.score).toBe(SCORE_LITERAL);
    });

    it("全拼前缀高于全拼子串", () => {
        const prefix = hit(
            forms({ latin: "主歌01.wav", compact: "zhuge01wav", initials: "zg" }),
            "zhuge",
        );
        expect(prefix?.score).toBe(SCORE_FULL_PREFIX);
        expect(prefix?.kind).toBe("pinyin");

        const inner = hit(
            forms({ latin: "翻唱主歌01.wav", compact: "fanchangzhuge01wav", initials: "fczg" }),
            "zhuge",
        );
        expect(inner?.score).toBe(SCORE_FULL_SUBSTRING);
    });

    it("初声精确 / 前缀 / 子序列", () => {
        const exact = hit(
            forms({ latin: "撤销.wav", compact: "chexiaowav", initials: "cx" }),
            "cx",
        );
        expect(exact?.score).toBe(SCORE_INITIALS_EXACT);

        const prefix = hit(
            forms({
                latin: "撤销并重做.wav",
                compact: "chexiaobingzhongzuowav",
                initials: "cxbzz",
            }),
            "cx",
        );
        expect(prefix?.score).toBe(SCORE_INITIALS_PREFIX);

        const subseq = hit(
            forms({ latin: "分割音频块.wav", compact: "fengeyinpinkuaiwav", initials: "fgypk" }),
            "fy",
        );
        expect(subseq?.score).toBe(SCORE_INITIALS_SUBSEQ);
    });

    it("忽略分隔符的字面命中走全拼档但记作 literal", () => {
        const info = hit(
            forms({ latin: "vocal_take_01.wav", compact: "vocaltake01wav" }),
            "vocaltake",
        );
        expect(info?.score).toBe(SCORE_FULL_PREFIX);
        expect(info?.kind).toBe("literal");
    });

    it("多音字变体参与匹配", () => {
        const info = hit(
            forms({
                latin: "重做.wav",
                compact: "zhongzuowav",
                variants: ["chongzuowav", "tongzuowav"],
            }),
            "chongzuo",
        );
        expect(info?.score).toBe(SCORE_FULL_PREFIX);
    });

    it("单字符查询不参与转写匹配", () => {
        const cjk = forms({ latin: "主歌01.wav", compact: "zhuge01wav", initials: "zg" });
        expect(hit(cjk, "z")).toBeNull();
        // 字面档不受此限。
        expect(hit(forms({ latin: "zebra.wav" }), "z")?.score).toBe(SCORE_LITERAL);
    });

    it("模糊子序列仅在 fuzzy 模式下、且查询至少三个字符", () => {
        const cjk = forms({ latin: "主歌01.wav", compact: "zhuge01wav", initials: "zg" });
        expect(hit(cjk, "zge", "smart")).toBeNull();
        expect(hit(cjk, "zge", "fuzzy")?.score).toBe(SCORE_FUZZY);
        expect(hit(cjk, "zge", "fuzzy")?.kind).toBe("fuzzy");
        // 两字符在 fuzzy 下走的是初声档，不是模糊档。
        expect(hit(cjk, "zg", "fuzzy")?.kind).not.toBe("fuzzy");
    });

    it("off 模式只有字面匹配", () => {
        const cjk = forms({ latin: "主歌01.wav", compact: "zhuge01wav", initials: "zg" });
        expect(hit(cjk, "zhuge", "off")).toBeNull();
        expect(hit(cjk, "主歌", "off")?.score).toBe(SCORE_LITERAL);
    });

    it("空查询不命中", () => {
        expect(hit(forms({ latin: "anything.wav" }), "")).toBeNull();
        expect(hit(forms({ latin: "anything.wav" }), "   ")).toBeNull();
    });

    it("档位严格递减", () => {
        const table = [
            SCORE_LITERAL,
            SCORE_FULL_PREFIX,
            SCORE_FULL_SUBSTRING,
            SCORE_INITIALS_EXACT,
            SCORE_INITIALS_PREFIX,
            SCORE_INITIALS_SUBSEQ,
            SCORE_FUZZY,
        ];
        for (let i = 1; i < table.length; i += 1) {
            expect(table[i - 1]).toBeGreaterThan(table[i]);
        }
    });
});

describe("isSubsequence", () => {
    it("按顺序出现即命中", () => {
        expect(isSubsequence("fgypk", "fg")).toBe(true);
        expect(isSubsequence("fgypk", "fk")).toBe(true);
        expect(isSubsequence("fgypk", "gf")).toBe(false);
        expect(isSubsequence("", "a")).toBe(false);
        expect(isSubsequence("abc", "")).toBe(true);
    });
});

describe("buildQuery", () => {
    it("拉丁查询的 compact 与 initials 相同（缩写直接对上汉字初声串）", () => {
        const query = buildQuery("CX");
        expect(query.literal).toBe("cx");
        expect(query.compact).toBe("cx");
        expect(query.initials).toBe("cx");
    });
});
