/*
 * 格式化层测试。
 *
 * 这一层是"文案不再手写拼接"的机制保证，因此测试锁的是**边界行为**：
 * 缺占位符时的降级、复数规则、以及不被识别的语系标识。
 */
import { describe, expect, test } from "vitest";

import {
    PLURAL_SEPARATOR,
    formatNumber,
    formatShortcutLabel,
    formatTemplate,
    formatUnit,
    primaryModifierLabel,
    selectPluralForm,
} from "./format";

describe("formatTemplate", () => {
    test("替换全部出现的占位符", () => {
        expect(formatTemplate("{a} and {a}", { a: "x" })).toBe("x and x");
    });

    test("数字会被转成字符串", () => {
        expect(formatTemplate("{n} clips", { n: 3 })).toBe("3 clips");
    });

    test("未提供的占位符原样保留", () => {
        // 宁可让缺失可见，也不要静默显示一个残缺的句子
        expect(formatTemplate("{a} {b}", { a: "x" })).toBe("x {b}");
    });

    test("值本身含花括号时不会被二次展开", () => {
        expect(formatTemplate("{a}", { a: "{b}" })).toBe("{b}");
    });
});

describe("selectPluralForm", () => {
    test("英语按 1 / 非 1 选形态", () => {
        expect(selectPluralForm("en-US", 1, "clip|clips")).toBe("clip");
        expect(selectPluralForm("en-US", 0, "clip|clips")).toBe("clips");
        expect(selectPluralForm("en-US", 2, "clip|clips")).toBe("clips");
    });

    test("单形态表示该语言不区分复数（中日韩）", () => {
        for (const locale of ["zh-CN", "zh-TW", "ja-JP", "ko-KR"]) {
            expect(selectPluralForm(locale, 1, "个音频块")).toBe("个音频块");
            expect(selectPluralForm(locale, 5, "个音频块")).toBe("个音频块");
        }
    });

    test("无法识别的语系标识不抛错", () => {
        /*
         * 注意：形如 `xx-YY` 的标识在结构上是合法的，`Intl.PluralRules` 会
         * 静默解析成引擎默认语系（而不是抛错），因此不能断言它一定退回英语规则。
         * 这里只锁"不抛错且返回两种形态之一"这一条 —— 目录里所有语系标识都是
         * 真实的，这条路径只用于防御。
         */
        expect(() => selectPluralForm("xx-YY", 1, "clip|clips")).not.toThrow();
        expect(["clip", "clips"]).toContain(selectPluralForm("xx-YY", 1, "clip|clips"));
    });

    test("第二个形态取第一个分隔符之后的全部内容", () => {
        expect(PLURAL_SEPARATOR).toBe("|");
        expect(selectPluralForm("en-US", 1, "a|b|c")).toBe("a");
        expect(selectPluralForm("en-US", 2, "a|b|c")).toBe("b|c");
        // 词典门禁（catalogIntegrity）保证实际不会出现多个分隔符
    });
});

describe("formatShortcutLabel", () => {
    test("把 {modifier} 换成当前平台的主修饰键", () => {
        const modifier = primaryModifierLabel();
        expect(modifier === "⌘" || modifier === "Ctrl").toBe(true);
        expect(formatShortcutLabel("Bold ({modifier}+B)")).toBe(`Bold (${modifier}+B)`);
    });

    test("没有占位符时原样返回", () => {
        expect(formatShortcutLabel("Bold")).toBe("Bold");
    });
});

describe("formatUnit", () => {
    test("按语系输出数字与单位", () => {
        /*
         * 不断言缩写形态（`kHz`）：短单位名来自 ICU 数据，Node 的 small-icu
         * 构建里 `kilohertz` 会输出全称。因此只锁"包含数字与单位"。
         */
        const out = formatUnit("en-US", 48, "kilohertz");
        expect(out).toContain("48");
        expect(out.toLowerCase()).toMatch(/hz|kilohertz/);
    });

    test("非法单位标识时降级为「数字 + 单位名」而不是抛错", () => {
        expect(formatUnit("en-US", 5, "definitely-not-a-unit")).toBe("5 definitely-not-a-unit");
    });
});

describe("formatNumber", () => {
    test("按语系分组", () => {
        expect(formatNumber("en-US", 1234567)).toBe("1,234,567");
    });
});
