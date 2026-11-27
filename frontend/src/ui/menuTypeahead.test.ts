import { describe, expect, test } from "vitest";

import {
    EMPTY_MENU_TYPEAHEAD,
    MENU_TYPEAHEAD_RESET_MS,
    matchMenuTypeahead,
    menuTypeaheadChar,
    pushMenuTypeahead,
} from "./menuTypeahead";

const LABELS = ["新建工程", "打开工程", "Open Recent", "Save", "Save As", "Close"];

describe("matchMenuTypeahead", () => {
    test("首字母跳到第一个匹配项", () => {
        expect(matchMenuTypeahead(LABELS, "s", -1)).toBe(3);
    });

    test("从当前项之后环形找：连按同一个字母在几个候选之间循环", () => {
        expect(matchMenuTypeahead(LABELS, "s", 3)).toBe(4);
        expect(matchMenuTypeahead(LABELS, "s", 4)).toBe(3);
    });

    test("拼出前缀后收窄（'sa' 只剩 Save / Save As）", () => {
        expect(matchMenuTypeahead(LABELS, "sa", -1)).toBe(3);
        // 从 Save 往后找，下一个以 "sa" 开头的是 Save As —— 前缀没有唯一化时
        // 连按会继续循环，而不是停在原地。
        expect(matchMenuTypeahead(LABELS, "sa", 3)).toBe(4);
        expect(matchMenuTypeahead(LABELS, "save a", -1)).toBe(4);
    });

    test("无匹配 ⇒ -1（调用侧据此不动，而不是跳回第一项）", () => {
        expect(matchMenuTypeahead(LABELS, "zz", 0)).toBe(-1);
    });

    test("大小写不敏感，中文标签按字符前缀匹配", () => {
        expect(matchMenuTypeahead(LABELS, "OPEN", -1)).toBe(2);
        expect(matchMenuTypeahead(LABELS, "打", -1)).toBe(1);
    });

    test("空数组 / 空前缀都返回 -1", () => {
        expect(matchMenuTypeahead([], "a", -1)).toBe(-1);
        expect(matchMenuTypeahead(LABELS, "", -1)).toBe(-1);
    });
});

describe("pushMenuTypeahead", () => {
    test("首字符开始新缓冲", () => {
        expect(pushMenuTypeahead(EMPTY_MENU_TYPEAHEAD, "s", 1000)).toEqual({
            query: "s",
            at: 1000,
        });
    });

    test("间隔内追加成前缀", () => {
        const first = pushMenuTypeahead(EMPTY_MENU_TYPEAHEAD, "s", 1000);
        expect(pushMenuTypeahead(first, "a", 1200)).toEqual({ query: "sa", at: 1200 });
    });

    test("超过重置窗口后重新开始", () => {
        const first = pushMenuTypeahead(EMPTY_MENU_TYPEAHEAD, "s", 1000);
        expect(pushMenuTypeahead(first, "a", 1000 + MENU_TYPEAHEAD_RESET_MS + 1)).toEqual({
            query: "a",
            at: 1000 + MENU_TYPEAHEAD_RESET_MS + 1,
        });
    });

    test("同一个字符连按不拼成 'ss'（否则循环跳项永远失效）", () => {
        const first = pushMenuTypeahead(EMPTY_MENU_TYPEAHEAD, "s", 1000);
        expect(pushMenuTypeahead(first, "s", 1100).query).toBe("s");
    });
});

describe("menuTypeaheadChar", () => {
    test("单字符进缓冲", () => {
        expect(menuTypeaheadChar("a", {})).toBe("a");
        expect(menuTypeaheadChar("A", {})).toBe("a");
    });

    test("空格、组合键、功能键都不进缓冲", () => {
        expect(menuTypeaheadChar(" ", {})).toBeNull();
        expect(menuTypeaheadChar("a", { ctrlKey: true })).toBeNull();
        expect(menuTypeaheadChar("a", { metaKey: true })).toBeNull();
        expect(menuTypeaheadChar("ArrowDown", {})).toBeNull();
        expect(menuTypeaheadChar("Enter", {})).toBeNull();
    });
});
