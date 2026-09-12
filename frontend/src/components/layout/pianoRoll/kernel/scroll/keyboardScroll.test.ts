/**
 * 参数编辑器内核 · 竖向键盘滚动目标解析单测。
 *
 * 【本测试要钉住的核心不变量】
 * 翻页量必须与**迁移前原生滚动**的行为逐值一致（实测 5 组视口高度，误差 0）。
 * 内核接管键盘后若步进变了，用户会立刻感到"翻页翻多了/翻少了"，而这属于手感变更，
 * 不在迁移范围内。
 */
import { describe, expect, it } from "vitest";

import { PAGE_SCROLL_OVERLAP_PX, resolveKeyboardScrollTarget } from "./keyboardScroll";

describe("resolveKeyboardScrollTarget（竖向键盘滚动）", () => {
    const base = { scrollTopPx: 533.5, viewportHeightPx: 823, maxScrollTopPx: 1600 };

    it("翻页量 = 视口高 − 20（与 Chromium 原生逐值一致）", () => {
        expect(PAGE_SCROLL_OVERLAP_PX).toBe(20);
        // 实测基线（旧实现走原生滚动）：视口高 823 → 803、623 → 603、323 → 303。
        expect(resolveKeyboardScrollTarget({ ...base, key: "PageDown" })).toBeCloseTo(
            533.5 + 803,
            6,
        );
        expect(
            resolveKeyboardScrollTarget({ ...base, key: "PageDown", viewportHeightPx: 623 }),
        ).toBeCloseTo(533.5 + 603, 6);
        expect(
            resolveKeyboardScrollTarget({ ...base, key: "PageDown", viewportHeightPx: 323 }),
        ).toBeCloseTo(533.5 + 303, 6);
    });

    it("PageUp 与 PageDown 反向等距", () => {
        const down = resolveKeyboardScrollTarget({ ...base, key: "PageDown" }) as number;
        const up = resolveKeyboardScrollTarget({ ...base, key: "PageUp" }) as number;
        expect(up).toBeCloseTo(base.scrollTopPx - (down - base.scrollTopPx), 6);
    });

    it("Home → 0，End → 上限（与内核同源，不由本函数夹取）", () => {
        expect(resolveKeyboardScrollTarget({ ...base, key: "Home" })).toBe(0);
        expect(resolveKeyboardScrollTarget({ ...base, key: "End" })).toBe(1600);
    });

    it("按键名大小写不敏感（KeyboardEvent.key 的实测取值）", () => {
        expect(resolveKeyboardScrollTarget({ ...base, key: "PageDown" })).toBe(
            resolveKeyboardScrollTarget({ ...base, key: "pagedown" }),
        );
        expect(resolveKeyboardScrollTarget({ ...base, key: "END" })).toBe(1600);
    });

    it("不相关的按键返回 null（调用方不得 preventDefault）", () => {
        for (const key of ["ArrowDown", "ArrowLeft", "a", " ", "Escape", "", "F5"]) {
            expect(resolveKeyboardScrollTarget({ ...base, key })).toBeNull();
        }
    });

    it("视口极矮时不产生负步进（PageDown 不得往回翻）", () => {
        const target = resolveKeyboardScrollTarget({
            ...base,
            key: "PageDown",
            viewportHeightPx: 10,
        }) as number;
        expect(target).toBeGreaterThanOrEqual(base.scrollTopPx);
        expect(target).toBeCloseTo(base.scrollTopPx, 6);
    });

    it("目标可以越界（钳制由内核统一负责，本函数不夹取）", () => {
        // PageDown 落在上限之外：必须原样返回，不能在这里夹到 1600——
        // 否则会出现两份上限来源（见函数说明的特殊说明 1）。
        const target = resolveKeyboardScrollTarget({
            scrollTopPx: 1500,
            viewportHeightPx: 823,
            maxScrollTopPx: 1600,
            key: "PageDown",
        }) as number;
        expect(target).toBeCloseTo(1500 + 803, 6);
        // PageUp 同理（可为负）。
        expect(
            resolveKeyboardScrollTarget({
                scrollTopPx: 100,
                viewportHeightPx: 823,
                maxScrollTopPx: 1600,
                key: "PageUp",
            }),
        ).toBeCloseTo(100 - 803, 6);
    });
});
