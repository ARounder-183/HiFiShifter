/**
 * 渲染内核 · 竖向键盘滚动目标解析单测（时间轴与参数编辑器共用）。
 *
 * 【本测试要钉住的核心不变量】
 * 翻页量必须与**迁移前原生滚动**的行为逐值一致（实测 5 组视口高度，误差 0）。
 * 内核接管键盘后若步进变了，用户会立刻感到"翻页翻多了/翻少了"，而这属于手感变更，
 * 不在迁移范围内。
 */
import { describe, expect, it } from "vitest";

import {
    PAGE_SCROLL_MIN_FRACTION,
    PAGE_SCROLL_OVERLAP_PX,
    resolveKeyboardScrollTarget,
    resolvePageScrollStepPx,
} from "./keyboardScroll";

describe("resolveKeyboardScrollTarget（竖向键盘滚动）", () => {
    const base = { scrollTopPx: 533.5, viewportHeightPx: 823, maxScrollTopPx: 1600 };

    it("翻页量在高视口下 = 视口高 − 20（与 Chromium 原生逐值一致）", () => {
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

    it("★ 矮视口下翻页量由占比项接管（只减 20 会少 1px，与原生的位移不一致）", () => {
        // 【为什么单列一条】上面那批基线全在 H ≥ 323，那里 `H − 20` 恰好更大，
        // 因此"只减 20"这个不完整实现也能全绿——测试没有判别力。
        // 时间轴的轨道头视口**实测只有 151px 高**，这个分支会被真的走到：
        //   H=151 → 原生实测 132，而 151 − 20 = 131（差 1px）
        // 原生取的是 max(H − 20, floor(0.875H))，与下面全部实测吻合。
        expect(PAGE_SCROLL_MIN_FRACTION).toBe(0.875);
        expect(resolvePageScrollStepPx(151)).toBe(132); // timeline legacy 实测
        expect(resolvePageScrollStepPx(151)).not.toBe(151 - 20);

        // 其余真实应用的实测基线（参数编辑器 legacy，误差 0）
        const measured: ReadonlyArray<readonly [number, number]> = [
            [823, 803],
            [623, 603],
            [423, 403],
            [323, 303],
            [1023, 1003],
        ];
        for (const [h, step] of measured) {
            expect(resolvePageScrollStepPx(h)).toBe(step);
        }
        // 端到端：矮视口下 PageDown 的落点
        expect(
            resolveKeyboardScrollTarget({ ...base, key: "PageDown", viewportHeightPx: 151 }),
        ).toBeCloseTo(533.5 + 132, 6);
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
        for (const viewportHeightPx of [10, 19, 20, 0, -5]) {
            const step = resolvePageScrollStepPx(viewportHeightPx);
            expect(step).toBeGreaterThanOrEqual(0);
            const target = resolveKeyboardScrollTarget({
                ...base,
                key: "PageDown",
                viewportHeightPx,
            }) as number;
            expect(target).toBeGreaterThanOrEqual(base.scrollTopPx);
        }
        // H=10 时占比项给出 floor(8.75)=8（原生同样会走这一步），不是 0。
        expect(resolvePageScrollStepPx(10)).toBe(8);
    });

    it("非法视口高不产生 NaN 步进（NaN 传染会让滚动位置失效）", () => {
        expect(resolvePageScrollStepPx(Number.NaN)).toBe(0);
        expect(resolvePageScrollStepPx(Number.POSITIVE_INFINITY)).toBe(0);
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
