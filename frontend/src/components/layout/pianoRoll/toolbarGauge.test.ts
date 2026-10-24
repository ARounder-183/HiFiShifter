// @vitest-environment jsdom
/*
 * 工具栏测量的契约：**预算固定、判据单一、失败可回退**。
 *
 * 【为什么能在 jsdom 里测】jsdom 没有布局引擎，`clientWidth/scrollWidth` 恒为 0。这里用
 * 合成宽度（`Object.defineProperty` 装一个 getter）把"某一级的可见内容有多宽"喂进去 ——
 * 于是被测的是**算法**（预算怎么取、表怎么截断、结束时把行留在哪一级），而不是浏览器
 * 的计算结果。浏览器里的一致性由手测确认（见方案文档第 3 节）。
 */

import { describe, expect, it } from "vitest";

import { chooseTier, layoutSignature, measureTierFits, readSlotRange } from "./toolbarGauge";

type SlotSpec = {
    /** 达到这一级隐藏。 */
    hideAt?: number;
    /** 从这一级起才显示。 */
    showAt?: number;
    /** 可见时占的宽度。 */
    width: number;
};

/**
 * 造一行"宽度可计算"的工具栏。
 *
 * - 行本身 `clientWidth === budget`（= 可用宽度），`scrollWidth` = max(内容, budget)；
 * - 行里只有一个直接子项（模拟左 / 右簇），它的 `clientWidth/scrollWidth` 按同一套内容
 *   宽度算 —— `fits` 的两条判据因此都成立。
 */
function buildRow(specs: readonly SlotSpec[], budget: number) {
    const row = document.createElement("div");
    const cluster = document.createElement("div");
    row.appendChild(cluster);

    const slots = specs.map((spec) => {
        const slot = document.createElement("span");
        if (spec.hideAt !== undefined) slot.dataset.hideAt = String(spec.hideAt);
        if (spec.showAt !== undefined) slot.dataset.showAt = String(spec.showAt);
        cluster.appendChild(slot);
        return { slot, width: spec.width };
    });

    const contentWidth = () =>
        slots.reduce(
            (sum, entry) => sum + (entry.slot.style.display === "none" ? 0 : entry.width),
            0,
        );

    Object.defineProperty(row, "clientWidth", { configurable: true, get: () => budget });
    Object.defineProperty(row, "scrollWidth", {
        configurable: true,
        get: () => Math.max(contentWidth(), budget),
    });
    // 簇是 flex: 1 1 auto 的拉伸项：拿到的是整份预算，内容超出时才被压到内容宽度
    // 以下（即 scrollWidth > clientWidth）—— 这正是 fits 的第二条判据。
    Object.defineProperty(cluster, "clientWidth", { configurable: true, get: () => budget });
    Object.defineProperty(cluster, "scrollWidth", {
        configurable: true,
        get: () => Math.max(contentWidth(), budget),
    });

    return { row, slots, contentWidth };
}

describe("measureTierFits", () => {
    it("按固定预算逐级试显隐，返回表并停在「要呈现的那一级」", () => {
        // 500 放不下（550），藏掉第 1 级那一项就放得下（450）。
        const { row, slots } = buildRow(
            [
                { hideAt: 1, width: 100 },
                { hideAt: 2, width: 100 },
                { hideAt: 3, width: 100 },
                { width: 250 },
            ],
            500,
        );

        const table = measureTierFits(row, 3, 0);

        expect(table).toEqual([false, true]);
        expect(chooseTier(table)).toBe(1);
        // 行被留在第 1 级：第 1 级隐藏，其余显示（不能停在"最后试过的那一级"）。
        expect(slots[0].slot.style.display).toBe("none");
        expect(slots[1].slot.style.display).toBe("contents");
        expect(slots[2].slot.style.display).toBe("contents");
    });

    it("第 0 级就放得下时立刻停止（只试一次，不是十二次）", () => {
        const { row, slots } = buildRow(
            [
                { hideAt: 1, width: 100 },
                { hideAt: 2, width: 100 },
            ],
            4000,
        );

        expect(measureTierFits(row, 11, 5)).toEqual([true]);
        expect(slots.every((entry) => entry.slot.style.display === "contents")).toBe(true);
    });

    it("全放不下时停在最大级（交给 overflow 裁掉，而不是永远收缩）", () => {
        const { row, slots } = buildRow(
            [
                { hideAt: 1, width: 400 },
                { hideAt: 2, width: 400 },
                // 固定底宽本身就超预算：即便全藏也放不下。
                { width: 600 },
            ],
            500,
        );

        const table = measureTierFits(row, 2, 0);
        expect(table).toEqual([false, false, false]);
        expect(chooseTier(table)).toBe(2);
        // 有 hideAt 的两个槽位都藏起来了；没有 hideAt 的底宽槽位不参与让位。
        expect(
            slots
                .filter((entry) => entry.slot.dataset.hideAt !== undefined)
                .every((entry) => entry.slot.style.display === "none"),
        ).toBe(true);
    });

    it("showAt 槽位只在它该出现的层级参与宽度", () => {
        // base 400 + 「⋯」触发器 150~250；静态参数在第 10 级被收走（300）。
        const wideTrigger = buildRow(
            [{ hideAt: 10, width: 300 }, { showAt: 10, width: 150 }, { width: 400 }],
            620,
        );
        // 第 10 级：400 + 150 = 550 ≤ 620 ⇒ 放得下。
        expect(measureTierFits(wideTrigger.row, 11, 0)).toEqual([
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
        ]);

        const tooWide = buildRow(
            [{ hideAt: 10, width: 300 }, { showAt: 10, width: 250 }, { width: 400 }],
            620,
        );
        // 触发器更宽 ⇒ 第 10 级也不放得下（650 > 620），只能到第 11 级。
        expect(chooseTier(measureTierFits(tooWide.row, 11, 0))).toBe(11);
    });

    it("行宽在测量途中变了就作废，并把行留在进入时的层级", () => {
        const { row, slots } = buildRow(
            [
                { hideAt: 1, width: 100 },
                { hideAt: 2, width: 100 },
            ],
            500,
        );
        // 第一次读（取预算）= 500，之后立刻变成 501：模拟"行被内容撑开 / 出现滚动条"。
        let reads = 0;
        Object.defineProperty(row, "clientWidth", {
            configurable: true,
            get: () => (reads++ === 0 ? 500 : 501),
        });

        expect(measureTierFits(row, 2, 1)).toEqual([]);
        // 回退到 currentTier = 1：第 1 级隐藏、第 2 级显示。
        // （若错写成"半张表的答案"，会停在别的层级上，且 React 永不修正。）
        expect(slots[0].slot.style.display).toBe("none");
        expect(slots[1].slot.style.display).toBe("contents");
    });

    it("读取槽位区间：缺省属性当「没写」", () => {
        const bare = document.createElement("span");
        expect(readSlotRange(bare)).toEqual({ hideAt: null, showAt: null });

        const full = document.createElement("span");
        full.dataset.hideAt = "3";
        full.dataset.showAt = "10";
        expect(readSlotRange(full)).toEqual({ hideAt: 3, showAt: 10 });

        const junk = document.createElement("span");
        junk.dataset.hideAt = "abc";
        expect(readSlotRange(junk).hideAt).toBeNull();
    });
});

describe("layoutSignature", () => {
    it("把显隐区间与行的直接子节点数算进签名（否则该重测时不重测）", () => {
        const { row } = buildRow([{ hideAt: 3, width: 10 }], 500);
        const before = layoutSignature(row);

        row.querySelector<HTMLElement>("[data-hide-at]")!.dataset.showAt = "10";
        expect(layoutSignature(row)).not.toBe(before);

        const afterRange = layoutSignature(row);
        row.appendChild(document.createElement("div"));
        expect(layoutSignature(row)).not.toBe(afterRange);
    });
});
