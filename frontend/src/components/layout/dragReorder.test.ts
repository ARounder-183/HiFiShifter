import { describe, expect, test } from "vitest";

import {
    REORDER_DRAG_THRESHOLD_PX,
    reorderInsertionIndex,
    reorderTargetIndex,
} from "./dragReorder";

/*
 * 拖拽排序的落点换算。
 *
 * 【为什么值得测】下标差一位不会抛错，只会让"拖到某处跳一格"。这里同时钉住"推过半行
 * 才换位"的手感与"当前顺序 → 移除后下标"的换算。
 */
describe("reorderInsertionIndex", () => {
    // 三行，中线分别在 10 / 30 / 50。
    const centers = [10, 30, 50];

    test("指针在第一行中线之上 → 插到最前", () => {
        expect(reorderInsertionIndex(centers, 0)).toBe(0);
        expect(reorderInsertionIndex(centers, 9)).toBe(0);
    });

    test("推过中线才落到下一行之前", () => {
        // 中线 10 之下、中线 30 之上 → 插到第 2 行之前。
        expect(reorderInsertionIndex(centers, 11)).toBe(1);
        expect(reorderInsertionIndex(centers, 29)).toBe(1);
        expect(reorderInsertionIndex(centers, 31)).toBe(2);
    });

    test("越过最后一行中线 → 插到末尾", () => {
        expect(reorderInsertionIndex(centers, 51)).toBe(3);
        expect(reorderInsertionIndex(centers, 9999)).toBe(3);
    });

    test("空列表返回 0（不会越界）", () => {
        expect(reorderInsertionIndex([], 100)).toBe(0);
    });
});

describe("reorderTargetIndex", () => {
    /*
     * `reorderUserVibratoPresets` 的下标是**移除被拖项之后**的数组下标，
     * 而插入位置说的是当前顺序 —— 被拖项之后的位置要减一。
     */
    test("插入位置在被拖项之后：减一", () => {
        // [a,b,c]，拖 a（下标 0）到 c 之前（插入位置 2）→ 移除后下标 1 → [b,a,c]。
        expect(reorderTargetIndex(2, 0)).toBe(1);
    });

    test("插入位置在被拖项之前或原地：不变", () => {
        expect(reorderTargetIndex(0, 2)).toBe(0);
        expect(reorderTargetIndex(1, 1)).toBe(1);
    });

    test("拖回原位得到原下标（等于不动）", () => {
        // 拖 b（下标 1）到 b 自己之前（插入位置 1）→ 下标 1 → 不变。
        expect(reorderTargetIndex(1, 1)).toBe(1);
        // 拖 b 到 c 之前（插入位置 2）→ 下标 1 → 仍是原地。
        expect(reorderTargetIndex(2, 1)).toBe(1);
    });

    test("落到末尾", () => {
        // [a,b,c] 拖 a 到最后（插入位置 3）→ 下标 2 → [b,c,a]。
        expect(reorderTargetIndex(3, 0)).toBe(2);
    });
});

describe("REORDER_DRAG_THRESHOLD_PX", () => {
    test("阈值小而可感：够区分点击与拖拽，又不会迟钝", () => {
        expect(REORDER_DRAG_THRESHOLD_PX).toBeGreaterThan(0);
        expect(REORDER_DRAG_THRESHOLD_PX).toBeLessThanOrEqual(8);
    });
});
