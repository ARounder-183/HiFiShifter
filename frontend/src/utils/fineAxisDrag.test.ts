import { describe, expect, test } from "vitest";

import { advanceFineAxisDrag, createFineAxisDragState } from "./fineAxisDrag";

describe("createFineAxisDragState", () => {
    test("raw 与 adjusted 同起点（否则首帧会凭空产生位移）", () => {
        expect(createFineAxisDragState(120, false)).toEqual({
            raw: 120,
            adjusted: 120,
            fineActive: false,
        });
    });
});

describe("advanceFineAxisDrag", () => {
    test("未按修饰键：累计位移等于原始位移", () => {
        const state = createFineAxisDragState(0, false);
        expect(advanceFineAxisDrag(state, 10, false)).toBe(10);
        expect(advanceFineAxisDrag(state, 25, false)).toBe(25);
    });

    test("全程按住修饰键：累计位移按比例缩小", () => {
        const state = createFineAxisDragState(0, true);
        // 0.2 倍。
        expect(advanceFineAxisDrag(state, 10, true)).toBeCloseTo(2, 9);
        expect(advanceFineAxisDrag(state, 25, true)).toBeCloseTo(5, 9);
    });

    /*
     * ★ 核心不变量：修饰键中途按下 / 松开时，累计量必须**连续**。
     *
     * 故障形态：每帧用"从起点算起的总位移 × 当前比例"重算 —— 中途一按 Ctrl，
     * 整个已累计的位移被重新缩小，数值瞬间跳回去（用户看到的"闪回"），正在
     * 进行的拖拽被打断。正确做法是只缩放**增量**。
     */
    test("中途按下修饰键：累计量不跳，只减慢此后的速度", () => {
        const state = createFineAxisDragState(0, false);
        const before = advanceFineAxisDrag(state, 100, false);
        expect(before).toBe(100);
        // 按下 Ctrl 的瞬间：累计量必须还在 100 附近，绝不回退。
        const atToggle = advanceFineAxisDrag(state, 100, true);
        expect(atToggle).toBeGreaterThanOrEqual(before);
        expect(atToggle).toBeCloseTo(before, 9);
        // 此后每帧只走 0.2 倍。
        const after = advanceFineAxisDrag(state, 110, true);
        expect(after).toBeCloseTo(before + 10 * 0.2, 9);
    });

    test("中途松开修饰键：累计量不跳，只恢复此后的速度", () => {
        const state = createFineAxisDragState(0, true);
        const before = advanceFineAxisDrag(state, 100, true);
        expect(before).toBeCloseTo(20, 9);
        const atToggle = advanceFineAxisDrag(state, 100, false);
        expect(atToggle).toBeCloseTo(before, 9);
        const after = advanceFineAxisDrag(state, 110, false);
        expect(after).toBeCloseTo(before + 10, 9);
    });

    test("起手就按住修饰键：不按「刚按下」的过渡比例处理", () => {
        const state = createFineAxisDragState(0, true);
        // 若误判成"刚按下"，首帧会走 0.65 而不是 0.2。
        expect(advanceFineAxisDrag(state, 100, true)).toBeCloseTo(20, 9);
    });

    test("刚按下那一帧走过渡比例（比稳态粗，避免突然拽不动的顿挫）", () => {
        const state = createFineAxisDragState(0, false);
        advanceFineAxisDrag(state, 0, false);
        const transition = advanceFineAxisDrag(state, 100, true);
        expect(transition).toBeGreaterThan(100 * 0.2);
        expect(transition).toBeLessThan(100);
        // 下一帧回到稳态比例。
        const steady = advanceFineAxisDrag(state, 200, true);
        expect(steady).toBeCloseTo(transition + 100 * 0.2, 9);
    });

    test("增量喂法：累计 delta 与绝对坐标等价", () => {
        const absolute = createFineAxisDragState(500, false);
        const relative = createFineAxisDragState(0, false);
        // 绝对坐标从 500 走到 520、540；累计位移从 0 走到 20、40。
        expect(advanceFineAxisDrag(absolute, 520, false)).toBe(520);
        expect(advanceFineAxisDrag(relative, 20, false)).toBe(20);
        expect(advanceFineAxisDrag(absolute, 540, false) - 500).toBe(40);
        expect(advanceFineAxisDrag(relative, 40, false)).toBe(40);
    });
});
