import { describe, expect, test } from "vitest";

import { pointerOverlayOffset } from "./pointerOverlayOffset";

describe("pointerOverlayOffset", () => {
    test("整数坐标原样通过", () => {
        expect(pointerOverlayOffset({ clientX: 600, clientY: 500 }, { left: 56, top: 40 })).toEqual(
            {
                left: 544,
                top: 460,
            },
        );
    });

    /*
     * ★ 核心不变量：结果必须落在整像素上。
     *
     * `clientX` / `clientY` 在 HiDPI 屏上是小数（devicePixelRatio > 1 时鼠标每移动
     * 一帧都可能落在 0.5 这种位置）。若原样当 `left/top` 用，气泡里的文字每帧换一个
     * 次像素相位、被反复重栅格化 —— 用户看到的就是"气泡跟着指针走，文字却在抖"。
     */
    test("小数坐标取整到整像素", () => {
        const out = pointerOverlayOffset(
            { clientX: 600.4, clientY: 500.6 },
            { left: 56.25, top: 40.75 },
        );
        expect(Number.isInteger(out.left)).toBe(true);
        expect(Number.isInteger(out.top)).toBe(true);
        expect(out).toEqual({ left: 544, top: 460 });
    });

    test("指针只移动小数距离时，画出来的位置不逐帧变化（相位恒定）", () => {
        const rect = { left: 56.5, top: 532.5 };
        const a = pointerOverlayOffset({ clientX: 600.1, clientY: 600.1 }, rect);
        expect(a).toEqual({ left: 544, top: 68 });
        // 次像素级的移动不改变绘制位置 —— 这正是"文字不再抖"的来源。
        expect(pointerOverlayOffset({ clientX: 600.2, clientY: 600.3 }, rect)).toEqual(a);
        expect(pointerOverlayOffset({ clientX: 600.4, clientY: 600.4 }, rect)).toEqual(a);
        // 越过半个像素才前进一格，且仍落在整数上。
        expect(pointerOverlayOffset({ clientX: 601.1, clientY: 601.1 }, rect)).toEqual({
            left: a.left + 1,
            top: a.top + 1,
        });
    });

    test("容器偏移为小数时同样取整（sticky 层被小数滚动量平移的情况）", () => {
        const out = pointerOverlayOffset({ clientX: 100, clientY: 200 }, { left: 0.3, top: 10.7 });
        expect(out).toEqual({ left: 100, top: 189 });
    });
});
