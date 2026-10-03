import { describe, expect, test } from "vitest";

import { pointerOverlayStyle } from "./pointerOverlayStyle";

describe("pointerOverlayStyle", () => {
    test("位移全部落在 transform 上，布局位置归零", () => {
        const style = pointerOverlayStyle({ clientX: 600, clientY: 500 }, { left: 56, top: 40 });
        expect(style.left).toBe(0);
        expect(style.top).toBe(0);
        expect(style.transform).toBe("translate3d(544px, 460px, 0) translateY(-100%)");
    });

    /*
     * ★ 核心不变量：**保留小数**，不要量化。
     *
     * 曾经在这里 `Math.round` 到整 CSS 像素，结果是"更糟" —— HiDPI 上 1 CSS 像素比
     * 1 设备像素粗（125% 缩放时 1.25 设备像素），连续位移被切成可见的台阶，浮层里的
     * 静态行开始一跳一跳。清晰度要求落在整**设备**像素上，而这件事由合成器负责。
     */
    test("小数指针坐标原样保留（不取整成整 CSS 像素）", () => {
        const style = pointerOverlayStyle(
            { clientX: 600.4, clientY: 500.6 },
            { left: 56.25, top: 40.75 },
        );
        expect(style.transform).toBe("translate3d(544.15px, 459.85px, 0) translateY(-100%)");
    });

    test("次像素级的移动必须产生次像素级的位移（否则就是量化抖动）", () => {
        const rect = { left: 56.5, top: 532.5 };
        const a = pointerOverlayStyle({ clientX: 600.1, clientY: 600.1 }, rect).transform;
        const b = pointerOverlayStyle({ clientX: 600.4, clientY: 600.4 }, rect).transform;
        expect(a).not.toBe(b);
        expect(b).toContain("543.9px");
        expect(a).toContain("543.6px");
    });

    test("声明 will-change，让浮层成为合成层（文字栅格只做一次）", () => {
        const style = pointerOverlayStyle({ clientX: 0, clientY: 0 }, { left: 0, top: 0 });
        expect(style.willChange).toBe("transform");
    });

    test("非有限坐标退化为 0 而不是产生 NaN 位移", () => {
        const style = pointerOverlayStyle(
            { clientX: Number.NaN, clientY: Number.POSITIVE_INFINITY },
            { left: 10, top: 20 },
        );
        expect(style.transform).toBe("translate3d(0px, 0px, 0) translateY(-100%)");
    });
});
