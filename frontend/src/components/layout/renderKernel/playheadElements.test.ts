/**
 * 播放头元素写入器单测。
 *
 * 【本测试要补的盲区】逐帧写 `style` 的去重键若只看位置，就会漏掉"元素在播放头静止
 * 时被重建"——新元素停在静态位置（左缘）不动，用户看到的是"倒三角停在工程起始处"。
 * 本模块的去重键是「元素身份 + 位置」，下面用例钉死这一条以及"槽位互不牵连"。
 */
import { describe, expect, it } from "vitest";

import { createPlayheadElementWriter } from "./playheadElements";

/** 假元素：只记录写入的样式属性。 */
function fakeElement(): { style: { left: string; transform: string } } {
    return { style: { left: "", transform: "" } };
}

function asElement(fake: { style: { left: string; transform: string } }): HTMLElement {
    return fake as unknown as HTMLElement;
}

describe("createPlayheadElementWriter（播放头元素写入器）", () => {
    it("位置未变时不重复写（逐帧去重仍然生效）", () => {
        const writer = createPlayheadElementWriter();
        const line = fakeElement();
        writer.writeLeft("rulerLine", asElement(line), 100);
        expect(line.style.left).toBe("100px");
        line.style.left = "(sentinel)";
        writer.writeLeft("rulerLine", asElement(line), 100.005);
        expect(line.style.left).toBe("(sentinel)");
    });

    it("位置变化超过阈值才写", () => {
        const writer = createPlayheadElementWriter();
        const line = fakeElement();
        writer.writeLeft("rulerLine", asElement(line), 100);
        writer.writeLeft("rulerLine", asElement(line), 140);
        expect(line.style.left).toBe("140px");
    });

    it("★ 元素重建后，即使位置未变也强制写一次（回归：三角停在工程起始处）", () => {
        const writer = createPlayheadElementWriter();
        const first = fakeElement();
        writer.writeLeft("rulerHead", asElement(first), 320);

        // 停靠重排搬动 DOM / 标尺子树重挂载：新元素没有 `left`，位置却与上次相同。
        const rebuilt = fakeElement();
        writer.writeLeft("rulerHead", asElement(rebuilt), 320);
        expect(rebuilt.style.left).toBe("320px");
    });

    it("★ 元素暂不可用时丢掉槽位，重新出现时无条件写（未挂载 → 挂载）", () => {
        const writer = createPlayheadElementWriter();
        const line = fakeElement();
        writer.writeLeft("rulerLine", asElement(line), 320);
        writer.writeLeft("rulerLine", null, 320);

        const remounted = fakeElement();
        writer.writeLeft("rulerLine", asElement(remounted), 320);
        expect(remounted.style.left).toBe("320px");
    });

    it("★ 槽位互不牵连：竖线被去重跳过时，三角仍然会写（回归：嵌套去重）", () => {
        const writer = createPlayheadElementWriter();
        const line = fakeElement();
        const head = fakeElement();
        writer.writeLeft("rulerLine", asElement(line), 500);

        // 同一帧里三角刚重建，位置与竖线上一次的写入值相同。
        writer.writeLeft("rulerLine", asElement(line), 500);
        writer.writeLeft("rulerHead", asElement(head), 500);
        expect(head.style.left).toBe("500px");
    });

    it("writeTranslateX 写 transform（轨道区 / 编辑器主体线）", () => {
        const writer = createPlayheadElementWriter();
        const body = fakeElement();
        writer.writeTranslateX("body", asElement(body), 12.5);
        expect(body.style.transform).toBe("translateX(12.5px)");
        expect(body.style.left).toBe("");
    });

    it("left 与 translateX 是两个独立槽位（同一元素也可各写一次）", () => {
        const writer = createPlayheadElementWriter();
        const element = fakeElement();
        writer.writeLeft("a", asElement(element), 8);
        writer.writeTranslateX("b", asElement(element), 8);
        expect(element.style.left).toBe("8px");
        expect(element.style.transform).toBe("translateX(8px)");
    });

    it("非有限位置不落地（不写 NaNpx / Infinitypx）", () => {
        const writer = createPlayheadElementWriter();
        const line = fakeElement();
        writer.writeLeft("rulerLine", asElement(line), Number.NaN);
        writer.writeLeft("rulerLine", asElement(line), Number.POSITIVE_INFINITY);
        expect(line.style.left).toBe("");
    });

    it("reset 后无条件重写（宿主重建）", () => {
        const writer = createPlayheadElementWriter();
        const line = fakeElement();
        writer.writeLeft("rulerLine", asElement(line), 100);
        line.style.left = "(sentinel)";
        writer.reset();
        writer.writeLeft("rulerLine", asElement(line), 100);
        expect(line.style.left).toBe("100px");
    });
});
