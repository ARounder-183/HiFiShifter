// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test } from "vitest";

import { SlotText, splitDigitRuns } from "./SlotText";

let host: HTMLDivElement;
let root: Root;
let originalRect: typeof Element.prototype.getBoundingClientRect;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    originalRect = Element.prototype.getBoundingClientRect;
    // jsdom 没有布局引擎：按 textContent 长度伪造一个稳定的宽度，让隐藏探针
    // 能测到 > 0 的宽度（生产实现依赖真实字体度量）。
    Element.prototype.getBoundingClientRect = function () {
        const width = (this.textContent?.length ?? 0) * 7;
        return {
            x: 0,
            y: 0,
            top: 0,
            left: 0,
            right: width,
            bottom: 0,
            width,
            height: 12,
            toJSON: () => ({}),
        } as DOMRect;
    };
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
    Element.prototype.getBoundingClientRect = originalRect;
});

test("splitDigitRuns 按连续数字分段，符号与分隔符保持自然宽度", () => {
    expect(splitDigitRuns("12.3.456")).toEqual([
        { text: "12", digits: true },
        { text: ".", digits: false },
        { text: "3", digits: true },
        { text: ".", digits: false },
        { text: "456", digits: true },
    ]);
    expect(splitDigitRuns("-3")).toEqual([
        { text: "-", digits: false },
        { text: "3", digits: true },
    ]);
    expect(splitDigitRuns("1:23.456")).toEqual([
        { text: "1", digits: true },
        { text: ":", digits: false },
        { text: "23", digits: true },
        { text: ".", digits: false },
        { text: "456", digits: true },
    ]);
});

test("textContent 保持原文，预留文本只存在于 data 属性与隐藏探针", () => {
    act(() => {
        root.render(<SlotText text="12.3s" reserveText="1800.0s" data-testid="slot" />);
    });
    const content = host.querySelector<HTMLElement>("[data-slot-text]");
    expect(content?.textContent).toBe("12.3s");
    expect(content?.getAttribute("data-reserve-text")).toBe("1800.0s");
    const container = host.querySelector<HTMLElement>("[data-slot-text-container]");
    expect(container?.getAttribute("data-testid")).toBe("slot");
});

test("数字被拆进固定宽度槽位，非数字仍为单个文本节点", () => {
    act(() => {
        root.render(<SlotText text="12.3s" reserveText="1800.0s" />);
    });
    const content = host.querySelector<HTMLElement>("[data-slot-text]")!;
    const fixed = [...content.querySelectorAll<HTMLElement>("span[style*='width']")];
    expect(fixed.length).toBe(3);
    expect(fixed.every((span) => span.style.width === "7px")).toBe(true);
    expect(content.textContent).toBe("12.3s");
});

test("reserveText 比 text 长时容器按 reserveText 预留宽度", () => {
    act(() => {
        root.render(<SlotText text="9.9s" reserveText="1800.0s" />);
    });
    const container = host.querySelector<HTMLElement>("[data-slot-text-container]")!;
    // "1800.0s" 长度为 7：7 × 7 + 2px 墨迹缓冲。
    expect(Number.parseFloat(container.style.minWidth)).toBeGreaterThanOrEqual(51);
});
