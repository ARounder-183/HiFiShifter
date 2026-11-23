// @vitest-environment jsdom
/**
 * 传输栏按钮原语的回归测试。
 *
 * 【为什么这些用例存在】用户看到的是"录音键的红色填充与播放键不齐"。根因不是颜色，
 * 而是同一行里混用了三套盒模型（文字 Button / 图标 IconButton / 被 Box 包裹的
 * IconButton，且在 soft 与 solid 之间切换）。这里钉住：
 * 1. 三种 fill 渲染出**同一个**盒模型标记；
 * 2. `active` 只切填充，不换盒模型；
 * 3. tooltip 同时是 `aria-label`（图标按钮没有可读文本）；
 * 4. 右键菜单事件可达（录音键的右键是唯一入口）。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { TransportIconButton, type TransportFill } from "./TransportIconButton";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

/**
 * 渲染并**立刻取快照**。
 *
 * 【为什么不能返回元素本身】React 在同一个 root 上重渲染时复用同一个 DOM 节点，
 * 元素引用会随下一次渲染被就地改写 —— 于是"idle"与"active"会变成同一个对象，
 * 断言永远比较的是最后一次的结果（这条用例的第一版正是这样误判的）。
 */
function renderButton(props: {
    fill?: TransportFill;
    active?: boolean;
    tone?: "neutral" | "accent" | "danger";
    onContextMenu?: (event: React.MouseEvent<HTMLButtonElement>) => void;
}): {
    className: string;
    transport: string | undefined;
    toolbarControl: string | undefined;
    ariaPressed: string | null;
    ariaLabel: string | null;
    tooltip: string | null;
    element: HTMLButtonElement;
} {
    act(() => {
        root.render(
            <TransportIconButton icon={<svg data-testid="icon" />} tooltip="play" {...props} />,
        );
    });
    const button = document.querySelector<HTMLButtonElement>("[data-hs-transport-button]");
    if (!button) throw new Error("transport button did not render");
    return {
        className: button.className,
        transport: button.dataset.hsTransportButton,
        toolbarControl: button.dataset.hsToolbarControl,
        ariaPressed: button.getAttribute("aria-pressed"),
        ariaLabel: button.getAttribute("aria-label"),
        tooltip: button.getAttribute("data-tooltip"),
        element: button,
    };
}

test("三种 fill 都渲染同一个盒模型标记", () => {
    for (const fill of ["ghost", "soft", "solid"] as const) {
        const snapshot = renderButton({ fill });
        expect(snapshot.transport).toBe("");
        expect(snapshot.toolbarControl).toBe("");
        expect(snapshot.className).toContain(`rt-variant-${fill}`);
    }
});

test("active 强制实心，但不改变盒模型标记", () => {
    const idle = renderButton({ fill: "soft", active: false });
    const active = renderButton({ fill: "soft", active: true });
    expect(idle.className).toContain("rt-variant-soft");
    expect(active.className).toContain("rt-variant-solid");
    expect(active.transport).toBe(idle.transport);
    expect(active.ariaPressed).toBe("true");
});

test("录音键：固定实心危险色，待机与录制中盒子不变", () => {
    const idle = renderButton({ tone: "danger", fill: "solid", active: false });
    const armed = renderButton({ tone: "danger", fill: "solid", active: true });
    expect(idle.className).toContain("rt-variant-solid");
    expect(armed.className).toContain("rt-variant-solid");
    expect(idle.className).not.toContain("rt-variant-soft");
    expect(armed.className).not.toContain("rt-variant-soft");
});

test("tooltip 同时是 aria-label", () => {
    const snapshot = renderButton({});
    expect(snapshot.ariaLabel).toBe("play");
    expect(snapshot.tooltip).toBe("play");
});

test("右键事件可达（录音菜单的唯一入口）", () => {
    const onContextMenu = vi.fn();
    const snapshot = renderButton({ onContextMenu });
    act(() => {
        snapshot.element.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, cancelable: true }),
        );
    });
    expect(onContextMenu).toHaveBeenCalledTimes(1);
});
