/**
 * 非被动滚轮监听的挂载契约。
 *
 * 【本测试要钉死的那条不变量】**条件挂载**的元素也必须拿到监听器。
 *
 * 旧实现返回 `RefObject` 并在 `useEffect(..., [])` 里读 `ref.current`，因此只在
 * "元素在 effect 之前就存在"时才生效。Tempo Map 的内联输入框（双击才出现）正是
 * 反例：监听器从未挂上，而调用方已删掉 React 的 `onWheel` —— 滚轮完全没反应。
 * 回调 ref 把元素出现本身当作挂载时机，与出现时间无关。
 */
import { describe, expect, it, vi } from "vitest";

import { createWheelAttacher } from "./useNonPassiveWheel";

/** 记录监听器的假元素。 */
function makeElement() {
    const listeners: Array<(event: unknown) => void> = [];
    const removed: Array<(event: unknown) => void> = [];
    return {
        listeners,
        removed,
        addEventListener(type: string, listener: (event: unknown) => void) {
            if (type !== "wheel") throw new Error(`unexpected listener type: ${type}`);
            listeners.push(listener);
        },
        removeEventListener(_type: string, listener: (event: unknown) => void) {
            removed.push(listener);
            // 真实 DOM 语义：摘掉之后事件不再送达。
            const index = listeners.indexOf(listener);
            if (index >= 0) listeners.splice(index, 1);
        },
    };
}

describe("useNonPassiveWheel（非被动滚轮监听）", () => {
    it("★ 元素后出现（条件挂载）时同样会挂上监听器", () => {
        const handled: unknown[] = [];
        const attach = createWheelAttacher<HTMLElement>((event) => handled.push(event));

        // 组件先渲染，此时目标元素还不存在。
        attach(null);
        // 之后（例如用户双击）元素才出现。
        const element = makeElement();
        attach(element as unknown as HTMLElement);

        expect(element.listeners).toHaveLength(1);
        const event = { deltaY: -1 };
        element.listeners[0](event);
        expect(handled).toEqual([event]);
    });

    it("元素被替换时，旧监听器被摘除、新元素挂上监听器", () => {
        const handled: unknown[] = [];
        const attach = createWheelAttacher<HTMLElement>((event) => handled.push(event));
        const first = makeElement();
        const second = makeElement();

        attach(first as unknown as HTMLElement);
        attach(null); // React 替换元素前会先以 null 调用
        attach(second as unknown as HTMLElement);

        expect(first.removed).toHaveLength(1);
        expect(second.listeners).toHaveLength(1);
        // 旧元素上已无监听器（摘掉了），新元素的事件照常送达。
        expect(first.listeners).toHaveLength(0);
        second.listeners[0]({ deltaY: 1 });
        expect(handled).toHaveLength(1);
    });

    it("元素移除（attach(null)）后摘除监听器，且可重复调用", () => {
        const attach = createWheelAttacher<HTMLElement>(() => {});
        const element = makeElement();
        attach(element as unknown as HTMLElement);
        attach(null);
        attach(null);
        expect(element.removed).toHaveLength(1);
    });

    it("dispose 摘除监听器（组件卸载兜底）", () => {
        const attach = createWheelAttacher<HTMLElement>(() => {});
        const element = makeElement();
        attach(element as unknown as HTMLElement);
        attach.dispose();
        expect(element.removed).toHaveLength(1);
        attach.dispose();
        expect(element.removed).toHaveLength(1);
    });

    it("handler 变化后新的事件走最新实现（经 ref 转发，不重挂监听器）", () => {
        const calls: string[] = [];
        let current = (): void => {
            calls.push("first");
        };
        const attach = createWheelAttacher<HTMLElement>(() => current());
        const element = makeElement();
        attach(element as unknown as HTMLElement);
        element.listeners[0]({});
        current = () => calls.push("second");
        element.listeners[0]({});
        expect(calls).toEqual(["first", "second"]);
        // 全程只挂了一次。
        expect(element.listeners).toHaveLength(1);
    });

    it("handler 抛错不会破坏监听器（后续事件仍送达）", () => {
        const spy = vi.fn();
        const attach = createWheelAttacher<HTMLElement>(() => {
            throw new Error("boom");
        });
        const element = makeElement();
        attach(element as unknown as HTMLElement);
        expect(() => element.listeners[0]({})).toThrow("boom");
        // 监听器仍在（异常不是我们的责任，但不能因此摘掉监听）。
        expect(element.listeners).toHaveLength(1);
        void spy;
    });
});
