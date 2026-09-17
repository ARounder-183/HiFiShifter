/**
 * 右键拖拽收尾守卫单测（`./rightDragContextMenuGuard`）。
 *
 * 【要锁住的回归】在任一处右键拖拽（时间轴框选音频块 / 参数编辑器右键拖拽），
 * 把指针移到**另一个表面**（轨道头 / 时间轴标尺 / 参数编辑器标尺）再松开右键时，
 * 会弹出那个表面的右键菜单。根因是抑制标记只覆盖发起手势的表面，而菜单由
 * **松开时指针所在的元素**决定。
 *
 * 本测试在 node 环境用记录型 window 桩驱动真实的事件流程（无 jsdom），覆盖：
 * 1. 超过阈值 = 拖拽 → 随后的 contextmenu 被吞掉；
 * 2. 未超过阈值 = 点击 → contextmenu 不被吞（正常弹菜单）；
 * 3. 武装只生效一次；
 * 4. `pointerdown` 无条件复位（下一次真正的右键点击不被误吞）；
 * 5. 失焦复位。
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
    armRightDragContextMenuGuard,
    disposeRightDragContextMenuGuard,
    installRightDragContextMenuGuard,
    isRightDragBeyondThreshold,
    shouldSwallowContextMenu,
} from "./rightDragContextMenuGuard";

describe("isRightDragBeyondThreshold（纯判定）", () => {
    it("恰好达到阈值算拖拽（与时间轴框选阈值同源语义）", () => {
        expect(isRightDragBeyondThreshold(5, 0, 5)).toBe(true);
        expect(isRightDragBeyondThreshold(3, 4, 5)).toBe(true);
    });

    it("低于阈值的位移不算拖拽（否则会误吞正常右键菜单）", () => {
        expect(isRightDragBeyondThreshold(0, 0, 5)).toBe(false);
        expect(isRightDragBeyondThreshold(4, 0, 5)).toBe(false);
        expect(isRightDragBeyondThreshold(3, 3, 5)).toBe(false);
    });

    it("非有限值保守判为「不是拖拽」（不吞菜单）", () => {
        expect(isRightDragBeyondThreshold(Number.NaN, 0, 5)).toBe(false);
        expect(isRightDragBeyondThreshold(0, Number.POSITIVE_INFINITY, 5)).toBe(false);
    });
});

describe("shouldSwallowContextMenu（纯判定）", () => {
    it("武装时吞，未武装时不吞", () => {
        expect(shouldSwallowContextMenu(true)).toBe(true);
        expect(shouldSwallowContextMenu(false)).toBe(false);
    });
});

/** 记录型 window / document 桩：保留处理器以便手动派发。 */
function installWindowStub() {
    const handlers = new Map<string, Set<(event: unknown) => void>>();
    const target = {
        addEventListener(type: string, listener: (event: unknown) => void) {
            const set = handlers.get(type) ?? new Set();
            set.add(listener);
            handlers.set(type, set);
        },
        removeEventListener(type: string, listener: (event: unknown) => void) {
            handlers.get(type)?.delete(listener);
        },
    };
    (globalThis as { window?: unknown }).window = target;
    (globalThis as { document?: unknown }).document = target;
    return {
        /** 派发一个事件；返回被调用的处理器数量与是否被 preventDefault。 */
        dispatch(type: string, event: Record<string, unknown>) {
            const set = handlers.get(type);
            const listeners = set ? [...set] : [];
            let stopped = false;
            const full: Record<string, unknown> & {
                preventDefault(): void;
                stopImmediatePropagation(): void;
                stopPropagation(): void;
                defaultPrevented: boolean;
            } = {
                ...event,
                preventDefault() {
                    full.defaultPrevented = true;
                },
                stopImmediatePropagation() {
                    stopped = true;
                },
                stopPropagation() {
                    stopped = true;
                },
                defaultPrevented: false,
            };
            let called = 0;
            for (const listener of listeners) {
                listener(full);
                called += 1;
                if (stopped) break;
            }
            return { called, defaultPrevented: full.defaultPrevented };
        },
        teardown() {
            delete (globalThis as { window?: unknown }).window;
            delete (globalThis as { document?: unknown }).document;
        },
    };
}

describe("右键拖拽守卫（事件流程）", () => {
    let stub: ReturnType<typeof installWindowStub>;

    beforeEach(() => {
        stub = installWindowStub();
        installRightDragContextMenuGuard();
    });

    afterEach(() => {
        disposeRightDragContextMenuGuard();
        stub.teardown();
    });

    it("【回归】右键拖拽后松手 → 随后的 contextmenu 被吞掉（无论 target 是哪个表面）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 100, clientY: 100 });
        stub.dispatch("pointerup", { button: 2, clientX: 160, clientY: 140 });
        // contextmenu 落在"另一个表面"（轨道头 / 标尺）——守卫按事件本身判定，
        // 与 target 无关，因此照样吞掉。
        const result = stub.dispatch("contextmenu", { button: 2 });
        expect(result.defaultPrevented).toBe(true);
    });

    it("拖拽越阈值即武装（不依赖 pointerup 先到，兼容按下即派发 contextmenu 的平台）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 100, clientY: 100 });
        stub.dispatch("pointermove", { button: 2, clientX: 120, clientY: 100 });
        // 尚未 pointerup，Linux/WebKitGTK 的重派发就到达了。
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(true);
    });

    it("阈值内的 pointermove 不武装", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 100, clientY: 100 });
        stub.dispatch("pointermove", { button: 2, clientX: 101, clientY: 101 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("右键点击（未超过阈值）→ contextmenu 正常放行", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 100, clientY: 100 });
        stub.dispatch("pointerup", { button: 2, clientX: 102, clientY: 101 });
        const result = stub.dispatch("contextmenu", { button: 2 });
        expect(result.defaultPrevented).toBe(false);
    });

    it("武装只生效一次（紧随其后的第二次 contextmenu 放行）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(true);
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("【关键】下一次 pointerdown 复位武装，真实右键点击不被误吞", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        // 用户随后真正想开菜单：先按下（复位），再松开（未拖动）→ 菜单放行。
        stub.dispatch("pointerdown", { button: 2, clientX: 10, clientY: 10 });
        stub.dispatch("pointerup", { button: 2, clientX: 10, clientY: 10 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("左键按下也复位武装（任何新交互都终结上一次拖拽的抑制）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        stub.dispatch("pointerdown", { button: 0, clientX: 5, clientY: 5 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("左键拖拽不武装（只处理右键）", () => {
        stub.dispatch("pointerdown", { button: 0, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 0, clientX: 200, clientY: 200 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("pointercancel 清除按下状态（不会把取消当成拖拽收尾）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointercancel", { button: 2 });
        stub.dispatch("pointerup", { button: 2, clientX: 90, clientY: 90 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("失焦复位：切回窗口后第一次右键点击不被误吞", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        stub.dispatch("blur", {});
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("键盘请求菜单的按键复位武装（该路径不经过 pointerdown）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        stub.dispatch("keydown", { key: "ContextMenu" });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("Shift+F10 同样复位；普通按键不复位（不削弱守卫）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        stub.dispatch("keydown", { key: "a" });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(true);

        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        stub.dispatch("keydown", { key: "F10", shiftKey: true });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("dispose 后守卫不再拦截", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointerup", { button: 2, clientX: 50, clientY: 50 });
        disposeRightDragContextMenuGuard();
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });
});

describe("显式武装（供阈值各异的手势精确调用）", () => {
    let stub: ReturnType<typeof installWindowStub>;

    beforeEach(() => {
        stub = installWindowStub();
        installRightDragContextMenuGuard();
    });

    afterEach(() => {
        disposeRightDragContextMenuGuard();
        stub.teardown();
    });

    it("【关键】低于全局阈值的手势也能精确武装（参数编辑器右键拖拽仅 2px）", () => {
        // 全局启发式阈值为 5px，此处的位移不构成"拖拽"——但手势自己知道
        // （它在自己的 2px 阈值处调用本 API），因此仍然吞掉松手后的菜单。
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointermove", { button: 2, clientX: 0, clientY: -3 });
        armRightDragContextMenuGuard();
        stub.dispatch("pointerup", { button: 2, clientX: 0, clientY: -3 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(true);
    });

    it("未显式武装且低于阈值时仍放行（不误吞正常右键菜单）", () => {
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        stub.dispatch("pointermove", { button: 2, clientX: 0, clientY: -3 });
        stub.dispatch("pointerup", { button: 2, clientX: 0, clientY: -3 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });

    it("显式武装同样只生效一次，且被下一次 pointerdown 复位", () => {
        armRightDragContextMenuGuard();
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(true);
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);

        armRightDragContextMenuGuard();
        stub.dispatch("pointerdown", { button: 2, clientX: 0, clientY: 0 });
        expect(stub.dispatch("contextmenu", {}).defaultPrevented).toBe(false);
    });
});
