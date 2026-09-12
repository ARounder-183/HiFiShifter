/**
 * 参数编辑器内核开关的单测。
 *
 * 【为什么要自带 storage 打桩】本工程 Vitest 跑在 **node** 环境（无 jsdom）：
 * `localStorage` / `window` / `document` 全为 undefined。开关实现因此必须经
 * `globalThis.localStorage` + typeof 守卫读取；测试也必须自己装桩，
 * 不能直接引用裸 `localStorage`（会 ReferenceError）。
 */
import { afterEach, describe, expect, it } from "vitest";

import { isPianoRollKernelEnabled, PIANO_ROLL_KERNEL_FLAG_KEY } from "./featureFlag";

/** 装一个最小可用的 localStorage 桩（只实现开关用到的 getItem）。 */
function installStorage(impl: (key: string) => string | null): () => void {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
    Object.defineProperty(globalThis, "localStorage", {
        value: { getItem: impl },
        configurable: true,
        writable: true,
    });
    return () => {
        if (descriptor) Object.defineProperty(globalThis, "localStorage", descriptor);
        else Reflect.deleteProperty(globalThis, "localStorage");
    };
}

describe("isPianoRollKernelEnabled", () => {
    let restore: (() => void) | null = null;
    afterEach(() => {
        restore?.();
        restore = null;
    });

    it("未显式设置 → 关闭（与时间轴内核不同：连 dev 也默认关）", () => {
        restore = installStorage(() => null);
        expect(isPianoRollKernelEnabled()).toBe(false);
    });

    it("显式 '1' → 开启", () => {
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_FLAG_KEY ? "1" : null));
        expect(isPianoRollKernelEnabled()).toBe(true);
    });

    it("显式 '0' → 关闭", () => {
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_FLAG_KEY ? "0" : null));
        expect(isPianoRollKernelEnabled()).toBe(false);
    });

    it("存储不可用（隐私模式 / 读抛错）→ 关闭，不抛异常", () => {
        restore = installStorage(() => {
            throw new Error("denied");
        });
        expect(isPianoRollKernelEnabled()).toBe(false);
    });
});
