/**
 * 参数编辑器内核开关的单测。
 *
 * 【为什么要自带 storage 打桩】本工程 Vitest 跑在 **node** 环境（无 jsdom）：
 * `localStorage` / `window` / `document` 全为 undefined。开关实现因此必须经
 * `globalThis.localStorage` + typeof 守卫读取；测试也必须自己装桩，
 * 不能直接引用裸 `localStorage`（会 ReferenceError）。
 */
import { afterEach, describe, expect, it } from "vitest";

import {
    isPianoRollGlSceneEnabled,
    isPianoRollKernelEnabled,
    PIANO_ROLL_KERNEL_FLAG_KEY,
    PIANO_ROLL_KERNEL_GL_FLAG_KEY,
} from "./featureFlag";

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

/**
 * GL 场景层子开关（阶段 2）。
 *
 * 【为什么单独守护】它是"GL 某层迁移出问题"时的**唯一回退手段**：只关 GL 层、
 * 保留阶段 1 已验证的滚动内核。因此必须确认它与内核开关**互不干扰**——把依赖
 * 写进任一侧，都会让"内核开 + GL 关"这一合法组合无法表达。
 */
describe("isPianoRollGlSceneEnabled", () => {
    let restore: (() => void) | null = null;
    afterEach(() => {
        restore?.();
        restore = null;
    });

    it("未显式设置 → 关闭", () => {
        restore = installStorage(() => null);
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });

    it("显式 '1' → 开启", () => {
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? "1" : null));
        expect(isPianoRollGlSceneEnabled()).toBe(true);
    });

    it("显式 '0' → 关闭", () => {
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? "0" : null));
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });

    it("只认 '1'，其它真值字符串不算开启", () => {
        for (const value of ["true", "yes", "on", "2", ""]) {
            restore = installStorage((key) =>
                key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? value : null,
            );
            expect(isPianoRollGlSceneEnabled()).toBe(false);
        }
    });

    it("与内核开关互不影响（内核开 + GL 关 必须可表达）", () => {
        // 只设内核开关：GL 应为关。
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_FLAG_KEY ? "1" : null));
        expect(isPianoRollKernelEnabled()).toBe(true);
        expect(isPianoRollGlSceneEnabled()).toBe(false);

        // 只设 GL 开关：内核应为关（GL 不得隐式拉启内核）。
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? "1" : null));
        expect(isPianoRollGlSceneEnabled()).toBe(true);
        expect(isPianoRollKernelEnabled()).toBe(false);
    });

    it("存储不可用（读抛错）→ 关闭，不抛异常", () => {
        restore = installStorage(() => {
            throw new Error("denied");
        });
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });
});
