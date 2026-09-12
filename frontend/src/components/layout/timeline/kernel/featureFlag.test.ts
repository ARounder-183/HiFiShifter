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
    isPianoRollCurveGlEnabled,
    isPianoRollGlSceneEnabled,
    isPianoRollKernelEnabled,
    PIANO_ROLL_CURVE_GL_FLAG_KEY,
    PIANO_ROLL_KERNEL_FLAG_KEY,
    PIANO_ROLL_KERNEL_GL_FLAG_KEY,
} from "./featureFlag";

/**
 * 未显式设置时四个开关都应回落到 `import.meta.env.DEV`。
 *
 * 【为什么要读这个值而不是写死 true/false】同一个测试文件在 dev 与生产构建下
 * 语义不同：`import.meta.env.DEV` 由构建工具在编译期替换。写死断言会在另一种
 * 构建下变成假失败——测试要守护的是"跟随 DEV"这条规则本身，不是某个具体取值。
 */
const EXPECT_UNSET = import.meta.env.DEV;

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

    it("未显式设置 → 跟随 import.meta.env.DEV（dev 开 / 生产关）", () => {
        restore = installStorage(() => null);
        expect(isPianoRollKernelEnabled()).toBe(EXPECT_UNSET);
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

    it("未显式设置 → 跟随 import.meta.env.DEV", () => {
        restore = installStorage(() => null);
        expect(isPianoRollGlSceneEnabled()).toBe(EXPECT_UNSET);
    });

    it("显式 '1' → 开启", () => {
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? "1" : null));
        expect(isPianoRollGlSceneEnabled()).toBe(true);
    });

    it("显式 '0' → 关闭", () => {
        restore = installStorage((key) => (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? "0" : null));
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });

    it("只认 '0' / '1'；其它字符串视为未设置 → 回落 DEV", () => {
        // 关键区分：这些值既不算"显式开启"也不算"显式关闭"，走默认策略。
        for (const value of ["true", "yes", "on", "2", ""]) {
            restore = installStorage((key) =>
                key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? value : null,
            );
            expect(isPianoRollGlSceneEnabled()).toBe(EXPECT_UNSET);
        }
    });

    it("与内核开关互不影响（内核开 + GL 关 必须可表达）", () => {
        // 两个开关都**显式**赋值，才能断言"互不影响"——靠未设置回落 DEV 的话，
        // 两侧同时为真，反而测不出依赖关系。
        //
        // 组合 1：内核开 + GL 显式关（"某一层迁移出问题"的回退姿势）。
        restore = installStorage((key) => {
            if (key === PIANO_ROLL_KERNEL_FLAG_KEY) return "1";
            if (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY) return "0";
            return null;
        });
        expect(isPianoRollKernelEnabled()).toBe(true);
        expect(isPianoRollGlSceneEnabled()).toBe(false);

        // 组合 2：内核显式关 + GL 开 —— GL 不得隐式拉启内核。
        restore = installStorage((key) => {
            if (key === PIANO_ROLL_KERNEL_FLAG_KEY) return "0";
            if (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY) return "1";
            return null;
        });
        expect(isPianoRollKernelEnabled()).toBe(false);
        expect(isPianoRollGlSceneEnabled()).toBe(true);
    });

    it("存储不可用（读抛错）→ 关闭，不抛异常", () => {
        restore = installStorage(() => {
            throw new Error("denied");
        });
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });
});

/**
 * 曲线 GL 子开关（阶段 3）。
 *
 * 【为什么单独守护】曲线层是阶段 3 新增的**最高风险层**（miter 连接、非整数线宽、
 * 虚线相位、裁剪），它必须能独立于阶段 2 的静态层回退——否则曲线出问题只能整体
 * 退回旧实现，连带丢掉已充分验证的滚动内核与静态 GL 层。
 */
describe("isPianoRollCurveGlEnabled", () => {
    let restore: (() => void) | null = null;
    afterEach(() => {
        restore?.();
        restore = null;
    });

    it("未显式设置 → 跟随 import.meta.env.DEV", () => {
        restore = installStorage(() => null);
        expect(isPianoRollCurveGlEnabled()).toBe(EXPECT_UNSET);
    });

    it("显式 '1' → 开启", () => {
        restore = installStorage((key) => (key === PIANO_ROLL_CURVE_GL_FLAG_KEY ? "1" : null));
        expect(isPianoRollCurveGlEnabled()).toBe(true);
    });

    it("显式 '0' → 关闭（曲线单独回退，不影响其余层）", () => {
        restore = installStorage((key) => {
            if (key === PIANO_ROLL_CURVE_GL_FLAG_KEY) return "0";
            if (key === PIANO_ROLL_KERNEL_GL_FLAG_KEY) return "1";
            return null;
        });
        expect(isPianoRollCurveGlEnabled()).toBe(false);
        // 关键：关曲线不得连带关掉静态 GL 层。
        expect(isPianoRollGlSceneEnabled()).toBe(true);
    });

    it("存储不可用（读抛错）→ 关闭，不抛异常", () => {
        restore = installStorage(() => {
            throw new Error("denied");
        });
        expect(isPianoRollCurveGlEnabled()).toBe(false);
    });
});
