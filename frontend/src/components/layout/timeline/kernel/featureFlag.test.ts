/**
 * 参数编辑器内核开关的单测。
 *
 * 【为什么要自带 storage 打桩】本工程 Vitest 跑在 **node** 环境（无 jsdom）：
 * `localStorage` / `window` / `document` 全为 undefined。开关实现因此必须经
 * `globalThis.localStorage` + typeof 守卫读取；测试也必须自己装桩，
 * 不能直接引用裸 `localStorage`（会 ReferenceError）。
 */
import { afterEach, describe, expect, it } from "vitest";

import FLAG_SOURCE from "./featureFlag.ts?raw";

import {
    isPianoRollCurveGlEnabled,
    isPianoRollGlSceneEnabled,
    isPianoRollKernelEnabled,
    isTimelineKernelEnabled,
    PIANO_ROLL_CURVE_GL_FLAG_KEY,
    PIANO_ROLL_KERNEL_FLAG_KEY,
    PIANO_ROLL_KERNEL_GL_FLAG_KEY,
    TIMELINE_KERNEL_FLAG_KEY,
} from "./featureFlag";

/**
 * 未显式设置时四个开关都应**默认开启**（dev 与生产构建一致）。
 *
 * 【为什么写死 true 而不是读 import.meta.env.DEV】内核已从 opt-in 路径转为默认
 * 路径：生产构建也必须走新实现，否则 `TAURI_UI_MODE=build` 跑的是旧渲染器，
 * 而真机（尤其 Windows/WebView2）的卡顿报告恰恰来自那个模式。
 *
 * 【为什么还需要"源码级守护"】Vitest 只在 DEV 下运行，`import.meta.env.DEV` 恒为
 * true——若实现写成 `return import.meta.env.DEV`，上面的断言**照样通过**，而生产
 * 构建会默认关闭。测试环境本身掩盖了要守护的行为，所以另加一条不依赖运行时的检查
 * （见下方「生产构建默认值（源码级守护）」）。
 */
const EXPECT_UNSET = true;

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

/**
 * 生产构建模拟：断言四个开关的默认值**不读取 `import.meta.env.DEV`**。
 *
 * 【为什么需要这一层】Vitest 只在 DEV 下运行，`import.meta.env.DEV` 恒为 true，
 * 于是"默认开启"的断言在实现写成 `return import.meta.env.DEV` 时也会通过——
 * 而那正好会让生产构建默认关闭。测试环境掩盖了要守护的行为。
 *
 * 做法：直接检查模块源码里这四个函数的函数体不含 `import.meta.env.DEV`。
 * 这比运行时断言弱（不是行为测试），但它是**唯一**能在 DEV-only 测试环境里
 * 发现"生产会回落"的手段；配套的构建产物检查（见计划文档）覆盖运行时那一半。
 */
describe("生产构建默认值（源码级守护）", () => {
    it("四个开关的实现不依赖 import.meta.env.DEV", () => {
        // Vite 的 `?raw` 在构建期把源码作为字符串内联——不需要 node:fs，
        // 因此浏览器 tsconfig 下也能通过类型检查。
        const code = FLAG_SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
        expect(code).not.toContain("import.meta.env.DEV");
        expect(code).not.toContain("import.meta.env.PROD");
    });
});

describe("isPianoRollKernelEnabled", () => {
    let restore: (() => void) | null = null;
    afterEach(() => {
        restore?.();
        restore = null;
    });

    it("未显式设置 → 默认开启（生产构建同样开启）", () => {
        restore = installStorage(() => null);
        expect(isPianoRollKernelEnabled()).toBe(true);
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

    it("未显式设置 → 默认开启（生产构建同样开启）", () => {
        restore = installStorage(() => null);
        expect(isPianoRollGlSceneEnabled()).toBe(true);
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

    it("只认 '0' / '1'；其它字符串视为未设置 → 回落默认（开启）", () => {
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

    it("未显式设置 → 默认开启（生产构建同样开启）", () => {
        restore = installStorage(() => null);
        expect(isPianoRollCurveGlEnabled()).toBe(true);
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

/**
 * 时间轴内核开关。
 *
 * 【为什么也要守护默认值】它与参数编辑器内核是**同一策略**：未显式设置时默认开启。
 * 生产构建下若回落为关闭，`TAURI_UI_MODE=build` 会静默退回旧渲染器——这正是
 * 内核落地过程中反复出现的坑（开发时看到新实现，打包后看到旧实现）。
 */
describe("isTimelineKernelEnabled", () => {
    let restore: (() => void) | null = null;
    afterEach(() => {
        restore?.();
        restore = null;
    });

    it("未显式设置 → 默认开启（生产构建同样开启）", () => {
        restore = installStorage(() => null);
        expect(isTimelineKernelEnabled()).toBe(true);
        expect(isTimelineKernelEnabled()).toBe(EXPECT_UNSET);
    });

    it("显式 '0' → 关闭（逃生门）", () => {
        restore = installStorage((key) => (key === TIMELINE_KERNEL_FLAG_KEY ? "0" : null));
        expect(isTimelineKernelEnabled()).toBe(false);
    });

    it("显式 '1' → 开启", () => {
        restore = installStorage((key) => (key === TIMELINE_KERNEL_FLAG_KEY ? "1" : null));
        expect(isTimelineKernelEnabled()).toBe(true);
    });

    it("存储不可用（读抛错）→ 关闭，不抛异常", () => {
        restore = installStorage(() => {
            throw new Error("denied");
        });
        expect(isTimelineKernelEnabled()).toBe(false);
    });
});
