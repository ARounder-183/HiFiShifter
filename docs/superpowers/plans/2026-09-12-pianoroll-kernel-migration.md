# PianoRoll Kernel Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the parameter editor (PianoRoll) panel onto the render-kernel architecture — self-drawn scrolling and GL rendering — with zero functional or visual loss, in three independently shippable phases.

**Architecture:** Phase 1 gives the panel a kernel-owned viewport (`ScrollKernel` + unified `TimelineAxis` projection + self-drawn scrollbars) while painting stays Canvas 2D; the vertical **value-domain** model (`center`/`span` ↔ 1600px) is preserved untouched. Phase 2 moves grid / keyboard axis / labels / selection / playhead to GL instanced geometry and activates the existing tested glyph pipeline for text. Phase 3 converts parameter curves to GL polyline triangle strips and migrates hit testing plus the gesture state machine.

**Tech Stack:** React 19, TypeScript 5.9, Redux Toolkit, Redux, WebGL2, Canvas 2D, Vitest, Playwright-core (`frontend/scripts/dev-shot.mjs`)

**Spec:** `docs/superpowers/specs/2026-09-12-pianoroll-kernel-migration-design.md`

**Verification prerequisites (used by every task):**
- dev server: `cd frontend && npx vite --port 5173 --strictPort`
- browser harness: `VW=1920 VH=1200 KERNEL=1 node scripts/dev-shot.mjs "<url>" <out.png> <waitMs> '<actionsJson>'`
- mock backend: URL carries `?mock=1`
- pre-existing failures: exactly 2 tests in `src/features/keybindings/keybindingMatch.test.ts` fail on a clean tree. Any other failure is caused by this work.

**Test environment (verified on this branch, applies to every task):** Vitest runs in the
**node** environment — there is no jsdom. A probe confirmed `typeof localStorage`,
`typeof window` and `typeof document` are all `undefined` inside tests. Therefore:
- production modules must not touch browser globals at module-evaluation time (guard with
  `typeof` / null checks, read via `globalThis`);
- tests install their own minimal stubs instead of assuming a DOM;
- anything DOM-shaped (layout, canvas pixels, gestures) is verified in the browser harness,
  not in Vitest.

---

## Phase 1 — Scroll/Viewport Kernel

**Exit criteria:** with `hifishifter.pianoRollKernel=1`, the parameter editor scrolls and zooms
by kernel-owned state with identical feel; the timeline sync feature still works; painting is
still Canvas 2D so visuals are unchanged by construction; `0` restores today's behaviour exactly.

**What changes mechanically:** the native scroller stops being the source of truth. Today the
panel treats `scroller.scrollLeft` / `scroller.scrollTop` as authoritative and pushes them into
every layer through `syncScrollLeft` → `applyScrollLayers` → bus → per-frame `reconcile`.
Phase 1 inverts that: `ScrollKernel` owns the values, and the DOM scroller becomes a passive
mirror used only as a legacy compatibility surface for code not yet migrated.

### Task 1: Add the Phase 1 feature flag

**Files:**
- Modify: `frontend/src/components/layout/timeline/kernel/featureFlag.ts`
- Test: `frontend/src/components/layout/timeline/kernel/featureFlag.test.ts` (create)

**Environment note (verified, not assumed):** this project runs Vitest in the **node**
environment — `localStorage`, `window` and `document` are all `undefined` inside tests
(verified by probe on this branch). The flag therefore must read storage through
`globalThis.localStorage` guarded by a `typeof` check, and the test must install its own
stub rather than assume a DOM. Do **not** write a test that touches bare `localStorage`.

- [x] **Step 1: Write the failing test**

```ts
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
        restore = installStorage((key) =>
            key === PIANO_ROLL_KERNEL_FLAG_KEY ? "1" : null,
        );
        expect(isPianoRollKernelEnabled()).toBe(true);
    });

    it("显式 '0' → 关闭", () => {
        restore = installStorage((key) =>
            key === PIANO_ROLL_KERNEL_FLAG_KEY ? "0" : null,
        );
        expect(isPianoRollKernelEnabled()).toBe(false);
    });

    it("存储不可用（隐私模式 / 读抛错）→ 关闭，不抛异常", () => {
        restore = installStorage(() => {
            throw new Error("denied");
        });
        expect(isPianoRollKernelEnabled()).toBe(false);
    });
});
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run featureFlag`
Expected: FAIL — `PIANO_ROLL_KERNEL_FLAG_KEY` / `isPianoRollKernelEnabled` are not exported.

- [x] **Step 3: Implement the flag**

Append to `frontend/src/components/layout/timeline/kernel/featureFlag.ts`:

```ts
/** 参数编辑器（PianoRoll）内核开关 key。 */
export const PIANO_ROLL_KERNEL_FLAG_KEY = "hifishifter.pianoRollKernel";

/**
 * 是否启用参数编辑器渲染内核。
 *
 * 规则与时间轴内核**刻意不同**：未显式设置时**默认关闭**（连 dev 也关）。
 *
 * 特殊说明 1：时间轴内核在 dev 默认开启，是为了让真机验证不必每次改 localStorage；
 * 参数编辑器分三个阶段落地，阶段 1 期间新路径尚不完整（绘制仍在 Canvas2D），
 * 默认开启会让日常开发一直跑在半迁移状态。需要验证时显式写 `"1"`。
 *
 * 特殊说明 2：必须经 `globalThis.localStorage` + `typeof` 守卫读取，不能直接引用
 * 裸 `localStorage`——本工程 Vitest 跑在 node 环境（无 jsdom），直接引用会让
 * **导入该模块的任何测试**在模块求值期就抛 ReferenceError。
 *
 * @returns 当前是否启用参数编辑器内核。
 */
export function isPianoRollKernelEnabled(): boolean {
    try {
        const storage = globalThis.localStorage;
        if (storage == null) return false;
        return storage.getItem(PIANO_ROLL_KERNEL_FLAG_KEY) === "1";
    } catch {
        return false;
    }
}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run featureFlag`
Expected: PASS (4 tests).

- [x] **Step 5: Verify the pre-existing timeline flag still behaves**

The existing `isTimelineKernelEnabled` reads bare `localStorage` inside a `try`, which is safe
in node only because it is never imported by a test today. Confirm this task did not change its
behaviour:

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit`
Expected: only the 2 known `keybindingMatch` failures; typecheck clean.

- [x] **Step 6: Commit**

```bash
git add frontend/src/components/layout/timeline/kernel/featureFlag.ts frontend/src/components/layout/timeline/kernel/featureFlag.test.ts
git commit -m "feat(pianoroll-kernel): add phase-1 feature flag (default off)"
```

### Task 2: Value-domain vertical scroll adapter (pure)

The kernel's `ScrollKernel` owns a pixel `scrollTop`. The parameter editor's vertical axis is
a **value domain**. This task adds the adapter that converts between them, reusing the existing
mapping functions verbatim so feel cannot drift.

**Files:**
- Create: `frontend/src/components/layout/pianoRoll/kernel/scroll/verticalValueScroll.ts`
- Test: `frontend/src/components/layout/pianoRoll/kernel/scroll/verticalValueScroll.test.ts`

- [x] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";

import {
    PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
    centerFromKernelScrollTop,
    kernelScrollTopFromCenter,
} from "./verticalValueScroll";

describe("verticalValueScroll（值域 ↔ 内核像素滚动）", () => {
    const bounds = { min: 0, max: 100 };

    it("滚动范围常量与旧实现一致（1600px）", () => {
        expect(PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX).toBe(1600);
    });

    it("中心值居中时滚动位置位于范围中点", () => {
        const top = kernelScrollTopFromCenter({ ...bounds, span: 50, center: 50 });
        expect(top).toBeCloseTo(800, 6);
    });

    it("中心越靠上 → 滚动位置越小（与旧 verticalScrollTopFromCenter 同向）", () => {
        const high = kernelScrollTopFromCenter({ ...bounds, span: 20, center: 80 });
        const low = kernelScrollTopFromCenter({ ...bounds, span: 20, center: 20 });
        expect(high).toBeLessThan(low);
    });

    it("往返转换是无损的（可逆）", () => {
        for (const center of [15, 37.5, 50, 88]) {
            const top = kernelScrollTopFromCenter({ ...bounds, span: 30, center });
            const back = centerFromKernelScrollTop({ ...bounds, span: 30, scrollTop: top });
            expect(back).toBeCloseTo(center, 6);
        }
    });

    it("span 覆盖整个范围时不可动（映射退化为中点）", () => {
        const top = kernelScrollTopFromCenter({ ...bounds, span: 100, center: 50 });
        expect(top).toBe(0);
        expect(centerFromKernelScrollTop({ ...bounds, span: 100, scrollTop: 999 })).toBeCloseTo(
            50,
            6,
        );
    });

    it("越界输入被钳制，不产生 NaN", () => {
        const top = kernelScrollTopFromCenter({ ...bounds, span: 20, center: 9999 });
        expect(Number.isFinite(top)).toBe(true);
        expect(top).toBeGreaterThanOrEqual(0);
        expect(top).toBeLessThanOrEqual(PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX);
    });
});
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run verticalValueScroll`
Expected: FAIL — module not found.

- [x] **Step 3: Implement the adapter**

Create `frontend/src/components/layout/pianoRoll/kernel/scroll/verticalValueScroll.ts`:

```ts
/**
 * 参数编辑器内核 · 竖向「值域 ↔ 像素滚动」适配。
 *
 * 【主要内容】
 * 把内核 `ScrollKernel` 的像素 `scrollTop` 与参数编辑器的**值域**视口
 * （`center` / `span`）互相转换，并给出两侧共用的滚动范围常量。
 *
 * 【作用】
 * 参数编辑器的竖向滚动语义与时间轴**根本不同**：时间轴滚的是像素行，
 * 而参数编辑器滚的是参数值域——`center` 是视口中心的值、`span` 是可见值跨度。
 * 内核的 `ScrollKernel` 只认像素，因此两者之间必须有一层适配。
 *
 * 【为什么不重写映射】
 * 真实换算仍是 `components/layout/pianoRoll/verticalScrollMapping.ts` 里的
 * `verticalScrollTopFromCenter` / `centerFromVerticalScrollTop`（既有实现、
 * 已被拖拽与滚轮路径调参验证过）。本模块只是把它接到内核像素域上：
 * 另写一套比值公式会让滚轮步进与拖拽比例的手感在两个渲染模式下分叉。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 在滚动条拖拽 / 滚轮 / 视口命令时调用。
 * - 下游：`verticalScrollMapping` 提供真正的映射算术。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 */

import {
    centerFromVerticalScrollTop,
    verticalScrollTopFromCenter,
} from "../../verticalScrollMapping";

/**
 * 竖向滚动条的可滚动像素范围。
 *
 * 必须与旧实现一致：面板过去用一个 1600px 的 spacer div 撑出滚动范围
 * （见 `PianoRollPanel` 的 `PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX`）。
 * 改这个值会同时改变滚轮步进与拖拽比例，属手感变更，不在本次迁移范围。
 */
export const PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX = 1600;

/** 值域边界与视口跨度的公共入参。 */
export interface ValueScrollMappingArgs {
    /** 该参数值域下界。 */
    readonly min: number;
    /** 该参数值域上界。 */
    readonly max: number;
    /** 视口可见的值跨度。 */
    readonly span: number;
}

/**
 * 值域中心 → 内核像素滚动位置。
 *
 * @param args 值域边界、跨度，以及视口中心值。
 * @returns 内核 `scrollTop`（CSS px，已钳制到 [0, 1600]）。
 */
export function kernelScrollTopFromCenter(
    args: ValueScrollMappingArgs & { readonly center: number },
): number {
    return verticalScrollTopFromCenter({
        min: args.min,
        max: args.max,
        span: args.span,
        center: args.center,
        scrollRangePx: PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
    });
}

/**
 * 内核像素滚动位置 → 值域中心。
 *
 * @param args 值域边界、跨度，以及内核 `scrollTop`。
 * @returns 视口中心值（已钳制到值域内合法区间）。
 */
export function centerFromKernelScrollTop(
    args: ValueScrollMappingArgs & { readonly scrollTop: number },
): number {
    return centerFromVerticalScrollTop({
        min: args.min,
        max: args.max,
        span: args.span,
        scrollTop: args.scrollTop,
        scrollRangePx: PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
    });
}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run verticalValueScroll`
Expected: PASS (6 tests).

- [x] **Step 5: Commit**

```bash
git add frontend/src/components/layout/pianoRoll/kernel/scroll/
git commit -m "feat(pianoroll-kernel): value-domain vertical scroll adapter"
```

### Task 3: Self-drawn scrollbar geometry for the parameter editor

Reuses the timeline kernel's `computeScrollbar` / `hitTestScrollbarThumb` /
`scrollDeltaFromThumbDrag` / `scrollTargetFromTrackClick` verbatim; this task only adds the
axis descriptors the panel needs (two axes, one of them value-domain).

**Files:**
- Create: `frontend/src/components/layout/pianoRoll/kernel/scroll/scrollbarSpec.ts`
- Test: `frontend/src/components/layout/pianoRoll/kernel/scroll/scrollbarSpec.test.ts`

- [x] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";

import { resolvePianoRollScrollbarGeometries } from "./scrollbarSpec";

describe("resolvePianoRollScrollbarGeometries", () => {
    it("两条轴都按视口尺寸与内容尺寸算出几何", () => {
        const { horizontal, vertical } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            contentWidthPx: 5000,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
        });
        expect(horizontal.scrollable).toBe(true);
        expect(vertical.scrollable).toBe(true);
        // 水平：视口 1000 / 内容 5000 → thumb = 1000 * 200 = 200
        expect(horizontal.thumbLengthPx).toBeCloseTo(200, 6);
    });

    it("内容不足一屏时该轴不可滚动", () => {
        const { horizontal } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            contentWidthPx: 400,
            scrollLeftPx: 0,
            scrollTopPx: 0,
            maxScrollLeftPx: 0,
            maxScrollTopPx: 1600,
        });
        expect(horizontal.scrollable).toBe(false);
    });

    it("竖向用 1600px 值域范围而不是内容像素", () => {
        const { vertical } = resolvePianoRollScrollbarGeometries({
            viewportWidthPx: 1000,
            viewportHeightPx: 600,
            contentWidthPx: 5000,
            scrollLeftPx: 0,
            scrollTopPx: 800,
            maxScrollLeftPx: 5000,
            maxScrollTopPx: 1600,
            verticalContentSizePx: 1600 + 600,
        });
        // 内容 2200 / 视口 600 → thumb = 600 * (600/2200)
        expect(vertical.thumbLengthPx).toBeCloseTo(600 * (600 / 2200), 6);
    });
});
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run scrollbarSpec`
Expected: FAIL — module not found.

- [x] **Step 3: Implement**

Create `frontend/src/components/layout/pianoRoll/kernel/scroll/scrollbarSpec.ts`:

```ts
/**
 * 参数编辑器内核 · 自绘滚动条几何装配。
 *
 * 【主要内容】
 * 把「视口尺寸 / 内容尺寸 / 当前滚动 / 滚动上限」装配成水平与竖直两条滚动条的几何，
 * 供宿主每帧写 DOM。
 *
 * 【作用】
 * 参数编辑器过去依赖原生滚动条（`overflow-x-scroll overflow-y-scroll`）。内核自绘
 * 滚动后浏览器不再提供滚动条，必须自己画。几何计算本身**复用时间轴内核**的
 * `computeScrollbar`——两条轴的数学完全相同，复制一份会在边界处理（内容不足一屏、
 * 除零）上分叉。
 *
 * 【竖向的特殊性】
 * 竖向滚动的「内容」不是像素内容而是**值域范围**（见 `verticalValueScroll`）：
 * 调用方传入的 `maxScrollTopPx` 即 1600px 值域范围，`verticalContentSizePx`
 * 应传「值域范围 + 视口高度」，这样 thumb 比例与拖拽行程与旧实现一致。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 每帧调用。
 * - 复用：`timeline/kernel/input/scrollbars`（几何与命中的单一来源）。
 * - 独立性：纯函数，不依赖 DOM / React。
 */

import {
    computeScrollbar,
    type ScrollbarGeometry,
} from "../../../timeline/kernel/input/scrollbars";

/** 装配入参。 */
export interface PianoRollScrollbarArgs {
    readonly viewportWidthPx: number;
    readonly viewportHeightPx: number;
    /** 时间轴内容宽度（工程秒 × pxPerSec）。 */
    readonly contentWidthPx: number;
    readonly scrollLeftPx: number;
    readonly scrollTopPx: number;
    /** 水平滚动上限（= 内容宽度，与时间轴语义一致）。 */
    readonly maxScrollLeftPx: number;
    /** 竖向滚动上限（= 值域滚动范围，通常 1600）。 */
    readonly maxScrollTopPx: number;
    /** 竖向「内容尺寸」= 值域范围 + 视口高度；缺省由 max + 视口高度推出。 */
    readonly verticalContentSizePx?: number;
}

/** 两条轴的几何。 */
export interface PianoRollScrollbarGeometries {
    readonly horizontal: ScrollbarGeometry;
    readonly vertical: ScrollbarGeometry;
}

/**
 * 解析两条滚动条的几何。
 *
 * @param args 见 `PianoRollScrollbarArgs`。
 * @returns 水平与竖直滚动条几何。
 */
export function resolvePianoRollScrollbarGeometries(
    args: PianoRollScrollbarArgs,
): PianoRollScrollbarGeometries {
    return {
        horizontal: computeScrollbar({
            contentSizePx: args.contentWidthPx,
            viewportSizePx: args.viewportWidthPx,
            scrollPx: args.scrollLeftPx,
            maxScrollPx: args.maxScrollLeftPx,
        }),
        vertical: computeScrollbar({
            contentSizePx:
                args.verticalContentSizePx ?? args.maxScrollTopPx + args.viewportHeightPx,
            viewportSizePx: args.viewportHeightPx,
            scrollPx: args.scrollTopPx,
            maxScrollPx: args.maxScrollTopPx,
        }),
    };
}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run scrollbarSpec`
Expected: PASS (3 tests).

- [x] **Step 5: Commit**

```bash
git add frontend/src/components/layout/pianoRoll/kernel/scroll/
git commit -m "feat(pianoroll-kernel): two-axis self-drawn scrollbar geometry"
```

---

## Phase 1 remaining tasks (expand in place when Phase 1 starts)

The tasks below are ordered and scoped, with their files and exit checks. Their step-level
detail (failing test first, exact code, exact commands) is written when the task starts, because
each depends on the previous task's concrete shape. This is the same granularity the timeline
kernel used: an adapter task, a host task, a wiring task, then a live-look verification task.

### Task 4: Kernel host skeleton

**Browser-measured baseline (taken on this branch before writing the task, `?mock=1`, 1920×1200):**

| 量 | 旧实现实测值 | 结论 |
|---|---|---|
| scroller `clientHeight` / `scrollHeight` | 823 / 2423 | 内容高 = 1600 spacer + 一个视口高 |
| **原生最大 `scrollTop`** | **1600** | 与 `PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX` 完全相等 |
| spacer 行内高 | `1600px` | 范围由 spacer 撑出，非布局偶然 |
| **原生最大 `scrollLeft`** | **9125** | = 绘制内容宽（spacer = 内容宽 + 视口宽） |
| `scrollWidth` / `clientWidth` | 10989 / 1864 | 9125 + 1864 ⇒ 上式的验证 |

**这张表是 Task 4 的验收基线**：内核必须复现「竖向上限 1600、横向上限 = 内容宽」。
两者恰好都是 `ScrollKernel` 的既有语义，因此**不需要改内核**：

- 竖向：`rowHeight = 0`、`trackCount = () => 0`、
  `extraContentHeightPx = () => RANGE + viewportHeightPx()` ⇒
  `maxScrollTop = (RANGE + vh) − vh = RANGE`，与旧实现逐值相等；
- 横向：`maxScrollLeft = projectSec × pxPerSec` = 内容宽，与旧实现逐值相等；
  且「同步时间轴」开启时旧的 `paddedContentWidth` 多出的 `offset` 只影响**原生**
  坐标，绘制坐标上限仍是内容宽 —— 内核持有的正是绘制坐标。

**Files:**
- Create: `frontend/src/components/layout/pianoRoll/kernel/host/pianoRollKernelData.ts`
- Create: `frontend/src/components/layout/pianoRoll/kernel/host/pianoRollKernelHost.ts`
- Test: `frontend/src/components/layout/pianoRoll/kernel/host/pianoRollKernelHost.test.ts`

**范围（Task 4 只做骨架，不做输入）**：宿主自持容器、两条滚动条的 thumb / 轨道、
量测、`ScrollKernel`、`createRenderLoop`、每帧帧提交与 `dispose`。**不**接管滚轮 /
键盘 / 中键拖拽（那些是 Task 7）与绘制（Phase 1 绘制仍由面板的 Canvas2D 拥有，
经 `onFrame` 回调注入）。

**阶段 1 宿主不引入 GL**：绘制仍在面板的 Canvas2D 上，宿主不需要 WebGL2 上下文，
因此本任务**可在 node 环境单测**（无 DOM 也能验证构造 / 销毁 / 钳制 / 值域往返）。

- [x] **Step 1: Write the failing test**

测试用「记录型桩」替掉 DOM：断言 `dispose()` 把加过的监听**逐条**摘掉（add/remove 配平）、
重复 `dispose()` 安全、横向钳制复现实测上限、值域往返无损、滚动条几何与实测比例一致。
`requestFrame` 注入为「捕获回调」，由测试手动 flush，使帧提交路径可确定性验证。

```ts
/**
 * 参数编辑器内核宿主 · 构造 / 销毁 / 钳制 / 值域往返单测。
 *
 * 【为什么可以在 node 环境测】阶段 1 的宿主**不碰 WebGL**（绘制仍在面板的
 * Canvas2D 上，经回调注入），只用到很少的 DOM API，因此可以用「记录型桩」替掉
 * 容器与滚动条元素，在无 jsdom 的 node 环境下验证生命周期与数值语义。
 *
 * 【本测试要钉住的核心不变量】
 * 1. `dispose()` 摘掉自己加过的**每一条**监听（add/remove 配平）——宿主模式最
 *    常见的缺陷就是漏摘 window 上的监听，卸载后仍持有回调；
 * 2. 横向上限 = 内容宽、竖向上限 = 1600（浏览器实测的旧实现基线，见计划 Task 4）；
 * 3. 值域 ↔ 像素往返无损；
 * 4. 重复 `dispose()` 不抛错。
 */
import { describe, expect, it } from "vitest";

import { createPianoRollKernelHost } from "./pianoRollKernelHost";
import { PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX } from "../scroll/verticalValueScroll";

/** 记录型 DOM 桩：统计每个目标上 add / remove 的次数。 */
function makeTarget() {
    const counts = new Map<string, { added: number; removed: number }>();
    return {
        counts,
        addEventListener(type: string) {
            const entry = counts.get(type) ?? { added: 0, removed: 0 };
            entry.added += 1;
            counts.set(type, entry);
        },
        removeEventListener(type: string) {
            const entry = counts.get(type) ?? { added: 0, removed: 0 };
            entry.removed += 1;
            counts.set(type, entry);
        },
        /** 全部类型都配平（加过几条就摘掉几条）。 */
        balanced(): boolean {
            for (const entry of counts.values()) {
                if (entry.added !== entry.removed) return false;
            }
            return true;
        },
        totalAdded(): number {
            let sum = 0;
            for (const entry of counts.values()) sum += entry.added;
            return sum;
        },
    };
}

/** 造一个测试用宿主（容器 / thumb / track 全是记录型桩）。 */
function makeHost() {
    const container = makeTarget();
    const vThumb = makeTarget();
    const hThumb = makeTarget();
    const rulerContent = { style: {} as CSSStyleDeclaration };
    const gridLayer = {};
    let pending: FrameRequestCallback | null = null;
    let handle = 0;
    let scrollLeftCommits = 0;

    const host = createPianoRollKernelHost({
        // node 无 DOM：桩只需满足宿主实际用到的成员（量测 + 事件 + style）。
        container: Object.assign(container, { clientWidth: 1864, clientHeight: 823 }) as never,
        hScrollbarThumb: hThumb as never,
        vScrollbarThumb: vThumb as never,
        data: () => ({
            projectSec: 100,
            pxPerSec: 91.25,
            valueDomain: { min: 0, max: 100, span: 50 },
        }),
        sync: {
            rulerContent: rulerContent as never,
            gridLayer: gridLayer as never,
        },
        onFrame: () => {},
        onScrollLeftCommit: () => {
            scrollLeftCommits += 1;
        },
        // 注入帧调度：手动 flush，避免依赖 node 里不存在的 rAF。
        requestFrame: (cb) => {
            pending = cb;
            handle += 1;
            return handle;
        },
        cancelFrame: () => {
            pending = null;
        },
    });

    return {
        host,
        container,
        vThumb,
        hThumb,
        scrollLeftCommits: () => scrollLeftCommits,
        /** 跑掉当前排队的帧（上限 8 次，防止自驱动的无限循环）。 */
        flush() {
            for (let i = 0; i < 8 && pending !== null; i += 1) {
                const cb = pending;
                pending = null;
                cb(0);
            }
        },
    };
}

describe("createPianoRollKernelHost", () => {
    it("构造即注册监听，dispose 后逐条摘除（add/remove 配平）", () => {
        const t = makeHost();
        expect(t.container.totalAdded()).toBeGreaterThan(0);
        t.host.dispose();
        expect(t.container.balanced()).toBe(true);
        expect(t.vThumb.balanced()).toBe(true);
        expect(t.hThumb.balanced()).toBe(true);
    });

    it("重复 dispose 是安全的空操作", () => {
        const t = makeHost();
        t.host.dispose();
        expect(() => t.host.dispose()).not.toThrow();
        expect(t.container.balanced()).toBe(true);
    });

    it("横向上限 = 内容宽（旧实现实测 9125）", () => {
        const t = makeHost();
        t.host.setScrollLeft(999999);
        expect(t.host.getViewport().scrollLeft).toBeCloseTo(100 * 91.25, 6);
        t.host.dispose();
    });

    it("竖向上限 = 1600（旧实现实测值），与视口高无关", () => {
        const t = makeHost();
        t.host.setScrollTop(999999);
        expect(t.host.getViewport().scrollTop).toBeCloseTo(
            PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
            6,
        );
        t.host.dispose();
    });

    it("值域中心往返无损（内核像素 ↔ 值域）", () => {
        const t = makeHost();
        t.host.setValueCenter(62.5);
        expect(t.host.getValueCenter()).toBeCloseTo(62.5, 6);
        t.host.dispose();
    });

    it("帧提交会写 DOM 与量化回调", () => {
        const t = makeHost();
        t.host.setScrollLeft(300);
        t.flush();
        expect(t.scrollLeftCommits()).toBeGreaterThan(0);
        t.host.dispose();
    });
});
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run pianoRollKernelHost`
Expected: FAIL — module not found.

- [x] **Step 3: Implement the data mirror + host**

`pianoRollKernelData.ts` 只声明宿主每帧读取的数据镜像（不含逻辑）。

`pianoRollKernelHost.ts` 的结构（**实现细节以源文件为准，本计划不复制约 350 行代码**，
避免同一份逻辑出现两个事实源）：

- **监听登记表**：内部 `registerListener(target, type, handler)` 统一
  `addEventListener` 并压入「摘除函数」数组；`dispose()` 倒序执行全部摘除函数并置
  `disposed = true`。这是「add/remove 配平」这一退出标准的实现方式，也是漏摘
  window 监听的根因治理。
- **量测**：构造时 `Math.max(1, container.clientWidth/clientHeight)` 取初值；
  有 `ResizeObserver` 时观察容器并在回调里更新量测 + `scroll.reclamp()`，
  **没有时跳过**（node 单测环境无 ResizeObserver，靠注入的量测驱动）。
- **ScrollKernel 配置**：`pxPerSec` 取初值；`projectSec: () => data().projectSec`；
  `rowHeight: 0`、`trackCount: () => 0`、
  `extraContentHeightPx: () => PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX + viewportHeightPx()`；
  `viewportHeightPx` 读量测值。由此竖向上限恒为 1600（见上方实测基线）。
- **帧提交**（`createRenderLoop` 的 `draw`）：读视口 → 更新两条滚动条 thumb →
  写 DOM 同步目标（`rulerContent.style.transform = translateX(-scrollLeft)`、
  `invokeGridRedrawHandler(gridLayer, scrollLeft)`）→ 调用 `onFrame(axis)` 交回面板
  （画布重绘 / 波形总线 / 播放头 DOM 仍由面板实现，行为逐字不变）→ 跨过
  `SCROLL_COMMIT_STEP_PX` 时 `onScrollLeftCommit`。
- **公开 API**：`setScrollLeft` / `setScrollTop` / `setViewport` / `getViewport` /
  `getAxis` / `getValueCenter` / `setValueCenter` / `getScrollbarGeometries` /
  `invalidate` / `dispose`。所有写入**只经 `ScrollKernel` 钳制一次**（与时间轴宿主
  同一约定），宿主自己不算上限。
- **值域适配**：`getValueCenter()` = `centerFromKernelScrollTop({...data().valueDomain,
  scrollTop})`；`setValueCenter(center)` = `setScrollTop(kernelScrollTopFromCenter(...))`。
- **滚动条几何**：复用 `resolvePianoRollScrollbarGeometries`（Task 3），thumb 样式
  按「值变化才写」的字符串 key 去重（与时间轴宿主同一写法）。
- **不注册滚轮 / 键盘 / 中键监听**：留给 Task 7，避免本任务的退出标准被输入语义污染。

- [x] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run pianoRollKernelHost`
Expected: PASS（6 tests）。

- [x] **Step 5: Verify no regression**

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit`
Expected: only the 2 known `keybindingMatch` failures; typecheck clean.

- [x] **Step 6: Commit**

```bash
git add frontend/src/components/layout/pianoRoll/kernel/host/
git commit -m "feat(pianoroll-kernel): kernel host skeleton (scroll ownership + dispose)"
```

### Task 5: Wire the panel behind the flag

- Modify: `PianoRollPanel.tsx` render of the scroller region, keeping the legacy subtree
  intact under the inverse flag.
- The kernel path renders the same sticky layer stack; `applyScrollLayers` becomes a kernel
  callback instead of being driven by `onScroll`.
- **Exit check:** with flag `0` the DOM is byte-identical to today (compare a screenshot and
  the scroller subtree); with flag `1` the panel scrolls via kernel state.

### Task 6: Timeline view sync on kernel state

- Modify: `PianoRollPanel.tsx` sync effects (`syncScrollLeft` / `reconcile` /
  `timelineViewportSync` subscription) to read/write kernel state.
- **Exit check:** with sync enabled, scrolling the timeline moves the parameter editor and
  vice versa; A/B against flag `0` in the same session.

### Task 7: Wheel / keyboard / scrollbar input parity

- Move `onScrollerWheelNative`, `onScrollerKeyDown`, `onScrollerAuxClick`, middle-drag pan,
  and scrollbar thumb/track interaction onto the kernel container.
- **Exit check:** enumerated checklist — wheel zoom, wheel scroll, shift/alt wheel variants,
  PageUp/Down/Home/End/arrows, middle-drag pan, thumb drag, track click page, all verified
  against flag `0`.

### Task 8: Phase 1 verification & docs

- Browser A/B pass on every Phase 1 gesture; screenshot comparison;
  record results in `docs/plans/2026-09-11-timeline-kernel-gap-completion.md` style.
- **Exit check:** `npx vitest run` (only the 2 known failures), `npx tsc -b --noEmit`,
  `npx eslint`, `npx prettier --check` all clean.

---

## Phase 1 — 完成记录（已完成）

**状态：** Task 1–8 全部完成，逐任务提交。开关未显式设置时**默认关闭**，因此合并后
线上行为与迁移前完全一致；显式写 `hifishifter.pianoRollKernel = "1"` 才走内核路径。

**提交序列：**

| 任务 | 提交 |
|---|---|
| Task 1 特性开关 | `d6e613c6` |
| Task 2 值域滚动适配 | `5cb2b3d5` |
| Task 3 双轴滚动条几何 | `306e7589` |
| Task 4 宿主骨架 | `571226c4`（+ 计划扩写 `ef817af1`） |
| Task 5 面板接线 | `5b3a4724` |
| Task 6 时间轴同步 | `70e98bbf` |
| Task 7 输入一致性 | `57d84f31` |

### 浏览器实测（本阶段的主要验收手段）

测试环境：`?mock=1`、1920×1200、DPR 2、同步时间轴开启（偏移实测 200px）。
截图比对脚本位于临时目录（不入库），逐像素比对 3840×2400。

**1. 视觉一致性（flag on vs flag off，静止状态）**

| 区域 | 差异像素 |
|---|---|
| 两条自绘滚动条矩形**之外** | **0（0.0000%）** |
| 竖向自绘滚动条矩形内 | 8 910 |
| 横向自绘滚动条矩形内 | 10 072 |
| 合计 | 18 982（0.2060%） |

滚动条矩形内的差异是**设计使然**：macOS 的原生 overlay 滚动条在静止时自动隐藏，
而内核自绘条常驻显示。除此之外**逐像素一致**。

**2. 回退保证（flag off vs 改动前基线）**

改动前先采一次基线截图，完成后在同等条件下重采：**0 像素差异**。即开关关闭时
行为与迁移前完全相同（这是"可一键回退"的实测依据，而非推断）。

**3. 手势清单（20 项，两模式同条件跑同一脚本）**

滚轮（shift / 普通 / ctrl / alt 四种变体）、PageUp / PageDown / Home / End /
四个方向键、中键拖拽平移、横向 thumb 拖拽、横向轨道点击翻页、竖向 thumb 拖拽、
竖向轨道点击翻页。

- **完全一致：** 大部分步骤两模式逐值相同。
- **亚像素差（0.17–0.50px）：** 原生 scroller 会把位置量化到 0.5 设备像素，内核
  持有精确浮点值。属于浏览器行为差异，非内核缺陷。
- **仅内核模式有位移的四项（滚动条拖拽 / 翻页）：** 旧实现的原生滚动条在静止时
  已自动隐藏，点击落在 scroller 上因此无翻页；内核自绘条常驻，轨道真实存在。
  这是自绘滚动带来的**能力增强**，不是回归。

**4. 滚动条几何与拖拽（对照解析期望）**

| 项目 | 解析期望 | 实测 |
|---|---|---|
| 横向 thumb 长度（原生 `scrollWidth` 口径） | 1864²/10989 = 316.18 | 316.179 |
| 竖向 thumb 长度 | 823²/2423 = 279.54 | 279.541 |
| 横向 thumb 拖拽 +200px | 602 + 200/1547.82×9125 = 1781.08 | 1781 |
| 竖向 thumb 拖拽 +150px | 800 + 150/543.46×1600 = 1241.62 | 1241.5 |
| 轨道点击翻页（竖向） | 一屏 = 823 | 823 |
| 轨道点击翻页（横向） | 一屏 = 1864 | 1864 |

**5. 竖向值域映射**

内核像素 ↔ 值域中心与旧实现同源（复用 `verticalScrollMapping`），实测
`scrollTop=1200 → center=57`，与旧公式独立复算结果一致；静止状态
`scrollTop 533.33 / center 72` 对应旧实现的 `533.5`（量化差）。

### 本阶段在浏览器比对中发现并修复的两个真实缺陷

两处都**无法靠类型检查发现**，只有与旧实现逐像素比对才会暴露：

1. **水平坐标域缺一段（Task 5）**：内核位置恒被钳到 `[0, max]`，而旧实现的
   **绘制**域是 `[−偏移, 内容宽]`（含负值）——同步留白无法表示，参数编辑器网格
   因此比时间轴少偏移那一段、整体错位 200px。修法：内核持有**原生**坐标
   （域 `[0, 内容宽 + 偏移]`，为此给 `ScrollKernel` 增加 `extraContentWidthPx`，
   与既有 `extraContentHeightPx` 对称），对外统一暴露绘制坐标。
   同时修正横向 thumb 的内容尺寸口径（`maxScroll + 视口`，即原生 `scrollWidth`；
   原先误用内容宽，thumb 偏长 20%）。
2. **竖向初始位置被覆盖（Task 5）**：宿主创建 effect 晚于「值域 → 竖向滚动条」
   的 layout effect，内核从 0 起步后首帧镜像回写把位置冲掉——钢琴键盘整体偏移
   **一个八度**。修法：宿主创建时采纳容器当前的两轴原生位置。

另修复 Task 6 发现的同步失效：入站同步写入目标位置后，一个仍按旧语义执行的
`syncScrollLeft` 又从**上一帧的**原生镜像读回旧值覆盖掉它（同步从 1200 拨回 0 时
参数编辑器不动）。内核模式下跳过该回读。

**3. 自绘滚动条拖拽未同步时间轴（收尾审计发现，提交 `44d34407`）**

镜像回写每帧都会写原生 scroller，因而必然触发 `scroll` 事件。这条事件不能一概
当作"用户滚动"（否则每帧都把内核状态推回共享视口），也不能一概忽略（自绘滚动条
拖拽是真实手势，忽略了就比旧实现**功能退化**——旧实现拖原生滚动条是会同步的）。
两者**无法靠数值区分**：拖拽后的镜像回写值与用户值完全相同。

修法：按**来源**区分。宿主新增 `onUserScrollLeft`，只在它自己解析的手势（拖 thumb、
点轨道翻页）后触发；面板据此推送共享视口，而镜像回写与命令式写入都不触发。

实测（同步偏移 200）：拖横向 thumb +300px → 共享视口 = 绘制坐标 + 200（正确推送）；
命令式写入 1500 → 共享视口不变（不再回推）；松手后 20 帧内收敛且无震荡。
这一缺陷是分头测「拖拽」与「同步」都各自通过、但**组合起来**才暴露的——记录在此
以说明为什么验收要求覆盖交叉路径，而不只是逐条手势。

### 工程化说明

- **`dispose()` 的监听配平**：宿主用统一登记表（`registerListener`）保证 add/remove
  配平；单测先断言"确实注册过"再验证配平，避免零监听时成为**空断言**。
- **几何单一来源**：滚动条的绘制、拖拽换算、轨道翻页判定共用 `scrollbarGeometries()`，
  避免三处各算一遍导致"画出来的 thumb 和能拖的范围不一致"。
- **坐标系只在宿主边界换算一次**，面板与各图层统一消费绘制坐标。

### 多窗口尺寸核对（避免"只在 1920×1200 正确"）

在同一会话下另测 1280×800 与 2560×1440（实测视口 1224×423 / 2504×1061）：

| 视口 | nativeTop（两模式） | 竖向上限（两模式） | valueCenter | 竖向 thumb 实测/解析 |
|---|---|---|---|---|
| 1224×423 | 533.5 | 1600 | 72 | 88.447 / 88.447 |
| 1864×823 | 533.5 | 1600 | 72 | 279.541 / 279.541 |
| 2504×1061 | 533.5 | 1600 | 72 | 423.044 / 423.044 |

横向 thumb 同样逐值吻合（144.765 / 316.180 / 539.171 对 1224²·(9125+vw)⁻¹ 等）。

**关键不变量**：竖向上限恒为 **1600**，与视口高无关——这是「额外高度 = 1600 + 视口高」
这一推导在任意窗口尺寸下成立的理由，也是与旧实现 1600px spacer 等价的条件。

### 监听生命周期核对（StrictMode 双挂载）

dev 环境启用了 `StrictMode`，因此挂载期实际经历 create→dispose→create。用
`PROBE_INIT` 在文档加载**前**注入计数器（事后注入会读到 0/0，什么都证明不了）：

| 开关 | window 拖拽监听 add / remove / 存活 |
|---|---|
| flag off | 27 / 14 / **13** |
| flag on | 33 / 17 / **16** |

存活数增量恰为 **3**（宿主的 pointermove / pointerup / pointercancel），与双挂载无关，
即 `dispose()` 未泄漏监听。

### 退出标准核对

| 检查 | 结果 |
|---|---|
| `npx vitest run` | 622 passed / 2 failed（均为 `keybindingMatch` 预先存在失败） |
| `npx tsc -b --noEmit` | 通过 |
| `npx eslint .` | 0 error（13 warning 全部为改动前既有） |
| `npx prettier --check` | 本阶段改动文件全部通过 |
| flag off 像素回退 | 与改动前基线 **0 像素差异**（多次复核） |

---

## Phase 2 — Render Kernel (outline)

Written as its own detailed plan after Phase 1 is verified, because the extraction path depends
on what Phase 1 actually shares. Scope: extract shared kernel modules to a neutral location with
re-export shims, then port grid / keyboard axis / labels / selection / playhead to GL and
activate the glyph pipeline for the 8 `fillText` sites.

**Key tasks:** shared-module extraction (+ timeline suite as the gate) → glyph activation spike
(measure real label set against atlas capacity) → grid/keyboard GL geometry → selection/playhead
→ value-axis labels via glyph program → pixel comparison → verification.

## Phase 3 — Curve GL + Interactions (outline)

Written as its own detailed plan after Phase 2, because curve fidelity findings determine the
final AA approach. Scope: `curveGeometry.ts` triangle-strip expansion, dashed reference lines,
all curve variants, then geometric hit testing and gesture state machine migration.

**Key tasks:** curve expansion pure function (+ tests) → GL curve program → wire all curve
variants → dashed/sub-pixel parity → hit testing → gesture migration (23 entry points, one
checklist item each) → full regression + verification.

---

## Self-Review

**Spec coverage:** the spec's three phases map to the three sections above; the spec's
constraints (value-domain vertical model, flag rollback, pixel parity, timeline sync) each have
a Phase 1 task (Tasks 2, 1, 5, 6). Curve GL and glyph activation are placed in Phases 2–3 as the
spec requires.

**Placeholder scan:** Phase 1 Tasks 1–3 carry complete failing tests, complete implementation,
and exact commands. Tasks 4–8 are declared as phase-entry expansion with explicit files and exit
checks rather than vague instructions — consistent with the spec's staged rollout and with how
the timeline kernel's own remaining items were tracked.

**Type consistency:** `PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX` (1600) is defined once in Task 2 and
consumed by Task 3 and Task 4; `kernelScrollTopFromCenter` / `centerFromKernelScrollTop` names are
used consistently; scrollbar geometry functions are imported from the existing timeline module
rather than re-declared.
