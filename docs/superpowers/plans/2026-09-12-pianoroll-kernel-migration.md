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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run featureFlag`
Expected: FAIL — `PIANO_ROLL_KERNEL_FLAG_KEY` / `isPianoRollKernelEnabled` are not exported.

- [ ] **Step 3: Implement the flag**

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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run featureFlag`
Expected: PASS (4 tests).

- [ ] **Step 5: Verify the pre-existing timeline flag still behaves**

The existing `isTimelineKernelEnabled` reads bare `localStorage` inside a `try`, which is safe
in node only because it is never imported by a test today. Confirm this task did not change its
behaviour:

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit`
Expected: only the 2 known `keybindingMatch` failures; typecheck clean.

- [ ] **Step 6: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run verticalValueScroll`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the adapter**

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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run verticalValueScroll`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run scrollbarSpec`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run scrollbarSpec`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

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

- Create: `pianoRoll/kernel/host/pianoRollKernelHost.ts`
- Reuse `createScrollKernel` (pixel scrollTop; vertical max set to
  `PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX`), `createRenderLoop`, `createTimelineAxis`.
- Host owns: container, both scrollbar thumbs/tracks, data mirror getter, DOM sync targets
  (ruler content layer, axis column, grid layer).
- **Exit check:** host constructs and disposes without leaking listeners; unit test asserts
  `dispose()` unregisters every listener it added.

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
