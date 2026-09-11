# 时间轴统一渲染内核 · 阶段 0（Spike）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个独立的时间轴渲染内核最小实现（自绘滚动 + 单 WebGL2 + 字形图集 + 几何命中雏形），在真机 400/1000 clip 场景下验证「滚动零几何重建」的帧率与手感，作为是否全面推进阶段 1 的决策依据。

**Architecture:** React 只保留外壳；`ScrollKernel` 持有视口状态并独占钳制逻辑；`RenderLoop` 以脏标记 + rAF 驱动；全部几何以内容坐标常驻 GPU，滚动帧只更新 `u_viewOrigin` uniform。Spike 不接入生产交互路径，通过 `hifishifter.timelineKernel` flag 在时间轴区域条件渲染。

**Tech Stack:** TypeScript / React 19 / WebGL2 / Vitest（`npm test`）/ 现有 `buildTimelineTicks`、`buildClipBodyInstance`、`timelineCanvasStyle`、`perfProject` 合成场景。

**参考设计：** `docs/plans/2026-09-11-timeline-unified-render-kernel-design.md`

**目录约定：** 新内核代码统一放在 `frontend/src/components/layout/timeline/kernel/` 下，Spike 期间不与现有 `runtime/` 目录交叉引用（除明确列出的复用模块）。

---

## Task 1: ScrollKernel（视口状态 / 钳制 / 缩放锚点）

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/scrollKernel.ts`
- Test: `frontend/src/components/layout/timeline/kernel/scrollKernel.test.ts`

**Step 1: 写失败测试**

```ts
import { describe, expect, it, vi } from "vitest";
import { createScrollKernel } from "./scrollKernel";

function makeKernel(overrides: Partial<Parameters<typeof createScrollKernel>[0]> = {}) {
    return createScrollKernel({
        pxPerSec: 100,
        rowHeight: 80,
        projectSec: () => 1000,
        trackCount: () => 10,
        viewportWidthPx: () => 800,
        viewportHeightPx: () => 400,
        ...overrides,
    });
}

describe("scrollKernel", () => {
    it("钳制 scrollLeft 到 [0, contentWidth - viewportWidth]", () => {
        const k = makeKernel();
        k.setScrollLeft(-50);
        expect(k.get().scrollLeft).toBe(0);
        k.setScrollLeft(999999);
        expect(k.get().scrollLeft).toBe(1000 * 100 - 800);
    });

    it("内容不足一屏时 maxScroll 为 0", () => {
        const k = makeKernel({ projectSec: () => 1 });
        k.setScrollLeft(500);
        expect(k.get().scrollLeft).toBe(0);
    });

    it("缩放保持指针锚点下的时间不变", () => {
        const k = makeKernel();
        k.setScrollLeft(300);
        const anchorScreenX = 200;
        const anchorSecBefore = (300 + anchorScreenX) / 100;
        k.setZoom(200, anchorScreenX);
        expect(k.get().pxPerSec).toBe(200);
        const anchorSecAfter = (k.get().scrollLeft + anchorScreenX) / 200;
        expect(anchorSecAfter).toBeCloseTo(anchorSecBefore, 6);
    });

    it("缩放后 scrollLeft 仍被钳制", () => {
        const k = makeKernel();
        k.setScrollLeft(0);
        k.setZoom(1000, 0);
        expect(k.get().scrollLeft).toBe(0);
    });

    it("状态变化通知订阅者，未变化不通知", () => {
        const k = makeKernel();
        const spy = vi.fn();
        k.subscribe(spy);
        k.setScrollLeft(100);
        expect(spy).toHaveBeenCalledTimes(1);
        k.setScrollLeft(100);
        expect(spy).toHaveBeenCalledTimes(1);
    });
});
```

**Step 2: 运行测试确认失败**

Run: `cd frontend && npx vitest run scrollKernel`
Expected: FAIL —— `Failed to resolve import "./scrollKernel"`

**Step 3: 实现**

要点：钳制只在这里做一次；`setZoom` 用"锚点秒不变"反算 `scrollLeft`；变更才通知。

```ts
/**
 * 时间轴内核 · 视口状态（自绘滚动）。
 *
 * 【主要内容】持有 scrollLeft / scrollTop / pxPerSec / rowHeight，并独占
 * 全部边界钳制逻辑；对外只暴露读写与订阅。
 *
 * 【作用】替代原生 scroller 的视口职责。原生滚动下"写入被浏览器钳制后回读"
 * 是错位问题的根源；这里所有钳制都在写入时一次算清，读取值恒为真值。
 *
 * 【与其他模块的关系】
 * - 上游：输入层（wheel / 滚动条 / 键盘）调用 set*；
 * - 下游：RenderLoop 订阅变更后标脏重绘；几何构建读 get() 取视口。
 */
export interface TimelineViewportState {
    readonly scrollLeft: number;
    readonly scrollTop: number;
    readonly pxPerSec: number;
    readonly rowHeight: number;
}

export interface ScrollKernelOptions {
    pxPerSec: number;
    rowHeight: number;
    /** 内容总时长（秒），用于算内容宽度。 */
    projectSec: () => number;
    /** 轨道总数，用于算内容高度。 */
    trackCount: () => number;
    viewportWidthPx: () => number;
    viewportHeightPx: () => number;
    minPxPerSec?: number;
    maxPxPerSec?: number;
}

export interface ScrollKernel {
    get(): TimelineViewportState;
    setScrollLeft(px: number): void;
    setScrollTop(px: number): void;
    /** 以 anchorScreenX（视口内 CSS px）为锚点缩放。 */
    setZoom(pxPerSec: number, anchorScreenX: number): void;
    subscribe(listener: () => void): () => void;
    contentWidthPx(): number;
    contentHeightPx(): number;
    maxScrollLeft(): number;
    maxScrollTop(): number;
}
```

完整实现按上述测试约束补全（含 `clamp`、`emit` 去重、`minPxPerSec/maxPxPerSec` 夹取）。

**Step 4: 运行测试确认通过**

Run: `cd frontend && npx vitest run scrollKernel`
Expected: PASS（5 passed）

**Step 5: 提交**

```bash
git add frontend/src/components/layout/timeline/kernel/scrollKernel.ts frontend/src/components/layout/timeline/kernel/scrollKernel.test.ts
git commit -m "feat(timeline-kernel): add scroll kernel with anchored zoom and clamping"
```

---

## Task 2: wheel 输入归一化

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/input/normalizeWheel.ts`
- Test: `frontend/src/components/layout/timeline/kernel/input/normalizeWheel.test.ts`

**Step 1: 写失败测试**

覆盖三种 `deltaMode`（0=pixel、1=line、2=page）与水平/竖直双轴；line 按行高换算、page 按视口高度换算。

```ts
it("deltaMode=1 按行高换算", () => {
    expect(normalizeWheelDelta(3, 1, { lineHeightPx: 16, pageHeightPx: 800 })).toBe(48);
});
it("deltaMode=2 按视口高度换算", () => {
    expect(normalizeWheelDelta(1, 2, { lineHeightPx: 16, pageHeightPx: 800 })).toBe(800);
});
it("deltaMode=0 原样返回", () => {
    expect(normalizeWheelDelta(-12.5, 0, { lineHeightPx: 16, pageHeightPx: 800 })).toBe(-12.5);
});
```

**Step 2: 运行确认失败** — `npx vitest run normalizeWheel`

**Step 3: 实现** —— 纯函数，无副作用；导出 `normalizeWheelDelta(delta, deltaMode, ctx)` 与 `readWheelPixels(event, ctx)`（从 `WheelEvent` 取双轴）。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): add wheel delta normalization`

---

## Task 3: 字形布局与测量缓存

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/glyph/glyphLayout.ts`
- Test: `frontend/src/components/layout/timeline/kernel/glyph/glyphLayout.test.ts`

**Step 1: 写失败测试**

`layoutText(text, fontKey, maxWidthPx, measure)`：逐字累加宽度；超宽截断并追加省略号；`measure` 结果按 `(char, fontKey)` 缓存，重复调用不重复测量。

```ts
it("按字符宽度切分为字形序列", () => {
    const measure = (t: string) => t.length * 10;
    const run = layoutText("abc", "12px sans", 100, measure);
    expect(run.glyphs.map((g) => g.char)).toEqual(["a", "b", "c"]);
    expect(run.glyphs[2].x).toBe(20);
    expect(run.width).toBe(30);
    expect(run.truncated).toBe(false);
});

it("超宽时截断并追加省略号", () => {
    const measure = (t: string) => t.length * 10;
    const run = layoutText("abcdef", "12px sans", 35, measure);
    expect(run.truncated).toBe(true);
    expect(run.width).toBeLessThanOrEqual(35);
    expect(run.glyphs.at(-1)?.char).toBe("…");
});

it("测量结果按 (char, fontKey) 缓存", () => {
    const measure = vi.fn((t: string) => t.length * 10);
    layoutText("aaa", "12px sans", 100, measure);
    layoutText("aa", "12px sans", 100, measure);
    expect(measure.mock.calls.filter(([t]) => t === "a")).toHaveLength(1);
});
```

**Step 2: 运行确认失败**

**Step 3: 实现** —— 纯逻辑，`measure` 由调用方注入（生产注入离屏 Canvas2D `measureText`，测试注入桩）。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): add glyph layout with measurement cache`

---

## Task 4: 字形图集分配器（shelf pack + 多页）

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/glyph/glyphAtlas.ts`
- Test: `frontend/src/components/layout/timeline/kernel/glyph/glyphAtlas.test.ts`

**Step 1: 写失败测试**

```ts
it("分配不重叠的矩形", () => {
    const atlas = createGlyphAtlas({ pageSizePx: 64, paddingPx: 1, maxPages: 1 });
    const a = atlas.allocate(20, 12)!;
    const b = atlas.allocate(20, 12)!;
    expect(a).not.toEqual(b);
    expect(a.y + 12 <= b.y || b.y + 12 <= a.y || a.x + 20 <= b.x || b.x + 20 <= a.x).toBe(true);
});

it("单页放满后溢出到新页", () => {
    const atlas = createGlyphAtlas({ pageSizePx: 32, paddingPx: 0, maxPages: 3 });
    let last = atlas.allocate(30, 30)!;
    for (let i = 0; i < 5; i += 1) last = atlas.allocate(30, 30)!;
    expect(last.page).toBeGreaterThan(0);
    expect(atlas.pageCount()).toBeLessThanOrEqual(3);
});

it("超过页数上限返回 null", () => {
    const atlas = createGlyphAtlas({ pageSizePx: 16, paddingPx: 0, maxPages: 1 });
    atlas.allocate(16, 16);
    expect(atlas.allocate(16, 16)).toBeNull();
});
```

**Step 2: 运行确认失败**

**Step 3: 实现** —— shelf packing：每页维护当前 shelf 的 `y / height / cursorX`；放不下则起新 shelf；页满则新页。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): add multi-page glyph atlas allocator`

---

## Task 5: 网格实例构建（内容坐标）

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/scene/gridInstances.ts`
- Test: `frontend/src/components/layout/timeline/kernel/scene/gridInstances.test.ts`

**Step 1: 写失败测试**

输入 `ticks`（`TimelineTick[]`，其 `contentPx` 已是内容坐标）、视口窗口、DPR；输出 `FlatInstance[]`（x/y/w/h/rgba），要求：仅包含窗口内刻度、x 设备像素吸附、强弱线颜色区分、纵向只覆盖 `[0, contentBottomPx]`。

```ts
it("只产出视口窗口内的网格线", () => {
    const ticks = [
        makeTick({ contentPx: 0, isStrongGridLine: true }),
        makeTick({ contentPx: 100 }),
        makeTick({ contentPx: 200 }),
    ];
    const out = buildGridInstances({
        ticks, viewportLeftPx: 150, widthPx: 100,
        viewportTopPx: 0, heightPx: 400, contentBottomPx: 400, dpr: 2,
        weakRgba: [1, 1, 1, 0.1], strongRgba: [1, 1, 1, 0.2],
    });
    // 内容坐标：窗口为 [150, 250]，仅 contentPx=200 命中。
    expect(out.map((i) => i.x)).toEqual([200]);
});
```

**Step 2: 运行确认失败**

**Step 3: 实现** —— 纯函数；`x` 经 `Math.round(x * dpr) / dpr` 吸附；线宽 `1 / dpr`。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): build grid instances in content coordinates`

---

## Task 6: clip 实例构建（复用现有样式与实例布局）

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/scene/clipInstances.ts`
- Test: `frontend/src/components/layout/timeline/kernel/scene/clipInstances.test.ts`
- 复用：`runtime/timelineClipGlRenderer.ts` 的 `buildClipBodyInstance`、`CLIP_INSTANCE_FLOATS`；`runtime/timelineCanvasStyle.ts` 的 `buildTimelineClipVisualStyle`

**Step 1: 写失败测试**

```ts
it("按可见窗口产出实例，且实例数 × 25 float 等于缓冲长度", () => {
    const out = buildClipInstances({
        clips: [makeClip({ id: "a", startSec: 0, lengthSec: 10 }), makeClip({ id: "b", startSec: 20, lengthSec: 10 })],
        pxPerSec: 100, viewportLeftPx: 0, widthPx: 800, rowHeight: 80, startTrackIndex: 0,
        rowByTrackId: new Map([["t1", 0]]), darkMode: false,
    });
    expect(out.count).toBe(1);
    expect(out.instances.length).toBeGreaterThanOrEqual(25);
});
```

**Step 2: 运行确认失败**

**Step 3: 实现** —— 复用现有实例布局与样式函数，输出 `{ instances: Float32Array; count: number }`；缓冲按需倍增、跨帧复用。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): build clip instances reusing existing style pipeline`

---

## Task 7: GL context + sdf-box program

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/gl/glContext.ts`
- Create: `frontend/src/components/layout/timeline/kernel/gl/sdfBoxProgram.ts`
- Test: `frontend/src/components/layout/timeline/kernel/gl/sdfBoxProgram.test.ts`（仅测纯逻辑：实例缓冲上传的分片计算、尺寸/DPR 光栅化参数）

**Step 1: 写失败测试** —— 覆盖"物理尺寸 = round(css × dpr)"与"缓冲增长策略"两个纯逻辑函数。

**Step 2: 运行确认失败**

**Step 3: 实现**
- `glContext.ts`：创建 `webgl2` context、resize（物理尺寸 + `u_resolution`）、dpr 读取；
- `sdfBoxProgram.ts`：着色器源码从 `runtime/timelineClipGlRenderer.ts` **复制改造**（Spike 阶段允许重复，阶段 1 再合并去重），支持 `u_viewOrigin` 平移 uniform；实例属性 25 float 与现有布局一致。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): add webgl2 context and sdf box program`

---

## Task 8: glyph-quad program + 字形光栅化

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/gl/glyphRasterizer.ts`
- Create: `frontend/src/components/layout/timeline/kernel/gl/glyphProgram.ts`
- Test: `frontend/src/components/layout/timeline/kernel/gl/glyphProgram.test.ts`（实例布局转换的纯逻辑）

**Step 1: 写失败测试** —— `layoutText` 输出 + 图集分配 → glyph quad 实例（x/y/w/h/u0/v0/u1/v1/rgba）的转换函数。

**Step 2: 运行确认失败**

**Step 3: 实现**
- `glyphRasterizer.ts`：离屏 Canvas2D 渲染单字符到像素缓冲（按 dpr），供上传图集；生产注入真实 `measureText`；
- `glyphProgram.ts`：纹理四边形实例化 program（`u_atlas` + `u_viewOrigin`）。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): add glyph quad program and rasterizer`

---

## Task 9: RenderLoop（脏标记 + rAF + 层序）

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/renderLoop.ts`
- Test: `frontend/src/components/layout/timeline/kernel/renderLoop.test.ts`

**Step 1: 写失败测试**

```ts
it("同一帧内多次 invalidate 只绘制一次", () => {
    const draw = vi.fn();
    const loop = createRenderLoop({ draw, requestFrame: (cb) => { cb(0); return 1; }, cancelFrame: () => {} });
    loop.invalidate();
    loop.invalidate();
    expect(draw).toHaveBeenCalledTimes(1);
});

it("滚动帧不触发几何重建（由调用方以 spy 断言）", () => {
    // 由 Task 10 的集成测试使用：构造 build 函数 spy，仅改变 viewOrigin 后 render，
    // 断言 build 调用次数不变。
});
```

**Step 2: 运行确认失败**

**Step 3: 实现** —— 注入 `requestFrame/cancelFrame` 以便测试；`invalidate()` 幂等合并；`stop()` 清理。

**Step 4: 运行确认通过**

**Step 5: 提交** — `feat(timeline-kernel): add dirty-flag render loop`

---

## Task 10: 宿主组件 + 自绘滚动 + flag 接入

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx`
- Create: `frontend/src/components/layout/timeline/kernel/input/scrollbars.ts`（自绘滚动条几何与命中，纯函数）
- Test: `frontend/src/components/layout/timeline/kernel/input/scrollbars.test.ts`
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（flag 开时在时间轴区域条件渲染内核视图）

**Step 1: 写失败测试** —— 滚动条几何（thumb 位置/长度/命中）纯函数。

**Step 2: 运行确认失败**

**Step 3: 实现**
- `scrollbars.ts`：`computeScrollbar({ contentSize, viewportSize, scroll, thicknessPx })` → `{ track, thumb }`；`hitTestScrollbar(...)`；
- `TimelineKernelView.tsx`：
  - 从 Redux 读 `tracks` / `clips`（只读，复用现有 selector）；
  - 创建 `ScrollKernel` + `RenderLoop` + GL 渲染器；canvas 尺寸随容器 `ResizeObserver`；
  - 输入：wheel（归一化 → 滚动或 Ctrl 缩放）、滚动条拖拽、键盘、中键平移；
  - 渲染：网格（Task 5）+ clip 块面（Task 6）+ clip 名称文字（Task 8）；
  - flag：`hifishifter.timelineKernel`（默认关，读 `localStorage`），未开启时不改变现有行为。

**Step 4: 运行确认通过**

Run: `cd frontend && npx vitest run scrollbars && npx tsc -b --noEmit`
Expected: PASS

**Step 5: 提交** — `feat(timeline-kernel): add spike host view behind feature flag`

---

## Task 11: 真机验证与性能报告

**Files:**
- Create: `docs/plans/2026-09-11-timeline-kernel-spike-report.md`

**Step 1: 生成性能工程**

在应用 dev 环境用 PERF 面板生成 400 clip（10×40）与 1000 clip（10×100）场景，开启 `hifishifter.timelineKernel`。

**Step 2: 采集数据**

用 Chrome DevTools Performance 录制：
- 连续水平滚动 5 秒（鼠标滚轮 + 触摸板各一次）
- 全览缩放 → 放大 → 缩小各一轮
- 记录：帧耗时分布（p50/p95/p99）、长任务数、`render()` 内几何重建次数（用内核 stats 计数）

**Step 3: 写报告**

报告需回答：
- 滚动帧是否零几何重建（stats 计数为 0）；
- p95 帧耗时是否 < 2ms、是否稳定 60fps；
- 滚动手感（触摸板惯性、快速滚动）是否可接受；
- 文字质量（与旧实现截图对比）；
- **结论：继续阶段 1 / 退回 WebGL2 + Canvas2D 细节层方案**。

**Step 4: 提交** — `docs(timeline-kernel): add spike performance report`

---

## 完成标准（Spike 出口条件）

1. `npm test` 全绿；`npx tsc -b --noEmit` 无错。
2. 1000 clip 场景：滚动帧零几何重建，p95 < 2ms，稳定 60fps。
3. 滚动手感与文字质量经真机确认可接受。
4. 报告给出明确的「继续 / 退回」结论。
