# PianoRoll Kernel Migration Design

## Status

Approved for implementation on branch `feature/timeline-unified-render-kernel`.

This document is based on the current branch implementation (HEAD `b62c4390` plus the
uncommitted kernel parity work). It is the follow-up to
`docs/plans/2026-09-11-timeline-unified-render-kernel-design.md`, which explicitly listed
the parameter editor (PianoRoll) migration as a non-goal for that phase:

> 参数编辑器（PianoRoll）迁移（后续复用同一渲染内核）

## Goal

Migrate the parameter editor (PianoRoll) panel onto the same render-kernel architecture as
the timeline, with **zero functional or visual loss**:

- replace the native scroller with kernel-owned, self-drawn scrolling on both axes;
- render the panel body (grid, piano keyboard axis, selection, playhead, parameter curves)
  through the render kernel instead of per-frame Canvas 2D painting;
- preserve every existing interaction, gesture, keyboard shortcut, and context menu;
- keep the existing vertical **value-domain** scrolling model (center/span), not a pixel-row
  model;
- keep the feature-flag rollback guarantee: disabling the flag restores the current path
  with zero impact.

## Problem Statement

The timeline track area has already been migrated to a self-drawn WebGL2 kernel. The
PianoRoll panel still uses the legacy architecture:

| Concern | Legacy PianoRoll | Timeline kernel |
|---|---|---|
| Scrolling | native `overflow-x-scroll overflow-y-scroll` + a 1600px spacer div | `ScrollKernel` owns `scrollLeft`/`scrollTop`; drawing follows via a content-coordinate window |
| Vertical model | value-domain `center`/`span` mapped onto a fixed 1600px scroll range | pixel rows (`scrollTop`) |
| Layer sync | `pianoRollViewportBus` → per-layer `paint` on every scroll event | single rAF frame commit; scroll frames only update a `u_viewOrigin` uniform |
| Panel body | one 2D canvas repainted per frame (curves, grid, keys, overlay text) | GL instanced geometry + Canvas 2D detail layer |
| Text | `ctx.fillText` (8 call sites) | Canvas 2D detail layer today; a complete GL glyph pipeline exists but is unwired |

The panel body is repainted wholesale on every frame that invalidates (including playback,
where only the playhead moves). That is the performance ceiling this migration removes.

## Scope

### In Scope

- new `frontend/src/components/layout/pianoRoll/kernel/` subtree mirroring the timeline
  kernel layout (`host/`, `gl/`, `scene/`, `interaction/`, `input/`);
- self-drawn horizontal + vertical scrollbars for the parameter editor;
- reuse of `ScrollKernel`, `createTimelineAxis`, `RenderLoop`, GL context/instance/glyph
  modules from the timeline kernel (extracted to a shared location where needed);
- GL rendering of: pitch/param grid, piano keyboard axis, value axis labels and ticks,
  selection rectangles, playhead line, reference/scale highlight bands;
- GL rendering of parameter curves (original, edited, selection overlay, secondary params,
  detected pitch reference lines, clipboard preview) via polyline geometry;
- activation of the existing, tested GL glyph pipeline (`glyph/*` + `gl/glyphProgram`) for
  all text the panel currently draws with `fillText`;
- geometric hit testing and a kernel-side gesture state machine covering every interaction
  the panel has today;
- a feature flag for stepwise rollout and one-click rollback.

### Out of Scope

- changing parameter/DSP semantics, backend protocol, or Redux state shape;
- redesigning the panel chrome (toolbar, param pills, dialogs, popovers, dropdowns) — those
  stay DOM;
- the MIDI import dialog, vowel chart, and other floating windows;
- the waveform surface already migrated by `2026-08-27-waveform-webgl-refactor-design.md`
  (it is registered as an independent viewport layer; the kernel must keep driving it);
- visual redesign — the target is pixel parity with the current rendering.

## Key Constraints

1. **Vertical scrolling keeps the value-domain model.** `center`/`span` ↔ 1600px virtual
   range mapping (`verticalScrollMapping.ts`) is preserved verbatim; the kernel draws the
   scrollbar and owns the drag, but the mapping arithmetic is unchanged. This avoids
   re-tuning wheel step and drag ratio feel.
2. **Feature flag rollback.** Disabling the flag must restore the current path exactly.
3. **Pixel parity.** Curves, key labels, grid colors, selection styling, and overlay text
   must match the current rendering. Any intentional deviation must be listed in the design
   doc and confirmed.
4. **`timelineViewportSync` must keep working.** Timeline↔parameter-editor view sync is a
   documented feature (`useTimelineState`, `pianoRollViewportBus`); the kernel becomes
   another consumer/producer of that same sync contract.

## Architecture

### Directory layout

```
pianoRoll/kernel/
  host/pianoRollKernelHost.ts     imperative host: GL resources, rAF loop, input wiring
  host/pianoRollKernelData.ts     data mirror consumed by the host (read per frame)
  scroll/verticalValueScroll.ts   value-domain ↔ 1600px scroll range adapter
  scene/                          render-model builders (pure, unit-testable)
    gridInstances.ts              pitch/param grid + octave separators
    keyboardInstances.ts          piano keys + key separators
    curveGeometry.ts              polyline → triangle-strip expansion
    selectionInstances.ts         selection rect + highlight bands
  interaction/                    pure geometry + gesture state machine
    hitTest.ts                    curve/selection/grid hit testing
    gestureState.ts               gesture dispatch and lifecycle
  gl/                             curveProgram.ts (+ reuse of timeline gl modules)
  glyph/                          reuse timeline glyph modules (activate)
```

Shared code currently living under `timeline/kernel/` that both panels need
(`glContext`, `instanceBuffer`, `instanceLayout`, `sdfBoxProgram`, `glyph/*`,
`glyphProgram`, `renderLoop`, `scrollKernel`, `timelineAxis`, `canvasRaster`,
`devicePixelLine`) moves to a neutral location (e.g. `components/layout/renderKernel/`)
with re-export shims left behind, so the timeline kernel keeps working unchanged and the
move is verifiable in isolation.

### Frame model

The kernel adopts the timeline's proven invariant set:

- geometry is built in **content coordinates** and cached on the GPU;
- a scroll frame updates only `u_viewOrigin` — no geometry rebuild, no CPU repaint;
- rebuilds are triggered by dirty flags and by scrolling past a margin;
- drawing and hit testing consume the **same** geometry functions ("what you see is what
  you can click");
- the value-domain vertical axis maps to content-space Y through one shared function used by
  geometry, hit testing, and the scrollbar.

### Curve rendering

Parameter curves are continuous polylines over thousands of points, so they do not fit the
box-instance model. The design uses **triangle-strip expansion** built on the CPU:

- a pure `curveGeometry.ts` converts `(frameIndex, value)` samples into a thick-line triangle
  strip in content coordinates, with a configurable half-width in CSS px expanded to device
  pixels;
- anti-aliasing comes from the same SDF-style coverage the timeline's box program uses (a
  smooth edge band), not from MSAA, so density is controllable and consistent across
  platforms;
- dashed reference lines (detected pitch, clipboard preview) are emitted as segment batches
  by the same builder — `setLineDash` semantics are reproduced by subdividing in content
  space, which is also what keeps dashes visually fixed under zoom;
- physical-pixel alignment reuses the timeline's `devicePixelLine` helpers so 1px lines do
  not shimmer.

This is the highest-risk part of the migration and is isolated in Phase 3 so that Phases 1
and 2 can ship independently.

## Data Flow

```
Redux session (paramView, pitchView, curves, selection, playhead)
        │  (low-frequency mirror, read per frame by the host)
        ▼
pianoRollKernelHost ── builds ──▶ scene/ render models (content coordinates)
        │                                   │
        │                                   ▼
        │                          GL instance/triangle buffers (uploaded on rebuild)
        │
   ScrollKernel (scrollLeft, vertical value-domain scrollTop)
        │
        └── per frame: update u_viewOrigin → single draw call per program
                       + glyph program draw call
                       + DOM sync (ruler, axis column, scrollbars)
```

## Rollout Plan

Three phases, each independently shippable, verifiable, and revertible.

| Phase | Content | Exit criteria |
|---|---|---|
| **1 · Scroll/viewport kernel** | Self-drawn scrollbars (both axes, value-domain vertical preserved), `ScrollKernel` + unified axis projection, rAF frame commit. Painting stays Canvas 2D. | Scroll/zoom feel identical; timeline sync unaffected; frame rate not worse; legacy flag off == today |
| **2 · Render kernel** ✅ **已完成** | Grid, keyboard axis, value labels/ticks, selection, playhead, highlight bands move to GL instanced geometry; the existing glyph pipeline takes over all `fillText`. Curves stay on Canvas 2D in the detail layer. | Pixel comparison within tolerance ✅ 0.0818%（96% 差异为 1/255）；text quality matches ✅（9px/bold 9px 完全一致）；playback frames no longer repaint curves ✅ 每帧 Canvas2D 绘图调用 318 → **0**。证据见 `plans/2026-09-12-pianoroll-kernel-phase2.md` 的完成记录 |
| **3 · Curve GL + interactions** | Polyline triangle-strip curve rendering (all curve variants); geometric hit testing + gesture state machine migration. | Curve fidelity comparison passes; every gesture regression-checked in the browser |
| ↳ **曲线 GL 部分** ✅ **已完成** | 曲线改为 GL 三角带（**实际用带距离属性的三角形**，见计划自审）；渲染层按设备像素列抽稀 | 像素保真 ✅ 同会话 A/B 差异 0.0352%，包络中位 1px；性能 ✅ longtask 30 次/2240ms → **0 次/0ms**；每帧 `lineTo` ✅ 213,955 → **0**。7 种变体中仅 3 种端到端比对（mock 限制，已记录）。证据见 `plans/2026-09-12-pianoroll-kernel-phase3.md` 的 Task 4/5 记录 |
| ↳ **手势迁移** ✅ **纯逻辑部分已完成** | 抽出 `gestureHitTest`（19 项单测）与 `dragArithmetic`（23 项），命中判定 / 选区坐标 / 拖拽增量 / 边缘自动滚动全部接线 | 浏览器复核 ✅ **9 个处理器 / 27 个观测点**：离散字段 0 差异、连续字段 0 超差、效果断言 10/10、dispatch 总和一致。**未达成**：「hook 可测量地缩小」按行数衡量不可达（3,876 → 3,897，新模块注释占比高），有效指标是**重复处数**（beat→帧 6→0、边缘滚动魔法数 2→0、曲线命中内联副本 3→0 份）。手势**状态机**本身未迁移，理由见计划 Task 6 |

Each phase has its own flag value (or its own flag) so a phase can be reverted without
reverting the previous ones.

**默认值状态（已变更）：** 四个开关的未显式设置默认值原为"跟随 `import.meta.env.DEV`"
（生产构建关闭）。现统一为**开启，与构建模式无关**——内核已是默认渲染路径，不再是
opt-in。开关只剩**逃生门**职责：显式写 `"0"` 即退回既有实现，无需重新发版。

这样改的原因：`TAURI_UI_MODE=build` 跑的是生产包（`DEV === false`），默认跟随 DEV 会让
**打包后静默退回旧渲染器**——开发时看到新实现、打包后看到旧实现。该问题实际发生过：
Windows 上的卡顿报告全部来自旧实现，与新内核无关（详见 Phase 3 计划 R8）。

验证用了三层（单测层测不出 DEV 回落）：运行期断言默认 `true`；**源码级**断言
`featureFlag.ts` 去注释后不含 `import.meta.env.DEV`；**构建产物**检查编译结果为
`e!=="0"` 且 `import.meta.env` 出现 0 次。浏览器端实测：默认输出的整页截图与显式写
`"1"` **字节完全相同**；显式写 `"0"` 时 GL 画布不再挂载。

**Shared-module extraction (done):** the spec's architecture section called for moving
the modules both panels share into a neutral location. That landed as Task 8 of the
Phase 2 plan: `components/layout/renderKernel/` now holds the GL programs, glyph
pipeline, scroll kernel, axis projection and rasterisation contracts, with dependencies
running `timeline` / `pianoRoll` → `renderKernel`. No re-export shims were left behind
(nothing outside `src/` referenced the old paths and the project has no path aliases,
so shims would have been dead code from the start); the directory README records the
rationale and the verification gates.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Curve GL fidelity (AA, width, dashes, pixel snapping) | Visual regression in the most visible element | Isolated in Phase 3; pure-function geometry + pixel-comparison harness; keep Canvas 2D curves as a flagged fallback until parity is proven |
| Gesture migration surface (23 event entry points, single 3875-line hook) | Interaction regressions are subtle and numerous | Enumerate every gesture as a checklist in the plan; migrate the state machine behind the kernel only after Phase 2 lands; browser self-test each gesture |
| Glyph atlas capacity (CJK labels, many sizes) | Missing/clipped text | Atlas is multi-page with a documented no-eviction policy; measure real label set in Phase 2 before committing |
| Shared-module extraction breaks the timeline kernel | Regression in already-shipped work | Extraction is its own task with re-export shims and the existing timeline kernel test suite as the gate |
| Vertical value-domain mapping drifts | Scroll feel changes subtly | Mapping stays in `verticalScrollMapping.ts` untouched; kernel consumes it, does not reimplement it |

## Testing Strategy

- **Pure-function unit tests** for every scene builder, the curve expansion, hit testing, and
  the value-domain scroll adapter (mirrors the timeline kernel's approach).
- **Invariant tests**: geometry builders and hit testing share inputs and produce consistent
  results; a scroll frame triggers zero geometry rebuilds (spy assertion).
- **Pixel comparison**: dev-shot screenshots of the panel in both modes, compared per phase.
- **Browser self-test**: every gesture and shortcut exercised through `dev-shot.mjs` against
  `?mock=1`, with A/B against the legacy path in the same session.
- **Existing suites stay green**: `npx vitest run`, `npx tsc -b --noEmit`, `npx eslint`,
  `npx prettier --check`.

### Test environment constraint (verified on this branch)

Vitest runs in the **node** environment — there is no jsdom. A probe on this branch confirmed
`typeof localStorage`, `typeof window`, and `typeof document` are all `undefined` inside tests.
Consequences for this migration:

- any module a test imports must not reference browser globals at module-evaluation time;
  storage access goes through `globalThis.localStorage` behind a `typeof`/null guard;
- tests that need browser APIs install their own minimal stubs rather than assuming a DOM;
- DOM-level verification (layout, canvas output, gestures) happens in the browser harness
  (`dev-shot.mjs`), not in Vitest.

This is why Phase 1's flag task ships with a self-contained storage stub instead of a test that
touches bare `localStorage`.

## Non-Goals

- Changing audible behavior, parameter smoothing, or DSP.
- Changing backend commands or payload shapes.
- Migrating panel chrome (toolbars, dialogs, popovers) into the kernel.
- Unifying the timeline and parameter-editor scroll models (they stay different by design).
