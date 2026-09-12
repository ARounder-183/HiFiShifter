# PianoRoll Kernel Migration — Phase 3 (Curve GL + Interaction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the parameter editor's curve layers onto WebGL2 polyline geometry, and extract the pointer/keyboard gesture logic out of the 3,875-line interactions hook into tested pure functions.

**Architecture:** Phase 2 left a hybrid: GL for the static layers (grid, keyboard, axis text) and the dynamic overlay (playhead, selection), Canvas2D for curves. Phase 3 adds a polyline program to `renderKernel/gl/`, builds curve geometry in content coordinates with per-frame x reprojection, and keeps Canvas2D as a flagged fallback until parity is proven. The gesture half moves hit-testing and drag arithmetic into pure modules (mirroring `timeline/kernel/interaction/*`) so they can be unit-tested instead of only browser-tested.

**Tech Stack:** React 19, TypeScript 5.9, Redux Toolkit, WebGL2, Canvas 2D, Vitest, Playwright-core (`frontend/scripts/dev-shot.mjs`)

**Spec:** `docs/superpowers/specs/2026-09-12-pianoroll-kernel-migration-design.md`
**Phase 2 record:** `docs/superpowers/plans/2026-09-12-pianoroll-kernel-phase2.md`

**Note on the frame-time claim (read R5 before promising anything):** Phase 3 is expected to reduce JS→native call counts, **not** frame time. Reconnaissance found no measurable frame-time headroom to reclaim on this machine, so the plan's exit metrics are call counts and pixel parity. See R5 for the control-group measurement that established this.

---

## Reconnaissance findings (measured before writing any task)

These change what Phase 3 can claim, so they come first.

**R1 — The curve layers, exactly (5 `drawCurveTimed` call sites + 2 clipped variants).**

| Layer | `render.ts` | Width | Dash | Notes |
|---|---|---|---|---|
| Reference pitch overlays | 889–915 | 3.2 highlighted / 2.6 | solid | per-overlay `strokeColor` |
| Detected pitch curves | 917–1000 | 2 | solid | 4-colour palette cycled by index |
| Secondary param curves | 1008–1040 | 2 | solid | per-param colour |
| Original curve | 1042–1060 | 1.8 | **dashed** `getFixedDashPattern(6,6)` | |
| Edited curve | 1063–1085 | 2.6 | solid | |
| Selection-highlighted curve | 1087–1120 | 3.6 | solid | **clipped** to selection rect |
| Clipboard preview | 1122–1170 | **2** | **dashed** `getFixedDashPattern(4,4)` | **clipped** to selection rect; **does not use `drawCurveTimed`** — it has its own loop anchored at `selStartSec` with raw inter-frame spacing |

**R2 — Stroke semantics that the GL port must reproduce.**
- `lineJoin` / `lineCap` are **never set anywhere** in the piano roll or `renderKernel` → Canvas2D defaults: **miter join, butt cap**, miter limit 10.
- Line widths are **non-integer CSS px** (1.8 / 2 / 2.6 / 3.2 / 3.6) and are **not** run through `strokePx`/`snapPx` — unlike the playhead. They are drawn at exact fractional widths with Canvas2D's own AA.
- Dash patterns are **quantized to whole device pixels at call time** by `getFixedDashPattern` (reads `window.devicePixelRatio`), so the GL path must call the same function.
- `globalAlpha = 1` at `render.ts:948` is a no-op (leftover); colours already carry their own alpha.
- The two `ctx.clip()` sites are axis-aligned rects → `gl.scissor` candidates.

**R3 — Curve scale is large, and the stride is 1.**
`usePianoRollData.ts:289` sets `stride = 1` with `framePeriodMs = 5`, i.e. **200 points per second**. A 12.4 s viewport can therefore contain ~2,500 points per curve. Measured with a 3,000-point injected curve: **1,648 `lineTo` calls per frame** for that one curve.

**R4 — The harness can render curves, but only with data injected.**
The mock backend produces **no** pitch data (`clipPitchCurves` starts empty) and always reports `is_playing: false`. Curves were reached by dispatching the real reducer:

```js
store.dispatch({ type: "session/setClipPitchData", payload: {
  clipId, curveStartSec: 0, midiCurve, framePeriodMs: 5 } });
```

This requires `editParam === "pitch"` **and** `rootTrack.composeEnabled` **and** the clip being in `groupTrackIds` and unmuted (`PianoRollPanel.tsx:3125–3137`). The mock sets `compose_enabled: true`, so injection alone is sufficient. **Verified: 22,328 ink pixels on the curve canvas and a visible teal sine curve** (see evidence section).

**R5 — ⚠️ Phase 3 cannot demonstrate a frame-time win in this environment.**
This is the single most important finding, and it contradicts the intuitive justification for the curve work.

| Measurement | With curve (3,000 pts) | Control (no curve) |
|---|---|---|
| Frame interval p50 | 16.7 ms | 16.7 ms |
| Frame interval p95 | **31.1 ms** | **32.0 ms** |
| Frames > 20 ms (of 120) | 16 | 20 |
| Long tasks (>50 ms) during 150 scroll frames | **0** | 0 |
| Time inside Canvas2D calls | **0.211 ms/frame** | — |
| `lineTo` calls/frame | 1,648 | 0 |

The control group — same scroll, **no curve at all** — shows the *same* p95. So the frame-interval tail is vsync/compositor jitter, not curve work, and `PerformanceObserver` reports **zero** long tasks while drawing a 3,000-point curve. The honest metric Phase 3 can move is **JS→native call count** (1,648 `lineTo` + 1 `stroke` per curve per frame, eliminated by batching into one buffer upload), not frame time.

Phase 3's real justifications are therefore: (a) eliminating per-point JS→native calls, which is the part that scales with curve count and zoom-out; and (b) architectural completion — today the kernel owns everything except curves, which is a hybrid state. **The plan must not claim a frame-rate improvement it cannot measure.**

**R5b — Polyline geometry cost at realistic scale (measured after Task 1).**
For a 1,650-point curve (R3's realistic visible count), `buildPolylineVertices` produces **14,838 vertices / 237 KB / ~1 ms** per curve, and the same for a deliberately jumpy curve. That is 6 vertices per segment plus 3 per miter join. Four curves on screen therefore means ~1 MB of vertex data and ~4 ms of JS per frame — non-trivial, and the dominant cost of Task 4, so Task 4 must measure buffer upload size and consider a straight-join fast path. Note the join triangles are what make it 9 vertices/point rather than 6: for a **nearly straight** join the apex coincides with the segment endpoint, so the triangle could be skipped without changing the rendered result. That optimisation is deliberately **not** taken in Task 1 (fidelity first, and the epsilon needs pixel evidence), but Task 4 should measure whether it is needed.

**R7 — ⛔ Curve GL was stopped: the measured head-to-head favours Canvas2D.**
After Tasks 1–3 were built and verified, a direct comparison of the two paths on the same 1,650-point polyline (same machine, same browser, 30 runs each, measured in-page) came out against the migration:

| Path | p50 | p95 |
|---|---|---|
| Canvas2D `stroke` (beginPath + 1649 `lineTo` + stroke) | **0.0 ms** | **0.1 ms** |
| Our GL geometry build (`buildPolylineVertices`) | **0.2 ms** | **1.2 ms** |

The GL path is **6–12× slower** at the step that replaced `stroke`, before adding the buffer upload or the draw call. Canvas2D's `stroke` is a highly optimised native path; computing miter triangles point-by-point in JS does not beat it. Combined with R5 (no frame-time headroom to reclaim: a no-curve control group shows the same p95), Phase 3's curve work had **no measurable benefit and a measured regression**.

**Decision (user-approved): stop curve GL here.** Tasks 1–3 are kept as finished, tested infrastructure (pure geometry, GL program, projections — all reusable if a future case justifies them, e.g. a platform where Canvas2D is slower), but the curve layers are **not** wired to GL and no sub-flag is added. Work moves to Task 6 (gesture extraction), which has independent value.

This is recorded rather than quietly dropped because the plan's own reconnaissance (R5) had already warned there was no frame-time win, and the honest conclusion is that the spec's Phase 3 curve row was justified by intuition ("GL is faster") rather than measurement. **Anyone resuming curve GL should re-run the R7 comparison first, and should test at 8+ curves and high zoom-out, where the ratio may differ** — that is the one regime where the JS-per-vertex cost could amortise against Canvas2D's per-point fill rate.

**R6 — The gesture surface is a 3,875-line hook, and a safety net already exists.**
`usePianoRollInteractions.ts` holds ~23 event entry points. `renderProjection.test.ts` already exists specifically as the "P3 pre-work snapshot" the spec asked for, comparing the legacy `timeToPixel` formula against `secToViewportPx` over random parameters — the x-projection conversion is therefore already guarded.

---

## Verification method

Same discipline as Phase 2, with one addition forced by R4.

1. **Curve fixtures.** Every curve comparison must first dispatch `setClipPitchData` with a deterministic `midiCurve` (a fixed sine series), then wait for the redraw. Without this the curve canvas is empty and any "0 pixels differ" result is vacuous — the trap R4 documents.
2. **Assert the fixture took.** A comparison that does not first prove the curve canvas has ink is not evidence. Capture the ink count (expect ~20k for the standard fixture) and record it alongside the diff.
3. **Determinism control.** Two captures of the same mode must differ by 0 px.
4. **Explained-or-fixed.** Any diff is fixed or explained by construction.
5. **Call-count metric.** Because frame time cannot show the win (R5), the curve task's exit metric is `lineTo`/`stroke` counts per frame, measured by injecting a hook over `CanvasRenderingContext2D.prototype` before document load (`PROBE_INIT`), plus GL-side buffer upload counts.

The existing helper lives at `/tmp/pd.py` (decode + region-bucketed diff); it is throwaway and not committed.

---

## File structure

```
renderKernel/
  gl/
    polylineProgram.ts         NEW — polyline shader/program (miter joins, AA edges)
    polylineGeometry.ts        NEW — pure: points → triangle vertices (+ dash, clip)
    polylineGeometry.test.ts   NEW
pianoRoll/kernel/
  scene/
    curvePoints.ts             NEW — pure: values[] + projection → viewport points
    curvePoints.test.ts        NEW
  host/pianoRollKernelHost.ts  MODIFY — curve GL layer behind a sub-flag
pianoRoll/
  render.ts                    MODIFY — skipCurves flag; keep Canvas2D path as fallback
  interactions/                NEW — pure gesture modules extracted from the hook
    gestureHitTest.ts
    gestureHitTest.test.ts
    dragArithmetic.ts
    dragArithmetic.test.ts
```

---

## Task 1: Pure polyline geometry with miter joins and device-pixel AA

**Why first:** everything else depends on it, it is pure arithmetic, and it is the piece where fidelity is won or lost. `lineJoin` defaulting to **miter** (R2) means segment quads alone will show notches at every vertex; with 0.75 px between samples those notches are dense, not occasional.

**Files:**
- Create: `frontend/src/components/layout/renderKernel/gl/polylineGeometry.ts`
- Test: `frontend/src/components/layout/renderKernel/gl/polylineGeometry.test.ts`

**Design:**
- Input: a point list in **content coordinates** (CSS px), a line width, a miter limit, and an **AA padding** (`aaPadPx`, default 0.5).
- **Why the padding is not optional:** the fragment shader computes coverage from the signed distance, so the geometry must extend *past* the geometric edge or there is no pixel left to carry the falloff and the line renders thinner than Canvas2D (whose stroke's AA footprint reaches roughly half a pixel beyond the edge). Geometry therefore spans `±(lineWidth/2 + aaPadPx)` while the shader's coverage threshold stays at `±lineWidth/2`.
- Output: a `Float32Array` of triangles, each vertex carrying `(x, y, u, v)` where `(u,v)` encodes **distance from the centreline in CSS px** as `(across, along)`. The fragment shader derives AA coverage from `across`, and dash phase from `along` — this is what makes dashes and AA shader-side rather than geometry-side.
- **Why encode distance in attributes rather than expanding to exact geometry:** a segment's normal changes at each vertex, so expanding outward by `lineWidth/2` cannot express both a 1.8 px wide line and sub-pixel AA at the same time. Carrying signed distance lets the shader compute coverage analytically, which is the only way to match Canvas2D's AA on fractional widths.

- [ ] **Step 1: Write the failing test**

```ts
/**
 * 折线几何构建（纯函数）单测。
 *
 * 【本测试守护什么】
 * 1. 每条线段产出 2 个三角形（6 个顶点），顶点携带**到中心线的有符号距离**，
 *    而不是被展开到精确外沿——因为后续要靠片元着色器做亚像素抗锯齿；
 * 2. 拐角用 **miter 连接**（Canvas2D 默认 lineJoin="miter"，
 *    见 render.ts 从未设置 lineJoin）；miter 长度超限时必须退化为 bevel，
 *    否则尖角会甩出长刺（实测检测曲线的八度跳变就是这种尖角）；
 * 3. 沿线的累积弧长写进 `along`，供虚线相位使用；
 * 4. 少于 2 个点、非有限坐标、零/负线宽都必须返回空而不是抛错或产出 NaN
 *    （NaN 会让整个顶点缓冲失效、整层消失）。
 */
import { describe, expect, it } from "vitest";

import { buildPolylineVertices, POLYLINE_FLOATS_PER_VERTEX } from "./polylineGeometry";

describe("buildPolylineVertices", () => {
    it("单段产出 2 个三角形（6 顶点）", () => {
        const out = buildPolylineVertices({
            points: [{ x: 0, y: 0 }, { x: 10, y: 0 }],
            lineWidth: 2,
            miterLimit: 10,
        });
        expect(out.length).toBe(6 * POLYLINE_FLOATS_PER_VERTEX);
    });

    it("顶点携带到中心线的有符号距离（几何外沿 = 半线宽 + AA 余量）", () => {
        const collectAcross = (a: Float32Array) => {
            const out: number[] = [];
            for (let i = 0; i < a.length; i += POLYLINE_FLOATS_PER_VERTEX) {
                out.push(a[i + 3]); // (x, y, along, across)，见实现处布局说明
            }
            return out;
        };
        // aaPadPx = 0 时，几何恰好到半线宽（纯几何语义）
        const bare = buildPolylineVertices({
            points: [{ x: 0, y: 0 }, { x: 10, y: 0 }],
            lineWidth: 4,
            miterLimit: 10,
            aaPadPx: 0,
        });
        const bareAcross = collectAcross(bare);
        expect(Math.min(...bareAcross)).toBeCloseTo(-2, 9);
        expect(Math.max(...bareAcross)).toBeCloseTo(2, 9);

        // 默认（有 AA 余量）时，几何必须**外扩**到 半线宽 + pad，
        // 否则最外圈像素没有空间承载覆盖率衰减，线会比 Canvas2D 更细。
        const padded = buildPolylineVertices({
            points: [{ x: 0, y: 0 }, { x: 10, y: 0 }],
            lineWidth: 4,
            miterLimit: 10,
        });
        const paddedAcross = collectAcross(padded);
        expect(Math.max(...paddedAcross)).toBeCloseTo(2.5, 9); // 2 + 0.5
        expect(Math.min(...paddedAcross)).toBeCloseTo(-2.5, 9);

        // 注意：`across` 记录的是**几何**距离（含 pad）。
        // 着色器的覆盖率阈值取 lineWidth/2，而不是 max|across|——
        // 因此 `u_halfWidth` 必须作为单独 uniform 传入，不能从几何反推。
    });

    it("沿线的累积弧长写入 along（供虚线相位使用）", () => {
        const out = buildPolylineVertices({
            points: [{ x: 0, y: 0 }, { x: 3, y: 4 }, { x: 3, y: 14 }],
            lineWidth: 2,
            miterLimit: 10,
        });
        const along = [];
        for (let i = 0; i < out.length; i += POLYLINE_FLOATS_PER_VERTEX) {
            along.push(out[i + 2]);
        }
        // 第一段长 5（3-4-5），第二段长 10
        expect(Math.min(...along)).toBeCloseTo(0, 9);
        expect(Math.max(...along)).toBeCloseTo(15, 9);
    });

    it("miter 未超限时走 miter，超限时退化为 bevel", () => {
        // 【用例必须按**转角**设计，不能凭坐标直觉】
        // Canvas2D 的 miterLimit 语义是 ratio = miter长度 / 半线宽 = 1 / cos(θ/2)，
        // 其中 θ 是两段方向的转角：
        //   θ=90°  -> ratio 1.414（limit=2 不超限，走 miter）
        //   θ=150° -> ratio 3.864（limit=2 超限，退化 bevel）
        // 曾经写过一个「浅 V」用例 (0,0)-(100,0.5)-(200,0)，它的转角只有 0.57°、
        // ratio≈1.000，**永远走不到 bevel 分支**，等于没测。
        const rightAngle = [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 10, y: 10 },
        ];
        // 150° 转角：从 +x 方向转到 150° 方向
        const sharpAngle = [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 10 + Math.cos((150 * Math.PI) / 180) * 10, y: Math.sin((150 * Math.PI) / 180) * 10 },
        ];
        // 用 aaPadPx: 0 隔离出**纯几何**行为，避免 AA 余量干扰 miter 断言。
        const maxAcross = (a: Float32Array) => {
            let m = 0;
            for (let i = 0; i < a.length; i += POLYLINE_FLOATS_PER_VERTEX) {
                m = Math.max(m, Math.abs(a[i + 3]));
            }
            return m;
        };
        // 90° + limit 2（ratio 1.41 未超限）：走 miter，顶点被推到半线宽之外
        const mitered = buildPolylineVertices({
            points: rightAngle,
            lineWidth: 10,
            miterLimit: 2,
            aaPadPx: 0,
        });
        expect(maxAcross(mitered)).toBeGreaterThan(5.5); // > 半线宽 5
        // 150° + limit 2（ratio 3.864 超限）：退化 bevel，顶点**不超过**半线宽
        const beveled = buildPolylineVertices({
            points: sharpAngle,
            lineWidth: 10,
            miterLimit: 2,
            aaPadPx: 0,
        });
        expect(maxAcross(beveled)).toBeLessThanOrEqual(5 + 1e-6);
        // 同样的 150° 若放宽 limit（10 > 3.864），则应走 miter 并显著超出半线宽
        const looseLimit = buildPolylineVertices({
            points: sharpAngle,
            lineWidth: 10,
            miterLimit: 10,
            aaPadPx: 0,
        });
        expect(maxAcross(looseLimit)).toBeGreaterThan(maxAcross(beveled) + 1);
    });

    it("少于 2 个点返回空数组", () => {
        expect(buildPolylineVertices({ points: [], lineWidth: 2, miterLimit: 10 }).length).toBe(0);
        expect(
            buildPolylineVertices({ points: [{ x: 1, y: 1 }], lineWidth: 2, miterLimit: 10 }).length,
        ).toBe(0);
    });

    it("非有限坐标 / 非法线宽返回空数组（防 NaN 污染顶点缓冲）", () => {
        const bad = [
            { points: [{ x: 0, y: 0 }, { x: Number.NaN, y: 1 }], lineWidth: 2, miterLimit: 10 },
            { points: [{ x: 0, y: 0 }, { x: 1, y: Number.POSITIVE_INFINITY }], lineWidth: 2, miterLimit: 10 },
            { points: [{ x: 0, y: 0 }, { x: 1, y: 1 }], lineWidth: 0, miterLimit: 10 },
            { points: [{ x: 0, y: 0 }, { x: 1, y: 1 }], lineWidth: -3, miterLimit: 10 },
        ];
        for (const args of bad) {
            expect(buildPolylineVertices(args).length).toBe(0);
        }
    });

    it("重合点不产生 NaN（相邻点相同时跳过该段）", () => {
        const out = buildPolylineVertices({
            points: [{ x: 5, y: 5 }, { x: 5, y: 5 }, { x: 15, y: 5 }],
            lineWidth: 2,
            miterLimit: 10,
        });
        for (let i = 0; i < out.length; i += 1) {
            expect(Number.isFinite(out[i])).toBe(true);
        }
    });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run polylineGeometry`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `polylineGeometry.ts`. Vertex layout must be documented in a header comment (repo convention):

```
每个顶点 4 个 float: [x, y, along, across]
  x, y    —— 内容坐标（CSS px）
  along   —— 沿折线的累积弧长（CSS px），供虚线相位
  across  —— 到中心线的有符号距离（CSS px），供片元做亚像素抗锯齿
```

Algorithm:
1. Validate: `points.length >= 2`, all finite, `lineWidth > 0`, `miterLimit > 0`; else return empty.
2. Build a filtered point list dropping consecutive duplicates (guard the NaN from zero-length normals).
3. For each segment compute the unit direction and left normal.
4. For each **interior vertex**, compute the miter vector: `miter = (n1 + n2) / |n1 + n2|²`, scaled by `lineWidth/2`; if `|miter| > miterLimit * lineWidth/2`, fall back to **bevel** (emit the two segment quads separately, no miter vertex).
5. Emit two triangles per segment; `along` is the cumulative arc length at each endpoint; `across` is `+w/2` on the left vertex and `-w/2` on the right.
6. For mitered interior vertices, additionally emit the join triangle(s) with the **miter-extended** position so the shader's `across` stays `±w/2` while the vertex is pushed outward.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run polylineGeometry`
Expected: PASS (7 tests).

- [ ] **Step 5: Verify no regression**

Run: `cd frontend && npx tsc -b --noEmit && npx vitest run`
Expected: typecheck clean; only the 2 known `keybindingMatch` failures.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/layout/renderKernel/gl/polylineGeometry.ts frontend/src/components/layout/renderKernel/gl/polylineGeometry.test.ts
git commit -m "feat(renderKernel): pure polyline geometry with miter joins"
```

---

## Phase 3 remaining tasks (expand in place when each starts)

Same convention Phase 2 used, and for the same reason: each depends on the previous task's concrete shape.

### Task 2: Polyline GL program (AA edges, dash phase, scissor clip)

- `polylineProgram.ts`: vertex shader places `(x, y)` via `u_viewOrigin` + `u_resolution`; fragment shader computes coverage from `|across|` against `u_halfWidth` (analytic AA, matching Canvas2D's fractional widths) and applies the dash pattern from `along`.
- **`u_halfWidth` is a separate uniform, not derived from the geometry.** The geometry's `across` values extend to `lineWidth/2 + aaPadPx` (Task 1), so the shader cannot recover the coverage threshold from `max|across|`. Passing it explicitly also keeps the AA pad tunable without rebuilding geometry.
- Dash pattern is a uniform pair `(dash, gap)`; a negative `dash` disables it. **The values must come from `getFixedDashPattern`** (R2: dpr-quantized at call time) — do not hardcode.
- **Dash phase must run from the first *visible* point, not the curve's absolute start.** `drawCurveTimed` skips samples before `viewportStartSec` with `started = false; continue`, and only the first surviving point issues `moveTo` (`render.ts:169–193`); the whole curve is therefore a **single subpath anchored at the first visible sample**, and `lineDashOffset` is never set (verified: zero occurrences). Canvas2D consequently restarts the dash pattern as you scroll, so the dashes visibly shift rather than staying pinned to content. Task 1's `along` already accumulates from the first point in the array it is given, so the GL path matches **only if the caller passes the visible slice** — passing the full curve with an absolute arc length would make dashes slide during scroll and is the most likely way to get this wrong.
- Clip via `gl.scissor` for the two axis-aligned selection-rect clips.
- **Exit check:** a single straight line rendered at widths 1.8 / 2 / 2.6 / 3.2 / 3.6 matches Canvas2D within tolerance, at dpr 2.

### Task 3: Curve point projection (pure)

- `curvePoints.ts`: two entry points, because two loops exist. `projectCurvePoints` reproduces `drawCurveTimed`'s loop exactly (**including** the `pitch + 0.5` mapping, the `break` past the right edge, and the `started = false; continue` prefix skip). `projectClipboardPreviewPoints` reproduces the clipboard branch's own loop (`render.ts:1150–1170`): anchored at `selStartSec` with **raw** inter-frame spacing (`selStartSec + i * cbFp / 1000`, no `startFrame`/`stride`), terminating at `selEndSec`.
- **Return only the visible slice** (the function's contract is "points that would be drawn"), because Task 2's dash phase depends on arc length starting at the first visible point. Do not return the full curve plus an offset.
- **Exit check:** equivalence test against a re-implementation of `drawCurveTimed`'s loop over many parameter combinations (the pattern that caught 12 defects in Phase 2).

### Task 4: Curve GL layer in the host, behind a sub-flag — ⛔ **不属于本轮范围（见 R7）**

> 停止原因：实测 GL 几何构建比 Canvas2D `stroke` 慢 6–12 倍，且无帧时间可回收。
> Task 1–3 的产出保留为已完成的基础设施，未接线。
> 若将来恢复：先复跑 R7 的对照，并覆盖 8+ 条曲线与高缩放出（那里比例可能不同）。
> 以下原始设计保留供参考。

- New sub-flag `hifishifter.pianoRollKernel.curveGl`, default off (same reasoning as `.gl`).
- Wire all 7 curve variants. Per-frame rebuild is expected and correct: scrolling/zooming changes every x, so unlike the grid there is no zero-rebuild path. Measure and record the upload cost.
- Keep the Canvas2D curve path intact behind `skipCurves`.
- **Exit check:** `lineTo`/`stroke` counts per frame drop to 0; pixel comparison within tolerance **with the fixture asserted** (R4).

### Task 5: Pixel comparison + verification record — ⛔ **不属于本轮范围（依赖 Task 4）**

- Fixture-injected comparison at 2 viewports and 2 dprs, across all 7 variants (toggle each: detected, original, edited, secondary, reference, selection-highlight, clipboard).
- Record the call-count metric and explicitly record that **frame time is not the metric** (R5), so a future reader does not mistake the absence of a frame-time win for a failed migration.
- Update the spec's phase table.

### Task 6: Extract gesture logic into pure modules

- Pull hit-testing (`gestureHitTest.ts`) and drag arithmetic (`dragArithmetic.ts`) out of `usePianoRollInteractions.ts`, following the `timeline/kernel/interaction/*` pattern.
- Enumerate all gesture entry points as a checklist (spec risk row: "23 event entry points, single 3875-line hook") and browser self-test each after extraction.
- **Exit check:** hook shrinks measurably; each extracted function has unit tests; every gesture re-verified in the browser.

---

## Self-Review

**Spec coverage:** the spec's Phase 3 row lists "Polyline triangle-strip curve rendering (all curve variants); geometric hit testing + gesture state machine migration", exit criteria "curve fidelity comparison passes; every gesture regression-checked in the browser". All 7 curve variants are enumerated in R1 and covered by Tasks 3/4/5; hit testing → Task 6; gesture migration → Task 6; fidelity comparison → Task 5; browser gesture re-check → Task 6's exit check.

**Deviation from the spec, recorded deliberately:** the spec says "triangle-strip". This plan uses **triangles with distance-encoded attributes** instead, because a triangle strip cannot express fractional line widths with AA, miter joins, or dash phase. Triangle strips are also a poor fit for polylines that need per-vertex normals. This is a mechanism change in service of the spec's actual requirement ("curve fidelity comparison passes"), and it is recorded here rather than silently substituted.

**Placeholder scan:** Task 1 carries a complete failing test, the vertex layout, the algorithm, and exact commands. Tasks 2–6 are declared as phase-entry expansions with explicit deliverables and exit checks — the same granularity Phase 2 used, which proved workable.

**Type consistency:** `POLYLINE_FLOATS_PER_VERTEX` (Task 1) is consumed by Task 2's program and Task 3's point builder. `buildPolylineVertices` (Task 1) is the sole geometry entry point for Task 4. The sub-flag name `hifishifter.pianoRollKernel.curveGl` (Task 4) follows the `.gl` naming from Phase 2 and must be added to `featureFlag.ts` alongside it.

**Risk ranking (highest first):**
1. **Dash fidelity** — two independent traps: `getFixedDashPattern` is dpr-quantized and re-evaluated per call (the shader must receive the same numbers), and the phase must accumulate by arc length **from the first visible point** (Canvas2D restarts the pattern per subpath, and the curve is one subpath starting at the first visible sample). Getting the second wrong makes dashes slide during scroll — visible, but easy to misread as a scroll bug rather than a dash bug. Tasks 2 and 3.
2. **Miter joins on sharp angles** — detected-pitch curves contain octave jumps producing near-180° turns; an unclamped miter produces long spikes. Task 1's `miterLimit` test guards this.
3. **Fractional line widths** — 1.8/2.6/3.2/3.6 are not device-pixel aligned and Canvas2D AA's them; matching that is what the distance-encoded design exists for. Task 2.
4. **Gesture regression** — 23 entry points in one hook. Task 6 is last and explicitly browser-re-verified per gesture.

**Honesty constraint carried into the plan:** because R5 shows no measurable frame-time win, Task 5 must record the JS→native-call metric as the benefit and state plainly that frame time was unchanged. Claiming otherwise would be unsupported by the measurements.
