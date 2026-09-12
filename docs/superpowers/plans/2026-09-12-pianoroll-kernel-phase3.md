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

**R7 — ✅ Curve GL is worth doing. My first "stop" call was wrong; so was my second explanation of *why*.**
Two corrections, both from the user pushing back, and the final numbers are platform-independent.

Correction 1 — **the first measurement was invalid.** I measured in headless Chromium and concluded "Canvas2D is 6–12× faster". Two defects: `stroke()` is **deferred**, so wrapping it in `performance.now()` times queueing rather than work (all my "0.0ms"/"0.211ms" Canvas2D numbers were artefacts), and the scene never actually reached the app's data path, because the mock's clips are only 4–12 s long so the "3-minute curve" was never drawn.

Correction 2 — **the cause is not the rasterisation backend.** I then blamed WKWebView/WebView2 software rasterisation. The user pointed out the jank happens on Chromium too. Measured head-to-head, with a forced flush and at the real canvas size (3728×1646 physical, 4px stroke):

| Points visible | Canvas2D | GL full path | Speed-up |
|---|---|---|---|
| 12,000 (1 min) | 9.2 ms | 1.7 ms | 5.4× |
| 36,000 (3 min) | **71.7 ms** | **4.6 ms** | **15.6×** |
| 200,000 (app's own clamp) | 214.2 ms | 28.0 ms | 7.6× |

And the same table under `--disable-gpu` gives **71.8 ms / 212.8 ms** — i.e. **identical**. The cost is Canvas2D's stroke itself, not the GPU backend, which is why it reproduces on Chromium, WebView2 and WKWebView alike. 36,000 points at 71.7 ms is ~14 fps, which matches the reported "1 minute and up it janks".

**The mechanism is overdraw.** At min zoom (`MIN_PX_PER_SEC = 4`) a 1864 px viewport covers **466 s**, so the whole curve is on screen and its samples pile up in the same pixel columns: 36,000 points over the curve's 720 CSS px is **25 points per column**, 200,000 is **139 per column**. Each sample is stroked independently. Decimating to ~2 points/column removes most of that (71.7 → 44.2 ms) but still strokes each surviving point; triangle rasterisation removes the redundancy rather than reducing it, hence GL's much larger win, and the gap widening with sample count.

**Decision (user-confirmed): resume curve GL (Tasks 4–5).** No platform gate is needed — the win holds on every backend measured. The earlier platform-gating idea is dropped as unnecessary.

**Lessons recorded:** never time a deferred Canvas2D API without forcing a flush; verify the fixture actually reaches the render path (a 3-minute curve in a mock whose clips are 12 s long proves nothing); and when a user reports jank that a synthetic benchmark does not reproduce, the benchmark is wrong until proven otherwise — not the report.

**R8 — 🔴 The kernel is OFF in the build the user is actually running.**
`backend/src-tauri/tauri.conf.json` runs `node ../scripts/tauri-before-dev.mjs` as its `beforeDevCommand`, and that script's `TAURI_UI_MODE` **defaults to `"build"`** (not `"dev"`). So a plain `tauri dev` — the normal way to run the app on Windows — does `npm run build` and serves the **production bundle**, where `import.meta.env.DEV === false`.

Every kernel flag is off in that bundle:

| Flag | Default in a production build |
|---|---|
| `isTimelineKernelEnabled()` | `import.meta.env.DEV` → **false** |
| `isPianoRollKernelEnabled()` | **false** (off even in dev, by design) |
| `isPianoRollGlSceneEnabled()` | **false** |
| `isPianoRollCurveGlEnabled()` | **false** |

**Consequence:** the jank the user reports on Windows is the **old** implementation. Phases 1–2 never ran there, and neither would Phase 3's curve work. This also explains why the user's reports and my Chromium measurements kept disagreeing: we were measuring two different renderers, and the reports were about the un-migrated one.

**This reframes the work.** The highest-value next step is **not** more curve-GL code — it is making the kernel actually reachable in the app the user runs, then measuring there. Options, in order of preference:
1. **Pass the mode through explicitly** so `tauri dev` runs the dev server (`TAURI_UI_MODE=dev`), or make `beforeDevCommand` use dev mode while a separate build command serves the bundle. This is what "dev" should have meant.
2. **Invert the piano-roll defaults to follow `import.meta.env.DEV`**, matching the timeline kernel, so a dev run exercises the new path by default.
3. Keep defaults off but surface a **visible in-app toggle** (the PERF overlay already has one for the timeline flag) so the path can be exercised on Windows without devtools.

Until one of these lands, any "is the new kernel faster?" measurement on the user's machine is measuring the wrong renderer. Recorded here rather than in chat because it invalidates how the whole phase was being verified.

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

### Task 4: Curve GL layer in the host, behind a sub-flag

> 已恢复（见 R7 修正后的测量）：曲线 GL 在**所有**后端都快 5–16 倍（GPU 与软件
> 栅格化的 Canvas2D 成本相同），无需平台开关。

- New sub-flag `hifishifter.pianoRollKernel.curveGl`, default off (same reasoning as `.gl`).
- Wire all 7 curve variants. Per-frame rebuild is expected and correct: scrolling/zooming changes every x, so unlike the grid there is no zero-rebuild path. Measure and record the upload cost.
- Keep the Canvas2D curve path intact behind `skipCurves`.
- **Exit check:** `lineTo`/`stroke` counts per frame drop to 0; pixel comparison within tolerance **with the fixture asserted** (R4).

### Task 5: Pixel comparison + verification record

- Fixture-injected comparison at 2 viewports and 2 dprs, across all 7 variants (toggle each: detected, original, edited, secondary, reference, selection-highlight, clipboard).
- Record the call-count metric and explicitly record that **frame time is not the metric** (R5), so a future reader does not mistake the absence of a frame-time win for a failed migration.
- Update the spec's phase table.
### Task 6: Extract gesture logic into pure modules

- Pull hit-testing (`gestureHitTest.ts`) and drag arithmetic (`dragArithmetic.ts`) out of `usePianoRollInteractions.ts`, following the `timeline/kernel/interaction/*` pattern.
- Enumerate all gesture entry points as a checklist (spec risk row: "23 event entry points, single 3875-line hook") and browser self-test each after extraction.
- **Exit check:** hook shrinks measurably; each extracted function has unit tests; every gesture re-verified in the browser.

#### 进度：两片已完成（`59277ad4`、`bd6ffea5`），任务仍未完成

**第一片** 落地 `kernel/gestureHitTest.ts`（19 项单测）；**第二片** 落地
`kernel/dragArithmetic.ts`（13 项单测）并接线全部命中判定。

| 抽出物 | 单测 | 已接线 | 浏览器验证 |
|---|---|---|---|
| `hitTestSelectionEdge` | ✅ | ✅ 两处（pointermove 光标、pointerdown 拉伸） | ✅ 15 点光标扫描逐位一致 |
| `curveValueAtPointerFrame` | ✅ | ✅ `getCurveValueAtPointerFrame` | ✅ 多手势截图字节相同 |
| `isPointerNearCurve` | ✅ | ✅ `getCurveValueNearPointer` | ✅ 多手势截图字节相同 |
| `selectionFrameRange` / `selectionIndexRange` | ✅ | ✅ 三处（morph 快照、拉伸 oldRange、buildDense nextRange） | ✅ 拉伸前后截图字节相同 |
| `hitTestSelectionBody` | ✅ | ✅ `isPointerNearDraggableSelection` | ✅ 拖动选区主体截图字节相同 |
| `frameToIndex` | ✅ | ⬜ **刻意不接线**（6 处内联全在逐采样点的循环体内，见下） | — |

**退出标准对照：**

| 标准 | 结果 |
|---|---|
| 每个抽出函数有单测 | ✅ 32 项（含 4 处我自己写错的用例前提，已修正） |
| 每个手势在浏览器复核 | ⚠️ **部分**：选区建立 / 边缘拉伸 / 拖动选区主体 / 画线 / 滚轮平移已验证；23 个入口未逐条枚举 |
| hook 可测量地缩小 | ❌ **未达成**：3,876 → 3,887 行（净 **+11**，见下） |

**为什么行数不是有效指标（修正上一片的判断）**：上一片记的是"要等其余三处接线
后才会下降"。接线完成后**行数依然没降**，原因是本工程硬性要求每个文件带完整文件头
注释与关键函数注释——两个新模块 378 行里相当一部分是注释，而抽走的算术本身只有
十几行。真正该看的是**重复被消除了多少**：

| 指标 | 基线 | 现在 |
|---|---|---|
| 内联「beat → 帧」换算 | **6 处** | **0 处** |
| 内联「帧 → 下标」换算 | 7 处 | 6 处（**刻意不接线**，见下） |

所以 Task 6 的退出标准里「hook 可测量地缩小」这条**应改为「重复处数下降」**才
可达成；按行数衡量在这个代码库里永远达不成（注释占比高）。这个判断错误已在此更正。

**为什么剩下 6 处「帧 → 下标」刻意不接线**：它们全部位于**逐采样点的 `for` 循环
体内**（`smoothed` / `packed.dense` / `overallLen` / `built.dense` 的展开写入）。
`frameToIndex` 每次调用要做 3 次 `Number.isFinite` 检查 + 一次 `Math.floor`（stride
归一），在下标计算本就只有一次减法一次除法的热点里，这层包装的**开销超过它省下的
重复**（这类循环在 200 点/秒 × 数秒的选区上会跑上万次）。留在内联是性能取舍，不是
遗漏；`frameToIndex` 仍服务于非热点的调用方。

**行为等价性是怎么验的（不是只跑单测）**：用 `git stash` 在基线重跑同一脚本，
四个场景的整页截图**字节完全相同**：
1. 选区建立 + Alt 拉伸右缘（拉伸确实生效：before/after 差异 0.1174%，maxdelta 50）
2. 多手势组合（选区 → Alt 拉伸 → 切画线工具拖一笔 → 滚轮水平平移）
3. 拖动选区主体（选区内两个不同 y 各拖一次，覆盖命中/未命中曲线两种分支）
4. 第一片的 15 点光标扫描（序列逐位相同）

这比"单测通过"强：单测只能证明纯函数自身正确，证明不了 hook 接线后行为不变。
**注意**：截图字节相同也说明这些手势路径**根本没被改动行为**——这是重构的目标，
但也意味着它证明不了"新的纯函数在真实手势下被走到了"；后者由单测 + 接线点的
代码审查共同保证。

**同时钉住一个既有不一致**：`stride` 归一化在 hook 里是 `Math.max(1, stride)`
（不取整），而渲染路径（`render.ts` / `curvePoints` / `selectionEditData`）统一用
`Math.max(1, Math.floor(stride))`。当前 `stride` 恒为整数故无实际差异；新模块取
渲染路径的规则，并在注释与单测里显式记录该选择。

**剩余工作清单**：
1. **枚举全部 23 个事件入口**，逐条用 stash 对照法补浏览器验证。已验 5 条
   （选区建立 / 边缘拉伸 / 拖动选区主体 / 画线 / 滚轮平移），其余未枚举——
   `onRulerMouseDown`、`onScrollerAuxClick`、`onScrollerScroll`、
   `onScrollerContextMenu`、`onScrollerKeyDown`、`onScrollerWheelNative`、
   `onCanvasPointerLeave`、`onCanvasPointerDown` 的其余分支等。
2. `frameToIndex` 的接线：若要接，需要先把它改成**接受已归一化的 stride**的
   轻量形式（去掉每次调用的 `Number.isFinite` 与 `Math.floor`），否则在逐采样点
   循环里不划算。当前选择是不接。
3. **本任务不做**（建议单列）：`dragArithmetic` 还应覆盖曲线拖动的手势增量换算
   （`secDelta` / 像素增量 → 帧增量的那段，hook 里在 `rawFrameDelta` 等处），
   本次只做了选区相关的坐标换算。

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

---

## Task 4 · 完成记录（曲线 GL 接线 + 渲染层抽稀）

**状态：** 已完成并提交（`45abcb7c` 修复接线缺陷、`7a88e9c7` 抽稀）。

### 实现过程中发现并修复的两个真实缺陷

两者都只在浏览器端到端验证中暴露，单测无法覆盖：

1. **帧内顺序错误**（`pianoRollKernelHost.draw`）。`drawGlCurves()` 读
   `data().curves`，而该字段是面板在 `onFrame` → `applyScrollLayers` →
   `drawRef.current()` 里**就地写入**的。原实现把 `onFrame` 排在曲线 GL **之后**，
   于是曲线层永远读到上一帧的图层列表——首帧更是空数组。现象是"曲线数据已到但
   屏幕上不出现，滚动一下才出来"，正是本阶段此前遗留的未解之谜。
2. **标脏只喂了面板自己的 rAF**（`PianoRollPanel.invalidate`）。曲线搬到 GL 后画面
   由两个循环驱动（面板 rAF 画 Canvas2D、宿主 rAF 画 GL），只调度前者会让 GL 层停在
   旧内容上。现在内核模式统一转交宿主，由宿主的帧提交回调 `onFrame` 同帧刷新两层。

### 抽稀是根因修复，不是微调

取数侧 `stride = 1`（200 点/秒）**不可改**——编辑路径依赖全分辨率
（`selectionEditData` 的零 IPC 快路径要求 `stride === 1`）。于是最小缩放
（4 px/s，视口覆盖 466 秒）下视口内有 ~93,200 个采样点，却只有 3,728 个设备像素列，
**每列 25 个点渲染在同一个像素上**。这些点全额进入几何构建：

| | 点数 | 顶点数 | 每帧成本 |
|---|---|---|---|
| 抽稀前 | 213,595 | 1,922,315（30 MB） | **32 ms** |
| 抽稀后 | ~22,000 | ~200,000（3 MB） | **~3 ms** |

抽稀按列保留 y 的 min/max，因此**不丢包络**；`along` 取自原始序列，虚线相位不漂移。
精度契约与单测见 `renderKernel/gl/polylineDecimation.ts`。

### 实测（同一会话，唯一变量是抽稀开关）

| 指标 | 旧实现（Canvas2D） | 内核 · 抽稀前 | 内核 · 抽稀后 |
|---|---|---|---|
| rAF 回调 p95 | 25.8 ms | 36.9 ms | **10.3 ms** |
| 回调 >16 ms 的帧数 | 40 / 150 | 40 / 150 | **0 / 151** |
| longtask 次数 | **30** | 1 | **0** |
| longtask 总时长 | **2,240 ms** | 62 ms | **0 ms** |
| 每帧 `lineTo` | 213,955 | 0 | 0 |

**像素保真度（同会话 A/B，抽稀前后）**：差异 0.0352%；曲线包络中位 **1 px**、
最大 6 px，且 3,580 列中仅 8 列中心位差 >1.5 px——**全部是 ySpan > 20 px 的近垂直
笔画**，该度量在其上本就不稳定。

**跨运行噪声底线实测为 0 像素**（同配置跑两次逐像素相同），因此上表可跨运行复现。

### ⚠️ 验证中发现的既有问题（非本任务引入，未修复）

**内核模式下最小缩放的竖直网格线与 Canvas2D 落在不同像素列**（实测位移 0～16
物理像素）。定位结论：

- 这是**竖线**（拍/小节网格），不是曲线；曲线包络本身与 Canvas2D 一致
  （`only in A / only in B` 的少量列集中在近垂直段，属度量噪声）。
- 与抽稀**无关**：GL 无抽稀与 GL 抽稀的竖线位置**逐列相同**。
- 根因在**跨层口径**：竖网格由 DOM 层 `BackgroundGrid` 绘制并用
  `deviceSnap = Math.round(cssX * dpr) / dpr` 吸附；而参数编辑器的网格/曲线走
  `secToViewportPx`。两者在 `pxPerBeat = 2`（4 px/s、120 BPM）这种极小步距下，
  取整相位差被放大到整条线错开。
- 影响范围：仅在缩放到下限（网格步距 ≈ 2 CSS px）时可见；中高缩放步距大，差异相对
  可忽略。**建议单列一项任务**，不要在曲线任务里顺带修改——它触及两个面板的网格
  对齐口径。

### 未覆盖项（诚实记录）

- **7 种曲线变体中只有 3 种在浏览器端到端比对过**：`detected`（单条）、
  `detected2`（两条，验证逐层叠加）、`selection`（选区）。其余 4 种
  （original / edited / secondary / reference / clipboard）**未构造出可达场景**：
  - `original` / `edited` 需要 `paramView` 有数据，即后端真实返回 `orig` / `edit`
    数组。mock 的 `get_param_frames` 走兜底实现返回 `{ok:true}`，无数组；
    本次验证通过注入合成数组绕过，但那同时会**替换掉被测的取数路径**，
    因此不作为对"真实数据流"的证据。
  - `secondary` / `reference` 需要选中子轨或参考轨，mock 只创建扁平根轨
    （Phase 2 已记录的同一限制）。
  - `clipboard` 需要剪贴板非空且有选区，面板的复制路径依赖上述 `paramView` 数据。
  - 这 4 种的**几何与投影**由 `polylineGeometry`（13 项）、`polylineCoverage`
    （12 项）、`curvePoints`（15 项，含两组 >10,000 / >5,000 点的旧实现等价性对照）
    覆盖；GL 接线复用已验证的同一 program 与同一适配层。
- **2 个视口 × 2 个 dpr 的矩阵未跑全**：本次只在 1920×1200 @ dpr 2 上比对
  （`--force-device-scale-factor=2`）。`decimatePolylinePoints` 的 dpr 依赖有单测
  覆盖（dpr 1/2/4 单调性），但像素级比对未在其它 dpr 下复跑。
- **播放态仍未验证**：mock 的 `is_playing` 恒为 false（Phase 2 已记录的同一限制）。

