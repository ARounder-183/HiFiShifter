# PianoRoll Kernel Migration — Phase 2 (Render Kernel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the parameter editor's static visual layers (grid, piano keyboard axis, value labels/ticks, selection band, highlight bands, playhead) onto WebGL2 instanced geometry and activate the existing glyph pipeline for all text, so that playback frames no longer repaint those layers.

**Architecture:** Phase 1 gave the panel a kernel-owned viewport. Phase 2 adds a WebGL2 scene layer alongside the existing Canvas 2D detail layer: static geometry is built in **content coordinates** and cached on the GPU, so a scroll frame updates only `u_viewOrigin` and a playhead frame updates only one instance. Text moves from `ctx.fillText` to the dormant-but-complete glyph pipeline (rasterizer → atlas → quads → program), which the Plan-E spike validated end-to-end in a real browser. Curves stay on Canvas 2D in this phase (they move in Phase 3).

**Tech Stack:** React 19, TypeScript 5.9, Redux Toolkit, WebGL2, Canvas 2D, Vitest, Playwright-core (`frontend/scripts/dev-shot.mjs`)

**Spec:** `docs/superpowers/specs/2026-09-12-pianoroll-kernel-migration-design.md`
**Phase 1 record:** `docs/superpowers/plans/2026-09-12-pianoroll-kernel-migration.md`

**Verification prerequisites (used by every task):**
- dev server: `cd frontend && npx vite --port 5173 --strictPort`
- browser harness: `VW=1920 VH=1200 node scripts/dev-shot.mjs "<url>" <out.png> <waitMs> '<actionsJson>'`
- mock backend: URL carries `?mock=1`
- pixel diff helper: the throwaway Python PNG differ used in Phase 1 (see "Verification method" below)
- pre-existing failures: exactly 2 tests in `src/features/keybindings/keybindingMatch.test.ts`. Any other failure is caused by this work.

**Test environment (verified, applies to every task):** Vitest runs in the **node** environment — no jsdom. `typeof localStorage`, `typeof window` and `typeof document` are all `undefined` inside tests. Therefore geometry/scene builders must be **pure functions** (unit-testable), and anything DOM/GL-shaped is verified in the browser harness.

**Phase 2 flag:** reuse `hifishifter.pianoRollKernel` (Phase 1's flag). Phase 2 must remain invisible when it is `0`: with the flag off, `PianoRollPanel` must render exactly as it does today, including the Canvas 2D path. A separate `hifishifter.pianoRollKernel.gl` sub-flag gates the GL scene layer so a GL-specific problem can be reverted without losing Phase 1 (see Task 3).

---

## Reconnaissance findings that shape this plan (measured, not assumed)

These were established by profiling the running app before writing any task. They change what "done" means, so they are recorded here rather than in an appendix.

**R1 — There is no static/dynamic layer split today.** `drawPianoRoll` unconditionally clears and repaints **both** canvases on every call (`render.ts:451-454` axis, `660-664` main). Every layer is repainted per frame, including during playback when only the playhead moves.

**R2 — Repaint count per frame is measurable and already improved by Phase 1.**

| Mode | Canvas clears per frame (20-frame windows) |
|---|---|
| flag off (legacy) | 5.8, 6.0, 6.0, 6.0, 6.1 |
| flag on (Phase 1 kernel) | 2.1, 2.0, 2.0, 2.2, 2.0 |

**R3 — Sustained-repaint frame time already meets budget in kernel mode.**

| Mode | p50 | p95 | max |
|---|---|---|---|
| legacy | 16.7 ms | **29.8–31.3 ms** | 63–68 ms |
| kernel (Phase 1) | 16.7 ms | **16.8–17.1 ms** | 58–59 ms |

So Phase 1 fixed scroll jank. Phase 2's remaining prize is the **playback** path (R1): moving static layers off the per-frame Canvas2D repaint.

**R4 — The glyph pipeline works end-to-end in a real browser (spike run).** Using `createGlyphRasterizer` with the app's real font stack:

| Check | Result |
|---|---|
| rasterizer construct (no-DOM guard passes) | non-null |
| axis glyph set (22 distinct: `#+-.0-9A-G e`) | **22/22 acquired** |
| overlay glyph set (CJK/Kana/Hangul, from the real i18n string) | **22/22 acquired** |
| atlas actually rasterized | 6,642 non-transparent px of 4,194,304 |
| `measure()` vs `ctx.measureText()` | **identical** (16.51025390625) |

**R5 — Atlas capacity is a non-issue.** Distinct glyphs total ~149 (22 axis/marks + 131 overlay across all 5 locales). At dpr 2 the slot heights are 20–29 physical px, so a 2048² page holds ~4,300–8,600 slots. `maxPages: 2` is ample.

**R6 — The rasterizer cannot parse `bold` (defect found by extending the spike).** `parseFontSizePx` is anchored (`/^(\d+(?:\.\d+)?)px/`) and `scaleFontKey` is anchored (`/^(\d+(?:\.\d+)?)px\s+(.*)$/`). The renderer uses `` `bold 9px ${fontFamily}` `` for the C-key labels (`render.ts:508`). Consequences: the size parse returns the **fallback 12**, and the dpr scale-up **silently fails** (key returned unchanged), so bold glyphs rasterize unscaled into a mis-sized slot — wrong size and wrong metrics, with no error. This must be fixed before any text is routed through the pipeline (Task 1).

**R7 — Text inventory to migrate (8 `fillText` sites, 5 font specs, 4 font keys).**

| Line | String | Font | Canvas |
|---|---|---|---|
| 510 | `midiToLabel(midi)` | `bold 9px` (C) / `9px` | axis |
| 519 | `midiToLabel(midi)` | `8px` | axis |
| 575, 599, 623, 633, 640 | `formatAxisMark(m, editParam)` | `10px` | axis |
| 1144 | `overlayText` (i18n) | `12px` | main |

Distinct font keys: `8px F`, `9px F`, `bold 9px F`, `10px F`, `12px F` (5).

**R8 — `formatAxisMark` can emit exponent notation.** `parseFloat(v.toPrecision(4)).toString()` yields `1e-7`, `1e+21` for extreme values. These are covered by the axis glyph set (`e`, `+`, `-`, digits), but the atlas must not assume "digits and a dot only".

**R9 — `overlayText` is localized (CJK / Kana / Hangul / Latin).** Confirms the atlas cannot be Latin-only; R4 already proves the real set rasterizes.

**R10 — Hazards the GL port must respect** (from the layer inventory):
- 3 real dash sites (`render.ts:255`, `1004`, `1100`); the rest are resets. Dash patterns are **dpr-quantized at call time** (`getFixedDashPattern`), not constants. Curves stay on Canvas 2D in Phase 2, so dashes are *not* in scope — but the morph overlay (M10) and reference lines are also curves and stay too.
- 3 `clip()` sites are all axis-aligned rects → `gl.scissor`; two are curve-related (stay on Canvas 2D), one is the black-key label clip (Task 5).
- 1 gradient (`render.ts:492`), one per black key per frame → needs shader-side ramp or a precomputed 1-D texture.
- 1 `arc` (`render.ts:277`) — morph handles, a curve-adjacent overlay that stays on Canvas 2D.
- Line widths in CSS px: `hairlineW = 1/dpr`, `strongW = 2/dpr`, plus 1.25/1.5/1.8/2/2.6/3.2/3.6. The sub-2px fractional widths belong to curve layers (Phase 3). Phase 2's lines are `1/dpr` and `2/dpr` — both map exactly to integer device widths, and the existing `INSTANCE_MODE_FLAT` box program already expresses "content-coord rect + width in CSS px".
- Inconsistent half-pixel conventions inside `render.ts` (`hairlineY` adds `0.5` CSS px; strong lines snap with no offset; `strokePx` adds `0.5/dpr`; the selection `strokeRect` uses a literal `0.5`). Phase 2 must preserve each layer's **current** convention, not normalize them — normalizing would change pixels.

---

## Verification method (used by every task)

Phase 1 established the standard: **pixel comparison against the legacy path in the same session**, with a control pair proving the harness is deterministic.

1. **Determinism control.** Capture two screenshots of the *same* mode; they must differ by **0 pixels**. Phase 1 measured exactly 0, so any nonzero diff in a kernel-vs-legacy comparison is real signal.
2. **Pixel diff.** Decode both PNGs, count differing pixels and the max per-channel delta, and bucket by region (so a localized regression is attributable). A throwaway differ is enough; it does not ship.
3. **Region buckets.** Split the full frame into named regions (ruler / graph body / axis column / each self-drawn scrollbar) so "0.2% differ" cannot hide a fully-broken layer.
4. **Explained-or-fixed.** Any diff must be either fixed or explained by construction (e.g. macOS overlay scrollbars auto-hide at rest while self-drawn bars persist). Unexplained diffs block the task.

Screenshots are 3840×2400 at `VW=1920 VH=1200` (device scale factor 2).

---

## File structure

```
pianoRoll/kernel/
  host/pianoRollKernelHost.ts          (Phase 1) — gains the GL scene layer in Task 4
  host/pianoRollKernelData.ts          (Phase 1)
  scroll/verticalValueScroll.ts        (Phase 1)
  scroll/scrollbarSpec.ts              (Phase 1)
  scene/                                NEW in Phase 2 — pure geometry builders
    gridInstances.ts                    pitch/param gridlines + scale highlight bands
    gridInstances.test.ts
    keyboardInstances.ts                key bodies + separators + black-key ramp rects
    keyboardInstances.test.ts
    axisMarkInstances.ts                value-axis tick marks (non-pitch params)
    axisMarkInstances.test.ts
    selectionInstances.ts               selection band + border + playhead
    selectionInstances.test.ts
    textRuns.ts                         which strings go where (pure; feeds the glyph path)
    textRuns.test.ts
  glyph/                                NEW in Phase 2 — thin adapter over timeline glyph modules
    pianoRollGlyphs.ts                  rasterizer + atlas + quad assembly for this panel
    pianoRollGlyphs.test.ts
```

Shared modules stay where they are. The spec's "move to a neutral location" is **deliberately deferred**: Phase 1 already imports `scrollKernel`, `renderLoop`, `scrollbars` and `timelineAxis` across the boundary without incident, and the extraction is a pure refactor whose value is architectural cleanliness, not function. Doing it first would add a large mechanical diff (10+ modules, re-export shims, 3 external call sites) with no user-visible benefit and would taint every subsequent pixel comparison. It is tracked as Task 8 so it can be done once Phase 2's behaviour is locked by tests.

---

## Task 1: Fix the glyph rasterizer's CSS font-shorthand parsing

**Why first:** R6. Every text layer depends on this, and the failure mode is silent (wrong size, no error). Pure functions, so this is a clean unit-tested fix.

**Files:**
- Modify: `frontend/src/components/layout/timeline/kernel/glyph/glyphRasterizer.ts`
- Test: `frontend/src/components/layout/timeline/kernel/glyph/glyphRasterizer.test.ts` (existing file — add a describe block)

- [ ] **Step 1: Write the failing test**

Append to `glyphRasterizer.test.ts`. These import the two helpers, so **Step 3 must export them** (they are currently module-private).

```ts
/**
 * CSS 字体简写解析：必须支持 `bold 9px …` 这类**带前缀**的简写。
 *
 * 【为什么单独守护】`render.ts` 的 C 音名标签用的是 `bold 9px ${family}`。
 * 原实现的字号正则带 `^` 锚定，遇到 bold 前缀会解析失败并**静默回退 12**，
 * 而 dpr 放大函数也匹配失败、原样返回 —— 结果是字形按未放大的字号光栅化到
 * 一个按 12px 算出的槽位里，尺寸与度量全错，且不报任何错。
 */
describe("CSS 字体简写解析（含 bold 前缀）", () => {
    it("解析普通简写的字号", () => {
        expect(parseFontSizePx("9px sans-serif")).toBe(9);
        expect(parseFontSizePx("12px \"Segoe UI\", Roboto")).toBe(12);
        expect(parseFontSizePx("8px sans-serif")).toBe(8);
    });

    it("解析带 bold 前缀的简写字号（回归：曾静默回退 12）", () => {
        expect(parseFontSizePx("bold 9px sans-serif")).toBe(9);
        expect(parseFontSizePx("  bold 12px \"Segoe UI\", Roboto  ")).toBe(12);
    });

    it("解析带 italic / 数字字重前缀的简写", () => {
        expect(parseFontSizePx("italic 10px sans-serif")).toBe(10);
        expect(parseFontSizePx("600 10px sans-serif")).toBe(10);
    });

    it("按 dpr 放大字号（回归：bold 曾原样返回、未放大）", () => {
        expect(scaleFontKey("9px sans-serif", 2)).toBe("18px sans-serif");
        expect(scaleFontKey("bold 9px sans-serif", 2)).toBe("bold 18px sans-serif");
        expect(scaleFontKey("italic 10px \"Segoe UI\"", 3)).toBe("italic 30px \"Segoe UI\"");
    });

    it("无法解析时回退到 12（既有契约不变）", () => {
        expect(parseFontSizePx("garbage")).toBe(12);
    });

    it("只改字号，不动字体族里的数字", () => {
        // 字体族可能含数字（如 "Arial 2"）；只有带 px 的那一段才是字号。
        expect(scaleFontKey("9px Arial 2", 2)).toBe("18px Arial 2");
    });
});
```

Update that file's import line to include the two helpers:

```ts
import {
    createGlyphRasterizer,
    parseFontSizePx,
    resolveGlyphRasterizerParams,
    scaleFontKey,
} from "./glyphRasterizer";
```

(Keep whatever else the existing file already imports; only add the two names.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run glyphRasterizer`
Expected: FAIL — `parseFontSizePx` / `scaleFontKey` are not exported (import error), or assertion failures on the `bold` cases.

- [ ] **Step 3: Implement**

In `glyphRasterizer.ts`, replace the two functions' regexes and export them.

```ts
/**
 * 解析 CSS 字体简写中的字号（CSS px）。
 *
 * 【为什么不能锚定行首】CSS 简写允许在字号前放 style / weight / stretch，
 * 例如 `bold 9px sans-serif`（`render.ts` 的 C 音名标签就是这么写的）。
 * 原实现用 `/^(\d+(?:\.\d+)?)px/` 锚定行首，遇到 bold 前缀解析失败并回退 12，
 * 于是槽位高度按 12px 计算、字形却按 9px 光栅化——尺寸与度量都错，且不报错。
 * 改为只要求「数字紧跟 px」，不限定出现位置。
 *
 * @param fontKey 字体标识（CSS font 简写）。
 * @returns 字号（CSS px）；解析失败回退 12（既有契约）。
 */
export function parseFontSizePx(fontKey: string): number {
    const match = /(\d+(?:\.\d+)?)px/.exec(fontKey.trim());
    return match ? Number(match[1]) : 12;
}

/**
 * 按比例缩放 CSS 字体简写中的字号，其余部分原样保留。
 *
 * 特殊说明：只替换**第一处**「数字 + px」——字体族里也可能出现数字（如
 * `"Arial 2"`），锚定行首或全局替换都会改错。`bold 9px X` 同样适用：
 * 第一处匹配即 `9px`。
 *
 * @param fontKey 字体标识（CSS font 简写）。
 * @param scale 缩放比例（通常为 dpr）。
 * @returns 缩放后的字体标识；无法解析时**原样返回**（调用方不应依赖此兜底，
 *          因为未放大的字体配合按 dpr 计算的槽位会产生错位）。
 */
export function scaleFontKey(fontKey: string, scale: number): string {
    return fontKey
        .trim()
        .replace(/(\d+(?:\.\d+)?)px/, (_match, size: string) => `${Number(size) * scale}px`);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run glyphRasterizer`
Expected: PASS (existing tests plus the 6 new ones).

- [ ] **Step 5: Verify no regression across the glyph suite**

Run: `cd frontend && npx vitest run glyph && npx tsc -b --noEmit`
Expected: all 4 glyph test files pass; typecheck clean.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/layout/timeline/kernel/glyph/glyphRasterizer.ts frontend/src/components/layout/timeline/kernel/glyph/glyphRasterizer.test.ts
git commit -m "fix(glyph): parse CSS font shorthand with a bold/italic/weight prefix"
```

---

## Task 2: Pure scene-geometry builders for the piano-roll grid

**Why:** gridlines are the largest static layer by count (up to 97 lines for formant-cents; 61 for pitch) and the simplest to express as content-coordinate rects. Establishing the builder pattern here (pure, content coords, unit-tested in node) sets the shape for the remaining layers.

**Files:**
- Create: `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.ts`
- Test: `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.test.ts`

**Contract to reproduce exactly (from `render.ts:671-821`):**

| Param kind | Step | Strong condition | Strong color (dark / light) | Weak color (dark / light) |
|---|---|---|---|---|
| pitch | 1 semitone, `midi` from `floor(min)` to `ceil(max)` **inclusive** | `pc === 0` | `pitchGridC` | `pitchGridOther` |
| child cents | 100 | `round(v) % 1200 === 0` | `rgba(255,255,255,0.14)` / `rgba(0,0,0,0.16)` | `rgba(255,255,255,0.07)` / `rgba(0,0,0,0.08)` |
| child degrees | 1 | `round(v) % 7 === 0` | same | same |
| formant cents | 50 | `round(v) % 600 === 0` | same | same |

Y conventions to preserve: weak lines use `hairlineY(y) = (round(y*dpr) + 0.5)/dpr`; strong lines use `round(y*dpr)/dpr` (no half-pixel). Widths: weak `1/dpr`, strong `2/dpr`.

Scale-highlight bands (time-varying x) are also emitted here when `highlightActive`.

- [ ] **Step 1: Write the failing test**

```ts
/**
 * 参数编辑器内核 · 网格实例构建（纯函数）单测。
 *
 * 【本测试守护什么】像素级复刻 `render.ts:671-821` 的网格语义：
 * 取值域的**整数半音 / 步进**、强弱线的判定阈值、以及两者**不同的**
 * 半像素取向（弱线 +0.5/dpr，强线不加）。取向写错不会报错，只会让网格
 * 整体偏移半个设备像素——表现为"发虚"，很难归因。
 */
import { describe, expect, it } from "vitest";

import { buildPitchGridInstances, buildValueGridInstances } from "./gridInstances";

/** 一个把值线性映射到视口 y 的桩（0 在上、h 在下），便于手算期望。 */
const makeValueToY = () => (v: number, _param: string, h: number) => ((100 - v) / 100) * h;

describe("buildPitchGridInstances", () => {
    it("按整数半音逐行产出，范围含端点（与 render.ts 的 <= 一致）", () => {
        const items = buildPitchGridInstances({
            view: { center: 50, span: 10 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: { r: 1, g: 0, b: 0, a: 1 },
            colorOther: { r: 0, g: 0, b: 1, a: 1 },
        });
        // span 10 / center 50 → min 45, max 55 → 45..55 共 11 行
        expect(items.length).toBe(11);
    });

    it("pc === 0（C）用 colorC，其余用 colorOther", () => {
        const items = buildPitchGridInstances({
            view: { center: 48, span: 4 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: { r: 1, g: 0, b: 0, a: 1 },
            colorOther: { r: 0, g: 0, b: 1, a: 1 },
        });
        const cRows = items.filter((i) => i.rgba.r === 1);
        // 46..50 → 48 是 C
        expect(cRows.length).toBe(1);
    });

    it("横线几何：x=0、w=视口宽、h=线厚（易错点：宽与厚不可互换）", () => {
        const items = buildPitchGridInstances({
            view: { center: 50, span: 2 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 2,
            valueToY: makeValueToY(),
            colorC: { r: 1, g: 0, b: 0, a: 1 },
            colorOther: { r: 0, g: 0, b: 1, a: 1 },
        });
        for (const item of items) {
            expect(item.x).toBe(0);
            expect(item.w).toBe(800); // 横向范围
            expect(item.h).toBeCloseTo(0.5, 9); // 1/dpr = 线厚
            // y 必须是 (round(v*2)+0.5)/2 的形式（弱线的半设备像素取向）
            expect(Math.abs((item.y * 2) % 1)).toBeCloseTo(0.5, 9);
        }
    });
});

describe("buildValueGridInstances", () => {
    it("cents 参数按 100 步进、%1200 为强线", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 300 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: { r: 1, g: 1, b: 1, a: 1 },
            weakRgba: { r: 0, g: 0, b: 0, a: 1 },
        });
        // -150..150 步进 100 → -100, 0, 100（0 为强线）
        expect(items.map((i) => Math.round(i.value)).sort((a, b) => a - b)).toEqual([-100, 0, 100]);
        expect(items.filter((i) => i.rgba.r === 1).length).toBe(1);
    });

    it("degrees 参数按 1 步进、%7 为强线", () => {
        const items = buildValueGridInstances({
            kind: "degrees",
            view: { center: 0, span: 7 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: { r: 1, g: 1, b: 1, a: 1 },
            weakRgba: { r: 0, g: 0, b: 0, a: 1 },
        });
        // -3.5..3.5 步进 1 → -3..3 共 7 行；强线只有 0
        expect(items.length).toBe(7);
        expect(items.filter((i) => i.rgba.r === 1).length).toBe(1);
    });

    it("formant 参数按 50 步进、%600 为强线", () => {
        const items = buildValueGridInstances({
            kind: "formantCents",
            view: { center: 0, span: 200 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: { r: 1, g: 1, b: 1, a: 1 },
            weakRgba: { r: 0, g: 0, b: 0, a: 1 },
        });
        // -100..100 步进 50 → -100,-50,0,50,100
        expect(items.length).toBe(5);
        expect(items.filter((i) => i.rgba.r === 1).length).toBe(1);
    });

    it("强线不加半像素、弱线加半像素（两种取向都必须保留）", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 300 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 2,
            valueToY: makeValueToY(),
            strongRgba: { r: 1, g: 1, b: 1, a: 1 },
            weakRgba: { r: 0, g: 0, b: 0, a: 1 },
        });
        for (const item of items) {
            const isStrong = item.rgba.r === 1;
            const frac = Math.abs((item.y * 2) % 1);
            if (isStrong) expect(frac).toBeCloseTo(0, 9);
            else expect(frac).toBeCloseTo(0.5, 9);
        }
    });

    it("强线线厚是弱线的两倍（1/dpr vs 2/dpr）", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 300 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 2,
            valueToY: makeValueToY(),
            strongRgba: { r: 1, g: 1, b: 1, a: 1 },
            weakRgba: { r: 0, g: 0, b: 0, a: 1 },
        });
        for (const item of items) {
            const expected = item.rgba.r === 1 ? 1 : 0.5; // 2/dpr vs 1/dpr
            expect(item.h).toBeCloseTo(expected, 9);
        }
    });

    it("span 非正时退化为单行（与 render.ts 的 1e-6 下限一致，不是空数组）", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 0 },
            heightPx: 100,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: { r: 1, g: 1, b: 1, a: 1 },
            weakRgba: { r: 0, g: 0, b: 0, a: 1 },
        });
        // render.ts:743 用 Math.max(1e-6, view.span)：span=0 时域退化为 [0,0]，
        // 循环仍会产出 v=0 这一行。**不是**空数组——写成空数组会让实现与既有
        // 渲染分叉（少一条零线）。
        expect(items.length).toBe(1);
        expect(items[0].value).toBeCloseTo(0, 9);
    });

    it("span 为 NaN / Infinity 时返回空数组（防死循环）", () => {
        // render.ts 的 `Math.max(1e-6, NaN)` 仍是 NaN，`for` 上界为 Infinity 时
        // 循环永不终止 —— 原实现没有这层保护，因为调用方保证 span 有限。
        // 几何构建跑在渲染热路径上，一旦死循环整个面板会卡死，故必须显式防御。
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            const items = buildValueGridInstances({
                kind: "cents",
                view: { center: 0, span: bad },
                heightPx: 100,
                dpr: 1,
                valueToY: makeValueToY(),
                strongRgba: { r: 1, g: 1, b: 1, a: 1 },
                weakRgba: { r: 0, g: 0, b: 0, a: 1 },
            });
            expect(items).toEqual([]);
        }
    });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run gridInstances`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `gridInstances.ts` with the file header comment (per repo convention: 主要内容 / 作用 / 与其他模块的关系), then:

```ts
/** 依据参数种类选择步进与强线间隔。 */
export type ValueGridKind = "cents" | "degrees" | "formantCents";
```

Implement `buildPitchGridInstances` and `buildValueGridInstances` returning `GridInstance[]` where

```ts
export interface GridInstance {
    /** 矩形左缘 x（内容坐标 CSS px）。网格线横跨整个视口，故由调用方填 0。 */
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
    readonly rgba: Rgba;
    /** 该行对应的参数值（供比例高亮与调试；网格线本身不需要）。 */
    readonly value: number;
}
```

Internally use a shared `hairlineY(y, dpr)` / `snapY(y, dpr)` pair mirroring `render.ts:440` and `render.ts:752`. Emit **content-coordinate** y (the GL program subtracts `u_viewOrigin`), width `1/dpr` (weak) or `2/dpr` (strong), and `x: 0`.

**Important:** clamp `span` with the same `1e-6` floor and the same `absMin/absMax` clamping as `render.ts:676-680` so the row set matches exactly. Note the consequences of that floor, which the tests pin: `span: 0` degenerates to a **single row at the centre value** (not an empty set — `Math.max(1e-6, 0)` is `1e-6`, so the domain is `[center, center]` and the loop still emits one line), while `span: NaN` stays `NaN` and `span: Infinity` gives an infinite loop bound. Guard the latter two by returning `[]` — this geometry runs on the render hot path, so an unbounded loop would hang the panel.

**Instance geometry for a horizontal line:** `GridInstance` is a plain axis-aligned rect, so for "a horizontal line spanning the viewport":
- `x = 0`, `y` = the line's content-coordinate vertical position (after the half-pixel convention),
- `w` = **horizontal extent** = the viewport width,
- `h` = **thickness** = `1/dpr` (weak) or `2/dpr` (strong).

Because the builder is pure and is also used off-screen, it takes `viewportWidthPx` as an explicit argument rather than reading it from a DOM node. The unit tests below assert `w` (extent) and `h` (thickness) separately, so a transposed rectangle fails loudly instead of drawing a full-height bar.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run gridInstances`
Expected: PASS.

- [ ] **Step 5: Verify no regression**

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit`
Expected: only the 2 known `keybindingMatch` failures; typecheck clean.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/layout/pianoRoll/kernel/scene/
git commit -m "feat(pianoroll-kernel): pure grid-instance builders for the phase-2 GL layer"
```

---

## Task 3: GL scene layer skeleton in the host, behind a sub-flag

**Why:** before porting more layers, establish the GL plumbing (context, flat-instance program, content-coordinate geometry cache, view-origin-only scroll repaint) and the ability to revert it independently of Phase 1.

**Files:**
- Modify: `frontend/src/components/layout/pianoRoll/kernel/host/pianoRollKernelHost.ts`
- Modify: `frontend/src/components/layout/timeline/kernel/featureFlag.ts`
- Test: `frontend/src/components/layout/timeline/kernel/featureFlag.test.ts` (add cases)

**Design:**
- Add `isPianoRollGlSceneEnabled()`, key `hifishifter.pianoRollKernel.gl`, default **off**, and only meaningful when `isPianoRollKernelEnabled()` is true.
- The host gains an optional `glCanvas: HTMLCanvasElement | null` arg. When the sub-flag is on and `createGlCanvas` succeeds, the host draws grid instances with the existing `createSdfBoxProgram` in `INSTANCE_MODE_FLAT`.
- Geometry is built once per **data/geometry change** and cached; a scroll frame only calls `sdfBox.repaint(target, viewOriginX, viewOriginY)` (the timeline host's proven zero-rebuild path).
- **Failure policy:** if WebGL2 is unavailable, or `createGlCanvas` returns null, or the program throws, the host must **log once and fall back to the Canvas 2D path** without breaking the panel. A GL failure must never blank the parameter editor.

- [ ] **Step 1: Write the failing test**

Add to `featureFlag.test.ts`:

```ts
describe("isPianoRollGlSceneEnabled", () => {
    it("未设置 → 关闭", () => {
        restore = installStorage(() => null);
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });

    it("只有 '1' 才开启（'true' / 'yes' 等不算）", () => {
        restore = installStorage((key) =>
            key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? "1" : null,
        );
        expect(isPianoRollGlSceneEnabled()).toBe(true);
        restore = installStorage((key) =>
            key === PIANO_ROLL_KERNEL_GL_FLAG_KEY ? "true" : null,
        );
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });

    it("存储抛错 → 关闭且不抛异常", () => {
        restore = installStorage(() => {
            throw new Error("denied");
        });
        expect(isPianoRollGlSceneEnabled()).toBe(false);
    });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run featureFlag`
Expected: FAIL — `isPianoRollGlSceneEnabled` / `PIANO_ROLL_KERNEL_GL_FLAG_KEY` not exported.

- [ ] **Step 3: Implement the flag**

Append to `featureFlag.ts`, following the existing guarded `globalThis.localStorage` pattern (node test environment), with a header comment explaining that this sub-flag exists so a GL-specific regression can be reverted without losing Phase 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run featureFlag`
Expected: PASS.

- [ ] **Step 5: Add the GL layer to the host**

In `pianoRollKernelHost.ts`:
- Extend `PianoRollKernelHostArgs` with `readonly glCanvas?: HTMLCanvasElement | null;`
- In the constructor, attempt `createGlCanvas(...)` + `createSdfBoxProgram(gl)` **only** when the sub-flag is on; wrap in `try/catch` and record a `glUnavailableReason` so the panel can surface it (dev-only) without crashing.
- Add `rebuildGlGeometry()` that consumes `buildPitchGridInstances` / `buildValueGridInstances` (Task 2) from `data()` and uploads via the instance buffer; call it on construction and whenever the geometry inputs change (tracked by a cheap signature string of the inputs, mirroring the timeline host's `sceneContentChanged`).
- In `draw()`, when GL is live: `glCanvas.resize(...)` → `glCanvas.clear()` → on geometry change `sdfBox.render(instances, count, target, view.scrollLeft, view.scrollTop)`, else `sdfBox.repaint(target, ...)`. Then still call `onFrame(axis)` so the Canvas 2D layer continues to paint (curves stay there until Phase 3).
- `dispose()` must release the GL program/buffer.

- [ ] **Step 6: Verify in the browser (both sub-flag states)**

Run with the sub-flag on and off, confirming the panel renders and the grid appears identical:
```bash
VW=1920 VH=1200 node scripts/dev-shot.mjs "http://localhost:5173/?mock=1" /tmp/gl-on.png 4000 "$(cat /tmp/actions-gl-on.json)"
```
Expected: no `pageerror`, grid visible, and the pixel diff against the Canvas-2D grid within tolerance (accounting for AA differences at hairline width).

- [ ] **Step 7: Verify no regression**

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit && npx eslint src/components/layout/pianoRoll/kernel/`
Expected: only the 2 known failures; 0 lint errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/layout/pianoRoll/kernel/host/ frontend/src/components/layout/timeline/kernel/featureFlag.ts frontend/src/components/layout/timeline/kernel/featureFlag.test.ts
git commit -m "feat(pianoroll-kernel): GL scene layer skeleton behind a sub-flag"
```

---

## Phase 2 remaining tasks (expand in place when each starts)

Same pattern Phase 1 used, and for the same reason: each depends on the previous task's concrete shape (the real instance layout, the real glyph adapter signature), so writing step-level detail now would be fiction.

### Task 4: Piano keyboard axis → GL (key bodies, separators, black-key ramp)

- Extract the geometry from `render.ts:462-532` into `keyboardInstances.ts` (pure, tested): white/black key rects in **value space**, the per-key separator lines, and the black-key right-edge ramp rect.
- The black-key gradient (`render.ts:492`) becomes either (a) a 1-D ramp texture sampled by the flat/fragment program, or (b) a fixed set of N nested alpha rects. Decision to be made with a pixel comparison; the gradient is only 5.6 CSS px wide, so (b) is acceptable if it matches within tolerance.
- Black-key **labels** are clipped to `rect(0, top, w*0.7, keyH)` → `gl.scissor`.
- **Exit check:** keyboard column compared against the Canvas 2D render at several pitch views (zoomed in/out, scrolled to each end) within tolerance.

### Task 5: Value-axis marks + tick labels via the glyph program

- `axisMarkInstances.ts` for the four tick-line variants (`render.ts:533-648`).
- `pianoRollGlyphs.ts` adapter: rasterizer + atlas + `buildGlyphQuads` for the 5 font keys, including the `bold 9px` C-label variant.
- Wire all 7 axis `fillText` sites (R7) plus the overlay hint text (`render.ts:1144`).
- **Exit check:** text position and size match `ctx.fillText` at dpr 1 and 2; the C labels are bold; CJK overlay text renders.

### Task 6: Selection band and playhead → GL

- `selectionInstances.ts` from `render.ts:823-834` and `1148-1164`.
- **Critical:** preserve the existing (inconsistent) pixel conventions — the selection border uses a literal `0.5` offset while the playhead uses `strokePx`'s `0.5/dpr`. Do **not** normalize them; normalizing changes pixels.
- The playhead becomes a single instance whose `u_viewOrigin`-relative position is updated per frame, so **playback frames repaint only that instance** — the phase's headline exit criterion.
- **Exit check:** with the GL layer live and the Canvas 2D static layers disabled, playback advances the playhead with no grid/keyboard repaint (verified by canvas-clear counting, the metric from R2).

### Task 7: Pixel comparison, playback profiling, and Phase 2 verification record

- Full-frame region-bucketed diff (flag off vs GL on) at 2 window sizes and 2 dprs.
- Re-run the R2/R3 measurements to prove the playback repaint actually dropped.
- Record results in this plan in the Phase 1 style; update the spec's phase table.
- **Exit check:** `npx vitest run` (only the 2 known failures), `npx tsc -b --noEmit`, `npx eslint`, `npx prettier --check` clean.

### Task 8: Extract shared kernel modules to a neutral location (deferred refactor)

- Move `glContext`, `instanceBuffer`, `instanceLayout`, `sdfBoxProgram`, `glyph/*`, `glyphProgram`, `renderLoop`, `scrollKernel`, `timelineAxis`, `canvasRaster`, `devicePixelLine` to a neutral directory (e.g. `components/layout/renderKernel/`) with re-export shims left behind.
- **Gate:** the timeline kernel's entire test suite plus the Phase 1 piano-roll suite must pass unchanged, and the timeline's browser pixel comparison must be unchanged.
- Deferred to last because it is behaviour-preserving by construction and would otherwise taint every Phase 2 pixel comparison.

---

## Phase 2 — 完成记录（Task 1–7）

执行日期：与阶段 1 同一分支 `feature/timeline-unified-render-kernel`，未推送。
开关：`hifishifter.pianoRollKernel` + `hifishifter.pianoRollKernel.gl`（两者都需为 `"1"`）。

### 提交清单

| 提交 | 任务 | 内容 |
|---|---|---|
| `bd4239af` | 1 | 字形光栅化器支持带前缀的 CSS 字体简写（`bold 9px …`） |
| `cb74ec58` | 2 | 网格实例构建器（pitch 半音线 + 三类值域步进线） |
| `87b875c8` | 3 | 调色板提取为共享模块 + GL 子开关 |
| `0ef88912` | 3 | GL 网格层（含 viewport origin 与 stroke 语义修正） |
| `e6106b02` | 4 | 键盘轴 GL 层（覆盖率修正的抗锯齿） |
| `9651b2d6` | 5 | 数值轴刻度几何（纯函数） |
| `c8d0dbb2` | 5 | 字形适配层接线 |
| `54212d11` | 5 | 轴文字与刻度线搬到 GL |
| `51e2f605` | 6 | 叠加层 + 静态缓存（播放不再重绘） |

### 关键指标（浏览器实测，1920×1200 @dpr 2）

**播放/重绘开销**（每帧持续 invalidate，200 帧）

| 指标 | 迁移前（Canvas2D） | 迁移后（GL） |
|---|---|---|
| 每帧清屏次数 | **1.98** | **0.00** |
| 曲线画布清屏次数（120 帧内） | 123 | **1** |
| 每帧 Canvas2D 绘图调用 | **318**（stroke 10149 / fillText 4776） | **0** |
| 帧间隔 p50 / p95 | 16.7 / 16.8 ms | 16.7 / 16.8 ms |

帧间隔两者相同是**预期**的：绘制工作已被移出关键路径，帧率本就被 vsync 钉在 16.7ms。
真正的收益是"每帧不再有 318 次 Canvas2D 调用"——这在低端设备与高分辨率下才是瓶颈。

**视觉一致性**

| 对比 | 差异 |
|---|---|
| 开关关闭 vs 迁移前基线 | **0 像素**（每次改动后复测，始终为 0） |
| GL 开启 vs Canvas2D，静置 | 7538 px / 184352（**0.0818%**），maxdelta 214 |
| GL 开启 vs Canvas2D，滚动到 drawing x=1200 | 7542 px（**0.0818%**），与静置一致 |

残差构成：**96% 的差异像素只差 1/255**；较大差异集中在
‑ 字形图集的亚像素量化（8px / 10px 各差 0.5 设备像素；9px 与 bold 9px **完全一致**）
‑ 黑键渐变右缘 4px 宽的一条（9 行）

### 本轮修掉的真实缺陷（全部由像素比对或等价性测试发现）

1. **`bold` 字体简写解析失败**（Task 1）。`parseFontSizePx` 锚定行首，`bold 9px` 回退到
   12；`scaleFontKey` 同时失配、未按 dpr 放大。后果是 C 音名按 12px 槽位光栅化 9px 字形。
   实测槽高 29（应为 22）。
2. **调色板转写错误**（Task 3）。像素比对先抓到深色 `playheadLine` 写成 0.28（应 0.25）；
   随后的脚本化逐值比对又抓到浅色 `blackKey` 写成 `#3a3d42`（应 `#3a3a3a`）——后者是
   单靠像素差异难以归因的。
3. **stroke 语义差**（Task 3）。Canvas2D `stroke` 以 y 为**中心**，GL 实例矩形以 y 为
   **上缘**，整组网格线低了半个线厚（dpr 2 下恰好 1 设备像素）。修正后差异从 1.83% 降到 0.10%。
4. **视口原点**（Task 3）。网格横线横跨视口（Canvas2D 就是 `moveTo(0,y)`），我却传了
   内容坐标原点，导致整组线按滚动量平移——静置时左侧 200px 空白，恰好等于同步偏移。
5. **`parseRgbaColor` 的洋红陷阱**（Task 4）。该函数对非 `rgb()/rgba()` 输入返回**不透明
   洋红**（刻意设计以暴露漏解析），而调色板里 `whiteKey`/`blackKey` 是 hex，整个键盘变洋红。
6. **图层次序**（Task 4）。`axisBorder` 在 Canvas2D 里先画、随后被琴键盖住；放到上层画布后
   浮在琴键之上（右缘 229 vs 255）。
7. **覆盖率抗锯齿**（Task 4）。GL 矩形是硬边、按像素中心采样，而 `fillRect`/`stroke` 按
   面积覆盖率混合。不足一个设备行的边界带会被**整条丢弃**——实测每个八度的 C 分隔线上方
   少一整行（Canvas2D 718 行 vs GL 708 行）。修正为"对齐到设备行 + alpha 乘覆盖率"后两者相等。
8. **渐变边界**（Task 4）。8 段近似在右边界差 60 个色阶；改为逐设备列精确取覆盖率。
9. **`isBlackKey` 的导入链**（Task 4）。从 `render.ts` 导出会让宿主单测经 `../timeline`
   桶文件拉进 Redux store，在 node 环境下因裸 `localStorage` 崩溃。移到无依赖的 `utils.ts`。
10. **回退分支步长**（Task 5）。`fallback` 必须用 `niceAxisStep(span, 4)`，我误用了 cents
    候选表，小跨度下差几个数量级（span 1e-6 时 1 个刻度 vs 5 个）。
11. **degrees 的重复 0 标签**（Task 5）。`render.ts` **无条件**补画 0 标签；而 degrees 步长
    是整数，视口含 0 时 0 已是普通刻度——原实现会在同一位置画两次、alpha 叠加（0.55 → 0.7975）。
    这是常见路径，必须复刻。
12. **清屏本身就是重绘**（Task 6）。只跳过绘制而不跳过 `clearCanvasPhysical` 时，每帧仍有
    1.98 次清屏——实测确认"跳过绘制"不足，必须整张跳过。

### 已知未覆盖项（诚实记录，未验证的不声称已验证）

- **值域轴（非 pitch 参数）的浏览器端到端未验证**：`child_pitch_offset_*` 要求选中**子轨**，
  而 mock 后端只创建扁平根轨。刻度几何与标签由 15 个单测 + 84022 次与 `render.ts` 的逐值
  对照覆盖；GL 接线复用已端到端验证的 pitch 路径（同一 program / 同一字形适配层 / 同一坐标约定）。
- **播放态未在 mock 下验证**：mock 的 `is_playing` 恒为 false，无法真正启动播放。改用
  "每帧 invalidate"等价负载测量（这正是播放时面板的行为：`onFrame` 驱动重绘）。
- **选中态与选区块拖拽未做像素比对**：选区块的 GL 绘制路径已接线，但未构造"有选区"的
  对照截图。

### 退出标准对照

| 标准 | 结果 |
|---|---|
| 像素比对在容差内 | ✅ 0.0818%，96% 差异为 1/255；开关关闭时 0 像素 |
| 文字质量一致 | ✅ 字形管线渲染；9px/bold 9px 完全一致，8px/10px 差 0.5 设备像素 |
| 播放帧不再重绘曲线 | ✅ 每帧 Canvas2D 绘图调用 318 → **0**；曲线画布清屏 123 → **1** |
| 单测全绿 | ✅ 676 passed（仅 2 个既有的 `keybindingMatch` 失败） |
| 类型 / lint / 格式 | ✅ `tsc -b --noEmit` 干净；`eslint` 0 error |

## Self-Review

**Spec coverage:** the spec's Phase 2 row lists "Grid, keyboard axis, value labels/ticks, selection, playhead, highlight bands move to GL instanced geometry; the existing glyph pipeline takes over all `fillText`. Curves stay on Canvas 2D in the detail layer." Grid → Task 2/3; highlight bands → Task 2 (scale-highlight segments); keyboard axis → Task 4; value labels/ticks → Task 5; selection + playhead → Task 6; glyph activation → Tasks 1 + 5; curves untouched → no task (correctly out of scope). The spec's exit criteria — pixel comparison within tolerance, text quality matches, playback frames no longer repaint curves — map to Tasks 4/5/6 exit checks and the Task 7 profiling.

**Deviation from the spec, recorded deliberately:** the spec's architecture section says shared modules "move to a neutral location with re-export shims" as part of this work. This plan defers that to Task 8 (last) because it is behaviour-preserving, carries the largest mechanical diff, and would add noise to every pixel comparison if done first. The functional dependency the spec assumed does not exist: Phase 1 already imports across the `timeline ↔ pianoRoll` boundary without incident.

**Placeholder scan:** Tasks 1–3 carry complete failing tests, complete implementation for the pure parts, exact commands, and concrete exit checks. Tasks 4–8 are declared as phase-entry expansions with explicit files and exit checks rather than vague instructions — the same granularity Phase 1 used successfully, and necessary because the instance layout and glyph adapter signatures are established by Tasks 2–3.

**Type consistency:** `GridInstance` (Task 2) is consumed by Task 3's `rebuildGlGeometry`. `parseFontSizePx` / `scaleFontKey` (Task 1) are consumed by Task 5's adapter. `PIANO_ROLL_KERNEL_GL_FLAG_KEY` / `isPianoRollGlSceneEnabled` (Task 3) gate the GL layer for Tasks 4–7. `ValueGridKind` (Task 2) uses the same three kind names as the renderer's `isChildPitchOffsetCentsParam` / `isChildPitchOffsetDegreesParam` / `isChildFormantOffsetCentsParam` branches.

**Risk ranking (highest first), with the mitigation each task carries:**
1. Sub-pixel conventions (four different ones coexist in `render.ts`) → unit-tested in Task 2, preserved-not-normalized in Task 6.
2. Text fidelity (baseline, bold, CJK, dpr) → spike already passed (R4); Task 1 fixes the one real defect found.
3. Black-key gradient in a non-gradient pipeline → Task 4 offers two implementations and picks by measurement.
4. GL context budget (the app already creates 2–3 contexts) → the piano-roll GL layer is one more canvas; if context creation fails, the host falls back to Canvas 2D (Task 3 failure policy).
