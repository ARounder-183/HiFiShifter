# 时间轴内核 · 剩余交互补全 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让时间轴渲染内核在交互上与旧实现完全对齐——先补齐价值最高的「clip header 交互族」（静音 / 共振峰 / 重命名 / 增益与速率编辑），再依次补 `OverlapEditLayer`、snap offset 拖拽、ghost 与素材拖入。

**Architecture:** 内核模式下 clip 是**自绘**的（GL 块面 + Canvas2D 细节层），没有任何 DOM 内容层，因此所有命中必须由几何计算得出。核心不变式是「**看到的 = 可点的**」：命中区与绘制区必须消费**同一份位置常量**。本计划的做法是——`clipHeaderControls` 纯函数**直接消费 `buildTimelineClipVisualStyle` 的返回值**（绘制端也在用同一个函数），而不是另抄一套偏移量。行内编辑（重命名 / 徽标输入）在旧实现里由 `ClipHeader`（DOM）承载，内核模式下需新建一个浮层组件，按内核视口把输入框定位到 clip 上。

**Tech Stack:** TypeScript（严格模式）、React 19、Redux Toolkit、WebGL2 + Canvas2D、Vitest、Tailwind、playwright-core（浏览器验证脚本 `frontend/scripts/dev-shot.mjs`）

**开关与验证前提：**
- 内核开关：`localStorage['hifishifter.timelineKernel']`（`"1"` 开 / `"0"` 关）
- 浏览器 mock 后端：URL 加 `?mock=1`
- 截图/交互脚本：`VW=1920 VH=1200 KERNEL=1 node scripts/dev-shot.mjs "<url>" <out.png> <waitMs> '<actionsJson>'`
- 内核调试出口：`window.__hfsKernel`（`getViewport()` / `getAxis()`）

---

## 背景：旧实现的 header 布局（单一事实来源）

`frontend/src/components/layout/timeline/runtime/timelineCanvasStyle.ts` 的
`buildTimelineClipVisualStyle()` 返回全部偏移量，**坐标系原点 = clip 左上角**：

| 控件 | 位置（相对 clip 左上角） | 尺寸 | 可见性开关 |
|---|---|---|---|
| 增益旋钮 | 圆心 `(gainKnobCenterOffsetX=15, gainKnobCenterOffsetY=10)` | 半径 `gainKnobRadius` | `showGainKnob` |
| 链徽标 | `(chainBadgeOffsetX = showGainKnob ? 28 : 8, 3)` | 20×14 | `showChainBadge` |
| 静音徽标 | `(muteBadgeOffsetX = showChainBadge ? chainX+22 : chainX, 3)` | 20×14 | `showMuteBadge` |
| 共振峰徽标 | `(formantBadgeOffsetX = muteX+22, 3)` | 20×14 | `showFormantBadge` |
| 名称区 | `x ∈ [leadingControlsWidth, clipWidth - trailingReservePx + 4]` | header 高 | `showName` |
| 增益标签 | 右对齐：`gainX = clipWidth - gainLabelWidth - 6` | 基线中心 `y=9` | `showGainLabel` |
| 速率标签 | `rateX = gainX - rateLabelWidth - 8` | 基线中心 `y=9` | `showPlaybackRate` |

绘制侧见 `timelineCanvasRenderer.ts` 的 `drawClipDetails()`（旋钮 578、链 615、静音/共振峰 660+、标签 707、名称 721）。

**旧实现的交互触发方式（需对齐）：**

| 交互 | 触发 | 旧实现落点 |
|---|---|---|
| 静音切换 | 单击静音徽标 | `ClipHeader.tsx:712` → `toggleClipMuted(clip.id, !clip.muted)` |
| 打开共振峰浮窗 | 单击 F 徽标 | `ClipFormantButton.tsx:66` → `openClipFormantToolWindow({clipId, anchor})` |
| 重命名 | **双击名称区** | `ClipHeader.tsx:810` → `onRenameClickCandidate` → `beginNameEditing()` → `onRenameStart(clipId)` |
| 增益/速率行内编辑 | 单击增益旋钮 / 增益标签 / 速率标签 | `ClipHeader.tsx:264` → `onBadgeEditStart(clipId, field)`（`field: "rate" \| "gain"`） |
| 速率高级编辑 | **右键**速率标签 | `ClipHeader.tsx` → `onRateBadgeMenu(clipId, screenX, screenY)` |

**已确认可复用、无需重写的部分：**
- `ClipRateEditorDialog`（速率高级编辑对话框）已在 `TimelinePanel.tsx:3618`，**位于内核开关之外** ✓ 只缺入口。
- `ClipFormantToolWindow` 已移出内核开关（commit `e89a9daa`）✓ 只缺入口。
- `hitTest` 的 `header` 分区已存在 ✓。
- 双击判定（同一 pointerId、位移 ≤6px、间隔 ≤500ms）已在 host 实现 ✓。

---

## Phase A：clip header 交互族（本计划执行）

### Task 1: 样式补充标签像素宽度（消除第二份事实来源）

**为什么：** 绘制端用 `ctx.measureText` 现算增益/速率标签宽度，命中端若也现算就会有两份宽度来源（字体或测量环境变化时会漂移）。改为在 `buildTimelineClipVisualStyle` 里算一次并返回，绘制与命中都消费它。

**Files:**
- Modify: `frontend/src/components/layout/timeline/runtime/timelineCanvasStyle.ts`（返回类型 ~477 行、实现 ~540 行、返回对象 ~698 行）
- Test: `frontend/src/components/layout/timeline/runtime/timelineCanvasStyle.test.ts`（若不存在则创建）

**Step 1: 写失败的测试**

在 `timelineCanvasStyle.test.ts` 追加：

```ts
import { describe, expect, it } from "vitest";
import { buildTimelineClipVisualStyle } from "./timelineCanvasStyle";

describe("buildTimelineClipVisualStyle · 标签宽度", () => {
    const base = {
        widthPx: 200,
        selected: false,
        muted: false,
        gain: 0,
        playbackRate: 1,
        name: "Take 1",
    };

    it("返回增益与速率标签的像素宽度（供命中测试复用）", () => {
        const style = buildTimelineClipVisualStyle(base);
        expect(style.gainLabelWidth).toBeGreaterThan(0);
        expect(style.rateLabelWidth).toBeGreaterThan(0);
    });

    it("速率 = 1 时仍返回速率标签宽度（标签文案为 x1）", () => {
        const style = buildTimelineClipVisualStyle(base);
        expect(style.playbackRateLabel.length).toBeGreaterThan(0);
        expect(style.rateLabelWidth).toBeGreaterThan(0);
    });

    it("宽度随文本变长而增大（测量真的生效）", () => {
        const short = buildTimelineClipVisualStyle(base);
        const long = buildTimelineClipVisualStyle({ ...base, gain: -11.5 });
        expect(long.gainLabelWidth).toBeGreaterThan(short.gainLabelWidth);
    });
});
```

**Step 2: 运行测试确认失败**

```bash
cd frontend && npx vitest run timelineCanvasStyle
```
Expected: FAIL —`style.gainLabelWidth` 为 `undefined`（`toBeGreaterThan` 报错）。

**Step 3: 实现**

`timelineCanvasStyle.ts` 的返回类型里（紧邻 `leadingControlsWidth: number;` / `trailingReservePx: number;`）加：

```ts
    /** 增益标签（如 "+0.0dB"）的像素宽度：命中测试复用，避免第二份测量来源。 */
    gainLabelWidth: number;
    /** 速率标签（如 "x1"）的像素宽度。 */
    rateLabelWidth: number;
```

实现处（`const rateLabelWidth = ...` 附近，~540 行）已有局部变量 `rateLabelWidth`；再补 `gainLabelWidth`：

```ts
    // 标签宽度在此算一次并返回：绘制端与命中端都消费它，避免两份测量来源。
    const gainLabelWidth = showGainLabel
        ? measureTextWidth(gainLabel, LABEL_FONT_STYLE, fontFamily)
        : 0;
```

返回对象里（`leadingControlsWidth,` 之后）加：

```ts
        gainLabelWidth,
        rateLabelWidth,
```

> 注意：`rateLabelWidth` 已存在，确认其声明处也带 `showPlaybackRate` 判定；若它只用于 `trailingReservePx` 计算而未被返回，直接加进返回对象即可。同时把 `gainLabelWidth` 并入 `trailingReservePx` 的现有算式（若算式里用的是 `gainLabelWidth + rateLabelWidth + 16` 这类表达式，保持等价）。

**Step 4: 让绘制端消费它（消除重复测量）**

`timelineCanvasRenderer.ts` 的 `drawClipDetails()`（~707 行）：

```ts
        if (style.showGainLabel) {
            ctx.fillStyle = style.textFill;
            ctx.font = `10px ${fontFamily}`;
            ctx.textBaseline = "middle";
            const gainX = clipLeft + clipWidth - style.gainLabelWidth - 6;
            if (style.showPlaybackRate) {
                const rateX = gainX - style.rateLabelWidth - 8;
                ctx.fillText(style.playbackRateLabel, rateX, clipTop + 9);
            }
            ctx.fillText(style.gainLabel, gainX, clipTop + 9);
        }
```

**Step 5: 运行测试确认通过**

```bash
cd frontend && npx vitest run timelineCanvasStyle
```
Expected: PASS（3 passed）。

**Step 6: 类型检查 + 提交**

```bash
cd frontend && npx tsc -b --noEmit && npx eslint src/components/layout/timeline/runtime --quiet
cd .. && git add frontend/src/components/layout/timeline/runtime
git commit -m "refactor(timeline): expose label widths from clip visual style"
```

---

### Task 2: `clipHeaderControls` 控件命中（纯函数 + TDD）

**为什么：** 内核需要把「header 内的局部坐标」映射为具体控件。本模块**直接消费 `style`**，与绘制端共享同一份偏移量。

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/interaction/clipHeaderControls.ts`
- Test: `frontend/src/components/layout/timeline/kernel/interaction/clipHeaderControls.test.ts`

**Step 1: 写失败的测试**

`clipHeaderControls.test.ts`：

```ts
import { describe, expect, it } from "vitest";
import { hitClipHeaderControl, type ClipHeaderControlStyle } from "./clipHeaderControls";

/**
 * 构造一个"全控件可见"的样式：与 buildTimelineClipVisualStyle 的偏移规则一致。
 * 旋钮圆心 (15,10) r=5；链 x=28；静音 x=50；共振峰 x=72；徽标均 20×14 @ y=3。
 */
function makeStyle(overrides: Partial<ClipHeaderControlStyle> = {}): ClipHeaderControlStyle {
    return {
        showMuteBadge: true,
        showChainBadge: true,
        showFormantBadge: true,
        showGainKnob: true,
        showGainLabel: true,
        showPlaybackRate: true,
        showName: true,
        muteBadgeWidth: 20,
        muteBadgeHeight: 14,
        muteBadgeOffsetX: 50,
        muteBadgeOffsetY: 3,
        chainBadgeWidth: 20,
        chainBadgeHeight: 14,
        chainBadgeOffsetX: 28,
        chainBadgeOffsetY: 3,
        formantBadgeWidth: 20,
        formantBadgeHeight: 14,
        formantBadgeOffsetX: 72,
        formantBadgeOffsetY: 3,
        gainKnobCenterOffsetX: 15,
        gainKnobCenterOffsetY: 10,
        gainKnobRadius: 5,
        leadingControlsWidth: 102,
        trailingReservePx: 60,
        gainLabelWidth: 40,
        rateLabelWidth: 16,
        ...overrides,
    };
}

describe("hitClipHeaderControl", () => {
    const CLIP_WIDTH = 300;

    it("命中静音徽标", () => {
        expect(hitClipHeaderControl({ localX: 60, localY: 10, clipWidthPx: CLIP_WIDTH, style: makeStyle() })).toBe("mute");
    });

    it("命中共振峰徽标", () => {
        expect(hitClipHeaderControl({ localX: 80, localY: 10, clipWidthPx: CLIP_WIDTH, style: makeStyle() })).toBe("formant");
    });

    it("命中链徽标", () => {
        expect(hitClipHeaderControl({ localX: 38, localY: 10, clipWidthPx: CLIP_WIDTH, style: makeStyle() })).toBe("chain");
    });

    it("命中增益旋钮（圆形判定，圆外不算）", () => {
        const style = makeStyle();
        expect(hitClipHeaderControl({ localX: 15, localY: 10, clipWidthPx: CLIP_WIDTH, style })).toBe("gain-knob");
        // 圆心右侧 6px：仍在 2px 容差外的圆外 → 不应命中旋钮
        expect(hitClipHeaderControl({ localX: 24, localY: 10, clipWidthPx: CLIP_WIDTH, style })).not.toBe("gain-knob");
    });

    it("命中增益标签（右对齐）", () => {
        // gainX = 300 - 40 - 6 = 254；中心取 274
        expect(hitClipHeaderControl({ localX: 274, localY: 9, clipWidthPx: CLIP_WIDTH, style: makeStyle() })).toBe("gain-label");
    });

    it("命中速率标签（在增益标签左侧）", () => {
        // rateX = 254 - 16 - 8 = 230；中心取 238
        expect(hitClipHeaderControl({ localX: 238, localY: 9, clipWidthPx: CLIP_WIDTH, style: makeStyle() })).toBe("rate-label");
    });

    it("命中名称区（左侧控件与右侧标签之间）", () => {
        expect(hitClipHeaderControl({ localX: 150, localY: 9, clipWidthPx: CLIP_WIDTH, style: makeStyle() })).toBe("name");
    });

    it("控件不可见时不命中（回落到名称区或 null）", () => {
        const style = makeStyle({ showMuteBadge: false, showGainLabel: false, showPlaybackRate: false });
        expect(hitClipHeaderControl({ localX: 60, localY: 10, clipWidthPx: CLIP_WIDTH, style })).not.toBe("mute");
        expect(hitClipHeaderControl({ localX: 274, localY: 9, clipWidthPx: CLIP_WIDTH, style })).not.toBe("gain-label");
    });

    it("clip 太窄时名称区退化为 null（不误判为名称）", () => {
        const style = makeStyle();
        expect(hitClipHeaderControl({ localX: 40, localY: 9, clipWidthPx: 80, style })).toBe(null);
    });

    it("纵向超出 header 高度时不命中任何控件", () => {
        expect(hitClipHeaderControl({ localX: 60, localY: 40, clipWidthPx: CLIP_WIDTH, style: makeStyle(), headerHeightPx: 16 })).toBe(null);
    });
});
```

**Step 2: 运行测试确认失败**

```bash
cd frontend && npx vitest run clipHeaderControls
```
Expected: FAIL —`Failed to resolve import "./clipHeaderControls"`。

**Step 3: 实现**

`clipHeaderControls.ts`：

```ts
/**
 * 时间轴渲染内核 · clip header 控件命中
 *
 * 【主要内容】
 * 把「clip header 内的局部坐标」映射为具体控件（静音 / 链 / 共振峰 / 增益旋钮 /
 * 增益标签 / 速率标签 / 名称区）。
 *
 * 【作用】
 * 内核模式下 clip 是自绘的，header 控件没有 DOM 可点。本模块让「看到的 = 可点的」
 * 成立：它**直接消费 `buildTimelineClipVisualStyle` 的返回值**（绘制端也用同一个
 * 函数），而不是另抄一套偏移量——两份位置常量迟早漂移。
 *
 * 【与其他模块的关系】
 * - 上游：宿主在 `hitTest` 判定为 `header` 分区后调用（见 `host/timelineKernelHost`）。
 * - 依赖：`ClipHeaderControlStyle` 是 `buildTimelineClipVisualStyle` 返回值的子集
 *   （结构化类型，无需导入运行时值，因此本模块保持纯函数、可直接单测）。
 * - 独立性：无 DOM / React 依赖。
 *
 * 【坐标系】
 * `localX` 以 **clip 左边缘**为原点，`localY` 以 **clip 顶边**为原点——与
 * `buildTimelineClipVisualStyle` 的偏移常量同一坐标系（渲染器里是
 * `clipLeft + style.xxxOffsetX`）。
 *
 * 【判定优先级】
 * 徽标（矩形）→ 增益旋钮（圆形）→ 速率标签 → 增益标签 → 名称区。
 * 右侧标签先判速率再判增益：两者相邻，速率在增益左侧，先判增益会把速率的
 * 左半边吞掉。
 */

/** header 内的可交互控件；null = 未命中任何控件。 */
export type ClipHeaderControl =
    | "mute"
    | "formant"
    | "chain"
    | "gain-knob"
    | "gain-label"
    | "rate-label"
    | "name"
    | null;

/** 控件命中所需的样式字段（`buildTimelineClipVisualStyle` 返回值的子集）。 */
export interface ClipHeaderControlStyle {
    readonly showMuteBadge: boolean;
    readonly showChainBadge: boolean;
    readonly showFormantBadge: boolean;
    readonly showGainKnob: boolean;
    readonly showGainLabel: boolean;
    readonly showPlaybackRate: boolean;
    readonly showName: boolean;
    readonly muteBadgeWidth: number;
    readonly muteBadgeHeight: number;
    readonly muteBadgeOffsetX: number;
    readonly muteBadgeOffsetY: number;
    readonly chainBadgeWidth: number;
    readonly chainBadgeHeight: number;
    readonly chainBadgeOffsetX: number;
    readonly chainBadgeOffsetY: number;
    readonly formantBadgeWidth: number;
    readonly formantBadgeHeight: number;
    readonly formantBadgeOffsetX: number;
    readonly formantBadgeOffsetY: number;
    readonly gainKnobCenterOffsetX: number;
    readonly gainKnobCenterOffsetY: number;
    readonly gainKnobRadius: number;
    readonly leadingControlsWidth: number;
    readonly trailingReservePx: number;
    readonly gainLabelWidth: number;
    readonly rateLabelWidth: number;
}

/** 命中参数。 */
export interface ClipHeaderControlArgs {
    /** clip 内相对 x（以 clip 左边缘为 0）。 */
    readonly localX: number;
    /** clip 内相对 y（以 clip 顶边为 0）。 */
    readonly localY: number;
    readonly clipWidthPx: number;
    readonly style: ClipHeaderControlStyle;
    /** header 高度（CSS px）：超出则不命中任何控件。缺省 16。 */
    readonly headerHeightPx?: number;
}

/** 徽标命中容差（CSS px）：比视觉边界各外扩一点，短按更容易点中。 */
const BADGE_SLOP_PX = 1;

/** 旋钮命中容差（CSS px）。 */
const KNOB_SLOP_PX = 2;

/** 标签命中区高度（以基线中心 ± 该值）。 */
const LABEL_HIT_HALF_HEIGHT_PX = 7;

/**
 * 判定点是否落在矩形内。
 *
 * @param x 点 x。@param y 点 y。
 * @param rx 矩形左上角 x。@param ry 矩形左上角 y。
 * @param rw 矩形宽。@param rh 矩形高。
 * @param slop 外扩容差。
 * @returns 命中为 true。
 */
function inRect(
    x: number,
    y: number,
    rx: number,
    ry: number,
    rw: number,
    rh: number,
    slop: number,
): boolean {
    return (
        x >= rx - slop && x <= rx + rw + slop && y >= ry - slop && y <= ry + rh + slop
    );
}

/**
 * 命中 clip header 内的控件。
 *
 * @param args 命中参数（局部坐标 + 样式）。
 * @returns 命中的控件；未命中为 null。
 */
export function hitClipHeaderControl(args: ClipHeaderControlArgs): ClipHeaderControl {
    const headerHeightPx = Number.isFinite(args.headerHeightPx)
        ? Math.max(0, args.headerHeightPx as number)
        : 16;
    // 纵向越界：落在 body 区，不属于任何 header 控件。
    if (args.localY < 0 || args.localY > headerHeightPx) return null;

    const style = args.style;
    const x = args.localX;
    const y = args.localY;
    const clipWidthPx = Math.max(1, args.clipWidthPx);

    // ── 左侧徽标（矩形）──
    if (
        style.showMuteBadge &&
        inRect(x, y, style.muteBadgeOffsetX, style.muteBadgeOffsetY, style.muteBadgeWidth, style.muteBadgeHeight, BADGE_SLOP_PX)
    ) {
        return "mute";
    }
    if (
        style.showFormantBadge &&
        inRect(x, y, style.formantBadgeOffsetX, style.formantBadgeOffsetY, style.formantBadgeWidth, style.formantBadgeHeight, BADGE_SLOP_PX)
    ) {
        return "formant";
    }
    if (
        style.showChainBadge &&
        inRect(x, y, style.chainBadgeOffsetX, style.chainBadgeOffsetY, style.chainBadgeWidth, style.chainBadgeHeight, BADGE_SLOP_PX)
    ) {
        return "chain";
    }

    // ── 增益旋钮（圆形）──
    if (style.showGainKnob) {
        const dx = x - style.gainKnobCenterOffsetX;
        const dy = y - style.gainKnobCenterOffsetY;
        const radius = Math.max(0, style.gainKnobRadius) + KNOB_SLOP_PX;
        if (dx * dx + dy * dy <= radius * radius) return "gain-knob";
    }

    // ── 右侧标签（右对齐，基线中心 y = 9）──
    // 先判速率（在增益左侧），否则增益会吞掉速率的左半边。
    if (style.showGainLabel) {
        const gainRight = clipWidthPx - 6;
        const gainLeft = gainRight - Math.max(0, style.gainLabelWidth);
        if (style.showPlaybackRate) {
            const rateRight = gainLeft - 8;
            const rateLeft = rateRight - Math.max(0, style.rateLabelWidth);
            if (
                x >= rateLeft - BADGE_SLOP_PX &&
                x <= rateRight + BADGE_SLOP_PX &&
                Math.abs(y - 9) <= LABEL_HIT_HALF_HEIGHT_PX
            ) {
                return "rate-label";
            }
        }
        if (
            x >= gainLeft - BADGE_SLOP_PX &&
            x <= gainRight + BADGE_SLOP_PX &&
            Math.abs(y - 9) <= LABEL_HIT_HALF_HEIGHT_PX
        ) {
            return "gain-label";
        }
    }

    // ── 名称区（左侧控件与右侧标签之间）──
    if (!style.showName) return null;
    const nameLeft = style.leadingControlsWidth;
    const nameRight = clipWidthPx - style.trailingReservePx + 4;
    // 可用宽度过窄时视为无名称区（绘制端同样要求 > 12px 才画）。
    if (nameRight - nameLeft <= 12) return null;
    if (x >= nameLeft && x <= nameRight) return "name";

    return null;
}
```

**Step 4: 运行测试确认通过**

```bash
cd frontend && npx vitest run clipHeaderControls
```
Expected: PASS（10 passed）。

**Step 5: 提交**

```bash
git add frontend/src/components/layout/timeline/kernel/interaction/clipHeaderControls.ts frontend/src/components/layout/timeline/kernel/interaction/clipHeaderControls.test.ts
git commit -m "feat(timeline-kernel): add clip header control hit testing"
```

---

### Task 3: `hitTest` 返回 clip 内局部坐标

**为什么：** 控件判定需要 clip 内相对坐标，而 `hitTest` 目前只返回 `region`。让它顺带返回 `localX` / `localY`，调用方无需重算（避免"两处各算一次"）。

**Files:**
- Modify: `frontend/src/components/layout/timeline/kernel/interaction/hitTest.ts`（`HitResult` 的 clip 分支 ~74-81、返回值 ~203）
- Test: `frontend/src/components/layout/timeline/kernel/interaction/hitTest.test.ts`（追加）

**Step 1: 写失败的测试**

在 `hitTest.test.ts` 追加：

```ts
describe("hitTest · 局部坐标", () => {
    it("命中 clip 时返回 clip 内相对坐标（原点 = clip 左上角）", () => {
        const result = hitTest({
            contentX: 120,
            contentY: 30,
            pxPerSec: 100,
            rowHeight: 80,
            tracks: [{ id: "t1" }],
            clipsByTrack: new Map([
                ["t1", [{ id: "c1", trackId: "t1", startSec: 1, lengthSec: 2 }]],
            ]),
            headerHeightPx: 16,
        });
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        // clip 左边缘 = 1s × 100 = 100px；顶边 = 第 0 行 = 0px
        expect(result.localX).toBeCloseTo(20, 5);
        expect(result.localY).toBeCloseTo(30, 5);
    });

    it("跨行命中时局部 y 以该行顶边为原点", () => {
        const result = hitTest({
            contentX: 120,
            contentY: 95,
            pxPerSec: 100,
            rowHeight: 80,
            tracks: [{ id: "t1" }, { id: "t2" }],
            clipsByTrack: new Map([
                ["t2", [{ id: "c2", trackId: "t2", startSec: 1, lengthSec: 2 }]],
            ]),
            headerHeightPx: 16,
        });
        expect(result.kind).toBe("clip");
        if (result.kind !== "clip") return;
        expect(result.localY).toBeCloseTo(15, 5);
    });
});
```

**Step 2: 运行测试确认失败**

```bash
cd frontend && npx vitest run hitTest
```
Expected: FAIL —`result.localX` 为 `undefined`。

**Step 3: 实现**

`HitResult` 的 clip 分支加两个字段：

```ts
    | {
          /** 命中 clip。 */
          readonly kind: "clip";
          readonly clip: HitTestClip;
          readonly region: ClipHitRegion;
          readonly sec: number;
          readonly trackIndex: number;
          /**
           * clip 内相对 x（以 clip **左边缘**为 0，CSS px）。
           *
           * 供 header 控件级命中复用（见 `clipHeaderControls`）：控件位置常量以
           * clip 左上角为原点，这里直接给出同一坐标系的值，调用方无需重算。
           */
          readonly localX: number;
          /** clip 内相对 y（以该 clip **顶边**为 0，CSS px）。 */
          readonly localY: number;
      };
```

返回值（~203 行）改为：

```ts
    return {
        kind: "clip",
        clip,
        region,
        sec,
        trackIndex,
        localX: args.contentX - clipLeftPx,
        localY,
    };
```

**Step 4: 运行测试确认通过**

```bash
cd frontend && npx vitest run hitTest
```
Expected: PASS（10 passed，含原有 8 条）。

**Step 5: 提交**

```bash
git add frontend/src/components/layout/timeline/kernel/interaction/hitTest.ts frontend/src/components/layout/timeline/kernel/interaction/hitTest.test.ts
git commit -m "feat(timeline-kernel): return clip-local coordinates from hit test"
```

---

### Task 4: 宿主分派 header 控件 → interactions 回调

**为什么：** 内核只做命中与手势，编辑语义交回面板（既有架构约定，见 `TimelineKernelInteractions` 注释）。

**Files:**
- Modify: `frontend/src/components/layout/timeline/kernel/host/timelineKernelHost.ts`
  - 新增 import（`buildTimelineClipVisualStyle`、`hitClipHeaderControl`、`resolveFontFamily` 已有）
  - `TimelineKernelInteractions` 接口（~239 行起）
  - `startPrimaryGesture`（~1669 行）
  - 宿主需要读取 clip 的业务字段 → `rebuildHitIndex` 的 `HitTestClip` 扩展

**Step 1: 扩展命中索引，携带控件判定所需的 clip 字段**

`hitTest.ts` 的 `HitTestClip` 加可选业务字段（保持 `hitTest` 自身不消费它们，只做透传）：

```ts
/** 命中测试所需的 clip 最小字段集。 */
export interface HitTestClip {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    /**
     * 以下字段 `hitTest` 本身**不消费**，只做透传给调用方（header 控件级命中
     * 需要它们构造 `buildTimelineClipVisualStyle`）。保持可选：现有测试与
     * 只关心几何的调用方无需补齐。
     */
    readonly muted?: boolean;
    readonly gain?: number;
    readonly playbackRate?: number;
    readonly name?: string;
    readonly groupId?: string;
    readonly isGroupActive?: boolean;
    readonly isGroupDisabled?: boolean;
    readonly isPitchAdjustment?: boolean;
}
```

`host` 的 `rebuildHitIndex()` 补齐这些字段（从 `ClipInfo` 读；`isGroupActive` / `isGroupDisabled` 由宿主的 `data()` 提供——若 `TimelineKernelData` 尚无这两个集合，加 `readonly activeGroupIds: readonly string[]` 与 `readonly disabledGroupIds: readonly string[]`，由 `TimelineKernelView` 从 Redux 传入）。

**Step 2: 接口新增回调**

`TimelineKernelInteractions` 追加：

```ts
    /**
     * 切换 clip 静音（单击 header 的静音徽标）。
     *
     * @param clipId 目标 clip。
     * @param nextMuted 目标状态（取反后的值，与旧实现一致——旧实现传 `!clip.muted`）。
     */
    readonly onToggleClipMute?: (clipId: string, nextMuted: boolean) => void;
    /**
     * 打开共振峰工具窗口（单击 header 的 F 徽标）。
     *
     * @param clipId 目标 clip。
     * @param screenX 浮窗锚点的视口坐标（旧实现取按钮右缘 +12）。
     * @param screenY 浮窗锚点的视口坐标（旧实现取按钮上缘）。
     */
    readonly onOpenClipFormant?: (clipId: string, screenX: number, screenY: number) => void;
    /**
     * 开始行内编辑 clip 的增益 / 速率（单击对应标签或旋钮）。
     *
     * @param clipId 目标 clip。
     * @param field 编辑字段。
     * @param screenX 输入框锚点的视口坐标。
     * @param screenY 输入框锚点的视口坐标。
     */
    readonly onBadgeEditStart?: (
        clipId: string,
        field: "gain" | "rate",
        screenX: number,
        screenY: number,
    ) => void;
    /**
     * 打开速率高级编辑（**右键**速率标签）。
     */
    readonly onRateBadgeMenu?: (clipId: string, screenX: number, screenY: number) => void;
    /**
     * 请求进入 clip 重命名（双击名称区）。
     *
     * 与 `onDoubleClickClip`（参数选区）互斥：名称区优先。
     */
    readonly onRenameClipStart?: (clipId: string, screenX: number, screenY: number) => void;
```

**Step 3: 在 `startPrimaryGesture` 的 clip 分支里分派**

在既有的双击判定**之后**、`fadeSide` 计算**之前**插入：

```ts
            // ── header 控件级分派 ──
            // 双击判定优先于单击控件（旧实现里名称区双击进入重命名，其他区域双击
            // 进入参数选区），因此本段位于双击分支之后。
            if (hit.region === "header") {
                const clipInfo = data().clips.find((item) => item.id === hit.clip.id);
                if (clipInfo !== undefined) {
                    const style = buildTimelineClipVisualStyle({
                        widthPx: Math.max(1, hit.clip.lengthSec * scroll.get().pxPerSec),
                        trackColor: clipInfo.color ?? undefined,
                        selected:
                            data().selectedClipId === clipInfo.id ||
                            data().multiSelectedClipIds.includes(clipInfo.id),
                        muted: clipInfo.muted === true,
                        gain: clipInfo.gainDb,
                        playbackRate: clipInfo.playbackRate,
                        name: clipInfo.name,
                        fontFamily: resolveFontFamily(),
                        isPitchAdjustment: clipInfo.isPitchAdjustment,
                        groupId: clipInfo.groupId ?? undefined,
                        isGroupActive: data().activeGroupIds.includes(clipInfo.groupId ?? ""),
                        isGroupDisabled: data().disabledGroupIds.includes(clipInfo.groupId ?? ""),
                        darkMode: data().darkMode,
                    });
                    const control = hitClipHeaderControl({
                        localX: hit.localX,
                        localY: hit.localY,
                        clipWidthPx: Math.max(1, hit.clip.lengthSec * scroll.get().pxPerSec),
                        style,
                        headerHeightPx: CLIP_HEADER_HEIGHT,
                    });
                    if (control === "mute") {
                        interactions?.onToggleClipMute?.(clipInfo.id, clipInfo.muted !== true);
                        return;
                    }
                    if (control === "formant") {
                        // 锚点与旧实现一致：按钮右缘 +12、上缘。
                        const rect = container.getBoundingClientRect();
                        const badgeScreenX =
                            rect.left + hit.localX + style.formantBadgeWidth + 12;
                        const badgeScreenY = rect.top + hit.localY - style.formantBadgeOffsetY;
                        interactions?.onOpenClipFormant?.(clipInfo.id, badgeScreenX, badgeScreenY);
                        return;
                    }
                    if (control === "gain-knob" || control === "gain-label") {
                        const rect = container.getBoundingClientRect();
                        interactions?.onBadgeEditStart?.(
                            clipInfo.id,
                            "gain",
                            rect.left + hit.localX,
                            rect.top + hit.localY,
                        );
                        return;
                    }
                    if (control === "rate-label") {
                        const rect = container.getBoundingClientRect();
                        interactions?.onBadgeEditStart?.(
                            clipInfo.id,
                            "rate",
                            rect.left + hit.localX,
                            rect.top + hit.localY,
                        );
                        return;
                    }
                    if (control === "name") {
                        // 名称区单击仍走选中 / 拖拽；只有双击才重命名（见双击分支）。
                        // 这里不 return，交给下方既有流程。
                    }
                    // 注意：`chain` 控件本期不接（旧实现由分组逻辑处理，需单独调研）。
                }
            }
```

**Step 4: 名称区双击 → 重命名（优先级高于参数选区）**

既有的双击分支目前无条件派发 `onDoubleClickClip`。改为：若双击落在名称区，则派发 `onRenameClipStart`：

```ts
            if (isDoubleClick) {
                // 名称区双击 → 重命名；其他区域双击 → 参数编辑器选区（旧实现里
                // 名称区的处理器会 stopPropagation，因此两者天然互斥）。
                const style = buildTimelineClipVisualStyle({ /* 同上 */ });
                const control = hitClipHeaderControl({
                    localX: hit.localX,
                    localY: hit.localY,
                    clipWidthPx: /* 同上 */,
                    style,
                    headerHeightPx: CLIP_HEADER_HEIGHT,
                });
                const rect = container.getBoundingClientRect();
                if (control === "name") {
                    interactions?.onRenameClipStart?.(
                        hit.clip.id,
                        rect.left + hit.localX,
                        rect.top + hit.localY,
                    );
                } else {
                    interactions?.onDoubleClickClip?.(hit.clip.id);
                }
                return;
            }
```

> 为避免重复构造，把上面这段抽成宿主内的局部函数 `resolveHeaderControl(hit): ClipHeaderControl`，两处共用。

**Step 5: 右键速率标签 → 高级编辑**

`onContextMenu` 的派发处（既有 `contextmenu` 处理）在 `clipsAtPointer` 之外，补一段：若指针落在 header 的 `rate-label` 控件上，优先派发 `onRateBadgeMenu`。

**Step 6: `TimelineKernelView` 透传新回调**

在 `stableInteractions` 里按既有模式补：

```ts
            onToggleClipMute: (clipId, nextMuted) =>
                interactionsRef.current?.onToggleClipMute?.(clipId, nextMuted),
            onOpenClipFormant: (clipId, screenX, screenY) =>
                interactionsRef.current?.onOpenClipFormant?.(clipId, screenX, screenY),
            onBadgeEditStart: (clipId, field, screenX, screenY) =>
                interactionsRef.current?.onBadgeEditStart?.(clipId, field, screenX, screenY),
            onRateBadgeMenu: (clipId, screenX, screenY) =>
                interactionsRef.current?.onRateBadgeMenu?.(clipId, screenX, screenY),
            onRenameClipStart: (clipId, screenX, screenY) =>
                interactionsRef.current?.onRenameClipStart?.(clipId, screenX, screenY),
```

**Step 7: 类型检查 + 提交**

```bash
cd frontend && npx tsc -b --noEmit && npx eslint src/components/layout/timeline/kernel --quiet
cd .. && git add frontend/src/components/layout/timeline/kernel
git commit -m "feat(timeline-kernel): dispatch clip header control interactions"
```

---

### Task 5: 面板接回调（静音 / 共振峰 / 速率菜单 —— 立即可用）

**Files:**
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（`kernelInteractions` ~1542、`handleKernelDoubleClickClip` 附近）

**Step 1: 实现三个回调**

```ts
    /** 内核静音：与旧实现同源（`toggleTrackLaneClipMuted` 内部处理乐观更新与提交）。 */
    const handleKernelToggleClipMute = React.useCallback(
        (clipId: string, nextMuted: boolean) => {
            void toggleTrackLaneClipMuted(clipId, nextMuted);
        },
        [toggleTrackLaneClipMuted],
    );

    /** 内核共振峰：锚点用内核给的视口坐标（旧实现取按钮右缘 +12 / 上缘）。 */
    const handleKernelOpenClipFormant = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            dispatch(openClipFormantToolWindow({ clipId, anchor: { x: screenX, y: screenY } }));
        },
        [dispatch],
    );

    /** 内核速率高级编辑：复用既有 `ClipRateEditorDialog`（已在开关之外，仅缺入口）。 */
    const handleKernelRateBadgeMenu = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            setRateEditorClipId(clipId);
            setRateEditorPosition({ x: screenX, y: screenY });
        },
        [],
    );
```

`kernelInteractions` 里补上这三项。`openClipFormantToolWindow` 需加进 sessionSlice 的 import 列表。

**Step 2: 浏览器验证（三个入口）**

```bash
cd frontend && VW=1920 VH=1200 KERNEL=1 node scripts/dev-shot.mjs "http://localhost:5173/?mock=1" /tmp/hfs-hdr1.png 5500 '[{"type":"eval","js":"const k=document.querySelector(\"[data-hs-timeline-kernel]\"); const r=k.getBoundingClientRect(); return {r:{x:r.x,y:r.y}}"},{"type":"click","x":430,"y":76},{"type":"wait","ms":400},{"type":"eval","js":"return {formantOpen: !!document.querySelector(\"[role=dialog]\") || document.body.innerHTML.includes(\"共振峰\")}"}]' 2>&1 | head -6
```

（坐标需按实际渲染微调：静音徽标约在 clip 左缘 +50 ~ +70、y = 行顶 +3 ~ +17；共振峰徽标再 +22。）

Expected: 点击静音徽标后 mock 收到 `set_clips_state` 类调用；点击 F 徽标后共振峰浮窗出现。

**Step 3: 提交**

```bash
git add frontend/src/components/layout/TimelinePanel.tsx
git commit -m "feat(timeline-kernel): wire mute/formant/rate-menu entries"
```

---

### Task 6: 内核态行内编辑浮层（重命名 / 增益 / 速率）

**为什么：** 旧实现的行内输入框在 `ClipHeader`（DOM）内，内核模式下不挂载。需要新建一个浮层组件，按内核视口把输入框定位到 clip 上。

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/KernelClipInlineEditor.tsx`
- Modify: `frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx`（渲染浮层 + 提供视口换算）
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（提供编辑状态与提交回调）

**Step 1: 设计（先写测试再实现）**

浮层职责：
- 接收 `{ clipId, field: "name" | "gain" | "rate", onCommit, onCancel }`
- 自己从内核宿主读视口（`getViewport()`）换算屏幕位置：`screenX = rect.left + clipLeftContent - scrollLeft`，`screenY = rect.top + rowTop - scrollTop`
- 渲染一个 `<input>`，`Enter` 提交 / `Esc` 取消 / `blur` 提交
- 定位随滚动更新：订阅内核视口（复用 `registerViewportLayer` 或直接在 rAF 里写 style）——**推荐后者**（浮层生命周期短，命令式写入足够）

**关键：不要在 React 里每帧更新位置**（会让输入框在滚动时抖动/失焦）。

**Step 2: 实现**

组件骨架：

```tsx
/**
 * 内核态 clip 行内编辑浮层（重命名 / 增益 / 速率）。
 *
 * 【为什么需要】
 * 旧实现的行内输入框在 `ClipHeader`（DOM）内，内核模式下 clip 是自绘的、
 * `ClipHeader` 不挂载，因此必须有一个独立的输入浮层。
 *
 * 【定位策略】
 * 输入框是**视口坐标**的绝对定位元素，位置由内核宿主在 rAF 内命令式写入
 * （内容坐标 − 滚动量）。用 React state 每帧更新位置会让输入框在滚动时抖动、
 * 甚至因重渲染丢失焦点。
 */
```

props：

```tsx
export interface KernelClipInlineEditorProps {
    /** 目标 clip（用于算初始位置与提交）。 */
    readonly clipId: string;
    readonly field: "name" | "gain" | "rate";
    /** 初始值。 */
    readonly initialValue: string;
    /** 输入框宽度（CSS px）。 */
    readonly widthPx: number;
    /** 内容坐标下的位置（clip 左缘 / 行顶 + 偏移）。 */
    readonly contentLeftPx: number;
    readonly contentTopPx: number;
    readonly onCommit: (value: string) => void;
    readonly onCancel: () => void;
}
```

**Step 3: 面板侧接入**

- `handleKernelRenameClipStart(clipId)` → `clipActions.setRenamingClipId(clipId)` + 记录锚点
- `handleKernelBadgeEditStart(clipId, field)` → `setEditingBadge({ clipId, field })` + 记录锚点
- 提交走既有的 `commitTrackLaneRename` / `commitTrackLaneBadgeEdit`

**Step 4: 浏览器验证**

双击名称区 → 出现输入框 → 输入新名 → Enter → mock 收到重命名调用。

**Step 5: 提交**

```bash
git add frontend/src/components/layout/timeline/kernel/KernelClipInlineEditor.tsx frontend/src/components/layout/TimelinePanel.tsx frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx
git commit -m "feat(timeline-kernel): add inline editor for rename and gain/rate badges"
```

---

### Task 7: 回归与收尾

**Step 1: 全量测试**

```bash
cd frontend && npx vitest run
```
Expected: `470 + 新增条数 passed`，仅 2 条 keybindings 预先存在的失败。

**Step 2: lint / 类型 / 构建**

```bash
cd frontend && npx tsc -b --noEmit && npx eslint src/components/layout --quiet && npx prettier --check src/components/layout/timeline/kernel
```

**Step 3: 浏览器逐项验证（旧实现对照）**

对每一项，分别用 `KERNEL=0` 与 `KERNEL=1` 截图/操作，确认行为一致：
- 单击静音徽标 → 徽标变色 + mock 收到状态写入
- 单击 F 徽标 → 共振峰浮窗出现
- 右键速率标签 → `ClipRateEditorDialog` 出现
- 双击名称区 → 输入框出现，Enter 提交
- 双击 clip 其他区域 → 参数编辑器选区（回归，不应被重命名抢走）
- 单击 header 空白（名称区）→ 仍然是选中 / 拖拽，不误触发编辑

**Step 4: 提交 + 更新记忆**

```bash
git add -A frontend/src/components/layout && git commit -m "test(timeline-kernel): verify clip header interactions in browser"
```

---

## Phase B / C / D：后续计划（需先调研，本计划不执行）

以下三项各自需要先做**只读调研**（旧实现的具体几何与状态机），调研结论写进新计划后再实施。本计划不臆测其实现细节。

### Phase B：`OverlapEditLayer` 等价物 —— 调研结论（2026-09-11）

**旧实现模型**（`OverlapEditLayer.tsx`，814 行，DOM 层 `z-[200]`，在 `TrackLane` 内挂载）：
它在每一对**重叠 clip** 之间按**位置**提供双方的控件，用一个独立的 DOM 层解决
「同一段屏幕空间同时属于两个 clip」的层叠问题。对每对 (earlier, later)：

| 区域 | 控件 | 归属 | 几何 |
|---|---|---|---|
| 重叠区**左缘** | 左边缘（trim/stretch） | later | 10px 宽、**整行高** |
| 重叠区**右缘** | 右边缘（trim/stretch） | earlier | 10px 宽、**整行高** |
| 重叠区内 later 的淡入部分 | 包络线小块 + 区域右缘竖条 | later | `buildFadeHitTargets`，裁剪到重叠区 |
| 重叠区内 earlier 的淡出部分 | 包络线小块 + 区域左缘竖条 | earlier | 同上 |
| later 的 snap offset 三角 | 吸附偏移手柄 | later | `snapOffsetHandleXPx` + 9×12 |
| 两条包络线交点 | 交叉点抓手（16px） | 双方 | `computeCrossfadeGripPoint`（二分求真实曲线交点） |

**判定优先级**（由 DOM 层叠顺序保证，**后 push 的在上**，因此内核要反序判定）：
clip 边缘（最优先）→ 淡化包络线 / 边缘竖线 → 交叉点抓手 / snap offset。

**淡变有效长度**：`autoFadeInSec > 0 ? autoFadeInSec : fadeInSec`（自动交叉淡化覆盖手动）。

**可直接复用的纯函数**（已确认）：
- `buildFadeHitTargets`（`fadeHitTargets.ts`）：输入 clip 几何 + 淡变参数，输出
  包络线小块（12px，沿弧长采样）与区域边缘竖条（6px），支持 `clipXFrom/clipXTo`
  裁剪到重叠区。**几何与绘制端 `drawFadeCurveStroke` 完全一致**。
- `computeCrossfadeGripPoint` / `fadeGainSigned`（`reaperFade.ts`）
- `snapOffsetHandleXPx`（`constants.ts`）

**内核的实际缺口（关键结论）**：内核的 `hitTest` 用二分取「最后一个 `startSec <= sec`」，
在重叠区里**永远是 later**。因此：
- later 的左边缘**可达**（它在重叠区左缘）✓
- **earlier 的右边缘与淡出控件完全不可达** ✗ ← 这是本阶段要解决的核心问题

**实施增量**（按价值排序）：
- **B-1**：`kernel/interaction/overlapControls.ts` —— 把「按位置解析」搬进纯函数，
  内核在通用命中之后用它改写命中结果（把 earlier 的右缘 / 淡出还给用户）
- **B-2**：淡变包络线 / 边缘竖线的拖拽（复用 `buildFadeHitTargets`，映射到既有
  `clip-fade` 手势）
- **B-3**：交叉点抓手（需新的「同时移动双方边缘」手势）
- **B-4**：淡变形状循环点击（Ctrl/可配置修饰键 + 单击）与双击重置曲率

### Phase C：snap offset 三角手柄拖拽 —— 调研结论（2026-09-11）

**几何（单一事实来源：`ClipItem.tsx` 的命中握把 + `constants.ts`）**

`SnapOffset` 是 clip 固有属性（秒，相对 clip 起点，与倒放无关）。手柄命中区是一个
**贴行底的透明矩形**（三角视觉由轨道级 Canvas 绘制）：

| 项 | 值 |
|---|---|
| 锚点 | `absolute bottom-0` → clip 局部 `y ∈ [clipHeight − 12, clipHeight]` |
| 高度 | `SNAP_OFFSET_HIT_HEIGHT_PX = 12` |
| 左缘 | `min(max(−4, snapOffsetHandleXPx(offset, pxPerSec) − 1), max(−4, width − 9))` |
| 宽度 | `SNAP_OFFSET_HANDLE_SIZE_PX + 3 = 12` |
| 三角 x | `snapOffsetHandleXPx(offset, pxPerSec) = offset > 0 ? offset × pxPerSec : 0`（**不钳制**，越界由绘制端裁剪） |

**优先级（旧实现由 z-index 决定，内核必须显式排序）**

`z-70` snap 手柄 > `z-65` 淡变角（横帽 22×14 @ `y=CLIP_HEADER_HEIGHT`；竖条 6px 宽，
下沿到 `fadeCornerReservePx(bodyH) = max(14, round(bodyH/3))`）> `z-60` 左右边缘
（全高）。→ **内核里 snap 手柄必须是最高优先级的 clip 分区**，在淡变角之前判定。

**状态机（`hooks/useSnapOffsetDrag.ts`）**

1. 按下：仅左键；`dispatch(beginInteraction())`；记 `baseOffset = clamp(snapOffsetSec, 0, clipLen)`、
   `clipStart`、`clipLen`、`anchorTrackId = clip.trackId`、`startPointerSec`
2. 位移阈值 **2px**（水平）后才 `dispatch(checkpointHistory())` —— **零位移单击不产生 undo 步**
3. 拖拽中：`rawAbs = clipStart + baseOffset + (pointerSec − startPointerSec)`；
   吸附开启时 `snapTimelineDetailed(rawAbs, "clip", { originSec: clipStart + baseOffset,
   anchorTrackId, excludeClipIds: {clipId}, highlight: { sources: [{trackId, clipId}] } })`
   —— 即**手柄的绝对时间线位置**作为被吸附对象（单点吸附），高亮 = 手柄所在行的亮条；
   吸附关闭时 `clearSnapHighlights`。落库前 `clamp(next, 0, clipLen)`
4. 收尾：清高亮；未越阈值 → 只 `endInteraction()`（**不写后端**）；否则
   `setClipStateRemote({ clipId, snapOffsetSec, checkpoint: true })` → `endInteraction()`
5. 全程 `beginSnapGesture()` / `endSnapGesture()` 包裹；失焦经 `registerDragAbort` 收尾

**内核实施增量**
- **C-1**：`hitTest` 新增 `snap-offset-handle` 分区（最高优先级）+ `HitTestClip` 补
  `snapOffsetSec`；需要 `pxPerSec` / `clipHeightPx`（后者已在 `hitTest` 内算出）
- **C-2**：宿主新增 `snap-offset-drag` 手势 + `onSnapOffsetPreview` / `onSnapOffsetCommit`
  回调（**只给几何位移，吸附与落库交回面板**——与 `onDragPreview` 同一架构约定）
- **C-3**：面板接回调，复用 `useSnapOffsetDrag` 的语义（阈值 / 吸附 / 单笔后端写入）

### Phase D：ghost 预览（copy 拖拽）与素材拖入 —— 调研结论（2026-09-11）

**D-1：ghost 预览（copy 拖拽）**

| 项 | 事实 |
|---|---|
| 触发 | `resolveClipDragCopyMode({ existingCopyMode, ctrlKey, modifierActive })`（`hooks/clipDragCopyMode.ts`）：**已配置的复制绑定为准**，非 macOS 额外保留 Ctrl 回退（macOS 上 ctrl 字段映射到 Command） |
| 状态 | `ghostDrag` 在 `useClipDrag.ts`（`useState`，550 行 set、645/663 行清）；只在 `copyMode` 时更新——**原 clip 不动，只更新 ghost 位置** |
| 字段 | `{ deltaSec, targetTrackId, targetTrackOffset, allowTrackMove, clipIds, initialById }`，带去重（同值不重复 setState） |
| 渲染 | 作为 `TrackLane` 的 prop 传入 → **位于内核开关的旧分支内**，内核模式下不渲染 |
| 几何 | `startSec = max(0, initialById[clipId].startSec + deltaSec)`（**内容坐标**） |

**D-2：素材拖入（`dropPreview` + drag&drop）**

| 项 | 事实 |
|---|---|
| 渲染 | `TimelinePanel` 旧分支内（`TrackLane` 里），`dropPreview` 来自 hook，`dropPreviewRef` 供命令式写位置 |
| 几何 | `left = max(0, dropPreview.startSec × pxPerSec)`、`top = rowTopForTrackId(dropPreview.trackId) + 8` → **内容坐标** |
| 尺寸 | 宽度 `pxPerSec × dropPreview.durationSec`（`durationSec > 0` 时） |
| 事件 | `onDragOver` / `onDrop` 挂在 **`TimelineScrollArea`** 上（旧分支内）；用 `tauriDraggedPathRef` 处理 Tauri 文件拖入，并区分 `dataTransfer.files` |
| 分支 | 落点后经 `importModeMenu` 分支（导入模式选择菜单） |

**内核实施增量（按依赖排序）**

- **D-1**：内核态 ghost 层。与 `SnapHighlightLayer` **同一模式**——内容坐标容器 +
  宿主 rAF 整层 `translate(-scrollLeft, -scrollTop)`（宿主已有 `snapHighlightContent`
  这条通道，可复用同一机制再挂一层）。手势侧：`clip-drag` 增加 copyMode 判定
  （复用 `resolveClipDragCopyMode`，宿主需读修饰键状态），copyMode 下不发
  `onDragPreview` 的"移动"语义而是"ghost 位置"语义。
- **D-2**：拖入处理迁到内核视口。`dropPreview` 同样是内容坐标 → 同一容器模式；
  `onDragOver`/`onDrop` 需挂到内核容器上（旧实现挂在 `TimelineScrollArea`，
  内核模式下该组件不挂载）。**注意**：`importModeMenu` 分支与 Tauri 路径要一并迁移，
  否则拖入会静默失效。
- **D-3**：`rowTopForTrackId` 这类"轨道 → 行顶"的换算，内核已有同源实现
  （`trackIndex × rowHeight`），应复用而不是另写。

**待确认（实施前需再读一次代码）**：`importModeMenu` 的完整分支条件、Tauri 拖入与
DOM 拖入的差异处理、`dropPreview` 的 duration 来源（是否已在拖入时解析音频头）。

---

#### ⚠️ D-1 实施前发现的计划缺口（2026-09-11，需先决策）

**结论：D-1 的"落库"不是一次 thunk 调用，而是一段约 40 行的编排。**
（`useClipDrag.ts:846-885`）

```
initialById → targetTrackIdByClipId → trackMapping
  → trackMode（same_track / explicit_mapping）
  → buildDuplicateClipsBulkPayload({ sourceClipIds, deltaSec, copyLinkedParams,
                                     applyAutoCrossfade, trackMode, renameCopies })
  → duplicateClipsBulkRemote(...) → createdClipIds
  → setMultiSelectedClipIds(created) + selectClipRemote(created[0])
  → 播放光标定位到副本中最靠前的起点
```

**问题**：这段逻辑目前**只存在于 `useClipDrag` 内部**（事件驱动、依赖 `drag` 局部状态）。
内核的 `clip-drag` 手势走 `onDragPreview` / `onDragCommit`，拿不到 `drag`，因此要么：

- **方案 A（抽取共享函数）**：把 846-885 抽成
  `copyClipsFromDrag({ sourceClipIds, initialById, deltaSec, targetTrackIdByClipId, ... })`，
  旧 hook 与内核面板都调用它。**优点**：单一事实来源，行为天然一致（这正是本次迁移
  反复强调的原则）。**代价**：要动旧实现的收尾路径，需一次回归验证。
- **方案 B（内核侧重实现）**：在面板的 `handleKernelDragCommit` 里按同样步骤再写一遍。
  **优点**：不碰旧实现。**代价**：**两份复制语义**——正是本计划一直在避免的模式
  （见 `snapTimelineDetailed` 的 `highlight` 漏传、`moveSnapOffsetSec: 0` 两次教训）。

**建议方案 A**（与既有迁移原则一致）。**决策后再实施 D-1。**

**D-1 还需补的前置**：宿主 `clip-drag` 需携带 `copyMode`（复用
`resolveClipDragCopyMode`，含"拖拽中允许从 false 变 true、不允许反向"的既有语义），
并在 `onDragPreview` / `onDragCommit` 上透出——否则内核模式下 ⌘+拖拽会**移动**原 clip
（用户预期是复制），属于**数据语义错误**，不只是缺视觉。

---

## 执行顺序与提交粒度

| 顺序 | 任务 | 产出 | 提交 |
|---|---|---|---|
| 1 | Task 1 样式补宽度 | `gainLabelWidth` / `rateLabelWidth` | 1 |
| 2 | Task 2 控件命中 | `clipHeaderControls.ts` + 10 条测试 | 1 |
| 3 | Task 3 局部坐标 | `hitTest` 返回 `localX/localY` + 2 条测试 | 1 |
| 4 | Task 4 host 分派 | 5 个新回调 | 1 |
| 5 | Task 5 简单回调 | 静音 / F / 速率菜单可用 | 1 |
| 6 | Task 6 行内编辑浮层 | 重命名 / 增益 / 速率可用 | 1 |
| 7 | Task 7 回归收尾 | 全绿 + 浏览器验证 | 1 |

**每完成一个 Task 都必须：** 跑该 Task 的测试 → `tsc -b --noEmit` → 提交 → 再进入下一个。
