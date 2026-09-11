# 时间轴内核 · 阶段 1：clip 拖拽移动 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让渲染内核接管轨道区后，用户仍能拖拽 clip 改变其时间位置与所属轨道，行为与旧实现一致。

**Architecture:** 内核只负责「命中 + 手势 + ghost 预览」，把拖拽意图（clipId / deltaSec / targetTrackId）在 `pointerup` 时回调给 React 侧；React 侧复用既有 `snapTimelineDetailed`（吸附）、`trackIdFromClientY`（跨轨道）、`moveClipStart` / `moveClipTrack`（乐观更新）与 `moveClipsRemote`（提交）——**不重写编辑语义**，只替换「手势来源」。

**Tech Stack:** TypeScript / React 19 / WebGL2 / Redux Toolkit / Vitest

**前置状态（已实现，勿重复造）:**
- `kernel/interaction/hitTest.ts`：`hitTest(args)` 返回 `{ kind: "clip", clip, region, sec, trackIndex }` 或 `{ kind: "empty", sec, trackId, trackIndex }`
- `kernel/host/timelineKernelHost.ts`：左键手势状态机已有 `pending-select`（4px 阈值区分点击/拖拽）与 `seek`；命中索引 `hitClipsByTrack` / `hitTracks`；`TimelineKernelInteractions` 已有 `onSeek` / `onSelectClip`
- `kernel/scene/clipInstances.ts`：`createClipInstanceBuilder().build({ clips, darkMode, ... })` 产出实例缓冲
- 宿主几何重建入口：`rebuildInstances(axis)`（内部用 `buildSparseClipRenderModel` 取 `model.drawClips`）

**范围界定（本期不做，列为后续任务）:**
- trim（裁短/延长）、fade 角拖拽
- 拖到空白处新建轨道（`allowDropToNewTrack`）
- 复制拖动（copyMode）、波纹（ripple）、自动交叉淡化调整

**为什么先做这一条链路:** 它是唯一还缺的核心编辑能力；先端到端打通「内核手势 → ghost → Redux 提交 → 视觉回读」，其余手势（trim/fade/框选）可复用同一套骨架。

---

## Task 1: 拖拽几何换算（纯函数 + 单测）

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/interaction/dragGeometry.ts`
- Test: `frontend/src/components/layout/timeline/kernel/interaction/dragGeometry.test.ts`

**职责:** 把「指针的内容坐标位移」换算为「时间位移 + 目标轨道下标」，并做边界钳制。**不含吸附**——吸附由 React 侧 `snapTimelineDetailed` 统一处理，避免两处吸附规则分叉。

**Step 1: 写失败测试**

```ts
import { describe, expect, it } from "vitest";
import { resolveDragDelta } from "./dragGeometry";

describe("resolveDragDelta", () => {
    it("按 pxPerSec 把水平位移换算为秒", () => {
        const out = resolveDragDelta({
            deltaContentXPx: 150,
            pxPerSec: 150,
            startSec: 10,
            lengthSec: 4,
            projectSec: 60,
        });
        expect(out.deltaSec).toBe(1);
        expect(out.startSec).toBe(11);
    });

    it("左移越界时钳制到 0", () => {
        const out = resolveDragDelta({
            deltaContentXPx: -3000,
            pxPerSec: 150,
            startSec: 1,
            lengthSec: 4,
            projectSec: 60,
        });
        expect(out.startSec).toBe(0);
        expect(out.deltaSec).toBe(-1);
    });

    it("右移越界时钳制到工程末端（clip 不越出工程长度）", () => {
        const out = resolveDragDelta({
            deltaContentXPx: 100000,
            pxPerSec: 150,
            startSec: 10,
            lengthSec: 4,
            projectSec: 60,
        });
        expect(out.startSec).toBe(56); // 60 - 4
    });

    it("pxPerSec 非法时不产生 NaN", () => {
        const out = resolveDragDelta({
            deltaContentXPx: 100,
            pxPerSec: 0,
            startSec: 5,
            lengthSec: 2,
            projectSec: 30,
        });
        expect(Number.isFinite(out.startSec)).toBe(true);
        expect(out.startSec).toBe(5);
    });
});
```

**Step 2: 运行确认失败**

Run: `cd frontend && npx vitest run dragGeometry`
Expected: FAIL —— `Failed to resolve import "./dragGeometry"`

**Step 3: 实现**

```ts
/**
 * 时间轴渲染内核 · 拖拽几何换算
 *
 * 【主要内容】把指针的内容坐标位移换算为 clip 的新起始时间，并钳制到工程范围。
 *
 * 【作用】内核手势只负责"几何"，编辑语义（吸附 / 事务 / 提交）留在 React 侧——
 * 两处都做吸附会让规则分叉（旧实现的吸附点在 `snapTimelineDetailed`）。
 *
 * 【设计约束】半开区间与命中测试一致；钳制在**写入时**一次算清（同 ScrollKernel 的约定）。
 */
export interface DragDeltaArgs {
    readonly deltaContentXPx: number;
    readonly pxPerSec: number;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly projectSec: number;
}

export interface DragDeltaResult {
    /** 钳制后的新起始时间（秒）。 */
    readonly startSec: number;
    /** 实际生效的时间位移（秒，已含钳制）。 */
    readonly deltaSec: number;
}

export function resolveDragDelta(args: DragDeltaArgs): DragDeltaResult {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    const rawDelta = pxPerSec > 0 ? args.deltaContentXPx / pxPerSec : 0;
    const length = Math.max(0, args.lengthSec);
    const maxStart = Math.max(0, args.projectSec - length);
    const startSec = Math.min(maxStart, Math.max(0, args.startSec + rawDelta));
    return { startSec, deltaSec: startSec - args.startSec };
}
```

**Step 4: 运行确认通过** — `npx vitest run dragGeometry` → 4 passed

**Step 5: 提交** — `feat(timeline-kernel): add drag delta resolution with clamping`

---

## Task 2: 内核拖拽手势 + ghost 预览

**Files:**
- Modify: `frontend/src/components/layout/timeline/kernel/host/timelineKernelHost.ts`
- Modify: `frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx`（透传新回调）

**Step 1: 扩展交互回调接口**

在 `TimelineKernelInteractions` 增加：

```ts
    /**
     * 拖拽中的预览（每帧回调，不经 React 渲染）。
     *
     * @param args 拖拽预览状态；null = 结束拖拽（清除 ghost）。
     */
    readonly onDragPreview?: (args: {
        readonly clipId: string;
        readonly deltaSec: number;
        readonly targetTrackId: string;
    } | null) => void;
    /**
     * 拖拽结束并提交。
     *
     * @param args 最终位置；cancelled = true 表示 Esc / pointercancel 取消（不提交）。
     */
    readonly onDragCommit?: (args: {
        readonly clipId: string;
        readonly deltaSec: number;
        readonly targetTrackId: string;
        readonly cancelled: boolean;
    }) => void;
```

**Step 2: 手势状态机扩展**

在宿主的 `Gesture` 联合类型中加入 `clip-drag`：

```ts
        | {
              kind: "clip-drag";
              clipId: string;
              /** 按下时的内容坐标（用于算位移）。 */
              startContentX: number;
              startContentY: number;
              /** 按下时 clip 的起始时间与所属轨道下标（用于算 delta 与跨轨）。 */
              originStartSec: number;
              originTrackIndex: number;
              lengthSec: number;
              /** 最近一次预览的目标轨道 id（避免重复回调）。 */
              lastTargetTrackId: string | null;
          }
```

`onGesturePointerMove` 的 `pending-select` 分支改为：

```ts
        if (gesture.kind === "pending-select") {
            const dx = event.clientX - gesture.startClientX;
            const dy = event.clientY - gesture.startClientY;
            if (dx * dx + dy * dy < DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) return;
            // 超过阈值：进入拖拽。用命中的 clip 信息初始化手势（按下的那一刻
            // 已经拿到 clipId，这里补齐几何量）。
            const clip = findHitClip(gesture.clipId);
            if (clip === null) {
                gesture = { kind: "none" };
                return;
            }
            const rect = container.getBoundingClientRect();
            const view = scroll.get();
            gesture = {
                kind: "clip-drag",
                clipId: gesture.clipId,
                startContentX: gesture.startClientX - rect.left + view.scrollLeft,
                startContentY: gesture.startClientY - rect.top + view.scrollTop,
                originStartSec: clip.startSec,
                originTrackIndex: clip.trackIndex,
                lengthSec: clip.lengthSec,
                lastTargetTrackId: null,
            };
            // 落点预览（同帧发一次，避免松手前的空白）。
            applyDragPreview(event);
            return;
        }
        if (gesture.kind === "clip-drag") {
            applyDragPreview(event);
        }
```

`applyDragPreview` 负责：内容坐标 → `resolveDragDelta`（Task 1）→ 目标轨道下标（`floor(contentY / rowHeight)`，钳制到 `[0, tracks.length-1]`）→ 写 `ghostClipId` / `ghostDeltaSec` / `ghostTrackIndex` → `sceneDirty = true; loop.invalidate()` → 回调 `onDragPreview`。

`onGesturePointerUp` 的 `clip-drag` 分支：取当前预览值 → 回调 `onDragCommit({ ..., cancelled: false })` → 清 ghost → 标脏。

**Step 3: ghost 渲染（复用 GL 实例，零新增 program）**

在 `rebuildInstances` 里，对拖拽中的 clip **追加一个额外实例**：

```ts
        // ghost：把被拖拽的 clip 以半透明画在目标位置（不改原实例——原位置保持
        // 可见，与旧实现的 ghost 语义一致）。
        if (ghostClipId !== null) {
            const source = model.drawClips.find((clip) => clip.id === ghostClipId);
            if (source !== undefined) {
                const ghostClip = {
                    ...source,
                    leftPx: source.leftPx + ghostDeltaPx,
                    topPx: ghostTrackIndex * view.rowHeight + (source.topPx % view.rowHeight),
                };
                // 复用同一套样式；alpha 由 style.mutedAlpha 的等价物承担——
                // 这里直接给实例的 body/header 乘 0.55（与旧实现 ghost 的 opacity-50 同量级）。
                clipBuilder.buildGhost(ghostClip, ...);
            }
        }
```

> **实现提示**：`clipBuilder` 需要新增一个 `buildGhost(clip)` 方法（复用 `buildTimelineClipVisualStyle` + `buildClipBodyInstance`，把颜色 alpha 乘 0.55）。若嫌改动大，**最省的做法**是直接把 ghost 画在细节层（Canvas2D）——细节层已有 `drawTimelineCanvas` 通道，加一个"额外 clip 列表"参数即可。**二选一，以改动小的为准**。

**Step 4: 浏览器验证**

Run（dev server 已起）:
```bash
cd frontend && VW=1920 VH=1200 KERNEL=1 node scripts/dev-shot.mjs \
  "http://localhost:5173/?mock=1" /tmp/drag.png 5000 \
  '[{"type":"drag","button":"left","from":[700,150],"to":[900,150]},{"type":"wait","ms":400},{"type":"shot","path":"/tmp/drag-after.png"}]'
```
Expected: 拖拽中出现 ghost（半透明块随指针移动），松手后 clip 落到新位置。

**Step 5: 提交** — `feat(timeline-kernel): add clip drag gesture with ghost preview`

---

## Task 3: React 侧提交（复用既有编辑语义）

**Files:**
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（新增两个 `useCallback` 并传给 `TimelineKernelView`）

**Step 1: 实现提交回调**

```tsx
    /**
     * 内核拖拽预览：只做本地乐观位置更新（不提交后端）。
     *
     * 特殊说明：与旧实现一致——拖拽期间**每帧**都写 Redux 乐观态（`moveClipStart` /
     * `moveClipTrack`），这样其它面板（参数编辑器、轨道头）能实时跟随；后端提交
     * 只在松手时发生一次。
     */
    const handleKernelDragPreview = React.useCallback(
        (args: { clipId: string; deltaSec: number; targetTrackId: string } | null) => {
            if (args === null) return;
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) return;
            // 吸附：复用既有 snapTimelineDetailed（内核不做吸附，避免规则分叉）。
            const snapped = snapTimelineDetailed(clip.startSec + args.deltaSec, "clip", {
                /* 参数与 useClipDrag 的调用一致，见该文件 510 行附近 */
            });
            batch(() => {
                dispatch(moveClipStart({ clipId: args.clipId, startSec: Math.max(0, snapped) }));
                dispatch(moveClipTrack({ clipId: args.clipId, trackId: args.targetTrackId }));
            });
        },
        [dispatch, sessionRef, snapTimelineDetailed],
    );

    /** 内核拖拽提交：结束事务并调用后端。 */
    const handleKernelDragCommit = React.useCallback(
        (args: { clipId: string; deltaSec: number; targetTrackId: string; cancelled: boolean }) => {
            if (args.cancelled) {
                // 取消：回滚到按下时的位置（乐观态已改，必须还原）。
                /* 参考 useClipDrag 的 `drag.initialById` 回滚分支 */
                return;
            }
            void dispatch(
                moveClipsRemote({
                    /* 参数与 useClipDrag 的调用一致：clipIds / startSec 或 delta */
                }),
            );
        },
        [dispatch],
    );
```

> **实现时必读**：`hooks/useClipDrag.ts` 的 440–520 行（事务开启：`checkpointHistory` + `beginInteraction` + `webApi.beginUndoGroup`）与 780–860 行（提交与收尾）。**本任务的正确性判据是"与旧实现的提交序列一致"**，因此这两段是唯一权威来源，不要凭记忆写。

**Step 2: 接入视图**

```tsx
                            <TimelineKernelView
                                {/* ...既有 props... */}
                                interactions={kernelInteractions}
                            />
```
（`kernelInteractions` 的 `useMemo` 里加入 `onDragPreview` / `onDragCommit`。）

**Step 3: 类型检查与回归**

Run: `cd frontend && npx tsc -b --noEmit && npx vitest run`
Expected: 类型通过；测试 424 passed / 2 failed（后者为预先存在）。

**Step 4: 浏览器端到端验证**

```bash
VW=1920 VH=1200 KERNEL=1 node scripts/dev-shot.mjs "http://localhost:5173/?mock=1" /tmp/drag2.png 5000 \
  '[{"type":"drag","button":"left","from":[700,150],"to":[950,250]},{"type":"wait","ms":600},{"type":"shot","path":"/tmp/drag-cross.png"}]'
```
Expected: clip 水平移动且落到下一条轨道；`window.__mockCalls` 中出现 `move_clips` 或 `set_clips_state_bulk`。

**Step 5: 提交** — `feat(timeline-kernel): commit clip drag through existing edit pipeline`

---

## Task 4: 取消路径与 Esc

**Files:**
- Modify: `kernel/host/timelineKernelHost.ts`

**Step 1:** `keydown` 处理器中，`gesture.kind === "clip-drag"` 且 `event.key === "Escape"` 时：回调 `onDragCommit({ ..., cancelled: true })` → 清 ghost → 标脏。

**Step 2:** `pointercancel` 与窗口失焦（`blur`）同样走取消路径（旧实现用 `registerDragAbort`，内核用同一语义）。

**Step 3:** 验证：拖拽中按 Esc，clip 回到原位。

**Step 4:** 提交 — `feat(timeline-kernel): cancel clip drag on escape or pointer cancel`

---

## 完成标准

1. 拖拽 clip 可水平移动、可跨轨道；松手后位置正确、可撤销（一次拖拽 = 一个撤销步）。
2. 拖拽期间有 ghost 预览，其它面板（参数编辑器）跟随乐观态。
3. Esc / pointercancel 能取消并回滚。
4. `npx tsc -b --noEmit` 通过；`npx vitest run` 无新增失败。
5. 浏览器 mock 端到端验证通过（截图 + `__mockCalls`）。

## 执行记录（2026-09-11，全部完成）

| 任务 | 提交 | 验证 |
|---|---|---|
| Task 1 拖拽几何换算 | `25fcf4ec` | 9 条单测通过（含左右越界、clip 比工程长、非法 pxPerSec、非法行高） |
| Task 2 手势 + ghost | `c98c7acd` | 浏览器拖拽：clip 右移并跨轨，细节层（名称 / M 徽标 / 速率标签 / fade 曲线）完整跟随 |
| Task 3 React 提交 | `c98c7acd` | `move_clip` 调用 1 次；全量 433 passed / 2 failed（预先存在） |
| Task 4 取消路径 | `c98c7acd` | 拖拽中按 Esc：`move_clip` 调用 **0 次**，clip 回到原位 |

**实施中的两处偏差（均优于计划）**：

1. **不做独立 ghost 图层**。计划里给了「GL 追加实例 / 细节层额外 clip 列表」两条路，实际发现
   两者都不必要：调用方写乐观位置 → Redux `clips` 引用变化 → 内核重建几何时把 clip 画在新位置，
   视觉上即"跟着指针走"。**只有一份位置真值**，不存在 ghost 与实体分叉的可能，也省掉一个图层。
2. **`resolveTargetTrackIndex` 一并实现**（计划里只在 Task 2 的实现提示里提到）：纵向 → 目标
   轨道下标的换算与钳制是纯函数，与 `resolveDragDelta` 同属"几何"，放在一起更好测。

**新增调试能力**：`dev-shot.mjs` 支持分离的 `down` / `up` 动作，可表达「拖拽中途按键」这类
手势（原 `drag` 动作是一次性 down→move→up，无法插入中间步骤）。

**踩坑记录**：拖拽预览必须用**按下时的原始位置** + 相对位移换算。若直接用
`clip.startSec + deltaSec`，由于上一次预览已改写 Redux 中的位置，位移会逐帧叠加
（表现为 clip 越拖越快）。内核回调的是相对位移，面板侧用 `kernelDragOriginRef` 记录基准。

## 后续任务（本计划外）

- trim（左右边缘裁短/延长）、fade 角拖拽（复用本计划的 ghost + 提交骨架，手势换成边缘命中）
- 框选（`useTimelineSelectionRect` 的坐标源改为 `ScrollKernel`）
- 浮层锚点（右键菜单 / tooltip / 吸附高亮按内核视口定位）
- 真机验证波形与 1000 clip 场景性能复测
