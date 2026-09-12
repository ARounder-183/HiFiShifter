# 内核回归修复与参数编辑器钢琴背景 · 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复「时间轴内核收归唯一路径」改造遗留的 5 个缺陷（2 个功能回归 + 3 个内核缺陷），并为参数编辑器音高视图新增黑键行背景带与恢复音阶高亮。

**Architecture:** 全部改动都在既有架构内，不引入新渲染路径。#1 修提交分支丢弃落点；#2 新增「只标脏不重建场景」的播放头重绘入口；#4 把主画布内容签名从 `join("|")` 字符串改为逐项 `Object.is` 引用比较（修掉对象塌缩）；#5 把孤儿组件的音高折线数学搬进细节层；#6 与钢琴背景都进网格实例缓冲，并同步更新几何签名。

**Tech Stack:** Tauri 2 · React 19 · TypeScript · Redux Toolkit · WebGL2（自研实例化内核）· Vitest（**node 环境，无 jsdom**）

---

## 关键背景（实现者必读）

### 仓库硬约定

1. **每个文件最上方必须有中文头注释**，说明：主要内容 / 作用 / 与其他模块的关系。
2. **每个关键函数必须有中文 doc 注释**，说明：流程 / 作用 / 特殊说明 / 参数。修改函数必须同步维护。
3. **陈旧注释视为实质缺陷**——本计划多处删除「与代码矛盾」的注释，那是交付物的一部分而不是顺手清理。
4. **绝不 push**。只做本地提交。
5. 分支：`feature/timeline-unified-render-kernel`。开工前 `git status --short` 必须为空。

### Vitest 环境约束（极易踩坑）

- `vitest` 跑在 **node** 环境，**没有 jsdom**。
- **135 个测试文件中没有任何一个 import `.tsx`**，因为 `.tsx` 会经 barrel 间接引入 Redux store / `localStorage` 而崩溃。
- **推论（本计划的核心约束）**：任何需要单测的逻辑，**必须**放进**纯 `.ts` 模块**且不 import 任何 `.tsx`。计划中每个"写失败测试"步骤都已按此设计；若实现时发现待测逻辑在 `.tsx` 里，正确做法是**抽成纯模块**，而不是放宽测试环境。
- 所以 Task 2、Task 3、Task 6 都包含「先把纯逻辑抽到 `.ts`」这一步——那不是重构洁癖，是**能否测试**的前提。

### `noUnusedLocals` 已开启

`tsconfig.app.json` 已开 `noUnusedLocals`。因此 `npx tsc -b --noEmit` 会报 `TS6133`（未使用变量）。删代码后靠它找出剩余孤儿是**有效手段**（计划中用到）。

注意：`tsc -b --noUnusedLocals` 是**非法**写法（`TS5094`），不要用。

### 基线数字（改动前后都要核对）

| 项目 | 基线值 |
| --- | --- |
| `npx vitest run` | **838 passed / 2 failed**（840 total，135 files） |
| 那 2 个失败 | `keybindingMatch.test.ts` 的「默认 Shift 变体」「Ctrl 微调变体」——**既有失败，`develop` 上同样失败。绝不要去"修"它们** |
| `npx tsc -b --noEmit` | 无输出 |
| eslint | 0 error / 12 warning |
| `npm run build` | 成功，产物 `import.meta.env` 出现 **0** 次 |

### 真机验证工具

dev server 可能已在 **http://127.0.0.1:5174**（从 `frontend/` 启动）。截图/交互脚本（从 `frontend/` 目录运行）：

```bash
VW=1920 VH=1200 node scripts/dev-shot.mjs "<url>" /tmp/out.png <waitMs> '<actionsJson>'
```

- `actionsJson` 是数组，支持 `wheel` / `drag` / `click` / `down` / `up` / `move` / `key` / `keyDown` / `keyUp` / `type` / `wait` / `shot` / `shotElement` / `eval` / `goto`。详见 `frontend/scripts/dev-shot.mjs` 头部注释。
- `PROBE_INIT=<file>` 环境变量可注入**页面加载前**执行的脚本。
- URL 加 `?mock=1` 使用 mock 后端（有示例工程数据）。
- **`eval` 动作之间不共享变量**：`await import(...)` 拿到的模块句柄不会跨 eval 保留。要在**同一个 eval 里**完成 import + 把 store 挂到 `window`，后续 eval 才能读 `window.__store`。
- **不要重复启动 5174**，也不要杀掉已有的 server。需要新端口就用别的。

### 内核容器几何（写探测脚本时要用的坐标）

`?mock=1`、1920×1200、dpr=2 下：

- 内核容器 `[data-hs-timeline-kernel]` 的 rect = `{left: 256, top: 112, right: 1920, bottom: 263}`（高 151px）。
- 轨 1 行中心 y ≈ **150**，轨 2 行中心 y ≈ **230**。
- **注意 x=1850 在轨 1 上不是空白**——那里落在超长 clip 内部。要测空白用 x≈400。
- 参数编辑器主画布 `[data-piano-roll-canvas]` 的 rect = `{left: 56, top: 353, w: 1864, h: 823}`。
- 参数编辑器默认 `toolMode` 是 `"draw"` 不是 `"select"`；测选区前必须先点工具栏「选择」按钮（约 `(132, 289)`）。

---

## 文件结构（先锁定拆分决策）

| 文件 | 动作 | 职责 |
| --- | --- | --- |
| `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.ts` | 修改 | 网格实例构建：新增黑键行背景带 + 音阶高亮强调线（**均先于网格线发射**） |
| `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.test.ts` | 修改 | 上述两者的纯单测 + 发射顺序断言 |
| `frontend/src/components/layout/pianoRoll/kernel/scene/gridView.ts` | 修改 | 几何签名：纳入背景色 + 音阶高亮输入（漏项 = 切主题/切音阶不生效） |
| `frontend/src/components/layout/pianoRoll/kernel/scene/gridView.test.ts` | 修改 | 签名对新增输入的敏感性单测 |
| `frontend/src/components/layout/pianoRoll/colors.ts` | 修改 | 新增 `blackKeyRowBand` 具名常量（深浅两套）+ 音阶高亮强调色 |
| `frontend/src/components/layout/pianoRoll/mainCanvasSignature.ts` | **新建** | **纯模块**：主画布内容签名的构造与比较（从 `PianoRollPanel.tsx` 抽出，使其可单测） |
| `frontend/src/components/layout/pianoRoll/mainCanvasSignature.test.ts` | **新建** | 签名比较单测 + 塌缩回归守卫 |
| `frontend/src/components/layout/pianoRoll/render.ts` | 修改 | 缓存改为数组逐项比较；删除已死的音阶高亮分支 |
| `frontend/src/components/layout/timeline/runtime/midiPitchCurve.ts` | **新建** | **纯模块**：从孤儿组件搬出的音高折线生成数学（含 Loop 回绕、环回标记位置） |
| `frontend/src/components/layout/timeline/runtime/midiPitchCurve.test.ts` | **新建** | 折线数学单测（trim / playbackRate / reversed / loop） |
| `frontend/src/components/layout/timeline/runtime/timelineCanvasModel.ts` | 修改 | clip 模型新增可选 `midiPitchCurvePx` 字段 |
| `frontend/src/components/layout/timeline/runtime/timelineCanvasRenderer.ts` | 修改 | 细节层新增折线描边 + 环回 ▽ 标记 |
| `frontend/src/components/layout/timeline/kernel/scene/playheadInvalidation.ts` | **新建** | **纯模块**：判定"播放头变化是否需要标脏"（把不可测的宿主决策抽出来） |
| `frontend/src/components/layout/timeline/kernel/scene/playheadInvalidation.test.ts` | **新建** | 上述判定单测 |
| `frontend/src/components/layout/timeline/kernel/host/timelineKernelHost.ts` | 修改 | 新增只标脏入口；接上 `args.playheadSec` getter |
| `frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx` | 修改 | 传 `playheadSec` getter；暴露播放头重绘回调 |
| `frontend/src/components/layout/timeline/hooks/kernelDropCommit.ts` | **新建** | **纯模块**：从提交分支抽出"落点 → dropToNewTrack / trackOffset"的解析（可单测） |
| `frontend/src/components/layout/timeline/hooks/kernelDropCommit.test.ts` | **新建** | 上述解析单测（含哨兵、跨轨、多媒体选择） |
| `frontend/src/components/layout/TimelinePanel.tsx` | 修改 | 提交分支改用上述解析、清空 ref、接真实建轨；删除孤儿 `playheadRef` |
| `frontend/src/components/layout/pianoRoll/usePianoRollInteractions.ts` | 修改 | 新建选区分支补 `invalidate()` |
| `frontend/src/components/layout/timeline/hooks/useTimelineState.ts` | 修改 | 删除孤儿 `playheadRef` 声明与写点 |
| `frontend/src/dev/mockBackend.ts` | 修改 | 给 MIDI clip 补 `midi_note_data`（否则 #5 无法视觉验证） |

---

## Task 1: 修饰键拖拽落点解析（缺陷 1，纯逻辑）

**为什么先做**：这是唯一一个"用户明确说 ghost 能到、落库不能到"的缺陷，且根因是**提交分支丢弃了预览分支已算出的信息**。先把解析逻辑抽成纯函数，才能在没有 DOM 的环境下锁定行为。

**Files:**
- Create: `frontend/src/components/layout/timeline/hooks/kernelDropCommit.ts`
- Create: `frontend/src/components/layout/timeline/hooks/kernelDropCommit.test.ts`

- [ ] **Step 1: 写失败测试**

创建 `frontend/src/components/layout/timeline/hooks/kernelDropCommit.test.ts`：

```ts
/**
 * 内核拖拽落点解析单测。
 *
 * 【为什么必须有这一层】#1 的根因是提交分支把 `dropToNewTrack` / `trackOffset`
 * 写死为 `false` / `0`，而预览分支用 `args.targetTrackId` 算出了真实落点。
 * 两处各写一份必然分叉（这正是缺陷本身），因此把解析收敛到本模块并在此锁定行为。
 */
import { describe, expect, it } from "vitest";

import { resolveKernelDropTarget } from "./kernelDropCommit";

const NEW_TRACK_SENTINEL = "__new_track__";

describe("resolveKernelDropTarget", () => {
    it("落点为新轨哨兵：dropToNewTrack 为真、trackOffset 为 0", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: NEW_TRACK_SENTINEL,
                trackIds: ["t1", "t2"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: true, trackOffset: 0, targetTrackIndex: -1 });
    });

    it("落到其他已有轨道：trackOffset 为下标差（缺陷 1 的核心）", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t3",
                trackIds: ["t1", "t2", "t3"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: 1, targetTrackIndex: 2 });
    });

    it("落到原轨：trackOffset 为 0", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t2",
                trackIds: ["t1", "t2", "t3"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: 0, targetTrackIndex: 1 });
    });

    it("向上跨轨得到负偏移", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t1",
                trackIds: ["t1", "t2", "t3"],
                anchorTrackIndex: 2,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }).trackOffset,
        ).toBe(-2);
    });

    it("落点不在轨道列表里且不是哨兵：回落原轨（不跨轨）", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "ghost-id",
                trackIds: ["t1", "t2"],
                anchorTrackIndex: 1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }),
        ).toEqual({ dropToNewTrack: false, trackOffset: 0, targetTrackIndex: -1 });
    });

    it("锚点下标非法（-1）：不跨轨", () => {
        expect(
            resolveKernelDropTarget({
                targetTrackId: "t2",
                trackIds: ["t1", "t2"],
                anchorTrackIndex: -1,
                newTrackSentinel: NEW_TRACK_SENTINEL,
            }).trackOffset,
        ).toBe(0);
    });
});
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd frontend && npx vitest run src/components/layout/timeline/hooks/kernelDropCommit.test.ts
```

Expected: FAIL —— `Failed to resolve import "./kernelDropCommit"`（文件还不存在）。

- [ ] **Step 3: 写最小实现**

创建 `frontend/src/components/layout/timeline/hooks/kernelDropCommit.ts`：

```ts
/**
 * 时间轴渲染内核 · 拖拽落点解析（纯函数）
 *
 * 【主要内容】
 * 把内核手势回调给出的 `targetTrackId` 解析为提交落库所需的三元组：
 * 是否新建轨道、相对锚点的轨道偏移量、目标轨道下标。
 *
 * 【作用】
 * 内核拖拽的回调把落点作为 `targetTrackId` 传回面板（拖到全部轨道之下时是
 * `NEW_TRACK_SENTINEL` 哨兵）。**预览分支与提交分支都必须用同一份解析**——
 * 原实现里预览分支老老实实按 `args.targetTrackId` 算，提交分支却把
 * `dropToNewTrack` / `trackOffset` 写死成 `false` / `0`，于是「幽灵预览能到新轨道
 * 和其他轨道，落库却永远留在原轨」。两处各写一份必然分叉，本模块是唯一来源。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 的 `handleKernelDragCommit`（copy 分支与 move 分支）。
 * - 下游：`copyClipsFromDrag` 依据 `dropToNewTrack` / `trackOffset` 解析目标轨。
 * - 独立性：纯函数，无 React / DOM / Redux 依赖，可在 node 环境单测。
 *
 * 【设计约束】
 * 1. 落点不在轨道列表中**且不是哨兵**时按"回落原轨"处理（`trackOffset = 0`）：
 *    这既覆盖手指/指针落在轨道区之外的情形，也覆盖重名 id 之类的异常输入。
 * 2. 锚点下标非法（< 0，例如该 clip 的轨道已被删除）时不跨轨，避免用一个
 *    无意义的差值把 clip 甩到别的轨道上。
 */

/** 解析入参。 */
export interface KernelDropTargetArgs {
    /** 内核回调给出的落点轨道 id；可能是 `newTrackSentinel`。 */
    readonly targetTrackId: string;
    /** 当前工程的轨道 id 列表（顺序 = 纵向排列顺序）。 */
    readonly trackIds: readonly string[];
    /** 拖拽锚点 clip 的轨道下标；非法时为 -1。 */
    readonly anchorTrackIndex: number;
    /** 「拖到全部轨道之下」的哨兵值。 */
    readonly newTrackSentinel: string;
}

/** 解析结果。 */
export interface KernelDropTarget {
    /** 是否落到新建轨道。 */
    readonly dropToNewTrack: boolean;
    /** 相对锚点轨道下标的偏移量（同轨为 0；哨兵落点为 0）。 */
    readonly trackOffset: number;
    /** 目标轨道下标；哨兵或解析失败时为 -1。 */
    readonly targetTrackIndex: number;
}

/**
 * 解析内核拖拽落点。
 *
 * 流程：哨兵判定 → 在 `trackIds` 中定位目标下标 → 与锚点下标相减得偏移量
 * （任一下标非法时偏移量归零）。
 *
 * @param args 落点、轨道列表与锚点下标。
 * @returns 新建标记、轨道偏移量与目标下标。
 */
export function resolveKernelDropTarget(args: KernelDropTargetArgs): KernelDropTarget {
    if (args.targetTrackId === args.newTrackSentinel) {
        return { dropToNewTrack: true, trackOffset: 0, targetTrackIndex: -1 };
    }
    const targetTrackIndex = args.trackIds.indexOf(args.targetTrackId);
    if (targetTrackIndex < 0 || args.anchorTrackIndex < 0) {
        return { dropToNewTrack: false, trackOffset: 0, targetTrackIndex };
    }
    return {
        dropToNewTrack: false,
        trackOffset: targetTrackIndex - args.anchorTrackIndex,
        targetTrackIndex,
    };
}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd frontend && npx vitest run src/components/layout/timeline/hooks/kernelDropCommit.test.ts
```

Expected: PASS，6 passed。

- [ ] **Step 5: 提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add frontend/src/components/layout/timeline/hooks/kernelDropCommit.ts frontend/src/components/layout/timeline/hooks/kernelDropCommit.test.ts
git commit -m "feat(timeline): 抽出内核拖拽落点解析（纯函数，含哨兵/跨轨/回落）"
```

---

## Task 2: 提交分支接上真实落点（缺陷 1 收尾）

**Files:**
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（约 `:1726-1760` 的 copy 分支）

- [ ] **Step 1: 确认基线行为（真机复现缺陷）**

先复现，才能证明修复有效。启动 dev server（若 5174 未在跑）：

```bash
cd frontend && npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

然后在**另一个**终端跑（把 mock 调用轨迹读出来）：

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t2-before.png 6000 '[
{"type":"eval","js":"window.__c=()=>{const a=(window.__mockCalls||[]).slice();window.__mockCalls.length=0;return a.filter(x=>/add_track|duplicate/.test(x));};return \"ok\";"},
{"type":"keyDown","key":"Meta"},
{"type":"drag","from":[700,150],"to":[700,255],"steps":14},
{"type":"keyUp","key":"Meta"},
{"type":"wait","ms":1200},
{"type":"eval","js":"return {AFTER_COPY_DRAG_BELOW_TRACKS: window.__c()};"}
]'
```

Expected（**缺陷证据**）：`["duplicate_clips_bulk"]` —— **没有** `add_track`。对照：不按 Meta 做同样拖拽会得到 `add_track`。

- [ ] **Step 2: 修改提交分支**

打开 `frontend/src/components/layout/TimelinePanel.tsx`，找到 copy 分支（`if (origin.copyMode) {` 这一块，约 `:1712`）。

在该分支**最前面**（`setKernelGhost(null);` 之后、`if (args.cancelled) return;` **之前**）加入落点解析与 ref 清空：

```ts
                // 落点解析必须**先于** cancelled 提前返回：cancel 路径同样要把
                // 「拖到新轨」的标记清掉，否则虚线新轨行会永久残留（实测：⌘ 拖到
                // 全部轨道之下再松手，虚线圈会一直挂在画面上，且之后任何普通移动
                // 都会被误判成哨兵落点）。
                const dropTarget = resolveKernelDropTarget({
                    targetTrackId: args.targetTrackId,
                    trackIds: sessionRef.current.tracks.map((track) => track.id),
                    anchorTrackIndex: origin.anchorTrackIndex,
                    newTrackSentinel: NEW_TRACK_SENTINEL,
                });
                if (kernelDropToNewTrackRef.current) {
                    kernelDropToNewTrackRef.current = false;
                    setKernelDropToNewTrack(false);
                }
```

然后把原来写死的两行替换掉：

```ts
                    // 内核拖拽的落点始终是已有轨道（`resolveTargetTrackIndex` 越界时
                    // 回落原轨），因此不涉及建新轨。
                    dropToNewTrack: false,
                    trackOffset: 0,
```

替换为：

```ts
                    // 落点来自与预览分支**同一份**解析（见 `resolveKernelDropTarget`）：
                    // 写死这两项是「幽灵能到新轨道、落库留在原轨」的根因。
                    dropToNewTrack: dropTarget.dropToNewTrack,
                    trackOffset: dropTarget.trackOffset,
```

**同时必须修 `resolveTrackIdByOffset`**（紧邻上方，约 `:1743-1750`）——否则上一步是**静默无效**：

```ts
                    // 每个参与者按各自初始轨道序号 + 同一偏移量解析目标轨。
                    resolveTrackIdByOffset: (clipId) => {
                        const participant = origin.participants.find(
                            (item) => item.clipId === clipId,
                        );
                        if (participant === undefined || participant.trackIndex < 0) return null;
                        return trackIds[participant.trackIndex] ?? null;
                    },
```

替换为：

```ts
                    // 每个参与者按各自初始轨道序号 + 同一偏移量解析目标轨。
                    //
                    // ★ 必须加上 `trackOffset`：此前这里直接 `trackIds[participant.trackIndex]`
                    //   返回**原轨**，即完全忽略偏移量。结果是即便上层把
                    //   `trackOffset` 传对了，`copyClipsFromDrag` 拿到的仍是原轨 id
                    //   —— 修复静默失效（"幽灵到了、落库还在原轨"依旧）。
                    //   旧实现的正确语义见 `useClipDrag.ts` 的 `resolveTrackIdByOffset`：
                    //   `targetIndex = sourceIndex + trackOffset`。
                    //   越界（含负下标）时 `?? null` 让调用方回落到原轨。
                    resolveTrackIdByOffset: (clipId) => {
                        const participant = origin.participants.find(
                            (item) => item.clipId === clipId,
                        );
                        if (participant === undefined || participant.trackIndex < 0) return null;
                        return trackIds[participant.trackIndex + dropTarget.trackOffset] ?? null;
                    },
```

再把两个假 creator 换成真实实现（**必须与上面同时落地**：`copyClipsFromDrag.ts:137-138` 在 `dropToNewTrack` 为真而 creator 返回 null 时抛 `create_track_failed`）：

```ts
                    createNewTracksForDrop: async () => [],
                    createNewTrackForDrop: async () => null,
```

替换为：

```ts
                    // copy 语义**不能**复用 `createNewTrackForKernelDrop`——那个函数
                    // 会顺手把 clip **移动**到新轨（它服务于 move 路径）。copy 的
                    // 副本位置由随后的 `duplicate_clips_bulk` 决定，这里只需要
                    // "建出空轨道并返回它们的 id"。
                    createNewTracksForDrop: (span: number) =>
                        createTrackIdsForDrop({ dispatch, sessionRef }, span),
                    createNewTrackForDrop: async () => {
                        const created = await createTrackIdsForDrop({ dispatch, sessionRef }, 1);
                        return created[0] ?? null;
                    },
```

并在 `frontend/src/components/layout/timeline/hooks/createNewTrackForDrop.ts` 末尾
（**同一个文件**，因为职责相同："建轨并解析新轨 id"）新增这个导出函数：

```ts
/** 纯建轨的依赖集合。 */
export interface CreateTrackIdsDeps {
    readonly dispatch: AppDispatch;
    readonly sessionRef: React.RefObject<SessionState>;
}

/**
 * 在轨道列表末尾新建 `count` 条**空**轨道，返回它们的新 id（按建轨顺序）。
 *
 * 流程（每条轨道）：记录建轨前的 id 集合 → `addTrackRemote` → 按**差集**解析新 id
 * （回退到回包的 `selected_track_id`，再回退到末条）。
 *
 * 【为什么与 `createNewTrackForKernelDrop` 分开】后者服务 move 路径，会顺手把
 * clip 移到新轨；copy 路径的副本位置由 `duplicate_clips_bulk` 决定，只需要空轨道。
 * 把两者混用会让 copy 先发生一次移动，再发生一次复制 —— 原 clip 被搬走。
 *
 * 【为什么按差集而不是取末条】并发建轨或后端返回顺序变化时，取末条会拿到别人的
 * 轨道（与 `createNewTrackForKernelDrop` 同一约束，见该文件头部设计约束 2）。
 *
 * @param deps 注入的 dispatch 与 sessionRef。
 * @param count 要新建的轨道数（<= 0 时返回空数组）。
 * @returns 新轨 id 列表；某条失败时该条被跳过（列表可能短于 `count`，调用方据此判定失败）。
 */
export async function createTrackIdsForDrop(
    deps: CreateTrackIdsDeps,
    count: number,
): Promise<string[]> {
    const createdIds: string[] = [];
    for (let index = 0; index < Math.max(0, Math.floor(count)); index += 1) {
        const before = new Set(deps.sessionRef.current?.tracks.map((track) => track.id) ?? []);
        const res = (await deps.dispatch(
            addTrackRemote({ name: undefined, parentTrackId: null }),
        ).unwrap()) as {
            tracks?: Array<{ id?: string }>;
            selected_track_id?: string | null;
        };
        const nextTracks = Array.isArray(res?.tracks) ? res.tracks : [];
        const created = nextTracks.find((track) => !before.has(String(track?.id)));
        const id =
            (created && String(created.id)) ||
            (res?.selected_track_id ? String(res.selected_track_id) : null) ||
            (nextTracks.length > 0 ? String(nextTracks[nextTracks.length - 1]?.id) : null) ||
            null;
        if (id) createdIds.push(id);
    }
    return createdIds;
}
```

> **实现提示**：`sessionRef` 的类型是 `React.RefObject<SessionState>`（见同文件
> `CreateNewTrackForDropDeps`）。上面用了 `?.` 是因为 mock / 早期渲染期它可能为
> `null`；若实际类型不允许，按现有写法调整（**照抄同文件既有风格**，不要新造模式）。

- [ ] **Step 3: 加 import**

在 `TimelinePanel.tsx` 顶部 import 区加入：

```ts
import { resolveKernelDropTarget } from "./timeline/hooks/kernelDropCommit";
import { createTrackIdsForDrop } from "./timeline/hooks/createNewTrackForDrop";
```

在 `createNewTrackForDrop.ts` 顶部 import 区确认已有：

```ts
import { addTrackRemote } from "../../../../features/session/sessionSlice";
```

（该文件已在用 `addTrackRemote` 实现 `createNewTrackForKernelDrop`，通常无需改动。）

- [ ] **Step 4: 类型检查**

```bash
cd frontend && npx tsc -b --noEmit
```

Expected: 无输出。若报 `createNewTrackForKernelDrop` 参数不匹配，按 Step 2 的实现提示调整。

- [ ] **Step 5: 真机验证（三条路径）**

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t2-after.png 6000 '[
{"type":"eval","js":"window.__c=()=>{const a=(window.__mockCalls||[]).slice();window.__mockCalls.length=0;return a.filter(x=>/add_track|duplicate/.test(x));};return \"ok\";"},
{"type":"keyDown","key":"Meta"},
{"type":"drag","from":[700,150],"to":[700,255],"steps":14},
{"type":"keyUp","key":"Meta"},
{"type":"wait","ms":1200},
{"type":"eval","js":"return {A_copy_to_NEW_track: window.__c()};"},
{"type":"keyDown","key":"Meta"},
{"type":"drag","from":[700,150],"to":[700,230],"steps":14},
{"type":"keyUp","key":"Meta"},
{"type":"wait","ms":1200},
{"type":"eval","js":"return {B_copy_to_EXISTING_track2: window.__c()};"},
{"type":"eval","js":"const stray=[...document.querySelectorAll(\"[data-hs-timeline-kernel] *\")].filter(e=>(e.textContent||\"\").includes(\"新轨道\"));return {C_strayNewTrackRowCount: stray.length};"}
]'
```

Expected：
- A：出现 `add_track`（且不是 fallback），并且 `duplicate_clips_bulk` 的 `trackMode` **不是** `same_track`；
- B：`duplicate_clips_bulk` 的 `trackMode` 为 `explicit_mapping`；
- C：虚线新轨行**不残留**（计数为 0）。

若 A 只出现 `duplicate_clips_bulk` 而没有 `add_track`，说明 creator 仍是假实现，回 Step 2。

- [ ] **Step 6: 全量测试**

```bash
cd frontend && npx vitest run 2>&1 | tail -5
```

Expected: `Tests 2 failed | 838 passed (840)` —— 与基线一致（Task 1 新增 6 项，故实际应为 **2 failed | 844 passed (846)**）。

- [ ] **Step 7: 提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add frontend/src/components/layout/TimelinePanel.tsx
git commit -m "fix(timeline): 修饰键拖拽提交接上真实落点（新轨/跨轨），并清理残留的新轨标记"
```

---

## Task 3: 播放头重绘（缺陷 2）

**注意**：本任务**只**修 #2（内核不重绘）。#3 的"播放头不动 / 不取消选中"是**另外两个独立缺陷**，在 Task 3b 处理。不要把两者混在一起改。

**Files:**
- Create: `frontend/src/components/layout/timeline/kernel/scene/playheadInvalidation.ts`
- Create: `frontend/src/components/layout/timeline/kernel/scene/playheadInvalidation.test.ts`
- Modify: `frontend/src/components/layout/timeline/kernel/host/timelineKernelHost.ts`
- Modify: `frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx`
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（**仅**播放头桥接部分）
- Modify: `frontend/src/components/layout/timeline/hooks/useTimelineState.ts`（**仅**删孤儿）

- [ ] **Step 1: 写失败测试**

创建 `frontend/src/components/layout/timeline/kernel/scene/playheadInvalidation.test.ts`：

```ts
/**
 * 播放头标脏判定单测。
 *
 * 【为什么需要这一层】宿主本身需要真实 WebGL2 上下文才能构造，无法直接单测；
 * 而 #2 的根因正是「播放头变化时**没有**请求重绘」这一决策。把它抽成纯函数，
 * 就能在没有 GPU 的环境下锁定行为（本工程 vitest 是 node 环境，无 jsdom）。
 */
import { describe, expect, it } from "vitest";

import { shouldRepaintForPlayhead } from "./playheadInvalidation";

describe("shouldRepaintForPlayhead", () => {
    it("位置变化 → 需要重绘", () => {
        expect(shouldRepaintForPlayhead(12.5, 4.82)).toBe(true);
    });

    it("位置未变 → 不需要重绘（空闲零成本）", () => {
        expect(shouldRepaintForPlayhead(12.5, 12.5)).toBe(false);
    });

    it("差异在亚像素量级以下 → 不重绘（避免浮点噪声刷帧）", () => {
        expect(shouldRepaintForPlayhead(12.5, 12.5 + 1e-9)).toBe(false);
    });

    it("首次绘制（上次为 NaN 哨兵）→ 需要重绘", () => {
        expect(shouldRepaintForPlayhead(12.5, Number.NaN)).toBe(true);
    });

    it("当前位置非有限 → 不重绘（防御 NaN 灌进样式）", () => {
        expect(shouldRepaintForPlayhead(Number.NaN, 12.5)).toBe(false);
    });
});
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd frontend && npx vitest run src/components/layout/timeline/kernel/scene/playheadInvalidation.test.ts
```

Expected: FAIL —— 无法解析 import。

- [ ] **Step 3: 写纯模块实现**

创建 `frontend/src/components/layout/timeline/kernel/scene/playheadInvalidation.ts`：

```ts
/**
 * 时间轴渲染内核 · 播放头标脏判定（纯函数）
 *
 * 【主要内容】
 * 判定「当前播放头位置」相对「上一次绘制用的位置」是否值得请求一次重绘。
 *
 * 【作用：修的是什么】
 * 内核的渲染循环是**纯脏标记驱动**的（`renderKernel/renderLoop`：`start()` 不绘制，
 * 也没有常驻 rAF）。而播放头位置的真值由面板的视觉插值 ref 持有、经数据镜像
 * 传进内核——**镜像变化不会自动标脏**。于是点标尺 seek 后：标尺播放头（React
 * 声明式渲染）动了，内核自绘的轨道区播放头却永不重绘，冻结在旧位置。
 *
 * 修复要在「播放头变了」时主动请求重绘，本模块就是这个判定的唯一来源。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost`/`timelineKernelHost` 的帧提交与外部重绘入口。
 * - 独立性：纯函数，无 DOM / WebGL / React 依赖，可在 node 环境单测。
 *
 * 【设计约束】
 * 1. 用**相对容差**而不是严格相等：播放头位置来自 `performance.now()` 外推，
 *    同一逻辑位置在两次读取间会有极小浮点抖动；严格相等会每帧都判定"变了"，
 *    把空闲状态也变成持续重绘。
 * 2. 上一次为 `NaN` 视为"尚未绘制过"，必须返回真（首帧不能跳过）。
 * 3. 当前值非有限时返回假：宁可不动，也不能把 NaN 写进 `style.transform`
 *    （NaN 会让该元素整层失效，且不会报错）。
 */

/** 位置比较的相对容差（秒）。远小于一个像素对应的时间，又足以吸收浮点抖动。 */
const PLAYHEAD_EPSILON_SEC = 1e-6;

/**
 * 播放头位置变化是否需要重绘。
 *
 * 流程：当前值有限性校验 → 上次值哨兵判定 → 相对容差比较。
 *
 * @param nextSec 本帧的播放头位置（秒）。
 * @param lastSec 上一次绘制用的位置（秒）；`NaN` 表示尚未绘制过。
 * @returns 需要请求一次重绘时为 true。
 */
export function shouldRepaintForPlayhead(nextSec: number, lastSec: number): boolean {
    if (!Number.isFinite(nextSec)) return false;
    if (!Number.isFinite(lastSec)) return true;
    return Math.abs(nextSec - lastSec) > PLAYHEAD_EPSILON_SEC;
}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd frontend && npx vitest run src/components/layout/timeline/kernel/scene/playheadInvalidation.test.ts
```

Expected: PASS，5 passed。

- [ ] **Step 5: 宿主新增「只标脏」入口并接上实时 getter**

打开 `frontend/src/components/layout/timeline/kernel/host/timelineKernelHost.ts`。

**(a)** 在宿主接口里（`invalidateScene(): void;` 附近，约 `:784`）新增方法声明，并写清它与 `invalidateScene` 的区别：

```ts
    /**
     * 仅请求一次重绘（不重建 GPU 几何）。
     *
     * 【为什么必须与 `invalidateScene` 分开】`invalidateScene` 会置 `sceneDirty`，
     * 下一帧重建**全部**实例几何并重新上传顶点缓冲。播放头是每帧变化的量（播放中
     * 60fps），用它触发场景重建等于每帧重传几何——这正是内核要消除的开销。
     * 本方法只做标脏，与 `renderLoop.invalidate()` 同义。
     */
    invalidatePlayhead(): void;
```

**(b)** 在实现对象里（`invalidateScene() { … }` 旁边，约 `:4111`）实现：

```ts
        invalidatePlayhead() {
            if (disposed) return;
            // 刻意不置 sceneDirty：见接口处的说明。
            loop.invalidate();
        },
```

**(c)** 把已声明但从未被读取的 `playheadSec` getter 接上。`args` 里已有
`readonly playheadSec?: () => number;`（约 `:294`），但它没被解构。在宿主内部
（`data` 取值处附近）加一个读取函数：

```ts
    /**
     * 读取本帧的播放头位置。
     *
     * 【为什么要走 getter 而不是数据镜像】`data().playheadSec` 由面板在
     * **render 期**写入镜像对象，而视觉插值 ref 是在 `useVisualPlayhead` 的
     * effect 里更新的——镜像因此**滞后一次提交**。用滞后值定位播放头会让它
     * 停在上一次的位置（实测：连续两次 seek，镜像恰好差一次）。
     * getter 直连面板的 ref，读到的是当帧真值。
     *
     * 缺省回退镜像值：宿主在无 getter 的场景（单测 / 未接线）仍要能工作。
     */
    function readPlayheadSec(): number {
        const live = args.playheadSec;
        if (live !== undefined) {
            const value = live();
            if (Number.isFinite(value)) return value;
        }
        return data().playheadSec;
    }
```

**(d)** 把 `syncDom()` 里读播放头的那一行（`const playheadContentX = data().playheadSec * view.pxPerSec;`，约 `:1816`）改为：

```ts
        const playheadContentX = readPlayheadSec() * view.pxPerSec;
```

**(e)** 把 `draw()` 里读播放头的那两行（约 `:1900`）改为：

```ts
        const playheadSec = readPlayheadSec();
        const playheadMoved = shouldRepaintForPlayhead(playheadSec, lastDrawnPlayheadSec);
        lastDrawnPlayheadSec = playheadSec;
```

并在文件顶部 import：

```ts
import { shouldRepaintForPlayhead } from "../scene/playheadInvalidation";
```

> **注意**：`lastDrawnPlayheadSec` 的初值应为 `Number.NaN`（首帧必须绘制）。若原实现
> 初始化成 `0`，改成 `NaN` 并确认首帧仍正常绘制。

- [ ] **Step 6: 视图传 getter 并暴露重绘入口**

打开 `frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx`。

**(a)** 把已有的 `getPlayheadSec` 同时传给宿主的 `playheadSec` 参数（它目前只喂 `buildData()`，约 `:369`）。在创建宿主的参数对象里加入：

```ts
                // 直连面板的视觉插值 ref：镜像会滞后一次提交（见宿主内
                // readPlayheadSec 的说明）。
                playheadSec: getPlayheadSec,
```

**(b)** 让 `getPlayheadSec` 参与宿主重建保护：它来自 `callbacksRef`，宿主创建时只取一次。确认 `TimelineKernelView` 的 props 里 `getPlayheadSec` 引用稳定（面板侧用 `React.useCallback(..., [])` 提供——`TimelinePanel.tsx:520` 已如此）。**若不稳定，改为在宿主参数里传 `() => callbacksRef.current.getPlayheadSec()`**，这是更稳的写法，推荐直接用。

**(c)** 通过 `hostRef` 暴露重绘入口。确认 `TimelineKernelView` 已把 `localHostRef` 同步给外部 `hostRef`（面板用 `kernelHostRef` 访问宿主）；若没有，补一个：

```ts
    // 把本地宿主句柄同步给面板：面板的播放头桥接需要在每帧请求重绘
    // （见 timelineKernelHost.invalidatePlayhead 的说明）。
    React.useEffect(() => {
        if (hostRef !== undefined) hostRef.current = localHostRef.current;
    });
```

- [ ] **Step 7: 面板每帧请求重绘**

打开 `frontend/src/components/layout/TimelinePanel.tsx`，找到 `TimelineTransportBridge` 的 `onFrame` 回调（约 `:270-326`）。

**(a)** 给 `TimelineTransportBridge` 的 props 增加一个重绘回调：

```ts
    /** 请求时间轴内核重绘播放头（见 kernelHost.invalidatePlayhead 的说明）。 */
    requestPlayheadRepaint: () => void;
```

并在解构里取出。

**(b)** 在 `onFrame` 内**写完所有播放头样式之后**调用它。找到 `onFrame` 里最后一段（`if (rulerPlayheadHeadRef.current) { ... }` 之后），加入：

```ts
                // 内核自绘的轨道区播放头只有一次 rAF 内的样式写入机会；
                // 渲染循环是纯脏标记驱动的，不主动请求就永不重绘
                // （#2 的根因：点标尺后标尺播放头动了、网格播放头不动）。
                requestPlayheadRepaint();
```

**(c)** 在 `onFrame` 的依赖数组里加入 `requestPlayheadRepaint`。

**(d)** 在 `<TimelineTransportBridge …/>` 处（约 `:5054`）传入：

```ts
                        requestPlayheadRepaint={handleRequestPlayheadRepaint}
```

**(e)** 在面板里定义它（挨着 `getVisualPlayheadSec`，约 `:520`）：

```ts
    /**
     * 请求时间轴内核重绘播放头。
     *
     * 【为什么走 hostRef 而不是 React state】播放头每帧都可能移动，走 React 会
     * 把 60fps 的更新灌进渲染；这里只做一次命令式标脏。
     *
     * 特殊说明：**不是** `invalidateScene()`——那个会置 `sceneDirty` 并在下一帧
     * 重建全部 GPU 几何，播放中每帧重建等于自毁性能（见宿主接口说明）。
     */
    const handleRequestPlayheadRepaint = React.useCallback(() => {
        kernelHostRef.current?.invalidatePlayhead();
    }, []);
```

- [ ] **Step 8: 删除孤儿 `playheadRef` / `scrollRef` 写点**

这些写点**永远是 no-op**（没有任何 JSX 挂载它们），但它们看起来像活写入点，会误导后续排查（本次排查就被误导过）。用 `tsc` 的 `noUnusedLocals` 找出全部残留：

```bash
cd frontend && grep -n "playheadRef" src/components/layout/TimelinePanel.tsx src/components/layout/timeline/hooks/useTimelineState.ts
```

逐个删除：
1. `TimelinePanel.tsx`：`TimelineTransportBridge` 的 `playheadRef` prop 与解构、`onFrame` 里的 `if (playheadRef.current) { … }` 块、`useLayoutEffect`（约 `:4013-4022`）整段、JSX 传参（约 `:5056`）。
2. `useTimelineState.ts`：`playheadRef` 的声明（`:395`）、类型字段（`:155`）、写点（`:516-517`、`:1141-1142`）、返回值（`:1348`）。
3. `scrollRef` 同理：先删 `TimelinePanel.tsx:606` 与 `:4078-4083` 的 `scroller` 分支，再跑 `tsc` 看还有哪些孤儿。

每删一批就跑一次：

```bash
cd frontend && npx tsc -b --noEmit
```

Expected: 逐步收敛到无输出（`TS6133` 会指出剩余孤儿）。

> **顺序很重要**：先删**写点**，让 `tsc` 指出剩余声明，再删声明。反过来会一次报出一大堆
> 无关错误。

- [ ] **Step 9: 真机验证（标尺路径）**

**用 store 值 + 元素 transform 双读数**，才能把 #2（不重绘）与 #3（store 没变）分开——只读 transform 会把两者混为一谈（本次排查就踩过这个坑）。

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t3.png 6000 '[
{"type":"eval","js":"return (async()=>{const m=await import(\"/src/app/store.ts\");window.__store=m.store;const host=document.querySelector(\"[data-hs-timeline-kernel]\");window.__ph=()=>{const e=[...host.querySelectorAll(\"div\")].find(d=>d.className.includes(\"bg-qt-playhead\"));return e?e.style.transform:null;};window.__st=()=>Math.round(window.__store.getState().session.playheadSec*1000)/1000;return {INIT:{ph:window.__ph(), store:window.__st()}};})();"},
{"type":"click","x":900,"y":88},
{"type":"wait","ms":900},
{"type":"eval","js":"return {A_RULER_CLICK_1:{ph:window.__ph(), store:window.__st()}};"},
{"type":"click","x":1300,"y":88},
{"type":"wait","ms":900},
{"type":"eval","js":"return {B_RULER_CLICK_2:{ph:window.__ph(), store:window.__st()}};"}
]'
```

Expected（本任务完成后）：A、B 两次的 **store 都变化**，且 **`ph` 也跟着变**、且两次 `ph` 互不相同。

> 关键：**store 变化而 ph 不变**才是 #2 的证据。修复前实测 A/B 的 `ph` 恒为
> `translateX(1875px)`（store 却从 12.5 → 4.293 → 6.96）。

- [ ] **Step 10: 播放路径验证（自动滚屏关闭）**

自动滚屏关闭是出厂默认（`backend/src-tauri/src/config.rs` 的 `auto_scroll: false`），必须单独验证——开启时滚动会顺带标脏，从而**掩盖**这个缺陷。

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t3b.png 6000 '[
{"type":"eval","js":"const host=document.querySelector(\"[data-hs-timeline-kernel]\");window.__kh=()=>{const e=[...host.querySelectorAll(\"div\")].find(d=>d.className.includes(\"bg-qt-playhead\"));return e?e.style.transform:null;};const btns=[...document.querySelectorAll(\"button[data-tooltip]\")];const auto=btns.find(b=>b.getAttribute(\"data-tooltip\").includes(\"自动滚屏\"));if(auto&&auto.className.includes(\"solid\"))auto.click();return {INIT:window.__kh(), autoScrollWasOn: !!(auto&&auto.className.includes(\"solid\"))};"},
{"type":"key","key":"Space"},
{"type":"wait","ms":2500},
{"type":"eval","js":"return {D_after_PLAY_2_5s: window.__kh()};"},
{"type":"key","key":"Space"}
]'
```

Expected：`D` 与 `INIT` **不同**（播放中播放头在动）。若相同，说明标脏仍未生效。

- [ ] **Step 11: 全量测试 + 静态检查**

```bash
cd frontend && npx vitest run 2>&1 | tail -5
cd frontend && npx tsc -b --noEmit
cd frontend && npx eslint src --max-warnings=99 2>&1 | tail -5
```

Expected：测试 `2 failed | 849 passed`（Task 3 新增 5 项）；tsc 无输出；eslint 0 error。

- [ ] **Step 12: 提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A frontend/src
git commit -m "fix(timeline): 播放头变化主动标脏内核重绘 + 删除孤儿 playheadRef"
```

---

## Task 3b: 点空白不取消选中 / 播放头不变（缺陷 3，两个独立缺陷）

**为什么单独一个 Task**：初查以为 #3 与 #2 同源，**实测推翻了**。用一个探测同时读 store 值与元素 transform，三次点击给出：

| 操作 | store `playheadSec` | 内核播放头 `transform` |
| --- | --- | --- |
| 点标尺 | 12.5 → **4.293** ✅ | 1875 → **1875px** ❌（＝#2 不重绘） |
| 再点标尺 | → **6.96** ✅ | 仍 **1875px** ❌ |
| 点轨道空白 | 6.96 → **6.96** ❌（＝store 没变） | → **1044px** ⚠️（反而重绘了） |

最后一行同时说明：**"点空白播放头不动"不是因为不重绘，而是因为 store 值压根没变**——即 #3 的两个症状各有独立根因，与 #2 无关。

**Files:**
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`
- Modify: `frontend/src/dev/mockBackend.ts`（仅为可验证性）

- [ ] **Step 1: 补齐内核 seek 的乐观写（症状：播放头不变）**

打开 `frontend/src/components/layout/TimelinePanel.tsx` 的 `handleKernelSeek`（约 `:1190-1226`）。

**根因**：内核路径只派发 `seekPlayhead(sec)`，而 `seekPlayhead.fulfilled`
（`sessionSlice.ts:5507-5520`）**只在**「前端值仍等于请求值 **且** 后端返回值与请求值差 > 0.001」时才写 `state.playheadSec`。真实后端 `commands/core.rs:115,155` 返回 `playhead_sec = v.max(0.0)`——非负请求下两者相等，**写入被跳过**。该 reducer 的注释写着"否则保持前端**已同步设好的** `state.playheadSec`"，即它**假定**调用方已经乐观写过；内核路径正是漏了这一步。

**(a)** 在 `commit=true` 分支里加乐观写：

```ts
            if (commit) {
                if (kernelSeekRafRef.current != null) {
                    cancelAnimationFrame(kernelSeekRafRef.current);
                    kernelSeekRafRef.current = null;
                }
                kernelSeekPendingRef.current = null;
                // 乐观写必须与 seekPlayhead 成对：`seekPlayhead.fulfilled` 只在
                // 「后端返回值与请求值不同」时才采纳后端值，而后端对非负请求原样
                // 返回（`playhead_sec = v.max(0.0)`），因此**缺少这一步时 store 的
                // playheadSec 根本不会变** —— 表现为"点空白播放头不动"。
                // 与标尺路径同一契约（见 `useTimelineState.setPlayheadFromClientX`）。
                dispatch(setplayheadSec(sec));
                void dispatch(seekPlayhead(sec));
                return;
            }
```

**(b)** 在 rAF 节流的拖拽分支里同样加（**两条分支都要**）：

```ts
            kernelSeekRafRef.current = requestAnimationFrame(() => {
                kernelSeekRafRef.current = null;
                const target = kernelSeekPendingRef.current;
                if (target == null) return;
                kernelSeekPendingRef.current = null;
                // 同 commit 分支：拖拽帧也必须乐观写，否则拖拽期间播放头不跟手。
                dispatch(setplayheadSec(target));
                void dispatch(seekPlayhead(target));
            });
```

**(c)** 在 import 里确认 `setplayheadSec` 已引入（若只有 `seekPlayhead`，补上）：

```bash
cd frontend && grep -n "setplayheadSec" src/components/layout/TimelinePanel.tsx | head -3
```

**(d)** 顺带核对同类缺陷：`handleKernelActivateTake`（约 `:3183`）也是只派发
`seekPlayhead`。若它同样期望播放头跟随，一并补 `setplayheadSec`；若语义上不需要
（例如切换 Take 只是顺带定位），在注释里写明**为什么可以不补**——不要默默放过。

- [ ] **Step 2: 恢复 `applySelectedClip: false` 契约（症状：不取消选中）**

**根因**：空白点击的取消选中是**纯本地**的（`useTimelineClipActions.ts:803-809`），
不通知后端，因此后端永远记着 `selected_clip_id`；当点击行 ≠ 当前轨时派发的
`selectTrackRemote` 用**纯字符串**形式，使其 fulfilled 用后端快照**复活**了刚清空的选中。

**(a)** 改 `TimelinePanel.tsx:1203`：

```ts
                if (
                    trackId != null &&
                    sessionRef.current.paramEditorTimelineClickSelectTrackEnabled &&
                    trackId !== sessionRef.current.selectedTrackId
                ) {
                    // `applySelectedClip: false` 是**必须**的，不是可选优化：
                    // 取消选中只发生在前端，后端仍记着旧 selected_clip_id；而
                    // `selectTrackRemote.fulfilled` 默认会用后端快照覆盖前端选中
                    // （`sessionSlice.ts` 的 applySelectedClip 闸门）。传纯字符串
                    // 会让 `typeof arg !== "object"` 判为"要恢复"，把刚清掉的选中
                    // 复活 —— 表现为"点空白切轨时选中没被取消"。
                    //
                    // 该契约由 019e93ed 建立（当时两处 DOM 调用点都传了
                    // applySelectedClip:false），内核补全时被写成纯字符串而失效。
                    // 修改此处前请先读该提交的历史。
                    void dispatch(
                        selectTrackRemote({ trackId, applySelectedClip: false }),
                    );
                }
```

**(b)** 核对其它调用点，**不要一刀切**：

```bash
cd frontend && grep -rn "selectTrackRemote(" src --include=*.ts --include=*.tsx | grep -v "thunks/"
```

对每个调用点判断它是否**期望**恢复后端选中：
- 轨道头的"点击切换当前轨道"、Alt+方向键切轨——用户意图是**只换轨道**，通常也应传
  `applySelectedClip: false`；
- 但若某处语义确实是"切轨并恢复该轨上次选中的 clip"，则保持纯字符串并在注释写明。

逐个决策并在注释里记下理由。

- [ ] **Step 3: 让 mock 能暴露这个缺陷**

`?mock=1` 的 `select_clip` / `select_track` 是 `() => ({ok:true})`
（`mockBackend.ts:418-419`），**不含 `selected_clip_id`**——而复活闸门要求
`payload.selected_clip_id !== undefined`，所以 **mock 会掩盖本缺陷**（这就是初查一直
复现不出来的原因）。

打开 `frontend/src/dev/mockBackend.ts`，给这两个桩补上真实后端会返回的字段。
先读 `buildTimelineState`（同文件内已有返回完整快照的函数）复用它：

```ts
        // 与真实后端同形：`to_payload()` **总是**带 selected_clip_id，而
        // `select_track` 刻意**不**清它（见 backend/src-tauri/src/state.rs）。
        // 若这里返回 `{ok:true}`，`selectTrackRemote.fulfilled` 的
        // `applySelectedClip` 闸门会因为字段缺失而永不触发，从而**掩盖**
        // "点空白切轨时选中被复活"的缺陷（本次排查因此一度无法复现）。
        select_clip: (clipId: string | null) => {
            mockSelectedClipId = clipId;
            return buildTimelineState();
        },
        select_track: (trackId: string) => {
            mockSelectedTrackId = trackId;
            // 刻意不改 mockSelectedClipId —— 与真实后端一致。
            return buildTimelineState();
        },
```

> **实现提示**：`buildTimelineState()` / `mockSelectedClipId` 等的实际名字以文件现状为准；
> 若现有代码用闭包内局部变量持有选中态，就在同处加可写变量，并让快照返回它。**保持改动
> 最小**，目标只是"让 mock 与实际后端在这两个方法上同形"。

- [ ] **Step 4: 真机验证（两个症状各一条）**

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t3b.png 6000 '[
{"type":"eval","js":"return (async()=>{const m=await import(\"/src/app/store.ts\");window.__store=m.store;const host=document.querySelector(\"[data-hs-timeline-kernel]\");window.__ph=()=>{const e=[...host.querySelectorAll(\"div\")].find(d=>d.className.includes(\"bg-qt-playhead\"));return e?e.style.transform:null;};window.__st=()=>{const s=window.__store.getState().session;return {ph:Math.round(s.playheadSec*1000)/1000, sel:s.selectedClipId, multi:s.multiSelectedClipIds.length, trk:s.selectedTrackId};};return {INIT:window.__st()};})();"},
{"type":"click","x":700,"y":150},
{"type":"wait","ms":900},
{"type":"eval","js":"return {A_click_clip_on_track1: window.__st()};"},
{"type":"click","x":400,"y":150},
{"type":"wait","ms":900},
{"type":"eval","js":"return {★B_blank_SAME_track_MUST_clear_and_seek: window.__st()};"},
{"type":"click","x":700,"y":150},
{"type":"wait","ms":900},
{"type":"click","x":400,"y":230},
{"type":"wait","ms":900},
{"type":"eval","js":"return {★C_blank_OTHER_track_MUST_clear_and_seek: window.__st()};"}
]'
```

Expected：
- **B**：`sel === null`、`multi === 0`（清空生效），且 `ph` **与 A 时不同**（seek 生效）；
- **C**：同上，且 `trk` 切到 `track-2`；
- B、C 的 `ph` 应互不相同。

> 修复前实测：B 的 `ph` 不变（6.96 → 6.96），C 的 `sel` 被复活。

- [ ] **Step 5: 端到端再确认 `applySelectedClip` 真的生效**

仅看前端 store 不够——本缺陷的关键是**异步 fulfilled 覆盖**。加长等待时间再读一次，
确保不是"看起来对了但异步又被打回去"：

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t3b2.png 6000 '[
{"type":"eval","js":"return (async()=>{const m=await import(\"/src/app/store.ts\");window.__store=m.store;window.__sel=()=>window.__store.getState().session.selectedClipId;return {INIT:window.__sel()};})();"},
{"type":"click","x":700,"y":150},
{"type":"wait","ms":900},
{"type":"click","x":400,"y":230},
{"type":"wait","ms":2000},
{"type":"eval","js":"return {★D_after_2s_settle_MUST_be_null: window.__sel()};"}
]'
```

Expected：`★D` 为 `null`。修复前（在带 `selected_clip_id` 的忠实桩下）会是 `track-1-clip-*`。

- [ ] **Step 6: 门禁 + 提交**

```bash
cd frontend && npx vitest run 2>&1 | tail -5
cd frontend && npx tsc -b --noEmit
```

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A frontend/src
git commit -m "fix(timeline): 内核 seek 补乐观写 + 恢复 applySelectedClip 契约（点空白的选中与播放头）+ mock 补 selected_clip_id"
```

---

## Task 4: 主画布内容签名改为引用比较（缺陷 4）

**这是本计划风险最高的一项**：签名比较语义变更会影响主画布的**每一个**缓存图层。

**Files:**
- Create: `frontend/src/components/layout/pianoRoll/mainCanvasSignature.ts`
- Create: `frontend/src/components/layout/pianoRoll/mainCanvasSignature.test.ts`
- Modify: `frontend/src/components/layout/pianoRoll/render.ts`
- Modify: `frontend/src/components/layout/PianoRollPanel.tsx`
- Modify: `frontend/src/components/layout/pianoRoll/usePianoRollInteractions.ts`

- [ ] **Step 1: 写失败测试**

创建 `frontend/src/components/layout/pianoRoll/mainCanvasSignature.test.ts`：

```ts
/**
 * 主画布内容签名单测。
 *
 * 【为什么必须有这一层】签名是"要不要重绘画布"的唯一判据，而它此前用
 * `[...].join("|")` 构造——`join` 会把每个对象/数组元素串成字面量
 * `"[object Object]"`，于是**两个完全不同的选区产生同一个签名**，缓存命中、
 * 旧选区框留在画布上不消失（用户的 #4 报告）。
 *
 * 本工程此前对签名层**零覆盖**（没有任何测试 import `drawPianoRoll`，也没有
 * 测试引用签名），因此这个缺陷能一路漏到真机。本文件是它的守卫。
 */
import { describe, expect, it } from "vitest";

import { isSameMainCanvasSignature } from "./mainCanvasSignature";

describe("isSameMainCanvasSignature", () => {
    it("引用相同 → 视为相同（空闲帧走缓存快路径）", () => {
        const selection = { aBeat: 1, bBeat: 2 };
        const a = [1920, 1.5, selection];
        const b = [1920, 1.5, selection];
        expect(isSameMainCanvasSignature(a, b)).toBe(true);
    });

    it("★ 回归守卫：内容不同的选区对象 → 必须判为不同", () => {
        // 这是 #4 的正向复现：若实现退回 join("|")，两项都会变成
        // "[object Object]" 而误判为相同，本断言随即失败。
        const a = [1, { aBeat: 4.1, bBeat: 6.7 }];
        const b = [1, { aBeat: 40, bBeat: 90 }];
        expect(isSameMainCanvasSignature(a, b)).toBe(false);
    });

    it("★ 回归守卫：选区从对象变为 null → 必须判为不同", () => {
        expect(isSameMainCanvasSignature([1, { aBeat: 1, bBeat: 2 }], [1, null])).toBe(false);
    });

    it("★ 回归守卫：对象数组内容不同的元素 → 必须判为不同", () => {
        const a = [[{ x: 1 }, { x: 2 }]];
        const b = [[{ x: 9 }, { x: 8 }]];
        expect(isSameMainCanvasSignature(a, b)).toBe(false);
    });

    it("长度不同 → 不同", () => {
        expect(isSameMainCanvasSignature([1, 2], [1, 2, 3])).toBe(false);
    });

    it("原始值变化 → 不同（滚动/缩放必须失效缓存）", () => {
        expect(isSameMainCanvasSignature([100, "pitch"], [101, "pitch"])).toBe(false);
    });

    it("数值 NaN 与自身：Object.is 语义下视为相同（避免 NaN 每帧刷帧）", () => {
        expect(isSameMainCanvasSignature([Number.NaN], [Number.NaN])).toBe(true);
    });

    it("首次比较（上次为 undefined）→ 不同", () => {
        expect(isSameMainCanvasSignature([1, 2], undefined)).toBe(false);
    });
});
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd frontend && npx vitest run src/components/layout/pianoRoll/mainCanvasSignature.test.ts
```

Expected: FAIL —— 无法解析 import。

- [ ] **Step 3: 写纯模块实现**

创建 `frontend/src/components/layout/pianoRoll/mainCanvasSignature.ts`：

```ts
/**
 * 参数编辑器 · 主画布内容签名（纯函数）
 *
 * 【主要内容】
 * 定义主画布（曲线 / 选区 / morph 手柄 / 剪贴板预览 / 中央提示文字）内容签名的
 * 类型与比较规则。
 *
 * 【作用：为什么要独立成模块】
 * 签名是"要不要重绘主画布"的**唯一判据**，漏项或误判的代价都是"图层停止更新"。
 * 而它此前在 `PianoRollPanel` 内以 `[...].join("|")` 构造，`join` 会把每个
 * 对象/数组元素串成字面量 `"[object Object]"`——于是
 *
 *     [{aBeat:4.1, bBeat:6.7}] 与 [{aBeat:40, bBeat:90}]  →  同一个签名
 *
 * 缓存命中、drawPianoRoll 在清屏前就 return，**新选区框画不出来、旧框不消失**；
 * 只有 `null ↔ object` 的转换能改变签名，这精确解释了"第一次选区可见、再次选择
 * 不更新，拖到边缘自动滚屏时才突然更新"（滚动量是真正的签名原语）。
 *
 * 顺带修复的同类塌缩项：`paramMorphOverlay`（morph 手柄拖拽同样静默失效，
 * 实测 3 点 → 9 点签名相同）、`paramViewsRef.current`、
 * `liveEditOverrideRef.current`、`detectedPitchCurves`、`referencePitchOverlays`、
 * `secondaryParamViews`。
 *
 * 独立成模块的另一个原因：**本工程 vitest 跑在 node 环境且不允许 import `.tsx`**
 * （会经 barrel 间接引入 Redux / localStorage 而崩溃），放在面板里就等于不可测。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel` 每帧构造签名；`render.ts` 的 `drawPianoRoll` 消费它。
 * - 独立性：纯函数 + 类型，无 React / DOM / Redux 依赖，可在 node 环境单测。
 *
 * 【设计约束】
 * 1. 用 `Object.is` 逐项比较，**不是**字符串化：对象/数组按**引用**参与。
 * 2. 引用比较能逐帧失效，是因为选区在被拖拽时每次都被赋一个**新对象**
 *    （`usePianoRollInteractions` 的 `selectionRef.current = {…}`）。
 *    调用方不得改成原地改字段——那会让引用保持不变、缓存永不失效。
 * 3. `Object.is` 下 `NaN` 与自身相等，正好避免"某输入恒为 NaN 时每帧刷帧"。
 */

/**
 * 主画布内容签名。
 *
 * 元素可以是原始值（数值 / 字符串 / 布尔）或**任意对象引用**；对象一律按引用
 * 参与比较，不做深比较（深比较会进入每帧热路径）。
 */
export type MainCanvasSignature = readonly unknown[];

/**
 * 比较两个主画布内容签名是否等价。
 *
 * 流程：任一侧缺失 → 不等价；长度不同 → 不等价；逐项 `Object.is` 比较。
 *
 * 特殊说明：`Object.is` 而不是 `===`，因为它对 `NaN` 的处理符合本场景需要
 * （`NaN` 视为与自身相同，避免某个输入恒为 NaN 时每帧刷帧；而 `-0`/`+0`
 * 的区分在本场景无影响）。
 *
 * @param next 本帧签名。
 * @param previous 上一帧签名；`undefined` 表示尚未绘制过。
 * @returns 等价（可走缓存、跳过重绘）时为 true。
 */
export function isSameMainCanvasSignature(
    next: MainCanvasSignature | undefined,
    previous: MainCanvasSignature | undefined,
): boolean {
    if (next === undefined || previous === undefined) return false;
    if (next.length !== previous.length) return false;
    for (let index = 0; index < next.length; index += 1) {
        if (!Object.is(next[index], previous[index])) return false;
    }
    return true;
}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd frontend && npx vitest run src/components/layout/pianoRoll/mainCanvasSignature.test.ts
```

Expected: PASS，8 passed。

- [ ] **Step 5: `render.ts` 的缓存改为签名数组**

打开 `frontend/src/components/layout/pianoRoll/render.ts`。

**(a)** 改模块级缓存类型（约 `:69`）：

```ts
const mainCanvasCache = new WeakMap<HTMLCanvasElement, MainCanvasSignature>();
```

**(b)** 改 `drawPianoRoll` 的参数类型。把 `mainContentSignature?: string;` 改为：

```ts
    /**
     * 主画布的内容签名（静态层缓存）。
     *
     * 契约：调用方必须把"主画布绘制的全部输入"与视口都编进签名。**漏掉任何一项
     * 都会让该层停止更新**（表现为"改了参数但画面不动"）。
     *
     * 特殊说明：对象/数组项按**引用**参与比较，不做字符串化——字符串化会把它们
     * 塌缩成 "[object Object]"，让不同内容产生同一签名（这正是选区框残留的根因，
     * 见 `mainCanvasSignature.ts`）。`undefined` 表示不做缓存（每帧重绘）。
     */
    mainContentSignature?: MainCanvasSignature;
```

> 保留原有的 `skipCurves` / `skipPlayhead` 等 `skip*` 说明注释不动。

**(c)** 改缓存闸门（约 `:723-726`）：

```ts
    if (mainContentSignature !== undefined) {
        if (isSameMainCanvasSignature(mainContentSignature, mainCanvasCache.get(canvas))) return;
        mainCanvasCache.set(canvas, mainContentSignature);
    }
```

**(d)** 在文件顶部 import：

```ts
import {
    isSameMainCanvasSignature,
    type MainCanvasSignature,
} from "./mainCanvasSignature";
```

- [ ] **Step 6: `PianoRollPanel.tsx` 改为传数组并补齐缺失项**

打开 `frontend/src/components/layout/PianoRollPanel.tsx`，找到 `mainContentSignature`（约 `:3490-3545`）。

**(a)** 去掉 `.join("|")`，让签名保持数组：

```ts
        const mainContentSignature: MainCanvasSignature = [
            // …（原有各项保持原顺序，一个都不删）
        ];
```

**(b)** 在数组里补上注释已要求、但实际漏掉的两项（约 `:3504` 的注释明确要求它们在）：

```ts
            pitchAnalysisPending,
            overlayText,
```

> `overlayText` 在本文件里是内联三元表达式（约 `:3559` 的
> `overlayText: !pitchEnabled ? (editParam === "pitch" ? pitchHardDisableReason : childPitchHardDisableReason) : null`）。
> **实现时先把它提取成一个局部变量**，再同时用于签名与传给 `drawPianoRoll`——
> 两处各写一遍就是本文件注释反复警告的"签名里写了 A、实际喂给绘制的是 B"漂移。

**(c)** 更新那段签名注释：原文写"数据与几何（**引用比较**）"，但实际是字符串化。改成如实描述：

```ts
            // 数据与几何：对象 / 数组按**引用**参与（`Object.is`），不做字符串化。
            // 刻意与传给 drawPianoRoll 的字段一一对应，避免"签名里写了 A、
            // 实际喂给绘制的是 B"这种漂移。
```

**(d)** 在文件顶部 import：

```ts
import type { MainCanvasSignature } from "./pianoRoll/mainCanvasSignature";
```

- [ ] **Step 7: 补上 pointerdown 的 invalidate**

打开 `frontend/src/components/layout/pianoRoll/usePianoRollInteractions.ts`，找到新建选区分支（约 `:3415-3417`）：

```ts
                    const startBeat = selectionBeatFromClientX(e.clientX, false);
                    selectionRef.current = { aBeat: startBeat, bBeat: startBeat };
                    updateSelectionUi(selectionRef.current);
```

在 `updateSelectionUi(...)` 之后补一行：

```ts
                    // 按下瞬间就要重绘：新选区此时是零宽的，若不重绘，画布上会
                    // 留着**上一次**的选区框直到指针移动（用户报告的"旧选择框
                    // 不消失，拖拽时才更新"）。与 onMove(:3437) / onUp(:3445)
                    // 的 invalidate 对齐。
                    invalidate();
```

- [ ] **Step 8: 类型检查 + 全量测试**

```bash
cd frontend && npx tsc -b --noEmit
cd frontend && npx vitest run 2>&1 | tail -5
```

Expected：tsc 无输出；测试 `2 failed | 857 passed`（Task 4 新增 8 项）。

- [ ] **Step 9: 真机验证（选区 + morph 两类图层）**

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t4.png 6000 '[
{"type":"click","x":132,"y":289},
{"type":"wait","ms":600},
{"type":"eval","js":"const el=document.querySelector(\"[data-piano-roll-canvas]\");window.__snap=()=>el.toDataURL();return {toolIsSelect:true};"},
{"type":"move","x":400,"y":700},
{"type":"down"},
{"type":"move","x":700,"y":700},
{"type":"move","x":1000,"y":700},
{"type":"wait","ms":400},
{"type":"up"},
{"type":"wait","ms":600},
{"type":"eval","js":"window.__A=window.__snap();return {A_marquee1_rendered:true};"},
{"type":"move","x":1300,"y":700},
{"type":"down"},
{"type":"wait","ms":600},
{"type":"eval","js":"const b=window.__snap();return {★_SECOND_POINTERDOWN_CHANGED_CANVAS: b!==window.__A, stillHasRect: b.length!==window.__A.length};"},
{"type":"up"}
]'
```

Expected：`★_SECOND_POINTERDOWN_CHANGED_CANVAS` 为 **true**（缺陷下是 `false`——画布逐字节不变）。

再验证 morph 手柄（同一塌缩缺陷）：选中一段后再按住 morph 修饰键拖动，画面必须同帧跟随。若 `?mock=1` 没有音高曲线导致 morph 不可用，则**在真机工程里验证**，并在提交信息里注明验证环境。

- [ ] **Step 10: 逐图层回归确认（本任务的核心风险）**

签名比较语义变更会影响**每一个**缓存图层。逐个确认仍会更新：

| 图层 | 怎么触发 | 期望 |
| --- | --- | --- |
| 选区框 | 拖出新选区 | 立即出现、旧框立即消失 |
| morph 手柄 | 选中后按 morph 修饰键拖动 | 同帧跟随 |
| 剪贴板预览 | 复制一段后移动指针 | 虚线预览跟随 |
| 音阶高亮 | 切「音阶高亮」按钮 | 画面变化（Task 6 之前此项仍失效，属已知） |
| 曲线 | 切参数 / 缩放 | 正常重绘 |
| 中央提示文字 | 禁用音高（切到无音高的轨道组） | 文字出现/消失 |

对每一项各截一张图对比（`shotElement` + 哈希）。任一项不更新，说明签名项与绘制输入漂移，必须定位到具体项后才可继续。

- [ ] **Step 11: 提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A frontend/src
git commit -m "fix(pianoroll): 主画布签名改为引用比较（修 object 塌缩导致的选区框残留）+ 补 pointerdown 重绘"
```

---

## Task 5: MIDI / 音高参考块内容（缺陷 5，功能回归）

**背景**：`MidiPitchTrackCanvas.tsx` 在 `develop` 上有、现在**无任何消费者**——它唯一的挂载点 `TrackLane.tsx` 被本次改造删除了。数据管线完好（`clip.midiNoteData`、Redux `clipPitchCurves`），所以**只需要画**。

**Files:**
- Create: `frontend/src/components/layout/timeline/runtime/midiPitchCurve.ts`
- Create: `frontend/src/components/layout/timeline/runtime/midiPitchCurve.test.ts`
- Modify: `frontend/src/components/layout/timeline/runtime/timelineCanvasModel.ts`
- Modify: `frontend/src/components/layout/timeline/runtime/timelineCanvasRenderer.ts`
- Modify: `frontend/src/dev/mockBackend.ts`

- [ ] **Step 1: 从孤儿组件搬出纯数学**

读 `frontend/src/components/waveform/MidiPitchTrackCanvas.tsx`，把下列**纯函数**（无 React、无 DOM）**逐字搬进**新文件 `frontend/src/components/layout/timeline/runtime/midiPitchCurve.ts`：

- `resolveLoopCycleDescriptor`（约 `:84-108`）——Loop 回绕描述
- `generateMidiCurveFromNotes`（约 `:117-215`）——音符 → 逐帧音高曲线
- `strokeColorForClip`（约 `:48-50`）与 `CLIP_COLOR_TO_STROKE`（约 `:36-46`）
- 曲线缓存 `midiCurveCache` / `getCachedMidiCurve`（约 `:227-270`）
- `FRAME_PERIOD_MS`（约 `:34`）
- `LoopCycleDescriptor` 接口（约 `:52-76`）

搬出后 `MidiPitchTrackCanvas.tsx` 改为从新模块 import（**不要留两份**）。若确认该组件在本分支已无其他用途，也可直接删除它——**实现时二选一，并在提交信息里写明选了哪个及理由**。

新文件必须有完整的中文头注释，说明：这些数学**逐字来自 `MidiPitchTrackCanvas`**（与后端 `emit_clip_pitch_data_for_clip` 逐帧对齐，重写会引入相位差），以及它与 `utils/loopRender.ts` 共享 helper 的关系。

- [ ] **Step 2: 写失败测试**

创建 `frontend/src/components/layout/timeline/runtime/midiPitchCurve.test.ts`，覆盖：

```ts
/**
 * MIDI 音高曲线生成单测。
 *
 * 【为什么必须有】这些数学从 `MidiPitchTrackCanvas` 搬来，与后端
 * `emit_clip_pitch_data_for_clip` 的 MIDI 分支**逐帧对齐**。搬迁过程中任何一处
 * 写错都不会报错，只会让折线与音频/后端出现恒定相位差——极难归因。
 */
import { describe, expect, it } from "vitest";

import { generateMidiCurveFromNotes, resolveLoopCycleDescriptor } from "./midiPitchCurve";

const NOTES = [
    { startSec: 0, endSec: 1, note: 60 },
    { startSec: 1, endSec: 2, note: 64 },
];

describe("generateMidiCurveFromNotes", () => {
    it("基本铺放：音符区间内的帧取该音高，区间外为 0", () => {
        const curve = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 1, false, false, null);
        expect(curve.length).toBeGreaterThan(0);
        expect(curve[0]).toBe(60);
        expect(curve[curve.length - 1]).toBe(64);
    });

    it("trim（源窗口）只保留窗口内音符", () => {
        const curve = generateMidiCurveFromNotes(NOTES, 1, 1, 2, 1, false, false, null);
        expect(curve.every((v) => v === 64 || v === 0)).toBe(true);
        expect(curve.includes(60)).toBe(false);
    });

    it("playbackRate 拉伸：速率 2 时内容占一半帧数", () => {
        const normal = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 1, false, false, null);
        const fast = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 2, false, false, null);
        const lastNormal = normal.lastIndexOf(64);
        const lastFast = fast.lastIndexOf(64);
        expect(lastFast).toBeLessThan(lastNormal);
    });

    it("reversed 倒放：音高顺序反转", () => {
        const curve = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 1, true, false, null);
        expect(curve[0]).toBe(64);
    });

    it("fillGaps 填补空隙", () => {
        const gapped = [{ startSec: 0, endSec: 0.5, note: 60 }, { startSec: 1.5, endSec: 2, note: 67 }];
        const filled = generateMidiCurveFromNotes(gapped, 2, 0, 2, 1, false, true, null);
        expect(filled.every((v) => v > 0)).toBe(true);
    });

    it("无音符 / 非法入参不抛错", () => {
        expect(() => generateMidiCurveFromNotes([], 1, 0, 1, 1, false, false, null)).not.toThrow();
        expect(() =>
            generateMidiCurveFromNotes(NOTES, 0, 0, 0, 1, false, false, null),
        ).not.toThrow();
    });
});

describe("resolveLoopCycleDescriptor", () => {
    it("loopEnabled 为假 → null（走非循环路径）", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: false,
                contentDurationSec: 4,
                sourceStartSec: 0,
                sourceEndSec: 2,
            }),
        ).toBeNull();
    });

    it("有媒体时长：周期取媒体时长、cycleFromMedia 为真", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: 4,
                sourceStartSec: 1,
                sourceEndSec: 3,
            }),
        ).toEqual({
            cycleSec: 4,
            fwdAnchorSec: 1,
            // 倒放锚点 clamp 到媒体时长上界（此处 3 < 4，不变）。
            revAnchorEndSec: 3,
            cycleFromMedia: true,
        });
    });

    it("纯 MIDI（无媒体时长）：退化为窗口跨度、cycleFromMedia 为假", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: null,
                sourceStartSec: 2,
                sourceEndSec: 7,
            }),
        ).toEqual({
            cycleSec: 5,
            fwdAnchorSec: 2,
            // 退化时**不** clamp 到媒体时长（没有媒体），保留原始 sourceEnd。
            revAnchorEndSec: 7,
            cycleFromMedia: false,
        });
    });

    it("倒放锚点只 clamp 到媒体时长上界、不做 max(0)（负 sourceEnd 交给 floor_mod）", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: 2,
                sourceStartSec: -1,
                sourceEndSec: -0.5,
            })?.revAnchorEndSec,
        ).toBe(-0.5);
        // 上界方向确实被 clamp：
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 9,
            })?.revAnchorEndSec,
        ).toBe(2);
    });

    it("周期退化到 0（窗口跨度为零且无媒体）→ null", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: null,
                sourceStartSec: 3,
                sourceEndSec: 3,
            }),
        ).toBeNull();
    });
});
```

并在 import 处补上 `resolveLoopCycleDescriptor`：

```ts
import { generateMidiCurveFromNotes, resolveLoopCycleDescriptor } from "./midiPitchCurve";
```

- [ ] **Step 3: 跑测试**

```bash
cd frontend && npx vitest run src/components/layout/timeline/runtime/midiPitchCurve.test.ts
```

Expected: 先 FAIL（模块不存在），实现搬迁后 PASS。

- [ ] **Step 4: 给 clip 模型加可选字段**

打开 `frontend/src/components/layout/timeline/runtime/timelineCanvasModel.ts`。

**(a)** 在 `TimelineCanvasClipModel` 类型里加字段（挨着 `silenceSpansPx`）：

```ts
    /**
     * MIDI / 音高参考块的音高折线（相对 clip 左缘的视口坐标点，CSS px）。
     *
     * 【为什么是可选且缺省不绘制】只有 `clip.midiNoteCount != null` 的 clip 有
     * 音高内容（音频 clip 画波形，MIDI clip 画折线）。空数组与缺省等价。
     *
     * 特殊说明：y 已经映射到 clip body 的局部坐标（`0..bodyHeightPx`），
     * 与 `silenceSpansPx` 同一约定——绘制端不需要再知道音高值域。
     */
    midiPitchCurvePx?: Array<{ x: number; y: number }>;
```

**(b)** 新增一个模块内构造函数（模仿 `buildSilenceSpansPx` 的写法：无有效内容时返回 `undefined`，而不是空数组）：

```ts
/**
 * 构造 MIDI / 音高参考块 body 内的音高折线点。
 *
 * 流程：取该 clip 的音符数据（`midiNoteData`，缺失时回退 Redux 推来的
 * `clipPitchCurves`）→ 用 `generateMidiCurveFromNotes` 生成逐帧曲线 →
 * 按 `axis` 投影为相对 clip 左缘的 x → 把 MIDI 值线性映射到 body 高度内的 y。
 *
 * y 映射沿用搬迁前 `MidiPitchTrackCanvas` 的语义：上下各留 10% padding、
 * 高音在上（`displayH - padding - normalized * (displayH - 2*padding)`）。
 *
 * 特殊说明 1：`framePeriodMs` 必须与曲线生成时用的一致，否则 x 会整体缩放错位。
 * 特殊说明 2：帧步长按 `axis.pxPerSec` 抽稀——最小缩放下视口可覆盖数百秒，
 * 逐帧产点会有上万点。
 *
 * @returns 折线点；非 MIDI clip / 无数据时为 undefined。
 */
```

> **实现细节**：函数体按 `buildSilenceSpansPx` 同款风格写；签名与调用点（约 `:372`）
> 与 `silenceSpansPx` 并列。`clipPitchCurves` 的读取方式以面板现有传入为准——先读
> `buildSparseClipRenderModel` 的 `args` 类型，按实际通道接（若当前没有该通道，
> 则在 Task 5 里从面板透传，并在提交信息里注明）。

- [ ] **Step 5: 细节层绘制折线**

打开 `frontend/src/components/layout/timeline/runtime/timelineCanvasRenderer.ts`。

在 `drawClipDetails` 的最后一遍（`silenceOverlays` / `takeLaneSeparators` 旁边，约 `:981`）加一个收集数组与绘制段。沿用**同一种"最后一遍落笔"**的写法（理由同既有注释：避免被后续批次的块面盖掉）：

```ts
    /**
     * MIDI / 音高参考块的音高折线（同样在最后一遍落笔）。
     *
     * 数据源：`MidiPitchTrackCanvas`——它在旧实现（`TrackLane`）里被挂载，
     * 该组件随内核改造被删除后成为孤儿，MIDI clip 的 body 因此变成空白
     * （功能回归）。折线数学已搬进 `midiPitchCurve.ts` 复用，不在此重写。
     */
    const midiPitchCurves: Array<{
        points: Array<{ x: number; y: number }>;
        color: string;
        alpha: number;
        /** 该 clip 的 body 矩形（内容坐标），用于逐条裁剪。 */
        clip: { left: number; top: number; width: number; height: number };
    }> = [];
```

收集（在既有 for 循环里，与 `silenceSpansPx` 并列）：

```ts
        if (clip.midiPitchCurvePx !== undefined && clip.midiPitchCurvePx.length >= 2) {
            midiPitchCurves.push({
                points: clip.midiPitchCurvePx,
                color: resolveMidiCurveColor(clip.trackColor),
                alpha: clip.muted ? 0.4 : 0.85,
                clip: {
                    left: item.left,
                    top: item.bodyTop,
                    width: item.width,
                    height: item.bodyHeight,
                },
            });
        }
```

绘制：

```ts
    if (midiPitchCurves.length > 0) {
        // 折线在 clip body 内裁剪：音高映射已钳制，但投影异常或极窄 clip 下
        // 仍可能溢出到相邻行。**逐条**裁剪——多个 clip 的折线在同一遍落笔，
        // 一次全局 clip 只能覆盖其中一个。
        ctx.save();
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.lineWidth = 1.5;
        for (const curve of midiPitchCurves) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(curve.clip.left, curve.clip.top, curve.clip.width, curve.clip.height);
            ctx.clip();
            ctx.beginPath();
            ctx.globalAlpha = curve.alpha;
            ctx.strokeStyle = curve.color;
            for (let index = 0; index < curve.points.length; index += 1) {
                const point = curve.points[index];
                // 点坐标是**相对 clip 左缘 / body 顶部**的，绘制时加回原点。
                const x = curve.clip.left + point.x;
                const y = curve.clip.top + point.y;
                if (index === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.restore();
        }
        ctx.restore();
    }
```

`resolveMidiCurveColor` 用搬迁来的 `strokeColorForClip`（按 clip 颜色色相给折线配色）。

- [ ] **Step 6: 给 mock 补 `midi_note_data`（否则无法视觉验证）**

打开 `frontend/src/dev/mockBackend.ts`，找到构造 MIDI clip 的那一行
（`...(trackIndex === 2 && index === clipCount - 1 ? { midi_note_count: 8 } : {})`，约 `:249`）。

改为同时给出音符数据：

```ts
                // MIDI clip 必须同时给 `midi_note_data`：前端以
                // `midiNoteCount != null` 判定 MIDI，但折线内容来自 note 数据。
                // 只写 count 会让该 clip 渲染为空白，从而**无法验证**折线图层
                // （本次排查中一度因此无法视觉复现 #5 回归）。
                ...(trackIndex === 2 && index === clipCount - 1
                    ? {
                          midi_note_count: 8,
                          midi_note_data: [
                              { start_sec: 0, end_sec: 0.5, note: 60, velocity: 100, channel: 0 },
                              { start_sec: 0.5, end_sec: 1.0, note: 64, velocity: 100, channel: 0 },
                              { start_sec: 1.0, end_sec: 1.6, note: 67, velocity: 100, channel: 0 },
                              { start_sec: 1.6, end_sec: 2.2, note: 72, velocity: 100, channel: 0 },
                              { start_sec: 2.2, end_sec: 2.8, note: 69, velocity: 100, channel: 0 },
                              { start_sec: 2.8, end_sec: 3.4, note: 65, velocity: 100, channel: 0 },
                              { start_sec: 3.4, end_sec: 4.0, note: 62, velocity: 100, channel: 0 },
                              { start_sec: 4.0, end_sec: 4.6, note: 60, velocity: 100, channel: 0 },
                          ],
                      }
                    : {}),
```

- [ ] **Step 7: 真机验证**

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t5.png 6000 '[
{"type":"eval","js":"const host=document.querySelector(\"[data-hs-timeline-kernel]\");const cs=[...host.querySelectorAll(\"canvas\")];const c=cs[cs.length-1];window.__h=()=>{const d=c.toDataURL();let h=0;for(let i=0;i<d.length;i++)h=(h*31+d.charCodeAt(i))|0;return h;};return {hash: window.__h()};"},
{"type":"wait","ms":1200},
{"type":"shotElement","selector":"[data-hs-timeline-kernel]","path":"/tmp/t5-kernel.png"},
{"type":"eval","js":"return {hashAfter: window.__h()};"}
]'
```

然后**看 `/tmp/t5-kernel.png`**：轨 3 的最后一个 clip（MIDI）body 内必须出现一条**彩色折线**。

Expected：折线可见（缺陷下 body 完全空白）。同时确认音频 clip 仍只显示波形（没有多余的折线）。

- [ ] **Step 8: 提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A frontend/src
git commit -m "fix(timeline): 恢复 MIDI/音高参考块的音高折线（孤儿组件数学搬入细节层）+ mock 补 midi_note_data"
```

---

## Task 6: 音阶高亮 + 钢琴背景（缺陷 6 + 新功能）

**Files:**
- Modify: `frontend/src/components/layout/pianoRoll/colors.ts`
- Modify: `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.ts`
- Modify: `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.test.ts`
- Modify: `frontend/src/components/layout/pianoRoll/kernel/scene/gridView.ts`
- Modify: `frontend/src/components/layout/pianoRoll/kernel/scene/gridView.test.ts`
- Modify: `frontend/src/components/layout/PianoRollPanel.tsx`
- Modify: `frontend/src/components/layout/pianoRoll/render.ts`

- [ ] **Step 1: 加配色常量**

打开 `frontend/src/components/layout/pianoRoll/colors.ts`，在 `PianoRollColors` 接口里加：

```ts
    /** 黑键行背景带（钢琴背景：只压暗黑键行，白键行保持原背景）。 */
    readonly blackKeyRowBand: string;
    /** 音阶高亮：非 Tempo Map 分段路径的强调色。 */
    readonly scaleHighlight: string;
```

`DARK_COLORS` 加：

```ts
    // 钢琴背景：黑键行压暗一档。深浅两端都取克制值——网格弱线在深色主题下
    // 已是 rgba(255,255,255,0.05)，背景带过重会让网格线糊掉。
    blackKeyRowBand: "rgba(0,0,0,0.08)",
    scaleHighlight: "rgba(255,200,80,0.22)",
```

`LIGHT_COLORS` 加：

```ts
    blackKeyRowBand: "rgba(0,0,0,0.06)",
    scaleHighlight: "rgba(200,120,20,0.22)",
```

> **必须用 `rgba()` 写法**：`parseRgbaColor` 只认 `rgb()/rgba()`，hex 会被解析成
> **不透明洋红**（该文件 `:122-149` 已记录这个血泪教训，`whiteKey`/`blackKey` 就是
> hex，靠 `normalizeCssColor` 兜底）。这里直接写成 `rgba()` 省掉一次 DOM 探针。

- [ ] **Step 2: 写失败测试**

在 `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.test.ts` 末尾追加：

```ts
describe("钢琴背景（黑键行）", () => {
    const band = [0, 0, 0, 0.08] as const;

    it("只为黑键半音产出背景带（pc ∈ {1,3,6,8,10}）", () => {
        const items = buildPitchGridInstances({
            view: { center: 60, span: 12 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
            blackKeyRowBandRgba: band,
        });
        const bands = items.filter((item) => item.value < 0);
        // 一个八度内 5 个黑键；span=12 时可见区间覆盖约一个八度。
        expect(bands.length).toBeGreaterThan(0);
        for (const item of bands) {
            const pc = ((-item.value - 1) % 12) % 12;
            expect([1, 3, 6, 8, 10]).toContain(pc);
        }
    });

    it("★ 背景带必须排在所有网格线之前（FLAT 按缓冲顺序合成）", () => {
        const items = buildPitchGridInstances({
            view: { center: 60, span: 12 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
            blackKeyRowBandRgba: band,
        });
        const lastBandIndex = items.map((i) => i.value < 0).lastIndexOf(true);
        const firstLineIndex = items.map((i) => i.value >= 0).indexOf(true);
        expect(lastBandIndex).toBeLessThan(firstLineIndex);
    });

    it("背景带横跨整个视口宽、高为键高（不是线厚）", () => {
        const items = buildPitchGridInstances({
            view: { center: 60, span: 12 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
            blackKeyRowBandRgba: band,
        });
        for (const item of items.filter((i) => i.value < 0)) {
            expect(item.x).toBe(0);
            expect(item.w).toBe(800);
            // 键高 ≤ 1 时仍保底 1px（与键盘轴一致，避免缩到底部出现空洞）。
            expect(item.h).toBeGreaterThan(0);
        }
    });

    it("缺省不产背景带（非 pitch / 未提供颜色时行为不变）", () => {
        const items = buildPitchGridInstances({
            view: { center: 60, span: 12 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        });
        expect(items.every((item) => item.value >= 0)).toBe(true);
    });
});

describe("音阶高亮", () => {
    it("音阶音级额外产出一条 2 倍线厚的强调线", () => {
        const base = {
            view: { center: 60, span: 12 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        };
        const plain = buildPitchGridInstances(base);
        const highlighted = buildPitchGridInstances({
            ...base,
            scaleNotes: [0, 4, 7],
            scaleHighlightRgba: [1, 0.78, 0.31, 0.22],
        });
        expect(highlighted.length).toBeGreaterThan(plain.length);
        const emphasis = highlighted.filter((item) => item.rgba[3] === 0.22);
        expect(emphasis.length).toBeGreaterThan(0);
        expect(emphasis[0].h).toBeGreaterThan(plain[0].h);
    });

    it("scaleNotes 为空 / 缺省时不产强调线", () => {
        const base = {
            view: { center: 60, span: 12 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        };
        expect(buildPitchGridInstances(base).length).toBe(
            buildPitchGridInstances({ ...base, scaleNotes: [] }).length,
        );
    });
});
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd frontend && npx vitest run src/components/layout/pianoRoll/kernel/scene/gridInstances.test.ts
```

Expected: FAIL —— `blackKeyRowBandRgba` / `scaleNotes` 不在入参类型里（`TS2353`），或背景带断言失败。

- [ ] **Step 4: 实现背景带 + 音阶高亮**

打开 `frontend/src/components/layout/pianoRoll/kernel/scene/gridInstances.ts`。

**(a)** `PitchGridArgs` 加两个可选入参：

```ts
    /**
     * 黑键行背景带的 RGBA；缺省不绘制。
     *
     * 【为什么只压暗黑键行】黑/白键行交替是钢琴卷帘的通用视觉语法
     * （REAPER / Logic / Ableton 同做法），能让人一眼定位音高落点。只做黑键行
     * 是刻意的：白键行加提亮带在浅色主题下会与背景糊在一起，反而削弱对比。
     *
     * 特殊说明：**必须用 rgba() 写法**——底层 `parseRgbaColor` 只解析该格式，
     * hex 会被解析成不透明洋红（见 `colors.ts` 的 normalizeCssColor 说明）。
     */
    readonly blackKeyRowBandRgba?: Rgba;
    /** 音阶音级（pitch class 集合）；缺省不产强调线。 */
    readonly scaleNotes?: readonly number[];
    /** 音阶强调线的 RGBA；缺省不产强调线。 */
    readonly scaleHighlightRgba?: Rgba;
```

**(b)** 在 `buildPitchGridInstances` 里，**在网格线循环之前**插入背景带发射。
背景带复用与键盘轴同一套覆盖率修正（`buildRectInstances` 已是本文件内的私有函数，
直接调用即可——**不要**另写一份边界处理）：

```ts
    // ── 钢琴背景：黑键行背景带 ─────────────────────────────────────
    //
    // 【必须在网格线之前发射】宿主按下标顺序写入实例缓冲，而 FLAT 模式在
    // `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)` 下按缓冲顺序合成——排在后面就会
    // 把网格线盖住。
    //
    // 【为什么复用 buildRectInstances】它与键盘轴共用同一套设备像素覆盖率修正
    // （边界落在设备像素内部时拆出边界列并乘覆盖率）。另写一份必然分叉，表现为
    // 背景带边缘比键盘轴暗一档。
    if (args.blackKeyRowBandRgba !== undefined) {
        for (let midi = startMidi; midi <= endMidi; midi += 1) {
            if (!isBlackKey(midi)) continue;
            const top = valueToY(midi + 1, heightPx);
            const bottom = valueToY(midi, heightPx);
            const rowTop = Math.min(top, bottom);
            const rowBottom = Math.max(top, bottom);
            // 键高保底 1px：与键盘轴一致，缩到底部时不出现空洞。
            const rowHeight = Math.max(1, rowBottom - rowTop);
            items.push(
                ...buildRectInstances({
                    x0: 0,
                    x1: viewportWidthPx,
                    y0: rowTop,
                    y1: rowTop + rowHeight,
                    rgba: args.blackKeyRowBandRgba,
                    // 背景带是**大面积**填充，边界列按覆盖率着色即可；
                    // 传入 1 让 buildRectInstances 走整数设备像素路径。
                    dpr: 1,
                }),
            );
        }
    }
```

> **实现提示**：`valueToY(midi + 0.5, …)` 是网格线的取值（线在键的**中心**），
> 而键体范围是 `[valueToY(midi+1), valueToY(midi)]`——两者**不同**，不要混用。
> 背景带用键体范围。实现后必须目视确认背景带与网格线对齐（网格线应正好落在带内）。

**(c)** 音频高亮同理放在**网格线之后**（强调线要盖在普通线上）：

在网格线循环内，普通线 push 之后追加：

```ts
        // 音阶高亮：该音级额外叠一条 2 倍线厚的强调线（与 Canvas2D 旧实现
        // 同一语义：同一 y 上再 stroke 一次、lineWidth 更大）。
        if (scaleNoteSet !== null && scaleNoteSet.has(pc)) {
            const emphasisThickness = thickness * 2;
            items.push({
                x: 0,
                y: rectTopFromCenter(centerY, emphasisThickness),
                w: viewportWidthPx,
                h: emphasisThickness,
                rgba: args.scaleHighlightRgba!,
                value: midi,
            });
        }
```

并在循环前构造集合：

```ts
    const scaleNoteSet =
        args.scaleNotes !== undefined && args.scaleNotes.length > 0 && args.scaleHighlightRgba !== undefined
            ? new Set(args.scaleNotes.map((note) => ((note % 12) + 12) % 12))
            : null;
```

**(d)** 在文件顶部 import：

```ts
import { isBlackKey } from "../../utils";
```

> 确认 `pianoRoll/utils.ts` 导出了 `isBlackKey`（它导出的是 `isBlackKey(midi)`）。
> 该模块是**无依赖叶子模块**（其头注释明确这一点），因此可以在 node 单测里引用。

**(e)** 更新 `GridInstance.value` 的注释：背景带用 `value < 0` 编码（`-(midi + 1)`），
这样调用方与测试都能把背景带与网格线区分开。在背景带 push 里把
`value` 设为 `-(midi + 1)`——`push` 时用 `buildRectInstances` 返回的多个矩形，
需要给每个都标上：

```ts
            for (const rect of buildRectInstances({ … })) {
                items.push({ ...rect, value: -(midi + 1) });
            }
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd frontend && npx vitest run src/components/layout/pianoRoll/kernel/scene/gridInstances.test.ts
```

Expected: PASS（含原有全部用例）。

- [ ] **Step 6: 签名纳入新输入**

打开 `frontend/src/components/layout/pianoRoll/kernel/scene/gridView.ts`。

**(a)** `GridGeometrySignatureArgs` 加：

```ts
    /** 黑键行背景带颜色；缺省表示不绘制。 */
    readonly blackKeyRowBandRgba?: readonly number[] | undefined;
    /** 音阶音级集合；缺省表示不高亮。 */
    readonly scaleNotes?: readonly number[] | undefined;
    /** 音阶强调线颜色。 */
    readonly scaleHighlightRgba?: readonly number[] | undefined;
```

**(b)** `gridGeometrySignature` 的数组里追加（**漏项 = 切主题/切音阶不生效**）：

```ts
        args.blackKeyRowBandRgba?.join(",") ?? "",
        args.scaleNotes?.join(",") ?? "",
        args.scaleHighlightRgba?.join(",") ?? "",
```

**(c)** 更新该函数 doc 注释，把新增的三项列进"必须包含"的清单里。

- [ ] **Step 7: 签名单测**

在 `gridView.test.ts` 追加：

```ts
    it("★ 钢琴背景色参与签名（漏项会让切主题不生效）", () => {
        const base = { kind: "pitch", view: { center: 60, span: 12 }, absMin: 36, absMax: 96,
            viewportWidthPx: 800, viewportHeightPx: 600, dpr: 1,
            strongRgba: [1, 1, 1, 0.1], weakRgba: [1, 1, 1, 0.05] };
        const a = gridGeometrySignature({ ...base, blackKeyRowBandRgba: [0, 0, 0, 0.08] });
        const b = gridGeometrySignature({ ...base, blackKeyRowBandRgba: [0, 0, 0, 0.06] });
        expect(a).not.toBe(b);
    });

    it("★ 音阶音级参与签名（漏项会让切音阶不生效）", () => {
        const base = { kind: "pitch", view: { center: 60, span: 12 }, absMin: 36, absMax: 96,
            viewportWidthPx: 800, viewportHeightPx: 600, dpr: 1,
            strongRgba: [1, 1, 1, 0.1], weakRgba: [1, 1, 1, 0.05] };
        const off = gridGeometrySignature(base);
        const on = gridGeometrySignature({ ...base, scaleNotes: [0, 4, 7],
            scaleHighlightRgba: [1, 0.78, 0.31, 0.22] });
        expect(off).not.toBe(on);
    });
```

- [ ] **Step 8: 面板传入新输入**

打开 `frontend/src/components/layout/PianoRollPanel.tsx` 的 `buildGridSpec()`（约 `:2383`）。

在 `base` 对象里加（注意 `toRgba` 已在本函数内定义，直接复用）：

```ts
            // 钢琴背景（黑键行）：只对 pitch 生效，非 pitch 参数没有"黑键"概念。
            blackKeyRowBandRgba: toRgba(colors.blackKeyRowBand),
```

在 `kind !== "pitch"` 提前返回**之前**，把音阶高亮也接上：

```ts
            // 音阶高亮：与 Canvas2D 旧实现同一份判定（`scaleHighlightMode` 为
            // "always" 且有音阶时才高亮）。这是 #6 的修复点——该图层此前只画在
            // Canvas2D 的 `skipGrid` 分支里，而 `skipGrid` 恒为 true，
            // 于是按钮能按、Redux 会变、但画面上什么都不发生。
            scaleNotes:
                scaleHighlightMode === "always" && effectiveProjectScale !== undefined
                    ? resolveScaleNotes(effectiveProjectScale)
                    : undefined,
            scaleHighlightRgba: toRgba(colors.scaleHighlight),
```

> **作用域已确认**：`buildGridSpec` 是组件内函数声明，`s`（`:590`，`useAppSelector`
> 的 session 切片）与 `effectiveProjectScale`（`:591`，`useMemo`）都在其闭包内，
> 因此直接可用（`s.scaleHighlightMode` 已在 `:3211` / `:3537` 以同样方式使用）。
> 仍**不要**在函数内直接读 store——本文件所有 spec 输入都来自 render 期快照。

在 import 区确认已引入：

```ts
import { resolveScaleNotes } from "../../utils/musicalScales";
```

**(b)** 更新 `gridGeometrySignature` 调用处（`pianoRollKernelHost.ts:631` 的 `gridSignature`）把三个新字段透传——照抄该对象里既有字段的写法。

- [ ] **Step 9: 删除 `render.ts` 里已死的音阶高亮分支**

打开 `frontend/src/components/layout/pianoRoll/render.ts`，`if (!skipGrid && editParam === "pitch") { … }` 整段是**死代码**（`skipGrid` 恒为 `true`）。删除整段（含 `highlightActive` 局部量、`segmentNotesList`、两处强调 stroke），并删除因此不再被读取的 `args` 项。

**这一步必须小心**：`args` 里有些项在**其它**分支仍被使用。逐项核对：

```bash
cd frontend && grep -n "scaleHighlightMode\|projectScale\|scaleSegments" src/components/layout/pianoRoll/render.ts
```

只删除**仅**在这段死分支里出现的项；其余保留。删完跑：

```bash
cd frontend && npx tsc -b --noEmit && npx eslint src/components/layout/pianoRoll/render.ts
```

Expected: 无输出（`noUnusedLocals` 会指出漏删的局部量）。

**(b)** 同步删掉 `PianoRollPanel.tsx` 里传给 `drawPianoRoll` 的那几项（`toolMode`、`snapToggleHeld`、`pitchSnapUnit` 等——**先确认它们在 `drawPianoRoll` 里已无任何读取点**再删；`grep` 每个名字在 `render.ts` 里的出现次数，只有 1 次（即签名声明本身）的才是死参数）。

- [ ] **Step 10: 真机验证**

```bash
cd frontend && VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/t6.png 6000 '[
{"type":"shotElement","selector":"[data-piano-roll-gl-scene]","path":"/tmp/t6-before.png"},
{"type":"eval","js":"const el=document.querySelector(\"[data-piano-roll-gl-scene]\");window.__h=()=>{const d=el.toDataURL();let h=0;for(let i=0;i<d.length;i++)h=(h*31+d.charCodeAt(i))|0;return h;};return {before: window.__h()};"},
{"type":"click","x":254,"y":289},
{"type":"wait","ms":900},
{"type":"eval","js":"return {★_AFTER_SCALE_TOGGLE_CHANGED: window.__h()!==undefined};"},
{"type":"shotElement","selector":"[data-piano-roll-gl-scene]","path":"/tmp/t6-after.png"}
]'
```

然后**比对两张 PNG**：

- `/tmp/t6-before.png`：黑键行应有半透明暗带（钢琴背景），音阶高亮关闭时**没有**琥珀色强调线。
- `/tmp/t6-after.png`：应出现琥珀色强调线（音阶高亮开启）。
- 两图哈希必须**不同**（缺陷下实测逐字节相同）。

并且目视确认：**横向网格线仍清晰可辨**（这是 §2.8 的强制验收点——深色主题下弱线只有 `rgba(255,255,255,0.05)`，背景带可能压掉它）。若网格线被压糊，把 `blackKeyRowBand` 的 alpha 调低一档再验。

- [ ] **Step 11: 全量门禁**

```bash
cd frontend && npx vitest run 2>&1 | tail -5
cd frontend && npx tsc -b --noEmit
cd frontend && npx eslint src --max-warnings=99 2>&1 | tail -5
cd frontend && npm run build 2>&1 | tail -5
grep -c "import.meta.env" dist/assets/*.js 2>/dev/null | head
```

Expected：测试 `2 failed | 865+ passed`（Task 6 新增 6 项）；tsc 无输出；eslint 0 error；构建成功；`import.meta.env` 计数为 0。

- [ ] **Step 12: 提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A frontend/src
git commit -m "feat(pianoroll): 黑键行钢琴背景 + 恢复音阶高亮（GL 网格层）+ 删除已死的 Canvas2D 分支"
```

---

## Task 7: 端到端验收与记录

**Files:**
- Create: `docs/superpowers/plans/2026-09-13-kernel-regression-fixes-acceptance.md`

- [ ] **Step 1: 逐条真机复核 6 项**

按 spec §6 的 7 条验收标准逐条跑一遍，把**实测值**（不是"通过"）记进验收文档。每一条都要有可复现的命令与读数。

- [ ] **Step 2: 回归对照 `develop`**

对 #5 与 #6 这两个**功能回归**，用 `git worktree` 在不打扰当前工作区的前提下对照
（**不要** `git checkout develop`，会丢掉当前工作区）：

```bash
cd /Users/guoqiangye/code/HiFiShifter
git worktree add /tmp/hfs-develop develop
# 在 /tmp/hfs-develop/frontend 起 dev server（用另一个端口，如 5175）
# 复现 #5/#6 在 develop 上的表现，截图留档
git worktree remove /tmp/hfs-develop
```

Expected：#5 在 `develop` 上 **有**音高折线；#6 在 `develop` 上**有**音阶高亮。把两份截图路径记进验收文档，作为"这是回归、不是从未实现"的证据。

> **注意**：`develop` 上 #5 需要 mock 也提供 `midi_note_data` 才能看到折线——若
> `develop` 的 mock 没有该字段，就在 worktree 里临时补上再截图（那个改动随 worktree
> 一起丢弃，不影响任何分支）。

- [ ] **Step 3: 跑 CPU-only 基准**

```bash
cd frontend && node scripts/cpu-render-bench.mjs
```

Expected：脚本正常产出（`MODE=legacy` 应被拒绝并退出码 2——那是上一轮修复的行为，本次不应回退）。记录新增实例数（钢琴背景 + 音阶高亮会小幅增加网格实例数）。

- [ ] **Step 4: 写验收文档**

照 `docs/superpowers/plans/2026-09-13-timeline-single-path-acceptance.md` 的体例写：逐条实测值 + 复现命令 + 未决项。

- [ ] **Step 5: 提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add docs/superpowers/plans/2026-09-13-kernel-regression-fixes-acceptance.md
git commit -m "docs(plan): 内核回归修复的真机验收记录"
```

---

## 完成后必做（不在任何 Task 内，但属于交付）

- [ ] **更新 spec 状态**：把 `2026-09-13-kernel-regression-fixes-design.md` 头部的
      `状态：待实施` 改为 `已实施（见 docs/superpowers/plans/2026-09-13-kernel-regression-fixes.md）`。
- [ ] **更新前置 spec**：在 `2026-09-13-timeline-single-path-design.md` 里补一句指向本次
      修复，说明"该改造引入的 2 个功能回归已由后续设计修复"——否则那份文档会一直声称
      改造完全成功。
- [ ] **陈旧注释总检**：`grep -rn "旧实现\|已随\|唯一路径" frontend/src/components/layout/timeline/ frontend/src/components/layout/TimelinePanel.tsx`
      逐条确认仍成立。本次改动删掉了若干"旧实现如何如何"的对照注释，凡指向已删文件/已删
      分支的描述都必须收敛。
- [ ] **不合并、不推送**：合并到 `develop` 需用户明确授权（用户此前为"先不合"）。
