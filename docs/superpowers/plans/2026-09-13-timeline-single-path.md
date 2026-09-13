# 时间轴渲染内核收归唯一路径 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除已被破坏的旧时间轴实现与全部内核开关，使内核成为唯一渲染路径，并把"WebGL2 不可用"做成可自助排障的明确报错。

**Architecture:** 分四阶段推进：先让旧分支不可达并真机验证（回滚成本 = 一个三元表达式），再以 `tsc -b` 报告为权威清单删除孤儿 state/hook，然后删除旧组件文件与拆分共享纯函数，最后移除四个开关并实现 GL 失败界面。参数编辑器（PianoRoll）**不删** Canvas2D 渲染器——它在内核模式下仍是活的细节层。

**Tech Stack:** React 19 + TypeScript + Redux Toolkit + Radix Themes + Vite + Vitest（node 环境，无 jsdom）+ WebGL2 + Tauri 2。

**依据 spec：** `docs/superpowers/specs/2026-09-13-timeline-single-path-design.md`

---

## 执行前必读

1. **测试基线**：当前 `848 passed / 2 failed`（133 文件）。2 项失败是 `frontend/src/features/keybindings/keybindingMatch.test.ts` 的「默认 Shift 变体」与「Ctrl 微调变体」，**既有问题、与本分支无关**（已在 `develop` worktree 复现）。本计划完成后期望约 **823 passed / 2 failed**（`featureFlag.test.ts` 的 25 项随开关删除）。**验收判据是"除明确删除的用例外无新失败"，不是数字不降。**
2. **不要推送。** 用户明确要求只提交、不推送。分支 `feature/timeline-unified-render-kernel` **无 upstream**。
3. **本项目注释即设计记录**（见 `~/.dsh/AGENTS.md` §4/§5）：每个文件头必须有「主要内容 / 作用 / 与其他模块的关系」，每个关键函数必须有流程与特殊说明。**修改代码必须同步改注释**，陈旧注释算实质缺陷。
4. **dev server**：用户自己的 `tauri dev` 占 5173；本计划的浏览器验证用 **5174**（若未运行，见任务 0 步骤 3 启动）。
5. **注释里的行号**仅供定位；改动会让行号漂移，以代码内容为准。

---

## 文件结构

### 删除
| 路径 | 行数 | 职责（删除理由） |
| --- | --- | --- |
| `frontend/src/components/layout/timeline/TimelineScrollArea.tsx` | 392 | 旧实现的滚动容器（原生滚动），仅被 barrel 再导出 |
| `frontend/src/components/layout/timeline/TimelineSurface.tsx` | 183 | 旧实现的绘制面，仅被 barrel 再导出 |
| `frontend/src/components/layout/timeline/TimelineCanvasViewport.tsx` | 188 | 旧实现的 Canvas2D 视口，仅被 `TimelineSurface` + barrel |
| `frontend/src/components/layout/timeline/ClipItem.tsx` | 945 | 旧实现的 clip DOM 组件，仅被 `TrackLane` + barrel |
| `frontend/src/components/layout/timeline/TrackLane.tsx` | 1006 | 旧实现的轨道 DOM；**其纯函数须先拆出**（任务 3） |
| `frontend/src/components/layout/timeline/kernel/kernelMount.ts` + `.test.ts` | — | `enabled` 维度消失后退化为伪抽象（任务 4 步骤 8 简化并保留行为测试） |

### 新建
| 路径 | 职责 |
| --- | --- |
| `frontend/src/components/layout/timeline/trackOverlap.ts` | 由 clips 计算各 clip 前置重叠秒数（从 `TrackLane.tsx` 拆出的纯函数，保留原实现） |
| `frontend/src/components/layout/timeline/trackOverlap.test.ts` | 上述纯函数的单测 |
| `frontend/src/components/layout/renderKernel/gl/glDiagnostics.ts` | WebGL2 可用性探测（纯函数，node 环境可测） |
| `frontend/src/components/layout/renderKernel/gl/glDiagnostics.test.ts` | 上述探测的单测 |
| `frontend/src/components/layout/timeline/kernel/KernelUnavailableNotice.tsx` | GL 失败界面（标题 + 原因 + 排查清单 + 诊断信息复制） |
| `frontend/scripts/cpu-render-bench.mjs` | CPU-only 渲染正确性与性能基准 |

### 修改（关键）
| 路径 | 改动 |
| --- | --- |
| `frontend/src/components/layout/TimelinePanel.tsx` | 阶段 1 改分支为无条件；阶段 2 删孤儿 state；阶段 3 删 import 与旧 JSX；阶段 4 接入失败界面 |
| `frontend/src/components/layout/PianoRollPanel.tsx` | 移除三个开关（任务 4） |
| `frontend/src/components/layout/TimelineDisplaySettingsDialog.tsx` | 移除内核总开关（任务 4） |
| `frontend/src/components/layout/timeline/kernel/featureFlag.ts` | 删除（任务 4） |
| `frontend/src/dev/perfProject.ts` | 移除内核切换按钮（任务 4） |
| `frontend/src/i18n/*.ts`（5 语言） | 移除 3 条旧文案（任务 4）；新增失败界面文案（任务 5） |

---

## 任务 0：准备与基线记录

**Files:** 无改动（只记录基线）

- [ ] **步骤 1：确认工作树干净**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git status --short
```
期望：无输出（干净）。若非空，先弄清未提交改动再继续。

- [ ] **步骤 2：记录测试基线**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx vitest run 2>&1 | tail -5
```
期望：`Tests  2 failed | 848 passed (850)`，`Test Files  1 failed | 132 passed (133)`。把这两行抄到本次工作的记录里。

- [ ] **步骤 3：确认 dev server 在 5174**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5174/
```
期望：`200`。若失败，在后台启动（不要占用用户的 5173）：

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend && nohup npx vite --port 5174 > /tmp/vite5174.log 2>&1 &
sleep 6 && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5174/
```

- [ ] **步骤 4：确认真机截图能力可用**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/task0.png 4000 '[{"type":"eval","js":"return {kernel: !!window.__hfsKernel};"}]' 2>&1 | grep -E "^EVAL"
```
期望：`EVAL: {"kernel":true}`（内核默认开启）。

- [ ] **步骤 5：无需提交**（本任务只记录基线，不产生改动）

---

## 任务 1：旧分支不可达（阶段 1）

**关键：本任务之后若真机发现内核缺能力，回滚 = 还原一个三元表达式。**

**Files:**
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（分支在 `{kernelActive ? (` 处）

- [ ] **步骤 1：先记录改动前的行为（作为回归对照）**

在浏览器里记录内核模式下的关键真值，任务结束后必须一致：

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/before-task1.png 4500 '[
 {"type":"eval","js":"const h=window.__hfsKernel; const v=h.getViewport(); return {pxPerSec:v.pxPerSec, scrollLeft:v.scrollLeft, scrollTop:v.scrollTop};"},
 {"type":"eval","js":"document.querySelector(\"[data-timeline-scroller]\").focus(); return 1;"},
 {"type":"key","key":"PageDown"},{"type":"wait","ms":400},
 {"type":"eval","js":"const h=window.__hfsKernel; return {afterPageDown_top:+h.getViewport().scrollTop.toFixed(2)};"}
]' 2>&1 | grep -E "^EVAL"
```
把两行 `EVAL` 输出抄下来。期望形如：第一行 `{"pxPerSec":150,"rowHeight":...,"scrollLeft":0,"scrollTop":0}`，第二行 `afterPageDown_top` 为 `132`（这是 `max(H-20, floor(0.875H))` 在 H=151 下的结果）。

- [ ] **步骤 2：把条件分支改为无条件渲染内核分支**

打开 `frontend/src/components/layout/TimelinePanel.tsx`，找到这一行（内容精确匹配，行号可能漂移）：

```tsx
                    {kernelActive ? (
```

将其改为下面三行（把条件替换为恒真，**保留 `?`**，否则语法非法）：

```tsx
                    {/* 内核是唯一渲染路径（旧实现已移除，见
                        docs/superpowers/specs/2026-09-13-timeline-single-path-design.md）。
                        此处不再有运行期二选一。 */}
                    {/* eslint-disable-next-line no-constant-condition -- 阶段 1：条件恒真，旧分支暂留以便一行回滚（任务 2/3 删除） */}
                    {true ? (
```

**⚠️ 不要把 `{kernelActive ? (` 改成 `{(`，也不要删掉 `?`**：这段 JSX 是三元表达式，末尾还有 `) : (` 与 `)}`。删掉 `?` 会得到 `(A) : (B)`，`:` 失去配对对象 → 语法错误。（本计划初稿写的正是 `{(`，是错的；实现者改用 `{true ? (` 才对。）

**为什么用 `{true ? (`**：本阶段要的是"不可达但可一行回滚"——`true ?` 保留了旧分支，回滚就是把 `true` 换回 `kernelActive`。

**注意**：本步骤**只改判断条件那一行**，`</>`、`) : (` 与 `)}` 保持原位不动 —— 旧分支 JSX 仍然留在文件里但**不可达**（任务 2 才删）。**不要**在本步骤删除 `) : (` 及其后的旧分支，否则无法单独回滚本阶段。

- [ ] **步骤 3：类型检查与全量测试**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -10
npx vitest run 2>&1 | tail -5
```
期望：`tsc` 无输出；测试仍为 `2 failed | 848 passed`。

**若 `tsc` 报 `kernelActive` 未使用**：这是预期的（任务 2 会处理），暂时在 `kernelActive` 声明前加一行 `// eslint-disable-next-line @typescript-eslint/no-unused-vars` **不要**——正确做法是保留 `kernelActive` 的声明并在本任务不改它（`tsc` 的 `noUnusedLocals` 只对局部变量报错，`const` 在模块/函数作用域内被声明未用会报 `TS6133`）。若确实报错，把该声明改为 `void kernelActive;` 之后的临时引用**或**直接进入任务 2 一起处理，并在提交信息里说明。

- [ ] **步骤 4：真机验证与步骤 1 的输出逐值比对**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/after-task1.png 4500 '[
 {"type":"eval","js":"const h=window.__hfsKernel; const v=h.getViewport(); return {pxPerSec:v.pxPerSec, scrollLeft:v.scrollLeft, scrollTop:v.scrollTop};"},
 {"type":"eval","js":"document.querySelector(\"[data-timeline-scroller]\").focus(); return 1;"},
 {"type":"key","key":"PageDown"},{"type":"wait","ms":400},
 {"type":"eval","js":"const h=window.__hfsKernel; return {afterPageDown_top:+h.getViewport().scrollTop.toFixed(2)};"}
]' 2>&1 | grep -E "^EVAL"
```
期望：与步骤 1 的输出**逐值相同**（含 `afterPageDown_top:132`；不 focus 会得到 0）。

- [ ] **步骤 5：目视确认渲染正常**

用 `read_image` 打开 `/tmp/after-task1.png`，确认能看到：标尺刻度、音轨行、clip 色块、波形、网格线、右侧与底部滚动条。若出现空白或报错，**停止并报告**（不要继续后续任务）。

- [ ] **步骤 6：提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add frontend/src/components/layout/TimelinePanel.tsx
git commit -m "refactor(timeline): 旧分支改为不可达（内核唯一路径 · 阶段 1）

把 kernelActive 三元判断改为无条件渲染内核分支。旧分支 JSX 暂时保留在文件中
但不可达，以便真机发现问题时以最小代价回滚（还原一个表达式）。

真机复验：内核视口真值与 PageDown 落点（132）与改动前逐值一致。"
```

---

## 任务 2：删除孤儿 state/hook（阶段 2）

**判定依据是工具报告，不是人工猜测。**

**Files:**
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`

- [ ] **步骤 1：取得权威清单**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | grep -E "TS6133" | sort -u
```
说明：`noUnusedLocals` / `noUnusedParameters` **已开启**于 `tsconfig.app.json`，所以普通 `tsc -b` 就会报 `TS6133`。**不要**用 `tsc -b --noUnusedLocals`（`--build` 与该选项互斥，报 `TS5094`）。

期望：报出一批「声明未读」的局部变量。**注意：此时 JD 报告可能为空**——因为旧分支 JSX 仍在文件里引用着那些变量（它只是不可达，变量仍被"读取"）。若为空，继续步骤 2。

- [ ] **步骤 2：先删旧分支 JSX（释放引用），再取报告**

定位任务 1 标记的那段不可达旧分支：从 `) : (` 到与之匹配的 `)}`（在 `</TimelineScrollArea>` 之后、`{/* 共振峰工具窗口 … */}` 之前）。

删除前的原文尾部形态（**以此作为锚点**，`…` 表示中间的旧分支 JSX，全部删除）：

```tsx
                                }
                            />
                        </>
                    ) : (
                        <>
                            …（旧分支 JSX，全部删除）…
                            </TimelineScrollArea>
                        </>
                    )}

                    {/* 共振峰工具窗口：`fixed` 定位（视口坐标），与渲染模式无关。 */}
```

删除 `) : (` 与其后的整个旧分支（含 `<>`、`</TimelineScrollArea>`、`</>`），把它替换为空——即让内核分支的 `</>` 直接接 `)}`：

```tsx
                                }
                            />
                        </>
                    )}

                    {/* 共振峰工具窗口：`fixed` 定位（视口坐标），与渲染模式无关。 */}
```

**不要删除** `{/* 共振峰工具窗口 … */}` 及其后的 `<ClipFormantToolWindow …>` —— 它们在分支之外，两种模式都用（实测确认）。

- [ ] **步骤 3：重新取权威清单**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | grep -E "TS6133" | sort -u
npx eslint src/components/layout/TimelinePanel.tsx 2>&1 | grep -E "no-unused-vars|unused" | head -20
```
期望：现在报出一批新的未使用项（原旧分支专属的 state / useCallback / ref / import）。**把这些行原样记录下来**，它们是本任务的删除清单。

- [ ] **步骤 4：按清单逐个删除（每个都要判断）**

对步骤 3 报告出的每一项：

1. 若它**只**服务于旧分支（如 `scrollRef`、`isTrackListMirrorEcho` 之外的旧滚动同步 state）→ 删除其声明。
2. 若它同时被内核分支使用（内核分支用到这 22 个标识符：`disabledGroupIds`、`getVisualPlayheadSec`、`handleKernelScrollLeftCommit`、`handleKernelUnavailable`、`handleTimelineDragOver`、`handleTimelineDrop`、`kernelActiveGroupIds`、`kernelHostRef`、`kernelInlineEditProp`、`kernelInteractions`、`pxPerSec`、`rowHeight`、`rulerContentRef`、`rulerPlayheadLineRef`、`s.showAllTakes`、`setPxPerSec`、`setRowHeight`、`setViewportWidth`、`state.syncScrollLeftFrame`、`timeRulerNode`、`tint`、`trackListScrollRef`）→ **保留**。
3. 若无法确定 → 保留，并在 `TimelinePanel.tsx` 该声明处加注释 `// 保留：暂不确定是否仅旧分支使用（阶段 2 未判定）`，在提交信息里列出待确认项。

- [ ] **步骤 5：每次删除后立即验证**

每删除 3-5 项就运行一次：

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -5
npx vitest run 2>&1 | tail -4
```
期望：`tsc` 无输出；测试保持 `2 failed | 848 passed`（**不得新增失败**）。

- [ ] **步骤 6：真机回归**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/task2.png 4500 '[
 {"type":"eval","js":"const h=window.__hfsKernel; return {kernel:!!h, kernelAttr:!!document.querySelector(\"[data-hs-timeline-kernel]\"), legacyOverflowAuto:!!document.querySelector(\".custom-scrollbar.overflow-auto\"), trackLanes:document.querySelectorAll(\"[data-track-lane]\").length};"},
 {"type":"eval","js":"document.querySelector(\"[data-timeline-scroller]\").focus(); return 1;"},
 {"type":"key","key":"PageDown"},{"type":"wait","ms":400},
 {"type":"eval","js":"return {afterPageDown_top:+window.__hfsKernel.getViewport().scrollTop.toFixed(2)};"}
]' 2>&1 | grep -E "^EVAL"
```
期望：`kernel:true`、`kernelAttr:true`、`legacyOverflowAuto:false`、`afterPageDown_top:132`（不 focus 会得到 0）。

**⚠️ 不要用 `[data-timeline-scroller]` 当"走的是内核"的判别信号**（初稿这么写，已实测证伪）：**两个实现都挂了该属性** —— 旧实现挂在原生滚动容器上（`TimelinePanel.tsx:4935`），内核视图也**刻意**回填了它（`TimelineKernelView.tsx:698`，因为 `measureTimelineViewportOffsetPx()` 与参数编辑器的 ResizeObserver 都按"轨道区视口元素"查询它；内核下不回填会让跨面板同步偏移恒为 0）。它在两种模式下**都**为 true，作为判别信号是假阳性。可靠的判别信号是 `[data-hs-timeline-kernel]`（内核独有）与 `.custom-scrollbar.overflow-auto`（旧实现的滚动容器）。

- [ ] **步骤 7：提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add frontend/src/components/layout/TimelinePanel.tsx
git commit -m "refactor(timeline): 删除旧分支与孤儿 state（内核唯一路径 · 阶段 2）

删除不可达的旧分支 JSX（约 513 行），并以 tsc -b 的 TS6133 报告为权威清单
删除随之孤立的 state / useCallback / ref / import。

判定依据说明：noUnusedLocals 已在 tsconfig.app.json 开启，普通 tsc -b 即
可报出未使用项（tsc -b --noUnusedLocals 与 --build 互斥，不可用）。

保留 ClipFormantToolWindow：它位于分支闭合之后，两种渲染模式都挂载。"
```

---

## 任务 3：拆分 `trackOverlap.ts` 并删除旧组件

**本任务最容易做错的地方**：`TrackLane.tsx` 导出的 `computeLeadingOverlapSecByClipId()` **仍被使用**——按文件名整体删除会连带删掉内核正在调用的函数。

**修正（实测）**：控制方初稿称它有"两个旧分支之外的消费者"（`TimelinePanel` 与
`TimelineWaveformSurface`），这是**基于任务 2 之前的代码状态**。任务 2（`66e7c8b4`）
已把 `TimelinePanel` 的 import **与唯一调用点**一起当作孤儿删除了，因此**实际只剩一个
消费者**：`TimelineWaveformSurface`（内核视图 `TimelineKernelView:708` 自行挂载），
它把结果交给 `waveform/sceneBuilder` 做重叠区的等权混合。

**Files:**
- Create: `frontend/src/components/layout/timeline/trackOverlap.ts`
- Create: `frontend/src/components/layout/timeline/trackOverlap.test.ts`
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`（改 import 来源）
- Modify: `frontend/src/components/layout/timeline/TimelineWaveformSurface.tsx`（改 import 来源）
- Modify: `frontend/src/components/layout/timeline/index.ts`（删再导出）
- Delete: `TrackLane.tsx`、`ClipItem.tsx`、`TimelineScrollArea.tsx`、`TimelineSurface.tsx`、`TimelineCanvasViewport.tsx`
  （**不含 `BackgroundGrid.tsx`** —— 参数编辑器也在用它，见步骤 7 的陷阱说明）

- [ ] **步骤 1：写新模块的失败测试**

创建 `frontend/src/components/layout/timeline/trackOverlap.test.ts`：

```ts
/**
 * 时间轴 · clip 前置重叠计算的单测。
 *
 * 【为什么单测这个函数】它从 `TrackLane.tsx` 拆出（旧轨道组件已随旧渲染路径
 * 删除），但**内核仍在用**：`TimelineWaveformSurface`（内核视图自行挂载）与
 * `TimelinePanel` 都依赖它。拆分类改动必须有用例固定其行为，否则"拆错了"
 * 只会表现为波形重叠区的颜色异常，极难归因。
 */
import { describe, expect, it } from "vitest";

import { computeLeadingOverlapSecByClipId } from "./trackOverlap";
import type { ClipInfo } from "../../../features/session/sessionTypes";

/** 造一个测试用 clip（只填本函数用到的字段）。 */
function clip(id: string, startSec: number, lengthSec: number): ClipInfo {
    return { id, startSec, lengthSec } as ClipInfo;
}

describe("computeLeadingOverlapSecByClipId", () => {
    it("单个 clip 无重叠", () => {
        expect(computeLeadingOverlapSecByClipId([clip("a", 0, 5)])).toEqual({ a: 0 });
    });

    it("完全不重叠时全为 0", () => {
        const r = computeLeadingOverlapSecByClipId([clip("a", 0, 2), clip("b", 5, 2)]);
        expect(r).toEqual({ a: 0, b: 0 });
    });

    it("★ 后一个 clip 覆盖前一个的尾部时，只算其左前导重叠段", () => {
        // a: [0,5)，b: [3,8) → b 的前导重叠 = 5 - 3 = 2
        const r = computeLeadingOverlapSecByClipId([clip("a", 0, 5), clip("b", 3, 5)]);
        expect(r.a).toBe(0);
        expect(r.b).toBeCloseTo(2, 9);
    });

    it("★ 重叠取「最远的那个前序 clip 末端」（多个前序时取 max）", () => {
        // a: [0,4)，b: [0,3)，c: [2,6) → c 的前导重叠 = max(4,3) - 2 = 2
        const r = computeLeadingOverlapSecByClipId([
            clip("a", 0, 4),
            clip("b", 0, 3),
            clip("c", 2, 4),
        ]);
        expect(r.c).toBeCloseTo(2, 9);
    });

    it("渲染顺序按 startSec 升序、同起点按 id 字典序（决定谁算「前序」）", () => {
        // 顺序影响"前序集合"。同起点时 id 字典序小的在前：
        // 因此对 "b" 而言 "a" 是前序 → b 的前导重叠 = (0+3) - 0 = 3
        const r = computeLeadingOverlapSecByClipId([clip("b", 0, 3), clip("a", 0, 3)]);
        expect(r.b).toBeCloseTo(3, 9);
        expect(r.a).toBe(0);
    });

    it("结果不含负数（不重叠时为 0 而非负值）", () => {
        const r = computeLeadingOverlapSecByClipId([clip("a", 0, 1), clip("b", 0, 1)]);
        for (const v of Object.values(r)) expect(v).toBeGreaterThanOrEqual(0);
    });

    it("空输入返回空对象（不抛异常）", () => {
        expect(computeLeadingOverlapSecByClipId([])).toEqual({});
    });
});
```

- [ ] **步骤 2：运行测试确认失败**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx vitest run src/components/layout/timeline/trackOverlap.test.ts 2>&1 | tail -8
```
期望：FAIL，报错形如 `Failed to resolve import "./trackOverlap"`（模块尚不存在）。

- [ ] **步骤 3：创建 `trackOverlap.ts`**

把 `TrackLane.tsx` 中的 `compareClipRenderOrder`（`:25-29`）与 `computeLeadingOverlapSecByClipId`（`:47-80`）**原样搬入**新文件。新文件内容：

```ts
/**
 * 时间轴 · clip 前置重叠计算
 *
 * 【主要内容】
 * 由一组 clip 计算每个 clip 在"自身左侧前导区"的重叠时长（秒），以及判定
 * clip 渲染顺序的比较函数。
 *
 * 【作用】
 * 重叠区需要做等权可视化混合，避免后绘制的 clip 完全盖住前一个 —— 因此必须
 * 知道每个 clip 左侧被前序 clip 覆盖了多长。渲染顺序（`startSec` 升序、同起点
 * 按 id 字典序）决定哪些 clip 算"前序"，故两个函数必须放在一起。
 *
 * 【为什么独立成模块】
 * 原本定义在 `TrackLane.tsx`（旧的轨道 DOM 组件）。旧渲染路径移除后，`TrackLane`
 * 组件被删除，但本函数**仍被使用**：
 * - `TimelinePanel`（时间轴面板）
 * - `TimelineWaveformSurface`（波形层，内核视图自行挂载）
 * 因此从组件文件里拆出，避免"删组件顺带删掉活代码"。
 *
 * 【与其他模块的关系】
 * - 上游：面板与波形层传入 `ClipInfo[]`。
 * - 下游：纯计算，返回 `clipId -> 重叠秒数`，无副作用、不依赖 DOM / React。
 */
import type { ClipInfo } from "../../../features/session/sessionTypes";

/** clip 渲染顺序：`startSec` 升序；起点相同按 id 字典序（保证顺序稳定）。 */
function compareClipRenderOrder(a: ClipInfo, b: ClipInfo): number {
    const d = (a.startSec ?? 0) - (b.startSec ?? 0);
    if (Math.abs(d) > 1e-9) return d;
    return String(a.id).localeCompare(String(b.id));
}

/**
 * 计算每个 clip 在"自身左侧前导区"的重叠时长（秒）。
 *
 * 流程：按 `compareClipRenderOrder` 排序 → 对每个 clip 遍历其**所有前序** clip →
 * 取「前序末端」的最大值（且须晚于本 clip 起点）→ 减去本 clip 起点即得前导重叠。
 *
 * 特殊说明 1：只算"左侧前导段"，不含被后续 clip 覆盖的尾部——混合语义只作用于
 * 每个 clip 的前导区（见文件头说明）。
 *
 * 特殊说明 2：用 `1e-9` 容差判断"是否真的重叠"，避免浮点误差把首尾相接判成重叠。
 *
 * @param clips 轨道内（或全部）clip 列表，顺序无关（内部会排序）。
 * @returns clipId → 前导重叠秒数（恒 ≥ 0；无重叠为 0）。
 */
export function computeLeadingOverlapSecByClipId(clips: ClipInfo[]): Record<string, number> {
    const sorted = [...clips].sort(compareClipRenderOrder);
    const leadingOverlapSecByClipId: Record<string, number> = {};

    for (let i = 0; i < sorted.length; i += 1) {
        const clip = sorted[i];
        const clipStart = clip.startSec;
        const clipEnd = clip.startSec + clip.lengthSec;
        let leadingOverlapEnd = clipStart;

        for (let j = 0; j < i; j += 1) {
            const other = sorted[j];
            const otherEnd = other.startSec + other.lengthSec;
            const overlapEnd = Math.min(clipEnd, otherEnd);
            if (overlapEnd <= clipStart + 1e-9) continue;
            if (overlapEnd > leadingOverlapEnd) {
                leadingOverlapEnd = overlapEnd;
            }
        }

        leadingOverlapSecByClipId[clip.id] = Math.max(0, leadingOverlapEnd - clipStart);
    }

    return leadingOverlapSecByClipId;
}
```

**以上实现体必须与 `TrackLane.tsx` 逐字一致**（搬运而非重写）。搬完后用下面命令确认搬运无损：

```bash
cd /Users/guoqiangye/code/HiFiShifter
diff <(git show HEAD:frontend/src/components/layout/timeline/TrackLane.tsx | sed -n '47,80p' | grep -v '^export function\|^ \* \|^/\*\*') \
     <(sed -n '/^export function computeLeadingOverlapSecByClipId/,/^}/p' frontend/src/components/layout/timeline/trackOverlap.ts | grep -v '^export function')
```
期望：无输出（除注释行差异外实现体一致）。若 `git show` 的旧行号已漂移，改用 `grep -n` 在旧文件中定位该函数再取范围。

- [ ] **步骤 4：运行测试确认通过**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx vitest run src/components/layout/timeline/trackOverlap.test.ts 2>&1 | tail -6
```
期望：`Tests  7 passed (7)`。

- [ ] **步骤 5：把消费者指向新模块**

**只有 `TimelineWaveformSurface.tsx` 需要改**（`TimelinePanel.tsx` 的该 import 已在
任务 2 随孤儿 state 一并删除，若你在此处找不到它，那是**正常的**——不要凭计划去
"补回"一个 import）。把
```ts
import { computeLeadingOverlapSecByClipId } from "./TrackLane";
```
改为
```ts
import { computeLeadingOverlapSecByClipId } from "./trackOverlap";
```

- [ ] **步骤 6：验证消费者仍工作**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -5
npx vitest run 2>&1 | tail -4
```
期望：`tsc` 无输出（若报 `TrackLane` 相关未解析，说明还有第三处 import，用 `grep -rn "TrackLane" src/ --include=*.ts --include=*.tsx` 找全）；测试 `2 failed | 855 passed`（848 + 7 新增）。

- [ ] **步骤 7：删除旧组件文件与 barrel 行**

**⚠️ 陷阱：`BackgroundGrid.tsx` 名字像旧实现，但它是两用的 —— 不要删。**
`PianoRollPanel.tsx` 从 barrel 导入它（`:84`）并在 `:6379` 渲染 `<BackgroundGrid …>`，
且**不在任何内核守卫内**（`visible={s.timelineSnap.gridVisible}`，与内核开关无关）。
实测确认参数编辑器下网格层的 canvas 确实由它承载。删掉它参数编辑器会直接编译失败。

控制方初稿把它列进了删除清单，是**错的**（当时的依据只是"仅被 TimelineSurface + barrel
引用"——漏了 PianoRollPanel 经由 barrel 的间接引用）。这类"经 barrel 间接引用"正是
按文件名删代码最容易踩的坑；本任务的删除清单已按真实 import 图修正。

其余 5 个文件（`TrackLane` / `ClipItem` / `TimelineScrollArea` / `TimelineSurface` /
`TimelineCanvasViewport`）经核实无其他消费者，可安全删除。

```bash
cd /Users/guoqiangye/code/HiFiShifter
git rm frontend/src/components/layout/timeline/TrackLane.tsx \
       frontend/src/components/layout/timeline/ClipItem.tsx \
       frontend/src/components/layout/timeline/TimelineScrollArea.tsx \
       frontend/src/components/layout/timeline/TimelineSurface.tsx \
       frontend/src/components/layout/timeline/TimelineCanvasViewport.tsx
```

然后打开 `frontend/src/components/layout/timeline/index.ts`，删除对上述 **5** 个已删模块的再导出行。该文件当前共 22 行，需删除这 5 行：

```ts
export * from "./BackgroundGrid";        // ❌ 不删！参数编辑器在用（见上方陷阱）
export * from "./ClipItem";              // 删
export * from "./TimelineScrollArea";    // 删
export * from "./TrackLane";             // 删
export * from "./TimelineCanvasViewport";// 删
export * from "./TimelineSurface";       // 删
```

**必须保留其余全部行**，特别是 `BackgroundGrid`（参数编辑器依赖）、`TimeRuler`、
`TrackList`、`SnapHighlightLayer`、`TimelineWaveformSurface`、`constants`、`math` 等
（内核与参数编辑器都在用）。

**验证**：删除后 `BackgroundGrid` 的再导出行必须仍在：
```bash
grep -n "BackgroundGrid" src/components/layout/timeline/index.ts
```
期望：输出 `export * from "./BackgroundGrid";`（若为空说明误删了）。

- [ ] **步骤 8：验证删除后无悬空引用**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -20
```
期望：无输出。任何 `TS2307`（找不到模块）都说明还有文件 import 了已删模块——逐个修正 import 目标（若目标是 `computeLeadingOverlapSecByClipId`，改指 `./trackOverlap`）。

- [ ] **步骤 9：全量测试与真机回归**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx vitest run 2>&1 | tail -4
VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/task3.png 4500 '[
 {"type":"eval","js":"const h=window.__hfsKernel; return {kernel:!!h};"},
 {"type":"eval","js":"document.querySelector(\"[data-timeline-scroller]\").focus(); return 1;"},
 {"type":"key","key":"PageDown"},{"type":"wait","ms":400},
 {"type":"eval","js":"return {afterPageDown_top:+window.__hfsKernel.getViewport().scrollTop.toFixed(2)};"}
]' 2>&1 | grep -E "^EVAL"
```
期望：测试 `2 failed | 855 passed`；`kernel:true`、`afterPageDown_top:132`（不 focus 会得到 0）。

- [ ] **步骤 10：提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A
git commit -m "refactor(timeline): 删除旧时间轴组件，拆出共享重叠计算（阶段 3）

删除旧渲染路径的 6 个组件文件（TimelineScrollArea / TimelineSurface /
TimelineCanvasViewport / TrackLane / ClipItem，共约 2714 行）
与 barrel 再导出。

关键：TrackLane.tsx 导出的 computeLeadingOverlapSecByClipId 有**两个旧分支之外
的消费者**（TimelinePanel 与内核自行挂载的 TimelineWaveformSurface），因此先拆
到独立的 trackOverlap.ts 并补 7 项单测，再删组件文件。按文件名整体删除会连带
删掉内核正在调用的代码。

保留：TimeRuler（两分支共用）、TimelineWaveformSurface / SnapHighlightLayer
（内核视图自行挂载）、renderKernel/*（共享底座）。"
```

---

## 任务 4：移除四个开关与相关界面（阶段 4）

**Files:**
- Delete: `frontend/src/components/layout/timeline/kernel/featureFlag.ts`、`featureFlag.test.ts`、`kernelMount.ts`、`kernelMount.test.ts`
- Modify: `frontend/src/components/layout/PianoRollPanel.tsx`
- Modify: `frontend/src/components/layout/pianoRoll/usePianoRollInteractions.ts`
- Modify: `frontend/src/components/layout/TimelineDisplaySettingsDialog.tsx`
- Modify: `frontend/src/dev/perfProject.ts`
- Modify: `frontend/src/i18n/{zh-CN,zh-TW,en-US,ja-JP,ko-KR}.ts`

- [ ] **步骤 1：先删开关的使用点，再删定义**

在 `PianoRollPanel.tsx` 中：

1. 删除 import（`:190-191` 附近的 `isPianoRollCurveGlEnabled, isPianoRollGlSceneEnabled`，以及 `isPianoRollKernelEnabled`）。
2. 删除三个常量声明（约 `:235-255`）：
```ts
const PARAM_EDITOR_KERNEL_ENABLED = isPianoRollKernelEnabled();
const PARAM_EDITOR_GL_SCENE_ENABLED = PARAM_EDITOR_KERNEL_ENABLED && isPianoRollGlSceneEnabled();
const PARAM_EDITOR_CURVE_GL_ENABLED = PARAM_EDITOR_GL_SCENE_ENABLED && isPianoRollCurveGlEnabled();
```
3. 把三个常量替换为 `true`（或按下方形态化简）。

   改动前的用量（`grep -o … | wc -l` 实测，含各自 1 处声明）：
   - `PARAM_EDITOR_KERNEL_ENABLED` 出现 18 次 → **17 处使用**
   - `PARAM_EDITOR_GL_SCENE_ENABLED` 出现 13 次 → **12 处使用**
   - `PARAM_EDITOR_CURVE_GL_ENABLED` 出现 3 次 → **2 处使用**

   **这些数字仅供参考**，本步骤会边改边变。**以 grep 的实际输出为准**（下一条命令），不要依赖上面的计数。

用下面的命令逐个核对（**不要**用无差别全局替换，`&&` 组合的表达式要看清）：

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
grep -n "PARAM_EDITOR_KERNEL_ENABLED\|PARAM_EDITOR_GL_SCENE_ENABLED\|PARAM_EDITOR_CURVE_GL_ENABLED" src/components/layout/PianoRollPanel.tsx
```

**注意几个不能简单替换成 `true` 的形态**（替换后要化简）：
- `if (PARAM_EDITOR_KERNEL_ENABLED && host != null)` → `if (host != null)`
- `if (!PARAM_EDITOR_KERNEL_ENABLED) return;` → 整句删除（恒不成立）
- `if (PARAM_EDITOR_KERNEL_ENABLED) return;` → 整句改为 `return;`（恒成立）
- `PARAM_EDITOR_GL_SCENE_ENABLED ? buildGridSpec() : null` → `buildGridSpec()`
- `mainContentSignature: PARAM_EDITOR_GL_SCENE_ENABLED ? mainContentSignature : undefined` → `mainContentSignature`
- 六个 `skip*: PARAM_EDITOR_GL_SCENE_ENABLED`（或 `PARAM_EDITOR_CURVE_GL_ENABLED`）→ `true`

**关于 `skip*` 改为恒 `true` 的后果（已知并接受，勿"顺手修"）**：BL 层是唯一绘制这些图层的地方，因此 WebGL2 运行期不可用时它们无人绘制（实测横向网格线 24 → 0）。这属**已知且有意接受**的限制，见 spec §2.4：无 WebGL2 的环境下参数编辑器本就无法正常工作，补一套 Canvas2D 回退等于把已迁走的渲染层再实现一遍。**不要**为此新增回调或运行期判断。

- [ ] **步骤 2：改 `usePianoRollInteractions.ts`**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
sed -n 1176,1182p src/components/layout/pianoRoll/usePianoRollInteractions.ts
```
把 `if (isPianoRollKernelEnabled()) return;` 改为无条件 `return;`（内核恒为唯一路径，回声永远忽略），删除该 import，并**同步更新该函数上方注释**（原注释解释"内核模式下才忽略"，改为"内核是唯一路径，原生 scroll 恒为回声"）：

```ts
    const onScrollerScroll = useCallback(
        (e: UIEvent<HTMLDivElement>) => {
            // 内核是唯一渲染路径：原生 scroller 只是被动镜像，其 scroll 事件恒为
            // 回声（每个真实输入都在写完原生位置后显式转发给内核）。忽略它，
            // 否则面板会把自己的镜像写回当成输入采纳，造成"阶梯感 / 被吸附感"。
            void e;
            return;
        },
        [],
    );
```

- [ ] **步骤 3：移除设置界面的内核总开关**

在 `TimelineDisplaySettingsDialog.tsx` 中删除：`isKernelRenderingEnabled` / `setKernelRenderingEnabled` 的 import、`kernelEnabled` / `kernelEnabledAtLoad` / `kernelDirty` 三个 state、以及整个复选框 JSX 块（含其上方解释注释）。

- [ ] **步骤 4：移除 dev 快捷切换**

在 `dev/perfProject.ts` 中删除内核切换按钮块（含 `kernelEnabled` 与其 `makeButton("kernel: …")` 调用），并删除 `isTimelineKernelEnabled` / `TIMELINE_KERNEL_FLAG_KEY` 的 import。

- [ ] **步骤 5：删除开关实现与其测试**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git rm frontend/src/components/layout/timeline/kernel/featureFlag.ts \
       frontend/src/components/layout/timeline/kernel/featureFlag.test.ts
```

- [ ] **步骤 6：类型检查（此时会暴露所有悬空引用）**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -20
```
期望：报出所有仍 import `featureFlag` 的位置。逐个修掉（应只剩 `TimelinePanel.tsx` 与 `kernelMount.ts`）。

- [ ] **步骤 7：`TimelinePanel.tsx` 移除开关**

删除 `isTimelineKernelEnabled` import 与 `TIMELINE_KERNEL_ENABLED` 常量、`resolveKernelMount` import 与 `kernelMount` 计算。同时把 `kernelUnavailable` 从 `boolean` 升级为**存失败原因**的 state（失败界面要展示原因），并删除只表达布尔量的旧 state：

删除这两行：

```ts
    const [kernelUnavailable, setKernelUnavailable] = React.useState(false);
```
（`resolveKernelMount({ enabled: …, unavailable: kernelUnavailable })` 那一处也一并删除。）

改为：

```ts
    /**
     * 内核运行期失败的原因（null = 未失败）。
     *
     * 内核是唯一渲染路径，没有回退可走，因此这里只表达"是否失败"以及"为何失败"
     * —— 后者要展示给用户，让他们能自助排障。判据经 `isKernelAvailable` 归约，
     * 见 `timeline/kernel/kernelAvailability`。
     */
    const [kernelUnavailableReason, setKernelUnavailableReason] = React.useState<string | null>(
        null,
    );
```

并把 `handleKernelUnavailable`（任务 5 步骤 8 会再次出现，以那次为准）改为：

```ts
    const handleKernelUnavailable = React.useCallback((reason: string) => {
        console.warn(`[TimelinePanel] 时间轴内核不可用（${reason}），显示排障界面`);
        setKernelUnavailableReason(reason);
    }, []);
```

**说明**：`kernelUnavailable` 这个标识符在本任务后不再存在；渲染判据统一写作 `isKernelAvailable(kernelUnavailableReason !== null)`（任务 5 步骤 8 会给出最终形态）。

- [ ] **步骤 8：简化 `kernelMount.ts` 并保留行为测试**

`kernelMount.ts` 的 `enabled` 维度已消失，保留它会造成"含恒真参数的伪抽象"。把它替换为不引入新模块的直接判断（删除该文件），**同时**把其测试改为守护"失败必须报错而非静默空白"这一行为——通过保留一个新测试文件覆盖面板的判定逻辑。

> **实施记录（与本文的偏差，2026-09-13 阶段 4 提交）**：实际执行时**保留了
> `kernelMount.ts` 与 `kernelMount.test.ts`**（仅修正其已成假的注释：上游改为"当前
> 无生产调用方"、`enabled` 的语义由"用户开关"改为"调用方意愿"），未按本节删除。
> 原因有二：
> 1. 本节原本的处置路径已在任务 4 步骤 7 之外由 `kernelAvailability.ts` + 测试
>    **另行承接**（该文件在阶段 4 之前就已存在），"删除以保住行为守护"的动机不再成立；
> 2. 阶段 4 的验收判据是**通过数恰为 838**（= 阶段 3 的 863 − `featureFlag.test.ts` 的
>    25）。`kernelMount.test.ts` 有 8 项用例，删掉会得到 830，与判据不符。
>
> 结论：删不删它是**独立决策**，不属于"移除开关"的任务范围。已在该文件头写明
> "当前无生产消费者、是否删除属独立决策"，避免下一位读者误以为是遗漏。
> 若将来决定删除，应同时把验收计数改为 830 并说明差异。

具体做法：删除 `kernelMount.ts` 与 `kernelMount.test.ts`，并在 `frontend/src/components/layout/TimelinePanel.tsx` 内联判定（步骤 7 的 `kernelAvailable`）。为保住行为守护，在任务 5 创建的 `KernelUnavailableNotice.tsx` 旁新增 `kernelAvailability.ts` + `.test.ts`：

创建 `frontend/src/components/layout/timeline/kernel/kernelAvailability.ts`：

```ts
/**
 * 时间轴内核 · 可用性判定
 *
 * 【主要内容】
 * 把"内核本次会话是否已失败"归约为一个可测的纯函数，供面板决定渲染内核还是
 * 渲染失败界面。
 *
 * 【作用：为什么值得单独成模块】
 * 内核是唯一渲染路径，失败时**没有回退**——只能显示失败界面。这条"失败必须
 * 报错而不是静默空白"的约束是用户可感知的行为，必须有测试守护；而面板是
 * `.tsx`，本工程 Vitest 跑在 node 环境（无 jsdom），组件无法被单测引用。
 * 因此把判定抽成纯函数。
 *
 * 【历史背景（勿删）】内核曾是 opt-in 路径，其默认值一度跟随
 * `import.meta.env.DEV`，导致打包后**静默退回旧渲染器**（Phase 3 计划 R8）——
 * Windows 真机上的卡顿报告全部来自旧实现。旧实现已移除，但"绝不静默降级"
 * 的结论保留在此：失败必须显式告知用户。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 传入 `TimelineKernelView` 回报的失败状态。
 * - 下游：面板据此渲染 `TimelineKernelView` 或 `KernelUnavailableNotice`。
 * - 独立性：纯函数，不依赖 DOM / React。
 */

/**
 * 内核是否可用于渲染。
 *
 * @param unavailable 本次会话内内核是否已回报不可用（WebGL2 创建失败等）。
 * @returns 可用时为 true；失败时为 false（调用方必须渲染失败界面）。
 */
export function isKernelAvailable(unavailable: boolean): boolean {
    return !unavailable;
}
```

创建 `kernelAvailability.test.ts`：

```ts
/**
 * 内核可用性判定的单测。
 *
 * 【守的是什么】内核是唯一渲染路径，失败时无回退。"失败必须显式报错、绝不静默
 * 空白"是用户可感知的行为，必须有用例钉住——这也是 Phase 3 R8 教训（默认值曾
 * 跟随构建模式、打包后静默退回旧渲染器）在行为层面的延续。
 */
import { describe, expect, it } from "vitest";

import { isKernelAvailable } from "./kernelAvailability";

describe("isKernelAvailable", () => {
    it("★ 未失败时可用（渲染内核）", () => {
        expect(isKernelAvailable(false)).toBe(true);
    });

    it("★ 已失败时不可用（必须渲染失败界面，不得静默空白）", () => {
        expect(isKernelAvailable(true)).toBe(false);
    });

    it("返回值恒为布尔量（不得把 undefined 当可用传下去）", () => {
        expect(typeof isKernelAvailable(false)).toBe("boolean");
        expect(typeof isKernelAvailable(true)).toBe("boolean");
    });
});
```

- [ ] **步骤 9：全量测试与生产构建**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -10
npx vitest run 2>&1 | tail -4
npm run build 2>&1 | tail -3
```
期望：`tsc` 无输出；测试通过数比任务 3 少 25（`featureFlag.test.ts` 删除）但**失败数仍为 2**；构建成功。

- [ ] **步骤 10：移除 5 语言的 3 条旧文案**

在 `frontend/src/i18n/{zh-CN,zh-TW,en-US,ja-JP,ko-KR}.ts` 中删除这三个 key（它们随设置项总开关一并作废）：

**这同时也是清理一处已成假的用户可见文案**：`render_kernel_enabled_desc` 当前仍写着
「关闭后回退到既有渲染实现」（5 语言都有），而阶段 1 起渲染处条件恒真、该开关**不会**
切回旧实现。同理 `featureFlag.ts` 的 :15/:49/:218 仍把逃生门描述为"退回既有实现"——
该文件在步骤 5 整体删除，两处一并消失。

（任务 1 的 spec 审查特意点名这两处，避免它们在分阶段改动中被漏掉。）

```
render_kernel_enabled
render_kernel_enabled_desc
render_kernel_restart_required
```

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
for f in zh-CN zh-TW en-US ja-JP ko-KR; do
  printf "%-6s " $f
  grep -c "render_kernel_enabled:\|render_kernel_enabled_desc:\|render_kernel_restart_required:" src/i18n/$f.ts
done
```
期望：全部为 `0`（删除后）。

- [ ] **步骤 11：提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A
git commit -m "refactor(kernel): 移除四个内核开关与相关界面（阶段 4）

内核已是唯一渲染路径，开关没有可退之处，故一并移除：
- featureFlag.ts 及其 25 项测试（24 项测开关本身，1 项是源码级守护；
  开关不存在后该守护失去对象而空洞化）
- 设置界面「使用新渲染内核」总开关与 5 语言的 3 条文案
- dev/perfProject 的快捷切换
- kernelMount.ts（enabled 维度消失后退化为含恒真参数的伪抽象）

为保住"失败必须显式报错、绝不静默空白"这一行为守护（Phase 3 R8 教训：
默认值曾跟随 import.meta.env.DEV，导致打包后静默退回旧渲染器），新增
kernelAvailability.ts + 测试承接。

参数编辑器只移除开关，保留其 Canvas2D 渲染器——它在内核模式下仍是活的细节层。"
```

---

## 任务 5：GL 失败界面

**Files:**
- Create: `frontend/src/components/layout/renderKernel/gl/glDiagnostics.ts`
- Create: `frontend/src/components/layout/renderKernel/gl/glDiagnostics.test.ts`
- Create: `frontend/src/components/layout/timeline/kernel/KernelUnavailableNotice.tsx`
- Modify: `frontend/src/components/layout/TimelinePanel.tsx`
- Modify: `frontend/src/components/layout/timeline/kernel/TimelineKernelView.tsx`
- Modify: `frontend/src/i18n/*.ts`（5 语言）

- [ ] **步骤 1：写诊断探测的失败测试**

创建 `frontend/src/components/layout/renderKernel/gl/glDiagnostics.test.ts`：

```ts
/**
 * WebGL2 可用性探测的单测。
 *
 * 【为什么要测】GL 失败界面是唯一路径下的**唯一**用户出口，其诊断信息必须准确
 * 且不得因环境差异抛错（它恰恰运行在"环境有问题"的机器上）。
 */
import { describe, expect, it, vi, afterEach } from "vitest";

import { collectGlDiagnostics } from "./glDiagnostics";

afterEach(() => {
    vi.unstubAllGlobals();
});

describe("collectGlDiagnostics", () => {
    it("★ 完全不支持 WebGL2 时给出 false 且不抛异常", () => {
        vi.stubGlobal("document", {
            createElement: () => ({ getContext: () => null }),
        });
        const d = collectGlDiagnostics();
        expect(d.webgl2).toBe(false);
        expect(d.webgl1).toBe(false);
        expect(typeof d.userAgent).toBe("string");
    });

    it("支持 WebGL2 时 webgl2 为 true", () => {
        vi.stubGlobal("document", {
            createElement: () => ({
                getContext: (type: string) => (type === "webgl2" ? { fake: true } : null),
            }),
        });
        expect(collectGlDiagnostics().webgl2).toBe(true);
    });

    it("★ document 缺失（非浏览器环境）时不抛异常", () => {
        vi.stubGlobal("document", undefined);
        expect(() => collectGlDiagnostics()).not.toThrow();
        expect(collectGlDiagnostics().webgl2).toBe(false);
    });

    it("getContext 抛异常时按不可用处理（老驱动的常见表现）", () => {
        vi.stubGlobal("document", {
            createElement: () => ({
                getContext: () => {
                    throw new Error("blocked");
                },
            }),
        });
        expect(collectGlDiagnostics().webgl2).toBe(false);
    });

    it("★ 不尝试读取 renderer/vendor（失败环境拿不到，不假装能给出型号）", () => {
        // 【为什么单列一条】diagnostics 在**已经失败**的环境里运行，此时再建
        // webgl2 上下文同样返回 null，读不到 WEBGL_debug_renderer_info。与其给出
        // undefined 让人误以为拿到了，不如契约上就不提供该字段。
        vi.stubGlobal("document", {
            createElement: () => ({ getContext: () => null }),
        });
        const d = collectGlDiagnostics() as Record<string, unknown>;
        expect(d).not.toHaveProperty("renderer");
        expect(d).not.toHaveProperty("vendor");
    });
});
```

- [ ] **步骤 2：运行测试确认失败**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx vitest run src/components/layout/renderKernel/gl/glDiagnostics.test.ts 2>&1 | tail -6
```
期望：FAIL，`Failed to resolve import "./glDiagnostics"`。

- [ ] **步骤 3：实现 `glDiagnostics.ts`**

```ts
/**
 * 渲染内核 · WebGL 可用性诊断
 *
 * 【主要内容】
 * 在**已经失败**的环境里收集可展示的排障信息：WebGL2 / WebGL1 是否可用、
 * userAgent、devicePixelRatio。
 *
 * 【作用】
 * 时间轴内核是唯一渲染路径，WebGL2 不可用时没有回退，用户看到的失败界面就是
 * 唯一出口 —— 它必须给出能自助排障的信息。
 *
 * 【为什么拿不到显卡型号（重要）】
 * renderer / vendor 需要创建 `webgl2` 上下文并读 `WEBGL_debug_renderer_info`，
 * 而在本函数运行的环境里这次探测**同样返回 null**。因此本模块**不提供**该字段，
 * 而不是给一个 undefined 让人误以为拿到了。
 *
 * 【与其他模块的关系】
 * - 上游：`KernelUnavailableNotice` 在渲染时调用。
 * - 下游：纯探测，不做任何写入或渲染。
 * - 独立性：只依赖 `document`（可缺失），不依赖 React；`document` 缺失或
 *   `getContext` 抛异常时按"不可用"处理，绝不向上抛错——它本就运行在有问题的机器上。
 */

/** 诊断结果。 */
export interface GlDiagnostics {
    /** 能否创建 WebGL2 上下文。 */
    readonly webgl2: boolean;
    /** 能否创建 WebGL1 上下文（用于判断"是否完全无 GL"）。 */
    readonly webgl1: boolean;
    /** 浏览器标识（Tauri 下为 WebView 的 UA）。 */
    readonly userAgent: string;
    /** 设备像素比（影响渲染成本，排障常需）。 */
    readonly devicePixelRatio: number;
}

/**
 * 探测某个 WebGL 上下文类型是否可用。
 *
 * 特殊说明：`createElement` / `getContext` 都可能不存在或抛错（非浏览器环境、
 * 老驱动的异常路径），一律按不可用处理。
 *
 * @param type 上下文类型（`"webgl2"` / `"webgl"`）。
 * @returns 可创建时为 true。
 */
function canCreateContext(type: string): boolean {
    try {
        const doc = globalThis.document;
        if (doc == null || typeof doc.createElement !== "function") return false;
        const canvas = doc.createElement("canvas") as HTMLCanvasElement;
        if (canvas == null || typeof canvas.getContext !== "function") return false;
        return canvas.getContext(type) != null;
    } catch {
        return false;
    }
}

/**
 * 收集 GL 诊断信息。
 *
 * 流程：分别探测 webgl2 / webgl1 → 读取 userAgent 与 devicePixelRatio（缺失时给
 * 安全默认值）→ 组装成可直接展示/复制的对象。
 *
 * @returns 见 `GlDiagnostics`。
 */
export function collectGlDiagnostics(): GlDiagnostics {
    return {
        webgl2: canCreateContext("webgl2"),
        webgl1: canCreateContext("webgl"),
        userAgent: typeof globalThis.navigator?.userAgent === "string" ? globalThis.navigator.userAgent : "未知",
        devicePixelRatio: typeof globalThis.devicePixelRatio === "number" ? globalThis.devicePixelRatio : 1,
    };
}
```

- [ ] **步骤 4：运行测试确认通过**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx vitest run src/components/layout/renderKernel/gl/glDiagnostics.test.ts 2>&1 | tail -6
```
期望：`Tests  5 passed (5)`。

- [ ] **步骤 5：新增 5 语言的失败界面文案**

在 5 个 i18n 文件中各加 6 条文案（放在 `timeline_display_settings` 附近）。`zh-CN` 的内容：

```ts
    kernel_unavailable_title: "时间轴无法渲染（WebGL2 不可用）",
    kernel_unavailable_reason: "浏览器或系统没有提供 WebGL2 图形上下文。",
    kernel_unavailable_hint_intro: "可依次尝试：",
    kernel_unavailable_hints: "1. 确认未禁用硬件加速与软件光栅（浏览器设置或启动参数）；2. 若在远程桌面 / 虚拟机中，尝试在本机运行；3. 更新显卡驱动；4. 关闭其它占用图形资源的窗口后重启应用。",
    kernel_unavailable_diagnostics: "复制诊断信息",
    kernel_unavailable_copied: "已复制",
```

其余 4 语言对应翻译（`zh-TW` 用繁体、`en-US`/`ja-JP`/`ko-KR` 用各自语言），key 名必须完全一致。

- [ ] **步骤 6：实现失败界面组件**

创建 `frontend/src/components/layout/timeline/kernel/KernelUnavailableNotice.tsx`：

```tsx
/**
 * 时间轴渲染内核 · 不可用提示界面
 *
 * 【主要内容】
 * WebGL2 不可用时替代时间轴轨道区显示的界面：标题、原因、按命中概率排序的排查
 * 清单，以及可一键复制的诊断信息。
 *
 * 【作用：为什么必须是"可自助排障"的界面】
 * 内核是唯一渲染路径 —— 没有 Canvas2D 回退，这个界面就是失败场景下用户能看到的
 * **全部**。因此它不能是一行红字：必须让用户（或提供支持的人）能判断原因并采到
 * 诊断信息。实测确认 WebGL2 在纯 CPU 环境（含无 GPU 机器）**是可用的**（ANGLE +
 * SwiftShader），只有显式禁止软件光栅一类配置才会失败，所以这里的清单按实际
 * 命中概率排序。
 *
 * 【不做的事】
 * - 不提供"重试"按钮：失败原因（无 GL）在会话内不会自愈，重试只会反复刷日志。
 * - 不静默降级、不显示空白：宁可明确报错，也不让用户面对一个没有内容的界面。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在 `isKernelAvailable()` 为 false 时渲染本组件。
 * - 下游：`collectGlDiagnostics()` 提供诊断数据；剪贴板写入沿用本工程既有的
 *   `navigator.clipboard` + `execCommand` 兜底模式（见 `TimelinePanel` 的
 *   复制播放头时间）。
 * - 独立性：只依赖 i18n 与诊断模块，不读 Redux。
 */
import React from "react";
import { Button, Flex, Text } from "@radix-ui/themes";

import { useI18n } from "../../../../i18n/I18nProvider";
import { collectGlDiagnostics } from "../../../renderKernel/gl/glDiagnostics";

interface Props {
    /** 失败原因（来自内核视图的回报，用于日志与展示）。 */
    readonly reason: string;
}

export const KernelUnavailableNotice: React.FC<Props> = ({ reason }) => {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [copied, setCopied] = React.useState(false);

    /**
     * 复制诊断信息到剪贴板。
     *
     * 流程：收集诊断 → 优先 `navigator.clipboard.writeText` → 失败则退回
     * `textarea` + `execCommand("copy")`（与面板既有的复制实现同一模式）→
     * 成功后短暂显示"已复制"。
     *
     * 特殊说明：两级兜底是必要的——Tauri / 非安全上下文下 `navigator.clipboard`
     * 可能不可用，而诊断信息正是排障时最需要交出去的东西。
     */
    const handleCopy = React.useCallback(async () => {
        const d = collectGlDiagnostics();
        const text = [
            `reason: ${reason}`,
            `webgl2: ${d.webgl2}`,
            `webgl1: ${d.webgl1}`,
            `devicePixelRatio: ${d.devicePixelRatio}`,
            `userAgent: ${d.userAgent}`,
        ].join("\n");
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            try {
                const textarea = document.createElement("textarea");
                textarea.value = text;
                textarea.style.position = "fixed";
                textarea.style.opacity = "0";
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            } catch {
                // 忽略复制失败：界面本身仍展示原因
            }
        }
        setCopied(true);
        window.setTimeout(() => setCopied(false), 2000);
    }, [reason]);

    return (
        <Flex
            direction="column"
            gap="3"
            align="center"
            justify="center"
            className="absolute inset-0 px-6 text-center"
        >
            <Text size="4" weight="bold">
                {tAny("kernel_unavailable_title")}
            </Text>
            <Text size="2" color="gray">
                {tAny("kernel_unavailable_reason")}
            </Text>
            <Text size="1" color="gray" className="max-w-[560px]">
                {tAny("kernel_unavailable_hint_intro")}
            </Text>
            <Text size="1" color="gray" className="max-w-[560px] whitespace-pre-line text-left">
                {tAny("kernel_unavailable_hints")}
            </Text>
            <Button size="1" variant="soft" onClick={() => void handleCopy()}>
                {copied ? tAny("kernel_unavailable_copied") : tAny("kernel_unavailable_diagnostics")}
            </Button>
        </Flex>
    );
};
```

- [ ] **步骤 7：让内核视图把失败**向上**回报（不再只显示红字）**

`TimelineKernelView.tsx`：`fatal` 状态下**不再**渲染那行红字（改由面板渲染完整界面）。找到：

```tsx
            {fatal !== null ? (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-red-500">
                    {`时间轴内核不可用：${fatal}`}
                </div>
            ) : null}
```

整块删除。这依赖已有的 `onUnavailable` 回报（面板据此切换子树），因此删除后失败界面由面板渲染。

- [ ] **步骤 8：面板接入失败界面**

在 `TimelinePanel.tsx` 的渲染处（原 `kernelActive ? (` 位置）把条件改为可用性判断。**`<TimelineKernelView>` 的既有 props 保持原样不动**（它有约 50 行 props，含 `onUnavailable={handleKernelUnavailable}`；不要重写它们，只改外层的条件表达式与新增 else 分支）：

把（任务 1 改写后的形态 —— 注意条件是恒真的 `true`，由任务 2 删除旧分支后才会变成无三元）：

```tsx
                    {/* 内核是唯一渲染路径（旧实现已移除，见
                        docs/superpowers/specs/2026-09-13-timeline-single-path-design.md）。
                        此处不再有运行期二选一。 */}
                    {/* eslint-disable-next-line no-constant-condition -- 阶段 1：条件恒真，旧分支暂留以便一行回滚（任务 2/3 删除） */}
                    {true ? (
                        <>
                            {timeRulerNode}
                            <TimelineKernelView
                                …（既有 props 全部保留，不要改动）…
                            />
                        </>
                    ) : (
                        …（旧分支，任务 2 删除）…
                    )}
```

若你执行本任务时**任务 2 已完成**（旧分支已删），则起点是 `{true ? ( … )}` 这次要变成 `{isKernelAvailable(...) ? ( … ) : (<KernelUnavailableNotice … />)}`；若任务 2 尚未执行，`true` 与旧分支都还在，同样只改条件行并在末尾补 else 分支。

改为：

```tsx
                    {/* 内核是唯一渲染路径（旧实现已移除，见
                        docs/superpowers/specs/2026-09-13-timeline-single-path-design.md）。
                        WebGL2 不可用时没有回退，改由排障界面接管（见 else 分支）。 */}
                    {isKernelAvailable(kernelUnavailableReason !== null) ? (
                        <>
                            {timeRulerNode}
                            <TimelineKernelView
                                …（既有 props 全部保留，不要改动）…
                            />
                        </>
                    ) : (
                        <KernelUnavailableNotice reason={kernelUnavailableReason ?? "未知原因"} />
                    )}
```

即：**只替换条件表达式那一行与末尾的 `)}`**，中间的子元素原样保留。

配套：`kernelUnavailableReason`（`string | null`）与 `handleKernelUnavailable` 已在**任务 4 步骤 7** 定义，本任务只需新增两个 import：

```ts
import { KernelUnavailableNotice } from "./timeline/kernel/KernelUnavailableNotice";
import { isKernelAvailable } from "./timeline/kernel/kernelAvailability";
```

**注意 `isKernelAvailable` 的入参语义**：它接收"是否已失败"（`kernelAvailability.ts` 内部做 `!unavailable`），因此写作 `isKernelAvailable(kernelUnavailableReason !== null)`。**不要**写成 `isKernelAvailable(!kernelUnavailable)` 或 `isKernelAvailable(!kernelUnavailableReason)` —— 双重取反会让失败界面在**内核正常**时弹出。

- [ ] **步骤 9：类型检查、测试与构建**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -10
npx vitest run 2>&1 | tail -4
npm run build 2>&1 | tail -3
```
期望：`tsc` 无输出；测试失败数仍为 2；构建成功。

- [ ] **步骤 10：真机验证失败界面**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
VW=1920 VH=1200 PROBE_INIT=/tmp/vprobe/nogl.js node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/task6-nogl.png 5000 '[
 {"type":"eval","js":"const t=document.body.innerText; return {hasTitle:/时间轴无法渲染/.test(t), hasHints:/硬件加速/.test(t), hasCopyBtn:/复制诊断信息/.test(t), kernelOff:!window.__hfsKernel};"}
]' 2>&1 | grep -E "^EVAL"
```
期望：`hasTitle:true`、`hasHints:true`、`hasCopyBtn:true`、`kernelOff:true`。

- [ ] **步骤 11：验证诊断信息可复制**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
VW=1920 VH=1200 PROBE_INIT=/tmp/vprobe/nogl.js node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/task6-copy.png 5000 '[
 {"type":"eval","js":"window.__clip=null; const orig=navigator.clipboard&&navigator.clipboard.writeText; if(orig) navigator.clipboard.writeText=(t)=>{window.__clip=t; return Promise.resolve();}; const btns=Array.from(document.querySelectorAll(\"button\")).filter(b=>/复制诊断信息/.test(b.textContent||\"\")); if(btns[0]) btns[0].click(); return {found:btns.length};"},
 {"type":"wait","ms":600},
 {"type":"eval","js":"return {clipboard:window.__clip, copiedLabel:/已复制/.test(document.body.innerText)};"}
]' 2>&1 | grep -E "^EVAL"
```
期望：`clipboard` 含 `webgl2: false`、`webgl1: false`、`reason`、`userAgent`；`copiedLabel:true`。

- [ ] **步骤 12：提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A
git commit -m "feat(timeline): GL 不可用时的自助排障界面（阶段 4）

内核成为唯一渲染路径后，WebGL2 不可用时没有任何回退，失败界面就是用户的唯一
出口。因此把原先的一行红字替换为：标题 + 原因 + 按实测命中概率排序的排查清单 +
可一键复制的诊断信息。

诊断只给布尔探测（webgl2 / webgl1 是否可用）+ UA + dpr，**不提供**显卡型号：
renderer/vendor 需要在已失败的环境里再建 webgl2 上下文，同样返回 null，给
undefined 会让人误以为拿到了。该限制写入代码注释与测试契约（断言不存在
renderer/vendor 字段）。

不做：不提供重试按钮（无 GL 在会话内不会自愈，重试只会刷日志）；不静默降级。

实测（PROBE_INIT 令 getContext(\"webgl2\") 返回 null）：界面出现、清单可见、
复制按钮工作且诊断内容含 webgl2:false。"
```

---

## 任务 6：CPU 渲染基准进仓库

**Files:**
- Create: `frontend/scripts/cpu-render-bench.mjs`

- [ ] **步骤 1：创建基准脚本**

创建 `frontend/scripts/cpu-render-bench.mjs`：

```js
/**
 * CPU-only 渲染基准（软件光栅下的正确性与帧时间）
 *
 * 【主要内容】
 * 以 SwiftShader 软件光栅启动 Chrome，程序化驱动时间轴横向滚动，输出帧间隔的
 * p50 / p95 / max 与慢帧计数；并打印所探测到的 GL renderer，证明确实走了软件栈。
 *
 * 【作用：为什么需要它】
 * 时间轴内核是唯一渲染路径，因此必须确认"没有 GPU 的机器也能正确渲染"。该结论
 * 曾以一次性手工验证得出（见 docs/superpowers/specs/2026-09-13-timeline-single-path-design.md），
 * 固化成脚本才能防止回归。
 *
 * 【为什么程序化驱动滚动，而不用 CDP 注入鼠标事件】
 * 实测：CDP 注入鼠标事件本身会产生 80~130ms 的**假慢帧**（idle 时 p50 16.7ms，
 * 一注入拖动就变成 89~131ms），会把注入开销算进渲染时间。这里改为在 rAF 循环里
 * 直接推进滚动位置（内核用 `setScrollLeft`，探针环境用 `scrollLeft`），测到的
 * 才是渲染本身。
 *
 * 【用法】必须在 frontend/ 目录下运行（脚本 import `playwright-core`，从别处
 * 运行会 ERR_MODULE_NOT_FOUND）：
 *     node scripts/cpu-render-bench.mjs
 *
 * 环境变量：
 * - `URL`  目标地址，默认 `http://127.0.0.1:5174/?mock=1`（需 dev server 已运行）
 * - `STEPS` 滚动帧数，默认 120
 */
import { chromium } from "playwright-core";

const url = process.env.URL ?? "http://127.0.0.1:5174/?mock=1";
const steps = Number(process.env.STEPS ?? 120);

const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    // SwiftShader = ANGLE 的纯 CPU 后端；enable-unsafe-swiftshader 允许在无 GPU
    // 时使用它（Chrome 曾默认禁用）；disable-gpu-sandbox 是本机直接启动所必需。
    args: [
        "--force-device-scale-factor=2",
        "--enable-unsafe-swiftshader",
        "--use-angle=swiftshader",
        "--disable-gpu-sandbox",
    ],
});
const page = await browser.newPage({ viewport: { width: 1920, height: 1200 }, deviceScaleFactor: 2 });
await page.goto(url, { waitUntil: "load" });
await page.waitForTimeout(5000);

/** 采样 stats 的公共部分。 */
const summarize = (frames) => {
    const f = frames.slice(3);
    if (f.length === 0) return null;
    const s = [...f].sort((a, b) => a - b);
    const q = (p) => +s[Math.min(s.length - 1, Math.floor(s.length * p))].toFixed(1);
    return {
        frames: f.length,
        p50: q(0.5),
        p95: q(0.95),
        max: +Math.max(...f).toFixed(1),
        slowOver33ms: f.filter((x) => x > 33).length,
        slowOver100ms: f.filter((x) => x > 100).length,
    };
};

// 证明确实走了软件光栅（否则这份数据的意义完全不同）。
const renderer = await page.evaluate(() => {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return null;
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    return gl.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
});

const idle = summarize(
    await page.evaluate(async (n) => {
        const frames = [];
        let last = performance.now();
        await new Promise((resolve) => {
            let i = 0;
            const tick = () => {
                const now = performance.now();
                frames.push(now - last);
                last = now;
                if (++i < n) requestAnimationFrame(tick);
                else resolve();
            };
            requestAnimationFrame(tick);
        });
        return frames;
    }, steps),
);

const scroll = summarize(
    await page.evaluate(async ({ n }) => {
        const el = document.querySelector("[data-timeline-scroller]");
        const host = window.__hfsKernel;
        const frames = [];
        let pos = 0;
        let last = performance.now();
        await new Promise((resolve) => {
            let i = 0;
            const tick = () => {
                const now = performance.now();
                frames.push(now - last);
                last = now;
                pos += 24;
                // 内核模式下必须经内核改视口；它有自绘滚动，原生 scroller 只是镜像。
                if (host) host.setScrollLeft(pos);
                else if (el) el.scrollLeft = pos;
                if (++i < n) requestAnimationFrame(tick);
                else resolve();
            };
            requestAnimationFrame(tick);
        });
        return frames;
    }, { n: steps }),
);

console.log(
    JSON.stringify(
        {
            url,
            softwareRenderer: renderer,
            kernelMounted: await page.evaluate(() => !!window.__hfsKernel),
            idle,
            scroll,
        },
        null,
        1,
    ),
);
await browser.close();
```

- [ ] **步骤 2：运行并确认走的是软件光栅**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
node scripts/cpu-render-bench.mjs
```
期望：输出 JSON，`softwareRenderer` 含 `SwiftShader`（若含 `Apple` / `NVIDIA` 等，说明未走软件栈，数据无效）；`idle.p50` 约 16.7、`idle.slowOver33ms` 为 0；`scroll.p50` 明显高于 idle（数十毫秒量级）。

- [ ] **步骤 3：确认脚本如实报告 GL 后端与内核挂载状态**

脚本输出必须含 `softwareRenderer` 与 `kernelMounted` 两个字段。验证它们确实反映了真实情况，而不是恒为某个值：

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
echo "— 软件光栅（应含 SwiftShader）—"
node scripts/cpu-render-bench.mjs | grep softwareRenderer
echo "— 对照：强制 WebGL2 不可用时 kernelMounted 应为 false —"
```

第二步对照需要临时把 `getContext("webgl2")` 置空。由于该脚本不读 `PROBE_INIT`，改为直接确认判据本身：脚本里的 `kernelMounted` 取自 `!!window.__hfsKernel`，`softwareRenderer` 取自 `WEBGL_debug_renderer_info`。若 `kernelMounted` 为 `false`（说明内核实际不可用），脚本输出的 `scroll` 数据不具代表性——此时应先去查失败界面（任务 5），而不是采信该性能数字。

- [ ] **步骤 4：提交**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add frontend/scripts/cpu-render-bench.mjs
git commit -m "test(perf): 固化 CPU-only 渲染基准脚本

时间轴内核是唯一渲染路径，因此"没有 GPU 的机器能否正确渲染"必须可回归验证。
脚本以 SwiftShader 软件光栅启动 Chrome，程序化驱动滚动并输出帧间隔分位数，
同时打印 GL renderer 以证明确实走了软件栈（否则数据无意义）。

关键设计：不用 CDP 注入鼠标事件驱动 —— 实测注入本身产生 80~130ms 假慢帧
（idle p50 16.7ms → 拖动 89~131ms），会把注入开销算进渲染时间。改为在 rAF
里直接推进滚动位置（内核经 setScrollLeft，探针环境经 scrollLeft）。

注意：必须在 frontend/ 下运行（import playwright-core）。"
```

---

## 后续清理（本计划范围外，记录以免丢失）

任务 3 删除 5 个旧组件后，**另有 7 个模块失去全部引用者**（约 2469 行）。它们仍能
编译、测试也仍通过（相关测试直接 import 它们），但已是死代码：

| 模块 | 行数 | 被谁孤立 |
| --- | --- | --- |
| `timeline/clip/ClipHeader.tsx` | 958 | `ClipItem`（4 处）+ `TrackLane`（1 处），均已删 |
| `timeline/OverlapEditLayer.tsx` | 712 | `TrackLane`（3）+ `ClipItem`（2） |
| `timeline/FadeHitLayer.tsx` | 277 | `ClipItem`（3） |
| `timeline/clip/ClipEdgeHandles.tsx` | 259 | `ClipItem`（2） |
| `timeline/runtime/timelineHitTest.ts` | 147 | `TrackLane`（1） |
| `hooks/useDebouncedPersist.ts` | 96 | 更早已孤立（其文件头仍称"当前使用者 TimelineScrollArea"，已成假注释） |
| `timeline/runtime/timelineViewportDispatch.ts` | 20 | 更早已孤立 |

**为什么不在本计划内删除**：任务是让渲染路径唯一化，判断依据始终是"是否仍有活引用"。
这批模块是**删除的副产物**而非目标；一次性删 2469 行会显著扩大本次改动面与回归风险，
且它们不影响任何行为。建议作为独立的死代码清理任务，用与任务 2 相同的办法（先确认
零引用、再 `tsc -b` 报告驱动）处理。

**注意 `useDebouncedPersist.ts` 的文件头已成假注释**——它写着"当前使用者：
TimelineScrollArea"，而该文件已删除。该注释**已在 `df430b91` 修正**（改为"当前无
使用者"并说明原因）。其余模块若也含此类表述，清理时应一并处理——本仓库把陈旧注释
视为实质缺陷。

### 附：第二个失效的逃生门（函数已删，仅 dev 按钮与 key 保留）

`isGlClipBodiesEnabled()`（key `hifishifter.glClipBodies`）曾定义在
`timeline/runtime/timelineClipGlRenderer.ts`，**无任何读取者**（唯一使用者
`TimelineCanvasViewport` 已在阶段 3 删除）。该函数**已在 `d3f6f516` 随注释修正一并删除**
——修正记录初稿曾误写为"函数保留"，实为已删。现在只剩 key 常量与
`dev/perfProject.ts` 的 `GL clip: on/off` 按钮：按钮仍读写该 key 并派发事件，但
**已无人监听**、按下不产生任何效果。

与上面 7 个模块不同的是：**其宿主模块整体仍是活代码**（内核用 `GlClipBodySink`、
`PianoRollPanel` 用 `parseRgbaColor`、`clipInstances` 用 `buildClipBodyInstance` /
`OFF_*`），因此只能删那个函数与按钮，不能删文件。注释已在 `d3f6f516` 修正（写明
逃生门已失效、历史用途与归属），函数与按钮留待与上面 7 个模块一并清理。

**另有一个待定项**：`timeline/kernel/kernelMount.ts` + 其 8 项测试**无生产消费者**
（`enabled` 维度随开关移除而消失）。任务 4 执行时**未删**，理由有二：其行为守护已由
`kernelAvailability.ts` 承接；且删除会使验收命中数变 830、与任务 4 的 838 判据不符。
**是否删除属独立决策**（若决定删除，应同时把验收计数改为 830 并说明差异），已在该
文件头与任务 4 的提交信息中写明。

## 任务 7：全量验收

- [ ] **步骤 1：完整验证套件**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
npx tsc -b --noEmit 2>&1 | head -10
npx eslint src/ 2>&1 | tail -6
npx vitest run 2>&1 | tail -5
npm run build 2>&1 | tail -3
```
期望：
- `tsc` 无输出
- eslint `0 errors`（warning 数量与本任务前的基线一致）
- 测试**失败数恰为 2**，且这 2 项是 `keybindingMatch.test.ts` 的既有失败；通过数 ≈ 823 + 本次新增用例（**实测 +15**：`trackOverlap` 7 + `glDiagnostics` 5 +
`kernelAvailability` 3；宿主 GL 那 2 项最终未落地）
- 构建成功

- [ ] **步骤 2：确认产物不含 `import.meta.env`（历史教训回归）**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
grep -c "import\.meta\.env" dist/assets/main-*.js
```
期望：`0`。（Phase 3 R8 教训：内核默认值曾跟随构建模式，导致打包后静默退回旧渲染器。旧实现已删，但构建产物仍不该出现 `import.meta.env`。）

- [ ] **步骤 3：确认旧实现与开关已彻底消失**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
echo "— 旧组件文件（应为空）—"
ls src/components/layout/timeline/{TimelineScrollArea,TimelineSurface,TimelineCanvasViewport,TrackLane,ClipItem}.tsx 2>&1 | grep -v "No such file" || echo "  已全部删除 ✓"
# BackgroundGrid 必须仍在（参数编辑器依赖，误删会让参数编辑器编译失败）：
test -f src/components/layout/timeline/BackgroundGrid.tsx && echo "  BackgroundGrid 保留 ✓" || echo "  ✗ BackgroundGrid 被误删！"
echo "— 开关残留（应为空）—"
grep -rn "isTimelineKernelEnabled\|isPianoRollKernelEnabled\|isPianoRollGlSceneEnabled\|isPianoRollCurveGlEnabled\|PARAM_EDITOR_KERNEL_ENABLED\|PARAM_EDITOR_GL_SCENE_ENABLED\|PARAM_EDITOR_CURVE_GL_ENABLED\|kernelActive" src/ 2>/dev/null || echo "  已全部移除 ✓"
echo "— hifishifter.*Kernel 键残留（应为空）—"
grep -rn "hifishifter.timelineKernel\|hifishifter.pianoRollKernel" src/ 2>/dev/null || echo "  已全部移除 ✓"
```

- [ ] **步骤 4：真机全流程回归**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
VW=1920 VH=1200 node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/final-ok.png 5000 '[
 {"type":"eval","js":"const h=window.__hfsKernel; return {kernel:!!h, pxPerSec:h.getViewport().pxPerSec};"},
 {"type":"eval","js":"document.querySelector(\"[data-timeline-scroller]\").focus(); return 1;"},
 {"type":"key","key":"PageDown"},{"type":"wait","ms":400},
 {"type":"eval","js":"return {pgdn_top:+window.__hfsKernel.getViewport().scrollTop.toFixed(1)};"},
 {"type":"wheel","x":1200,"y":190,"deltaY":-240},{"type":"wait","ms":500},
 {"type":"eval","js":"return {afterZoom_pxPerSec:window.__hfsKernel.getViewport().pxPerSec};"}
]' 2>&1 | grep -E "^EVAL"
```
期望：`kernel:true`、`pxPerSec:150`、`pgdn_top:132`、`afterZoom_pxPerSec` ≠ 150（滚轮缩放生效）。再用 `read_image` 确认截图内容完整。

- [ ] **步骤 5：CPU-only 双态回归**

```bash
cd /Users/guoqiangye/code/HiFiShifter/frontend
node scripts/cpu-render-bench.mjs | head -20
VW=1920 VH=1200 PROBE_INIT=/tmp/vprobe/nogl.js node scripts/dev-shot.mjs "http://127.0.0.1:5174/?mock=1" /tmp/final-nogl.png 5000 '[
 {"type":"eval","js":"const t=document.body.innerText; return {notice:/时间轴无法渲染/.test(t), kernelOff:!window.__hfsKernel};"}
]' 2>&1 | grep -E "^EVAL"
```
期望：基准显示 `softwareRenderer` 含 SwiftShader 且 `kernelMounted:true`；`notice:true`、`kernelOff:true`。

- [ ] **步骤 6：更新 spec 状态标记**

把 `docs/superpowers/specs/2026-09-13-timeline-single-path-design.md` 头部的

```
- 状态：已确认，待实施
```
改为
```
- 状态：已实施（见 docs/superpowers/plans/2026-09-13-timeline-single-path.md）
```

- [ ] **步骤 7：提交验收记录**

```bash
cd /Users/guoqiangye/code/HiFiShifter
git add -A
git commit -m "chore(timeline): 唯一路径改造验收记录

- tsc 干净；eslint 0 error；生产构建成功且产物不含 import.meta.env
- 全量测试失败数恰为 2（keybindingMatch 既有失败，与 develop 一致）
- 真机：内核挂载、PageDown 纵向 132、滚轮缩放生效、渲染完整
- CPU-only（SwiftShader）：内核正常挂载并渲染；WebGL2 被禁时显示排障界面
- 旧组件 6 个文件与四个开关、5 语言旧文案均已移除，无残留引用"
```

---

## 自查记录（写计划时对 spec 的逐条核对）

| spec 要求 | 对应任务 |
| --- | --- |
| §2.1 CPU-only 可运行 + 像素一致 | 任务 6（固化脚本）、任务 7 步骤 5 |
| §2.2 移除四个开关与设置项、i18n、dev 切换 | 任务 4 |
| §2.3 参数编辑器保留 Canvas2D、移除三个开关 | 任务 4 步骤 1-2 |
| §2.4 参数编辑器 GL 失败时不可用（已知并接受，不修） | 有意不设任务 ✓ |
| §3.1 删除 6 个旧组件 + 旧分支 JSX | 任务 2（JSX）、任务 3 步骤 7（文件） |
| §3.2 拆分 `computeLeadingOverlapSecByClipId` | 任务 3 步骤 1-6 |
| §3.3 保留共享模块 | 任务 3 步骤 7 的保留清单、步骤 8 的 `tsc` 校验 |
| §3.4 不做渲染重写 | 计划中无相关任务 ✓ |
| §3.5 `featureFlag.test.ts` 处置 + R8 留档 | 任务 4 步骤 5、步骤 8 |
| §4 GL 失败界面（含"拿不到型号"限制） | 任务 5 |
| §5 分期（先不可达再删） | 任务 1 → 2 → 3 → 5 |
| §6 验证方式 | 任务 7 全量验收 |
| §8 验收标准 1-7 | 任务 7 步骤 1-5 |

**计划阶段发现、但经用户决定不修的问题**：写计划时实测发现参数编辑器在 GL 运行期失败后横向音高网格整片消失（判据：贯穿宽度的横向网格线 24 条 → 0 条），根因是 `skip*` 取自模块加载期常量而 GL 失败发生在运行期。用户明确要求**不修**（GL 失败即视为不可用），故不在计划内设任务；已作为"已知并接受的限制"记入 spec §2.4，并在任务 4 步骤 1 加了"勿顺手修"的显式说明。
