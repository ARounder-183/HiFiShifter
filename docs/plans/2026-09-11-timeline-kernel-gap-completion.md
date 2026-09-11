# 时间轴渲染内核 · 功能缺口补全 实施计划

> 输入：`docs/plans/2026-09-11-timeline-kernel-gap-analysis.md`（21 项缺口）
> 开关：`localStorage['hifishifter.timelineKernel']`（`"1"` 开 / `"0"` 关；dev 默认开）
> 验证：`?mock=1` + `VW=1920 VH=1200 KERNEL=1 node scripts/dev-shot.mjs "<url>" <out.png> <waitMs> '<actions>'`
> 原则：**旧实现只做「模式分派」，不重写语义**；能复用旧函数就复用（单一事实来源）。

---

## 批次 A：功能恢复（P0-1 ~ P0-4）

### A-1 模式无关的视口访问器（新文件）

**为什么**：内核模式下 `scrollRef.current === null`，导致拖入落点、自动滚屏、键盘缩放、聚焦光标、视图同步五处一起失效。这些都是「读/写视口」的同一件事，应收敛成一个入口。

- Create `timeline/hooks/timelineViewportAccess.ts`
  - `createTimelineViewportAccess({ scrollRef, kernelHostRef })` → 稳定对象
  - `getRect()`：旧 `scroller.getBoundingClientRect()`；内核 `host.getContainerRect()`
  - `getScrollLeft() / getScrollTop() / getViewportWidth()`
  - `setScrollLeft(px): number`：旧 `applyNativeScrollLeft(scroller, px)`；内核 `host.setScrollLeft(px)` 后回读
- Modify `kernel/host/timelineKernelHost.ts`（`TimelineKernelHost` 公开句柄）
  - 新增 `getContainerRect(): DOMRect | null`
  - 新增 `setScrollLeft(px: number): void`
  - 新增 `setViewport(next: { pxPerSec?: number; scrollLeft?: number }): { pxPerSec: number; scrollLeft: number }`（原子提交；pxPerSec 变化时回调 `onZoomChange`）
- Modify `kernel/scrollKernel.ts`
  - 新增 `setViewport(next): TimelineViewportState`（一次 commit，用**目标 pxPerSec** 算上限，返回新状态）

**测试**：`scrollKernel.test.ts` 追加 3 条（同时改缩放与滚动只通知一次；越界被目标上限钳制；非法值忽略）。

### A-2 拖入落点与预览（P0-1）

- Modify `hooks/useTimelineDragDrop.ts`：`scrollRef` → `viewport: TimelineViewportAccess`；4 处 `scroller.xxx` 改 `viewport.xxx`
- Modify `TimelinePanel.tsx`：新增 `resolveTrackIdAtClientY(clientY)`（内核用 `getContainerRect() + scrollTop`，旧用 `trackIdFromClientY`）；`handleTimelineDragOver/Drop` 改用它；`useTimelineDragDrop` 传 `viewport` 与 `resolveTrackIdAtClientY`

**验证**：mock 下拖入 → 内核容器出现 dropPreview 且落在指针所在轨道；`?mock=1` + `hifi-file-drag` 走同一路径。

### A-3 自动滚屏（P0-2）

- Modify `TimelinePanel.tsx:220-237`：`scrollRef.current` → `viewport.getViewportWidth()/getScrollLeft()/setScrollLeft()`；播放头 `screenLeft` 也用 `viewport.getScrollLeft()`

**验证**：mock 下开自动滚屏播放 → 内核 `getViewport().scrollLeft` 递增。

### A-4 键盘缩放 / 聚焦播放光标 / 粘贴后聚焦（P0-4）

- Modify `hooks/useTimelineEventHandlers.ts`：新增 `viewport` 入参；`hifi:zoomTimelineFocus` 与 `hifi:focusCursor` 改走 `viewport`；内核分支用 `viewport.setViewport({ pxPerSec, scrollLeft })`
- Modify `TimelinePanel.tsx:556-580`（`pendingPlayheadRevealSec`）：内核分支 `viewport.setScrollLeft(...)`

**验证**：mock 下 `hifi:zoomTimelineFocus` → 内核 pxPerSec 变化；`hifi:focusCursor` → scrollLeft 变化；粘贴后播放头进入视口。

### A-5 参数编辑器同步（P0-3）

- Modify `hooks/useTimelineState.ts`：新增 `viewport` 入参；`paramEditorSyncTimeline` 的 apply 分支与 layout effect 增加内核分支（`viewport.setViewport`）；内核视口变化时回写 `timelineViewportSync.setViewport`

**验证**：mock 下开同步 → 参数编辑器滚轮后内核 scrollLeft/pxPerSec 跟随。

---

## 批次 B：数据语义（P0-5 ~ P0-7）

### B-1 免吸附修饰键（P0-5）
- 宿主已透出 `modifiers`；面板侧统一 `effectiveSnap = s.snapEnabled !== noSnapHeld`（复用 `computeEffectiveSnap`）
- 覆盖：`handleKernelDragPreview`、`handleKernelTrimPreview`、`handleKernelFadePreview`、`handleKernelSnapOffsetPreview`

### B-2 多选 / 编组联动拖动（P0-6）
- 面板按下时展开参与集合：`multiSelectedClipIds`（命中项在其中时）+ `expandClipIdsWithGroups`（`ignoreGrouping` 时不展开）
- 预览/提交按集合批量（`setClipsStateBulkRemote` / `moveClipsRemote` 多元素）
- trim / fade 同样按集合（`useEditDrag` 的规则：fade_in/fade_out/gain 不展开组）

### B-3 波纹编辑（P0-7）
- 预览/提交阶段复用 `buildRippleFollowers` + `applyRippleFollowerShift`

---

## 批次 C：手册对齐（P1-1 ~ P1-4、P1-9）

| 任务 | 内容 |
|---|---|
| C-1 | `Alt` 拖边缘 = stretch（`modifier.clipStretch`）；`Alt` 拖 body = slip（`modifier.clipSlipEdit`） |
| C-2 | `Alt+Shift` 竖直拖 clip = 调该 clip 音高（`modifier.clipPitchDrag`） |
| C-3 | 增益旋钮拖动调值 + 双击重置 0 dB（单击改为双击进输入，与手册一致） |
| C-4 | 分组锁链徽标点击（`onToggleGroupDisabled`）+ 编组激活金色描边 |
| C-5 | 空白点击清空选择 + 按设置点击切轨 |

---

## 批次 D：大件（P1-5 ~ P1-8）

| 任务 | 内容 |
|---|---|
| D-1 | 淡化专属右键菜单（`requestOpenFadeContextMenu`）+ 淡变悬停 tooltip |
| D-2 | 多 Take：lane 分界线绘制 + 点击 inactive lane 切换活跃 Take |
| D-3 | 静音检测红色预览层（内容坐标层） |
| D-4 | 拖到轨道区下方新建轨道（ghost + 落库） |

---

## 批次 E：一致性收尾（P2）

- E-1 多选修饰键走键位绑定 + Shift 范围选择
- E-2 Esc 覆盖 `crossfade-grip` / `snap-offset-drag` / `box-select`
- E-3 滚动条 track 点击跳转
- E-4 Vertical Lock 行高亮提示
- E-5 陈旧注释与 `glyph/*` 死代码处理

---

## 提交约定

每批一次提交（`feat(timeline-kernel): ...` / `fix(timeline-kernel): ...`），提交前跑：
`npx vitest run`（仅 2 条 keybindings 预先存在失败）+ `npx tsc -b --noEmit` + `npx eslint <改动目录> --quiet` + `npx prettier --check <改动目录>`。
