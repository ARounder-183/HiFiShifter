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

## 批次 B：数据语义（P0-5 ~ P0-7）—— 已完成（`feat(timeline-kernel): multi-clip/group/ripple drag semantics`）

### B-1 免吸附修饰键（P0-5）✅
- 宿主：`onTrimPreview` / `onFadePreview` / `onSnapOffsetPreview` 补 `modifiers` 快照
- 面板：拖拽 / trim / snap offset 预览统一 `computeEffectiveSnap(s.snapEnabled, isModifierActive(noSnapKb, mods))`
- **淡变角不参与吸附**（旧实现 `useEditDrag` 的 `shouldSnap` 只覆盖 trim/stretch，已逐字核对），
  故淡变预览不加吸附；`modifiers` 为后续「Alt 调曲率」预留

### B-2 多选 / 编组联动拖动（P0-6）✅
- 新增纯函数模块 `hooks/kernelEditSet.ts`（15 条单测）：
  `resolveKernelEditParticipants`（多选 + 编组展开 + `ignoreGrouping` / `disabledGroupIds`）、
  `applyKernelEditDelta`（共享位移先按 `-minStart` 钳制以**保持整组间距**，跨轨按各自初始序号 + 同一偏移量）
- 面板：拖拽 origin 快照参与集合；预览/提交整组批量（`moveClipsRemote` 多元素）；
  copy 模式 ghost 整组渲染、落库走 `copyClipsFromDrag`（每成员按各自轨道解析目标轨）
- **仍未做**：trim / fade / snap offset 的**多选批量**（拖拽移动已覆盖多选与编组；
  trim 目前仍是单 clip，属已知残留）

### B-3 波纹编辑（P0-7）✅
- 拖拽开始时 `buildRippleFollowers`（origin = 参与者最早起点、`session.rippleMode`）；
  预览按**钳制后的共享位移** `applyRippleFollowerShift`；copy 模式与取消路径恢复原位
- 提交后的权威波纹仍由后端计算（与旧实现同源）

### B-4（本轮新发现并修复）框选后弹出右键菜单、吞掉下一次左键 ✅
- **现象**：右键拖拽框选后画面上仍弹出 clip 菜单，随后的左键拖拽完全无效
- **根因**：macOS / 部分 Chromium 在右键**按下**时就触发 `contextmenu`，而框选是否成立要等指针
  移动超过阈值才知道——「框选成立后吞掉一次 contextmenu」的写法永远晚了一步
- **修复**：右键交互期间一律 `preventDefault + stopPropagation` 并记住位置；松手时
  框选成立 → 丢弃；未成立（右键单击）→ 在该位置**补发**菜单（与旧实现
  `useTimelineSelectionRect` 的重放语义一致）；`suppressNextContextMenu` 改为在下一次
  `pointerdown` 清除，兼容「松手后才触发 contextmenu」的平台
- **同批修复**：内核拖拽提交后补 `applyAutoCrossfade` / `applyDetachedAutoCrossfadeClears`
  （旧实现有、内核缺失——拖拽落库后自动交叉淡化不生效）

---

## 批次 C：手册对齐（P1-1 ~ P1-4、P1-9）—— 部分完成

### 已完成

| 任务 | 内容 | 验证 |
|---|---|---|
| C-4a | 分组锁链徽标点击 → `onToggleGroupDisabled`（复用旧实现 `toggleGroupDisabled`） | 点击命中 `control: chain`；后端 `toggle_group_disabled` 调用 ✓ |
| C-4b | 编组激活的**深金外圈描边** + 锁链徽标「已禁用」配色 | 细节画布金色像素 0 → **930** ✓ |
| C-5 | 空白点击清空选择 + 按 `允许时间轴点击切换轨道` 切轨；点击 clip 改走旧实现的 `selectTrackLaneClipRemote`（含 `selected_clip` 落库与点击切轨） | 点击 clip → `select_clip`；点击空白 → `select_track` ✓ |

**C-4b 的根因（重要）**：编组状态（`activeGroupIds` / `disabledGroupIds`）从未进入渲染内核的
渲染模型与 GL 实例构建器——旧实现的 `buildSparseClipRenderModel` 会算 `activeGroupIds` 但
不返回、`drawTimelineCanvas` 与 `clipInstances.build` 的编组参数没人传，于是描边恒不绘制、
徽标恒为「未禁用」。修复：模型返回 `activeGroupIds`；宿主把编组状态同时传给模型、GL 构建器
与细节层。
**同时新增** `drawTimelineCanvas` 的 `groupOutlineOverGl` 选项：细节层位于波形**之上**，
若照旧画整块会把波形盖掉——该模式让块面仍归 GL，Canvas2D 只补那一圈描边。

**未验证（mock 限制）**：禁用状态的实际视觉切换依赖后端返回的 `disabled_group_ids`
（mock 的 fallback 不返回 TimelineState），需真机确认。

### C-3 增益旋钮（已完成，提交 `feat(timeline-kernel): gain knob drag and double-click reset`）

- 新内核手势 `gain-drag`：旋钮**竖直拖动调值**（3px 起手阈值，旧实现语义）；
  单击不再进编辑；双击 → `onGainReset` → `commitTrackLaneGain(clipId, 0)`
- 面板换算与旧实现同源：`deltaDb = ΔY × CLIP_GAIN_DRAG_DB_PER_PX`（常量提到
  `timeline/constants.ts` 作为单一来源）、`advanceFineAxisDrag` 处理精细修饰键、
  `applyBulkGainDeltaDb` 钳制 ±12dB、提交 `setClipsStateBulkRemote`（读乐观值）
- 徽标改为**双击**进行内编辑（手册语义）；单击只走选中/拖拽
- Esc / pointercancel 回滚到按下时增益
- **踩坑（重要）**：旋钮分支**不能**自己做双击判定——外层 clip 分支已把
  `lastClipPress` 覆写成"本次按下"，自判必然恒为双击（表现为"每次按旋钮都把增益
  重置成 0dB"）。直接复用外层的 `isDoubleClick`。
- 验证：拖动 40px → gain 0.3162（-10dB）；双击 → 1.0；单击徽标 inputs=0；
  双击徽标 inputs=1（值 "0"）

### C-1 拉伸（已完成 stretch 部分）

- **抽取**（先做，避免两份语义）：
  - `hooks/stretchGroup.ts` 新增 `computeClipStretch`（单 clip 拉伸几何：对侧边缘固定、
    速率反算与钳制、**用钳制后的速率回算长度**、淡变与 SnapOffset 比例缩放）+ 12 条单测；
    `scaleSnapOffsetForStretch` 从 `useEditDrag` 迁入（单一来源）
  - 新增 `hooks/stretchParams.ts`：`stretchLinkedParams` / `stretchTrackLinkedParams`
    从 `useEditDrag` 抽出（旧实现与内核共用「锁定参数线时的曲线时域映射」）
  - `MIN_CLIP_LENGTH_SEC` 提到 `timeline/constants.ts`（宿主 + 面板同源）
- **内核接入**：边缘手势在**按下时**按 `modifier.clipStretch`（Alt）定模式（裁切 / 拉伸），
  预览写五个字段（起点 / 长度 / 速率 / 两侧淡变 / SnapOffset），提交走
  `setClipsStateBulkRemote`，落库后二次写回速率并在 `lockParamLinesEnabled` 时映射参数线；
  取消路径五个字段一起回滚
- **顺带修复（既有缺陷）**：内核裁切提交原先只发 `startSec/lengthSec`，**漏了源区间**
  → 后端按旧源区间重新解释内容（波形与音频对不上）。已补 `sourceStartSec/sourceEndSec`
- 验证：Alt+右缘 +100px → `length 4.667 / rate 0.857 / fadeIn 0.7`（左缘固定 2.0）；
  Alt+左缘 +50px → `start 2.333 / length 3.667 / rate 1.0909`（**右缘固定 6.0**）；
  无 Alt 对照 → 只改 `length` 与 `sourceEndSec`、速率不变

### C-1b slip（已完成）

- **抽取**：新增 `hooks/slipWindow.ts`（`computeSlipWindow` + `readSlipClip`），
  把 `useSlipDrag` 的源窗口平移几何（倒放方向反转 / loop 取模环绕 / 非 Loop 正放的
  派生窗口 / 其余保持跨度）抽成共享纯函数，`useSlipDrag` 改为引用。
- **内核接入**：`clip-drag` 在按下时若 `modifier.clipSlipEdit`（Alt）按住则进入 slip 模式
  （优先于 copy）；逐帧按**增量**应用（用当前 Redux 值 + 增量，避免累计位移重复施加）；
  提交用交互数学结果（`lastSourceById`，不回读 Redux，与旧实现 `lastById` 同源）；
  零位移不写后端；取消回滚到按下时源窗口。
- **已知差异（记录）**：未实现旧实现的「loop 边界吸附」（只在 loop 开启且窗口跨过
  素材边界时影响落点），其余分支逐条对齐。
- 验证：Alt+拖 body +100px → `{sourceStartSec: 0.6667, sourceEndSec: 4.6667}`（只有源窗口变）；
  反向 −100px → `{-0.6667, 3.3333}`（允许越出媒体＝渲染静音）；无 Alt 对照 → `move_clips` ✓

### C-6 多选 trim / fade（已完成）

- **裁切多选**：按下时解析参与集合（多选 + **编组展开**，与旧实现 `useEditDrag` 的
  `supportsGroupExpansion` 对 trim 展开一致）；预览以**锚点位移**为基准逐 clip 换算
  （源位移 = 时间轴位移 × **各自播放速率**），提交全部参与者、取消全部回滚。
- **淡变多选**：参与集合**不展开编组**（旧实现排除 fade / gain）；应用的是**同一个淡变
  长度值**（`applyBulkFadeValue`，各自按长度钳制）而不是同一增量；提交 / 取消同样覆盖全部。
- **踩坑（重要）**：参与集合在按下时解析，回调必须把 `multiSelectedClipIds` 放进依赖数组——
  漏掉时闭包停在挂载时的空选择，表现为「框选多个后仍只裁切一个」。
- **顺带修复**：锚点的源位移原先漏乘播放速率（rate ≠ 1 时源域与时间轴不同步）。
- 验证：框选 2 clip 后拖 clip-1 右缘 +100px → 两个 update（clip-1 `length 4.667`；
  clip-2 速率 1.5 → `length 7.667 / sourceEnd 8`，源位移按速率折算 ✓）；
  框选后拖淡入角 +100px → 两个 clip 都是 `fadeInSec 1.2667`（同值语义）✓

### C-2 clip 音高拖拽（已完成，走「委托」路线）

- 内核新增 `onClipPointerDownIntercept`：clip 左键按下时先问面板是否接管；
  面板在 `modifier.clipPitchDrag`（Alt+Shift）按住时构造最小鸭子类型事件，
  **整体交给旧实现的 `useClipPitchDrag`**（自带 window 监听 / 参数帧预览 /
  undo group / 收尾 / tooltip 发布）。
- **为什么不重写**：该手势是一台带异步状态机的完整实现（先取基准帧 → 节流写预览 →
  收尾提交或回滚）；内核侧重写会产生第二份音高语义，而 tooltip 浮层本就由面板渲染、
  与渲染模式无关。这比 C-1/C-1b 的「先抽取」更彻底：**零重复**。
- 验证：Alt+Shift 竖直拖 → `get_param_frames` + 多次 `set_param_frames`、
  **无 `move_clips`**（拦截生效）；无修饰键水平拖 → `move_clips`（拦截不误触发）✓

### 未完成（下一批）

| 任务 | 内容 | 说明 |
|---|---|---|
| C-6b | trim / fade 拖拽期间的**自动交叉淡化预览**（旧实现 `previewAutoCrossfadeNow()`），以及收尾的 `applyAutoCrossfade` | 拖拽移动已有（B 批）；trim/fade 尚缺 |
| C-7 | **组拉伸**（多选 + `Alt` 拖边缘）：旧实现用 `buildStretchGroupState` / `computeStretchGroupUpdate` 做整组等比缩放 | 纯函数已在 `stretchGroup.ts`；内核当前只拉伸被拖的那一个 |
| C-8 | slip 的 **loop 边界吸附**（`loopSnapThresholdSec`） | 仅 loop 开启且窗口跨素材边界时影响落点 |
| D | 淡化专属右键菜单 + 淡变 tooltip、多 Take、静音检测预览、拖到空白新建轨道 | — |
| E | 多选修饰键走键位绑定 + Shift 范围选择、Esc 覆盖抓手/snap/框选、滚动条 track 点击跳转、Vertical Lock、陈旧注释与 `glyph/*` 死代码 | — |

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
