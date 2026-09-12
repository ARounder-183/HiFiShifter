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

### C-6b trim / fade 的自动交叉淡化（已实现，**mock 无法验证**）

- 按下时快照：`xfadeClipIds`（= 参与者）、`initialCrossfadeSides`
  （`computeInitialCrossfadeSides`）、`editSides`（**按拖拽的边缘 / 侧限定可调整侧**——
  裁切左缘只允许自动调整 `fadeIn`，右缘只允许 `fadeOut`；淡变同理）。
- 预览：裁切 / 拉伸 / 淡变的每个预览分支末尾调
  `previewAutoCrossfade(store.getState().session, ids, dispatch, affectedSides, editSides)`。
- 收尾：提交落库后按开关走 `applyAutoCrossfade`（`affectedSides` + `editSides`）或
  `applyDetachedAutoCrossfadeClears`；取消路径按已还原几何重算预览（回到按下时关系）。
- **验证受限（已尝试 A/B）**：mock 下裁切提交只观察到 `set_clips_state_bulk`，没有
  `set_clip_state`（自动淡化写回）。用 KERNEL=0 做对照时**旧实现的拖拽没有命中边缘**
  （只产生 `select_clip`），因此对照无效。判断为 **mock 限制**（批量提交的 fulfilled
  把 mock 的 `{ok:true}` 当 TimelineState 应用后，自动淡化字段/重叠关系已不完整），
  **需真机确认**。

### C-7 组拉伸（已完成）

- 参与集合：**裁切与拉伸统一**为「多选 + 编组展开」（旧实现 `supportsGroupExpansion`
  对 trim / stretch 都展开，只排除 fade / gain）——原先拉伸分支硬编码「仅锚点」。
- 组状态由纯函数判定：`buildStretchGroupState`（选区 ≥ 2 且**锚点位于选区边界**，
  否则返回 null → 退化为单 clip 拉伸）；预览走 `computeStretchGroupUpdate`
  （整组等比缩放 + 各成员速率反算钳制 + 淡变与 SnapOffset 按比例缩放）。
- 吸附的 `excludeClipIds` 改为**整组**（旧实现排除 `selectedClipIds`）：组内其他成员
  随本次拉伸一起移动，不应成为自己的吸附目标。
- 提交：整组一次 `setClipsStateBulkRemote`；落库后二次写回非 1 的速率；锁定参数线时
  按**根轨道**聚合成员的时域映射（`stretchTrackLinkedParams`）并 `bumpParamsEpoch`；
  取消路径五个字段整组回滚。
- **顺带补齐**：单 clip 拉伸的提交原先**漏了自动交叉淡化写回**（旧实现
  `shouldApplyAutoCrossfade` 覆盖 stretch）——现已与裁切同源。
- 验证（读实际提交参数）：
  - 框选 clip-1 + clip-2（锚点 clip-1 为选区最左）+ `Alt` 拖左缘 −100px →
    `set_clips_state_bulk` **两条**：clip-1 `start 1.3333 / length 4.2614 / rate 0.9387 /
    fadeIn 0.6392`，clip-2 `start 4.7425 / length 7.4575 / rate 1.4080 /
    snapOffset 0.2663`（= 0.25 × 1.0654）✓ 与手算逐项一致
  - 对照 A（未框选）：只 1 条 `length 4.6667 / rate 0.8571`（单 clip 拉伸）✓
  - 对照 B（框选但锚点**不在**选区边界：拖 clip-1 右缘）：只 1 条，左缘固定 2.0 ✓

### C-8 slip 的 loop / 内容边界吸附（已完成）

- 新增 `slipWindow.toBoundarySnapClip`：媒体边界吸附视图（内容时长 D 的解析规则
  「帧数/采样率 → durationSec → 音高参考块覆盖值」的**单一来源**），
  `useSlipDrag` 改为引用它（原先自己拼一份快照）。
- 内核 slip 预览：命中候选时把**累计位移**替换为吸附值，再以「目标累计 − 已应用累计」
  驱动增量（与旧实现同一算法，仅位移正负号约定相反）；发布 / 清除循环节专用高亮
  （只亮**真正对齐**的那一侧，用 `slipBoundaryAlignedSides`）。
- 参与条件与旧实现逐条对齐：`(isContentBearing || loopEnabled) &&
  snapClipsToSourceMedia && effectiveSnap && snapDistancePx > 0`。
- 验证（A/B 读实际提交参数，`Alt` 拖 body +599px = 3.99333s，候选 −4 距 0.00667s）：
  - 吸附开 → `sourceStartSec 4 / sourceEndSec 8`（**被吸附**，精确落在候选上）✓
  - 吸附总开关关 → `sourceStartSec 3.99333 / sourceEndSec 7.99333`（未吸附）✓
  - 注：`Alt+Shift` 不能用作免吸附对照——那是 `clipPitchDrag`（C-2 的委托拦截），
    因此改用工具栏「吸附」总开关做对照。
- **未覆盖**：loop 分支（`loopEnabled` 的 mod-D 相位族）在 mock 工程里没有 loop Clip，
  需真机或后续给 mock 增加 loop 素材后复测；非 loop 的有限候选族已验证。

### 未完成（下一批）

> **2026-09-12 更新**：下表三项**已全部完成**（提交见下方 D-4 / D-1b / 残留波纹各节）。
> 当前缺口清单为空——21 项缺口（P0 7 / P1 9 / P2 5）已全部关闭。

| 任务 | 内容 | 说明 |
|---|---|---|
| ~~D~~ | 淡化专属右键菜单 ✅ / 多 Take ✅ / 静音检测预览 ✅ / 淡变 tooltip ✅ / 拖到空白新建轨道 ✅ | 全部完成 |
| ~~E~~ | 多选修饰键走键位绑定 + Shift 范围选择 ✅、Esc 覆盖抓手/snap/框选 ✅、滚动条 track 点击跳转 ✅、Vertical Lock ✅、陈旧注释与 `glyph/*` 死代码 ✅ | 全部完成 |
| ~~残留~~ | 裁切 / 拉伸的**波纹跟随预览** ✅ | 已完成 |

---

## 批次 D：大件（P1-5 ~ P1-8）

### D-1 淡变专属右键菜单（已完成）

- **根因**：菜单由三层 DOM 命中块发起（`ClipItem` 角部 / `FadeHitLayer` /
  `OverlapEditLayer`），内核模式下它们都不挂载；菜单宿主 `FadeContextMenuHost`
  本身在两种模式下都已挂载（走全局总线），缺的只是「谁发起」。
- 宿主 `dispatchContextMenuAt` 先解析淡变角 / 交叉点抓手 → 构造载荷
  （形状 / 方向 / **生效长度**：自动交叉淡化覆盖手动值，与绘制端同源）→
  `onFadeContextMenu` → 面板转发 `requestOpenFadeContextMenu`（并先关掉通用菜单）。
- 抓手 = 双列：`primary` 是**前一个** clip 的淡出、`secondary` 是后一个的淡入
  （与旧实现 `crossfadeSides.out / .in` 列序一致）。
- 验证：右键淡变线 → 「淡入 -0.50 淡化曲率…」；右键交叉点 → 「淡出 +0.00 / 淡入 -0.50」✓

### D-2 多 Take（已完成）

- 模型：clip 新增 `takeLaneSeparatorOffsetsPx`，由**共享**的
  `takeLanes.resolveTakeLaneLayouts` 推出（与波形面 / 旧命中同一套数学）；
  `SparseRenderClip.takes` 必须保留 `sourcePath`（lane 布局按它筛选音频 take）。
- 渲染：分界线与静音覆盖层同一遍最后落笔（1px 深色，全 clip 宽，首条 lane 不画）。
- 命中：clip body 上**无修饰键**的按下解析 inactive lane，收尾未超过阈值时切换
  （拖动仍是移动、编辑修饰键优先——旧实现语义）；面板走 `setClipActiveTakeRemote`
  并在**非播放**时把播放光标带到点击位置。
- 验证：分界线在 bodyTop+30 处（alpha 46 = 0.18）且横向恰好等于 clip 宽度；
  点击 lane 1 → `set_clip_active_take(clip-1, take-2)` + seek；再点 lane 0 → 切回 ✓

### D-3 静音检测红色预览层（已完成）

- 数据链：`session.silencePreviewSegments` → 内核数据 → 模型
  `silenceSpansPx`（clip 内相对像素，**防御性钳制**到 clip 本体）。
- 渲染：**单独一遍最后落笔**。踩坑：原先在各自 clip 的细节阶段画，重叠区里后一个
  clip 的块面属于后续批次（前导重叠触发 `barrier`），会把先画的盖掉——实测只有
  第一段可见。
- **脏标记教训复现**：新增可视输入必须同时进 `TimelineKernelView` 的场景重建依赖
  数组，否则「数据变了但画面不动」。
- 验证：两段红色恰好落在 450–630 / 780–870 px（clip 起点 2s、150px/s）；mock 里
  故意越界的第三段**没有**画到 clip 之外 ✓

### D-4 拖到轨道区下方新建轨道（已完成）

- **内核哨兵**：新增纯函数 `isContentYBelowTracks`（`kernel/interaction/dragGeometry`），
  按**行下标整数比较**判定是否越过最后一行——用浮点内容高度比较会在末行底部留下
  一条"看着在行内、却被判越界"的窄带。`applyDragPreview` 命中时把 `targetTrackId`
  换成共享的 `NEW_TRACK_SENTINEL`（不区分的话 `resolveTargetTrackIndex` 会把落点
  钳回末行，用户的"往下拖出新轨"被静默吞掉）。
- **预览**：内核视图新增内容坐标层 `newTrackDrop`（虚线框 + 半透明 clip 预览），
  行位置 `tracks.length × rowHeight`——哨兵轨不在 `tracks` 里，不能靠 `findIndex`
  定位。面板按 `clip.trackId === NEW_TRACK_SENTINEL` 取乐观位置填充，宿主整层平移
  跟随视口（与 ghost / dropPreview 同一机制）。
- **落库**：新增共享编排 `hooks/createNewTrackForDrop.ts`（**单一事实来源**，
  抽取理由与 `copyClipsFromDrag` 相同）——建轨 → 批量移动 → 选中新轨，新轨 id
  按**差集**解析（取末条会在并发建轨时拿到别人的轨道）。
- **踩坑（重要）**：回滚必须用**同步的乐观 reducer**（`moveClipStart` /
  `moveClipTrack`），最初写成 `moveClipRemote` thunk——它是异步的，`void …` 之后
  调用方不再等待，回滚 dispatch 可能落在渲染之后，表现为「建轨失败后 clip 停在
  哨兵轨上、从画面里消失」（实测：mock 的 `add_track` 只回 `{ok:true}` 无轨道列表，
  正好暴露了这条路径）。
- 验证（`?mock=1`，临时把 mock 轨道数改为 2 以让"下方"区域进入视口，验完已还原）：
  拖到末行之下 → 幽灵行出现（`layer.children.length === 1`）、参与者写入
  `__hs_new_track__`；松手后 mock 无法建轨 → **回滚到 `{start 2, track track-1}`**
  且哨兵轨上不留残骸 ✓

### D-1b 淡变悬停 tooltip（已完成）

- **方案**：不新建会吞事件的浮层锚点，而是把**内核容器本身**标记为 AppTooltip 的
  载体（`data-hs-fade-tooltip-anchor` + 空 `data-tooltip`）。容器本就是指针事件的
  目标，复用它即可整体复用浮标的显示 / 钉住 / 随指针跟随 / 菜单打开时收起语义。
- **踩坑（重要）**：`data-tooltip` 必须在**创建时**就置空串写上。AppTooltip 先按
  `closest("[data-tooltip], [data-hs-rich-tooltip]")` 解析"指针下的元素"再查注册表；
  首次悬停时若标记尚不存在，`currentElement` 为 null，随后的内容注册因"元素不等于
  当前元素"而不刷新浮标——表现为**第一次悬停永远不显示**，移开再回来才正常。
- **内容拼装留在面板**（形状名 i18n、长度按时间轴显示设置格式化、内联曲线图标），
  与旧实现共用 `buildSingleFadeInfoContent` / `buildCrossfadeGripInfoContent`；
  长度取**生效值**（自动交叉淡化优先于手动）。抓手 = 双列（前块淡出在前）。
- **去重**：只在命中**身份**变化时回调（指针沿包络线移动会命中相邻采样块，逐帧
  回调会让浮标内容重建、位置抖动）。
- 验证：悬停淡入线 → 与旧实现**逐字相同**（`淡入类型：/ 长度：0.1.200 / 0:0.600 /
  曲率：-0.50`）；悬停交叉点 → 双列（淡出 + 淡入）；移出 → 浮标消失 ✓

### 残留：裁切 / 拉伸的波纹跟随预览（已完成）

- **抽取**：`computeRegionRightEdgeDelta` 从 `useEditDrag` 迁入 `stretchGroup`
  （单一事实来源，+5 条单测）：波纹驱动量 = 编辑**区域右缘**净位移（不是锚点右缘、
  不是各成员位移之和），且**必须带符号**——负位移被吞掉会表现为"向右有波纹、向左没有"。
- **内核接入**：`handleKernelTrimPreview` 的三个出口（单 clip 拉伸 / 组拉伸 / 裁切）
  统一调 `applyKernelTrimRipplePreview`；快照在按下时按旧实现同源规则建
  （原点 = 参与者最早起点、轨道集 = 参与者所在轨道）；用
  `store.getState().session`（同步新鲜，batch 内 `sessionRef` 落后一帧）。
- 取消路径还原跟随集原位（提交时不还原——后端权威结果写回，保留乐观位置避免回跳）。
- 验证：`rippleMode=track` 拖右缘 +0.633s → 两个跟随 clip 同步 +0.633；Esc →
  全部回到原位；对照 `rippleMode=off` → 跟随集不动 ✓

---

## 批次 E：一致性收尾（P2）—— 全部完成

| 任务 | 内容 | 验证 |
|---|---|---|
| E-1 | 多选修饰键走键位绑定（`modifier.clipMultiSelectToggle` / `clipRangeSelect`）+ Shift 范围选择 + **普通单击维护范围锚点** | ⌘ 点击 → 追加选择（与旧实现逐字相同）；Shift 点击 → 4 个 clip 范围选择（与旧实现对照一致）✓ |
| E-2 | Esc 覆盖 `crossfade-grip` / `snap-offset-drag` / `box-select`，并补齐 snap offset 的**乐观值回滚** | 框选 3 个 → Esc → 选择回空且不弹菜单；抓手拖拽 → Esc → 双方 length/start 全回滚；snap offset 0.25→0.5967 → Esc → 0.25 且**无后端写入** ✓ |
| E-3 | 滚动条 track 点击翻页（原生滚动条的等效交互）：新增 `scrollTargetFromTrackClick`（+7 单测） | 点 thumb 右侧 0→1664→3328，点左侧 3328→1664（每次恰好一屏）✓ |
| E-4 | Vertical Lock：内核手势**钳零水平位移** + 行高亮 + `Vertical Lock` 徽标（配色逐值与旧实现一致） | 拖到相邻轨 → 行底 y=192 h=80、徽标显示、`startSec` 保持 2 不变而轨道改为 track-2（与旧实现对照一致）✓ |
| E-5 | 陈旧注释（`featureFlag` / 宿主 / 视图 props / `TimelinePanel` 头注释）与 `glyph/*` 死代码判定 | 注释-only 改动（`git diff` 无非注释行）；`glyph/*` + `gl/glyphProgram` + `gl/glyphQuads` 确认零调用方（见下方说明）✓ |

### E-1 / D-1b 的**共同根因（重要）**

`TimelineKernelView` 的 `stableInteractions` 是一份**手工维护的回调转发清单**
（内核创建时只取一次引用）。新增回调或新增参数若不同步加进去，宿主调用的就是
`undefined` 或旧签名——表现为「功能完全没反应，但没有任何报错」。
本批在 `onSelectClip` 新增参数与 `onFadeHover` 上**各踩一次**。
现已：① 补齐全部 32 个回调；② 在清单上方写明该约束。
另补齐 `kernelInteractions` 依赖数组漏列的 13 个回调（原先一直持有首帧闭包）。

### E-5 关于 `glyph/*` 的结论（**未删除，仅判定**）

`glyph/glyphAtlas` / `glyphLayout` / `glyphRasterizer` + `gl/glyphQuads` +
`gl/glyphProgram` 共 6 个文件**确无调用方**（`glyphProgram` 零导入；其余只被彼此的
`import type` 与自身测试引用；`timeline/index.ts` 桶文件不覆盖 `kernel/`）。
但**内核的文字走 Canvas2D 细节层**（`drawTimelineCanvas` → `drawClipDetails` 的
`fillText`），即该 WebGL 字形管线是"架构上被绕过"而非"待接线"。
**本轮只清理陈旧注释、保留代码**：删除属破坏性操作且与"功能对齐"目标无关，
建议单独决策（若要删，6 个文件连同 3 个测试一并删，无能力损失）。

---

## 提交约定

每批一次提交（`feat(timeline-kernel): ...` / `fix(timeline-kernel): ...`），提交前跑：
`npx vitest run`（仅 2 条 keybindings 预先存在失败）+ `npx tsc -b --noEmit` + `npx eslint <改动目录> --quiet` + `npx prettier --check <改动目录>`。

---

## 追加修复（2026-09-12，用户反馈）

### F-1 `Alt` + 拖拽方向反了（已修复）

**现象**：`Alt` + 拖 clip 中部（slip / 内部偏移）时，素材移动方向与指针相反。

**根因（坐标域混淆）**：`computeSlipWindow(clip, deltaSec)` 的 `deltaSec` 是
**窗口平移量**（正 = 源窗口向素材后段平移），而不是屏幕位移。两个调用方分属不同域：

| 调用方 | 传入的量 | 域 | 结果 |
|---|---|---|---|
| 旧实现 `useSlipDrag` | `起点指针 − 当前指针` | 窗口域（向右拖为**负**） | 正确 |
| 渲染内核 `handleKernelDragPreview` | 内核回调的 `deltaSec` | **屏幕域**（正 = 向右拖） | 方向整体反过来 |

内核侧的注释**误判了这一点**（原文写「只是位移正负号约定相反」，但取反只做在吸附
分支的 `rawWindowShift` 上，**真正应用窗口的 `computeSlipWindow` 调用漏了负号**）。

**修复**：内核调用改为 `computeSlipWindow(clip, -dApplied)`，并在调用点与函数头
**双向**写明参数约定与「屏幕向右拖 = 窗口负平移」的推导。

**验证**（`?mock=1`，KERNEL=1 vs KERNEL=0 逐字对照）：

| 手势 | 旧实现 | 修复前（内核） | 修复后（内核） |
|---|---|---|---|
| Alt + 向右拖 100px | `srcStart 0 → -0.6667` | `0 → +0.6667` ❌ | `0 → -0.6667` ✓ |
| Alt + 向左拖 100px | `0 → +0.6667` | `0 → -0.6667` ❌ | `0 → +0.6667` ✓ |

单测：`slipWindow.test.ts` +9 条（方向约定、倒放镜像、速率换算、loop 环绕、
非 loop 倒放保持跨度、屏幕域换算自证）。

### F-2 `TAURI_UI_MODE=build` 下界面与 dev 差异较大（已定位：**非缺陷，设计如此**）

**结论**：不是构建损坏，而是**渲染路径不同**——build 模式下内核默认关闭。

**机制**：`scripts/tauri-before-dev.mjs` 的 `build` 模式跑 `npm run build` +
`vite preview`，即加载**生产包**；而 `featureFlag.isTimelineKernelEnabled()` 在未显式
写 `localStorage` 时返回 `import.meta.env.DEV`——生产包恒为 `false`，于是时间轴走
**旧 DOM 实现**；dev 模式走**新渲染内核**。两者是两套渲染实现，观感不同属预期。

**取证（排除"构建确实坏了"）**，1920×1200 空工程，dev vs 生产包：

| 检查项 | 结果 |
|---|---|
| 主题变量（`--qt-*`）/ body 字体 / 字号 | **完全一致** |
| DOM 骨架（root 子树逐行比较） | **0 处差异**（81 行全同） |
| 计算样式（563 个共同节点 × 25 个属性） | **0 处差异** |
| 内核画布几何（8 块 canvas 的尺寸与位置） | **完全一致** |
| 唯一结构差异 | dev 多出 PERF 悬浮面板的 10 个节点（dev-only，预期内） |

把两边**都强制 `KERNEL=1`** 后再比：计算样式 **0 差异**、结构差异仅剩 dev-only 面板。
即「生产包本身没有渲染退化」，差异 100% 来自内核开关的默认值。

**因此未改代码**，仅记录结论。若要 build 模式与 dev 观感一致，有三种选择（需决策）：

1. **加显隐开关**：把 `TAURI_UI_MODE` 透传成 `VITE_*` 变量，build 模式也默认开内核
   （改动最小，但会改变「生产默认关」的既有发布策略）；
2. **保持现状**：把 `TAURI_UI_MODE=build` 理解为「验证旧实现 / 发布路径」，
   dev 模式才验证内核（与当前文档定位一致）；
3. **反转默认**：内核全环境默认开、旧实现退化为逃生门（对齐设计文档里
   「稳定后默认开再删旧路径」的目标态；前提是新旧对齐已完成——本轮刚补完全部缺口，
   真机回归尚未做）。

**顺带发现（不影响本结论）**：生产包在无后端（纯 `vite preview`）时会打印
`Backend invoke failed: pywebview:get_ui_settings / get_runtime_info` 并弹错误提示——
因为生产包不安装 dev-only 的 mock 后端（`?mock=1` 被 `import.meta.env.DEV` 门控）；
在 Tauri 里由真实后端提供这些命令，属预期行为。

### F-3 纵向拖拽时间轴有「吸附感」（已修复）

**现象**：纵向上拖拽时间轴时，末段感觉「卡住 / 吸附」一下。

**根因（两侧竖直上限不一致，差 32px）**：竖直滚动范围由两处**各自**计算：

| 位置 | 内容高度公式 | 缺少的元素 |
|---|---|---|
| 左侧轨道头（`TrackList` 的滚动容器） | 真实 DOM 高度，含底部「添加轨道」行 | — |
| 渲染内核（`ScrollKernel.contentHeightFor`） | `tracks × rowHeight` | **`TRACK_ADD_ROW_HEIGHT`（32px）** |

旧实现的原生滚动容器内容层用的是面板的 `contentHeight`
（= `tracks × rowHeight + TRACK_ADD_ROW_HEIGHT`，见 `useTimelineState`），
所以旧实现两侧**天然同源**；内核自绘滚动后只按轨道数算高，于是：

- 内核竖直上限 329px，轨道头 361px（实测差值恒为 32px）；
- 在**轨道头**里拖拽超过 329px 后，轨道头继续滚到 361，**时间轴已停住** ——
  两侧行错位、指针还在动但内容不动，松手后视觉上"回弹/吸附"。

**取证（修复前）**：在轨道头中键拖拽到底 →
`{"finalK": 329, "finalTL": 361}`（内核与轨道头相差 32px）；
旧实现同姿势实测 `legacyScroller.max = 361`、`trackList.max = 361`（**delta 0**，完全同步）——
即**轨道头是正确的一方**，内核算少了。

**修复**：
1. `ScrollKernel` 新增可选 `extraContentHeightPx`（**额外量**而非绝对高度：
   内核不必知道「添加轨道行」这个领域概念，调用方给差额）；
2. 宿主注入 `() => TRACK_ADD_ROW_HEIGHT`；
3. 新增宿主私有 `verticalContentSizePx()`，让**自绘滚动条**的 thumb 长度 /
   拖拽换算 / 轨道翻页量与滚动上限同源（否则滚到底时 thumb 到不了轨道末端、
   点轨道翻页的目标值也与实际上限不符——末段会"跳一下"）；
4. **刻意不动**两处 `contentBottomPx = tracks × rowHeight`：那是网格线与行分界线的
   绘制下界（添加轨道行不画网格），与滚动范围是两件事。

**验证**：
- 修复后同姿势拖拽：`{"finalK": 361, "finalTL": 361}`，逐帧日志两侧完全同步；
- 滚到底时 thumb：`106.467 + 44.533 = 151` = 轨道全长（精确到末端）；
- **逐行对齐**（按 Track 名称比对，排除虚拟化窗口干扰）：scrollTop = 361 时
  Track 3/4/5/6 的实测边界与内核算出的期望边界**逐项相同**（`allAligned: true`），
  0 / 100 / 200 位置同样对齐。
- 单测：`scrollKernel.test.ts` +5 条（额外高度计入上限、缺省 0 行为不变、
  非法/负值不放大上限、函数形式在 `reclamp` 后生效）。
