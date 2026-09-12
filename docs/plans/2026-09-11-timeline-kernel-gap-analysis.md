# 时间轴渲染内核 · 功能缺口分析（对照旧实现与用户手册）

> 调研日期：2026-09-11 · 分支 `feature/timeline-unified-render-kernel`
> 对照基准：旧实现（`timeline/` 下 DOM 路径）与 `docs/i18n/USERMANUAL.md`
> 目的：列出内核模式下**确实缺失或行为分叉**的能力，给出证据、手册依据与修复思路，作为补全计划的输入。

---

## 0. 结论摘要

内核已覆盖：滚动 / 缩放 / 中键平移 / 自绘滚动条 / 键盘翻页、选中（点击 + 多选 + 框选）、拖拽移动 / trim / 淡变角 / 交叉点抓手 / snap offset 手柄、右键菜单（clip / 轨道区 / 速率）、行内编辑（重命名 / 增益 / 速率）、静音 / 共振峰 / 速率菜单、copy 拖拽 ghost、HTML5 素材拖入（事件已挂内核容器）、吸附（含高亮）、波形层。

**缺口 21 项**，按影响分为三档：

| 档 | 含义 | 项数 |
|---|---|---|
| **P0** | 数据语义错误或手册明确功能失效（用户可感知的「不能用了」） | 7 |
| **P1** | 手册明确描述、内核未实现的功能 | 9 |
| **P2** | 一致性 / 细节（行为偏差、提示缺失、陈旧代码） | 5 |

---

## 1. P0：数据语义 / 可用性缺口

### P0-1 素材拖入（文件浏览器 / 系统文件 / Tauri）落点与预览全部失效

**现象**：内核模式下把文件拖到时间轴，落点永远是**当前选中轨道**、无 dropPreview；文件浏览器面板的拖拽（`hifi-file-drag` 自定义通道）完全不工作。

**证据**：
- `trackIdFromClientY`（`hooks/useTimelineState.ts:877-886`）第一步 `if (!scroller) return null` —— 内核模式 `scrollRef.current === null`。
- 三处落点解析全部走它：HTML5 DnD `TimelinePanel.tsx:2743 / 2798`；`hifi-file-drag` `hooks/useTimelineDragDrop.ts:402 / 445`；Tauri 原生 `hooks/useTimelineDragDrop.ts:151`。
- `hifi-file-drag` 的预览/落库还额外依赖 `scrollRef.current` 的 bounds 与 `scrollLeft`（`useTimelineDragDrop.ts:142-155, 370-401, 443-447`），内核模式直接 `isOverTimeline === false` → 静默无操作。
- D-2 只把 HTML5 `onDragOver/onDrop` 挂到了内核容器（`TimelinePanel.tsx:3031-3032`），未解决 `trackIdFromClientY` 的视口来源。
- 面板传 dropPreview 时 `trackId === null` 会被丢弃（`TimelinePanel.tsx:3013-3015`）→ 预览不可见。

**手册依据**：第四章「左键拖拽文件可以将一个或多个媒体文件批量地跨时间添加到时间轴中。右键拖拽文件可以弹出 `跨时间添加` / `跨轨道添加` 的选单」。

**修复思路**：引入「模式无关」的落点解析——`trackIdFromClientY` 改为可注入视口（内核取 `host.getViewport().scrollTop` + 容器 rect），或新增 `resolveDropTarget(clientX, clientY, containerEl)` 由面板按模式分派；`useTimelineDragDrop` 的 scroller 访问改为注入访问器（与既有 `dragScrollLeftOf` 同一模式）。规模 **M**。

### P0-2 自动滚屏失效（播放时视图不跟随播放头）

**现象**：开启「自动滚屏」后播放，时间轴不再水平跟随。

**证据**：`TimelinePanel.tsx:220-237` 的自动滚动逻辑 `if (autoScrollEnabled && transport.isPlaying)` 内第一步 `const scroller = scrollRef.current; if (scroller) {...}` —— 内核模式恒为 null，整段跳过。

**手册依据**：工具栏「`自动滚屏`：启用后，在播放时，自动对界面进行水平滚动操作，以跟随播放头位置」。

**修复思路**：把「写滚动位置」抽成模式无关调用（旧：`applyNativeScrollLeft(scroller, next)`；内核：`host.setScrollLeft(next)`），自动滚屏复用同一入口。规模 **S**。

### P0-3 参数编辑器「同步时间轴视图」失效

**现象**：启用同步后，轨道区与参数编辑器的水平位置/缩放不再双向联动。

**证据**：`hooks/useTimelineState.ts:564-598`（订阅共享视口并写 `scroller`）、`:604-630`（layout effect 落地）、`:616-630` 均在 `scrollRef.current` 为 null 时 return；内核模式下 `timelineViewportSync` 的写入无人消费。

**手册依据**：第五章「参数编辑器标题左侧的 `同步时间轴视图` 按钮……两者的水平位置与缩放完全一致并双向联动」。

**修复思路**：把「应用共享视口」的落地分支改为模式无关（内核 `scroll.setZoom` + `setScrollLeft`），并把内核视口变化回灌 `timelineViewportSync`。规模 **M**。

### P0-4 键盘缩放（时间轴放大/缩小）与「聚焦播放光标」失效

**现象**：`时间轴放大/缩小` 快捷键、`聚焦播放光标` 快捷键在内核模式下无反应；粘贴后自动聚焦播放光标也失效。

**证据**：
- `hifi:zoomTimelineFocus`：`hooks/useTimelineEventHandlers.ts:370-371` `if (!scroller) return`。
- `hifi:focusCursor`：同文件 `:449-450` `if (!scroller) return`。
- 粘贴后聚焦：`TimelinePanel.tsx:556-560` `if (!scroller) { dispatch(setPendingPlayheadReveal(null)); return; }`（直接丢弃请求）。
- 键盘缩放落地 layout effect：`useTimelineState.ts:633-645`。

**修复思路**：三者都只需「写滚动位置」这一件事——统一走 P0-2 抽出的模式无关入口；`pendingPlayheadRevealSec` 的分支改为内核 `host.setScrollLeft`。规模 **S**。

### P0-5 内核手势不支持「免吸附修饰键」（Shift 临时切换吸附）

**现象**：拖拽/trim/淡变/snap offset 过程中按住 `modifier.clipNoSnap`（默认 Shift）不会临时取反吸附；旧实现支持。

**证据**：`TimelinePanel.tsx:1471-1472` 自述「内核所有手势共有的缺口」；内核预览回调只读 `s.snapEnabled`（`handleKernelDragPreview` `TimelinePanel.tsx:1241`、trim/fade/snapOffset 同）。

**手册依据**：「修饰键 `Shift`：按住后，可以切换是否吸附到网格」。

**修复思路**：与 copyMode 同一模式——宿主在预览/提交回调上透出修饰键快照（`KernelDragModifiers` 已存在），面板统一 `effectiveSnap = snapEnabled XOR noSnapHeld`。规模 **S**。

### P0-6 内核拖拽只作用于「单个 clip」：多选整体拖动与编组联动缺失

**现象**：选中多个 clip 后拖动其中一个，只有它移动（旧实现整组移动）；同组 clip 也不联动。

**证据**：内核 `onDragPreview/onDragCommit` 只带一个 `clipId`（`host:2654, 2670`）；面板提交只发一条 move（`TimelinePanel.tsx:1340-1351`）。旧实现 `useClipDrag.ts:255-267` 用 `multiSelectedClipIds` + `expandClipIdsWithGroups` 展开，`:571-605` 批量预览，提交走 bulk。

**手册依据**：「所有同组的音频块，在执行一些编辑操作时，会被联动编辑」。

**修复思路**：面板侧在按下时展开「实际参与集合」（复用 `expandClipIdsWithGroups` + 多选集合 + `ignoreGrouping`），预览/提交按集合批量；内核仍只给锚点 clip 的位移。规模 **M**。

### P0-7 波纹编辑未参与内核拖拽

**现象**：开启「波纹编辑」后拖动 clip，后续 clip 不跟随前移。

**证据**：旧实现 `useClipDrag.ts:290-297`（`buildRippleFollowers`）+ `:595-603`（跟随位移）；内核路径无任何 ripple 处理。

**手册依据**：「`波纹编辑`：启用后，会自动对轨道上音频块的一些编辑操作执行自动跟进」。

**修复思路**：面板预览/提交阶段复用 `buildRippleFollowers` / `applyRippleFollowerShift`。规模 **M**。

---

## 2. P1：手册明确描述、内核未实现

### P1-1 `Alt` + 拖边缘 = 拉伸（stretch），`Alt` + 拖 body = slip（内部偏移）

**证据**：内核 `clip-trim` 只做裁切（`handleKernelTrimPreview` 改 `sourceStart/EndSec`）；无 stretch / slip 分支。旧实现 `clip/ClipEdgeHandles.tsx:84,177`（`modifier.clipStretch`）、`useClipDrag.ts:240-248` + `useSlipDrag.ts`（`modifier.clipSlipEdit`）。
**手册依据**：「修饰键 `Alt`：按住后，拖动音频块开头/结尾可以拉伸音频块，拖动音频块中间可以调整音频块的内部偏移量」。
**规模** M。

### P1-2 `Alt + Shift` 竖直拖 clip = 调整该 clip 范围内音高

**证据**：内核无 `modifier.clipPitchDrag` 分支；旧实现 `ClipItem.tsx:594-600` + `useClipPitchDrag.ts`（参数编辑器音高线实时跟随）。
**手册依据**：「修饰键 `Alt + Shift`：按住后在音频块上竖直拖动，可以整体调整该音频块的音高，参数编辑器中的音高线会实时跟随变化」。
**规模** M。

### P1-3 增益旋钮拖拽调整 / 双击重置 0 dB

**证据**：内核把 `gain-knob` 单击映射为「打开行内编辑」（`host:2239-2246` → `onBadgeEditStart`）；旧实现旋钮是**拖动调值**（3px 阈值后 `startEditDrag(...,"gain")`，`clip/ClipHeader.tsx:489-558`）+ **双击重置 0 dB**（`:559-564`）。
**手册依据**：「音频块左上角的音量旋钮依然可以直接上下拖动，双击恢复为 0 dB」。
**规模** S。

### P1-4 分组锁链徽标点击（启用/禁用编组联动）+ 编组激活的金色描边

**证据**：`host:2248` 明确注释「分组交互本期未接」；旧实现 `clip/ClipHeader.tsx:592-699` → `onToggleGroupDisabled(groupId)`；`scene/clipInstances.ts:28-29` 记录「编组激活的 clip 旧实现退回 Canvas2D 补外圈描边，内核仅缺少该描边」。
**手册依据**：「此时点击音频块左上角的锁链按钮，可以临时禁用或者启用该编组的联动编辑效果」。
**规模** S（描边 S / 徽标点击 S）。

### P1-5 淡化专属右键菜单与淡变悬停提示缺失

**证据**：内核 `contextmenu` 只处理速率标签与通用 clip/轨道菜单（`host:2817-2844`），无 `requestOpenFadeContextMenu`；`FadeContextMenuHost`（`TimelinePanel.tsx:3705`）在内核模式下无触发源。淡变富 tooltip（`publishFadeRichTooltip`）同样无触发源。
**手册依据**：「右键淡化包络线或淡化区域边缘竖线，会打开淡化专属上下文菜单：上方为 7 个形状按钮……下方为曲率滑块与迷你曲线预览」；「鼠标悬停在包络线、边缘竖线、交叉点抓手等淡化控件上时，会显示浮动提示」。
**规模** M（菜单）/ S（tooltip）。

### P1-6 多 Take：平铺显示与点击切换活跃 Take

**证据**：内核 GL 只画单个块面，无 take lane 分界线；`hitTest` 无 inactive take lane 分区。波形面经 `TimelineWaveformSurface` 已按 `showAllTakes` 展开 lane（`TimelineWaveformSurface.tsx:70-95`），因此**波形可能已平铺**，缺的是分界线与命中切换。旧实现 `ClipItem.tsx:605-663` + `takeLanes.ts:17-63`。
**手册依据**：「`视图 -> 显示所有 Take (如果空间足够)`……多 Take 音频块在轨道高度足够的情况下会平铺显示所有 Take」。
**规模** M。

### P1-7 静音检测预览覆盖层缺失

**证据**：红色静音区间覆盖层渲染在 `ClipItem.tsx:901-921`（旧分支 DOM），内核模式不挂载。
**手册依据**：「对话框打开后会自动预览分析结果：被判定为静音的区间会以红色覆盖显示在音频块上」。
**规模** S（可复用内容坐标层机制）。

### P1-8 拖到轨道区下方新建轨道（ghost + 落库）

**证据**：内核 `resolveTargetTrackIndex` 越界钳制回原轨，`dropToNewTrack: false`（`TimelinePanel.tsx:1313`）；旧实现 `clipDropNewTrack` + `newTrackGhostClips`（`TimelinePanel.tsx:3307-3360`）+ `buildDropToNewTrackMoves`。
**手册依据**：无逐字描述，但属 Reaper 类拖拽语义；且旧实现有。
**规模** M。

### P1-9 点击空白清空选择 + 点击切换当前轨道

**证据**：内核空白点击只做 seek（`host:2296-2299`），不清空选中、不切轨；旧实现 `TimelinePanel.tsx:3170-3220`（`deselectAllTrackLaneClips` + `paramEditorTimelineClickSelectTrackEnabled` 时 `selectTrackRemote`）。
**手册依据**：「`允许时间轴点击切换轨道`：默认启用。启用后，点击时间轴中的音频块或空白区域会自动切换当前轨道」。
**规模** S。

---

## 3. P2：一致性 / 细节

| 编号 | 缺口 | 证据 | 规模 |
|---|---|---|---|
| P2-1 | 多选修饰键写死 Ctrl/⌘，未走键位绑定 `modifier.clipMultiSelectToggle`；**Shift 范围选择**（`clipRangeSelect`）内核无对应 | `host:2744`；旧实现 `ClipItem.tsx:577-588`、`TrackLane.tsx:436-453` | S |
| P2-2 | Esc 取消不覆盖 `crossfade-grip` / `snap-offset-drag` / `box-select` | `host:2975-3012` | S |
| P2-3 | 自绘滚动条不支持「点击 track 空白跳转」（`hitTestScrollbarThumb` 已实现但未被宿主使用） | `input/scrollbars.ts:103-112` | S |
| P2-4 | 垂直换轨锁定的行高亮与 "Vertical Lock" 徽标提示缺失 | 旧实现 `TrackLane.tsx:302, 730-736, 790-803` | S |
| P2-5 | 陈旧注释与死代码：`featureFlag.ts:29`、`host:30-32` 仍写「Spike 未接入标尺/波形/交互」；`TimelineKernelView.tsx:84-101` 注释块错位；`glyph/*` 全模块无调用方（约 5 文件） | — | S |

**另需注意的偏差（非缺失，但属行为分叉）**：
- 徽标编辑触发方式：旧实现是**双击**徽标进入输入（`ClipHeader.tsx:878-949`），内核是**单击**（`host:2239-2246`）。手册写「双击徽标即可在原位置输入数值」。
- 内核 `onSelectClip` 的 additive 只识别 ctrl/meta，未识别键位绑定后的等价修饰键。
- `TimelinePanel.tsx:2410` 的 `viewportHeightPx: scrollRef.current?.clientHeight ?? 0` 在内核模式恒为 0（该 render model 仅旧分支消费，风险低，但应清理）。

---

## 4. 已确认「非缺口」（避免重复劳动）

- 标尺（含 Tempo Map 行、时间单位菜单、悬浮标签）：DOM 共用，内核模式正常。
- 左侧轨道头（含拖拽排序/编组、音量旋钮、电平表、颜色、重命名）：DOM 共用，且滚动已由内核转发。
- clip 右键菜单、轨道区右键菜单、速率高级编辑对话框、共振峰浮窗、导入模式菜单、工程文件菜单：均在开关之外，入口已接。
- 吸附引擎与高亮、自动交叉淡化（拖拽预览期）、copy ghost、框选、交叉点抓手、snap offset：已完成。
- 全局快捷键（S 切割 / G·U 编组 / T 切换 Take / Delete / Ctrl+C·X·V / Ctrl+A）：`useTimelineEventHandlers` 在开关之外，内核模式仍生效。

---

## 5. 建议补全批次

| 批次 | 内容 | 价值 | 规模 |
|---|---|---|---|
| **A** | P0-1 拖入落点与预览、P0-2 自动滚屏、P0-3 视图同步、P0-4 键盘缩放/聚焦 | 手册明确的功能恢复，用户最容易感知 | M |
| **B** | P0-5 免吸附、P0-6 多选/编组联动、P0-7 波纹 | 数据语义正确性 | M |
| **C** | P1-1 stretch/slip、P1-2 音高拖拽、P1-3 增益旋钮、P1-4 分组徽标+描边、P1-9 空白点击语义 | 手册逐条对齐 | M |
| **D** | P1-5 淡变菜单/tooltip、P1-6 多 Take、P1-7 静音检测预览、P1-8 拖到空白新建轨道 | 手册逐条对齐 | L |
| **E** | P2 全部 | 一致性收尾 | S |

**建议顺序**：A → B → C → D → E（每批独立可验证、可提交）。
