# 内核收归唯一路径后的回归修复与参数编辑器钢琴背景 · 设计

- 日期：2026-09-13
- 状态：待实施
- 相关：`docs/superpowers/specs/2026-09-13-timeline-single-path-design.md`（本次修复的**前置改造**）、`docs/superpowers/specs/2026-09-12-pianoroll-kernel-migration-design.md`

## 1. 背景

`2026-09-13-timeline-single-path-design.md` 把时间轴内核收归唯一路径，删除了旧实现（`TimelineScrollArea` / `TimelineSurface` / `TimelineCanvasViewport` / `TrackLane` / `ClipItem`）。该改造的验收全部通过，但**上线前的手工验收只覆盖了「内核自己实现的行为」**——没有覆盖「旧实现提供、内核未接管的次要绘制与失效路径」。

用户随后在真机上报告了 4 个缺陷，并在追问中追加了 1 条线索。排查后确认为 **5 个独立缺陷**，其中 **2 个是本次改造引入的功能回归**（`develop` 上有、现在没有），3 个是内核自有的交互/失效缺陷。

**本设计的目标**：修复全部 5 个缺陷，并新增 1 项用户体验改进（参数编辑器钢琴背景）。不扩大范围，不重构无关代码。

### 1.1 缺陷清单与严重度

| # | 缺陷 | 类型 | 根因一句话 |
| --- | --- | --- | --- |
| 1 | 修饰键 + 拖拽复制无法落到新轨道（也不能落到其他已有轨道） | 内核缺陷 | 提交路径把 `dropToNewTrack` / `trackOffset` 写死为 `false` / `0`，丢弃了预览路径已算出的落点 |
| 2 | 点标尺后标尺播放头动、网格播放头不动 | 内核缺陷 | 播放头变化**从不**标脏内核渲染循环（循环是纯脏标记驱动，无空闲 rAF） |
| 3 | 点本轨空白偶尔不取消选中 / 不改变播放头 | 内核缺陷（**与 #2 同源**，见 §4.3） | 空白点击的 seek 确实到达 store，但内核不重绘，用户看到的是「没反应」 |
| 4 | 参数编辑器选区框残留，拖拽时才更新 | 内核缺陷 | 新建选区时只写 `selectionUi`，**未调用 `invalidate()`** |
| 5 | MIDI / 音高参考块内部为空 | **改造回归** | `MidiPitchTrackCanvas` 唯一挂载点（`TrackLane`）被删，组件成孤儿 |
| 6 | 音阶高亮按钮失效 | **改造回归** | 该图层只画在 Canvas2D 的 `skipGrid` 分支里，而 `skipGrid` 恒为 `true` |

> 编号 5/6 在用户报告里是"追问项"，但排查证据表明它们是**同一类**问题（删除旧实现时的未审计附带损失），因此并入本次一并修复（用户已确认范围）。

## 2. 决策记录

### 2.1 为什么 #2 和 #3 合并为一条根因

用户的原始描述把 #3 说成两个症状：「点本轨空白不取消选中」和「偶尔点空白不改变播放头」。实测（`?mock=1`，1920×1200，直接读 Redux store）表明：

| 操作 | `selectedClipId` | `multiSelectedClipIds` | 内核播放头元素 |
| --- | --- | --- | --- |
| 点 clip（轨 1） | — | 1 | — |
| 点同轨空白 | `null` | 0 | **不动** |
| 点他轨空白 | `null` | 0 | **不动** |

即**选中清空一直是正确的**；两个症状里真正能复现的是「播放头不动」，而它的根因就是 #2：seek 已经派发（`set_transport` 可在 mock 调用轨迹里看到）并写入了 store，但内核渲染循环没有被标脏，所以自绘的轨道区播放头永不重绘。用户看到的「没反应」是**重绘缺失**，不是状态没变。

因此 #3 不再单独立项，其验收并入 #2 的验收（「点空白后播放头必须可见地移动」）。

### 2.2 为什么用「新增只标脏入口」而不是「把 playheadSec 加进 invalidateScene 依赖」

`invalidateScene()`（`timelineKernelHost.ts:4111-4113`）会置 `sceneDirty = true`，下一帧**重建全部 GPU 几何**。播放头是每帧变化的量（播放中 60fps），用它触发场景重建等于每帧重传顶点缓冲——这正是内核设计要消除的开销。

因此新增一个**只标脏、不置 sceneDirty** 的入口，与 `renderLoop.invalidate()` 同义。这也与参数编辑器既有的正确做法同构（`PianoRollPanel.tsx:2006-2008` 在 `playheadSec` 变化时 `invalidate()`）。

### 2.3 为什么 #5 走 Canvas2D 细节层而不是 GL 实例

内核的**细节层**（`runtime/timelineCanvasRenderer`，z-index 2，由宿主每帧 `drawTimelineCanvas` 提交）已经承载全部「每 clip 的路径/文字」绘制：淡变曲线、增益/速率角标、静音徽标、静默检测红块、多 Take 分界线。MIDI 音高线正是这一类（一条 per-clip 折线）。

放进 `clipInstances.ts` 需要新增一种实例类型（折线/LINE_STRIP）+ 独立缓冲 + 独立 program，而内核明确把「逐 clip 路径绘制」放在细节层（见 `timelineKernelHost.ts:1133` 的说明）。因此复用细节层，**且折线生成数学必须从孤儿组件抽取复用**，不重写。

### 2.4 为什么钢琴背景放进**网格**实例缓冲，而音阶高亮需要改签名

钢琴背景是「按半音行铺满视口宽的半透明矩形」，与网格线**同一坐标系、同一重建时机**（都只依赖视口 + dpr + 主题），且 `GridInstance` 已经是 `{x,y,w,h,rgba}`，`writeFlatInstance` 的矩形语义就是 `(x,y,w,h)`——一个 `h = 行高` 的矩形与一条 `h = 线厚` 的线同形。放进同一缓冲零新增基础设施。

**但有一个必须处理的约束**：宿主按下标顺序从 0 写入（`pianoRollKernelHost.ts:682-688`），而 `INSTANCE_MODE_FLAT` 在 `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)` 下**按缓冲顺序合成**。因此背景带必须排在网格线**之前**，否则会盖住网格线。

**更关键的约束**：`gridGeometrySignature`（`gridView.ts:159`）当前**不含**键色。签名是"要不要重建几何"的唯一判据，漏项的表现就是「切主题不生效」。新增背景色必须进签名。这与该函数自述的失败模式完全一致。

### 2.5 为什么音阶高亮也需要改签名（而不只是"补上绘制"）

音阶高亮依赖 `projectScale` 与 `scaleSegments`，二者都是**低频**变化（改音阶 / 改 Tempo Map），但都不在 `gridGeometrySignature` 里。若只把绘制搬到 GL，切音阶后高亮不会更新——同一个漏项陷阱。因此签名必须同时纳入「音阶高亮是否开启 + 生效音阶音级集合 + 分段」。

### 2.6 钢琴背景的具体观感（用户已确认）

- **只给黑键行加半透明深色带**，白键行保持原背景。理由：这是 REAPER / Logic / Ableton 的通行做法，最不干扰曲线读数；也给不出「白键带在浅色主题下与背景糊在一起」的问题。
- **常开，不加开关**。少一处状态与持久化。
- **音阶高亮叠在钢琴背景之上**：钢琴背景是底层纹理（黑键/白键），音阶高亮是语义高亮（该音阶的音级），两者语义不冲突。
- 强度取「克制的」档：深色主题 alpha ≈ 0.08，浅色主题 ≈ 0.06。用户可后续要求调整（本设计把它做成 `colors.ts` 里的具名常量，便于一处调整）。

## 3. 实施范围（精确清单）

### 3.1 缺陷 1：修饰键拖拽落点

**改动点**：`frontend/src/components/layout/TimelinePanel.tsx` 的 `handleKernelDragCommit` copy 分支（约 `:1712-1756`）。

1. 在**清空之前**捕获落点：`const dropToNewTrack = kernelDropToNewTrackRef.current;`
2. 在同分支内清空 ref + state（修复 `:1755` 提前 `return` 造成的**虚线新轨行永久残留**状态泄漏）。
3. `trackOffset` 由 `args.targetTrackId` 与 `origin.anchorTrackIndex` 推导，复用预览分支（`:1590-1594`）的同一算法——不另写一份。
4. `createNewTracksForDrop` / `createNewTrackForDrop` 换成真实实现（复用 move 分支已有的 `createNewTrackForKernelDrop`）。**必须与第 1 步同时落地**：`copyClipsFromDrag.ts:137-138` 在 `dropToNewTrack` 为真而 creator 仍返回 `null` 时抛 `create_track_failed`。
5. 删除 `:1733-1734` 的失效注释（它声称"内核拖拽的落点始终是已有轨道"，与 `:1597` 直接矛盾）。

**顺带修正**：`TimelinePanel.tsx:4457-4460` 的新轨幽灵行条目取自 Redux 里 `trackId === NEW_TRACK_SENTINEL` 的 clip；copy 模式下原 clip 不动，因此该行**预览为空**。改为从 `kernelGhost` 取（移动模式继续走 Redux，或两者统一走 ghost——实现时按最小改动选一条并在注释里写明理由）。

### 3.2 缺陷 2（含 #3）：播放头重绘

**改动点**：

1. `timelineKernelHost.ts`：新增一个**只标脏**的公开入口（命名表达"仅播放头重绘"，例如 `invalidatePlayhead()`），内部只调 `loop.invalidate()`，**不**置 `sceneDirty`。
2. 同文件：把已声明但**从未被解构读取**的 `args.playheadSec?: () => number`（`:294`）接上——`syncDom()`（`:1816`）与 `draw()`（`:1900`）改为经一个内部 `readPlayheadSec()` 读取，优先取该 getter，缺省回退 `data().playheadSec`。这消除「数据镜像滞后一次提交」的次要缺陷（镜像在 render 期写，而视觉插值 ref 在 effect 里更新）。
3. `TimelineKernelView.tsx`：把已有的 `getPlayheadSec`（`:569`）同时传给宿主的 `playheadSec` 参数（目前只喂 `buildData()`，`:369`）。
4. `TimelinePanel.tsx` 的 `TimelineTransportBridge.onFrame`（`:270-326`）在写完 `visualPlayheadRef.current` 后请求一次播放头重绘——经 `TimelineKernelView` 暴露的稳定回调 / `kernelHostRef` 调用上述入口。
5. `TimelinePanel.tsx`：删除孤儿 `playheadRef` 的全部写点（`:217`、`:236`、`:303-305`、`:4013-4022`、`:5056`）与 `useTimelineState.ts` 中的声明（`:155`、`:395`、`:1348`）。`scrollRef` 同理（`:606` 分支已注明"恒为 null"）——一并核实后清理，避免它们继续看起来像活写入点。

**验收口径**：点标尺、点空白、播放（**自动滚屏关闭**）三条路径下，网格播放头都必须可见移动。

### 3.3 缺陷 4：参数编辑器选区框

**改动点**：`frontend/src/components/layout/pianoRoll/usePianoRollInteractions.ts` 的新建选区分支（`:3356-3447`）。

`selectionRef.current = {...}` + `updateSelectionUi(...)` 之后**补 `invalidate()`**——与拖拽分支（`:3437`）一致。当前 `updateSelectionUi` 只走 React state，而 `selectionRef` 是 `drawPianoRoll` 读取的真值，两者之间没有任何重绘触发（已实测：第二次按下后主画布 `toDataURL()` 与按下前**逐字节相同**）。

同时核实 `pointerup` 分支（`:3445`）的 `invalidate()` 是否足够——若按下即选区已坍缩为零宽，抬起时的重绘同样需要在按下时就发生。

### 3.4 缺陷 5：MIDI / 音高参考块内容

**改动点**：

1. **新建** `frontend/src/components/layout/timeline/runtime/midiPitchCurve.ts`：从 `MidiPitchTrackCanvas.tsx` **逐字搬出**纯函数 `generateMidiCurveFromNotes`、`resolveLoopCycleDescriptor`、`strokeColorForClip`、曲线缓存（`WeakMap` + 每 notes 最多 4 个几何变体）与那些已经落在 `utils/loopRender.ts` 的共享 helper 的引用关系。搬出后 `MidiPitchTrackCanvas.tsx` 改为从这里 import（若该组件确认无其他消费者，也可直接删除——实现时二选一并说明）。
2. `runtime/timelineCanvasModel.ts`：给 `TimelineCanvasClipModel` 增加**可选**字段，承载该 clip 的音高折线（视口坐标点序列 + 描边色 + 透明度），仅在 `clip.midiNoteCount != null` 时构建。需要 `clip.midiNoteData`、`sourceStartSec/sourceEndSec`、`playbackRate`、`reversed`、`loopEnabled`、`lengthSec`、`color`、`muted`——`ClipInfo` 全部已有。
   - **兜底数据源**：`midiNoteData` 缺失时回退 Redux `clipPitchCurves`（后端 `clip_pitch_data` 事件路径）。这是既有两条数据源，必须都保留。
3. `runtime/timelineCanvasRenderer.ts`：在细节层新增一段 per-clip 折线描边，y 映射沿用 `MidiPitchTrackCanvas.tsx:543` 的 `displayH - padding - normalized*(displayH-2*padding)`（padding = 10%），`globalAlpha = muted ? 0.4 : 0.85`。折线必须只在 clip body 区间内绘制（沿用 `item.bodyTop` / `item.bodyHeight`）。
4. **环回标记**（`drawLoopMarkers` 的 ▽）同样丢失，一并恢复；它复用 `utils/loopRender.ts` 的既有实现。

**不迁移**：`MidiPitchTrackCanvas` 的 rAF + `timelineViewportBus` 架构。细节层已由宿主每帧提交，自带视口跟随，不需要第二个滚动订阅者（这也是它当初与内核不同帧的根源）。

### 3.5 缺陷 6：音阶高亮

**改动点**：

1. `pianoRoll/kernel/scene/gridInstances.ts`：`PitchGridArgs` 增加音阶高亮输入（是否开启、生效音级集合 `pc[]`、可选分段 `{startSec,endSec,scale}` 与视口投影）。在 `buildPitchGridInstances` 里，对每个 `pc ∈ scaleNotes` 的半音行**额外**产出一条 2× 线厚的强调线（沿用 Canvas2D 的 `rgba(255,200,80,0.22)` / `rgba(200,120,20,0.22)` 两套主题色，配色下沉到 `colors.ts`）。
2. `gridView.ts`：`gridGeometrySignature` 纳入「音阶高亮开关 + 音级集合 + 分段签名」，否则切音阶不重建几何。
3. `PianoRollPanel.tsx` 的 `buildGridSpec()`：把 `s.scaleHighlightMode`、`effectiveProjectScale`、`segCache.result` 写入 `PianoRollGridSpec`。
4. 删除 `render.ts` 中已死的 `scaleHighlightMode` / `projectScale` / `scaleSegments` 绘制分支（`skipGrid` 恒真，它永远不执行），并把 `drawPianoRoll` 签名里随之失效的参数一并清理——**但保留 `args` 里仍被其它分支使用的项**，逐项核对。

### 3.6 新功能：参数编辑器钢琴背景

**改动点**：

1. `pianoRoll/colors.ts`：新增具名常量（如 `blackKeyRowBand`），深浅两套主题各一档（≈ 0.08 / 0.06 alpha）。**必须用 `rgb()/rgba()` 写法**——`parseRgbaColor` 只认这两种，hex 会被解析成不透明洋红（该文件 `:122-149` 已用血泪教训记录了这一点）。
2. `pianoRoll/kernel/scene/gridInstances.ts`：`buildPitchGridInstances` 在最前面（**先于任何网格线**，见 §2.4）为每个黑键半音产出一条 `x=0, w=viewportWidthPx, y=该半音上缘, h=该半音高度` 的半透明矩形。复用已有的 `isBlackKey`（`pianoRoll/utils.ts`，是无依赖叶子模块，node 单测可引用）与既有的设备像素对齐取向。
   - **只对 pitch 生效**：非音高参数没有「黑键」概念，`buildValueGridInstances` 不产背景带。
   - 键高下限沿用 `Math.max(1, …)`（与键盘轴一致，避免缩到底部时出现空洞）。
3. `gridView.ts`：`gridGeometrySignature` 纳入背景色（否则切主题不生效）。
4. `PianoRollPanel.tsx` 的 `buildGridSpec()`：传入背景色 RGBA。
5. `render.ts`：`skipGrid` 分支里若也有等价绘制则删除（该分支已死，实际无需改动——实现时确认）。

**不做**：不加开关（§2.6），不加持久化字段，不动 Redux。

## 4. 验证方式

1. **全量测试**：`npx vitest run`。当前基线 **838 passed / 2 failed**（2 项为 `keybindingMatch.test.ts` 既有失败，已在 `develop` worktree 复现，与本分支无关）。新增单测只加不减；验收判据是**没有任何用例从通过变为失败**。
2. **类型与静态检查**：`npx tsc -b --noEmit`（仓库已开 `noUnusedLocals`，故新删的死代码会立刻报 `TS6133`）、eslint 0 error。
3. **生产构建**：`npm run build`，并确认产物中 `import.meta.env` 为 0 次。
4. **真机渲染**（`?mock=1` + `scripts/dev-shot.mjs`），逐条对照 §1.1：
   - #1：修饰键拖到末轨之下 → 断言 mock 收到建轨请求且 `trackMode` 不是 `same_track`；拖到其他已有轨道 → 断言 `trackMode` 为 `explicit_mapping`；并断言虚线新轨行**不残留**。
   - #2：读内核播放头元素的 `style.transform`，点标尺 / 点空白 / 播放（自动滚屏**关**）三条路径都必须变化。
   - #4：第二次按下后主画布 `toDataURL()` 必须与按下前**不同**（当前是逐字节相同）。
   - #5：MIDI clip 的 body 内必须出现折线。**注意 mock 数据不含 `midi_note_data`**，因此需要先给 mock 补上该字段（或在验证脚本里注入），否则该 clip 在 `develop` 上同样是空的——这是本次排查中一度无法视觉复现的原因。
   - #6：切换音阶高亮按钮后，GL 画布必须**不再**逐字节相同（当前实测完全相同）。
   - 钢琴背景：截图像素比对确认黑键行被压暗、白键行不变，且网格线仍可见（不被背景盖住）。
5. **CPU-only**：确认 `frontend/scripts/cpu-render-bench.mjs` 仍可运行（本次不预期性能变化；渲染层新增的实例数应记录在案）。

## 5. 风险

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| **签名漏项 → 图层停止更新** | `gridGeometrySignature` 是本工程反复出问题的点（该函数自述："漏项的代价是该输入变化后几何不更新"） | 新增输入（背景色、音阶高亮）必须同时进签名；每一步用「切主题 / 切音阶」真机验证 |
| **实例顺序 → 背景盖住网格线** | 宿主按下标顺序写入，FLAT 按缓冲顺序合成 | 背景带必须先于网格线发射；单测断言输出顺序 |
| **播放头重绘引入每帧 GPU 重建** | 若误用 `invalidateScene()` | 新增入口只标脏；验收时读帧耗时确认播放帧无几何重建 |
| **#1 的建轨与落库必须原子落地** | `copyClipsFromDrag` 在 creator 返回 null 时抛错 | §3.1 第 1、4 步同一提交 |
| **#5 迁移折线数学引入视觉漂移** | 该数学与后端 `emit_clip_pitch_data_for_clip` 逐帧对齐 | 逐字搬迁不重写；对搬出的纯函数补单测 |
| **删除 `render.ts` 死分支时误删活代码** | `skipGrid` 分支内混有 `highlightActive` 等局部计算 | 逐项核对 args 的其余消费者（本工程已有一次同类误删教训） |
| **孤儿 `playheadRef` / `scrollRef` 清理波及面** | 写点分散在两个 hook | 先删写点、跑 `tsc`（`noUnusedLocals` 会指出剩余孤儿），再删声明 |

## 6. 验收标准

1. 修饰键 + 拖拽复制可落到**新轨道**与**其他已有轨道**，落库与幽灵预览一致，且虚线新轨行不残留。
2. 点标尺、点轨道空白、播放（自动滚屏关闭）三条路径下网格播放头均可见移动。
3. 参数编辑器：新建选区时旧选区框立即消失，无需拖动。
4. MIDI / 音高参考块在时间轴 body 内显示音高折线（含环回标记），数据源为 `midiNoteData`，缺失时回退 `clipPitchCurves`。
5. 参数编辑器音高视图背景显示黑键行半透明暗带；音阶高亮按钮恢复可见效果，且叠加在背景之上；切主题 / 切音阶后两者都立即生效。
6. 全量测试无新增失败；tsc 干净；eslint 0 error；生产构建成功且产物不含 `import.meta.env`。
7. 五个缺陷各自的真机验证截图/测量值写入验收记录；两个回归（#5/#6）额外注明「`develop` 有、改造后丢失」，避免后人误判为从未实现。
