# 时间轴统一渲染内核设计（自绘滚动 + 单 WebGL2）

- 日期：2026-09-11
- 状态：设计已确认，待实施
- 范围：`frontend/src/components/layout/TimelinePanel.tsx` 及其 timeline 子树（轨道区 + 标尺 + 左侧轨道头）
- 目标：消除滚动卡顿，把时间轴全部视觉收敛到单一 WebGL2 渲染器，取消跨层同步机制（sync）

---

## 1. 背景与问题

### 1.1 现状架构

时间轴当前由 4 个视觉层 + 1 个 DOM 交互层组成：

| 层 | 技术 | 滚动时的行为 |
|---|---|---|
| 背景网格 | SVG（2 条 path） | 重建 path `d` 字符串 + 2 次 `setAttribute` |
| clip 块面 + 细节 | Canvas2D（sticky） | 全物理清屏 + 全量重绘，含逐 clip `drawClipDetails`（文字 / 徽标 / 旋钮 / fade 曲线） |
| 波形 | WebGL2（sticky） | 平移帧只更新 `u_viewOrigin` uniform；超 margin（128–512px）全量重建 |
| 标尺 / 播放头 | DOM | 命令式写 `transform` / `style.left` |
| clip 交互层 | 透明 DOM（`ClipItem`） | 随原生滚动移动（浏览器合成器负责，免费） |

滚动/缩放走**同步命令式**链路：

```
scroller onScroll
 → useTimelineState.syncScrollLeft()            （同步）
     ├─ 标尺 transform / 播放头 style.left ×3
     ├─ timelineViewportBus.emit()               （同步派发）
     │    └─ frameCommitter.commit(axis)         （同步遍历图层）
     │         ├─ clip-body  paint → Canvas2D 全量重绘
     │         ├─ waveform   paint → uniform 或全量重建
     │         └─ gridOverlay paint → 重建 SVG path
     └─ rAF 量化 setState（水平 256px / 竖直 2×rowHeight 一次）
```

同步机制存在的原因：sticky 画布层不随原生滚动移动，而 DOM 内容层随原生滚动移动，两者必须在同一帧内提交位移，否则出现可见的分层漂移（`utils/timelineViewportBus.ts` 文件头注释明确"严禁改回异步派发"）。

### 1.2 瓶颈（按量级）

1. **clip 体 Canvas2D 每帧全量重绘**（主瓶颈，O(可见 clip)）：清屏 + 逐 clip `prepareClip` + 逐 clip `drawClipDetails`（`measureText` / `fillText` / fade 曲线最多 1200 点）。离线基准（`npm run bench`，`timelineClipPerf.bench.ts`）显示纯计算仅 **0.5ms/帧 @400 clip**，瓶颈在 Canvas2D 实际绘制而非计算。
2. **对角滚动每帧提交两轮**：`TimelineScrollArea.onScroll` 先 emit `(新scrollLeft, 旧scrollTop)`，`TimelinePanel.onScroll` 再 emit `(新scrollLeft, 新scrollTop)`；两次 axis 不等 → `axisEquals` 去重失效 → 三个图层各画两遍。
3. **GL 模式下的纯浪费**：`prepareClip` 仍构建 `fills`/`strokes`，随后被 `pending.filter(!useGl)` 丢弃（`timelineCanvasRenderer.ts:904`）。
4. **波形重建路径**：超出 margin 或 `rows` 引用变化（竖直滚动窗口变动）时全量 scene + geometry + 上传。
5. **React 量化提交的二次绘制**：竖直滚动时 `model` / `rows` 引用变化触发 `TimelineCanvasViewport` 与 `WaveformSurface` 的 layout effect 再画一次。
6. 其他：标尺 `transform` 每帧重复写两次；常驻 rAF 对账循环；播放时每帧全量重绘。

### 1.3 核心判断

卡顿根因**不是 sync 本身**，而是"每次提交都要重绘 O(clip) 内容"。

- 把 sync 从"N 层各自全量重绘"收敛成"一次提交"，约省 10–30%；
- 把"滚动 = 重绘"变成"滚动 = 平移"，才能省 80–95%，而这天然导向统一渲染器；
- 只要保留原生滚动容器 + 随滚动移动的 DOM 内容层，同帧提交就不可去掉。"彻底不用 sync"的唯一路径是**取消原生滚动 + 全 canvas 自绘 + 几何命中**。

---

## 2. 已确认决策

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 滚动载体 | **完全自绘滚动**：`overflow: hidden`，自维护 `scrollLeft/Top`，自绘滚动条，自实现手势 |
| 2 | 渲染器形态 | **单 WebGL2**，文字走**字形图集**（自建 atlas + 布局缓存，中文按需 + LRU） |
| 3 | 覆盖范围 | 时间轴面板**全部视觉**：轨道区 + 标尺（含 Tempo 行）+ 左侧轨道头（含电平表）。参数编辑器（PianoRoll）本期不动 |
| 4 | 落地策略 | **先内核验证（Spike），再分阶段替换**；feature flag 切换新旧，可随时回退 |

---

## 3. 总体架构

**分层原则**：React 只保留外壳与低频 UI，高频滚动 / 渲染 / 交互全部下沉到命令式 runtime。

```
TimelinePanel（React 外壳：布局、菜单、对话框、低频 UI）
└─ TimelineViewport（新内核宿主）
     ├─ ScrollKernel        自绘滚动：scrollLeft/scrollTop/pxPerSec/rowHeight
     │                      输入源：wheel / 自绘滚动条拖拽 / 键盘 / 缩放锚点
     ├─ RenderLoop          rAF 调度：脏标记驱动，按固定层序提交 draw call
     ├─ SceneBuilder        纯函数：session 数据 → 可见窗口+余量的绘制清单
     ├─ GeometryStore       内容坐标几何缓存（网格 / clip 块面 / 文字 / 波形）
     ├─ GlyphAtlas          字形图集（按 dpr 生成，中文按需 + LRU）
     ├─ HitTestIndex        几何命中索引（轨道分桶 + clip 时间排序）
     └─ InteractionController 手势状态机（拖拽 / trim / fade / 框选 / 播放头）
```

**数据流**：输入事件 → `ScrollKernel` / `InteractionController`（纯 runtime，不经 React）→ 标脏 → rAF → `RenderLoop` 绘制；只有**低频结果**（编辑提交、选中变化、菜单请求）才回流 Redux / React。

**核心不变式**：所有几何以**内容坐标**常驻 GPU，滚动帧只更新 `viewOrigin` uniform；只有缩放、内容变化、超出余量时才重建几何。这是"滚动零重绘"的根基，也是取消 sync 的根基——没有任何 DOM 内容层需要对齐，渲染完全由 rAF 掌控。

**DOM overlay 边界**（保留 DOM）：重命名输入框、badge 编辑、右键菜单、tooltip、各类对话框、参数编辑器浮窗。其定位由 runtime 的 `worldToScreen` 提供，在 rAF 内更新。

---

## 4. 滚动与输入内核

**宿主结构**：`overflow: hidden` 的宿主 div + 单个 `<canvas>` + DOM overlay 容器。滚动状态（`scrollLeft / scrollTop / pxPerSec / rowHeight`）由 `ScrollKernel` 持有，只存在 runtime，不经 React。

**输入源**：

| 输入 | 行为 |
|---|---|
| 无修饰键 wheel | 双轴自由滚动（`deltaMode` pixel/line/page 归一化）；触摸板惯性由系统持续派发 wheel，直接 1:1 跟随，不做二次插值 |
| Ctrl+wheel / 绑定键 | 水平缩放（指针锚点保持，复用 `resolveHorizontalWheelZoom`）；Alt+滚动条悬停 = 该轴缩放 |
| 自绘滚动条拖拽 | 水平 / 竖直两条，位置由 rAF 更新 |
| 键盘 | PageUp/Down、Home/End、方向键，走现有 keybinding 体系 |
| 中键拖拽 | 平移（沿用现有 pan 手势） |

**滚动条**：用 DOM（两个 div）而非 canvas 自绘——拖拽命中、悬停态、样式成本最低，且滚动条属于 chrome 而非内容。现有 `nativeScrollbarZoneAt` 的原生滚动条判定改为自绘滚动条的命中检测。

**边界与钳制**：内容宽 = `projectSec × pxPerSec`，竖直 = `轨道数 × rowHeight`；缩放上下限沿用 `resolveTimelineMinPxPerSec` / `MAX_PX_PER_SEC`。所有钳制只在 `ScrollKernel` 内做一次，消除现有"写原生 `scrollLeft` 被浏览器钳制后回读"的整类问题。

**手感与无障碍**：wheel 直接 1:1 跟随（渲染在 rAF 内，最多 1 帧延迟，无撕裂）；宿主保留 `tabindex`、`role`、`aria-*` 与键盘焦点。与参数编辑器的 `timelineViewportSync` 本期保留（PianoRoll 未迁移）。

---

## 5. 渲染内核（单 WebGL2）

**Program 划分（3 个）**：

| Program | 承载内容 |
|---|---|
| `sdf-box`（实例化） | 网格线、轨道分界线、clip 块面 / 分隔缝 / 描边、badge 底、旋钮底、选区框、播放头、Tempo 标记 |
| `glyph-quad`（实例化 + 图集） | 全部文字（clip 名、增益 / 速率、badge 标签、标尺刻度数字、轨道头文字） |
| `wave-vertex`（顶点流） | 波形（复用现有 `surfaceRenderer` 顶点管线） |

**内容坐标 + uniform 平移**：所有几何以内容坐标构建，每帧只更新 `u_viewOrigin(scrollLeft, scrollTop)`、`u_resolution`、`u_dpr`。滚动帧 = 改 uniform + 重发 draw call，**零几何重建**。这是现有波形 `repaint()` 已验证的模式，推广到全部图层。

**几何缓存与余量**：横向余量沿用 `max(1.5s, 512px)`、纵向沿用 4 行 overscan；只在"超出余量 / 缩放 / 内容编辑"时重建。纯计算实测 0.5ms @400 clip，重建帧预算充裕；clip 编辑用 `bufferSubData` 局部更新实例，不整表重建。

**字形图集**：单张 2048² RGBA 图集，用离屏 Canvas2D 按 dpr 渲染字形（保持现有文字质量），shelf-pack 分配；ASCII 预热、中文按需 + LRU 分页。文字布局（`measureText` 结果）按 `(text, font)` 缓存；clip 名超宽时在几何层裁剪字形。dpr 变化重建图集。

**层序与预算**：网格 → 分界线 → clip 块面 → 波形 → clip 细节 → 网格覆盖 → 选区 / ghost / 预览 → 播放头。目标 **< 10 draw call / 帧**。

**降级**：WebGL2 不可用时回退 Canvas2D 全量渲染（保留现有渲染器，仅作 fallback）。

---

## 6. 命中测试与交互 Controller

**核心不变式："看到的 = 可点的"**。绘制与命中必须共用同一套几何常量与换算函数（`CLIP_HEADER_HEIGHT`、`snapOffsetHandleXPx`、`takeLanes`、`fadeCornerReserve` 等）。现在这些几何散在 `ClipItem` 的 DOM 样式里，新内核把它们抽成纯几何模块，绘制与命中同时消费——这是消除"DOM 命中区与 canvas 视觉漂移"的根本手段。

**HitTestIndex**：在现有 `timelineHitTest.ts`（轨道分桶 + clip 时间排序 + 二分）基础上扩展命中区类型：

| 层级 | 命中区 |
|---|---|
| 轨道头 | 名称 / 静音 / 独奏 / 电平表 / 右键区 |
| 标尺 | 刻度 seek / Tempo 标记 / 播放头拖拽 |
| clip | header（名称 / 编辑）/ body / trim 左右 / fade 角 / 旋钮 / 三个 badge / take lane / snap 三角 |

判定按 z 序自上而下短路；几何规则全部为纯函数，直接单测。

**InteractionController**：统一接管 pointer / wheel / key / contextmenu，内部是显式状态机：`idle → press 候选（位移阈值）→ 手势（drag / trim / fade / 框选 / pan / 播放头）→ 提交或取消`。手势期间以世界坐标差值计算（沿用既有 Drag Rules），拖拽到边缘时驱动 `ScrollKernel` 自动滚动；Esc / blur / pointercancel 走统一取消路径（沿用现有 `registerDragAbort` 语义）。

**与 Redux 的边界**：hover、ghost、选区框、trim 预览等高频临时态只存在 runtime 并直接进渲染清单；**手势结束才提交** Redux action / remote thunk；选中变化因需驱动外部 UI，作为低频提交。

**DOM overlay 定位**：rename 输入框、badge 编辑、右键菜单、tooltip 由 runtime 的 `worldToScreen` 计算位置并在 rAF 内更新；菜单锚点仍用 client 坐标。

---

## 7. 迁移路径、验证与风险

### 7.1 阶段划分

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| **0 · Spike** | 独立新目录的最小内核：自绘滚动 + 单 WebGL2（网格 / clip 块面 / 文字）+ 几何命中雏形。**不接入生产路径**，用 PERF 场景驱动 | 真机跑 400/1000 clip：滚动 60fps、手感可接受、文字质量达标 → 决策点（不达标则退回 WebGL2 + Canvas2D 细节层方案） |
| **1 · 内核接入** | 按区域切换：先轨道区（渲染 + 自绘滚动 + 交互 controller），再标尺，再轨道头；`timelineViewportBus` 在此阶段退化为"一次 commit" | 各区视觉与交互与旧版一致，帧率达标 |
| **2 · 清理** | 删除旧路径：`ClipItem` / `TrackLane` / `OverlapEditLayer` DOM 交互、`timelineViewportBus` / `timelineFrameCommitter` 多图层同步、`gridRedrawBridge`、旧 wheel / hit 逻辑 | 无死代码，全量回归通过 |

### 7.2 Feature flag

`hifishifter.timelineKernel`（沿用现有 PERF 开关模式），默认关、PERF 面板可切；稳定后默认开再删旧路径。**可一键回退是这一阶段的硬要求。**

### 7.3 测试策略

- 纯函数单测：命中几何、手势状态机、字形布局、几何构建、缩放锚点 / 钳制
- 不变量测试：① 绘制与命中共用几何函数（同一输入同一结果）；② "滚动帧不重建几何"（spy 断言构建函数零调用）
- 性能验收：1000 clip 场景滚动帧 render < 2ms、稳定 60fps；拖拽 / 框选 ≥ 55fps
- 视觉回归：与现有实现逐像素对比（clip / 网格 / 波形 / 文字）

### 7.4 主要风险与缓解

| 风险 | 缓解 |
|---|---|
| 中文字形图集容量 | 分页 + LRU；极端情况降级为省略号截断 |
| 文字渲染质量 | 按 dpr 渲染 + 物理像素吸附；Spike 阶段真机比对 |
| 滚动手感（触摸板 / 惯性 / 快速滚动） | Spike 阶段重点验证，这是最可能推翻方案的项 |
| 交互语义回归 | 现有行为清单 + 单测 |
| 性能回退 | feature flag 一键回退 |

---

## 8. 非目标

- 参数编辑器（PianoRoll）迁移（后续复用同一渲染内核）
- Redux / 后端协议语义变更
- UI 视觉改版（保持现有视觉逐像素一致）
- 借重构改造无关模块
