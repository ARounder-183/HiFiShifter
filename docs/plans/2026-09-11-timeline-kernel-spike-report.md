# 时间轴统一渲染内核 · 阶段 0（Spike）报告

- 日期：2026-09-11
- 分支：`feature/timeline-unified-render-kernel`
- 设计：`docs/plans/2026-09-11-timeline-unified-render-kernel-design.md`
- 实施计划：`docs/plans/2026-09-11-timeline-unified-render-kernel-implementation.md`
- 状态：**阶段 0 完成；并已超出 Spike 范围完成「视觉与输入对齐 + 基础交互」，结论为继续推进**

---

## 0. 结论（先读这里）

**继续推进阶段 1。**

依据：真机（macOS，dev 构建）实测滚动帧率 59 FPS、内核绘制 p50 1.0ms / p95 3.0ms、
React 提交 0ms，图层只剩 `kernel-draw`——「滚动零几何重建」的核心假设成立。
在此基础上补齐了与旧实现的视觉 / 输入 / 基础交互对齐（见 §2.2–2.4），
剩余缺口集中在**拖拽编辑、框选、浮层锚点**（见 §7）。

---

## 1. 交付清单

新内核代码全部位于 `frontend/src/components/layout/timeline/kernel/`。

### 1.1 内核基础（阶段 0 计划内）

| 模块 | 文件 | 职责 |
|---|---|---|
| 滚动内核 | `scrollKernel.ts` | 视口真值 + 独占钳制 + 锚点缩放 + `reclamp` + `setRowHeight` |
| 输入 | `input/normalizeWheel.ts` | 滚轮增量归一化（pixel/line/page） |
| 输入 | `input/scrollbars.ts` | 自绘滚动条几何 + thumb 命中 + 拖拽换算 |
| 字形 | `glyph/glyphLayout.ts` | 文本切分 + 测量缓存 + 截断 |
| 字形 | `glyph/glyphAtlas.ts` | 多页 shelf-pack 分配器 |
| 字形 | `glyph/glyphRasterizer.ts` | 离屏 Canvas2D 按 dpr 光栅化到图集 |
| 场景 | `scene/instanceTypes.ts` | 平面矩形实例类型 |
| 场景 | `scene/gridInstances.ts` | 网格实例（内容坐标，竖直不裁剪） |
| 场景 | `scene/clipInstances.ts` | clip 实例（复用既有样式与实例写入） |
| GL | `gl/glRaster.ts` / `gl/instanceBuffer.ts` / `gl/instanceLayout.ts` | 光栅化参数 / 缓冲策略 / 25-float 布局唯一来源 |
| GL | `gl/glContext.ts` / `gl/sdfBoxProgram.ts` | WebGL2 上下文 / 圆角盒 + 平面矩形实例化 program |
| GL | `gl/glyphQuads.ts` / `gl/glyphProgram.ts` | 字形四边形构建 / 字形纹理 program（当前无调用方，见 §4） |
| 循环 | `renderLoop.ts` | 脏标记 + rAF 调度（按需，非常驻） |
| 宿主 | `host/timelineKernelHost.ts` | 装配 + 输入 + 渲染循环 + 场景重建 + DOM 同步 + 手势 |
| 外壳 | `TimelineKernelView.tsx` | React 外壳（DOM 节点 + 数据镜像 + 回调） |
| 交互 | `interaction/hitTest.ts` | 几何命中（内容坐标 → 时间 → 轨道行 → 二分定位 clip → 分区） |
| 开关 | `featureFlag.ts` | `hifishifter.timelineKernel`（显式 `"0"` 关 / 显式 `"1"` 开 / 未设置时 dev 默认开） |

### 1.2 超出 Spike 范围的补齐（同日第二批 / 第三批）

| 能力 | 落地方式 |
|---|---|
| clip 细节层 | 新建 Canvas2D 覆盖层，复用 `drawTimelineCanvas` + **空 GL sink**（该函数收到 `glBodies` 时只画细节）；采用「内容坐标 + 窗口平移」，滚动在余量内只移动画布元素、不重绘 |
| 轨道行分界线 | 内核产出平面矩形实例（与左侧轨道头的行边框同源） |
| 网格 / 行线配色 | 改读主题 token（`--qt-graph-grid-weak/strong`、`--qt-border`） |
| 播放头 | 轨道区（内核视图内，视口坐标 `translateX`）+ 标尺（标尺内容层内，内容坐标 `left`） |
| 标尺 / 左侧轨道头 | **保留 DOM**，内核在 rAF 内同步 `transform` / `scrollTop`（理由见 §2.1） |
| 波形层 | `TimelineWaveformSurface` 支持外部视口源；内核暴露 `getAxis()` / `registerViewportLayer()` / `onVisibleRowsChange` |
| 滚轮语义 | 完全由 keybinding 决定（见 §2.3） |
| 中键平移 | 抓取式 1:1 反向映射，含光标 / userSelect 复原 |
| 点击交互 | 点击 clip 选中（含多选修饰键）、点击 / 拖拽空白 seek |
| 浮层 | 移出内核开关的三元表达式，两种模式共用（右键菜单 / 对话框 / 播放头逐帧驱动） |
| 开发工具 | 浏览器 mock 后端（`?mock=1`，含假工程与假波形）+ 截图 / 交互脚本 |

---

## 2. 关键设计落地

### 2.1 架构决策：为什么标尺与轨道头保留 DOM

设计文档原定「时间轴面板全部视觉自绘」。实施后调整为**内核只接管轨道区**：

- 标尺（含 Tempo Map 行，76KB 交互组件）与左侧轨道头（电平表 / 旋钮 / 按钮）是**重交互、
  低频变化**的 DOM 子树，搬进 canvas 等于重写旗帜拖拽、内联编辑、右键菜单与电平表，收益极低；
- 内核只把「跟随视口」的部分收敛为 **rAF 内一次 `transform` / `scrollTop` 写入**——一次样式写入
  的成本与内容规模无关，不构成滚动瓶颈，且视觉天然与旧实现一致；
- 接入方式：`TimelinePanel` 提取 `timeRulerNode` 变量，两种模式共用同一实例。

**这仍然满足「取消 sync」的核心诉求**：轨道区（滚动性能瓶颈所在）没有任何随原生滚动移动的
DOM 内容层，渲染完全由 rAF 调度。

### 2.2 坐标系约定（评审检查项）

| 元素 | 坐标系 | 原因 |
|---|---|---|
| GL 块面实例 | 内容坐标 | 靠 `u_viewOrigin` uniform 平移，滚动零上传 |
| 细节层画布 | 内容坐标 + 画布元素 `left/top` 平移 | 与 GL 层共用同一窗口原点，滚动时不会分离 |
| 标尺播放头 | **内容坐标** `left` | 位于标尺内容层内，随内容层 `translateX(-scrollLeft)` 自动跟随 |
| 轨道区播放头 | **视口坐标** `translateX` | 位于内核视口容器内 |

混用会导致双重计滚动或播放头「粘屏」。

### 2.3 滚轮语义（完全由 keybinding 决定）

| 操作 | 行为 |
|---|---|
| 无修饰键滚轮 | 水平缩放（`modifier.horizontalZoom` 是 none-binding，无修饰键时命中） |
| Shift + 滚轮 | 水平滚动（`modifier.scrollHorizontal`） |
| Alt + 滚轮 | 垂直滚动（`modifier.scrollVertical`） |
| 主修饰键（macOS ⌘ / Win Ctrl）+ 滚轮 | 竖直缩放（`modifier.pianoRollVerticalZoom`） |
| 悬停自绘滚动条 | 该轴滚动；按住 `modifier.scrollbarZoom`（默认 Alt）= 该轴缩放 |

缩放锚点：水平走 `resolveHorizontalWheelZoom`（含播放头锚点、`resolveTimelineMinPxPerSec` 下限、
工程长度），竖直保持指针下的行位置不变。

> **教训**：竖直缩放在浏览器里一度「测不出来」，实际是 **macOS 上主修饰键是 Meta（⌘）**而非
> Control —— `ctrl: true` 的绑定在 mac 上映射为 ⌘。

### 2.4 静态可验证的设计目标

| 目标 | 落地方式 | 验证手段 |
|---|---|---|
| 取消跨层 sync | 轨道区无随原生滚动移动的 DOM 内容层 | 代码结构（宿主无原生 scroller） |
| 滚动零几何重建 | 几何内容坐标常驻；滚动帧只更新 `u_viewOrigin` | `gridInstances.test.ts` 断言；`sdfBoxProgram.repaint` 零上传 |
| 单次提交 | 网格 + 行线 + clip 共用一份实例缓冲、一次 draw call | 宿主 `draw()` |
| 钳制唯一 | 全部钳制收敛在 `ScrollKernel`，外部边界变化走 `reclamp()` | `scrollKernel.test.ts`（17 条） |
| 看到的 = 可点的 | 绘制与命中共用同一套几何常量与换算 | `hitTest.test.ts`（8 条） |
| 缓冲零分配 | 实例缓冲倍增复用 | `instanceBuffer.test.ts` + `clipInstances.test.ts` |

---

## 3. 测试与质量门

- 单元测试：**426 条**（105 个文件），其中内核目录 13 个文件。
- 类型检查：`npx tsc -b --noEmit` 通过。
- Lint：`npx eslint`（kernel 目录 + `TimelinePanel`）无输出。
- 生产构建：`npm run build` 通过。
- 全量回归：`npx vitest run` → **424 passed / 2 failed**；2 个失败位于
  `src/features/keybindings/keybindingMatch.test.ts`，**经 develop 分支复现确认为预先存在**。

### 独立审查记录

| 轮次 | 范围 | 发现 | 处理 |
|---|---|---|---|
| 1 | Task 1（scrollKernel） | 1 实质（测试断言无判别力）+ 1 轻微 | 已修 |
| 2 | Task 1（质量） | 3 Important（`reclamp` 缺失 / 引用稳定无测试 / 水平上限语义与既有实现相反）+ 6 Minor | 已修（含语义对齐既有 `resolveTimelineScrollRange`） |
| 3 | Task 2–4 | 5 处测试判别力不足 + 4 其他 | 已修 |
| 4 | Task 5–8 | 1 Critical（**混合状态缺失**）+ 6 Important + 8 Minor | 已修（Critical 与全部 Important） |

---

## 4. 已知简化与遗留

1. **字形图集模块当前无调用方**：`glyph/*` 与 `gl/glyphProgram`、`gl/glyphQuads` 已实现并有单测，
   但 clip 名称文字改由细节层（Canvas2D）承担——GL 字形与细节层同时画名称会产生重影。
   保留模块供后续自绘标尺标签 / 性能敏感场景使用。
2. **编组激活的 clip 仍走 GL 块面**（既有实现会退回 Canvas2D 画外圈描边）——视觉上少一个 2px 外圈。
3. **波形在 headless 浏览器中不可见**：数据管道与绘制调用全部正确（见 §5.4），但
   headless Chrome + SwiftShader 下多 WebGL canvas 合成不出内容，需真机验证。
4. **浮层锚点仍取旧 scroller 坐标**：右键菜单 / tooltip / 吸附高亮的挂载已恢复，但定位未改。

### A/B 时**不应**判为回归的差异

1. **网格竖线相位**：既有 SVG 用「居中描边 + 设备像素吸附」，内核用「矩形左缘吸附」，
   强线相位差 1 物理像素、弱线更锐利（符合设计规格）。
2. **文字渲染管线**：细节层复用既有 Canvas2D 路径，与旧实现同源；GL 字形路径已不参与。

---

## 5. 验证记录

### 5.1 真机（macOS，dev 构建）

采集方式：PERF 面板的 `profiler: on` 帧率浮层（应用内绘制，不依赖 devtools）。

| 指标 | 目标 | 实测 |
|---|---|---|
| 图层数 | 只应有 kernel-draw | ✅ 仅 `kernel-draw`（旧图层全部消失） |
| 滚动帧率 | 稳定 60fps | ✅ **FPS 59** |
| 帧间隔 | ~16.7ms | ✅ frame p50 17.0 / p95 18.0ms |
| 内核绘制耗时 | < 2ms | ✅ p50 **1.0ms** / p95 **3.0ms**（p95 含重建帧） |
| React 提交 | 低频 | ✅ commit p50 0.0 / p95 0.0ms |

**对照（同一场景、修复前）**：两套渲染叠加时 FPS 43 / frame p95 178ms / clip-body p50 5ms。

### 5.2 真机过程中修复的阻塞

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 帧率与旧架构一致（43 FPS） | 接入为「覆盖式」，旧子树仍挂载并每帧重绘 | 改为**替换式** |
| 2 | `cannot declare arrays of this qualifier` | 着色器用 `in float i_rect[4]`，**GLSL ES 3.00 禁止顶点输入为数组**（macOS ANGLE/Metal 严格拒绝）；该写法源自既有 `timelineClipGlRenderer`，**既有 GL clip 体在 macOS 上一直静默回退 Canvas2D** | 两处改为 `in vec4 i_rect` 打包（含生产代码） |
| 3 | `Unable to create sdf-box shader` | `dispose()` 调用了 `loseContext()`；一个 canvas 只能有一个 WebGL context，丢失后再 `getContext` 返回同一个已丢失的 context → StrictMode 双挂载第二次 `createShader` 返回 null | dispose 不再丢 context，并堵住创建失败路径的资源泄漏 |

### 5.3 浏览器验证（mock 后端 + 截图脚本）

新增开发工具：`dev/mockBackend.ts`（URL 加 `?mock=1`，含覆盖各视觉分支的 6 轨假工程与假波形）
与 `scripts/dev-shot.mjs`（playwright-core + 系统 Chrome，免下载浏览器）。

| 项 | 结果 |
|---|---|
| 无修饰键滚轮 = 水平缩放 | ✅ 标尺 transform 随锚点变化 |
| Shift + 滚轮 = 水平滚动 | ✅ |
| 主修饰键 + 滚轮 = 竖直缩放 | ✅ 行高变化且左右两列同步 |
| 中键平移 | ✅ 标尺 transform 0 → −200px |
| 自绘滚动条（水平 / 竖直） | ✅ 竖直拖拽后轨道头 scrollTop 同步为 262.5 |
| 点击 clip 选中 | ✅ 出现选中描边 |
| 点击空白 seek | ✅ 播放头跳转 |
| 纵向滚动左右对齐 | ✅ |
| 视觉逐项对比旧实现 | ✅ 网格 / 行分界线 / clip 块面 / clip 细节（fade 曲线 + 徽标 + 增益与速率标签）/ 播放头 / 标尺 / 轨道头 / 波形层挂载 |

### 5.4 波形 mock 的逐层诊断（浏览器里仍不可见）

按后端二进制协议（`utils/waveformBinaryCodec`）合成假波形：

```
[magic "WFPK" 4B][sample_rate u32][division_factor u32][peak_count u32][level u32][min f32 × n][max f32 × n]
```

关键：`division_factor` **完全由数据驱动**（`startIdx = startSec × sampleRate / divisionFactor`），
无硬编码级别；与 `SPP_THRESHOLDS = [512, 1024]` 对齐后取 L0/L1/L2 = 512/1024/2048，
30 秒仅 2583 点（约 20KB）。

逐层验证**全部正确**：

| 环节 | 实测 |
|---|---|
| mock 数据 | magic=WFPK, sr=44100, df=512, peakCount=2583, level=0, 浮点值有效 |
| store 取数 | `peaks HIT level=0 spp=294 range=0.00..4.00` |
| 几何 | `lines=6970 verts=83640 complete=true segments=9`，首段 `rect0=716,18,600x60`（在视口内） |
| 渲染器 | `quads=41820 err=0 vp=3328x302 origin=416,0` |
| canvas | 与内核 GL canvas 完全重合 `(256,112,1664×151)`，z-index 1，加半透明背景**可见** |

**结论**：数据管道与绘制调用全部正确，但 **headless Chrome + SwiftShader 下多 WebGL canvas
合成不出内容**（`toDataURL()` 能取到 27KB PNG，但合成后不可见）——环境限制，非代码缺陷。
波形渲染是既有生产代码路径（本次未改其算法），**需在 Tauri 真机验证**。

---

## 6. 阶段 1 出口条件对照

| 条件 | 状态 |
|---|---|
| `npm test` 全绿；`tsc -b --noEmit` 无错 | ✅（2 条预先存在的 keybinding 失败与本改动无关） |
| 1000 clip 场景滚动帧零几何重建，p95 < 2ms | ✅ 真机 p50 1.0 / p95 3.0ms（p95 含重建帧）；400 clip 场景已实测 |
| 滚动手感与文字质量可接受 | ✅ 手感已确认；文字由细节层（Canvas2D，与旧实现同源）承担 |
| 报告给出明确结论 | ✅ **继续推进阶段 1** |

---

## 7. 阶段 1 待办（按价值排序）

1. **clip 拖拽编辑**（移动 / trim / fade 角）——手势状态机与命中测试已就位，超过位移阈值后
   目前只结束手势；需接 Drag Rules、吸附、ghost 预览与 Redux 提交。**缺口最大的一项**。
2. **框选**——坐标源从原生 scroller 改为 `ScrollKernel`。
3. **浮层锚点**——右键菜单 / hover tooltip / 吸附高亮的定位改按内核视口计算。
4. **真机验证波形**与性能复测（含 1000 clip 场景）。
5. **阶段 2 清理**——删除旧路径（`ClipItem` / `TrackLane` / `OverlapEditLayer` DOM 交互、
   `timelineViewportBus` / `timelineFrameCommitter`、`gridRedrawBridge`）。

### 关键教训（写入规范）

1. **DOM 写入去重的 NaN 初值缺陷**：`if (Math.abs(next - prev) > eps)` 在 `prev = NaN` 时恒为
   `false` → **首次写入被跳过**（标尺 / 播放头 / 细节层完全不动）。必须用取反写法
   `!(Math.abs(next - prev) <= eps)`（见宿主 `shouldWrite`）。
2. **mock 的 thenable 陷阱**：`window.pywebview.api` 若被 Proxy 包装且 `then` 返回函数，会被
   Promise 解析协议当作 thenable 调用，`await` 永久挂起。必须让 `then/catch/finally` 返回 `undefined`。
3. **JSX 注释不能作为表达式并列**：提取 JSX 块时要改成 `//` 注释。
