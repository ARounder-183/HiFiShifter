# 时间线渲染架构重构设计（方案 B：统一 GL 场景 + 视口变换）

- 日期：2026-09-01
- 范围：`frontend/src/waveform/*`、`frontend/src/components/layout/timeline/**`、`frontend/src/components/layout/TimelinePanel.tsx`、`frontend/src/components/layout/timeline/hooks/useTimelineState.ts`
- 目标：让**平移与缩放**在「10 轨 / 400 clip / 全览缩放」下都稳定 60 fps，且每帧成本与 clip 数解耦
- 已确认决策：采用**方案 B**（统一 GL 场景 + 视口 uniform + LOD 档位缓存）；分 P0→P6 落地，每阶段独立提交、独立回退
- 前置文档：`docs/plans/2026-08-31-timeline-waveform-perf-design.md`（其阶段 3 的离线基准已完成；其「R4 per-clip 开销」假设已被实测推翻，见 §2.2）

---

## 1. 目标与非目标

### 目标

1. **平移帧**：一次 uniform 更新 + 1~3 次 draw call，成本与 clip 数、波形量无关。
2. **缩放帧**：LOD 档位内同样退化为 uniform 更新；跨档才重建几何，且一次手势内触发次数有界。
3. **React 零重渲染**：平移/缩放期间不触发任何 React 提交，删除 `flushSync`。
4. 保留可回退的离线基准与真机验收手段。

### 非目标

- 不改变时间线的视觉设计语言（配色、圆角、淡变曲线形状、loop 标记、网格与标尺样式）。
- 不改动 mipmap 的级别划分与选级阈值（`DIV_FACTORS` / `SPP_THRESHOLDS`）。
- 不把标尺迁入 GL（它需要文本输入与右键交互）。
- 不引入 in-app 性能探针（沿用离线基准 + Chrome DevTools 验证）。

---

## 2. 现状分析

### 2.1 数据流

滚动/缩放事件 → 两条并行路径，**各画一遍**：

```
滚轮/滚动
  ├─(A) 总线同步路径
  │    syncScrollLeft()                        useTimelineState.ts:440
  │      └─ timelineViewportBus.emit(...)
  │           └─ frameCommitter.commit(axis)   timelineFrameCommitter.ts:99
  │                ├─ clip-body   paint        TimelineCanvasViewport.tsx:99
  │                ├─ waveform    paint        WaveformSurface.tsx:243
  │                └─ gridOverlay paint        BackgroundGrid
  └─(B) React 路径（同一帧内再次触发）
       setScrollLeft(...)  rAF 合并            useTimelineState.ts:476
       / flushSync(setPxPerSec, ...)            TimelineScrollArea.tsx:307
         └─ TimelinePanel 整树重渲染
              ├─ buildTimelineRenderModel      TimelinePanel.tsx:1271
              ├─ buildSparseClipRenderModel    TimelinePanel.tsx:1372
              └─ TimelineWaveformSurface.rows  TimelineWaveformSurface.tsx:46
                   └─ layout effect 再画一遍   WaveformSurface.tsx:169 / TimelineCanvasViewport.tsx:92
```

### 2.2 实测基线（`npm run bench`，Node/V8，视口 1500 px，10 轨）

| 阶段 | 400 clip / fitContent 0.625 px·s⁻¹ | 400 clip / zoomFloor 4 px·s⁻¹ | 400 clip / 40 px·s⁻¹ | 40 clip / fitContent |
|---|---|---|---|---|
| renderModel（React 侧） | 0.005 ms | 0.004 ms | 0.004 ms | 0.001 ms |
| scene | 0.036 ms | 0.008 ms | 0.002 ms | 0.004 ms |
| geometry | 0.52 ms | 0.30 ms | 0.27 ms | 0.31 ms |
| quads | 0.70 ms | 0.72 ms | 0.73 ms | 0.70 ms |
| **full frame** | **1.24 ms** | 1.04 ms | 0.99 ms | 1.02 ms |
| 可见 segments | 400 | 70 | 10 | 40 |
| pixelColumns | 14,993 | 14,991 | 14,946 | 14,983 |
| vertexBytes | 703 KB | 703 KB | 701 KB | 702 KB |

**三个结论：**

1. **波形全帧仅 ~1.2 ms，不是主瓶颈。** 40 clip 与 400 clip 的 full frame 几乎相同（1.02 vs 1.24 ms）——`pixelColumns` 恒等于「视口宽 × 轨道数」，与 clip 数无关。2026-08-31 设计中的 R4（per-clip 固定开销）假设**不成立**，其优先级应下调。
2. **最大单项是 clip 体画布。** `drawTimelineCanvas`（`timelineCanvasRenderer.ts:266-619`）对每个 clip 依次执行 `roundRect + clip()` + 3~5 次 `fillRect` + 2 次 `stroke` + 旋钮 `arc`×2 + 3 次 `measureText` + `fillText`，外加淡变曲线的自适应细分（单条上限 1200 点）。400 clip 粗估 5~20 ms/帧。
3. **两张画布每帧各画两遍**（路径 A + 路径 B），可直接减半。

### 2.2b clip 体链路实测基线（`timelineClipPerf.bench.ts`）

| 阶段 | 400 clip / fitContent | 400 clip / 4 px·s⁻¹ | 40 clip / fitContent |
|---|---|---|---|
| renderModel（全 clip 分组 + 窗口裁剪） | 0.005 ms | 0.005 ms | 0.001 ms |
| sparseClipRenderModel（几何投影 + 重叠检测） | 0.040 ms | 0.039 ms | 0.005 ms |
| clipVisualStyle × 3 宽度档 × 全 clip | 0.49 ms | 0.49 ms | 0.050 ms |
| **纯计算合计** | **0.54 ms** | **0.53 ms** | **0.057 ms** |

**这条基线推翻了初期的代码估算（原以为 clip 体是 5~20 ms）。说明：**

1. clip 体的**纯计算**与波形（0.76 ms）同量级，不是主要矛盾；
2. 真正的开销在**基准测不到的 Canvas2D 调用**上——每个 clip 一轮
   `roundRect + clip()` 遮罩、约两次 `stroke`、数次 `fillRect`，400 clip
   合计约 2400 次操作，按 Chrome 单次成本粗估约 10 ms；
3. 因此 clip 体的优化方向是「**降 canvas 操作数**」（合批 / 去遮罩），
   而不是继续抠模型构建。

注意该基准是**下界**：`drawTimelineCanvas` 的实际绘制与浏览器侧
`measureText` 都不在覆盖范围内。

### 2.3 缩放帧的额外成本

`TimelineScrollArea.tsx:307` 的 `flushSync` 强制同步全树渲染，其存在理由见 `:139-146` 注释——内容宽度必须先按新 `pxPerSec` 重排，否则写 `scrollLeft` 会被浏览器按旧宽度钳制。代价是：

- 一帧内两次全树 React 渲染（`flushSync` 一次 + `syncScrollLeft` 的 rAF 一次）；
- `contentWidth` 变化 → 滚动内容区重排重绘；
- 全部派生缓存重建：`buildTimelineRenderModel`（**遍历全部 clip** 建分组 Map）、`buildSparseClipRenderModel`（400 对象 + `timelineCanvasModel.ts:174-189` 的 **O(n²) 重叠检测**，单轨 400 clip 即 8 万对）、`TimelineWaveformSurface.rows`（10 行 take 展开 + 每行 `computeLeadingOverlapSecByClipId([...clips])` 拷贝排序）。

### 2.4 一个关键的既有事实

```141:144:frontend/src/components/layout/timeline/runtime/timelineAxis.ts
export function secToContentPx(axis: TimelineAxis, sec: number): number {
    if (!Number.isFinite(sec)) return 0;
    return sec * axis.pxPerSec;
}
```

clip 体模型的所有字段（`leftPx` / `topPx` / `widthPx` / `fadeInPx` / `fadeOutPx` / `snapOffsetPx` / `leadingOverlapPx`）都经 `secToContentPx` / `durationToWidthPx` / `secToSpanPx` 投影，**不含 `scrollLeftPx` / `scrollTopPx`**。即 clip 体画布的内容在纯平移下逐像素完全相同——复用是**语义精确**的，不是近似的。这是整个方案 B 的立足点。

---

## 3. 目标架构

### 3.1 三层分工

| 层 | 技术 | 内容 | 重绘时机 |
|---|---|---|---|
| **GL 场景层** | 单个 WebGL2 上下文 | 波形、clip 体（SDF）、淡变曲线、标记、网格 | 仅 LOD 跨档 / 余量耗尽 / 数据变更 |
| **文字覆盖层** | Canvas2D（独立画布） | clip 名称、增益/变速标签、徽标字母 | 视口停止变化 ~100 ms 后 |
| **DOM 交互层** | React | 原生滚动容器、选中手柄、重命名输入框、右键菜单、标尺 | 数据/交互变更 |

### 3.2 统一内容坐标系 + 视口 uniform

所有几何在**内容坐标**里构建并常驻 GPU，视口变换只更新 uniform：

```glsl
uniform vec2 u_viewOrigin;   // 视口左上角的内容坐标
uniform vec2 u_viewScale;    // 档内缩放残差（恒为 1，跨档过渡期除外）
vec2 screen = (a_contentPos - u_viewOrigin) * u_viewScale;
```

- **平移** → 改 `u_viewOrigin`，零重建。
- **缩放** → 档内改 `u_viewScale`（瞬时响应），跨档重建几何。

对波形而言这是**简化**：`sceneBuilder` 从 `secToViewportPx` 改为 `secToContentPx`，不再耦合 `scrollLeft`，2026-08-31 设计里那套「余量 + 锚点 + 失效判定」整个不需要了。

### 3.3 LOD 档位缓存

```ts
level = Math.floor(Math.log2(pxPerSec / BASE_PX_PER_SEC));
```

- 几何按 **level** 缓存，而非按精确 `pxPerSec`；
- 档内缩放/平移 = 纯 uniform，**零重建**；
- 跨档才重建几何 —— 一次手势内触发次数 = 跨档次数（有界，通常 1~3 次）；
- 档位过渡的视觉误差 = 采样密度差（≤ 2×），波形采样密度的视觉容差本来就大；
- **手势结束后按精确 `pxPerSec` 补一次全质量重建**，消除稳态误差；
- 空闲时**预取相邻档位**（level ± 1），抹掉跨档尖峰。

### 3.4 几何窗口 + 余量

不能构建整条工程（2400 s × 100 px·s⁻¹ = 24 万列）。只构建「视口 + 余量」的内容窗口：

- 余量 `marginPx = clamp(round(viewportWidthPx * 0.35), 256, 768)`；
- 400 clip / fitContent 下窗口 ≈ 2500 列 × 10 行 ≈ 25,000 列 ≈ 1.2 MB 顶点，可接受；
- 平移超出余量 → 重建窗口一次（~1.7 ms），而非每帧；
- 跨档重建时以视口中心为局部原点，规避长工程 + 高缩放下的 float32 精度问题（内容坐标可达 10⁵ 量级，float32 尾数 24 位仍有 ~0.01 px 精度，局部原点更稳）。

### 3.5 波形顶点格式

把 `expandLineSegmentsToQuads`（0.70 ms + 2.16 MB/帧上传，`surfaceRenderer.ts:200-202`）搬到 GPU：

- 改为**实例化绘制**：per-instance 传 `(x1, y1, x2, y2, r, g, b, a)` = 8 float；
- 顶点着色器用单位四边形 + `a_side` 沿法线偏移 half-width 展开；
- 实际所有段都是**轴对齐**的（`geometry.ts:215` 逐像素竖列、`:238` 标记横线），法线退化为常量偏移，实现很轻；
- 上传量：25,000 列 × 32 B = **800 KB**（对比现状 2.16 MB），且只在跨档/余量耗尽时上传。

配套去掉 `surfaceRenderer.ts:55-62` 的 `corners` 数组字面量（每帧约 10.6 万次分配）与 `geometry.ts:247` 的 `vertexScratch.slice()`（每帧 703 KB 拷贝，改为调用方提供的缓冲池）。

---

## 4. clip 体的 GL 表达

### 4.1 主体：SDF 圆角盒实例化

per-instance：`(x, y, w, h, radius, colorRGBA, headerColorRGBA, flags)`。

片元着色器内一个圆角盒 SDF 完成：body 填充、header 带、前导重叠区半透、未选中收边、选中提亮、编组金边。400 实例 ≈ 十几 KB 上传，跨档重建只需填一次数组。

### 4.2 细节元素：按尺寸 LOD

淡变曲线、snap 三角、增益旋钮、链/静音/共振峰徽标**只在 clip 超过宽度阈值时绘制**。可绘制数量被视口钳住（≈ 视口宽 / 阈值 × 行数），每帧重建也无所谓，走独立小批次。

- 淡变曲线：按 `(shape, dir, mode, width档, height档)` 缓存归一化折线点集，绘制时 `translate/scale` 复用，去掉 `drawFadeCurveStroke` 每帧最多 1200 点的自适应细分（`timelineCanvasRenderer.ts:39-133`）。

### 4.3 文字：不进 GL

clip 名称 / 增益 / 变速 / 徽标字母保留 **Canvas2D 覆盖层**：

- 只在「视口停止变化 ~100 ms」或「档位切换完成」时重绘；
- 手势中用 CSS `transform` 跟随（与 §3.2 同一套位移量）；
- 由此**规避整个动态 CJK 字形图集子项目**。

### 4.4 顺带修掉的每帧固定开销

- `resolveFontFamily()`（`timelineCanvasStyle.ts:83`）与 `--qt-border`（`timelineCanvasRenderer.ts:213-216`）的 `getComputedStyle` → 主题切换时读一次并缓存；
- `measureTextWidth(CHAR_SAMPLE, ...)`（`timelineCanvasStyle.ts:501`）目前**每个 clip 每帧测一次** → 模块级缓存（只依赖字体）；
- `frameCommitter.commit()` 的 `slice().sort()`（`timelineFrameCommitter.ts:104-106`）→ 有序插入。

---

## 5. React 脱钩

`pxPerSec` / `scrollLeft` / `scrollTop` **不再驱动任何 React state**：

- 视口变更只经 `timelineViewportBus` → GL 图层（已有的每帧对账机制 `useTimelineState.ts:516-542` 保留）；
- 保留的 React state 只有**窗口化**相关（轨道窗口、可见 clip 集合），改为**阈值触发**（跨行 / 跨 1/4 屏才更新）；
- **`flushSync` 彻底删除** —— 它只为协调视觉层而存在，视觉层不再经过 React；
- `timelineAxis`（`TimelinePanel.tsx:1361`）的 useMemo 依赖从 `[pxPerSec, scrollLeft, scrollTop, viewportWidth]` 缩为 `[pxPerSec, viewportWidth]`，让滚动帧不再产生新 axis 对象（这是所有下游 memo 能生效的前提）。

---

## 6. 分阶段落地与当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P0** | 真机 Chrome Performance 量化各层帧耗时 | **跳过**——当前环境无法录制火焰图。改为靠两条离线基准（`waveformPerf.bench.ts` / `timelineClipPerf.bench.ts`）+ 结构回归测试推进；代价是 React 侧与 Canvas2D 绘制的实际占比仍未知 |
| **P1** | ① 消除重复绘制 ② 内容轴稳定 `drawClips` 引用 ③ O(n²) 改排序扫描 ④ 去每帧 `getComputedStyle` 与逐 clip `measureText` | **已完成**（`8ff3ea1b`） |
| **P2a** | 波形顶点缓冲池 + quad 展开去分配 | **已完成**（`9b52126a`）波形 full frame 1.24 → **0.76 ms** |
| **P2c** | 波形几何改窗口局部坐标 + `u_viewOrigin` uniform + `repaint()` | **已完成**（`c6c83cb9`）**平移帧 ≈ 0** |
| **P2b** | 波形实例化绘制（quad 展开上 GPU） | **推迟**——P2a 后 quads 仅 0.25 ms，收益有限；若后续波形仍需压榨再做 |
| **P2d** | 波形 LOD 档位缓存（让**缩放**也 ≈ 0） | **待做**——缩放帧当前 0.76 ms，性价比低于 clip 体，故排在其后 |
| **P2e** | clip 体合批 + 去圆角遮罩 | **已完成**（`ac303963`）400 次 `clip()` → 0；合批调用数与 clip 数解耦 |
| **P3** | GL clip 体渲染器（SDF 实例化），与波形并入同一上下文 | 待评估——P2e 之后需重新测，若 clip 体已足够快则可不做 |
| **P4** | 文字覆盖层（Canvas2D，手势期 CSS transform 跟随） | 待评估 |
| **P5** | React 脱钩：视口出 state，删 `flushSync` | 待评估——**很可能是下一个大头**（TimelinePanel 2100 行整树每帧重渲染，完全未测） |
| **P6** | 网格并入 GL；标尺保持 DOM | 待评估 |

**P1 是先决条件**：不消除重复绘制、不 stabilize `drawClips` 引用，后续所有缓存每帧都会被判失效。

**优先级已被实测调整过两次**：

1. 波形 per-clip 开销（原 R4 假设）不成立 → 波形优化降级；
2. clip 体纯计算仅 0.54 ms → 优化目标从「模型构建」改为「canvas 操作数」。

**当前最不确定的一项**：React 侧每帧整树重渲染 + 浏览器 layout/paint 的
占比完全未测。若 P2e 落地后缩放/平移仍卡，下一刀应落在 **P5（React 脱钩）**
而不是继续优化绘制。

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **WebGL 不可用** | 回退面变大 | P3 之后评估回退范围；至少保留波形层的 Canvas2D 回退（现有 `Canvas2dWaveformRenderer` 已实现） |
| **亚像素对齐** | 1px 描边/文字发虚，与 DOM 层错位 | 几何构建沿用现有 `snapPx`（`timelineAxis.ts:223`）；GL uniform 按设备像素对齐；验收含 dpr=1 / 1.25 / 1.5 / 2 四档 |
| **float32 精度** | 长工程 + 高缩放下坐标抖动 | 跨档重建以视口中心为局部原点（§3.4） |
| **预取策略抖动** | 空闲预取抢占主线程 | 预取走 `requestIdleCallback`，且可被任何交互打断 |
| **P5 依赖面广** | 哪些 UI 真正需要 state 驱动需逐个梳理 | 放在 P3/P4 之后，此时视觉层已不依赖 React，梳理面大幅收窄 |
| **跨档重建尖峰** | 手势中偶发掉帧 | 空闲预取相邻档位 + 余量窗口（§3.3 / §3.4） |

---

## 8. 验收标准（真机，10 轨 / 400 clip / 全览）

- **水平平移**：稳定 60 fps，Performance 面板中主线程只有 uniform 更新与 draw call，无周期性 GC。
- **缩放**：稳定 60 fps；跨档重建帧耗时 < 2 ms 且一次手势内 ≤ 3 次。
- **React**：平移/缩放全程 Profiler 提交次数为 0。
- **视觉**：与重构前逐像素比对一致（dpr 四档）；LOD 过渡与手势期末的全质量重建无可见跳变。
- **回归**：`npm test`、`npm run lint`、`npm run bench`（三阶段耗时不劣化）。
