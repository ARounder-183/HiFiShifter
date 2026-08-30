# 时间线统一坐标重构设计（Timeline Unified Axis）

- 日期：2026-08-29
- 分支：`feature/timeline-unified-axis`（自 `develop` 拉出）
- 范围：时间线（TimelinePanel）+ 参数编辑器（PianoRollPanel）
- 目标：让**网格、标尺、clip、波形、播放头**在同一缩放/滚动状态下，像素位置与缩放比例严格一致

---

## 1. 目标与非目标

### 目标

1. 全图层共享**唯一**的时间↔像素投影，任何图层不得自行执行 `sec * pxPerSec`。
2. 画布（Canvas2D / WebGL2）共享**唯一**的 DPR 光栅化契约，物理像素缩放比严格等于 `dpr`。
3. 一次视口变更（滚动/缩放）在**同一帧**内按固定顺序提交所有图层，无 rAF 延迟。
4. 网格线与标尺刻度由**同一份 tick 数据**驱动，不存在两套生成逻辑。
5. 上述不变量由属性测试守护，并有 dev 期可视化自检面板可肉眼验收。

### 非目标

- 不改变任何交互行为（拖拽、trim、吸附、缩放锚点手感）。
- 不改变视觉风格（配色、线宽、字号）。本次只修"错位"，不顺带改设计。
- 不重构波形 mipmap 缓存与 WebGL 顶点生成算法（性能足够，另案处理）。
- 不引入 feature flag 开关新旧路径（靠阶段化 commit 回退，见第 6 节）。

---

## 2. 现状分析

### 2.1 时间线侧：五层五套换算

| 图层 | 换算表达式 | 坐标系 | 滚动补偿 |
|---|---|---|---|
| 网格 | `x = index * stepPx - offset`（`BackgroundGrid.tsx:218`）；Tempo Map 路径 `x = lineXs[i] - offset`（`:243`） | 内容坐标 | sticky 层手减 `offset = scrollLeft`（`:162`） |
| 标尺 | `left = tick.sec * pxPerSec`（`TimeRuler.tsx:111`） | 内容坐标 | 外层 `translateX(-scrollLeft)` |
| Clip 体画布 | `leftPx = clip.startSec * args.pxPerSec`（`timelineCanvasModel.ts:143`） | 内容坐标 | `ctx.translate(-scrollLeft, -scrollTopPx)` |
| 波形 | `x = (clip.startSec + local - viewportStartSec) * pxPerSec`（`sceneBuilder.ts:312`），其中 `viewportStartSec = scrollLeft / pxPerSec`（`WaveformSurface.tsx:78`） | **屏幕坐标** | 先除后乘 |
| 播放头 | `sec * pxPerSec - scrollLeft`（`TimelineSurface.tsx:136`） | 屏幕坐标 | 直接减 |

### 2.2 Piano Roll 侧：三套公式并存

| 对象 | 表达式 | 位置 |
|---|---|---|
| 网格 / 标尺 | 与时间线共用 `BackgroundGrid`、`TimeRuler`、`buildRulerTicks` | `PianoRollPanel.tsx:5388-5407` |
| 曲线（pitch/param） | `x = (tSec - visibleStartSec) / visibleDurSec * w` | `pianoRoll/render.ts:157` → `pianoRoll/utils.ts:44-52` |
| 选区 / 剪贴板预览 | `x = beat * pxPerBeat - scrollLeft` | `render.ts:1150-1151`、`:1352-1353`、`:1395-1396` |
| 播放头（canvas） | `phx = playheadSec * pxPerSec - scrollLeft`，画在 `phx + 0.5` | `render.ts:1457` |
| 波形 | 与时间线共用 `WaveformSurface`，`viewportStartSec = scrollLeftPx / pxPerSec` | `PianoRollWaveformSurface.tsx:57-66` |

> Piano Roll 无音符块渲染；MIDI 音高线在时间线侧 `components/waveform/MidiPitchTrackCanvas.tsx`。

### 2.3 五类根因

**R1 — 数学路径不等价。** 波形走"先除后乘" `(t − s/p)·p`，其余走"先乘后减" `t·p − s`。二者 IEEE754 差异约 1e-13，本身不可见，但叠加 clamp、取整与 R2 后被放大。

**R2 — DPR 光栅化契约不一致（真正的缩放偏差）。**

```ts
// Clip 画布：严格 dpr
const displayWidth  = Math.max(1, Math.ceil(widthRef.current));      // TimelineCanvasViewport.tsx:40
const internalWidth = Math.max(1, Math.floor(displayWidth * dpr));   // :43
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);                             // :53

// 波形 WebGL2：实际缩放比 = round(w·dpr)/w ≠ dpr
const width = Math.max(1, Math.round(widthPx * dpr));               // surfaceRenderer.ts:15
gl.viewport(0, 0, internal.width, internal.height);                 // :113
gl.uniform2f(this.resolutionLocation, widthPx, heightPx);           // :117  ← CSS px
```

顶点着色器 `clip = a_position / u_resolution * 2 - 1`，`a_position` 是 CSS px、`u_resolution` 也是 CSS px，于是 NDC 被拉伸到 `round(w·dpr)` 物理像素 —— 实际缩放比 `round(w·dpr)/w`，在 dpr=1.25/1.5 或宽度为奇数时，边缘最多差半像素且**随窗口宽度变化跳动**。这就是波形"整体比 clip 宽/窄一点点"的来源。

Piano Roll 主画布用 `Math.floor(w*dpr)`（`render.ts:651`），与波形的 `round` 又差一档。

**R3 — 提交时序不同步。** 网格走 `useLayoutEffect` + `gridRedrawBridge` 命令式桥；Clip 画布走总线同步订阅立即重绘；波形走 **rAF**（`WaveformSurface.tsx:48-54`）；标尺走 React 渲染周期。缩放瞬间各层不在同一 commit 提交，出现"扇形分离"的一帧。

**R4 — 取整策略各写一套。** 波形用 `Math.ceil` 取像素列（`geometry.ts:148-149`）；网格不取整靠 SVG 亚像素；标尺用 `left: -1; width: 2` 手工半像素补偿（`TimeRuler.tsx:127-129`）；Clip 画布完全不取整。1px 级错位由此产生。

**R5 — 宽度 clamp 不统一。** Clip 有 `Math.max(1, …)` 下限（`timelineCanvasModel.ts:145`），波形没有 → 极小 clip 处两者宽度不一致。

### 2.4 Piano Roll 侧的附加问题

- **`scrollLeft` 语义分裂**：state/ref 存的是"绘制坐标"，DOM 与 `timelineViewportSync` 用"原生坐标"，二者差一个动态量测的 `timelineOffsetPx`（`utils/timelineViewportSync.ts:98,103`；`PianoRollPanel.tsx:981-998`）。
- **两条总线契约不对称**：`timelineViewportBus` 有 `scrollLeft/pxPerSec/viewportWidth/scrollTopPx/rowHeight/revision`，`pianoRollViewportBus` 只有前三个（`pianoRollViewportBus.ts:14-18`），缺 `scrollTopPx` 与 `revision`，无法判定快照是否变更，且残留快照会跨面板实例泄漏。
- **缩放锚点已共用**：`timelineScrollRange.ts:69-141` 供两侧使用，这部分**不在**本次改动范围内，只需保证它输出的 `scrollLeft` 喂给 axis 时语义一致。

---

## 3. 设计

### 3.1 `TimelineAxis`：唯一投影

新增 `frontend/src/components/layout/timeline/runtime/timelineAxis.ts`。

```ts
export interface TimelineAxis {
    readonly pxPerSec: number;
    /** 绘制坐标（内容坐标原点相对视口左缘的偏移），非 DOM 原生 scrollLeft。 */
    readonly scrollLeftPx: number;
    readonly scrollTopPx: number;
    readonly viewportWidthPx: number;
    readonly dpr: number;
    readonly revision: number;
}

secToContentPx(sec: number): number   // sec * pxPerSec              —— 全局唯一乘法
contentPxToSec(px: number): number
secToViewportPx(sec: number): number  // secToContentPx(sec) - scrollLeftPx
viewportPxToSec(px: number): number
durationToWidthPx(sec: number): number // max(MIN_FEATURE_PX, sec * pxPerSec) —— clip 与波形共用
snapPx(px: number): number             // Math.round(px * dpr) / dpr
strokePx(px: number, width: number): number // 线宽居中：snapPx(px) + (width % 2 ? 0.5 / dpr : 0)
with(patch: Partial<...>): TimelineAxis     // 结构共享，便于 React.memo 比较
```

**强制约束（写入文件头注释，评审检查项）**

1. 任何图层禁止直接读 `pxPerSec` / `scrollLeft` 做乘法或减法。
2. 禁止 `scrollLeft / pxPerSec` 还原成秒 —— R1 的根源。
3. 位置与线宽必须经 `snapPx` / `strokePx` —— R4 的解法。
4. `durationToWidthPx` 是 clip 宽度与波形宽度的**唯一**来源 —— R5 的解法。
5. axis 只接受**绘制坐标**；原生坐标只在 DOM 边界（`measureTimelineViewportOffsetPx` / `timelineViewportStateToNative`）转换。

### 3.2 `canvasRaster`：统一 DPR 契约

新增 `frontend/src/components/layout/timeline/runtime/canvasRaster.ts`。

```ts
export interface RasterTarget {
    cssWidthPx: number; cssHeightPx: number;
    physicalWidth: number; physicalHeight: number;  // = Math.round(css * dpr)
    dpr: number;
    /** WebGL u_resolution 必须传这个值，而非 cssWidthPx。 */
    resolutionWidth: number; resolutionHeight: number;  // = physical / dpr
}
export function rasterize(canvas: HTMLCanvasElement, cssW: number, cssH: number, dpr: number): RasterTarget
```

契约：**绘制坐标一律 CSS px；物理尺寸一律 `Math.round(css * dpr)`；WebGL 的 `u_resolution` 传 `physical / dpr`。**

推导：顶点在 CSS 空间 → NDC = `pos / (physical/dpr) * 2 - 1` → 物理像素 = `(pos·dpr/physical) · physical = pos · dpr`，与 Canvas2D 的 `setTransform(dpr,…)` 严格等价。R2 消除。

所有画布（时间线 Clip 体、波形 WebGL2、波形 Canvas2D 回退、Piano Roll 主画布与轴画布）统一改调 `rasterize`。

### 3.3 `TimelineFrameCommitter`：单一帧提交

新增 `frontend/src/components/layout/timeline/runtime/timelineFrameCommitter.ts`。

```ts
export interface TimelineLayer {
    readonly name: string;
    paint(axis: TimelineAxis): void;
}
register(layer: TimelineLayer, order: number): () => void
commit(axis: TimelineAxis): void   // 同步、按 order 升序调用 paint，axis 未变则跳过（幂等去重）
```

图层顺序（order）：`GridBack(10) → ClipBody(20) → Waveform(30) → GridOverlay(40) → Playhead(50) → DomOverlay(60)`。

`timelineViewportBus.emit` 与 `pianoRollViewportBus.emit` 内部改为调用 `commit(axis)`。波形的 rAF 路径删除（R3 消除）；以"axis 未变则跳过"做同帧去重，重复 emit 不产生额外绘制，性能不降反升。

`gridRedrawBridge.ts`（WeakMap 命令式桥接）在 P2 后可整体删除。

### 3.4 统一 tick 源：网格与标尺同源

新增 `frontend/src/components/layout/timeline/runtime/buildTimelineTicks.ts`。

```ts
export interface TimelineTick {
    sec: number;
    contentPx: number;          // = axis.secToContentPx(sec)
    strength: 0 | 1 | 2;        // 弱线 / 强线 / 小节线
    isBarStart: boolean;
    primaryLabel: string;
    secondaryLabel: string | null;
}
export function buildTimelineTicks(args: {
    axis: TimelineAxis; tempoMap: TempoMap | null;
    timeUnit: TimeUnit; gridSize: GridSize; beatsPerBar: number;
    minSpacingPx?: number; swingPercent?: number;
}): TimelineTick[]
```

网格消费 `contentPx + strength` 画竖线，标尺消费同一数组画刻度与标签。**一次生成、两处消费**，Tempo Map 与非 Tempo Map 只是 tick 生成的两种策略，不再有 `buildTempoGridLineXsForViewport` 与 `resolveGridLineSamplingPlan` 双路径。

### 3.5 Piano Roll 接入

1. 三套 x 公式全部收敛为 `axis.secToViewportPx(sec)`（曲线、选区、剪贴板预览、播放头、检测音高线、音阶高亮）。
2. `pianoRollViewportBus` 与 `timelineViewportBus` 合并为**共享实现** `createViewportBus()` 工厂，两个面板各持一个实例，契约统一（含 `scrollTopPx` / `revision`）。
3. `pianoRoll/render.ts:651` 的 `Math.floor(w*dpr)` 改为统一 `rasterize`（round）。
4. `seekPlayheadMapping.ts` 保留语义（刻意绕开 beat 换算以避免 BPM 变更漂移），但内部改用 axis。
5. `timelineScrollRange.ts`（缩放锚点）保持不动，仅确认其输出喂给 axis 时为绘制坐标。

---

## 4. 分阶段实施计划

每阶段独立可运行、独立 commit、独立验收；上一阶段验收通过才进入下一阶段。

| 阶段 | 内容 | 涉及文件 | 交付物 |
|---|---|---|---|
| **P0** | `TimelineAxis` 内核；时间线侧接入：波形 `sceneBuilder`、播放头 `TimelineSurface`、Clip 模型 `timelineCanvasModel` | 新增 `timelineAxis.ts`；改 `sceneBuilder.ts`、`TimelineCanvasViewport.tsx`、`TimelineSurface.tsx`、`timelineCanvasModel.ts` | R1、R5 消除 |
| **P1** | `canvasRaster` 统一 DPR；`TimelineFrameCommitter` 取代波形 rAF 与各层分散订阅 | 新增 `canvasRaster.ts`、`timelineFrameCommitter.ts`；改 `surfaceRenderer.ts`、`WaveformSurface.tsx`、`timelineViewportBus.ts` | R2、R3 消除（**最直观**） |
| **P2** | 统一 tick 源；网格与标尺同源；删除 `gridRedrawBridge.ts` | 新增 `buildTimelineTicks.ts`；改 `BackgroundGrid.tsx`、`TimeRuler.tsx`、`TimelinePanel.tsx` | R4 消除，代码量净减少 |
| **P3** | Piano Roll 接入：三套公式收敛、scrollLeft 语义统一、双总线共享实现、DPR 统一 | 改 `PianoRollPanel.tsx`、`pianoRoll/render.ts`、`pianoRoll/utils.ts`、`PianoRollWaveformSurface.tsx`、`pianoRollViewportBus.ts` | 两侧一致 |
| **P4** | 属性测试 + dev 对齐自检面板 + 死代码清理 | 新增 `*.property.test.ts`、`dev/TimelineAlignmentProbe.tsx`；删 `viewportStore.ts`、`weightedLru.ts` | 防回归 |

> 代码规范：每个新增/修改文件头部维护"主要内容、作用、与其他模块的关系"注释；每个关键函数头部维护"流程、作用、特殊规则、参数说明"注释（遵循仓库既有约定）。

---

## 5. 验证

### 5.1 属性测试（P4，但 P0 起同步补基础用例）

`timelineAxis.property.test.ts`：随机 `pxPerSec ∈ [1, 2000]`、随机 `scrollLeft`、随机 `sec`、随机 `dpr ∈ {1, 1.25, 1.5, 2, 3}`，断言：

1. **往返一致**：`secToViewportPx(viewportPxToSec(x))` 与 `x` 误差 < 1e-9。
2. **线性性**：`secToViewportPx(t + d) - secToViewportPx(t)` 与 `d` 成正比，误差 < 1e-9。
3. **跨层一致**：同一 `sec` 经网格 tick `contentPx`、Clip `leftPx`、波形 `screenRect.x + scrollLeftPx` 三条路径投影，结果**严格相等**（不是近似）。
4. **DPR 一致**：`rasterize` 后 `resolutionWidth * dpr === physicalWidth`，且 `physicalWidth === Math.round(cssW * dpr)`。
5. **clamp 一致**：`durationToWidthPx` 在极小 duration 下与波形宽度同源。

### 5.2 dev 对齐自检面板

`frontend/src/components/layout/timeline/dev/TimelineAlignmentProbe.tsx`，仅在 `import.meta.env.DEV` 下渲染：

- 在视口内取一组固定时间点（小节线、秒整点、clip 边界、随机点）。
- 分别用**网格、标尺、Clip 画布、波形**四条路径投影，各画一条同色半透明竖直参考线。
- 完全重合 → 面板显示绿色 "ALIGNED"；任一路径偏差 ≥ 0.5px → 红色告警 + `console.warn` 输出偏差明细与 axis 快照。

### 5.3 人工冒烟

- 拖动水平滚动条 / 滚轮缩放 / `Ctrl+滚轮` / 缩放按钮 / 拖拽 clip 边界，观察四条参考线是否始终重合。
- dpr=2 与 dpr=1.5（系统缩放）各验一遍。
- Tempo Map 开启/关闭各验一遍（验证不等距网格）。

---

## 6. 风险与回退

| 风险 | 缓解 |
|---|---|
| P1 去掉波形 rAF 后，滚动时波形重建几何的耗时可能放大到单帧内 | 先测：滚动帧内波形 `buildWaveformGeometry` 耗时 < 2ms 则直接同步；否则在 committer 内做"滚动中降精度"（复用低一级 mipmap），停顿后恢复全精度 |
| P2 统一 tick 源后 Tempo Map 极端密度下的性能 | 保留现有 `buildTempoGridLineXsForViewport` 的"按步长预缩放 + stride 抽取"有界策略（`tempoMap.ts:997`），迁移时不改算法 |
| P3 改动 Piano Roll 主画布公式可能影响曲线编辑手感 | P3 前先对 `render.ts` 的曲线 x 换算做快照测试（旧公式 vs 新投影，误差 < 1e-9），确认等价后再替换 |
| 任一阶段引入回归 | 每阶段一个 commit，可 `git revert` 单独回退，不影响已验收阶段 |

---

## 7. 文件变更清单

**新增**
- `frontend/src/components/layout/timeline/runtime/timelineAxis.ts`
- `frontend/src/components/layout/timeline/runtime/canvasRaster.ts`
- `frontend/src/components/layout/timeline/runtime/timelineFrameCommitter.ts`
- `frontend/src/components/layout/timeline/runtime/buildTimelineTicks.ts`
- `frontend/src/components/layout/timeline/dev/TimelineAlignmentProbe.tsx`
- 对应 `*.property.test.ts`

**修改**
- `frontend/src/waveform/sceneBuilder.ts`、`WaveformSurface.tsx`、`surfaceRenderer.ts`、`geometry.ts`
- `frontend/src/components/layout/timeline/TimelineSurface.tsx`、`BackgroundGrid.tsx`、`TimeRuler.tsx`、`TimelineCanvasViewport.tsx`、`runtime/timelineCanvasModel.ts`、`TimelinePanel.tsx`
- `frontend/src/utils/timelineViewportBus.ts`
- `frontend/src/components/layout/PianoRollPanel.tsx`、`pianoRoll/render.ts`、`pianoRoll/utils.ts`、`pianoRoll/PianoRollWaveformSurface.tsx`、`pianoRoll/pianoRollViewportBus.ts`、`pianoRoll/seekPlayheadMapping.ts`

**删除**
- `frontend/src/components/layout/timeline/gridRedrawBridge.ts`（P2）
- `frontend/src/waveform/viewportStore.ts`、`frontend/src/waveform/weightedLru.ts`（P4，已确认无生产引用）
