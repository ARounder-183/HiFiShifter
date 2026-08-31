# 时间线波形缩放/平移性能优化设计

- 日期：2026-08-31
- 范围：`frontend/src/waveform/*`、`frontend/src/components/layout/timeline/runtime/timelinePerfScenario.ts`
- 目标：解决「10 轨道 / 400 clip / 1 分钟音频 / 全览缩放」下缩放与平移卡顿
- 已确认决策：平移复用走 **shader uniform 方案**；**不引入** in-app 性能探针（仅用离线基准 + Chrome DevTools 验证）

---

## 1. 目标与非目标

### 目标

1. 建立可复现的**离线基准**，量化波形渲染各阶段耗时，作为优化依据与回归防线。
2. 消除波形绘制热路径上的**每帧对象分配**（GC 压力）。
3. 让**水平平移**不再重建波形几何，退化为一次 GPU uniform 更新。
4. 降低**缩放**过程中的单帧重建成本。

### 非目标

- 不改变波形的视觉表现（颜色、线宽、包络形状、淡入淡出、loop 标记）。
- 不改动 mipmap 的级别划分与选级阈值（`DIV_FACTORS` / `SPP_THRESHOLDS` 已验证正确）。
- 不重构 `TimelinePanel` 的 React 数据流（阶段 2 才处理，且需单独评估）。
- 不启用后端已实现但前端未接入的 tile 化 mipmap 路径（`hfspeaks_v2` 的 `to_tile_binary`）。

---

## 2. 现状分析

### 2.1 数据流

```
滚动/缩放
  → useTimelineState.syncScrollLeft()      (hooks/useTimelineState.ts:440)
      ├─ timelineViewportBus.emit(...)     同步派发（utils/timelineViewportBus.ts:74）
      │    └─ frameCommitter.commit(axis)  (runtime/timelineFrameCommitter.ts:99)
      │         └─ waveform 图层 paint      (WaveformSurface.tsx:243)
      │              └─ draw()  → buildWaveformScene + buildWaveformGeometry + render
      └─ setScrollLeft(...)                rAF 合并的 React state 更新（:476）
```

关键：**每次滚动/缩放事件都同步跑完整的 `draw()`**（`WaveformSurface.tsx:104-148`），
scene 与 geometry **无任何跨帧缓存**。

### 2.2 单帧成本估算（待基准验证）

场景：10 轨 × 40 clip = 400 clip，每段 60 s，跨度 ≈ 2400 s，视口 1500 px
→ `pxPerSec ≈ 0.625`，**每个 clip 仅 37.5 px 宽**。

| 环节 | 位置 | 每帧成本 |
|---|---|---|
| 视口裁剪失效 | `sceneBuilder.ts:226-230` | 400 clip 全部进入后续流程 |
| 段对象分配 | `sceneBuilder.ts:366-392` | 400 segment + 400 嵌套 `screenRect` |
| `getPeaks` × 400 | `waveformMipmapStore.ts:494` | 800 个 `subarray` 视图对象 |
| 逐像素列 | `geometry.ts:154` | 400 × 38 ≈ **15,200 列** |
| 列内 min/max 扫描 | `geometry.ts:179-182` | ≈ **25.8 万次**迭代 |
| 顶点写入 | `geometry.ts:215-216` | **18.2 万次**浮点写入 |
| 顶点缓冲拷贝 | `geometry.ts:247` `slice()` | **每帧新分配 ≈ 730 KB** |
| 线段展开 | `surfaceRenderer.ts:31-75` | 54.7 万次写入 + **≈10.6 万次数组分配** |
| GPU 上传 + 绘制 | `surfaceRenderer.ts:201-202` | **2.19 MB** / 30,400 三角形 |

### 2.3 四类根因

**R1（最严重）— `expandLineSegmentsToQuads` 在段循环内分配数组。**

```55:62:frontend/src/waveform/surfaceRenderer.ts
        const corners = [
            [x1 + nx, y1 + ny, base + 2], // A（起点色）
            [x2 + nx, y2 + ny, base + 8], // B（终点色）
            [x2 - nx, y2 - ny, base + 8], // C（终点色）
            [x1 + nx, y1 + ny, base + 2], // A
            [x2 - nx, y2 - ny, base + 8], // C
            [x1 - nx, y1 - ny, base + 2], // D（起点色）
        ] as const;
```

15,200 段 × 7 个数组 = **每帧约 10.6 万次对象分配**。这是纯浪费，语义上等价于
内联展开 6 次写入。

**R2 — 平移时几何不变却全量重算。**
`pxPerSec` 不变时，每个 clip 的包络形状完全不变，只有 x 整体平移
`−ΔscrollLeft`。当前代码不感知这一点，仍从 400 个 clip 重算一遍。

**R3 — 每帧大块内存分配。**
`vertexScratch.slice(0, used)`（`geometry.ts:247`）每帧拷贝 ≈730 KB。注释声称
「跨帧复用、稳态零分配」，但最后的 `slice` 使复用失效。

**R4 — per-clip 固定开销被 400 倍放大。**
37 px 的 clip 只产出 38 个像素列，却要付 6 次函数调用 + 3 个对象分配 + 2 个
`subarray` 的固定成本。视口裁剪在全览缩放下完全失效，无从削减。

### 2.4 附带发现

- `timelinePerfProbe.ts` / `timelinePerfScenario.ts` **是死代码**，仅被各自的
  测试引用，从未接入生产链路。场景生成器的 clip 参数写死（`lengthSec: 1.2`、
  `startSec = clipIndex * 1.5`），与真实场景不符。
- `frameCommitter.commit()` 每次执行 `layers.slice().sort()`
  （`timelineFrameCommitter.ts:104-106`），图层数少时可忽略，但可顺手改为
  有序插入。

---

## 3. 阶段 3：量化（先做）

### 3.1 为什么用离线基准

`buildWaveformScene` / `buildWaveformGeometry` / `expandLineSegmentsToQuads`
**都是纯函数**，不依赖 DOM / React / WebGL。可直接在 Node 下计时，既能指导优化
优先级，又能进 CI 防回归，且不给生产代码增加任何运行时开销。

### 3.2 改动清单

1. **扩展 `buildTimelinePerfScenario`**（`runtime/timelinePerfScenario.ts`）
   新增参数 `clipLengthSec`（默认 60）、`gapSec`（默认 0）、`viewportWidthPx`
   （默认 1500），返回值增加 `axis`——按「全部内容可见」反算出全览 `pxPerSec`。
   保留现有默认行为，不破坏 `timelinePerfScenario.test.ts`。

2. **新增合成 peaks 源**（新文件 `frontend/src/waveform/perfFixtures.ts`）
   按 L2 密度（`44100 / 4096 ≈ 10.8` peaks/s）生成**确定性**伪随机 min/max，
   长度与真实数据一致（60 s → ≈646 点）。实现 `WaveformPeakResolver` 接口，
   额外记录调用次数，用于验证 R4 的 per-clip 开销假设。

3. **新增基准**（`frontend/src/waveform/waveformPerf.bench.ts`）
   - `scene`：400 clip 全览，仅 `buildWaveformScene`
   - `geometry`：喂入上一步 scene，含 `getPeaks`
   - `quads`：对上一步顶点做 `expandLineSegmentsToQuads`
   - `full-frame`：三个阶段串起来

4. `package.json` 增加 `"bench": "vitest bench"`。
   Vitest 内置 benchmark 能力，无需新增依赖。

### 3.3 验收

- 三个阶段耗时占比明确，能确认 R1 / R3 / R4 的实际量级。
- 若结果推翻 2.2 的估算，按实测重排第 4 节的优先级。

---

## 4. 阶段 1：低风险修补

### 4.1 消除 per-segment 数组分配（对应 R1）

`surfaceRenderer.ts:41-73`：删除 `corners` 字面量数组，改为内联 6 次写入。
行为完全等价，`surfaceRenderer.test.ts` 直接作为回归防线。

> 可独立提交，风险最低。

### 4.2 平移复用：shader uniform（对应 R2）

**核心洞察**：`pxPerSec` 不变时，顶点形状不变，只有 x 整体平移
`−ΔscrollLeft`。顶点数据已经躺在 GPU buffer 里，平移只需改一个 uniform。

**着色器改动**（`surfaceRenderer.ts:125-136`）：

```glsl
uniform vec2 u_offsetPx;
void main() {
    vec2 zeroToOne = (a_position + u_offsetPx) / u_resolution;
    ...
}
```

**超量绘制（overdraw）**：平移会露出新区域，几何必须按「视口 + 余量」构建。

余量宽度 `marginPx = clamp(round(viewportWidthPx * 0.25), 128, 512)`。

实现上**不需要改 `sceneBuilder.ts`**——用 `withAxis` 派生一个更宽的 axis 即可：

```ts
const buildAxis = withAxis(axis, {
    scrollLeftPx: axis.scrollLeftPx - marginPx,
    viewportWidthPx: axis.viewportWidthPx + 2 * marginPx,
});
```

这样 `secToViewportPx` 自然产出 `[-marginPx, widthPx + marginPx]` 范围的坐标，
`buildWaveformScene` 的裁剪边界也随之扩展。

**偏移量推导**：以 `S0` 为构建时的 `scrollLeftPx`，则 `S0' = S0 − marginPx`。
之后处于 `S1` 时，需要
`u_offsetPx.x = (S0 − marginPx) − S1`。

**渲染器接口**新增平移重绘：

```ts
interface WaveformSurfaceRenderer {
    render(geometry, widthPx, heightPx, dpr): void;
    /** 仅平移：复用 GPU 上已有的顶点，只更新 offset uniform 后重绘。 */
    repaintPan(widthPx: number, heightPx: number, dpr: number,
               offsetPx: { x: number; y: number }): void;
}
```

- WebGL 路径：`uniform2f` + `drawArrays`，**零 CPU 顶点成本**。
- Canvas2D 路径：顶点在 CPU 侧，退化为 `setTransform(dpr, 0, 0, dpr, offsetPx.x * dpr, offsetPx.y * dpr)`，
  仍需重放 path（比全量重建便宜，因为省掉了 scene + geometry）。

**复用判定**（`WaveformSurface.draw()`）：

```
可复用 ⟺ 有缓存
      ∧ pxPerSec 未变
      ∧ viewportWidthPx 未变
      ∧ rows 引用未变
      ∧ color 未变
      ∧ |S1 − S0| ≤ marginPx
```

任一条不满足即全量重建并重新锚定缓存。

**竖直方向**：`u_offsetPx.y` 同样处理，但行窗口化会让 `rows` 引用随
`startTrackIndex` 变化而失效，因此竖直滚动仍走全量重建。竖直余量列为后续项。

### 4.3 顶点缓冲复用（对应 R3）

`geometry.ts` 的 `vertexScratch.slice()` 改为**由调用方提供的缓冲**：
`buildWaveformGeometry` 增加可选 `out` 参数；`WaveformSurface` 维护一个两槽
缓冲池，`render()` 结束后立即归还（顶点已上传 GPU，CPU 侧不再需要）。

配合 4.2 后，平移帧**完全零分配**；缩放帧也省掉每帧 730 KB 的拷贝。

### 4.4 减少 per-clip 分配（对应 R4）

- `getBestSliceView` 每次新建 2 个 `subarray`（400 clip → 800/帧）。改为让
  `geometry.ts` 直接消费 `(peaks, startIdx, endIdx)`，需要调整
  `WaveformPeakView` 接口，避免视图对象。
- `buildWaveformScene` 的 segment 改为**扁平结构**（`x/y/w/h` 直接作为字段，
  去掉嵌套 `screenRect`），减半对象数。
- `frameCommitter.commit()` 的 `slice().sort()` 改为有序插入（顺手）。

### 4.5 测试

- 既有测试全绿：`sceneBuilder.test.ts` / `geometry.test.ts` /
  `surfaceRenderer.test.ts` / `waveformMipmapStore` 相关测试。
- **新增不变量测试（关键）**：在 `S0` 以 `marginPx` 余量构建几何，令
  `S1 = S0 + Δ`（Δ ≤ marginPx），断言
  `几何(S0) 平移后 == 几何(S1)` 逐顶点相等（容差 1e-4）。这是 4.2 正确性的
  核心保障，覆盖正向/反向平移、跨 clip 边界、loop tile 三种情形。
- 新增：`expandLineSegmentsToQuads` 展开结果与改前实现逐字节一致。

---

## 5. 阶段 2：React 数据流（后续，需单独评估）

`pxPerSec`（`useTimelineState.ts:345`）与 `scrollLeft`（`:342`）都是 React state：

- 缩放 → `setPxPerSec` → 同步触发 `TimelinePanel`（123 KB 巨型组件）整棵子树重渲染；
- 平移 → `setScrollLeft`（rAF 合并）→ 同样每帧一次。

目标是把二者迁到总线订阅，让视口变更只驱动画布重绘而不触发 React 渲染。
**风险**：依赖面广，需先梳理哪些 UI 真正需要 state 驱动（标尺刻度文本、
虚拟化窗口边界、`sliceVisibleClipIds` 对 400 clip 的 filter/map）。
本阶段不在本次范围内。

---

## 6. 实施顺序与验证

| 步骤 | 内容 | 验证 |
|---|---|---|
| 1 | 阶段 3 基准 | `npm run bench` 产出三阶段耗时基线 |
| 2 | 4.1 消除 `corners` 分配 | 既有测试 + 基准对比 |
| 3 | 4.3 顶点缓冲复用 | 既有测试 + 基准对比 |
| 4 | 4.4 per-clip 分配削减 | 既有测试 + 基准对比 |
| 5 | 4.2 shader uniform 平移 | 新增不变量测试 + 基准 + DevTools 实测平移帧 |
| 6 | 全量回归 | `npm test`、`npm run lint`、真机全览缩放/平移手感 |

每步独立提交，便于二分回退。

### 验收标准（真机，10 轨 / 400 clip / 全览）

- 水平平移：稳定 60 fps，Chrome Performance 面板中无周期性大块 GC。
- 缩放：帧耗时较基线显著下降（具体目标值待步骤 1 的基线数据确定）。
- 波形视觉与优化前逐像素一致。
