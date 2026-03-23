# Tempo Map（变速 BPM）设计方案

## 概述

为 HiFiShifter 添加变速 BPM 支持，允许在时间线上定义多个 Tempo 变化点（BPM + 拍号），时间线网格、吸附、小节号随之变化，Clip 的 `playback_rate` 自动跟随 Tempo 变化。

## 设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 变化方式 | 仅阶梯式（Step） | BPM 在变化点位置瞬间跳变，不支持渐变 |
| 拍号变化 | 支持 | Tempo Map 同时支持 BPM 和拍号变化 |
| 编辑交互 | 时间线标尺上方 Tempo 轨道 | 可显示/隐藏 |
| Clip playback_rate | Tempo Map 驱动 | 自动根据 tempo 变化调整 Clip 时间拉伸 |
| Clip 原始 BPM | 隐式确定 | Clip 被放置时所在位置的 BPM 即为其"原始 BPM"，不需额外字段 |
| 主要工作量 | 前端 | 后端以绝对时间为单位，基本不变 |

## 职责划分

### 前端（主要工作量）

- Tempo Map 数据存储与管理（Redux state）
- `tick ↔ 秒` / `tick ↔ 小节:拍:tick` 双向坐标转换
- 时间线网格绘制（不等距网格）
- 吸附逻辑（查询 Tempo Map）
- Tempo 轨道 UI（显示/编辑变化点）
- 计算 Clip 的有效 `playback_rate`

### 后端（极小改动）

- `TimelineState` 新增 `tempo_map` 字段（纯数据透传，后端不使用）
- `TimelineStatePayload` 对应新增字段
- 保存/加载自动跟随序列化，无需额外逻辑

## 数据模型

### 前端 TypeScript

```typescript
interface TempoPoint {
  id: string;
  positionTicks: number;   // tick 绝对位置
  bpm: number;
  numerator: number;       // 拍号分子
  denominator: number;     // 拍号分母
}

interface TempoMap {
  ticksPerBeat: number;    // 如 480
  points: TempoPoint[];    // 按 positionTicks 排序，第一个点固定在位置 0
}
```

### 后端 Rust（TimelineState 新增字段）

```rust
// state.rs - TimelineState
#[serde(default)]
pub tempo_map: Option<Vec<TempoPointData>>,

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoPointData {
    pub id: String,
    pub position_ticks: u64,
    pub bpm: f64,
    pub numerator: u32,
    pub denominator: u32,
}
```

### sessionSlice 新增状态

```typescript
// sessionSlice.ts - SessionState
tempoMap: TempoMap;          // Tempo Map 数据
tempoTrackVisible: boolean;  // Tempo 轨道是否显示
```

- 现有 `bpm` / `beats` 字段保留，作为 `tempoMap.points[0]` 的同步镜像
- `tempoMap` 初始值：`{ ticksPerBeat: 480, points: [{ id: "default", positionTicks: 0, bpm: 120, numerator: 4, denominator: 4 }] }`

## 交互逻辑

### Tempo 轨道

- **位置**：时间线标尺与第一个轨道之间，高度约 24-30px
- **显示/隐藏**：ActionBar 上新增切换按钮控制，隐藏时该行完全收起
- **变化点样式**：标记旗帜，标注 `BPM` 和 `拍号`（如 `140 BPM 3/4`）

### 添加变化点

- 在 Tempo 轨道空白区域双击 → 插入新 TempoPoint
- 新点默认继承前一个点的 BPM 和拍号
- 插入后弹出内联编辑气泡（popover），可修改 BPM 和拍号

### 编辑变化点

- 单击选中（高亮）
- 双击弹出编辑气泡：BPM 数字输入（10-300）、拍号分子/分母选择
- 左右拖拽移动位置，自动吸附到拍/小节线
- 第一个点（位置 0）不可移动、不可删除，只能编辑 BPM 和拍号

### 删除变化点

- 选中后按 Delete 键删除
- 或右键菜单删除

### ActionBar BPM 输入框行为

- **Tempo 轨道关闭时**：修改全局 BPM（等同现有行为，只有 `points[0]`）
- **Tempo 轨道开启时**：修改当前 cursor 位置的 BPM — 如果 cursor 位置已有变化点则修改它，否则在 cursor 位置新增变化点

### Clip playback_rate 联动

- Clip 的"原始 BPM" = 被放置时所在位置的 Tempo Map BPM
- 编辑 Tempo Map 时，受影响区域的 Clip 自动重算 `playback_rate`：
  - `新 playback_rate = 新BPM / 旧BPM * 原 playback_rate`
- 重算后同步后端

## 时间线网格与吸附

### 网格绘制

- 每个 BPM 区域内等间距，不同区域间距不同（BPM 高的区域拍间距更密）
- 遍历 TempoMap 每个区间，分段计算网格线像素位置

### 小节号标尺

- 根据每段区间的拍号累加计算小节号（4/4 段每 4 拍一小节，3/4 段每 3 拍一小节）

### 吸附逻辑

- 给定秒数位置，查询所属区间，计算最近的拍/小节线位置

### 核心工具函数（前端 TempoMap 工具模块）

```typescript
// utils/tempoMap.ts
ticksToSeconds(ticks: number, tempoMap: TempoMap): number
secondsToTicks(seconds: number, tempoMap: TempoMap): number
ticksToBarBeatTick(ticks: number, tempoMap: TempoMap): { bar: number; beat: number; tick: number }
getTempoAt(ticks: number, tempoMap: TempoMap): { bpm: number; numerator: number; denominator: number }
getGridLines(startSec: number, endSec: number, tempoMap: TempoMap, subdivision: number): GridLine[]
snapToGrid(seconds: number, tempoMap: TempoMap, gridSize: string): number
```

## 工程保存/加载

- 后端 `TimelineState` 新增 `tempo_map: Option<Vec<TempoPointData>>` 字段
- `#[serde(default)]` 确保旧工程加载时自动为 `None`（向后兼容）
- `None` 时退化为现有全局 `bpm` 行为
- 前端加载工程时从 `TimelineStatePayload` 取出 `tempo_map` 存入 Redux
- 无需改动保存/加载流程代码

## 实现步骤

1. **后端数据透传**：`state.rs` / `models.rs` 加 `tempo_map` 字段
2. **前端 TempoMap 工具模块**：`utils/tempoMap.ts`，实现所有坐标转换函数
3. **Redux 状态**：`sessionSlice` 新增 `tempoMap` / `tempoTrackVisible`，加 reducers
4. **Tempo 轨道 UI 组件**：`TempoTrack.tsx`，显示/编辑变化点
5. **ActionBar 集成**：Tempo 轨道显示/隐藏按钮，BPM 输入框双模式
6. **时间线网格改造**：`TimelinePanel` 使用 TempoMap 工具函数绘制网格
7. **吸附逻辑改造**：snap 相关逻辑使用 TempoMap
8. **Clip playback_rate 联动**：编辑 tempo 时自动重算
9. **工程保存/加载集成**：前后端字段对接
