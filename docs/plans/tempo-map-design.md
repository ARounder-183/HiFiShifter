# Tempo Map（速度 / 拍号 / 音阶图谱）设计文档

## 概述

为 HiFiShifter 添加工程级 Tempo Map：在时间轴的不同位置定义不同的 BPM、拍号（每小节拍数）与音阶，时间标尺、背景网格、网格吸附、小节/节拍标签以及所有依赖音阶的处理逻辑（音高吸附、音阶高亮、按度数移调、量化、均值量化、子轨道“度数差”渲染等）全部按 Tempo Map 分段生效。支持从 MIDI（作为音高参考块导入时）与 REAPER 工程（.rpp）导入 Tempo Map 数据，并支持导出到 MIDI。

## 设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 变化点坐标 | 时间锚定（绝对秒） | HiFiShifter 时间轴以秒为绝对坐标（clip、播放头、参数帧），时间锚定模型保证编辑 Tempo Map 不会移动任何音频，音乐时间（拍/小节）完全由 Tempo Map 推导 |
| 变化方式 | 阶梯式（Step） | 每个点携带 BPM/拍号/音阶，自该秒位置生效直到下一个点；REAPER 线性渐变段在导入时采样为若干阶梯点近似 |
| 拍号 | 分子 + 分母 | 一小节拍数 = 分子 × 4 ÷ 分母；小节对齐在拍号变化点重置（与 REAPER 语义一致），段末尾不足一小节的余拍计为“不完整小节” |
| 音阶 | 可选覆盖（null = 跟随工程音阶） | 某位置生效音阶 = 该位置前最近一个携带音阶的变化点；无覆盖时使用工程音阶 |
| 工程基准值 | 与 0 位置点同步 | `bpm` / `beats_per_bar` 始终镜像 0 位置点，删除 Tempo Map 后工程回退一致 |
| 持久化 | `TimelineState.tempo_map` | 随工程文件自动序列化（`#[serde(default)]`，旧工程为 None），参与后端撤销历史 |
| 渲染缓存 | 音阶签名比较 | 仅当 Tempo Map 音阶部分（或工程音阶）发生变化时失效渲染缓存并触发后台预渲染（尊重“后台预渲染”开关） |

## 数据模型

```typescript
// 前端（utils/tempoMap.ts）
interface TempoMapScaleData { key?: string; name?: string; notes?: number[] }
interface TempoPoint {
    id: string;
    positionSec: number;   // 绝对秒，时间锚定
    bpm: number;           // 10-960
    numerator: number;     // 1-32
    denominator: number;   // 1|2|4|8|16|32
    scale: TempoMapScaleData | null;  // null = 跟随工程音阶
}
interface TempoMap { points: TempoPoint[] }  // 升序，首点位于 0
```

```rust
// 后端（state.rs / models.rs）
pub struct TempoPointData { id, position_sec, bpm, numerator, denominator, scale: Option<TempoScaleData> }
// TimelineState.tempo_map: Option<Vec<TempoPointData>>（serde default → None）
// TimelineStatePayload.tempo_map: Option<Vec<TempoPointPayload>>（始终序列化该字段）
```

## 职责划分

### 前端

- `utils/tempoMap.ts`：核心转换（秒↔拍、小节.拍分解、网格生成、吸附、音阶查询、编辑辅助、序列化）。
- `features/session/sessionSlice.ts`：`tempoMap` / `tempoMapVisible` 状态、`setTempoMap` 等 reducer、`applyTimelineState` 解析 `tempo_map`、`setTempoMapRemote` thunk；工程音阶变更时 bump `paramsEpoch`。
- `components/layout/timeline/TempoMapRulerRow.tsx`：Tempo Map 标尺行（段说明、变化点旗帜、拖拽移动、双击新建）与变化点编辑对话框。
- `components/layout/timeline/TimeRuler.tsx`：标尺集成（行显示、动态高度、右键菜单 Tempo Map 分区）。
- `components/layout/timeline/timeFormat.ts` / `BackgroundGrid.tsx` / `hooks/useTimelineState.ts`：Tempo Map 感知的刻度生成、显式网格线、吸附。
- `TimelinePanel.tsx` / `PianoRollPanel.tsx`：两处标尺接线、角框高度对齐（`rulerHeight.ts`）。
- `MenuBar.tsx`：视图菜单开关（默认开）、“工程音阶”选项的 Tempo Map 影响提示（`pianoRollSelectionBus` 读取参数编辑器选区）。
- `ActionBar.tsx`：BPM/拍号显示与编辑作用于播放头位置的生效值。
- `MidiTrackSelectDialog.tsx`：导入为 Tempo Map 选项组（仅音高参考块目标显示）。
- 音阶联动：`usePianoRollInteractions.ts`（吸附/度数拖拽/粘贴预览按帧取音阶）、`pianoRoll/render.ts`（音阶高亮分段）、`childPitchOffsetPaste.ts`、`paramValuePreviewLogic.ts`。

### 后端

- `state.rs`：`TempoPointData` / `TempoScaleData`、`TimelineState.tempo_map`、`normalize_tempo_map`、`effective_scale_notes_at_sec`、`scale_segments`。
- `models.rs` / `commands/timeline.rs`：`set_timeline_tempo_map` 命令（校验、撤销检查点、音阶签名比较、缓存失效与后台预渲染）。
- `commands/project.rs`：工程音阶变更时失效渲染缓存 + 后台预渲染；拍号设置与 Tempo Map 0 点同步。
- `commands/core.rs`：`set_transport` BPM 与 Tempo Map 0 点同步。
- `import/midi_import.rs`：解析 FF 58（拍号）/ FF 59（调号）事件；`build_tempo_map_points_from_midi` 构建时间锚定变化点。
- `commands/midi.rs`：`get_midi_tracks` 返回拍号/调号统计；`import_midi_as_clip` 支持导入为 Tempo Map。
- `import/reaper_parser.rs`：TEMPOENVEX 点解析（含 slowcurv 打包拍号解码，参见 [RPR GetSetEnvelopeState](https://wiki.cockos.com/wiki/index.php/RPR_GetSetEnvelopeState)）。
- `import/reaper_import.rs` / `commands/reaper.rs`：.rpp 导入时构建并应用 Tempo Map（线性渐变采样；REAPER MIDI item 的变速不参与）。
- `pitch_editing.rs` / `commands/midi_export.rs`：子轨道“度数差”逐帧按生效音阶渲染；MIDI 导出写入 Tempo Map（变速/变拍/变调 meta 事件）。

## 关键交互

- 添加：时间标尺右键 → `在此添加速度变化…` / `在此添加拍号变化…` / `在此添加音阶变化…`；或 Tempo Map 行空白处双击。
- 编辑：双击变化点旗帜 / 右键 `编辑此变化点…`（BPM、拍号分子/分母、音阶）。
- 移动/删除：非首点拖拽移动（吸附网格）、右键或对话框删除、`清除 Tempo Map`。
- 显示：`视图 → Tempo Map`（默认开）；工程无 Tempo Map 数据时不显示行。
