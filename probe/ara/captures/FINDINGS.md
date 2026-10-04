# Task 1 FINDINGS —— REAPER 通过 ARA 实际提供了什么

> 证据：`captures/` 下四份真实 dump。`ara-model.awkward.json`（2 源 / 2 modification /
> 5 region，含拉伸与淡化）是主要依据；`ara-model.reaper.json` 是最小对照。
> 采集方式：隔离 REAPER 实例（见 `../README.md`），插件为插桩后的 ARA SDK Test PlugIn。
> 日期：2026-10-04 · ARA SDK 2.3.0 · REAPER 7.81

---

## 1. ARA 实际给出的字段（全部为实测，非文档推断）

### `audioSource`

| 字段 | 实测值 | 备注 |
| --- | --- | --- |
| `persistentID` | **素材绝对路径** `E:\...\tone44100.wav` | 见 §3 第 1 条，这是最关键的一项 |
| `name` | 文件名 | |
| `sampleRate` | `44100` / `48000` | 每源各自保真，不随工程 |
| `sampleCount` | `88200` / `96000` | |
| `durationSeconds` | `2` | 由 sampleRate/sampleCount 得出 |
| `channelCount` | `1` | |
| `merits64BitSamples` | `true` | REAPER 建议 64 位采样访问 |
| `sampleAccessEnabled` | `true`（最小采集）/ `false`（awkward 采集） | **未决，见 §4** |
| `deactivatedForUndoHistory` | `false` | |

### `musicalContext` / `regionSequence`

| 字段 | 实测值 | 备注 |
| --- | --- | --- |
| `musicalContext.name` | `null` | REAPER 未设置 |
| `musicalContext.orderIndex` | `0` | |
| `regionSequence.name` | 轨道名（`awkward-44k` / `awkward-48k`） | **regionSequence ↔ REAPER 轨道** |
| `regionSequence.orderIndex` | `0` / `1` | 与轨道顺序一致 |

### `audioModification`

| 字段 | 实测值 | 备注 |
| --- | --- | --- |
| `persistentID` | **与所属 source 相同** | 本插件上 1 source ↔ 1 modification |
| `name` | `null` | |
| `playbackRegionCount` | `4` / `1` | 多 region 挂同一 modification |

### `playbackRegion`

| 字段 | 实测值 | 备注 |
| --- | --- | --- |
| `startInModificationTime` | `0` | 源内窗口起点（秒） |
| `durationInModificationTime` | `2` / **`1`** | **拉伸时此值变小** |
| `startInPlaybackTime` | `0` / `3` / `6` / `9` | 时间线位置（秒） |
| `durationInPlaybackTime` | `2` / `1` | |
| `isTimestretchEnabled` | 恒 `false` | 本插件不声明支持拉伸 |
| `isTimeStretchReflectingTempo` | 恒 `false` | 同上 |
| `hasContentBasedFadeAtHead` / `Tail` | 恒 `false` | 本插件不声明支持内容淡化 |

进度：`sampleAccessEnabled` 与全部数值型字段都来自实测输出，未做任何推测性填值。

---

## 2. 与 REAPER item 模型的同构程度

| REAPER 概念 | ARA 对应 | 判定 |
| --- | --- | --- |
| item 的源文件 | `audioSource`（按源聚合） | **同构**，且多 item 共享同源时自动合并 |
| item 的位置 / 长度 | `startInPlaybackTime` / `durationInPlaybackTime` | **同构**，直接是秒 |
| take 的裁剪窗口 | `startInModificationTime` / `durationInModificationTime` | **同构** |
| take 的播放速率 | 两个 duration 的比值 | **可无损反推**（§3 第 2 条） |
| REAPER 轨道 | `regionSequence` | **同构**（名字即轨道名） |
| take | `audioModification` | **部分同构**：本插件 1:1，多 take 行为未验证 |
| item 淡化 | 仅两个布尔标志 | **不同构**，见 §3 第 4 条 |
| item 增益 / 声道模式 | 无 | **ARA 无此概念** |
| 参数曲线（pitch/tension/…） | 无 | **ARA 无此概念**（预期之中） |

---

## 3. 四条对映射有实质影响的结论

**1. `persistentID` 就是素材绝对路径。** 这是 `Clip.source_path` 的直接对应物，
也是 spec §5.2 映射表里最不确定的一行。现在有实测支撑。
**但要注意**：spec §4.2 已定"宿主是权威源"，而 ARA 同时给了路径与
`sampleAccessEnabled` 两条路径 —— 走文件 IO 还是走 ARA 采样访问，是实现期要定的分叉，
两者在缓存键上的后果不同（见 `render_key.rs` 的源身份/文件签名部分）。

**2. `playback_rate` 必须从两个 duration 反推，不能读标志位。**
拉伸 region 实测 `durationInModificationTime=1 / durationInPlaybackTime=1`，
源本身 2 秒。即
`rate = durationInModificationTime / durationInPlaybackTime`。
这条同时说明 `isTimestretchEnabled` 对映射**不是必需的**。

**3. 采样率逐源如实上报，不做统一。** 44.1k 与 48k 源在同一工程里各带自己的
`sampleRate`，因此"模型域固定 44.1kHz"不会造成错位；重采样责任明确落在渲染侧。

**4. 淡化只能拿到两个布尔，拿不到形状与曲率。**
本插件声明不支持 content-based fades，REAPER 因此从不置位。
HiFiShifter 的 `fade_in_shape` / `fade_in_dir`（`fade_curves.rs` 那套 REAPER 实测反推的
曲线模型）**在 ARA 里没有对应来源**。

---

## 4. 丢失字段清单（Step 5 的核心产出）

按"渲染是否需要"分级。渲染入口是
`audio/mixdown.rs::render_mixdown_interleaved`，以下按它实际消费的输入逐条核对。

### A. 渲染必需、ARA 不提供 —— 必须由 HiFiShifter 自己承担

| 字段 | 渲染中的用途 | ARA 状况 |
| --- | --- | --- |
| 参数曲线（pitch / breath / tension / formant / volume / pan / dyn） | `maybe_apply_pitch_edit_to_clip_segment` 等 | **ARA 无此概念**。这是 HiFiShifter 的核心价值，本就该自持 |
| 淡化形状 `fade_in_shape` / 曲率 `fade_in_dir` | `fade_curves.rs` 包络计算 | 只有两个布尔标志，形状不可得（§3 第 4 条） |
| Clip 增益 `clip.gain` | 混音增益 | ARA 无 item 增益概念 |
| Take 级音量 / 声道模式（`channel_mode`） | 条件化发生在渲染输入段 | ARA 无对应字段 |
| Take 选择（哪个 take 是 active） | `active_take_id` | 多 take 行为未验证；本插件 1:1 |

### B. 渲染需要、ARA 以**间接形式**提供 —— 需换算，但不丢信息

| 字段 | ARA 形式 | 换算 |
| --- | --- | --- |
| 播放速率 `playback_rate` | 两个 duration 的比值 | §3 第 2 条 |
| Clip 位置 / 长度 | `startInPlaybackTime` / `durationInPlaybackTime` | 秒，直接可用 |
| 源窗口 | `startInModificationTime` / `durationInModificationTime` | 秒，直接可用 |
| 源身份 | `persistentID`（绝对路径） | 直接可用 |
| 源音频内容 | `sampleAccessEnabled` + 内容读取器 | 需经 ARA 读，或回落文件 IO |

### C. 未决 —— 必须先澄清才能进 Task 3

| 项 | 现象 | 为什么必须澄清 |
| --- | --- | --- |
| `sampleAccessEnabled` | 最小采集 `true`，awkward 采集 `false` | 插件**必须**能读样本才能做音高分析。若是持续为假，则 ARA 路径下无法分析，整个"宿主供源 + 本地合成"的链路不成立 |

### D. 明确不在 ARA 范围（非缺陷）

- tempo map / 拍号 / 调号：**存在**，但走内容读取器（`kARAContentTypeTempoEntries` /
  `kARAContentTypeBarSignatures`），不在对象模型里。Task 1 的 dump 未采集，
  因为计划要求的 region 映射不依赖它（见 ledger 的对应 Ruling）。
- ARA 的 `ARAColor`：与本仓库渲染无关，已刻意省略。

---

## 5. 对 spec 的回写建议

1. §5.2 映射表：把 `ARAAudioSource → Clip.source_path` 从"推断"升级为"已实测"，
   并补注 `persistentID` 即绝对路径。
2. §5.2 映射表：`playback transformation → playback_rate` 应改写为
   **由两个 duration 的比值反推**，而不是"读变换标志"。
3. §5.2 / §6：新增一条明确缺失项 —— **淡化形状与曲率 ARA 不提供**，
   必须由 HiFiShifter 模型自持。这会进入 Task 3 的"丢失字段"测试。
4. §6 风险表：R2（映射是否无损）目前证据**倾向于成立**，但以 §4.C 的
   `sampleAccessEnabled` 未决为前提；在该项澄清前不应标为"已验证"。
