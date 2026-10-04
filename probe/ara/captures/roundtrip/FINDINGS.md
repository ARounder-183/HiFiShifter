# Task 3 FINDINGS —— `ARA region → TimelineState` 映射原型

> 写于 2026-10-04。对应 plan 的 `Task 3`。代码：`probe/ara/mapping/`（一次性）。
> 实测口径：转换器落点是**产品真实类型** `backend_lib::__test_internals::{Clip, TimelineState}`，
> 不是探针自造的影子结构。11 条测试全通过。

---

## 1. 结论

**R2 成立（在 ARA 表达得到的范围内）：映射对渲染所需的字段无损。**

- 位置 / 长度 / 源身份 / 源内偏移 / 拉伸速率 / 逐源采样率，全部一一对应且值一致；
- ARA 表达不到的输入集中在**一份 7 条的显式清单**里，全部以"显式降级 + 测试钉住"处理，
  没有一处是悄悄填默认值。

**一处必须更正的既有结论**（见 §4）：Task 1 的 awkward 样本**并没有携带任何拉伸信号**，
ledger 里"拉伸 = 时长差，awkward 已证明"这句的**证据不成立**（公式本身成立，见 §5）。

## 2. 怎么做的（以及一个关键前提）

落点选择上曾有一个看起来会撞墙的问题：`backend/src-tauri/src/lib.rs` 里的模块几乎全是
私有的（`mod state;`、`mod mixdown;`），外部 crate 拿不到 `TimelineState` / `Clip`。
但 backend 已经有一个 `#[doc(hidden)] pub mod __test_internals`，专门供 `tests/` 集成目标
使用，它公开 re-export 了 `state::{Clip, TimelineState}`。

**因此探针不需要给 backend 加任何 `pub`，也就不违反"不改 backend/"。**

构造方式：把映射结果写成 JSON 再 `serde_json::from_value` 成真实结构，最后调用
`Clip::normalize_takes()` 把 take 物化到扁平投影 —— 这正是产品加载工程文件走的同一条路径。

## 3. 映射口径（与 spec §5.2 一致）

| ARA 字段 | TimelineState 落点 |
| --- | --- |
| `regionSequence.name` | `Track.name`（REAPER 实测：序列名 = 轨道名） |
| `audioSource.persistentID` | `Clip.source_path`（绝对路径） |
| `playbackRegion.startInPlaybackTime` / `durationInPlaybackTime` | `Clip.start_sec` / `length_sec`（秒，无需换算） |
| `startInModificationTime` | `take.source_start_sec`（`source_end_sec` = start + duration） |
| `durationInModificationTime / durationInPlaybackTime` | `take.playback_rate`（**不是** `isTimestretchEnabled` 标志位） |
| `audioSource.sampleRate` / `channelCount` / `durationSeconds` | take 的 `source_sample_rate` / `source_channels` / `duration_sec`，逐源保真 |

**已知近似（探针限定）**：Task 1 的 dump 没有记录 region → regionSequence 这条边，
所以原型按各序列声明的 `playbackRegionCount` 顺序切分扁平的 region 列表。
真实 ARA 里这条边是直接存在的（`createPlaybackRegion` 带 sequence 参数），产品实现必须用它。

## 4. 更正的既有结论：awkward 样本里没有拉伸

awkward 样本的 5 个 region，**每一个**都满足
`durationInModificationTime == durationInPlaybackTime`：

| region | 源 | durMod | durPlay | 比值 |
| --- | --- | --- | --- | --- |
| 1 | tone44100 | 2 | 2 | 1.0 |
| 2 | tone44100 | 1 | 1 | 1.0 |
| 3 | tone44100 | 2 | 2 | 1.0 |
| 4 | tone44100 | 2 | 2 | 1.0 |
| 5 | tone48000 | 2 | 2 | 1.0 |

Task 1 的 ledger 把第 2 行的"1 秒 vs 2 秒素材"读成了拉伸证据，但那只是**区间被裁短到 1 秒**；
两个时长相等意味着**没有拉伸**。合理解释是：当时的测试插件在加载时声明不支持时间拉伸，
所以 REAPER 根本没把拉伸写进 ARA 模型。

测试 `task1_awkward_fixture_does_not_actually_carry_a_time_stretch` 把这一事实钉住，
避免后续实现者拿这份样本去"验证"拉伸。

**后续必须补的实验**：用 Task 2 那个已经能跑的插件，声明
`supportedPlaybackTransformationFlags = Timestretch | ReflectTempo | ContentFades`，
再采一次带真实拉伸/淡化的样本。这是目前唯一还没被宿主级观测覆盖的映射分支。

## 5. 拉伸公式本身是成立的（合成样本验证）

`synthetic_duration_ratio_becomes_playback_rate`：durMod=2、durPlay=1 →
`clip.playback_rate == 2.0`、`length_sec == 1.0`、源窗口 `[0, 2)`。
公式与 `state.rs` 里消费侧的窗口定义一致
（`playback_window_sec`：正放 `[ss, ss+len·r)`）。

## 6. 丢失字段清单（渲染需要、ARA 不给）

| 字段 | ARA 为什么给不了 | 本原型的处理 |
| --- | --- | --- |
| 倒放 | ARA 的 playback transformation 只有 Timestretch / ReflectTempo / ContentFade，**没有反向位**（已核对 `ARAInterface.h`） | 显式 `reversed = false` |
| 淡化形状与曲率 | ARA 只有"头/尾是否有基于内容的淡化"两个布尔 | 显式 0 |
| 淡化长度 | ARA 不给出淡化时长 | 显式 0 |
| Loop source | ARA 的 region 模型没有 REAPER / VEGAS 的 loop 语义 | 显式 `loop_enabled = false` |
| Tempo Map | ARA 对象模型不带 tempo（要经 content reader 另取） | `tempo_map = None` + 默认 BPM 120 |
| Item / Take 增益 | ARA 对象模型不提供 item gain | 交给 HiFiShifter 自己的模型 |
| 源文件内容指纹 | ARA 只给 persistentID（路径），不给内容哈希 | 显式 `None`；**渲染缓存键必须另行处理**（spec §4.2） |

判定：**没有任何一条属于"缺了就无法产生正确音频"** —— 淡化 / Loop / 增益 / 反放在
spec 里本来就归 HiFiShifter 自己的模型；内容指纹由 ARA 的内容变更通知替代。
唯一需要盯住的是**倒放**：如果宿主只以"改源内容"的方式表达倒放，插件读到的是已反向的源，没问题；
如果宿主想用 region 属性表达，ARA 没有这个位 —— 这一点仍未验证。

## 7. Step 6（逐样本音频比对）做不了，原因与最小解法

plan 要求把映射出的 `TimelineState` 喂给现有 `render_mixdown_interleaved` 逐样本比对。
**这一步无法按原样完成**：该函数（以及 `MixdownOptions`）没有经 `__test_internals` 暴露，
外部 crate 拿不到，而给 backend 加 `pub` 违反探针约定。

最小解法（需要用户批准，因为它改 backend/）：在既有的
`backend/src-tauri/src/lib.rs` 的 `__test_internals` 里加两行 re-export：

```rust
pub use crate::audio::mixdown::{render_mixdown_interleaved, MixdownOptions};
```

（并按需补 `pub(crate)`→`pub` 的可见性。）加上之后，本 crate 的测试即可直接把映射结果渲染出来，
与"手工拼出的同内容 TimelineState"逐样本比较，阈值按 plan 取 1e-6。

**在那之前，本任务对 R2 的结论只覆盖"字段级无损"，不覆盖"渲染输出一致"。**

## 8. 复现

```powershell
. .\tools\msvc-env.ps1
$env:CARGO_TARGET_DIR = "<worktree>\backend\src-tauri\target"   # 复用 backend 已编译的依赖
cd probe\ara\mapping
cargo test --jobs 1 --offline
```

**踩坑记录**：新 crate 必须**先拷一份 backend 的 `Cargo.lock`** 再构建。否则 cargo 会按
自己的解析结果拉入不同版本的 `windows-core`，导致 `backend_lib` 在
`webview2_accelerators.rs` 报 `cast()` 找不到（trait 版本不匹配）——
这是"作为外部依赖编译"与"包内自建"的特性统一差异，不是 backend 的缺陷。
