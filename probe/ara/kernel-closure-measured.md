# 内核依赖闭包 —— 实测记录

> 测于 2026-10-04，分支 `codex/ara-plugin`，HEAD `bc735223`。
> 方法：从 `backend/src-tauri/src/lib.rs` 解析 `mod` 声明 → 从种子 `mixdown` 出发做
> `crate::X` 引用闭包 → 整体排除设备层 `audio_engine` → 统计模块数 / 字节数 /
> 是否出现 `tauri::`。脚本与探针期的 `kernel-closure2.ps1` 同类（目录型模块扫描全部子文件）。
>
> **这份是"施工清单"的唯一权威来源。** 设计文档 §4.2 里 43 模块这个数是**拆分
> `state.rs` 之前**测的，两者不可混用。

## 1. 三次测量的对比

| 测量时点 | 种子 | 排除 | 模块数 | 字节 | 碰 `tauri::` 的模块 |
| --- | --- | --- | --- | --- | --- |
| 拆分前（探针期） | `{mixdown, state}` | 无 | 43 | 2 568 908 | `state`、`pitch_clip`、`pitch_analysis`、`recording`、`audio_engine` |
| 拆分前 | `mixdown` | `audio_engine` | 42 | 2 280 554 | `pitch_analysis`、`pitch_clip`、`recording`、`state` |
| **拆分后（现状）** | `mixdown` | `audio_engine` | **42** | **2 284 602** | `pitch_analysis`、`pitch_clip`、`recording`、`state` |
| **拆分后 + 假设 `state` 只剩 `model.rs`** | `mixdown` | `audio_engine` | **40** | **2 149 796** | **`pitch_clip`、`pitch_analysis`** |

第 4 行是**目标态**：把 `AppState` 从闭包里彻底摘掉之后，内核是 **40 模块 / 2.15 MB**，
只剩两个模块碰 `tauri::`。

## 2. 目标态的内核模块清单（40 个）

```
audio_utils          clip_rendering_state  config               dml_adapters
encode               fcpe_onnx             formant_cache        formant_morph
glottal_rd           gpu_info              hfspeaks_v2          hnsep_dsp
hnsep_onnx           media                 mel_utils            midi_import
mixdown              models                notebook_assets       nsf_hifigan_onnx
pitch_analysis       pitch_clip            pitch_config         pitch_editing
project              rd_tension            render_cache         render_key
renderer             soundtouch            sstretch             state
streaming_pitch      streaming_world       synth_clip_cache     temp_manager
time_stretch         vibrato               vocoder_ort_session  vslib
world_vocoder
```

## 3. 两条实测结论（都推翻了原计划的假设）

### 3.0 【更正】本文档第 1 节的两行数字来自一个有 bug 的脚本

第 1 节里的脚本用正则 `^\s*(?:pub )?mod\s+` 匹配模块声明，**漏掉了 `pub(crate) mod`**。
于是 `commands` / `channel_policy` / `channel_mode` / `channel_decision` / `stereo_detect`
这些模块根本没进候选表，遍历时也就无法经它们继续扩散 —— **闭包被低估了**。

修正后的实测（同样从 `mixdown` 出发、`state` 只算 `model.rs`、排除 `audio_engine`）：

| 口径 | 模块数 | 碰 `tauri::` 的模块 |
| --- | --- | --- |
| 含测试代码 | **46** | `commands`、`recording`、`pitch_analysis`、`pitch_clip` |
| 仅生产代码 | **44** | `commands`、`recording`、`pitch_analysis`、`pitch_clip` |

多出来的 6 个是 `commands`、`recording`、`search`、`system_clipboard`、`linux_clipboard`
以及 `commands` 的子模块 —— **全部是 app 层的东西**。

### 3.0.1 那它们是怎么被拉进来的：**一条真实的边，正好是 HostServices 要修的那条**

排除测试代码后，唯一的"生产代码越界"是：

```
pitch_analysis/schedule.rs:438  crate::commands::playback::AUTO_BG_RENDER_ENABLED
pitch_analysis/schedule.rs:440  crate::commands::playback::BG_RENDER_PITCH_PENDING
pitch_analysis/schedule.rs:475  crate::commands::playback::request_background_render(app)
```

`pitch_analysis` 是内核模块，它却直接伸手去够 app 层的后台渲染开关与请求函数。
而 `commands` 一旦被拉进来，它自己的依赖（`recording` / `search` / `system_clipboard` /
`linux_clipboard`）就跟着全进来了 —— 46 个模块里有 6 个是这么来的。

**这条边正是设计文档 §4.3 那张表里的第 2 类："向音频引擎投递命令"。**
`HostServices` 要吸收的就是它。

**结论**：只要 `HostServices` 把这条边改掉（外加把 `project.rs` 的测试挪走，
那些测试里也有 `crate::commands::channel_scan`），闭包就回到
**39 个模块 + `state/model`**，也就是设计文档 §4.2 那个数。**原来的估计是对的，
只是当时不知道"对"依赖于先做掉 HostServices。**

所以执行顺序必须是：**先 `EngineCommand` + `HostServices`，再大搬迁。**
反过来做会一路撞墙，而且撞的是同一面墙。

### 3.0.2 另一处较小的越界

`project.rs` 的 `#[cfg(test)]` 里有 `crate::commands::channel_scan::{collect_targets, plan}`
（至少 5 处）。那是"通道扫描"的集成测试。搬 `project.rs` 时这些测试要么挪回 app,
要么把 `channel_scan` 的纯函数部分也拉进内核 —— 二选一，在搬迁那一步决定。

### 3.1 设计文档 §4.2 的预测被部分证伪

原预测：`hfspeaks_v2` / `notebook_assets` / `temp_manager` / `recording` 会离开闭包。

实测：**`recording` 离开了，另外三个没有。** 说明它们不只被 `AppState` 引用，
模型侧（`project` / `models` 这条链）也引用了它们。当时已把该说法标注为推断、
并写明"以重算为准"，所以没有造成误施工，但结论按实测改写。

### 3.2 `state/model` 不是叶模块

它引用 `project`（`CustomScale`）、`models`、`midi_import`、`audio_utils`、`time_stretch`。

**因此顺序必须反过来：先做机械搬迁，再搬 `state/model`。** 原计划的
Task 3（先搬 model）—— Task 6（后搬其余）是反的。

## 4. 为什么 40 模块里有 `soundtouch` / `sstretch` / `vslib`

这三个模块背后是**由 `backend/src-tauri/build.rs` 编译的原生代码**：

| 模块 | 原生形态 | 链接方式 |
| --- | --- | --- |
| `sstretch` | Signalsmith Stretch（头文件库）+ C 包装 `sstretch-c.cpp` | **静态链接**（`rustc-link-lib=static=signalsmith_stretch`） |
| `soundtouch` | CMake 构建的 `SoundTouchDLL.dll` | 运行时加载（LGPL 规避） |
| `vslib` | `vslib_x64.dll` + 导入库 | 动态链接（仅 x86_64 Windows） |

**把它们 `git mv` 进内核，在 app 里会假性通过**（链接参数来自 app 的 build 脚本，
最终二进制的链接不受影响），**但插件单独构建内核时会链接失败**。
处置见设计文档 §4.9（建议：把这两个原生依赖的构建搬进内核自己的 `build.rs`）。
