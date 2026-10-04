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
