# HiFiShifter 作为 ARA2 插件接入 DAW · 设计文档

> 状态：待评审
> 日期：2026-10-04
> 分支：`feature/ara-bridge-probe`
> 相关计划：`docs/superpowers/plans/2026-10-04-ara-bridge-probe.md`（待编写）

---

## 1. 目标

让 HiFiShifter 能以 ARA2 插件的形式挂在 DAW 里，并且**时间线内容随 DAW 走**：DAW 侧的切片、位置、走带变化能反映到 HiFiShifter 的编辑模型上，HiFiShifter 的修音结果能作为音频回到 DAW 的播放/渲染链里。

**成功判据（v1）**：在 REAPER 里挂上插件、把一段人声素材切片，HiFiShifter 能呈现与 DAW 一致的 clip 布局，调参后的结果在 DAW 里播放正确，且工程重新打开后不需要重新合成。

## 2. 背景与已确认的事实

本节记录已由代码或外部资料证实的事实。推测项在 §6 单独列出。

### 2.1 为什么双进程桥接不可行

ARA 的 renderer 是**宿主在插件进程内回调**的接口。若 PCM 需要跨进程向 HiFiShifter 本体索取再返回，只能退化为"实时推流"，等于放弃 ARA 的内容缓存并引入同步漂移。**因此"切片/位置/走带跟随 DAW"这一目标要求进程内 ARA。**

### 2.2 后端已有条件（按模块统计 `tauri::` / `cpal::` 引用数）

| 模块 | 行数 | Tauri | cpal |
|---|---|---|---|
| `audio/` | 11489 | 0 | 0 |
| `renderer/` | 2947 | 0 | 0 |
| `vocoder/` | 8654 | 0 | 0 |
| `render_cache/` | 1845 | 0 | 0 |
| `import/` | 10797 | 0 | 0 |
| `synth_clip_cache.rs` / `render_key.rs` / `pitch_editing.rs` / `formant_cache.rs` | — | 0 | 0 |
| `pitch/` + `pitch_analysis/` | 3962 | 8 | 0 |
| `state.rs` | — | 11 | 0 |
| `audio_engine/` | 5791 | 21 | 6 |
| `commands/` | 18702 | 196 | 0 |

**整个合成与离线渲染内核已与 Tauri、cpal 解耦。** 耦合集中在设备边界（`audio_engine/engine.rs` 建 cpal 流）、`commands/` 的 IPC 包装、以及 `lib.rs` 装配。

### 2.3 ARA 需要的三件事，仓库已各自实现

1. **任意区间离线渲染**：`audio/mixdown.rs` 的 `render_mixdown_interleaved(timeline, opts) -> Result<(u32, u16, f64, Vec<f32>), String>`，`MixdownOptions` 含 `start_sec` / `end_sec` / `sample_rate`。
2. **内容哈希 + 跨进程持久化缓存**：`render_key.rs` 统一产键，`render_cache/` 落盘。其注释已明确"缓存存的是整条 clip、clip 局部帧 0 起"，与 ARA 的 audio modification 语义同构。
3. **预渲染缓冲 + 回调只读缓存**：`audio_engine/mix.rs` 的 `render_callback_f32(data, out_channels, snapshot, ...)` 不关心缓冲区去向；`snapshot.rs` 构建快照时读 `global_rendered_clip_cache()`。

### 2.4 ARA 侧事实

- ARA SDK 自 2021 年起以开源许可发布（[Celemony/ARA_SDK](https://github.com/Celemony/ARA_SDK)，Apache-2.0）。
- ARA 插件仍需一个 VST3（或 AU）外壳才能被宿主加载。
- Rust 侧无成熟绑定：只有较新的第三方 crate（[ara2-bridge](https://github.com/eas4ai/ara2-bridge)），主流是 [官方示例](https://github.com/Celemony/ARA_Examples) 与 [JUCE 的 ARA 支持](https://raw.githubusercontent.com/juce-framework/JUCE/master/docs/ARA.md)，均为 C++。
- 宿主 ARA 支持是 DAW 侧既成事实。**FL Studio 不支持 ARA。**

## 3. 范围

### v1 范围内

- 单实例：`一个插件实例 = 一条人声编辑轨`。
- ARA 文档 → HiFiShifter 时间线模型的映射（源、区间、放置、拉伸、淡化）。
- 进程内渲染：复用现有 mixdown/renderer/vocoder/render_cache。
- 参数编辑通道：使本体的编辑能驱动插件渲染（见 §5.4）。
- REAPER 作为验证宿主。

### v1 范围外（明确不做）

- 多实例（一轨一实例）。理由见 §4.1。
- 插件内嵌参数编辑器 UI。
- macOS / Linux：ARA 宿主主要存在于 Windows/macOS，而 `vslib` 仅 Windows，故 **Linux 在本路线出局**。macOS 是否纳入 v1 待定（见 §8）。
- vslib 算法的分发（见 §7）。

## 4. 关键设计决策

### 4.1 决策：v1 单实例

**选择**：一个 ARA 实例对应一条"人声编辑轨"，而非一轨一实例。

**理由**（三条独立理由指向同一结论）：

1. **算力**：渲染是 ONNX 推理（HiFiGAN + hnsep），不是轻量 DSP。多实例等于在宿主进程内并发多份推理；且 ORT 会话与 DirectML 设备本就是进程级单例，多实例只会争抢同一 GPU。
2. **贴合既有语义**：HiFiShifter 以"轨道组 + 根轨道 `C` 开关 + 一套参数线"为单位，天然是一个文档对应一个编辑对象。
3. **保住最大的一笔便宜**：单实例意味着约 40 个进程级单例（`GLOBAL_RENDERED_CLIP_CACHE`、`GLOBAL_SYNTH_CLIP_CACHE`、`GLOBAL_FORMANT_CACHE`、`GLOBAL_BREATH_NOISE_CACHE`、`GLOBAL_CLIP_PITCH_CACHE`、`global_clip_rendering_state`、`HNSEP_CACHE`、`SHARED_SESSION` 等）**基本不需要改造** —— 单例在一个实例里是正确的。需要改的只剩真正的设备边界。

**代价（须明确接受）**：v1 无法让 DAW 里每条轨道都独立修音。

### 4.2 决策：宿主的音频源是权威源

ARA 的 `ARAAudioSource` 指向宿主托管的音频。一旦源的权威在 DAW：

- 插件**不得**再依赖 `decode_audio_f32_interleaved(Path::new(source_path))` 直接开文件，必须经 ARA 的内容读取接口获取 PCM。
- **渲染缓存键必须纳入"源内容版本"**。现有键包含源文件身份与文件签名（`render_key.rs`、`GLOBAL_FILE_SIG_CACHE`）；在 DAW 里改源而不改路径是常规操作，若键不变就会静默复用过期缓存 —— 这是会静默出错的那类缺陷，必须在设计里封死。

### 4.3 决策：渲染产物交回 ARA 模型

用 `storeAudioSourceContent` 把渲染结果写回宿主的 ARA 模型，使 DAW 负责其保存与迁移。这与 `render_cache/` 已实现的"内容哈希键 + 落盘 + 重开免重算"同构，是 Melodyne 一类插件的通行做法。

### 4.4 决策：v1 不在插件内做参数编辑器

插件声明无编辑器窗口。参数编辑仍在本体完成。这使得 v1 不必把 WebView2 + React 塞进宿主进程（这是单独就能否决"直接把现有应用变成插件"的障碍）。

### 4.5 决策：模型映射以既有 REAPER 语义为准

`Clip` / `Take` / 拉伸标记 / 淡化形状 / loop 语义均已有对齐 REAPER 的实现。映射应**复用这些已逆向的语义**，而不是为 ARA 另立一套。

## 5. 架构

### 5.1 组件

```
┌─ DAW (REAPER) ─────────────────────────────────────────────┐
│  ARA 宿主：提供 audioSources / audioModifications /         │
│            playbackRegions / tempo map / transport          │
└───────────────┬────────────────────────────────────────────┘
                │ ARA2 (进程内)
┌───────────────▼────────────────────────────────────────────┐
│ 新 crate: hifishifter-plugin                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ARA 宿主适配层（新写）                                │  │
│  │  · ARA 文档 → TimelineState 映射                      │  │
│  │  · AudioSourceReader（经 ARA 读源 PCM）               │  │
│  │  · storeAudioSourceContent（回写渲染产物）            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 复用 backend_lib 内核（不依赖 Tauri / cpal）          │  │
│  │  audio/ · renderer/ · vocoder/ · render_cache/        │  │
│  │  pitch/ · synth_clip_cache · render_key               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 设备边界替换（新写，替代 audio_engine 的 cpal 部分）  │  │
│  │  宿主回调 → render_callback_f32                       │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────┬────────────────────────────────────────────┘
                │ 参数编辑通道（§5.4）
┌───────────────▼────────────────────────────────────────────┐
│ HiFiShifter 本体（沿用现有 Tauri 应用）                     │
│  · 参数编辑器 UI（现有 PianoRoll）                          │
│  · 不再自建音频设备（或仅在独立运行时自建）                 │
└────────────────────────────────────────────────────────────┘
```

### 5.2 数据映射

| ARA 概念 | HiFiShifter 对应物 | 备注 |
|---|---|---|
| `ARAAudioSource` | `Clip.source_path` + `source_start_sec` / `source_end_sec` | 权威在宿主（§4.2） |
| `ARAAudioModification` | `Take` | 已对齐 REAPER Take / VEGAS Take |
| `ARAPlaybackRegion` | `Clip` | `start_sec` / `length_sec` |
| playback transformation | `playback_rate` + 拉伸标记分段 | 已有 REAPER 拉伸标记展开实现 |
| loop source | `loop_enabled` | 已对齐 REAPER Loop source |
| region 淡化 | `fade_in_shape` / `fade_in_dir` | `fade_curves.rs` 为依据 REAPER 实测反推 |
| 宿主 tempo map | 工程 Tempo Map | 已有实现 |
| `storeAudioSourceContent` | `render_cache` | 内容哈希键 + 落盘 |

### 5.3 数据流（渲染）

1. 宿主变更（切片/移动/走带）→ ARA 通知 → 适配层更新 `TimelineState`。
2. 渲染请求 → `render_mixdown_interleaved` 或按 clip 的 `render_cache` 路径 → PCM。
3. 若 ARA 要求 renderer 语义：经 `storeAudioSourceContent` 回写宿主模型。
4. 播放回调 → 复用 `render_callback_f32`，快照来源改为"ARA 内容的读取器"而非本地文件解码器。

### 5.4 参数编辑通道（一等需求，非可选项）

**问题**：§4.1 的单实例与 §4.4 的"编辑器在本体"合起来要求插件实例与本体进程之间存在一条状态通道。参数曲线是渲染输入，插件渲染时必须能读到；曲线不可能只存在于本体内存里。

**v1 方案（待评审确认）**：本体作为该通道的客户端，插件实例为服务端。

- 通道承载：参数曲线（pitch / breath / tension / formant / volume 等）、播放头位置、选中对象。
- 需要明确的三件事：**谁持有曲线的权威副本**、**并发编辑的提交语义**、**插件实例消亡时曲线的归属**。
- 恢复既有工程时，插件实例需要能从工程文件重建曲线，而不能依赖本体进程恰好在线。

**这是本设计里最未收敛的部分**，§6 的探针不覆盖它；它需要在实现计划里独立成任务。

## 6. 风险与待验证假设

按风险排序。**每一条都有对应的杀死判据**（详见探针计划）。

| # | 假设 | 若失败意味着 | 验证成本 |
|---|---|---|---|
| R1 | Rust 侧能以可接受的成本接触 ARA（`ara2-bridge` 可用，或自写绑定规模可控） | 映射再完美也是零 | 1–2 天 |
| R2 | ARA 的 region 模型与 `TimelineState` 的映射对渲染所需字段无损 | 方案死 | 2 天 |
| R3 | 单实例设备边界可替换（cpal 流出，宿主回调进） | 需重做播放模型 | ~1 周（把握较大） |
| R4 | 渲染回调在 ARA 提前渲染窗口内总能给出音频 | 会出现可听见空洞 | 随 R2 |
| R5 | 参数编辑通道的权威/并发语义可收敛 | §5.4 需重新设计 | 未知 |

**R4 补充**：现有回调在缓存未命中时静音等待（`render_callback_f32` 的 `data.fill(0.0)` 分支与快照的"静音等待渲染"），在本体是正确的（传输层由自己控制），在 ARA 宿主里会表现为空洞。ARA renderer 角色提供提前渲染窗口，故大概率可解，但这是主要工程风险。

## 7. 明确不承诺的事项

1. **vslib 的分发**。`vslib` 是闭源 DLL + 闭源 C API（C API 为文件 IO 型，每次渲染写 `%TEMP%\hs_vslib_{uuid}.wav`），且仅 Windows。在 MIT 项目里作为插件分发需要单独处理，v1 不承诺。
2. **FL Studio 支持**。宿主不支持 ARA。
3. **Linux 支持**。见 §3。
4. **目标机器上的 ARA 宿主可得性**。见 §8。

## 8. 未决问题

1. **macOS 是否纳入 v1**：ARA 宿主在 macOS 同样存在，但 `vslib` 缺失，且打包链路需另做。
2. **验证宿主**：本机已确认有 FL Studio（不支持 ARA）。REAPER 待安装（用户已确认愿意装）。
3. **§5.4 通道的权威与并发语义**。
4. **稳定基线**：本分支的 `cargo test` 基线尚未取得（构建环境受干扰，见计划文档的"环境前提"）。此项应在任何代码改动前补齐。

## 9. 参考

- [Celemony/ARA_SDK](https://github.com/Celemony/ARA_SDK) — ARA SDK 伞形仓库
- [Celemony/ARA_Examples](https://github.com/Celemony/ARA_Examples) — 官方示例（含 ARADemoPlugin）
- [JUCE ARA 文档](https://raw.githubusercontent.com/juce-framework/JUCE/master/docs/ARA.md) — 宿主/插件两侧的 ARA 支持说明
- [eas4ai/ara2-bridge](https://github.com/eas4ai/ara2-bridge) — 纯 Rust 的 ARA2 绑定尝试
- [Dreamtonics ARA Bridge 模式](https://sv2.docs.dreamtonics.com/en/ara-bridge) — "重 UI 留在本体"的先例
