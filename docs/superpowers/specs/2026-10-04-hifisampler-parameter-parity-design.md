# HifiSampler 参数对齐设计（gender / tension）

## Goal

对齐 OpenUtau 0.1.571-beta 内置 hifisampler 的两个音色参数语义：

1. **共振峰（gender）**：把现有 `formant_shift_cents` 的实现从「mel 域 bin 线性插值」换成
   OpenUtau 的 **PitchAdjustableMelSpectrogram**（按 keyShift 伸缩 FFT/窗长后重新投影），
   值域由 ±500 扩到 ±1200 cents。
2. **张力（tension）**：把 `hifigan_tension` 从「声码器后立体声总线频谱倾斜」迁移为
   OpenUtau 的 **LF 声门模型 Rd 重塑**，作用于 **HNSep 谐波支**、在 **mel 分析之前**。

并一并修掉迁移过程中确认的两个既有缺陷（见「附带修复」）。

## Background

### 参考实现

OpenUtau `0.1.571-beta`（commit `ec7ba520583173c67aabfc5feab33390b4f720a4`）的
`OpenUtau.Core/Classic/Hifisampler/` 是从 [openhachimi/hifisampler](https://github.com/openhachimi/hifisampler)
`backend/resampler.py` 移植的 C# 版；其中 tension 已被 OpenUtau **替换**为
`OpenUtau.Core/Analysis/GlottalRd.cs` + `HifiRdTension.cs`（上游原版的
`pre_emphasis_base_tension` 频谱倾斜未被采用）。本设计对齐的是 **OpenUtau 的 Rd 版**。

### 现状差异

| 维度 | OpenUtau hifisampler | HiFiShifter 现状 |
|---|---|---|
| gender 机制 | 伸缩 `n_fft`/窗长 + 原 mel 基投影 | mel 域 bin 索引线性插值 |
| gender 值域 | ±100 = ±1200 cents | ±500 cents |
| gender 符号 | 正 = 共振峰**下移** | 正 = 共振峰**上移** |
| tension 时机 | 声码器**前**、仅谐波支 | 声码器**后**、整个立体声混音 |
| tension 算法 | LF 声门 Rd 拟合（Itakura–Saito） | 无模型，dB 线性斜坡 |
| tension 归一化 | 第一谐波恒 1，不重归一化 | RMS 回原响度 + 峰值限 0.98 |
| tension 上限 | 由 Rd 比值决定 | `MAX_TENSION_DB = 17.0` |

### 关于现有 tension 实现的定位（重要）

`third_party/tension.pd`（Pure Data 参考补丁，144 行）表明 HiFiShifter 现有实现是**忠实移植**：

- `pitchEstimation` 子补丁：`sigmund~ -maxfreq 1500` → `mtof` → `clip 100 1000` → `* 2`
  ⇒ 拐点 = `clamp(f0_hz, 100, 1000) × 2`，与 `hifigan_tension.rs::tension_center_hz` 逐字一致；
- `filter` 子补丁：`tension / 100 * 17` → `db2lin~` → `clip~`
  ⇒ 与 `MAX_TENSION_DB = 17.0` 一致。

因此**现有实现不是 bug，而是另一套设计**。`docs/tension_analysis.md` 描述的是 HachiTune 的
实现（固定 1500 Hz 拐点、12 dB 上限），与 `tension.pd` **并非同一算法**；该文档不应作为
本次迁移的依据。

迁移到 Rd 方案的理由是**听感更好**（见「为什么 Rd 更好」），而非修复移植错误。

### 为什么 Rd 更好

用 `LfModel` 数值计算源 Rd=1.0、f0=220 Hz 时 `tension = +100`（Rd→0.5）的逐谐波增益：

```
谐波:      1      2      3      5      8     12     16     24     32     40
增益:  +0.00  +5.81  +8.74 +10.48 +12.17 +13.90 +14.93 +15.86 +15.82 +15.63  dB
```

对照现有倾斜（`MAX_TENSION_DB=17`，拐点 `2×f0`，在 `4×f0` 处撞上平台）：

```
MIDI 60 (f0=262Hz): 262Hz -8.50dB | 523Hz 0.00 | 1047Hz +17.00 | 2093Hz +17.00 | 8kHz +17.00
```

1. **不动噪声**。现有实现对 1 kHz 以上**一切**（含齿音、气声、嘶声）无差别 +17 dB，是「刺耳 /
   金属感」的直接来源；Rd 只重塑谐波结构。
2. **形状是物理的**。Rd 增益按谐波收敛并**回落**（+15.86 → +15.63）；现有实现是无界线性斜坡
   硬撞平台，`4×f0` 以上恒 +17 dB，等效一个高通。
3. **不改变响度**。Rd 第一谐波恒为 1，感知响度天然稳定；现有实现砍掉基频 −8.5 dB 后再做 RMS
   回归一化，补偿不干净且等于整体抬高低频，与「调音色」意图不符。
4. **按目标音高摆放增益**。两者在此点一致（都在目标 f0 的谐波上放置），迁移后保持。

## Scope

### In Scope

- 移植 `LfModel` / `GlottalRd` / Rd 张力施加逻辑到 HiFiShifter（Rust）
- 复活并改造 `mel_from_audio` 为逐帧 keyShift 版本，替换 `shift_mel_formant`
- `formant_shift_cents` 值域 ±500 → ±1200
- `hifigan_tension` 语义改为「谐波支 mel 域 Rd 重塑」；删除后处理实现与其专用缓存层
- 一并修复：tension 导出不生效、breath 曲线越界语义不一致
- `RENDER_PIPELINE_VERSION` 递增（算法变更，持久化缓存必须全失效）
- 单元测试与回归测试

### Out of Scope

- **子轨共振峰偏移（`child_formant_offset`，±2400）叠加后可能超出 ±1200 的情况**：按用户决定
  **不处理**，保持现有叠加与钳制行为不变
- 新增 `voicing` 参数（HNSEP 谐波增益曲线）
- 新增 `growl`、`P`（响度归一化）等 OpenUtau 其余 flag
- 修改 `breath_gain` 的值域或映射曲线
- 导入/导出 OpenUtau 工程文件（本次不涉及格式互通）

## Design

### 一、gender：`PitchAdjustableMelSpectrogram`

#### 算法（对齐 OpenUtau `HifiMelSpectrogram.Compute`）

对每个分析帧：

```
factor      = 2^(keyShift / 12)
nfftNew     = round(2048 * factor)
winSizeNew  = round(2048 * factor)
hop         = 512                     // 不变
padded      = reflect_pad(x, (winSizeNew - hop) / 2, (winSizeNew - hop + 1) / 2)
mag         = |STFT(frame, nfftNew, hop, hann(winSizeNew))|     // 帧数不变
bins'       = resize(mag, 1025)       // 多截少补零，回到模型固定 mel 基维度
binScale    = winSize / winSizeNew    // keyShift == 0 时为 1.0
mel         = melFilterbank[1025 → 128] · (bins' * binScale)
mel         = ln(max(mel, 1e-9))      // 逐元素
```

关键点：**`nfftNew` 会改变频点数**（`nfftNew/2+1`），而 NSF-HiFiGAN 需要固定 128 bins。
因此必须把幅度谱 resize 回 `n_fft/2+1 = 1025` 再做固定矩阵乘法。这正是 OpenUtau
`Math.Min(size, row.Length)` + 缺 bin 补零所做的事。

**frame 数不变**：`winSizeNew - hop` 的 reflect pad 与 `hop` 共同保证
`frames = 1 + (n - hop) / hop`，与 `keyShift` 无关（OpenUtau `FrameCount` 注释同此）。

#### 三处有意的偏离（必须显式记录）

1. **窗函数保留对称 Hann**。OpenUtau 用 `Hnsep.PeriodicHann`（`0.5(1−cos(2πn/N))`，
   librosa/torch 约定）；HiFiShifter 的 `hann_window` 用对称形式
   （`0.5(1−cos(2πn/(N−1)))`）。采用**对称窗**以保证 `keyShift = 0` 时与现有
   `mel_from_audio_fast` 逐样本一致——为对齐 OpenUtau 而改动会让所有既有工程
   的渲染结果发生无谓变化。此处与 OpenUtau **不逐样本相等**，属已知且接受的差异。

2. **keyShift 量化**。连续曲线会使每帧产生不同的 `nfftNew`，导致每帧重新 `FftPlanner::plan_fft_forward`
   （OpenUtau 用 `scratches` 按 `(nfftNew, winSizeNew)` 缓存 + Bluestein 兜底，
   因为 `round(2048 × 2^(k/12))` 通常不是 2 的幂）。HiFiShifter 的 `self.fft` 是构造时
   绑定 2048 的**单个 plan**，无法直接复用。

   方案：把 keyShift 量化到 **1/4 半音**（±12 半音 → 97 档），并按 `nfftNew` 缓存
   `(Arc<dyn Fft>, Vec<f32> window, f32 bin_scale)`。97 档在听感上不可区分，
   但把 FFT plan 数量收敛为常数（每档一个），避免逐帧建 plan 的开销。

   `nfftNew` 可能不是 2 的幂（如 `round(2048 × 2^(1/48)) = 2078`），
   `rustfft::FftPlanner` 支持任意长度（内部用 Bluestein/Radix4 混合），无需自行实现。

3. **`nfftNew` 下限保护**。极端 keyShift 下 `nfftNew` 可能退化到不可用；
   需 `nfftNew.max(4)`（与 OpenUtau `BitOperations.IsPow2(length) && length >= 4`
   的可用性判据同量级），并保证 `winSizeNew >= 1`。

#### 符号与换算

保留 `formant_shift_cents` 的**现有符号**（正 = 共振峰上移），与现有
`shift_mel_formant` 的 `ratio = 2^(shift/1200)`、`hz_src = hz_m / ratio` 语义一致
（该函数 1478 行注释明确「正值 → 共振峰上移 → 声音变细」）。

转换到 keyShift（半音）：

```
keyShift_semitones = +formant_shift_cents / 100
```

即 ±1200 cents ↔ ±12 半音。

**符号推导（务必按此实现，容易搞反）**：keyShift 与共振峰方向的关系是**同号**的。
`nfftNew = round(2048 × 2^(k/12))`，bin 宽度变为 `sr / nfftNew`，而 mel 基是按**原始**
`nfft = 2048` 建立的。因此频率 `f` 的内容落在 bin `f / (sr/nfftNew)`，被基当作
`(sr/2048) × f / (sr/nfftNew) = f × nfftNew/2048 = f × 2^(k/12)` 读出：

- `k = +12` ⇒ 内容出现在 `2f` ⇒ 共振峰**上移**；
- `k = −12` ⇒ 内容出现在 `f/2` ⇒ 共振峰**下移**。

OpenUtau 侧可交叉验证：`keyShift = -0.12 × gender`（`HifiFeatures.cs:130`），
其单测 `GenderCurveMovesFormantsAsWorldline` 断言 `gender=100` 时 1000 Hz → 500 Hz
（下移一个八度）⇒ `keyShift = -12` 对应**下移**，与上式一致。

与 OpenUtau `gender` 的关系：

```
gender = -formant_shift_cents / 12        // gender ±100 ↔ cents ∓1200
```

**语义影响**：扩域后既有工程的 `formant_shift_cents` 曲线**方向与含义不变**，
同一数值的听感强度也不变（同参数、同公式，只是放宽了允许范围）。
既有曲线值落在 ±500 内，不会因扩域而变化。

#### 集成点

- `nsf_hifigan_onnx.rs`：新增 `mel_from_audio_shifted(&mut self, audio: &[f32], shifts: &[f32])`；
  若 `shifts` 全为 0 或为空，**直接走现有 `mel_from_audio_fast`**（保证零 shift 路径逐样本不变）
- 两处调用点（chunked 路径 ~1316、mel-stretch 路径 ~1616）：先算 `shifts`，再选择函数
- **删除** `shift_mel_formant`（1480-1548）及其两处调用
- `renderer/chain.rs`：`formant_shift_cents` 值域 `±500.0 → ±1200.0`
- `renderer/hifigan.rs`：`formant_shift_at_time` 保持返回 cents，换算在 `nsf_hifigan_onnx` 内完成

### 二、tension：LF 声门 Rd 重塑

#### 移植内容（三个模块）

| 源文件 | 行数 | 目标 |
|---|---|---|
| `OpenUtau.Core/Analysis/LfModel.cs` | 202 | `backend/src-tauri/src/audio/glottal_rd.rs`（LF 模型 + 频谱 + Brent 求根） |
| `OpenUtau.Core/Analysis/GlottalRd.cs` | 176 | 同上（Rd 拟合 / 平滑 / 增益 / 插值） |
| `OpenUtau.Core/Classic/Hifisampler/HifiRdTension.cs` | 99 | `backend/src-tauri/src/audio/rd_tension.rs`（STFT 谐波峰 + 施加增益） |

三个文件均属 OpenUtau（MIT）与 hifisampler（Apache-2.0），与 HiFiShifter（MIT）兼容。
移植文件头部必须注明来源、原许可与「已修改」说明（沿用现有 `third_party/` 归因惯例）。

算法要点（`HifiRdTension.Apply`）：

```
1. 谐波支 x 补零到 hop 整数倍，STFT(2048, hop=256, Hann, center=True)
2. 每帧：用 sourceF0 在 |f - k*f0| < 0.3*f0 内找幅度峰（对数幅度抛物线插值），
   最多 min(8000 / f0, 80) 个谐波
3. rd[m] = GlottalRd.Fit(amplitudes, f0)        // 先除唇辐射，再 Itakura-Saito 拟合
4. rd = GlottalRd.Smooth(rd, voiced, round(0.02 * sr / hop))   // 0.02 s 窗
5. 对每帧，若 |tension| > 1e-9：
     rd2   = clamp(rd * 2^(-tension/100), 0.02, 3.0)
     gains = GlottalRd.Gains(rd, rd2, targetF0, sr/2/targetF0)
     spec[k] *= GlottalRd.GainAt(gains, targetF0, k * binHz)
6. ISTFT 回时域，截回原长度
```

关键特性（必须保持）：

- **第一谐波恒为 1**（`Gains` 以第一谐波归一化），**不做任何重新归一化**
- 增益施加在**目标音高**的谐波位置上（变调后音色仍正确）
- 未浊音帧用最近浊音帧填充（`Smooth` 内部），全未浊音则退化为 rd=1.0

#### 源 f0 与目标 f0 的来源

Rd 拟合需要**源 f0**（定位源频谱的谐波峰），增益放置需要**目标 f0**。

现有 `ClipProcessContext` 已同时携带两者（无需新增字段）：

- `clip_midi` = **源音高**。由 FCPE 对**源文件**分析得到
  （`build_clip_pitch_key` 的 hash 前缀为 `clip_pitch_v4_fcpe_source_midi`，
  输入仅 `source_path` + 文件签名 + `frame_period_ms`），再经
  `trim_and_resample_midi` 截取到 clip 消费窗口并按 `playback_rate` 重采样到**时间线帧网格**。
- `pitch_edit` = **目标音高**（绝对 MIDI，0 = 无编辑）。
- 两者的有效值合取由 `renderer/hifigan.rs::midi_fn` 完成
  （`clip_midi_at_time` 取原始，`edit_midi_at_time_or_none` 取编辑，缺编辑时回退原始）。

**时间基一致性（关键）**：`clip_midi` 已按 `playback_rate` 重采样到时间线网格，
因此"时间线绝对时间 → 源 f0"可直接用 `clip_midi_at_time(fp, clip_start_sec, clip_midi, abs_t)`。
对处理器收到的**源速率** PCM 段，第 `i` 个样本对应的时间线绝对时间为：

```
abs_t = seg_start_sec + i / sample_rate / playback_rate
```

（与 `nsf_hifigan_onnx.rs:1894` 的 `chunk_start_sec = start_sec + chunk_start / sr / playback_rate`
同构。）当处理器不自己拉伸时（`ctx_playback_rate == 1.0`），PCM 已在外部拉伸，
映射退化为 `abs_t = seg_start_sec + i / sample_rate`。

由于 Rd 需要**源 f0 的连续性**（谐波峰定位对 f0 抖动敏感），实现中应对 source f0 做
与 `HifiRdTension` 一致的"0 值跳过 + 最近浊音帧填充"处理，而非直接用带 0 的曲线。

#### 施加位置与采样率

选定：在 `process_breath` 内、`render_with_formant` **之前**，对 HNSEP 返回的
`harmonic` 施加 Rd，STFT 用 `cc.sample_rate`。

理由与已知偏差：

- HNSEP 返回的 `harmonic`/`noise` 与输入同采样率（`hnsep_onnx.rs:495-497`
  在非 44100 时重采样回 `sample_rate`），因此此处 `harmonic` 位于 `cc.sample_rate`。
- 重采样到模型采样率发生在 `render_with_formant` **内部**
  （`nsf_hifigan_onnx.rs:1316`）。若要在模型采样率（44100）上做 Rd，
  必须把重采样提前或把 Rd 塞进声码器内部，两者都会显著增加耦合。
- OpenUtau 的输入在读取阶段就被统一重采样到 44100（`Wave.GetSamples`），
  所以它总是在 44100 上做 Rd。**在 44100 素材上本设计与 OpenUtau 等价**；
  非 44100 素材（如 48 kHz）上 Rd 的 STFT bin 宽度不同（`sr/2048`），
  拟合结果会有细微差异。这是为降低耦合而接受的偏差。

`HifiRdTension.Apply` 本就以 `sampleRate` 为参数（OpenUtau 传 `config.SampleRate`），
其 hop=256 与 `MaxFitHz=8000` 均与采样率无关地成立（bin 宽度随 `sr` 缩放），
因此直接传 `cc.sample_rate` 无需改动算法。

#### 挂载点：为什么必须进 `HiFiGanStage`

Rd 张力**只作用于谐波支**，而唯一存在 harmonic/noise 分支的地方是
`HiFiGanStage::process_breath`（`renderer/chain.rs:327-453`）——它调用
`hnsep_onnx::infer_harmonic_noise_mono` 拿到 `(harmonic, noise)`。

因此本次迁移的**架构前提**是：tension 变为 `HiFiGanStage` 内部的 mel 域操作，
并且 **tension 激活时强制走 HNSEP 路径**（与 OpenUtau 的
`NeedsSeparation = HasTension || ...` 判据一致）。

新流程（顺序对齐 OpenUtau `HifiFeatures.Generate`，逐步可对照）：

```
mono PCM (源速率)
  1. HNSEP 分离                      -> (harmonic, noise)
  2. （voicing 增益，本次不实现；OpenUtau 在此乘 voicing/100）
  3. Rd 张力施加于谐波支**波形**       -> STFT(2048,hop=256) 改谱 -> ISTFT -> voiced 波形
  4. 混合：x[i] = breath_gain*(wave[i]-noise_gain...) 形式见下
  5. （OpenUtau 在此把峰值 >0.5 的信号整体缩放并记录 scale；本次**不引入**，
      见下方说明）
  6. mel 分析（含 gender 的逐帧 keyShift）
  7. log 压缩 -> 声码器
```

第 4 步与 OpenUtau 的对应关系：OpenUtau 写 `x[i] = NoiseGain(breath)*(wave[i]-h[i]) + voiced[i]`，
其中 `voiced[i]` 已被第 3 步重塑；HiFiShifter 的 HNSEP 满足 `noise = wave - harmonic`
（见 `hnsep_onnx` 的 `Sub_7` 说明），且现有实现已按 `h + n*breath_gain` 混合。
迁移后需改为 `voiced + noise * NoiseGain(breath_gain)` 的同构形式，
把被 Rd 重塑后的 `voiced` 代入谐波位置。

> **实现约束（此处最易写错）**：Rd 的 STFT→增益→ISTFT 发生在**波形域**，
> 输出是一段**波形**；gender 的 mel 分析发生在其**之后**，作用于整段混合波形。
> 两者不在同一阶段，必须按上面 1→7 的顺序实现。
> 直觉：Rd 改变的是"谐波的相对强度"这一物理事实，应先落实成波形，
> 再让 mel 分析（含 gender 的频率轴伸缩）去观察它。

> **不引入 OpenUtau 的峰值预缩放（第 5 步）**：OpenUtau 在 mel 分析前把峰值压到
> ≤0.5 并在声码器输出后除回 `scale`。HiFiShifter 的 `mel_from_audio_fast` 与
> `render_with_formant` 都**没有**这一层，引入它会改变所有既有工程的渲染结果
> （等于一次隐式的响度归一化）。本次保持 HiFiShifter 现状：不缩放。
> 代价是极端电平素材上 Rd 拟合的稳定性略低于 OpenUtau，属已知且接受的差异。

#### 缓存层简化

迁移后 tension 变成 mel 域操作 ⇒ 自动被 `RenderedClipCache` 的 `extra_curves`
哈希覆盖。因此**删除**：

- `audio/hifigan_tension.rs`（298 行，全部）
- `commands/playback.rs::ensure_hifigan_tension_cache`（384-482）及其调用点（2212）
- `synth_clip_cache.rs`：`compute_hifigan_tension_hash`（991）、
  `TensionRenderedClipCacheKey`（1043）、`TensionRenderedClipCacheEntry`（1051）、
  `TensionRenderedClipCache` + impl（1060-1090）、`global_tension_rendered_clip_cache`（1110）、
  `get_latest_tension_rendered_pcm`（1373）、
  以及 `invalidate_clip_all_caches` 中的第 3 步（1246-1256）与
  `params.rs::invalidate_rendered_clip_caches_for_child_track` 中的对应块（99-105）
- `render_cache`：`EntryKind::Tension`（format.rs:54-65）、`load_tension` / `store_tension`
  （mod.rs:457-471 / 551-565）及其统计分支（755-782）
- `synth_clip_cache.rs::include_rendered_extra_curve`（767-776）中
  `hifigan_tension` 的排除项（`breath_gain` 仍保留排除，理由见下）

> `breath_gain` 必须**继续保留排除**：预览路径靠 `breath_noise_stereo` +
> `audio_engine/mix.rs` 实时混音，把它混进底层哈希会导致每次拖曲线都重合成。

#### 附带修复 ①：tension 导出不生效

现状：`apply_tension_to_stereo` 全仓库只有 **1 个调用点**
（`commands/playback.rs:459`，位于 `ensure_hifigan_tension_cache` 内），
而该函数只被 `render_background_pass` 调用（`playback.rs:2212`）。
导出路径 `audio/mixdown.rs` 中 `tension` 出现次数为 **0** ⇒ **导出时 tension 静默丢失**。

迁移后此问题**自然消失**：tension 进入 `HiFiGanStage`，而 `mixdown.rs:1120` 走的正是
同一个 `maybe_apply_pitch_edit_to_clip_segment` → processor 链路。
需补一条导出回归测试钉住（见 Testing）。

#### 附带修复 ②：breath 曲线越界语义不一致

| 路径 | 曲线越界（超出末点）行为 |
|---|---|
| `renderer/chain.rs:207-209`（导出走这条） | 返回**默认值 1.0** |
| `audio_engine/mix.rs:139-144`（预览走这条） | **保持末值**（hold-last） |

同一条 `breath_gain` 曲线，预览与导出结果不同。

统一为 **hold-last**（与 `mix.rs` 及 `hifigan_tension.rs` 的既有惯例一致，
也符合用户"画到哪就保持到哪"的直觉）：修改 `chain.rs::sample_curve_at_abs_sec`
的越界分支，钳制到末值而非返回 `default_value`。

> 注意：该函数同时被 `formant_shift_cents` 使用，改动会影响共振峰曲线的越界语义，
> 但不影响曲线**存在**时的行为（只影响超出末点的区段）。这是一个行为变更，
> 需在 CHANGELOG / 提交信息中显式说明。
>
> 现有测试 `renderer/chain.rs:568 sample_curve_beyond_end_returns_default` 断言的正是
> **即将改变**的行为（`assert_eq!(frac_beyond, 0.0)` / `assert_eq!(far, 0.0)`），
> 必须改写为 hold-last 断言。该测试的注释记录了一次历史修复（避免"末值↔default 振荡"），
> 改写时须保留该意图：越界区段必须是**常量**（末值），不得重新引入振荡。

### 三、遗留 `tension_edit` 字段的处置

`TrackParamsState.tension_edit`（`state.rs:517`）是旧版遗留字段：

- `set_param_frames` 把裸参数名 `"tension"` 路由到它（`params.rs:527`），
  而**唯一**暴露该 id 的入口是 `frontend/src/components/layout/PianoRollPanel.tsx:2186`
  的短标签分支（`case "tension":`）——当前 `param_descriptors()` **不返回** `"tension"`，
  所以前端永远不会请求它；
- **没有任何渲染器读取它**（全仓库仅在 `engine.rs:1097` 被作为失效判据比较、
  在测试 `engine.rs:2137` 被写入）；
- `project.rs:436-437` 在保存时清空全零值。

结论：`tension_edit` 是**渲染无效**的死字段。本次**不删除**它（避免牵连工程
文件迁移与 `state.rs` 的 linked-param 抽样逻辑，超出本次范围），但需在
`engine.rs:1097` 旁补注释说明"该字段不参与渲染，仅保留以兼容旧工程"，
避免后续误以为它有效而重复实现一套 tension。

> 与本次迁移相关的**唯一**真实 tension 通路是 `extra_curves["hifigan_tension"]`，
> 即 `hifigan_tension_curve_for_clip` 读取的那条。

### 四、RENDER_PIPELINE_VERSION

gender 与 tension 都改变了渲染算法，持久化渲染缓存必须整体失效：

```
synth_clip_cache.rs:561   RENDER_PIPELINE_VERSION: 4 -> 5
```

### 五、前端改动

`formant_shift_cents` 的值域由**后端描述符**驱动（`PianoRollPanel.tsx:2071-2087`
自动初始化视口、3388-3395 从描述符读边界），因此**值域扩域无需前端改动**。

需要检查的点：

- `paramConversion.test.ts:23`、`paramAxisUnits.test.ts:37,141`：断言与 ±500 无关
  （只断言"不支持 dB 轴"），预期不需改
- 工具栏排序（`getParamToolbarRank`）、量化步长（`paramShiftStep.ts`，按
  `range/40` 从描述符推导）均自动跟随
- **`clampParamWriteValue`（`paramRanges.ts:377-403`）没有 `formant_shift_cents`
  分支**（只有子轨 `child formant` ±2400），因此前端无硬编码值域需同步

结论：本次**预期无前端代码改动**；若 `cargo` / `vitest` 暴露断言冲突再最小化修正。

## Testing

### 单元测试（新增）

| 测试 | 断言 |
|---|---|
| `lf_model_matches_reference` | `LfModel::Spectrum` 在若干 (rd, f0, freq) 上与 OpenUtau 参考值一致 |
| `glottal_rd_fit_recovers_known_rd` | 用已知 Rd 生成谐波幅度，`Fit` 还原误差 < 0.05 |
| `glottal_rd_tense_rd_halves` | `TenseRd(1.0, 100) == 0.5`，`TenseRd(1.0, -100) == 2.0`，且钳在 [0.02, 3.0] |
| `rd_gains_first_harmonic_unity` | `Gains(...)[0] == 1.0`（不重归一化的契约） |
| `rd_tension_zero_is_identity` | `tension = 0` 时输出与输入逐样本一致（对应 OpenUtau `ZeroTensionKeepsTheSignal`） |
| `rd_tension_moves_rd_on_source_harmonics` | 在合成分音上测得的逐谐波增益与 `GlottalRd.Gains` 预测一致 |
| `mel_shift_zero_is_identical_to_fast_path` | `shifts` 全 0 时 `mel_from_audio_shifted` 与 `mel_from_audio_fast` **逐样本一致** |
| `mel_shift_moves_formant_by_expected_ratio` | 1 kHz 正弦在 `shift = +1200 cents`（keyShift `+12`）下峰值移到 2 kHz、`shift = -1200` 下移到 500 Hz（**方向断言必须存在**，防止符号写反） |
| `mel_shift_frame_count_is_shift_invariant` | 任意 shift 下帧数恒等于 `1 + (n - hop) / hop` |
| `mel_shift_quantization_buckets_bounded` | 量化后参与计算的 `nfftNew` 取值数 <= 97 |

### 回归测试（新增，钉住附带修复）

| 测试 | 断言 |
|---|---|
| `export_applies_tension` | 导出链路（`render_mixdown_interleaved`）在 tension 激活时输出**不同于**未激活的输出（修复前二者相同） |
| `breath_curve_beyond_end_is_hold_last` | `sample_curve_at_abs_sec` 在超出末点时返回末值而非 default |
| `preview_and_export_breath_agree` | 同一 clip + 同一越界曲线，预览与导出的 breath 混合结果在容差内一致 |

### 既有测试影响

- `render_key.rs` / `synth_clip_cache.rs` 的渲染键测试：`RENDER_PIPELINE_VERSION`
  变更会改变所有哈希值，若测试硬编码哈希常量需同步更新
- `renderer/chain.rs` 的 `sample_curve_beyond_end_returns_default`（568）测试
  断言的正是**即将改变**的行为，必须改写为 hold-last 断言
- `audio/hifigan_tension.rs` 无测试模块，删除无测试损失
- `paramConversion.test.ts` / `paramAxisUnits.test.ts`：预期不需改

### 验证命令

```
cd backend/src-tauri && cargo test
cd frontend && npm test
```

## Risks

| 风险 | 说明 | 缓解 |
|---|---|---|
| Rd 拟合依赖稳定的源 f0 | `clip_midi` 来自 FCPE，未浊音帧为 0；谐波峰定位对 f0 抖动敏感 | 沿用 `GlottalRd.Smooth` 的最近浊音帧填充 + 20 ms 平滑；补测试 |
| 性能回退 | Rd 需逐帧 STFT(2048/hop256) + 谐波峰搜索 + 64 点网格拟合，比现有倾斜贵约一个量级 | 仅在 tension 曲线非默认时触发（沿用 `hifigan_tension_active_for_clip` 门禁）；`Smooth` 的 0 值短路保留 |
| tension 强制走 HNSEP | 未开 breath 的用户开启 tension 时也会跑 HNSEP 推理，首次渲染变慢 | 与 OpenUtau 行为一致；HNSEP 结果已有缓存（`HnsepCache`） |
| 曲线越界语义变更 | 修复 ② 会改变超出末点的渲染结果 | 显式写入提交信息；`RENDER_PIPELINE_VERSION` 递增使旧缓存失效 |
| 平衡性 97 档量化 | 极细的共振峰曲线会被量化 | 1/4 半音（≈3 cents）远低于 JND；测试钉住档数上界 |
| 非 44100 素材的 Rd 偏差 | Rd 在 `cc.sample_rate` 而非模型 44100 上运行 | 44100 素材与 OpenUtau 等价；非 44100 的差异已记录，不额外补偿 |
| 峰值预缩放未移植 | OpenUtau 在 mel 分析前压峰值到 0.5，本设计不引入 | 保持 HiFiShifter 现状以避免改变既有工程渲染结果；极端电平素材拟合稳定性略低 |
| 符号写反 | keyShift 与共振峰方向同号，极易实现成异号 | 测试 `mel_shift_moves_formant_by_expected_ratio` 显式断言双向；spec 内含推导 |
| 删除 tension 缓存层 | 涉及 `synth_clip_cache` / `render_cache` / `params.rs` / `snapshot.rs` 多处 | 按「缓存层简化」清单逐项删除，编译器保证无遗漏引用 |

## Rollout

单次 PR，分三个逻辑提交便于审阅。三者有依赖顺序（1 独立，2 依赖 1 的
`mel_from_audio_shifted` 不会与之冲突但共享 `nsf_hifigan_onnx.rs`，3 依赖 2 的挂载点）：

1. `feat(vocoder): align formant shift with hifisampler PitchAdjustableMelSpectrogram`
   —— gender 算法替换 + 值域扩到 ±1200 + `RENDER_PIPELINE_VERSION` 4→5
2. `feat(vocoder): Rd-based tension on the harmonic stem`
   —— 移植 `glottal_rd.rs` / `rd_tension.rs`，tension 进 `HiFiGanStage`，
   删除后处理实现与其专用缓存层。
   **附带修复 ①**：tension 因此在导出链路自动生效（无需额外改动，但必须有回归测试钉住）
3. `fix(render): unify breath curve out-of-range semantics`
   —— `chain.rs::sample_curve_at_abs_sec` 越界改为 hold-last，改写对应测试

分支：`feature/hifisampler-param-parity`（基于 `origin/develop` @ `2e40fe71`）

### 验收标准

- `cargo test` 与 `npm test` 全绿
- 新增的 10 个单元测试与 3 个回归测试通过
- 手工验证：同一工程分别导出与预览，`hifigan_tension` 与 `breath_gain`
  的听感一致（修复前 tension 导出完全缺失）
- 手工验证：`formant_shift_cents` 从 0 拉到 +1200，人声共振峰**上移**（变细），
  拉到 −1200 **下移**（变粗）—— 方向不得相反
