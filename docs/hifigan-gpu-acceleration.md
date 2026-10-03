# HiFiGAN GPU 加速：根因分析与修复记录

本文记录 `NSF-HiFiGAN` 声码器在 macOS/Apple Silicon 上「GPU 加速无法使用」的排查过程、
实测数据和最终修复方案。同时覆盖 FCPE（音高检测）与 HNSEP（谐波/噪声分离）两个模型的
GPU 收益评估。

所有数字均在 **Apple Silicon（8 核，M 系列）/ macOS / ONNX Runtime 1.28.0** 上实测，
模型为 `pc_nsf_hifigan_coreml.onnx`（56 MB）、`fcpe.onnx`（43 MB）、`hnsep.onnx`（93 MB）。
`rtf` = 实时倍率，即「生成的音频时长 ÷ 推理耗时」，越大越好。

---

## 1. 症状

- 在「推理设备」里选择 GPU 后，渲染速度不升反降；
- 内置基准测试显示 GPU 比 CPU 慢；
- 菜单里的「推理设备」标签恒为 `GPU (CoreML)`，无法判断底层到底跑在哪个后端。

## 2. 根因

`ort_session.rs` 为了让 CoreML EP 能编译 NSF-HiFiGAN 的动态图，给会话做了**维度固定**：

```rust
.with_dimension_override("time", 4096)
.with_dimension_override("batch", 1)
.with_static_input_shapes(true)   // 仅 Vocoder 角色
```

这带来三个后果：

1. **静态编译出来的 CoreML 图比 CPU 还慢。** 固定 `time=4096` 后，CoreML EP 会把
   整张图编译成一个全静态的 MLProgram，实测反而比 CPU EP 慢 1.65 倍。
2. **每个分块都被补齐到 4096 帧。** `nsf_hifigan_onnx.rs` 里的 `session_time_frames()`
   会把所有输入 pad 到 4096 帧再裁回，短片段大量空算。
3. **ONNX Runtime 1.28 已经不需要这个 workaround。** 实测动态 shape 的 CoreML 会话
   能正常编译，输出与 CPU 逐样本一致，而且快得多。

也就是说：**让 GPU「能跑起来」的那个补丁，正是让 GPU「快不起来」的原因。**

## 3. 实测数据

### 3.1 NSF-HiFiGAN，4096 帧（≈47.6 s 音频）

| 配置 | 中位耗时 | rtf | 输出与 CPU 参考对比 |
|---|---|---|---|
| CPU（应用原配置） | 6187 ms | 7.69x | 参考基准 |
| **CoreML：原配置（固定 time=4096）** | **10230 ms** | **4.65x** | 正确（`max_abs ≤ 2e-5`），但比 CPU 慢 1.65 倍 |
| **CoreML：不固定维度（动态 time）** | **801 ms** | **59.34x** | 正确（`max_abs ≤ 2e-5`），比 CPU 快 7.7 倍 |
| CoreML：固定 + `static_shapes=false` | — | — | 直接失败 |
| CoreML：`CPUAndNeuralEngine` + 固定 | 6895 ms | 6.90x | 与 CPU 持平 |
| WebGPU（mem_pattern 开 / 关） | — | — | 全部失败：`{1,4096,1} != {1,2097152,1}` 缓冲区复用冲突 |

### 3.2 不固定维度后的分块长度扫描

| 帧数 | 输出样本数 | rtf | 输出 RMS |
|---|---|---|---|
| 128 | 65 536 | 56.34x | 0.4308 |
| 256 | 131 072 | 57.44x | 0.4307 |
| 512 | 262 144 | 57.44x | 0.4307 |
| 1024 | 524 288 | 57.38x | 0.4306 |
| 2048 | 1 048 576 | 58.19x | 0.4306 |
| 4096 | 2 097 152 | 58.34x | 0.4305 |

线性、稳定、输出一致，无需任何 padding。

### 3.3 CPU 侧单因子隔离（1024 帧）

早期的矩阵测试把多个参数混在一起改，导致归因错误。重新做单因子隔离后：

| 配置 | intra_threads | memory_pattern | parallel_execution | 中位耗时 | rtf |
|---|---|---|---|---|---|
| A（应用原配置） | cores/2 | ON | ON | 1586 ms | 7.49x |
| **B** | **cores** | ON | ON | **1202 ms** | **9.89x** |
| C | cores/2 | OFF | ON | 1508 ms | 7.89x |
| D | cores/2 | ON | OFF | 1557 ms | 7.64x |
| E | cores | OFF | ON | 1098 ms | 10.82x |

结论：主因是 `intra_threads`（−24%），**不是** `parallel_execution`。
`memory_pattern=OFF` 还能再快约 9%，但会让 ORT 放弃缓冲区复用、增加分配抖动，
本次没有采用。

## 4. 其他模型：FCPE 与 HNSEP

| 模型 | 输入规模 | CPU | CoreML | 加速比 | 输出一致性 |
|---|---|---|---|---|---|
| FCPE | mel `[1,1000,128]`（10 s） | 22.7 ms | 7.7 ms | **2.95x** | `max_abs` 1e-6 |
| FCPE | mel `[1,4000,128]`（40 s） | 79.2 ms | 27.4 ms | **2.89x** | `max_abs` 5e-6 |
| HNSEP（**波形域旧模型**） | wav `[1,220500]`（5 s） | 310 ms | 304 ms | 1.02x | 完全一致 |
| HNSEP（**波形域旧模型**） | wav `[1,441000]`（10 s） | 553 ms | 531 ms | 1.04x | 完全一致 |
| HNSEP（旧模型，`All` 计算单元，5 s） | — | 310 ms | 304 ms | 1.02x | — |
| HNSEP（旧模型，`CPUAndNeuralEngine`，5 s） | — | 306 ms | 301 ms | 1.02x | — |

> ⚠ 上表是**波形域旧模型**的数据，已被下面的 mask-only 模型取代，见 §4.1。

- **FCPE 本来就已经在跑 CoreML。** `ep_choice_for_role()` 对 PitchDetector 同样返回
  CoreML，而且因为 `pinned = matches!(role, Vocoder)`，它一直用的是**未固定维度**配置
  ——恰好就是实测最快的那一套。本次没有改它的 EP 策略，只是把 EP 纳入了状态上报。
- **旧模型上 GPU 基本无收益（1.5%~4%）。** 原因是**模型形态**而非 EP 配置：
  旧模型是波形域版，ONNX 图内含 STFT + 编码器 + 24 个 LSTM + 解码器 + ISTFT。
  LSTM 串行递归不可并行，图内的 STFT/ISTFT（ConvTranspose）又要么不被 EP 支持
  而回退 CPU，要么把图切成碎片。已换用 mask-only 模型，见 §4.1。
- **顺带修掉一个真 bug：** 原代码里 `ep_choice_for_role()` 把 Separator 的
  `return "cpu"` 写在环境变量判断**之前**，导致 `HIFISHIFTER_HNSEP_ORT_EP` 这个
  环境变量完全失效（死代码）。现在改成「按角色默认值兜底」，显式指定优先，
  需要的人可以用 `HIFISHIFTER_HNSEP_ORT_EP=coreml` 把 HNSEP 手动放到 GPU 上。

### 4.1 HNSEP 换用 mask-only 模型（频谱域）

旧模型是**波形域**版：ONNX 图内含 STFT、编码器、**24 个 LSTM**、解码器与 ISTFT，
输入输出都是波形 `[1, N]`。这解释了 §4 里 GPU 无收益的观测 —— 瓶颈是模型形态：

1. **LSTM 是串行递归结构**，逐帧依赖，GPU 无法并行化；
2. **STFT/ISTFT 在图内**（以 ConvTranspose 实现），这些算子要么不被 CoreML/DirectML
   支持而回退 CPU，要么把图切成大量碎片，kernel launch 开销吃掉收益。

已换用 mask-only 导出（`third_party/hnsep_mask_only/`）：ONNX 里**只有 mask 网络**
（输入 `spec [1,2,1025,T]`、输出 `mask [1,2,1025,T]`），STFT/ISTFT 回到 Rust
（`vocoder/hnsep_dsp.rs`）。这与 OpenUtau 的做法一致，其 `Hnsep.cs` 同样把 STFT
留在宿主、只把网络交给 ONNX。

**等价性已验证**（同一 1 秒合成信号，两模型各自的 harmonic 输出）：

```
|h_old - h_new| RMS / input RMS = 0.0000   (-122.9 dB)
correlation(h_old, h_new)      = 1.000000
[old] h + n 残差                = 0.000e0
```

差值在浮点噪声量级，LSTM 节点数两版均为 24 —— 是**同一个网络**，只是 I/O 边界不同。
模型体积也从 88 MB 降到 56 MB。

**分段计时**（release，5 s 音频，448 帧）：

| 阶段 | 耗时 |
|---|---|
| STFT（Rust） | 6.3 ms |
| mask 网络（CPU） | 255.6 ms |
| ISTFT（Rust） | 5.6 ms |

DSP 只占约 4.5%，成本几乎全在网络 —— 所以"把 STFT 移出图外"本身不省时间，
它的价值是**让 EP 能接手那个网络**。

**CoreML vs CPU 随输入长度变化**（release，仅网络推理）：

| 音频时长 | CPU | CoreML | 加速比 |
|---|---|---|---|
| 2 s | 154.9 ms | 119.6 ms | 1.29x |
| 5 s | 368.1 ms | 211.6 ms | 1.74x |
| 10 s | 698.9 ms | 368.2 ms | 1.90x |
| 20 s | 1378.8 ms | 681.2 ms | 2.02x |
| 30 s | 2130.0 ms | 880.7 ms | 2.42x |
| **45 s** | 3356.7 ms | 1640.4 ms | **2.05x ← 拐点** |
| **50 s** | 3731.9 ms | 4953.2 ms | **0.75x（反而更慢）** |
| 60 s | 4377.2 ms | 6749.4 ms | 0.65x（更慢） |

两点结论：

1. **短音频上 GPU 收益有限**（2 s 时仅 1.29x），因为 CoreML 每次推理有固定开销，
   而 HNSEP 有 clip 级缓存、每个 clip 只跑一次，固定开销摊不掉。
2. **约 45~50 s 处有拐点，超过之后 CoreML 反而显著更慢**（50 s 时是 CPU 的 0.75x，
   60 s 时 0.65x）。这是 CoreML 在长输入上的已知退化行为，**不是噪声**：
   45 s 与 50 s 各复测两次，结果稳定（45 s: 1625/1640 ms；50 s: 4953/5064 ms）。

因此**不能简单把 HNSEP 默认切到 GPU**。当前策略保持 `Separator` 默认 CPU
（见 `default_ep_for_role`），需要的人可用 `HIFISHIFTER_HNSEP_ORT_EP=coreml` 手动开启
—— 该路径在 45 s 以内的 clip 上确有约 2x 收益。

> **待办**：若要默认启用 GPU，需要一个按输入长度选择的策略（例如 >40 s 走 CPU），
> 或先查清 CoreML 长输入退化的根因。
>
> **后续已查明（见下方「HNSEP 是整段推理」一节）**：更可能的根因是
> **HNSEP 不做分块**，而非 CoreML 本身的缺陷 —— HiFiGAN 恰好在 4096 帧
> （≈47.6s）处分块，因此 CoreML 从不接触超长输入；HNSEP 整段推理，
> 于 ~45–50s 之后撞上长序列退化。两个阈值吻合。

#### HNSEP 是整段推理（不分块），与 HiFiGAN 不同

**结论：HNSEP 不分块，长 clip 会整段一次性推理。** 这不是本次改动引入的 ——
旧波形域模型同样把整段波形塞进一次 `Run`（`Tensor::from_array([1, len])`）。

对比两者的分块策略：

| | HiFiGAN | HNSEP |
|---|---|---|
| 分块 | ✅ `CHUNK_MAX_FRAMES = 4096` 帧（≈47.6s @ hop 512） | ❌ 无，整段 |
| 重叠 | ✅ `HIFISHIFTER_ONNX_OVERLAP_SEC`（默认 0.1s）线性 crossfade | — |
| 分块级缓存 | ✅ `chunk_cache_get/put`，可只重渲染脏块 | ❌ 只有整段级缓存 |
| 可配置 | ✅ `HIFISHIFTER_ONNX_CHUNK_SEC` | ❌ 无 |

**为什么没有顺手给 HNSEP 也加上分块：它不安全。** 模型里的 LSTM 是
**双向（bidirectional）** 的 —— 直接读 ONNX 属性可确认：

```
/stg1_low_band_net/lstm_dec2/lstm  direction = "bidirectional"
共 10 处 direction 属性，其中 5 处为 bidirectional
```

双向 LSTM 会**从后往前**扫过整个序列，因此每一帧的输出都依赖**它之后的全部帧**。
分块会切断这个反向依赖。

实测验证（12s 音频，CPU，把后半段单独推理后与整段对比 mask）：

| 比较位置 | max\|Δmask\| | mean\|Δmask\| |
|---|---|---|
| 后半段**首帧**（跨段边界） | 1.0132 | 0.0467 |
| 后半段**末帧**（远离边界） | 0.3803 | 0.0466 |

**远离边界处差异依然显著**（0.38），这正是双向依赖的特征 —— 若是普通前向 LSTM，
边界远处应当基本一致。因此分块会造成可闻的分离错误，**不能直接照搬 HiFiGAN 的方案**。

**长 clip 的实际代价（结构性分配，实测）：**

| 时长 | 帧数 | mask 输入 | 频谱 | OLA | 合计 |
|---|---|---|---|---|---|
| 30s | 2,592 | 21.3 MB | 42.5 MB | 21.3 MB | **85 MB** |
| 60s | 5,184 | 42.5 MB | 85.0 MB | 42.5 MB | **170 MB** |
| 180s | 15,520 | 127.3 MB | 254.5 MB | 127.2 MB | **509 MB** |
| 300s | 25,856 | 212.0 MB | 424.0 MB | 211.8 MB | **848 MB** |

线性增长。频谱占大头（`Complex<f64>` = 16 B/点，是 mask 的 2 倍）。
分离缓存同样是**整段**粒度，默认容量 128 条：

| 时长 | 单条（harmonic+noise） | 128 条上限 |
|---|---|---|
| 1 min | 21.2 MB | 2.7 GB |
| 5 min | 105.8 MB | 13.5 GB |
| 10 min | 211.7 MB | 27.1 GB |

**这很可能解释了 §4.1 的 CoreML 拐点。** HiFiGAN 恰好在 4096 帧（≈47.6s）处分块，
所以 CoreML 从不接触超长输入；HNSEP 不分块，于是 ~45–50s 之后撞上长序列退化。
两个数字吻合到这种程度不像巧合：

> **修正**：§4.1 我最初把 CoreML 在 50s 后变慢归因于"CoreML 对超长序列的已知退化"。
> 更可能的解释是**输入长度本身**（不分块），而非 CoreML 的问题 ——
> 也就是说在 CPU 上同样存在随长度增长的非线性劣化风险，只是 CoreML 更早暴露。

**可行的改进方向（未实施，工作量较大）：**

1. **重叠分块 + crossfade**：按帧率而非音频长度分块，块间给足重叠（需覆盖双向
   LSTM 的有效感受野，可能要数秒），重叠区线性 crossfade。代价是算力成倍增加，
   且需要实测确定"感受野多大才够"，风险是仍然听得出接缝。
2. **换非双向模型**：若存在单向 LSTM 或纯卷积的同类模型，分块即可安全且低成本。
3. **降内存**：`spectrum` 用 `Complex<f32>`（省一半），或分块只为省内存而用
   "STFT 整段 + mask 分段 + 各段独立 ISTFT 后 OLA"——但双向依赖问题依旧。

**当前建议**：保持整段推理（与旧版行为一致，正确性优先）。若遇到超长 clip 的
内存或耗时问题，优先用**缩短 clip 长度**规避，而不是改分块。

#### DirectML（Windows）注意事项

Windows 侧的 GPU 路径是 **DirectML**（`or` 平台条件编译，无法在 macOS 上运行验证）。
换模型时审查出两处**平台相关**问题，均已修：

1. **烟测探针张量膨胀 2000 倍。** `SMOKE_TEST_FRAMES = 4096` 原本是为声码器的
   1-D `time` 轴选的，但烟测会把**每个**动态维都替换成该值。HNSEP 的 mask-only
   输入是 `[batch, 2, 1025, n_frames]`，于是探针会变成 `[1, 2, 1025, 4096]`
   = **32 MB**（旧波形域模型只要 16 KB）。已改为按角色取值
   （`smoke_probe_frames`：Separator 用 32 帧 = 一个 segment；声码器/音高检测沿用
   4096），并加了三条回归测试防止再次膨胀。
   注意：HNSEP 默认走 CPU，而 **CPU 路径不做烟测**，所以这个膨胀只在用户显式
   `HIFISHIFTER_HNSEP_ORT_EP=directml` 时才会发生 —— 属于"手动开 GPU 才踩到"的坑。

2. **维度覆盖按名字匹配，对 HNSEP 是 no-op。** DirectML 构建器里：
   ```rust
   .with_dimension_override("batch", 1)                        // 按维度名
   .with_dimension_override_by_denotation("time", 4096)        // 按 denotation
   ```
   实测三个模型的导出维度名：

   | 模型 | 维度名 |
   |---|---|
   | NSF-HiFiGAN | `batch`, `time` |
   | FCPE | `batch`, `time` |
   | **HNSEP（mask-only）** | **`batch_size`, `n_frames`** |

   对 HNSEP 两条覆盖都不匹配 —— 不报错，但也固定不住任何维度（DML 仍走通用
   kernel，性能略差）。已把 `time` 覆盖**按角色门控**只用于声码器/音高检测：
   这既如实反映"该优化不适用于 HNSEP"，也避免将来 HNSEP 维度若被改名成 `time`
   时，把 4096 静默套到一个真实帧数与 32 整除约束都无关的 4-D 输入上。

**Windows 上仍需实测确认的一点**：DirectML 的 **strict 模式**
（`with_disable_cpu_fallback()`）要求**图里每个算子都被 DML 支持**，否则会话创建
直接失败。HNSEP 图含 `LSTM`，这一算子在部分 DML 版本上支持情况不一 ——
旧波形域模型同样含 LSTM，所以这不是本次引入的风险，但换模型后**应回归确认**
`HIFISHIFTER_HNSEP_ORT_EP=directml` 能建会话；失败时既有逻辑会回退
（strict 失败 → 非 strict 重试 → 仍失败则 CPU），不会卡住渲染。

## 5. 修复内容

### P0 — 让 GPU 真正可用

| 文件 | 改动 |
|---|---|
| `vocoder/ort_session.rs` | 移除 macOS ARM64 分支的 `with_dimension_override("time"/"batch")`；`build_coreml_ep()` 的 `with_static_input_shapes` 恒为 `false`；删除 `COREML_FIXED_TIME_FRAMES` / `coreml_active` / `set_coreml_pinned` / `reset_coreml_pinned_state` 状态机 |
| `vocoder/nsf_hifigan_onnx.rs` | 删除 `session_time_frames()`；`run_model()` 不再做 4096 补齐与裁剪 |

### P1 — 让「GPU 是否生效」可观测

| 文件 | 改动 |
|---|---|
| `vocoder/nsf_hifigan_onnx.rs` | `ACTIVE_EP` 由 `OnceLock<String>` 改为 `RwLock<String>`，切 EP 后能反映真实后端；新增 `active_backend_name()` |
| `state.rs` | `runtime_info()` 的 `gpu_backend` 改为读 `active_backend_name()`，不再返回编译期硬编码常量 |
| `models.rs` / `types/api.ts` / `MenuBar.tsx` | 字段注释与菜单文案改为「实际生效的后端」；`auto` 模式也显示真实后端 |
| `vocoder/nsf_hifigan_onnx.rs` | 基准测试的 GPU 候选按平台枚举（macOS: `coreml` → `webgpu`；Linux: `webgpu`），逐个尝试并回报实际生效者。原实现只认 `WebGpuExecutionProvider`，WebGPU 探测一失败就整个跳过 GPU 基准 |

### P2 — 性能与健壮性

| 文件 | 改动 |
|---|---|
| `vocoder/ort_session.rs` | 新增 `cpu_intra_threads()`：macOS ARM64 用满核（实测 −24%），Windows/Linux 保持 `cores/2` 以避免大小核 / NUMA 上的超订阅劣化 |
| `commands/ui_settings.rs` | 新增 `apply_ort_ep_settings()` 去重：`get_ui_settings()` 是读路径且调用频繁，原先每次读取都会销毁重建全部 ORT 会话；同时修掉「只改 DirectML 设备 ID 不会重建会话」的问题 |
| `vocoder/nsf_hifigan_onnx.rs` | `run_model_batch()` 只在批量内所有条目**等长**时才走批量推理（见下） |

### 关于批量推理的等长约束（重要）

`run_model_batch()` 把不同长度的分块零填充到最长长度后一起推理。实测发现这样会**改变
有效区域内的输出**：

| 批量场景 | 相对逐条推理的 `rel_l2` |
|---|---|
| 等长批量（4 × 1024 帧） | 3e-6（一致） |
| 非等长批量（256 帧填充到 1024） | **0.086（不一致）** |

原因是模型的 f0 source-generator 子图横跨整个时间轴，尾部补零会改变结果。
且 **CPU 与 CoreML 的偏差完全相同**，说明这是模型本身的性质，与执行后端无关。

修复前 macOS 走的是逐条推理（因为 CoreML 会话被固定维度），Linux/Windows CPU 走的是
带填充的批量推理。改为等长才批量之后：

- 输出在所有平台上与逐条推理一致；
- 常见的等长分块场景仍然走批量快路径；
- GPU 在 4096 帧上是计算密集而非调度密集，逐条调用的额外开销可以忽略。

## 6. 修复前后对比

内置基准测试（`--benchmark`），1024 帧：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| CPU rtf | 7.89x | **10.76x** |
| GPU (CoreML) rtf | 4.14x（比 CPU 慢 1.9 倍） | **60.07x**（比 CPU 快 5.6 倍） |
| `gpuAvailable` | 依赖 WebGPU 探测，易误判 | 按平台候选逐个探测 |
| `gpuBackendName` | — | `CoreML` |
| 菜单显示设备 | 恒为编译期常量 | 实际生效后端 |

> 注：修复前 GPU 的 4.14x 是在 4096 帧下测得，修复后是 1024 帧；rtf 与帧数无关，
> 因此可以直接比较。

## 7. 复现与验证方法

排查时使用的临时探针（`backend/src-tauri/examples/` 下，验证完成后已删除）：

- `ort_gpu_probe.rs` — EP 配置矩阵 + 输出正确性校验（对比 CPU 参考的 `rel_l2`）
- `ort_model_probe.rs` — 通用模型探针，用于 FCPE / HNSEP 的元数据读取与性能测量
- `ort_batch_check.rs` — 批量推理 vs 逐条推理的一致性校验
- `ort_bench_check.rs` — 直接调用应用内的 `run_vocoder_benchmark_cli()`

运行应用内基准：

```bash
cd backend/src-tauri
HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR=target/debug/models/nsf_hifigan \
  cargo run --bin HiFiShifter -- --benchmark
```

## 8. 跨平台注意事项

- **Windows（DirectML）**：完整的 `build_dml_session_inner()` 仍会固定 `batch=1` 与
  `time=4096`，未做改动。DirectML 的 `batch_pinned_to_one` 检测与逐条推理路径保持原样。
  本次没有在 Windows 上实测，因此保守地保留了原有策略。
- **Linux（WebGPU）**：EP 优先级与会话构建逻辑未变，仅新增了批量推理的等长约束
  与 CPU 线程数策略（Linux 仍是 `cores/2`）。
- **macOS ARM64**：CoreML 为主力路径，不再固定维度；`intra_threads` 用满核。
- 所有平台共享的 `SMOKE_TEST_FRAMES`（GPU 会话创建后的冒烟测试长度）保持 4096，
  避免 ORT 缓冲区复用优化与模型固定中间形状冲突。
