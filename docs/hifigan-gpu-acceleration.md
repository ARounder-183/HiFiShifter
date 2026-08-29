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
| HNSEP | wav `[1,220500]`（5 s） | 310 ms | 304 ms | 1.02x | 完全一致 |
| HNSEP | wav `[1,441000]`（10 s） | 553 ms | 531 ms | 1.04x | 完全一致 |
| HNSEP（`All` 计算单元，5 s） | — | 310 ms | 304 ms | 1.02x | — |
| HNSEP（`CPUAndNeuralEngine`，5 s） | — | 306 ms | 301 ms | 1.02x | — |

- **FCPE 本来就已经在跑 CoreML。** `ep_choice_for_role()` 对 PitchDetector 同样返回
  CoreML，而且因为 `pinned = matches!(role, Vocoder)`，它一直用的是**未固定维度**配置
  ——恰好就是实测最快的那一套。本次没有改它的 EP 策略，只是把 EP 纳入了状态上报。
- **HNSEP 上 GPU 基本无收益（1.5%~4%），保持 CPU 默认。** 三个理由：
  1. 各种 CoreML 计算单元都试过，提升都在噪声范围内；
  2. CoreML 会话创建额外要 0.6~1.2 s（CPU 只要 ~50 ms），而 HNSEP 结果有 clip 级
     LRU 缓存、每个 clip 只跑一次，编译成本摊不掉；
  3. 与声码器争抢 GPU 资源会拖慢真正的热点路径。
- **顺带修掉一个真 bug：** 原代码里 `ep_choice_for_role()` 把 Separator 的
  `return "cpu"` 写在环境变量判断**之前**，导致 `HIFISHIFTER_HNSEP_ORT_EP` 这个
  环境变量完全失效（死代码）。现在改成「按角色默认值兜底」，显式指定优先，
  需要的人可以用 `HIFISHIFTER_HNSEP_ORT_EP=coreml` 把 HNSEP 手动放到 GPU 上。

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
