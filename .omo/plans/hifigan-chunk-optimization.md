# HiFi-GAN 长音频分块推理优化方案

> 参考：HachiTune 源码分析文档 (`hifigan_optimization.md` Section 3 + Section 5)
> 分支：`feat/hifigan-long-chunk-optimization`

---

## 一、现状分析

当前 `nsf_hifigan_onnx.rs` 已有分块推理 (`infer_pitch_edit_chunked`)，但存在以下瓶颈：

| 问题 | 影响 | HachiTune 做法 |
|------|------|---------------|
| **每块独立提取 mel** | 相邻块重叠区域 mel 计算 2 次，浪费 | 全音频提取 mel 一次，按块切片 |
| **`Tensor::from_array()` 每次分配新内存** | 推理热路径堆分配，延迟毛刺 | 预分配 scratch buffer，仅填充数据 |
| **时间维度分块（10s）** | 与 vocoder 帧率耦合不直观 | 帧维度分块（512 帧），与模型语义对齐 |
| **ONNX Session 未开图优化** | 算子融合/常量折叠未启用 | `ORT_ENABLE_ALL` + `EnableMemPattern` |
| **Crossfade 使用 sin/cos 等功率** | 虽好，但与参考实现不一致 | 线性 crossfade（简单且足够平滑） |

## 二、优化目标

仅优化**分块推理路径**，不改动：
- mel 提取算法本身
- 音高编辑/共振峰偏移逻辑
- 缓存层 / renderer 调度
- 异步推理（下一阶段）

## 三、改动清单

### 3.1 ONNX Session 配置增强 (`build_session_with_ep`)

参考 HachiTune Section 2.2，启用：
- `ORT_ENABLE_ALL` 图优化（算子融合、常量折叠、layout 优化）
- `EnableMemPattern()` 内存复用
- `EnableCpuMemArena()` CPU 大块预分配
- GPU 后端减少线程：`SetIntraOpNumThreads(1)`, `SetInterOpNumThreads(1)`
- CPU 后端：`num_threads = max(hardware_concurrency/2, 2)`

### 3.2 Pre-allocated Tensor Scratch Buffers

在 `NsfHifiganOnnx` 结构体中新增：

```rust
mel_scratch: Vec<f32>,          // mel 数据暂存（列主序）
f0_scratch: Vec<f32>,           // f0 数据暂存
mel_shape_scratch: Vec<i64>,    // [1, numMels, frames]
f0_shape_scratch: Vec<i64>,     // [1, frames]
```

新增方法 `infer_from_mel_f0_fast()`：直接使用预分配 buffer 填充数据后推理，避免 `Tensor::from_array()` 的每次分配。

### 3.3 帧级分块推理（新入口 `infer_pitch_edit_chunked_fast`）

```
全音频 → mel_from_audio_fast() 提取完整 mel 矩阵 [n_mels, T]
       → 构建 f0 向量 [T]
       → for chunk in chunks (512帧步进, 16帧重叠):
            mel_slice = mel[:, chunk_start..chunk_end]
            f0_slice = f0[chunk_start..chunk_end]
            waveform_chunk = run_model_fast(mel_slice, f0_slice)
            linear_crossfade(waveform_chunk, output)
       → 拼接返回
```

参数常量（匹配 HachiTune）：
- `CHUNK_MAX_FRAMES = 512`（≈5.9s @ 44100/512）
- `OVERLAP_FRAMES = 16`（≈186ms）
- `STEP = CHUNK_MAX_FRAMES - OVERLAP_FRAMES = 496`

### 3.4 Crossfade 改为线性

```rust
// 线性 crossfade（匹配 HachiTune reference）
for i in 0..overlap_samples {
    let t = i as f32 / overlap_samples as f32;
    out[dst + i] = prev[i] * (1.0 - t) + curr[i] * t;
}
```

### 3.5 Wire 到 Renderer

修改 `hifigan.rs` 的 `render_with_formant`，当输入为长音频时走新的快速路径。

## 四、不改动的部分

- ❌ 不添加 IoBinding / DirectML / CoreML 后端（后续阶段）
- ❌ 不添加异步推理管线（后续阶段）
- ❌ 不添加增量合成 / dirty region（后续阶段）
- ❌ mel 提取算法自身不变（`mel_from_audio_fast` 已经高效）

### 3.5 分块级缓存（新增）

利用现有 `SynthClipCache`（key = `clip_id + param_hash`，已含 `start_frame/end_frame`），
在 chunked 路径中对每个 chunk 独立缓存其推理后的 waveform。

```
for chunk in chunks (512帧步进):
    chunk_key = SynthClipCacheKey {
        clip_id,
        param_hash(chunk_start_frame, chunk_end_frame, ...)
    }
    if cache_hit(chunk_key):
        chunk_waveform = cached
    else:
        mel_slice = mel[:, chunk_range]
        f0_slice = f0[chunk_range]
        chunk_waveform = run_onnx(mel_slice, f0_slice)
        cache_insert(chunk_key, chunk_waveform_stereo)
    linear_crossfade(chunk_waveform, output)
```

**收益**：用户编辑部分 pitch 曲线时，只有覆盖到的 chunk 需要重渲染，其余 chunk 直接命中缓存。

### 3.6 Wire 到 Renderer

修改 `hifigan.rs` 的 `render_with_formant`：
- 短音频（≤512 帧）：保持现有逻辑（完整 per-segment 缓存）
- 长音频（>512 帧）：走新的分块+逐块缓存路径

## 四、不改动的部分

- ❌ 不添加 IoBinding / DirectML / CoreML 后端（后续阶段）
- ❌ 不添加异步推理管线（后续阶段）
- ❌ 不添加增量合成 / dirty region（后续阶段）
- ❌ mel 提取算法自身不变（`mel_from_audio_fast` 已经高效）
- ❌ 缓存系统本身不变（复用已有 `SynthClipCache` 和 `compute_param_hash`）

## 五、验证标准

- [ ] `cargo check` 通过
- [ ] lsp_diagnostics 无新增错误
- [ ] 功能：长音频（≥512帧）推断结果与旧路径一致（可接受微小浮点差异）
- [ ] 缓存：分块缓存命中时跳过 ONNX 推理
- [ ] 性能：长音频推断中 mel 只提取一次，后续重建只渲染脏 chunk
