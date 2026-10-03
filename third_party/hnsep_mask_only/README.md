# hnsep_240512 (mask-only 导出)

本目录是**导出说明与许可**，即"这个模型是什么、从哪来、怎么导出"。
**模型二进制不在这里**（原因见下），运行时用的是
`backend/src-tauri/resources/models/hnsep/hnsep.onnx`。

## 本目录为什么不存模型二进制

这里曾放一份 `model.onnx`，它与运行时使用的那份**逐字节相同**（同一个 sha256），
即同一份 56 MiB 数据在仓库里存了两遍，而**只有 `resources/` 那份会被加载**：

- 模型路径解析（`vocoder/hnsep_onnx.rs::resolve_model_path`）只查
  `HIFISHIFTER_HNSEP_ONNX`、`HIFISHIFTER_HNSEP_MODEL_DIR` /
  `hnsep_model_dir()`、`resources/models/hnsep/`（或可执行文件同级
  `models/hnsep/`）——**从不查 `third_party/`**；
- 全仓库代码、CI 与脚本对本目录**零引用**。

因此删掉重复副本，只保留说明文档。二进制可校验信息见下方「来源与许可」，
需要时可按 sha256 重新获取并做校验。

## 模型标识

```
sha256     0b46499d71799a3b47a060997f26e94c513776461bbbf9d592a4b5af8dc8a80c
大小       59,046,980 字节
对应文件   resources/models/hnsep/hnsep.onnx（运行时加载的那一份）
```

`model.onnx` —— **谐波/噪声分离模型，频谱域（mask-only）**。

## 为什么用这一版

本仓库原先用的是 `third_party/vocal-remover/model/hnsep_240512_vr/hnsep.onnx`
（波形域版）：ONNX 图里**包含 STFT、编码器、24 个 LSTM、解码器与 ISTFT**，
输入输出都是波形 `[1, N]`。

那份模型在 GPU 上几乎没有收益（实测 CoreML 相对 CPU 仅 1.02~1.04x，见
`docs/hifigan-gpu-acceleration.md`），原因是：

1. **24 个 LSTM 是串行递归结构**，逐帧依赖，GPU 无法并行化；
2. **STFT/ISTFT 在图内**（以 ConvTranspose 实现），这些算子要么不被
   CoreML/DirectML 支持而回退 CPU，要么把图切成大量碎片，kernel launch
   开销吃掉收益。

本版把 STFT/ISTFT 移出 ONNX（与原版 `export.py` 中 `CascadedNetONNX._forward`
的拆分一致：`_forward` 只做 mask，`forward` 才包 STFT/ISTFT），
**Rust 侧只跑 mask 网络**。这样 GPU 上剩下的是纯卷积/LSTM 网络，
与 OpenUtau 的做法一致（其 `Hnsep.cs` 也是「STFT 在宿主、只有网络在 ONNX」）。

## 网络是同一个

LSTM 节点数两版均为 **24**，`stg1/stg2/stg3/out` 结构与权重一致 ——
区别仅是 I/O 边界。因此分离质量不应改变，但**必须实测验证**。

## 参数（`hnsep.yaml`）

```
sample_rate: 44100
n_fft:       2048
hop_length:  512
```

与本仓库现有 HNSEP 配置逐字一致。

## I/O 形状

```
输入  spectrum  [B, 2 (re, im), 1025, T]   T 必须是 32 的倍数
输出  mask      [B, 2 (re, im), 1025, T]
```

谐波 = `istft(spectrum * mask)`；噪声 = 原信号 − 谐波。

## 来源与许可

- 上游网络：`yxlllc/vocal-remover`（MIT），源自 `tsurumeso/vocal-remover`
- 本 mask-only 导出：OpenUtau 依赖包 `hnsep_240512`，取自
  `https://github.com/stakira/vocal-remover/releases/download/hnsep_240512-oudep/hnsep_240512.oudep`
- sha256（.oudep 归档）：`8f8c1046d0cacd40363e6f71d48bc1663e06a01d4625af0569055648d771ff92`
- 许可见同目录 `LICENSE`（MIT, Copyright (c) 2019 tsurumeso）
