# KARA2 主唱 / 背景和声分离

公开版本 0.25，内部更新 037。输入为已经分离出的单声道或双声道人声；原唱与伴奏的完整混音应先做人声分离。

## 使用

模型与离线处理工具位于 `engines/hamood`。命令行入口：

```bat
engines\hamood\kara2.cmd --input "人声.wav" --output-dir "分离结果"
```

默认 CPU，可通过 `--start 45 --duration 22` 测试片段。输出 `lead.wav`、`backing.wav` 和 `separation.json`，均保留相同时间起点与长度。输出为 44.1 kHz 双声道浮点 WAV，避免独立归一化或削波改变主唱 / 和声的相对音量。已有结果不会覆盖，应使用新的输出目录。

`--check` 验证模型哈希与运行库；`--provider directml` 可手动使用已有 DirectML，默认 CPU 为已验证路径。GPU 的速度与结果需按设备单独验证。

此更新提供本地分离工具与模型配置，尚未添加编辑器菜单或新的 MCP 分离命令。可将输出 WAV 导入伴奏 / 参考轨继续分析。模型可能保留混响、叠唱或主唱泄漏，不能将输出直接等同于准确的和声乐谱。

## 文件与来源

- 原版 `UVR_MDXNET_KARA_2.onnx`：52,786,726 字节。
- SHA256：`bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4`。
- 权重：https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models
- 原项目与推理实现：https://github.com/Anjok07/ultimatevocalremovergui
- 模型转换方的输入输出说明：https://huggingface.co/musetric/uvr-mdxnet-kara2-onnx （本包使用原版权重，未包含转换版）。

STFT / iSTFT 与 UVR MDX 流程一致；使用 5120 FFT、1024 hop、2048 频点、256 帧分块。返回的背景声已乘 1.065 补偿，主唱为输入减去背景声；两者相加还原输入。权重只从随包固定路径读取，运行时不联网。

## 完整配布范围

包括编辑器、WCSNDM 0.0803、Python / CPU PyTorch、DiffSinger 运行库与发音资源、NSF-HiFiGAN ONNX / WCSNDM HF 模型、Beat This! small0、BTC 170 类和 KARA2。

复用一份随包 Python，HAMOOD 配置使用相对路径。音源库、用户工程、试听音频、工程缓存、源码 / 构建目录、历史备份、独立的 UniverSR 实验环境不属于此编辑器配布包。Basic Pitch / CREPE 原唱分析实验也未接入编辑器，不计入本包。
