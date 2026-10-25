# GAME 默认识别（0.25 内部更新 060）

录音导入默认使用 GAME Medium + FCPE。设置 → 算法中的 GAME 和 FCPE 路径可以留空；程序自动查找 EXE 旁的 `models/game/medium` 与 `models/fcpe/fcpe.onnx`。模型类型默认 `medium`，也保留 `large` / `small` 自定义模型支持。

旧设置保存过 `large` 但未指定路径、且包内没有 Large 模型时自动采用 Medium。明确的自定义路径、可用的 Large 模型及 Small 选择保留。设置窗口保存后会写入实际默认选择。

GAME 负责音符分割和逐音符音高；FCPE 负责连续 F0，包括滑音、颤音。此默认设置用于新导入 / 主动重新分析的录音。打开已有工程沿用工程中已有的音符与编辑结果，DS 的 MIDI 唱法预测不受影响。

模型缺失或推理失败时使用内置 native-hq；命令行 `--inspect-analysis` 和 MCP `analysis_status` 可核对实际路径、模型档位与后端。仅文件就绪状态不能代替实际推理验证。

## 模型来源

- GAME：OpenVPI 官方 v1.0.3 `GAME-1.0.3-medium-onnx.zip`，https://github.com/openvpi/GAME/releases/tag/v1.0.3 。下载大小 179775226 字节；SHA-256 `5b7a21e64c6310efac399f5d12838fffa70565be162436b5a4a65f290721e7d8`。模型采用 CC BY-NC-SA 4.0；代码 MIT。配布时保留作者、来源和许可，模型权重需遵守非商业及相同方式共享条款。
- FCPE：CN_ChiTu 的 FCPE，使用 HiFiShifter 项目提供的 ONNX，https://github.com/ARounder-183/HiFiShifter/blob/main/backend/src-tauri/resources/models/fcpe/fcpe.onnx 。SHA-256 `013b507acd406de122b61ee497300ac6be082511101bc2136b886e48d9737e5d`；随模型保留 FCPE 与 HiFiShifter 的 MIT 许可。

模型目录的 `model-info.json` 记录来源和校验信息；原始 GAME ONNX 内容未修改。CMake 构建会将源码根目录已准备的 `models/game/medium` 与 `models/fcpe` 复制到 EXE 旁。源码归档如不含被忽略的权重，需要按上述来源补齐。

## 验证

使用 `--smoke-game-defaults <输出目录>` 核对便携发现、默认值、旧配置兼容、自定义选择与保存重开；使用 `juce/tests/game_fcpe_smoke.py` 验证标准 220 Hz 输入的 FCPE 输出与实际人声的 GAME + FCPE 推理。验证报告保存在 `test-output/update060`。
