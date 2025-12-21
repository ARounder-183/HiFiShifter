# Vocal Shifter 模块文档

本软件包提供了一个用于音高修正和合成的 GUI 工具，基于神经声码器（特别是 NSF-HiFiGAN）。

## 主要功能

*   **音高编辑**: 在钢琴卷帘界面上直接绘制音高曲线。
*   **实时合成**: 使用修改后的音高曲线合成音频。
*   **工程管理**: 保存和加载工程文件 (`.hsp`)，包含音频路径、模型路径、音高编辑和参数设置。
*   **音频导出**: 将合成后的音频导出为 WAV 格式。
*   **长音频优化**: 自动将长音频分段，支持增量合成，显著提高编辑时的性能和响应速度。
*   **撤销/重做**: 支持音高编辑操作的完整撤销和重做功能。
*   **播放控制**: 空格键播放/停止（返回起点），支持时间轴点击定位。

## 模块结构

### 1. `vocal_shifter.audio_processor`

该模块处理所有与音频处理和模型推理相关的繁重工作。它将 PyTorch 和音频处理逻辑与 GUI 隔离开来。

**类: `AudioProcessor`**

*   **`__init__(self)`**: 初始化处理器，检测 CUDA 设备。
*   **`load_model(self, folder_path)`**: 加载模型检查点和配置。
*   **`load_audio(self, file_path)`**:
    *   加载音频文件并重采样。
    *   提取 Mel 频谱图和 F0 (音高)。
    *   **分段处理**: 基于静音检测自动对音频进行分段 (`segment_audio`)，以支持增量合成。
*   **`synthesize(self, f0_midi)`**: 合成完整音频。
*   **`synthesize_segment(self, segment_idx, f0_midi_segment)`**: 仅合成音频的特定片段，用于性能优化。

### 2. `vocal_shifter.widgets`

该模块包含专为钢琴卷帘界面定制的 PyQtGraph 组件。

*   **`CustomViewBox`**: 处理自定义鼠标交互（中键平移，Ctrl/Alt+滚轮缩放）。
*   **`PianoRollAxis`**: 在 Y 轴显示音名 (C4, D#4)。
*   **`BPMAxis`**: 在 X 轴显示音乐时间（小节-拍），支持点击定位播放头。

### 3. `vocal_shifter.main_window`

该模块包含主要的 GUI 应用程序逻辑。

**类: `VocalShifterGUI`**

*   **UI 组件**:
    *   **菜单栏**: 文件 (打开/保存/导出), 编辑 (撤销/重做), 播放控制。
    *   **控制栏**: 快速调整移调、BPM 和拍号参数。
    *   **钢琴卷帘**: 用于编辑音高的 `pg.PlotWidget`。
*   **状态管理**:
    *   `is_dirty`: 跟踪是否有更改需要重新合成。
    *   `segment_states`: 跟踪哪些音频片段需要重新合成。
    *   `undo_stack` / `redo_stack`: 管理编辑历史。
*   **交互逻辑**:
    *   **智能合成**: 仅重新合成被修改的音频片段，确保响应速度。
    *   **播放**: 使用 `sounddevice` 处理音频播放，并与光标同步。

### 4. `vocal_shifter.main`

*   **`main()`**: 入口点函数。

## 使用方法

运行应用程序：

```bash
python run_gui.py
```

或者作为模块运行：

```bash
python -m vocal_shifter
```

### 快捷键

*   **空格键**: 播放 / 停止 (返回起点)
*   **Ctrl + Z**: 撤销
*   **Ctrl + Shift + Z** / **Ctrl + Y**: 重做
*   **Ctrl + O**: 打开工程
*   **Ctrl + S**: 保存工程
*   **Ctrl + Shift + S**: 另存为工程
*   **鼠标左键**: 绘制音高
*   **鼠标右键**: 擦除 / 还原音高
*   **鼠标中键**: 平移视图
*   **Ctrl + 滚轮**: 缩放时间 (X轴)
*   **Alt + 滚轮**: 缩放音高 (Y轴)
