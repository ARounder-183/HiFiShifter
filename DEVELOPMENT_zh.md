# HifiShifter 开发与使用手册

HifiShifter 是一个基于深度学习神经声码器（NSF-HiFiGAN）的图形化音高修正工具。它允许用户加载音频文件，在钢琴卷帘界面上直观地编辑音高曲线，并利用预训练的声码器模型实时合成修改后的音频。

## 1. 核心架构与工作原理

HifiShifter 采用了模块化的设计，将 GUI 交互、音频处理和数据管理分离。

### 1.1 系统架构图

```mermaid
graph TD
    GUI[Main Window (PyQt6)] -->|User Input| Processor[Audio Processor]
    GUI -->|Visuals| Widgets[Custom Widgets (PyQtGraph)]
    GUI -->|Data| Track[Track Object]
    
    Processor -->|Load/Infer| Model[NSF-HiFiGAN Model]
    Processor -->|Extract| Features[Mel & F0]
    
    Track -->|Store| AudioData[Waveform]
    Track -->|Store| PitchData[F0 Curve]
    Track -->|Store| State[Edit History]
```

### 1.2 核心模块详解

#### A. `hifi_shifter.audio_processor` (音频处理核心)
这是整个系统的引擎。它负责所有与 PyTorch 模型和音频信号处理相关的任务。
*   **模型加载**: 读取 `.ckpt` 和 `.yaml` 配置文件，初始化 NSF-HiFiGAN 生成器。
*   **特征提取**:
    *   **Mel 频谱图**: 使用 STFT 将波形转换为 Mel 频谱，作为内容的表征。
    *   **F0 (基频)**: 使用 Parselmouth (Praat) 算法提取原始音高。
*   **智能分段 (Segmentation)**: 为了实现实时编辑，长音频会被自动切分为多个片段（基于静音检测）。
    *   **增量合成**: 当用户修改音高时，只有受影响的片段会被重新送入模型合成，而不是整首歌曲。这使得编辑响应速度极快。
*   **合成 (Synthesis)**: 接收 Mel 频谱和修改后的 F0，输出波形。

#### B. `hifi_shifter.track` (数据模型)
每个音轨都是一个 `Track` 对象，封装了该音轨的所有状态：
*   **原始数据**: 原始音频波形、采样率、原始 F0、Mel 频谱。
*   **编辑数据**: 用户修改后的 F0 曲线 (`f0_edited`)。
*   **状态标志**: `muted` (静音), `solo` (独奏), `volume` (音量), `start_frame` (时间轴偏移)。
*   **缓存**: `synthesized_audio` 存储合成后的结果，避免重复计算。
*   **历史记录**: `undo_stack` 和 `redo_stack` 用于撤销/重做。

#### C. `hifi_shifter.timeline` (时间轴面板)
负责多轨管理和宏观视图。
*   **轨道管理**: 显示所有音轨的控制面板（静音、独奏、音量）和波形概览。
*   **时间控制**: 包含标尺（Ruler），控制播放头位置。
*   **交互**: 支持拖拽音轨进行时间对齐，支持框选和缩放。

#### D. `hifi_shifter.main_window` (主界面与逻辑)
协调所有组件的中央控制器。
*   **事件循环**: 处理播放定时器 (`QTimer`)，更新播放头位置。
*   **绘图逻辑**: 使用 `pyqtgraph` 绘制钢琴卷帘（Piano Roll）和波形。
*   **同步机制**: 确保时间轴面板和编辑面板的视图同步（或解耦）。
*   **工程管理**: 处理 `.hsp` 文件的序列化和反序列化。

#### E. `utils.i18n` (国际化支持)
负责多语言管理。
*   **资源加载**: 从 `assets/lang/` 目录加载 `.json` 语言文件。
*   **动态切换**: 通过 `config_manager` 保存语言设置，重启后生效。

## 2. 功能特性与使用指南

### 2.1 基础操作
1.  **加载模型**: 启动后首先点击“文件 -> 加载模型”，选择包含 `model.ckpt` 和 `config.json` 的文件夹。
2.  **导入音频**: 点击“文件 -> 加载音频”或直接拖入音频文件。
3.  **视图操作**:
    *   **中键拖动**: 平移视图。
    *   **Ctrl + 滚轮**: 缩放时间轴（横向）。
    *   **Alt + 滚轮**: 缩放音高轴（纵向）。
4.  **音高编辑**:
    *   **左键绘制**: 直接在钢琴卷帘上画线修改音高。
    *   **右键擦除**: 还原为原始音高。

### 2.2 高级功能
*   **多语言支持**: 在“设置 -> 语言”中切换中文或英文（需重启）。
*   **多轨混音**: 支持导入伴奏（BGM）和多条人声轨。在轨道左侧面板可以调节音量、静音和独奏。
*   **时间对齐**: 在时间轴面板（下方区域），可以拖动音轨块来调整其开始时间。
*   **参数设置**:
    *   **移调 (Shift)**: 全局调整音高（半音为单位）。
    *   **BPM / 拍号**: 设置项目的速度和节拍，影响网格线的显示。
*   **导出**: 将所有非静音轨道混合并导出为 WAV 文件。

## 3. 工程文件结构 (.hsp)

工程文件是标准的 JSON 格式，支持相对路径，方便在不同设备间迁移。

```json
{
    "version": "2.1",
    "model_path": "models/nsf_hifigan",  // 模型路径（支持相对路径）
    "params": {
        "bpm": 120.0,
        "beats": 4
    },
    "tracks": [
        {
            "name": "vocal_track",
            "file_path": "audio/vocal.wav", // 音频路径（支持相对路径）
            "type": "vocal",                // 类型: 'vocal' 或 'bgm'
            "shift": 0.0,
            "muted": false,
            "solo": false,
            "volume": 1.0,
            "start_frame": 0,               // 时间轴偏移量
            "f0": [ ... ]                   // 修改后的 F0 数据 (MIDI 编号数组)
        }
    ]
}
```

## 4. 二次开发指南

如果您想为 HifiShifter 添加新功能，请参考以下建议：

### 4.1 添加新的音频效果
1.  在 `AudioProcessor` 类中添加处理方法（如混响、EQ）。
2.  在 `Track` 类中添加相应的参数存储。
3.  在 `TrackControlWidget` (`timeline.py`) 中添加 UI 控件。
4.  在 `main_window.py` 的 `mix_tracks` 方法中应用这些效果。

### 4.2 修改声码器模型
目前支持 NSF-HiFiGAN。如果想支持其他模型（如 DiffSinger 的声码器）：
1.  修改 `AudioProcessor.load_model` 以加载新模型结构。
2.  修改 `AudioProcessor.synthesize` 以适配新模型的输入格式（例如，如果新模型需要能量值 Energy，则需要增加提取和输入逻辑）。

### 4.3 自定义 UI 组件
所有的绘图组件都位于 `widgets.py`。
*   如果需要修改网格样式，请编辑 `MusicGridItem` 或 `PitchGridItem`。
*   如果需要修改音符块的显示，请编辑 `timeline.py` 中的 `AudioBlockItem`。

### 4.4 调试
*   使用 `run_gui.py` 启动程序。
*   控制台会输出详细的加载和处理日志。
*   `AudioProcessor` 中的 `segment_audio` 方法是性能优化的关键，修改时需谨慎。

## 5. 常见问题 (FAQ)

*   **Q: 为什么没有声音？**
    *   A: 检查是否加载了模型。检查轨道是否被静音。检查系统音频设备设置。
*   **Q: 为什么编辑后声音有爆音？**
    *   A: 这通常是分段合成的边界问题。我们已经实现了 Context Padding 来缓解此问题，但极端的音高突变仍可能导致伪影。尝试平滑音高曲线。
*   **Q: 工程文件打不开？**
    *   A: 检查音频文件或模型文件夹是否被移动或删除。虽然支持相对路径，但如果文件结构发生巨大变化，仍需手动定位。

---
Copyright © 2025 HifiShifter Team.
