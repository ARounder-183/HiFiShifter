# HifiShifter 开发手册

HifiShifter 是一个基于深度学习神经声码器（NSF-HiFiGAN）的图形化音高修正工具。本文档旨在为开发者提供项目的架构概览、模块说明以及扩展指南。

## 1. 项目概览

### 1.1 目录结构

```text
HifiShifter/
├── assets/                 # 资源文件
│   └── lang/               # 语言包 (zh_CN.json, en_US.json)
├── configs/                # 模型配置文件 (.yaml)
├── hifi_shifter/           # 核心源码包
│   ├── __init__.py
│   ├── audio_processor.py  # 音频处理与模型推理核心
│   ├── config_manager.py   # 配置与国际化管理
│   ├── main_window.py      # 主窗口 GUI 逻辑
│   ├── timeline.py         # 时间轴与轨道管理
│   ├── track.py            # 音轨数据模型
│   └── widgets.py          # 自定义 UI 组件 (PyQtGraph)
├── models/                 # 预定义模型结构 (NSF-HiFiGAN, UnivNet 等)
├── modules/                # 神经网络基础模块
├── utils/                  # 工具函数 (音频处理, 配置文件)
├── run_gui.py              # 程序启动入口
├── requirements.txt        # 项目依赖列表
└── ...
```

### 1.2 核心架构

HifiShifter 采用 Model-View-Controller (MVC) 的变体架构，实现了数据、视图与逻辑的分离：

*   **Model (数据层)**: `Track` 类封装了音频波形、F0 曲线、Mel 频谱以及用户的编辑状态（如静音、独奏、音量）。
*   **View (视图层)**: `MainWindow` 和 `Timeline` 使用 `PyQt6` 构建界面框架，利用 `pyqtgraph` 进行高性能的波形和钢琴卷帘渲染。
*   **Controller (控制层)**: `AudioProcessor` 负责业务逻辑（特征提取、模型推理、音频合成），`MainWindow` 负责协调用户交互与后台处理。

## 2. 核心模块详解

### 2.1 音频处理 (`audio_processor.py`)
这是系统的核心引擎，负责所有与 PyTorch 模型和信号处理相关的任务。
*   **模型加载**: 解析 `.yaml` 配置文件，根据配置实例化对应的生成器模型，并加载 `.ckpt` 权重。
*   **特征提取**:
    *   **F0 (基频)**: 使用 `Parselmouth` (基于 Praat 算法) 提取高精度的 F0 曲线。
    *   **Mel 频谱**: 使用 STFT 将波形转换为 Mel 频谱，作为内容的声学表征。
*   **智能分段 (Segmentation)**:
    *   为了优化性能和实现实时编辑，长音频会被基于静音阈值自动切分为多个 `Segment`。
    *   **增量合成**: 当用户修改音高时，系统仅重合成受影响的片段，而非整首歌曲，从而实现毫秒级的编辑反馈。

### 2.2 轨道管理 (`track.py` & `timeline.py`)
*   **Track 对象**: 每个音轨是一个独立的对象，存储了原始数据（Raw Data）和编辑数据（Edited Data）。它还维护了撤销/重做栈（Undo/Redo Stack）。
*   **Timeline**: 负责多轨混音逻辑。它管理所有音轨的静音（Mute）、独奏（Solo）状态和音量增益，并计算最终的混合音频输出。
*   **视图同步**: 时间轴面板（Timeline Widget）与主编辑窗口（Piano Roll）通过信号机制保持同步，支持拖拽对齐音轨时间。

### 2.3 国际化 (`config_manager.py`)
项目内置了轻量级的国际化（i18n）支持。
*   **语言文件**: 位于 `assets/lang/` 目录，采用 JSON 格式存储键值对。
*   **加载机制**: `ConfigManager` 在启动时读取配置文件，加载对应的语言包。
*   **使用方法**: 在代码中通过 `self.cfg.get_text("key_name")` 获取当前语言的文本。
*   **添加新语言**:
    1. 在 `assets/lang/` 下新建 `xx_XX.json`。
    2. 复制 `en_US.json` 的内容并翻译所有 Value。
    3. 重启软件并在设置中选择新语言。

## 3. 开发指南

### 3.1 环境搭建
建议使用 Python 3.10+ 环境。
```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
pip install -r requirements.txt
```

### 3.2 运行与调试
```bash
python run_gui.py
```
**调试建议**:
*   使用 VS Code 或 PyCharm。
*   关键断点位置：
    *   `MainWindow.synthesize_audio`: 检查合成触发逻辑。
    *   `AudioProcessor.process_segment`: 检查模型推理输入输出。
    *   `Timeline.paint`: 检查自定义绘图逻辑。

### 3.3 常见扩展任务
*   **添加新声码器支持**:
    1. 在 `models/` 目录下添加新的模型定义文件。
    2. 修改 `audio_processor.py` 中的 `load_model` 方法，添加新模型的初始化逻辑。
    3. 确保新模型的输入（Mel + F0）和输出（Waveform）格式与现有管线兼容。
*   **修改 UI 交互**:
    *   主要交互逻辑（鼠标点击、拖拽）位于 `main_window.py` 的事件处理函数中。
    *   如果需要修改绘图样式（如颜色、线条粗细），请查看 `widgets.py`。

## 4. 已知问题 (Known Issues)

*   **音量调节延迟**: 在播放过程中实时调节音轨音量，可能不会立即生效，或者存在轻微的延迟。
*   **长音频卡死**: 导入非常长的音频文件（例如超过 10 分钟）时，初始的特征提取（F0 和 Mel 计算）可能会导致界面长时间无响应（假死）。建议预先将长音频切割为较短的片段。
*   **内存占用**: 加载多个高采样率音轨会消耗大量内存，因为每个音轨都保存了完整的浮点波形数据和频谱图。

