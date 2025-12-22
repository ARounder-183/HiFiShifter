# HifiShifter

[中文](README.md) | [English](#english)

HifiShifter 是一个基于深度学习神经声码器（NSF-HiFiGAN）的图形化音高修正工具。它允许用户加载音频文件，在钢琴卷帘界面上直观地编辑音高曲线，并利用预训练的声码器模型实时合成修改后的音频。

HifiShifter is a graphical pitch correction tool based on deep learning neural vocoders (NSF-HiFiGAN). It allows users to load audio files, visually edit pitch curves on a piano roll interface, and synthesize modified audio in real-time using pre-trained vocoder models.

## 安装 / Installation

### 1. 克隆仓库 / Clone Repository
```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
```

### 2. 安装依赖 / Install Dependencies
请确保已安装 Python 3.10+。
Ensure Python 3.10+ is installed.

```bash
pip install -r requirements.txt
```

如果 `requirements.txt` 不存在，请手动安装以下库：
If `requirements.txt` is missing, install manually:

```bash
pip install PyQt6 pyqtgraph sounddevice numpy scipy torch torchaudio pyyaml
```

## 快速开始 / Quick Start

1. **运行程序 / Run Application**:
   ```bash
   python run_gui.py
   ```

2. **加载模型 / Load Model**:
   - 点击 `文件` -> `加载模型` (File -> Load Model)。
   - 选择包含 `model.ckpt` 和 `config.json` 的文件夹。
   - Select the folder containing `model.ckpt` and `config.json`.

3. **加载音频 / Load Audio**:
   - 点击 `文件` -> `加载音频` (File -> Load Audio)。
   - 选择 `.wav` 或 `.flac` 文件。
   - Select a `.wav` or `.flac` file.

4. **编辑与合成 / Edit & Synthesize**:
   - 使用左键在钢琴卷帘上绘制音高曲线。
   - Use Left Click to draw pitch curves on the piano roll.
   - 点击 `播放` -> `合成并播放` (Playback -> Synthesize & Play) 听取效果。
   - Click `Synthesize & Play` to hear the result.

## 文档 / Documentation

- [开发手册 (中文)](DEVELOPMENT_zh.md)
- [Development Manual (English)](DEVELOPMENT_en.md)

## License

MIT License
