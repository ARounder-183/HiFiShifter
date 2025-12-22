# HifiShifter

[中文](README.md) | [English](README_en.md)

HifiShifter 是一个基于深度学习神经声码器（NSF-HiFiGAN）的图形化音高修正工具。它允许用户加载音频文件，在界面上直观地编辑音高曲线，并利用预训练的声码器模型实时合成修改后的音频。

## 安装

### 1. 克隆仓库
```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
```

### 2. 安装依赖
请确保已安装 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果 `requirements.txt` 不存在，请手动安装以下库：

```bash
pip install PyQt6 pyqtgraph sounddevice numpy scipy torch torchaudio pyyaml
```

## 快速开始

1. **运行程序**:
   ```bash
   python run_gui.py
   ```

2. **加载模型**:
   - 点击 `文件` -> `加载模型`。
   - 选择包含 `model.ckpt` 和 `config.json` 的文件夹。

3. **加载音频**:
   - 点击 `文件` -> `加载音频`。
   - 选择 `.wav` 或 `.flac` 文件。

4. **编辑与合成**:
   - 使用左键在钢琴卷帘上绘制音高曲线。
   - 点击 `播放` -> `合成并播放` 听取效果。

## 已知问题

目前存在许多问题，如播放时无法修改音量和导入长音频有大概率会导致卡死。

## 文档

- [开发手册](DEVELOPMENT_zh.md)

## 致谢

本项目使用了以下开源库的代码或模型结构：
- [SingingVocoders](https://github.com/openvpi/SingingVocoders)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)

## License

MIT License
