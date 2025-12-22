# HifiShifter

[中文](README.md) | [English](README_en.md)

HifiShifter is a graphical pitch correction tool based on deep learning neural vocoders (NSF-HiFiGAN). It allows users to load audio files, visually edit pitch curves on a piano roll interface, and synthesize modified audio in real-time using pre-trained vocoder models.

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
```

### 2. Install Dependencies
Ensure Python 3.10+ is installed.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install PyQt6 pyqtgraph sounddevice numpy scipy torch torchaudio pyyaml
```

## Quick Start

1. **Run Application**:
   ```bash
   python run_gui.py
   ```

2. **Load Model**:
   - Click `File` -> `Load Model`.
   - Select the folder containing `model.ckpt` and `config.json`.

3. **Load Audio**:
   - Click `File` -> `Load Audio`.
   - Select a `.wav` or `.flac` file.

4. **Edit & Synthesize**:
   - Use Left Click to draw pitch curves on the piano roll.
   - Click `Playback` -> `Synthesize & Play` to hear the result.

## Known Issues

There are currently many issues, such as the inability to change volume during playback and a high probability of freezing when importing long audio files.

## Documentation

- [Development Manual](DEVELOPMENT_en.md)

## Acknowledgements

This project uses code or model structures from the following open-source repositories:
- [SingingVocoders](https://github.com/openvpi/SingingVocoders)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)

## License

MIT License
