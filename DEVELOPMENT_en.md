# HifiShifter Development Manual

HifiShifter is a graphical pitch correction tool based on deep learning neural vocoders (NSF-HiFiGAN). This document aims to provide developers with an overview of the project architecture, module descriptions, and extension guides.

## 1. Project Overview

### 1.1 Directory Structure

```text
HifiShifter/
├── assets/                 # Resource files
│   └── lang/               # Language packs (zh_CN.json, en_US.json)
├── configs/                # Model configuration files (.yaml)
├── hifi_shifter/           # Core source package
│   ├── __init__.py
│   ├── audio_processor.py  # Audio processing & model inference core
│   ├── config_manager.py   # Configuration & i18n management
│   ├── main_window.py      # Main Window GUI logic
│   ├── timeline.py         # Timeline & Track management
│   ├── track.py            # Track data model
│   └── widgets.py          # Custom UI widgets (PyQtGraph)
├── models/                 # Predefined model structures (NSF-HiFiGAN, UnivNet, etc.)
├── modules/                # Neural network basic modules
├── utils/                  # Utility functions (Audio processing, Config utils)
├── run_gui.py              # Application entry point
├── requirements.txt        # Dependency list
└── ...
```

### 1.2 Core Architecture

HifiShifter adopts a variant of the Model-View-Controller (MVC) architecture, achieving separation of data, view, and logic:

*   **Model (Data Layer)**: The `Track` class encapsulates audio waveforms, F0 curves, Mel spectrograms, and user edit states (e.g., mute, solo, volume).
*   **View (View Layer)**: `MainWindow` and `Timeline` use `PyQt6` to build the interface framework and utilize `pyqtgraph` for high-performance waveform and piano roll rendering.
*   **Controller (Control Layer)**: `AudioProcessor` handles business logic (feature extraction, model inference, audio synthesis), while `MainWindow` coordinates user interaction and background processing.

## 2. Core Modules Detail

### 2.1 Audio Processing (`audio_processor.py`)
This is the core engine of the system, responsible for all tasks related to PyTorch models and signal processing.
*   **Model Loading**: Parses `.yaml` configuration files, instantiates the corresponding generator model based on the config, and loads `.ckpt` weights.
*   **Feature Extraction**:
    *   **F0 (Fundamental Frequency)**: Uses `Parselmouth` (based on Praat algorithm) to extract high-precision F0 curves.
    *   **Mel Spectrogram**: Uses STFT to convert waveforms into Mel spectrograms as the acoustic representation of content.
*   **Smart Segmentation**:
    *   To optimize performance and enable real-time editing, long audio is automatically split into multiple `Segments` based on silence thresholds.
    *   **Incremental Synthesis**: When the user modifies pitch, the system only re-synthesizes the affected segments rather than the entire song, achieving millisecond-level editing feedback.

### 2.2 Track Management (`track.py` & `timeline.py`)
*   **Track Object**: Each track is an independent object storing Raw Data and Edited Data. It also maintains an Undo/Redo Stack.
*   **Timeline**: Manages multi-track mixing logic. It controls the Mute, Solo states, and Volume gain of all tracks, and calculates the final mixed audio output.
*   **View Synchronization**: The Timeline Widget and the main editor window (Piano Roll) are synchronized via signal mechanisms, supporting drag-and-drop track alignment.

### 2.3 Internationalization (`config_manager.py`)
The project has built-in lightweight internationalization (i18n) support.
*   **Language Files**: Located in the `assets/lang/` directory, stored as JSON key-value pairs.
*   **Loading Mechanism**: `ConfigManager` reads the configuration at startup and loads the corresponding language pack.
*   **Usage**: Retrieve localized text in code via `self.cfg.get_text("key_name")`.
*   **Adding New Languages**:
    1. Create a new `xx_XX.json` in `assets/lang/`.
    2. Copy the content of `en_US.json` and translate all Values.
    3. Restart the software and select the new language in Settings.

## 3. Development Guide

### 3.1 Environment Setup
Python 3.10+ environment is recommended.
```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
pip install -r requirements.txt
```

### 3.2 Run & Debug
```bash
python run_gui.py
```
**Debugging Suggestions**:
*   Use VS Code or PyCharm.
*   Key Breakpoints:
    *   `MainWindow.synthesize_audio`: Check synthesis trigger logic.
    *   `AudioProcessor.process_segment`: Check model inference input/output.
    *   `Timeline.paint`: Check custom drawing logic.

### 3.3 Common Extension Tasks
*   **Adding New Vocoder Support**:
    1. Add new model definition files in the `models/` directory.
    2. Modify the `load_model` method in `audio_processor.py` to add initialization logic for the new model.
    3. Ensure the new model's input (Mel + F0) and output (Waveform) formats are compatible with the existing pipeline.
*   **Modifying UI Interaction**:
    *   Main interaction logic (mouse clicks, dragging) is located in the event handling functions of `main_window.py`.
    *   If you need to modify drawing styles (e.g., colors, line thickness), check `widgets.py`.

## 4. Known Issues

*   **Volume Adjustment Latency**: Adjusting track volume in real-time during playback may not take effect immediately or may have a slight delay.
*   **Long Audio Freezing**: When importing very long audio files (e.g., over 10 minutes), the initial feature extraction (F0 and Mel calculation) may cause the interface to become unresponsive (freeze) for a long time. It is recommended to pre-cut long audio into shorter segments.
*   **Memory Usage**: Loading multiple high-sample-rate tracks will consume a large amount of memory because each track stores complete floating-point waveform data and spectrograms.

