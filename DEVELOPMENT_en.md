# HifiShifter Development Manual

HifiShifter is a graphical pitch correction tool based on deep learning neural vocoders (NSF-HiFiGAN). It allows users to load audio files, visually edit pitch curves on a piano roll interface, and synthesize modified audio in real-time using pre-trained vocoder models.

## 1. Core Architecture and Principles

HifiShifter adopts a modular design, separating GUI interaction, audio processing, and data management.

### 1.1 System Architecture

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

### 1.2 Core Modules

#### A. `hifi_shifter.audio_processor` (Audio Processing Core)
The engine of the system. Handles all tasks related to PyTorch models and audio signal processing.
*   **Model Loading**: Reads `.ckpt` and `.yaml` config files, initializes NSF-HiFiGAN generator.
*   **Feature Extraction**:
    *   **Mel Spectrogram**: Converts waveform to Mel spectrogram using STFT.
    *   **F0 (Fundamental Frequency)**: Extracts original pitch using Parselmouth (Praat) algorithm.
*   **Segmentation**: For real-time editing, long audio is automatically split into segments (based on silence detection).
    *   **Incremental Synthesis**: When pitch is modified, only affected segments are re-synthesized, ensuring fast response.
*   **Synthesis**: Takes Mel spectrogram and modified F0 to output waveform.

#### B. `hifi_shifter.track` (Data Model)
Each track is a `Track` object, encapsulating its state:
*   **Raw Data**: Waveform, sample rate, original F0, Mel spectrogram.
*   **Edit Data**: User-modified F0 curve (`f0_edited`).
*   **State Flags**: `muted`, `solo`, `volume`, `start_frame` (timeline offset).
*   **Cache**: `synthesized_audio` stores results to avoid re-computation.
*   **History**: `undo_stack` and `redo_stack`.

#### C. `hifi_shifter.timeline` (Timeline Panel)
Manages multi-track and macro view.
*   **Track Management**: Controls for Mute, Solo, Volume, and waveform overview.
*   **Time Control**: Ruler and playback head.
*   **Interaction**: Dragging tracks for alignment, box selection, zooming.

#### D. `hifi_shifter.main_window` (Main Interface & Logic)
Central controller coordinating all components.
*   **Event Loop**: Handles playback timer (`QTimer`).
*   **Drawing Logic**: Uses `pyqtgraph` for Piano Roll and Waveform.
*   **Sync Mechanism**: Ensures synchronization between timeline and editor views.
*   **Project Management**: Handles `.hsp` file serialization.

#### E. `utils.i18n` (Internationalization)
Manages multi-language support.
*   **Resource Loading**: Loads `.json` language files from `assets/lang/`.
*   **Dynamic Switching**: Saves language preference via `config_manager`, effective after restart.

## 2. Features and Usage Guide

### 2.1 Basic Operations
1.  **Load Model**: Click "File -> Load Model" (or Open Project), select folder with `model.ckpt` and `config.json`.
2.  **Import Audio**: Click "File -> Load Audio" or drag & drop files.
3.  **View Operations**:
    *   **Middle Click Drag**: Pan view.
    *   **Ctrl + Scroll**: Zoom Time (Horizontal).
    *   **Alt + Scroll**: Zoom Pitch (Vertical).
4.  **Pitch Editing**:
    *   **Draw (Left Click)**: Draw lines on piano roll to modify pitch.
    *   **Erase (Right Click)**: Restore to original pitch.

### 2.2 Advanced Features
*   **Multi-language**: Switch between Chinese/English in "Settings -> Language" (Restart required).
*   **Multi-track Mixing**: Support for BGM and multiple vocal tracks. Adjust volume, mute, solo in the left panel.
*   **Time Alignment**: Drag track blocks in the timeline panel to adjust start time.
*   **Parameters**:
    *   **Shift**: Global pitch shift (semitones).
    *   **BPM / Time Sig**: Set project tempo and time signature for grid lines.
*   **Export**: Mix all non-muted tracks and export as WAV.

## 3. Project File Structure (.hsp)

Standard JSON format, supporting relative paths.

```json
{
    "version": "2.1",
    "model_path": "models/nsf_hifigan",
    "params": {
        "bpm": 120.0,
        "beats": 4
    },
    "tracks": [
        {
            "name": "vocal_track",
            "file_path": "audio/vocal.wav",
            "type": "vocal",
            "shift": 0.0,
            "muted": false,
            "solo": false,
            "volume": 1.0,
            "start_frame": 0,
            "f0": [ ... ]
        }
    ]
}
```

## 4. Secondary Development Guide

### 4.1 Adding Audio Effects
1.  Add processing method in `AudioProcessor`.
2.  Add parameter storage in `Track`.
3.  Add UI controls in `TrackControlWidget` (`timeline.py`).
4.  Apply effects in `mix_tracks` in `main_window.py`.

### 4.2 Modifying Vocoder Model
Currently supports NSF-HiFiGAN. To support others:
1.  Modify `AudioProcessor.load_model`.
2.  Modify `AudioProcessor.synthesize` to adapt to new input formats.

### 4.3 Custom UI Components
All drawing components are in `widgets.py`.
*   Edit `MusicGridItem` or `PitchGridItem` for grid styles.
*   Edit `AudioBlockItem` in `timeline.py` for note blocks.

### 4.4 Debugging
*   Run `run_gui.py`.
*   Console outputs detailed logs.
*   `segment_audio` in `AudioProcessor` is key for performance.

## 5. FAQ

*   **Q: No sound?**
    *   A: Check if model is loaded, track is muted, or system audio device.
*   **Q: Popping sound after edit?**
    *   A: Likely segmentation boundary issues. Context padding is implemented but extreme pitch shifts may still cause artifacts. Smooth the curve.
*   **Q: Project file won't open?**
    *   A: Check if audio/model files were moved. Relative paths are supported but major structure changes require manual fix.

---
Copyright © 2025 HifiShifter Team.
