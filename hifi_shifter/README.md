# Vocal Shifter Package Documentation

This package provides a GUI tool for pitch correction and synthesis using neural vocoders (specifically NSF-HiFiGAN).

## Key Features

*   **Pitch Editing**: Draw pitch curves directly on the piano roll interface.
*   **Real-time Synthesis**: Synthesize audio using the modified pitch curve.
*   **Project Management**: Save and load projects (`.hsp` files) containing audio path, model path, pitch edits, and parameters.
*   **Audio Export**: Export the synthesized audio to WAV format.
*   **Long Audio Optimization**: Automatically segments long audio files to allow for incremental synthesis, significantly improving performance during editing.
*   **Undo/Redo**: Full undo/redo support for pitch editing operations.
*   **Playback Control**: Spacebar to play/stop, "Return to Start" cursor behavior, and click-to-seek on the timeline.

## Module Structure

### 1. `vocal_shifter.audio_processor`

This module handles all the heavy lifting related to audio processing and model inference. It isolates the PyTorch and audio processing logic from the GUI.

**Class: `AudioProcessor`**

*   **`__init__(self)`**: Initializes the processor, detects CUDA device.
*   **`load_model(self, folder_path)`**: Loads the model checkpoint and configuration.
*   **`load_audio(self, file_path)`**:
    *   Loads an audio file and resamples it.
    *   Extracts Mel Spectrogram and F0 (Pitch).
    *   **Segmentation**: Automatically segments the audio based on silence (`segment_audio`) to enable incremental synthesis.
*   **`synthesize(self, f0_midi)`**: Synthesizes the entire audio.
*   **`synthesize_segment(self, segment_idx, f0_midi_segment)`**: Synthesizes only a specific segment of the audio, used for optimization.

### 2. `vocal_shifter.widgets`

This module contains custom PyQtGraph widgets tailored for the piano roll interface.

*   **`CustomViewBox`**: Handles custom mouse interaction (Middle Mouse Pan, Ctrl/Alt+Wheel Zoom).
*   **`PianoRollAxis`**: Displays note names (C4, D#4) on the Y-axis.
*   **`BPMAxis`**: Displays musical time (Bars and Beats) on the X-axis. Supports click-to-seek.

### 3. `vocal_shifter.main_window`

This module contains the main GUI application logic.

**Class: `VocalShifterGUI`**

*   **UI Components**:
    *   **Menu Bar**: File (Open/Save/Export), Edit (Undo/Redo), Playback controls.
    *   **Controls Bar**: Quick access to Shift, BPM, and Time Signature parameters.
    *   **Piano Roll**: `pg.PlotWidget` for editing pitch.
*   **State Management**:
    *   `is_dirty`: Tracks if changes have been made to trigger re-synthesis.
    *   `segment_states`: Tracks which audio segments need re-synthesis.
    *   `undo_stack` / `redo_stack`: Manages edit history.
*   **Interaction Logic**:
    *   **Smart Synthesis**: Only re-synthesizes modified segments of the audio to ensure responsiveness.
    *   **Playback**: Handles audio playback using `sounddevice`, with cursor synchronization.

### 4. `vocal_shifter.main`

*   **`main()`**: The entry point function.

## Usage

To run the application:

```bash
python run_gui.py
```

Or as a module:

```bash
python -m vocal_shifter
```

### Shortcuts

*   **Space**: Play / Stop (Return to Start)
*   **Ctrl + Z**: Undo
*   **Ctrl + Shift + Z** / **Ctrl + Y**: Redo
*   **Ctrl + O**: Open Project
*   **Ctrl + S**: Save Project
*   **Ctrl + Shift + S**: Save Project As
*   **Mouse Left**: Draw Pitch
*   **Mouse Right**: Erase / Restore Pitch
*   **Mouse Middle**: Pan View
*   **Ctrl + Wheel**: Zoom Time (X)
*   **Alt + Wheel**: Zoom Pitch (Y)
