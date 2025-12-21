# Vocal Shifter Tool for SingingVocoders

This is a GUI tool for pitch correction using the SingingVocoders models.

## Requirements
- Python 3.8+
- PyQt6
- pyqtgraph
- sounddevice
- torch, torchaudio, numpy

## Installation
1. Ensure you have the `SingingVocoders` dependencies installed.
2. Install GUI dependencies:
   ```bash
   pip install PyQt6 pyqtgraph sounddevice
   ```

## Usage
1. Run the script:
   ```bash
   python vocal_shifter_gui.py
   ```
2. **Load Model**: Click "Load Model Folder" and select the directory containing your model checkpoint (`.ckpt`) and config (`.yaml` or `.json`).
   - Example: `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`
3. **Load Audio**: Click "Load Audio" and select a `.wav` file.
4. **Edit Pitch**:
   - **View/Drag**: Standard pan/zoom.
   - **Draw Pitch**: Select this mode in the dropdown. Left-click and drag on the plot to draw the pitch curve.
   - **Shift**: Use the spinbox to apply a global pitch shift (semitones). Note: This resets manual drawings.
5. **Synthesize**: Click "Synthesize & Play" to generate audio with the modified pitch and play it.
6. **Play Original**: Play the original loaded audio.

## Notes
- The tool uses the loaded vocoder model to re-synthesize audio.
- The quality depends on the model and how well it handles the modified F0.
- "Draw Pitch" modifies the F0 curve directly.
