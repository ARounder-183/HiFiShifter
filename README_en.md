# HifiShifter

[中文](README.md) | [English](README_EN.md)

HifiShifter is a graphical vocal editing and synthesis tool based on deep learning neural vocoders (NSF-HiFiGAN). It allows you to load audio, edit parameter curves (such as pitch, tension) directly on the piano roll, and synthesize modified audio in real-time/incrementally using pre-trained vocoders.

**Note: This project is still under development and iteration. Full pipeline testing has not been completed, and there may be many bugs or stability issues.**

## Feature Overview

- **Multi-parameter editing**: Supports editing not only pitch but also tension, with abstract interfaces reserved for future parameter extensions.
- **Selection editing (generic abstraction)**: Supports selecting sample points, highlighting selected segments, and dragging entire selections up/down (applies to current parameter).
- **Long audio incremental synthesis**: Automatic segmentation (based on silence/fragment strategy), only re-synthesizes dirty segments to ensure smooth interaction.
- **Project management**: Save/load projects (`.hshp`) containing timeline and clip information; supports "recent projects", window title shows project name and unsaved marker (`*`), prompts for unsaved changes when closing window.
- **Undo/Redo (backend authority)**: Maintains Undo/Redo stack in Tauri/Rust backend, frontend `Ctrl+Z / Ctrl+Y` directly calls backend undo/redo to avoid "frontend undo being overwritten by backend state".
- **Bottom parameter panel (Pitch/Tension, independent window)**: Bottom parameter editor has independent `zoom(pxPerBeat)` and `horizontal scroll(scrollLeft)` (not forced to sync with timeline); top provides time ruler and BPM grid; background shows waveform preview of "selected root track (root + subtracks) mix input" for easy alignment.
- **Root track `C` (synthesis) toggle**: Root track can toggle synthesis output on/off; when `C` is on and editing pitch parameter, backend generates/updates `pitch_orig` for that root track based on "root track (root + subtracks) mix input" (same as parameter panel background waveform). Bottom pitch panel enters loading state (can show progress bar) until pitch detection completes automatically; when `C` is off, pitch panel only shows waveform and prompts to enable `C`.
- **Pitch curve timeline alignment**: `pitch_orig` generation aligns with timeline length/playback rate (audio may be time-stretched before analysis if needed), so after time stretching, curves scale with timeline and maintain same scale as waveform/ruler.
- **Pitch editing affects playback/synthesis/export (default WORLD, switchable to ONNX)**: When `C` is on, `pitch_edit` drawn in bottom parameter panel affects final mix in "play original (real-time engine) / synthesize and play / export WAV" paths (default uses WORLD vocoder for pitch shifting; also supports switching to NSF-HiFiGAN ONNX inference, see below).
- **Pitch algorithm switching (per root track)**: Pitch panel can switch analysis algorithm for current root track (currently integrated WORLD DLL; `none` means no generation).
- **Playback and export**: Play original/synthesized audio; export WAV (supports mix or separate tracks depending on GUI entry point).
- **Real-time playback mixing**: Adjust volume faders, mute tracks, solo tracks take effect immediately during playback (no need to stop and restart).
- **Multi-language (i18n) and themes**: Supports Chinese/English and dark/light themes.
- **Modern interface**: Dark theme interface designed with DAW reference, providing intuitive operation experience.
- **Audio editing features**:
  - **Audio slicing**: Split selected audio clips at playhead position (shortcut: S)
  - **Fade in/out**: Add fade effects to clips, supports custom fade durations
  - **Time stretching**: Adjust clip playback rate (0.1x - 10x) for time stretching/shortening
  - **Audio trimming**: Precise control over clip start and end positions

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
```

### 2. Install Dependencies

Ensure you have the following tools installed:

- **Node.js** (recommended 18+) and npm
- **Rust toolchain** (see `rust-toolchain.toml`)
- **Tauri 2 CLI**: `cargo install tauri-cli --version "^2"`

Install frontend dependencies:

```bash
npm --prefix frontend install
```

## Quick Start

### Run Development Mode

```bash
cd backend/src-tauri
cargo tauri dev
```

**Note:** WORLD vocoder and Signalsmith Stretch are now statically compiled via cc crate, no additional configuration needed. First build will automatically compile C++ source code (takes about 1-2 minutes).

### ONNX Inference (Optional)

To switch pitch editing algorithm to **NSF-HiFiGAN ONNX inference** (experimental, may be slower):

- Enable feature during compilation: `cargo tauri dev --features onnx`
- Method A (recommended): Select `NSF-HiFiGAN (ONNX)` in the `Algo` dropdown in bottom parameter panel (Pitch).
- Method B (debug/force override): Set `HIFISHIFTER_PITCH_EDIT_ALGO=nsf_hifigan_onnx`
- Provide model path (choose one):
  - `HIFISHIFTER_NSF_HIFIGAN_ONNX=...\pc_nsf_hifigan.onnx`
  - or `HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR=...\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`
- (Optional) `HIFISHIFTER_NSF_HIFIGAN_CONFIG=...\config.json` (default uses `config.json` in model directory)

Example (PowerShell):
```powershell
$env:HIFISHIFTER_PITCH_EDIT_ALGO = "nsf_hifigan_onnx"
$env:HIFISHIFTER_NSF_HIFIGAN_MODEL_DIR = "E:\Code\HifiShifter\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
```

### Pitch Edit Application Method

- Default: `v2` (pitch shift per clip then mix back with fade, avoids boundary artifacts from global pitch shift on entire mixdown)
- Fallback to old implementation (global mixdown pitch shift):
  - `HIFISHIFTER_PITCH_EDIT_APPLY=v1`
  - or `HIFISHIFTER_PER_CLIP_PITCH_EDIT=0`
- Force use v2:
  - `HIFISHIFTER_PITCH_EDIT_APPLY=v2`
  - or `HIFISHIFTER_PER_CLIP_PITCH_EDIT=1`

### Playback Instructions

- Uses Rust-side low-latency real-time audio engine (`cpal` output stream callback mixing), playback startup doesn't depend on "full offline rendering".
  - To avoid stuttering during timeline changes, audio decoding/resampling prepares asynchronously in background; during initial loading of audio segments, clips may output silence briefly then automatically resume when ready.
- When Pitch Edit algorithm switches to **NSF-HiFiGAN ONNX**, playback briefly waits for pre-buffering; bottom-left status bar shows "Rendering..." prompt.
  - Optional parameters (environment variables): `HIFISHIFTER_ONNX_STREAM_PRIME_SEC` (default 0.25), `HIFISHIFTER_ONNX_STREAM_PRIME_TIMEOUT_MS` (default 4000).
  - To disable waiting: `HIFISHIFTER_ONNX_PITCH_STREAM_HARD_START=0`.
  - ONNX real-time pitch shifting performs inference on entire voiced segments (unvoiced segments pass through directly) to reduce boundary noise from fixed-time window segmentation.
- **Audio source formats**: Real-time playback side attempts to decode common audio formats via `symphonia`; offline export/mixdown format support may differ from real-time playback.
- **Waveform/duration preview**: After importing audio, backend extracts duration and waveform preview for timeline display. Waveform prioritizes "on-demand peaks (min/max) + caching" drawing method: zooming in requests more columns for clearer details; WAV prioritizes `hound` (supports 16/24/32-bit int + 32-bit float), other formats use `symphonia` generic decoding as fallback.
- **Waveform caching (performance)**: Backend writes base peaks for each audio file to disk cache to reduce repeated calculations; can manually clear via menu `View` → `Clear Waveform Cache`.
- **Project save/load**: Manage project files (`.hshp`) via menu `File` → `New Project / Open Project / Save / Save As / Recent Projects`.
- **Unsaved changes prompt**: When project has unsaved changes, window title shows `*`, closing window prompts to save.

### Loading Audio

- Click `File` → `Load Audio`, supports `.wav` / `.flac` / `.mp3`.
- Alternatively, drag audio files directly to timeline track area for import.

### Editing and Synthesis

- Use `Edit:` dropdown in top bar to select parameter to edit (pitch/tension).
- Bottom parameter panel also provides parameter toggle buttons (synchronized with top dropdown).
- **Manual refresh**: When waveforms/parameter curves don't update promptly, click `Refresh` in top-right corner of bottom parameter panel to force refetch visible window data.
- **Draw curves**: Left-click to draw edit curves (solid lines); right-click to restore original curves (dashed lines).
- **Selection copy/paste**: Switch to Select mode, left-click drag to create vertical time selection; `Ctrl+C` copies edit curves in selection, `Ctrl+V` pastes to selection start point.
- **Zoom/scroll (parameter panel independent)**:
  - Mouse wheel: Horizontal timeline zoom (centered on cursor).
  - Ctrl + mouse wheel: Vertical parameter axis zoom (centered on cursor).
  - Middle mouse button drag: Pan view (timeline).
  - Horizontal scrollbar: Horizontal scrolling.
- Click `Play` → `Synthesize and Play` to hear results (when root track `C` is on, applies pitch panel edit curves to synthesis output).

## Edit Mode and Selection Mode

### Edit Mode
- **Left-click**: Edit current parameter curve (follows parameter panel selection).
- **Right-click**: Restore current parameter to "original curve" (effective for Pitch/Tension).

### Selection Mode
- **Left-click drag**: Create vertical time selection (covers entire height).
- **Copy/paste**: `Ctrl+C` copies "edit curves" within selection range, `Ctrl+V` writes from selection start point.

## Axis Display (Parameter-dependent)

- When editing **pitch**: Left side shows piano roll-style pitch axis (C2 → C8) with semitone lines and C note names for scale alignment.
- When editing **tension** and other linear parameters: Left axis switches to numerical scale (0.0 / 0.5 / 1.0) for intuitive linear parameter alignment.

This mechanism is abstracted: when adding new parameters, only need to implement parameter's "axis type/mapping/formatting" to reuse existing UI.

## Common Shortcuts

| Operation | Shortcut / Mouse |
| :--------------------------- | :------------------------------ |
| Pan view (timeline) | Middle mouse button drag |
| Horizontal zoom (timeline) | Mouse wheel (centered on cursor) |
| Vertical zoom (track height, timeline) | Ctrl + mouse wheel |
| Vertical zoom (parameter axis, parameter panel) | Ctrl + mouse wheel (inside parameter panel) |
| Play/pause | Space |
| Play/stop | Enter |
| Undo / Redo | Ctrl + Z / Ctrl + Y |
| New project | Ctrl + N |
| Open project | Ctrl + Shift + O |
| Save | Ctrl + S |
| Save As | Ctrl + Shift + S |
| Export audio | Ctrl + E |
| Mode Toggle (Select/Draw) | Tab |
| Delete selected clips | Delete / Backspace |
| Copy selected clips (internal clipboard) | Ctrl + C |
| Paste at playhead position | Ctrl + V |
| Parameter panel copy selection curves | Ctrl + C (Select mode) |
| Parameter panel paste to selection start | Ctrl + V (Select mode) |
| Split clip | S (split selected clip at playhead position) |
| Add track | Ctrl + T |
| Quick search | Ctrl + F |

Additional notes:
- Supports dragging audio files directly to timeline track area for import.
- When dropping outside any track (e.g., blank row/outside track area), automatically creates new track and places clip.
- Left track list supports dragging: up/down drag reorders tracks; drag to another track with right offset sets it as subtrack (nesting).
- Clicking timeline ruler, timeline blank area, or clip (item) during playback positions cursor and stops playback.
- Dragging clips (move/cross-track move) doesn't change playhead position; only "clicking clip" updates cursor/playhead.
- Timeline has clear project duration boundaries, BPM grid not displayed beyond boundaries; clips extending beyond boundaries automatically extend project boundaries.
- Allows playback even when project has no audio (for checking timeline/cursor behavior).

## Audio Clip Editing

After selecting audio clips in timeline panel, perform following operations in bottom properties panel:

Also supports direct editing on timeline (closer to DAW interaction):

- **Snap to grid**: Clip move/trim defaults to grid snapping; hold `Shift` to temporarily disable snapping.
- **Trim/extend range**: Drag clip left/right boundaries to trim or extend; when clip length exceeds source audio available range, excess becomes "blank/silence" (waveform preview shows blank), doesn't loop repeat.
- **Time Stretch**: Hold `Alt` + left-click drag clip left/right boundaries to stretch audio (changes playback rate and clip length simultaneously).
  - High-quality "pitch-preserving" real-time stretching uses Signalsmith Stretch (MIT), statically compiled via cc crate.
- **Internal offset (Slip-Edit)**: Hold `Alt` + left-click drag clip body to slide internal content left/right (equivalent to modifying Trim In), doesn't change clip timeline position (no snapping); offset limited to "±1x source audio duration", allows leading/trailing silence.
- **Fade in/out**: Drag top-left/top-right corner handles to adjust fade durations.
- **Fade effect on waveform**: Waveform preview updates amplitude in real-time with fades for intuitive audio-visual alignment.
- **Gain (dB)**: Drag top-left knob (up/down) to adjust gain, top-right corner shows current dB.
- **Gain-linked waveform**: Adjusting gain synchronously changes waveform amplitude (visual preview only).
- **Clip mute (M)**: Top-left `M` button mutes clip, muted clips appear grayed out.
- **Box selection multi-select**: Hold right-click drag in timeline blank area to box-select multiple clips.
- **Group move**: After multi-select, dragging any clip moves all selected clips together.
- **Cross-track move**: Drag clip up/down to switch to other tracks (only allows entire group cross-track move when selected clips are on same track).
- **Copy drag**: Hold `Ctrl` while dragging clip creates copy at target position while keeping original clip in place (copy completes on mouse release).
- **Glue**: Right-click clip, select "Glue" (requires same track and at least 2 clips).
- **Split**: Select clip, press `S` to split at playhead position.

Copy/paste rules:
- `Ctrl + C` copies selected clips to internal clipboard.
- Also attempts to write to system clipboard (failure ignored, doesn't affect internal copy).
- `Ctrl + V` aligns "leftmost start point among selected clips" to playhead position, other clips maintain relative spacing; ensures start point ≥ 0.
- After paste/copy drag completion, automatically selects newly created copies.

Property panel parameters:
- **Length (Len)**: Adjust clip length
- **Gain**: Adjust clip volume (0-2x)
- **Playback Rate (Rate)**: Adjust playback speed for time stretching (0.1x-10x)
- **Fade In**: Set fade-in duration (in beats)
- **Fade Out**: Set fade-out duration (in beats)
- **Trim In**: Adjust clip start position
- **Trim Out**: Adjust clip end position

Clips display fade in/out visual effects for intuitive audio processing status visualization.

## Performance Optimization

### Pitch Analysis Acceleration (v3)

Pitch analysis performance significantly optimized via **parallel processing, intelligent caching, incremental refresh**, typical project speedup **3-9x**:

| Scenario | Old Duration | Current Target | Optimization Method |
| --------------------------- | ------------ | ---------- | ---------------------------- |
| **Initial analysis** (10 clips) | 22-45s | 3-7s | Multi-core parallel (rayon) |
| **Repeat analysis** (cached) | 22-45s | <100ms | LRU memory cache |
| **Incremental refresh** (edit single clip) | 22-45s | 1-4s | Snapshot comparison (only reanalyze changed items) |
| **Position change** (drag clip) | 22-45s | <100ms | Position-independent cache keys |

**Key Features**:

1. **Intelligent caching**:
   - Generate cache keys (Blake3 hash) based on audio content, parameters, BPM, etc.
   - Position changes (dragging clips) don't trigger reanalysis
   - LRU strategy automatically manages 100 clips capacity (~300-500MB)

2. **Incremental refresh**:
   - Record previous timeline snapshot, compare to detect changes
   - Only reanalyze added/modified clips
   - Unchanged clips read from cache in milliseconds

3. **Parallel processing**:
   - Use Rayon thread pool to parallel analyze multiple clips
   - Sort by workload (duration × cache miss coefficient) for optimal load balancing
   - Progress bar weighted by duration, shows real-time analysis progress

4. **Cache management**:
   - Cache hit rate: >95% in typical workflows (repeat refresh scenarios)
   - Cache statistics: Query via Tauri commands (cached_clips, capacity, hit_rate)
   - Clear cache: Supports manual clearing or automatic LRU eviction

**Usage Recommendations**:

- Wait 3-7 seconds for initial analysis when opening project or making extensive changes
- Dragging clips, adjusting positions requires no waiting (direct cache reuse)
- Editing single clip parameters completes incremental update in ~1-4 seconds
- If memory usage too high, clear cache via backend commands

Detailed implementation see [DEVELOPMENT.md - Pitch analysis performance optimization](DEVELOPMENT.md#pitch-analysis-performance-optimization-v3)

## Documentation

- [Development Manual](DEVELOPMENT.md)
- [User Manual](USERMANUAL.md)
- [Update Plan](todo.md)

## Acknowledgments

This project uses code or model architectures from the following open-source libraries:
- [WORLD](https://github.com/mmorise/World) — High-quality speech analysis and synthesis system
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) — High-quality audio time stretching library (MIT)
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) — Audio analysis and synthesis library
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) — Singing voice synthesis vocoders (OpenVPI)
- [HiFi-GAN](https://github.com/jik876/hifi-gan) — High-fidelity generative adversarial network vocoder

## License

This project is released under the [MIT License](LICENSE).