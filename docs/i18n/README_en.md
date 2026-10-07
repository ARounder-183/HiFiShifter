# HiFiShifter

[简体中文](../../README.md) | [繁體中文](README_zh-TW.md) | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

HiFiShifter is a graphical vocal editing and synthesis tool. It supports multi-track audio clip processing and uses various vocoders to achieve pitch correction and parameter adjustment for vocal, integrating splicing and tuning for Jinriki VOCALOID production.

**The project is still under active development. Full-chain testing has not been completed, so there may be many bugs or instability issues.**

![Preview](../preview.png)

## Features

- **Track editing**: a DAW-like multi-track timeline with clip cropping, stretching, slip editing, fades and crossfades, grouping, take management, silence detection, ripple editing, tempo mapping (BPM / time signature / scale), metronome, recording and more.
- **Parameter editing**: adjust parameter lines such as pitch, volume, dynamics, pan, formant, breath and tension per track group with a piano-roll style parameter editor; supports draw / line / vibrato tools, multi-range editing and vibrato preset management, and child tracks can stack `Cents offset` / `Degree offset` / `Formant offset` to build harmonies.
- **Three vocoder algorithms**: nsf-hifigan (PC-NSF-HiFiGAN), World and VsLib — see [Algorithms](#algorithms).
- **Interoperability**: import REAPER (`.rpp`) and VocalShifter (`.vshp` / `.vsp`) projects, with two-way read/write of the REAPER and VocalShifter clipboards; MIDI can be imported as pitch reference clips / pitch parameters / tempo maps, and audio clips and pitch lines can be exported to MIDI.
- **Import & export**: import common audio / video formats (the audio track is extracted from videos automatically); export the project and stems as `wav` / `mp3` / `flac`.
- **VST3 / ARA Plugin**: besides the standalone app, a Windows x64 VST3 (ARA) plugin is also provided, opening the same editing interface directly inside REAPER: the host handles clip geometry, BPM and fade audio, HiFiShifter handles parameter curves, and edits are applied back to the host automatically. See the [VST3 / ARA plugin chapter](USERMANUAL_en.md#8-vst3--ara-plugin-reaper) of the user manual for details.
- **Miscellaneous**: built-in file browser and quick search, notepad, auto backup, multi-language UI (简体中文 / 繁體中文 / English / 日本語 / 한국어), light and dark themes, inference device selection and benchmarking.

## Installation

Download the package for your operating system and architecture from the [Releases](https://github.com/ARounder-183/HiFiShifter/releases) page:

- **Windows**: NSIS installer (`installer`) or portable zip archive (`portable`), available for x86_64 and arm64.
- **macOS**: unsigned dmg (Apple Silicon → `arm64`, Intel → `x86_64`). The first launch requires manual approval; if macOS reports the file as "damaged", follow the steps in the [user manual](USERMANUAL_en.md#1-installation).
- **Linux**: AppImage (x86_64 / arm64).

**VST3 / ARA Plugin**: Windows x64 also has a VST3 (ARA) plugin form, usable as an embedded interface inside REAPER (currently verified against REAPER only). See the [VST3 / ARA plugin chapter](USERMANUAL_en.md#8-vst3--ara-plugin-reaper) of the user manual for installation, runtime dependencies and usage limitations.

GPU acceleration: Windows uses DirectML (DirectX 12), macOS (Apple Silicon) uses CoreML + WebGPU, and Linux x86_64 uses WebGPU (Dawn/Vulkan); other platforms fall back to CPU. You can switch the device and run the benchmark in-app via `Options → Inference Device`.

## How it works

HiFiShifter uses an offline rendering approach similar to UTAU, processing, rendering, and caching each audio clip on the timeline before feeding it into the playback system, resulting in faster processing for short clips.

HiFiShifter provides a unified rendering interface to facilitate future algorithm additions.

## Recommended workflow

Our recommended workflow is:

1. Prepare short clip sources needed for vocal using other DAWs or slicing software.
2. Complete audio splicing and tuning in HiFiShifter.

If you already work in REAPER, you can also use the [VST3 / ARA plugin](USERMANUAL_en.md#8-vst3--ara-plugin-reaper) to do the splicing and tuning inside the REAPER project directly, without importing the audio into the standalone app.

HiFiShifter also supports the following operations to facilitate migration from other software:

1. Directly open VocalShifter projects.
2. Directly open Reaper projects.
3. Parse VocalShifter clipboard content, allowing parameters from VocalShifter to be pasted into HiFiShifter's parameter area.
4. Parse Reaper clipboard content, allowing Reaper items to be pasted directly into HiFiShifter.

## UI and algorithms

### Layout

HiFiShifter is roughly divided into the track panel at the top and the parameter panel at the bottom: the track panel handles clip editing and arrangement, while the parameter panel handles parameter tuning. Each panel can be docked, floated and rearranged (`View → Window / Layout`).

Tracks support nesting: drag one track under another to form a track group (root track + child tracks). A track group shares a single algorithm and a single set of parameter lines, which apply to every audio clip in the group by position. Press the track's Compose button `C` before editing parameters.

### Algorithms

HiFiShifter currently supports three algorithms:

- **World**: a classic vocoder. Supports editing of the `Pitch`, `Volume`, `Dynamics` and `Pan` parameters.
- **PC-NSF-HiFiGAN** (displayed as `nsf-hifigan` in the in-app algorithm list): OpenVPI's open-source HiFi-GAN vocoder specialized for singing voices, and the default algorithm. Supports editing of the `Pitch`, `Formant Shift`, `Breath Gain`, `Tension`, `Volume`, `Dynamics` and `Pan` parameters. `Breath Gain` and `Tension` depend on the `Harmonic Separation` switch: when enabled, each audio clip is split into harmonic and noise parts (using the hnsep model), which **adds extra rendering cost** (the first pass performs a separation for every clip and can be slow); when disabled, no separation is performed at all — these two parameters are greyed out, their curves remain visible but cannot be edited, and they take no part in synthesis.
- **VsLib** (displayed as `vslib` in the in-app algorithm list): the algorithm library provided by the official VocalShifter. Supports editing of the `Pitch`, `Formant Shift`, `Breathiness`, `Volume`, `Dynamics`, `Pan` and `Synth Mode` parameters. **Available only on Windows x86_64**; because the official DLL only supports file I/O, processing takes more time compared to VocalShifter itself.

For detailed operations on the track and parameter panels (fade editing, snap settings, vibrato presets, export and recording, etc.), please read the [user manual](USERMANUAL_en.md).

## Keyboard shortcuts

> All shortcuts can be customized in-app under `Options → Keyboard Shortcuts...`; the table below lists the defaults. On macOS, `Ctrl` corresponds to `⌘` and `Alt` to `⌥`.

| Action | Shortcut / Mouse |
| :----------------------------------- | :----------------------------------------------------------- |
| Play / Pause (does not return to the start point) | `Space` |
| Play / Stop (returns to the start point) | `Enter` |
| Toggle metronome | `K` |
| Pan the view | Middle mouse button drag |
| Scroll the timeline | Mouse wheel (free two-axis scrolling) |
| Scroll horizontally / vertically | `Shift` / `Alt` + mouse wheel |
| Zoom track height | `Ctrl` + mouse wheel |
| Undo / Redo | `Ctrl + Z` / `Ctrl + Shift + Z` (or `Ctrl + Y`) |
| Cut / Copy / Paste | `Ctrl + X` / `Ctrl + C` / `Ctrl + V` |
| Delete selected clips | `Delete` |
| Split clip (at playhead) | `S` |
| Group / Ungroup | `G` / `U` |
| Cycle takes | `T` (`Shift + T` for the previous one) |
| Marquee multi-select | Hold the right mouse button and drag on empty timeline |
| Copy-drag | Hold `Ctrl` while dragging a clip |
| Stretch clip / Slip editing | Hold `Alt` and drag the clip's edge / body |
| Temporarily toggle snap | Hold `Shift` while dragging |
| New project / Open project | `Ctrl + N` / `Ctrl + Shift + O` |
| Import media / Export audio | `Ctrl + O` / `Ctrl + E` |
| Save / Save as | `Ctrl + S` / `Ctrl + Shift + S` |
| New track / Record | `Ctrl + T` / `Ctrl + R` |
| Mode toggle (select ↔ draw-type tools) | `Tab` |
| Quick search | `Ctrl + F` |
| Shift the whole param line up / down | `=` / `-` (`[` / `]` for the selected range), `Shift` for large steps, `Ctrl` for fine steps |
| Multi-range selection in the param editor | Hold `Ctrl` and drag to add a range; click an existing range to remove it |

For the complete list of shortcuts and modifier keys, see the [user manual](USERMANUAL_en.md#3-track-view) or the in-app shortcut settings.

## Development

This section is for developers; regular users can skip it.

### 1. Clone the Repository

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HiFiShifter
```

### 2. Install Dependencies

#### Windows

Make sure the following tools are installed:

- **Node.js** (20+ recommended, CI uses 24) and npm
- **Rust toolchain** (see `rust-toolchain.toml`)
- **Tauri 2 CLI**: `cargo install tauri-cli --version "^2"`
- **CMake** (required to build the SoundTouch library)

ONNX Runtime (DirectML) is automatically downloaded by the ort crate at build time — no extra configuration needed.

Alternatively, run `scripts/install_deps_windows.ps1`, which installs NSIS / CMake / LLVM via Chocolatey and installs tauri-cli (Chocolatey must already be installed).

Install frontend dependencies:

```bash
npm --prefix frontend install
```

#### macOS

```bash
chmod +x ./scripts/install_deps_macos.sh
SKIP_FRONTEND=0 bash ./scripts/install_deps_macos.sh
```

#### Linux

Make sure the following tools are installed:

- **Node.js** (20+ recommended, CI uses 24) and npm
- **Rust toolchain** (see `rust-toolchain.toml` — the project auto-selects the correct platform stable toolchain)
- **Tauri 2 CLI**: `cargo install tauri-cli --version "^2"`
- **CMake**, **pkg-config**, and system build tools
- **GTK3, WebKit2GTK, ALSA** and other Tauri runtime dev libraries (see install script below)

Run the one-click install script:

```bash
chmod +x ./scripts/install_deps_linux.sh
bash ./scripts/install_deps_linux.sh
```

This script installs system dependencies, Node.js (if missing), appimagetool, and frontend npm dependencies.

Install frontend dependencies (if not using the script):

```bash
npm --prefix frontend ci
```

### 3. Third-party Sources

SoundTouch, WORLD and Signalsmith Stretch are all built from source at compile time. They are **auto-cloned** on first build — no manual steps required.

For offline builds, you can pre-clone them manually, for example SoundTouch:

```bash
cd backend/src-tauri/third_party/soundtouch-static
git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch
```

### 4. Development and Build

```bash
# Development mode (hot reload)
cd backend
cargo tauri dev

# Build Release
# Windows / macOS (default features: onnx + vslib)
cargo tauri build

# Linux AppImage (vslib is Windows-only; exclude the default feature)
cargo tauri build --bundles appimage -- --no-default-features --features onnx
# Or use the helper script: bash scripts/build-linux-appimage.sh

# Windows portable ZIP
.\scripts\pack-portable.ps1 -SkipBuild

# Windows VST3 ZIP + installer (a complete Release delivery already exists)
.\scripts\pack-portable.ps1 -PackageTarget Plugin -SkipBuild -Installer
```

Double-clicking `pack-portable.bat` lets you choose App, VST3 or both; see `scripts/pack-portable.ps1` and the scripts under `tools/` for the available switches.

You can switch the frontend startup mode via the `TAURI_UI_MODE` environment variable:

- `dev`: development mode (default, uses the Vite dev server with hot reload)
- `build`: build mode (builds the frontend static assets first, then starts)

Linux/macOS (bash/zsh):

```bash
cd backend
TAURI_UI_MODE=build cargo tauri dev
```

Windows PowerShell:

```powershell
cd backend
$env:TAURI_UI_MODE='build'; cargo tauri dev
```

**Note:** The first compilation will take a long time. Please be patient.

### 5. GPU Acceleration

HiFiShifter automatically enables GPU-accelerated inference on supported platforms. You can choose between Auto / CPU / GPU from the **Inference Device** menu in the menu bar, and compare per-device latency using **Run Benchmark...**.

| Platform                        | GPU Technology                        | Description                                                               |
| ------------------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| Windows x86_64 / ARM64          | DirectML (DirectX 12)                 | Proven, stable GPU path; supports NVIDIA / AMD / Intel Arc                |
| macOS ARM64 (Apple Silicon)     | CoreML + WebGPU (Dawn/Metal)          | CoreML leverages the Apple Neural Engine; WebGPU as a supplementary GPU backend |
| macOS x86_64 (Intel)            | —                                     | CPU only (uses the ort-tract alternative backend)                         |
| Linux x86_64                    | WebGPU (Dawn/Vulkan)                  | Dawn accesses the GPU through the Vulkan API; falls back to CPU if no GPU is present |
| Linux ARM64                     | —                                     | CPU only (no prebuilt WebGPU ONNX Runtime binary for this target)        |

> **Note**: WebGPU is not enabled on Windows. Its Dawn/D3D12 backend can cause native crashes on some GPU/driver combinations. DirectML is the mature, stable GPU path for Windows.

ONNX Runtime binaries are automatically downloaded by the ort crate at build time via the `download-binaries` feature — no manual setup needed. GPU providers (DirectML / WebGPU / CoreML) are enabled automatically at compile time for each target platform; no extra `--features` flags are required.

**WSL2 users**:

- WSL2 does not expose hardware Vulkan to the Linux guest. WebGPU/Dawn can only use Lavapipe (CPU software rendering), which is extremely slow. For GPU acceleration, use the Windows native build (DirectML) instead.
- Due to missing FUSE support, the Tauri bundler's linuxdeploy step may fail (error: `failed to run linuxdeploy`). This is a known WSL2 limitation and does not affect the actual AppImage output — the AppDir is correctly assembled at `target/release/bundle/appimage/`. You can set `APPIMAGE_EXTRACT_AND_RUN=1` and run `appimagetool` manually to package. This issue does not exist on real Linux machines or in CI.

## Logs and Troubleshooting

The app automatically writes its run log to the platform-standard log directory — no command-line flags required:

| OS | Log directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\com.arounder.hifishifter\logs` |
| macOS | `~/Library/Logs/com.arounder.hifishifter` |
| Linux | `~/.local/share/com.arounder.hifishifter/logs` |

- Inside the app, use **Help → Open Log Folder** to jump straight to the logs, or **Help → Export Diagnostics** to generate a diagnostics package (system info + all logs + inference-device benchmark results) to attach to an issue.
- Logs rotate automatically by size: 8 MiB per file, with up to 3 historical copies kept (`hifishifter.1.log` ~ `hifishifter.3.log`).
  - Frequently repeating errors / warnings are throttled automatically: a given log site emits at most one message per 10-second window, and suppressed messages are summarized in a `[throttled]` line before the next one.
- Frontend and backend errors are written to the same log file, so a single file tells the whole story.

Advanced options:

- Launch flag `--log-file=<path>`: write the log to a specific path; `--log-file=-` disables file logging.
- Environment variable `HIFISHIFTER_LOG=debug` (or `trace` / `info` / `warn` / `error`): adjust log verbosity.
- Environment variable `HIFISHIFTER_LOG_DIR`: override the default log directory.
- Run `HiFiShifter --benchmark` in a terminal: run the inference-device benchmark and print JSON results.

## Documentation

- User manual: [简体中文](../../docs/i18n/USERMANUAL.md) · [繁體中文](USERMANUAL_zh-TW.md) · [English](USERMANUAL_en.md) · [日本語](USERMANUAL_ja.md) · [한국어](USERMANUAL_ko.md)

For developers:

- Extension API: [docs/extension-api.md](../../docs/extension-api.md)
- UI text style guide (i18n Style Guide): [docs/i18n/style-guide.md](../../docs/i18n/style-guide.md)

## Acknowledgements

This project uses code or model architectures from the following open-source libraries:

- [WORLD](https://github.com/mmorise/World) - High-quality speech analysis and synthesis system
- [SoundTouch](https://www.surina.net/soundtouch/) - Audio time stretching and pitch shifting library (LGPL)
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) - High-quality audio time stretching library (MIT)
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) - Voice analysis and synthesis library
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) - Singing voice vocoder (OpenVPI)
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - High-fidelity GAN vocoder
- [vocal-remover (hnsep)](https://github.com/stakira/vocal-remover) - Harmonic / noise separation model (used for Harmonic Separation)

## License

This project is released under the [MIT License](../../LICENSE).
