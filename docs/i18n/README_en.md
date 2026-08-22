# HiFiShifter

[简体中文](../../README.md) | [繁體中文](README_zh-TW.md) | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

HiFiShifter is a graphical vocal editing and synthesis tool. It supports multi-track audio clip processing and uses various vocoders to achieve pitch correction and parameter adjustment for vocal, integrating splicing and tuning for Jinriki VOCALOID production.

**The project is still under active development. Full-chain testing has not been completed, so there may be many bugs or instability issues.**

![Preview](../preview.png)

## Installation

Download and install the appropriate release version for your system from the sidebar of the repository.

## Basic Principle

HiFiShifter uses an offline rendering approach similar to UTAU, processing, rendering, and caching each audio clip on the timeline before feeding it into the playback system, resulting in faster processing for short clips.

HiFiShifter provides a unified rendering interface to facilitate future algorithm additions.

## Recommended Workflow

Our recommended workflow is:

1. Prepare short clip sources needed for vocal using other DAWs or slicing software.
2. Complete audio splicing and tuning in HiFiShifter.

HiFiShifter also supports the following operations to facilitate migration from other software:

1. Directly open VocalShifter projects.
2. Directly open Reaper projects.
3. Parse VocalShifter clipboard content, allowing parameters from VocalShifter to be pasted into HiFiShifter's parameter area.
4. Parse Reaper clipboard content, allowing Reaper items to be pasted directly into HiFiShifter.

## Feature Introduction

### Layout

HiFiShifter can be roughly divided into two functional areas: the upper track panel and the lower parameter panel. The track panel is mainly responsible for audio clip processing, while the parameter panel handles parameter adjustments.

### Track Panel

HiFiShifter provides a fairly complete track panel and audio clip editing functionality, similar to most modern DAWs.

#### Importing Media (Audio / Video)

HiFiShifter supports three ways to import media files. Video files automatically use their audio track:

1. Drag and drop audio or video files from the system file manager directly onto a track.
2. Click the folder icon on the toolbar to open the built-in file browser and drag media files onto a track.
3. Press `Ctrl + F` to open quick search, select media files, and import them onto a track (the quick search file path matches the built-in file browser's current path).

#### Audio Editing

- **Snap to Grid**: Clip movement/cropping snaps to grid by default; hold `Shift` to temporarily disable snap.
- **Crop/Stretch Range**: Drag the left/right edges of a clip to crop or extend.
- **Time Stretch**: Hold `Alt` + left mouse button and drag the left/right edges of a clip to stretch the audio.
- **Slip-Edit**: Hold `Alt` + left mouse button and drag the main body of a clip to slide its internal content left or right.
- **Fade In/Out**: Drag the top-left/top-right corner of a clip to adjust fade in/out duration.
- **Gain (dB)**: Drag the knob at the top-left of a clip (up/down) to adjust gain; the current dB is displayed at the top-right.
- **Clip Mute (M)**: Click the `M` button at the top-left of a clip to mute it; the clip will turn grey.
- **Marquee Select**: Hold the right mouse button and drag in an empty area of the timeline to select multiple clips.
- **Copy Drag**: Hold `Ctrl` while dragging a clip to create a copy at the target position (the original clip remains unchanged; copying takes effect upon release).
- **Glue**: Right-click a clip and select "Glue" (requires at least 2 clips on the same track).
- **Split**: Select a clip and press `S` to split it at the playhead position.
- **Copy/Paste**: Select a clip and press `Ctrl + C` to copy it to the application clipboard. `Ctrl + V` aligns the leftmost start of the selected clips to the playhead position, preserving relative spacing.

Note that tracks support nesting: you can drag a track under another track to form a track group, which will be very useful during parameter adjustment.

### Parameter Panel

HiFiShifter's parameter panel provides operation support similar to VocalShifter for convenient parameter adjustment.

Note that there is a special `C` button on each track. Only when this button is pressed can audio on that track be processed by subsequent parameter adjustments.

During parameter adjustment, HiFiShifter operates on track groups. The root track's `C` button determines the algorithm and parameter curve shared by the entire group. The parameter curve applies to each audio clip based on its position.

Each algorithm in HiFiShifter offers different adjustable parameters; the common parameter is pitch.

When first opened, HiFiShifter takes some time to analyze the pitch of clips. After analysis, the solid line in the panel represents the group's current overall pitch, the dashed line represents the original overall pitch, and the colored lines represent each clip's own original pitch.

Other parameter panels are similar to the pitch panel but do not display individual clip original pitches.

The small eye icon next to a panel toggles its visibility when not selected.

### Algorithms

HiFiShifter currently supports three algorithms.

#### World Algorithm

A classic vocoder.  
Supports only `Pitch` editing.

#### PC-NSF-HiFiGAN

OpenVPI's open-source hifigan vocoder specialized for singing voices.  
Supports editing of `Pitch`, `Breath`, `Tension`, `Formant Shift`, and `Volume`.  
Note that breath editing requires additional enabling; it uses the hnsep UVR model for breath separation, which may take a long time on first use. If you need to edit tension, ensure breath is enabled.

#### Vslib

Algorithm library provided by VocalShifter.  
Supports editing of `Pitch`, `Pan`, `Formant Shift`, `Volume`, and `Breath`.  
Because the official DLL only supports file I/O, processing takes longer compared to VocalShifter itself.

## Common Shortcut Keys

| Action                              | Shortcut / Mouse                        |
| :---------------------------------- | :-------------------------------------- |
| Pan view (timeline)                 | Middle mouse button drag                |
| Horizontal zoom (timeline)          | Mouse wheel (centered on cursor)        |
| Vertical zoom (track height)        | Ctrl + Mouse wheel                      |
| Vertical zoom (parameter axis)      | Ctrl + Mouse wheel (in parameter panel) |
| Play / Pause                        | Space                                   |
| Play / Stop                         | Enter                                   |
| Undo / Redo                         | Ctrl + Z / Ctrl + Y                     |
| New Project                         | Ctrl + N                                |
| Open Project                        | Ctrl + Shift + O                        |
| Save                                | Ctrl + S                                |
| Save As                             | Ctrl + Shift + S                        |
| Export Audio                        | Ctrl + E                                |
| Toggle Mode (Select/Draw)           | Tab                                     |
| Delete Selected Clips               | Delete                                  |
| Copy Selected Clips (app clipboard) | Ctrl + C                                |
| Paste at Playhead                   | Ctrl + V                                |
| Copy Selection Curve (parameter)    | Ctrl + C (Select mode)                  |
| Paste to Selection Start            | Ctrl + V (Select mode)                  |
| Split Clip                          | S (splits selected clip at playhead)    |
| New Track                           | Ctrl + T                                |
| Quick Search                        | Ctrl + F                                |

## Development Environment Setup

This section is for developers; regular users can skip it.

### 1. Clone the Repository

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HiFiShifter
```

### 2. Install Dependencies

#### Windows

Make sure the following tools are installed:

- **Node.js** (recommended 18+) and npm
- **Rust toolchain** (see `rust-toolchain.toml`)
- **Tauri 2 CLI**: `cargo install tauri-cli --version "^2"`
- **CMake** (required to build the SoundTouch library)

ONNX Runtime (DirectML) is automatically downloaded by the ort crate at build time — no extra configuration needed.

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

- **Node.js** (recommended 20+) and npm
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

#### Linux AppImage Build

The `vslib` algorithm is Windows-only, so Linux builds must disable the default feature:

```bash
# Run from the backend/ directory (paths in tauri.conf.json are relative to it)
cd backend
cargo tauri build --bundles appimage -- --no-default-features --features onnx
```

Or use the provided helper script:

```bash
bash scripts/build-linux-appimage.sh
```

> **Note:** On WSL2, the Tauri bundler's linuxdeploy step may fail due to missing FUSE support (error: `failed to run linuxdeploy`). This is a known WSL2 limitation and does not affect the actual AppImage output — the AppDir is correctly assembled at `target/release/bundle/appimage/`. Set `APPIMAGE_EXTRACT_AND_RUN=1` and run `appimagetool` manually to package. This issue does not exist on real Linux machines or in CI.

### 3. SoundTouch Source

The SoundTouch audio time stretching library is built from source at compile time. It is **auto-cloned** on first build - no manual steps required.

For offline builds, you can pre-clone manually:

```bash
cd backend/src-tauri/third_party/soundtouch-static
git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch
```

### 4. GPU Acceleration

HiFiShifter automatically enables GPU-accelerated inference on supported platforms. You can choose between Auto / CPU / GPU from the **Inference Device** menu in the menu bar, and compare per-device latency using **Run Benchmark**.

| Platform                        | GPU Technology                        | Description                                                               |
| ------------------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| Windows x86_64 / ARM64          | DirectML (DirectX 12)                 | Proven, stable GPU path; supports NVIDIA / AMD / Intel Arc                |
| macOS ARM64 (Apple Silicon)     | CoreML + WebGPU (Dawn/Metal)          | CoreML leverages the Apple Neural Engine; WebGPU as a supplementary GPU backend |
| macOS x86_64 (Intel)            | —                                     | CPU only (uses the ort-tract alternative backend)                         |
| Linux x86_64                    | WebGPU (Dawn/Vulkan)                  | Dawn accesses the GPU through the Vulkan API; falls back to CPU if no GPU is present |
| Linux ARM64                     | —                                     | CPU only (no prebuilt WebGPU ONNX Runtime binary for this target)        |

> **Note**: WebGPU is not enabled on Windows. Its Dawn/D3D12 backend can cause native crashes on some GPU/driver combinations. DirectML is the mature, stable GPU path for Windows.
>
> **WSL2 users**: WSL2 does not expose hardware Vulkan to Linux guests. WebGPU/Dawn can only use Lavapipe (CPU software rendering), which is extremely slow. For GPU acceleration on WSL2, use the Windows native build with DirectML instead.

#### All Platforms

ONNX Runtime binaries are automatically downloaded by the ort crate at build time via the `download-binaries` feature — no manual setup needed. GPU providers (DirectML / WebGPU / CoreML) are enabled automatically at compile time for each target platform; no extra `--features` flags are required.

```bash
# Development mode (hot reload)
cd backend
cargo tauri dev

# Build Release
# Windows / macOS (default features: onnx + vslib)
cargo tauri build

# Linux (vslib is Windows-only; exclude default feature)
cargo tauri build --bundles appimage -- --no-default-features --features onnx

# Windows portable ZIP
.\scripts\pack-portable.ps1 -SkipBuild
```

## Quick Start

### Run Development Mode

```bash
cd backend/src-tauri
cargo tauri dev
```

You can switch frontend startup mode via `TAURI_UI_MODE`:

- `dev`: development mode (default, uses Vite dev server with hot reload)
- `build`: build mode (build frontend static assets first, then start)

Linux/macOS (bash/zsh):

```bash
cd backend/src-tauri
TAURI_UI_MODE=build cargo tauri dev
```

Windows PowerShell:

```powershell
cd backend/src-tauri
$env:TAURI_UI_MODE='build'; cargo tauri dev
```

**Note:** The first compilation will take a long time. Please be patient.

## Documentation

- [User Manual](USERMANUAL_en.md)
- [Todo List](../../todo.md)

## Acknowledgements

This project uses code or model architectures from the following open-source libraries:

- [WORLD](https://github.com/mmorise/World) - High-quality speech analysis and synthesis system
- [SoundTouch](https://www.surina.net/soundtouch/) - Audio time stretching and pitch shifting library (LGPL)
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) - High-quality audio time stretching library (MIT)
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) - Voice analysis and synthesis library
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) - Singing voice vocoder (OpenVPI)
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - High-fidelity GAN vocoder

## License

This project is released under the [MIT License](../../LICENSE).
