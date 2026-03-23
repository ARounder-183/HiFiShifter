# HifiShifter

[中文](README.md) | [English](README_en.md)

HifiShifter is a graphical vocal editing and synthesis tool. It supports multi-track audio slicing and uses various vocoders—organized by track groups—to perform vocal pitch correction and "humanized" parameter tuning, achieving an integrated workflow for "Jinri" (Human-Vocaloid) production.

**Note: This project is still under active development. The full pipeline has not been exhaustively tested; you may encounter bugs or stability issues.**

## Installation

Please go to the **Releases** section on the sidebar of this repository to download and install the version compatible with your operating system.

## Basic Principles

HiFiShifter utilizes an offline rendering approach similar to UTAU. It processes, renders, and caches every audio slice on the timeline before feeding it into the playback system. This architecture ensures high processing efficiency for short audio clips.

HiFiShifter provides a unified rendering interface to allow for the addition of more algorithm support in the future.

## Recommended Workflow

We recommend the following workflow:

1.  Prepare the short audio slices (phonemes/samples) required for vocal synthesis using other DAWs or slicing software.
2.  Complete the arrangement (stitching) and tuning within HiFiShifter.

Additionally, HiFiShifter supports the following features for easy migration from other software:

1.  Open **VocalShifter** projects directly.
2.  Open **Reaper** projects directly.
3.  Parse **VocalShifter clipboard** data (supports pasting parameters from VocalShifter into the HiFiShifter parameter area).
4.  Parse **Reaper clipboard** data (supports pasting Reaper items directly into HiFiShifter).

## Feature Introduction

### Layout Overview

HiFiShifter is divided into two main functional areas: the **Track Panel** at the top and the **Parameter Panel** at the bottom. The Track Panel handles audio slicing and arrangement, while the Parameter Panel focuses on parameter automation and tuning.

### Track Panel

HiFiShifter offers a fully-featured track panel and audio slicing functionality, similar to most modern DAWs.

#### Audio Import

HiFiShifter supports three ways to import audio:

1.  Drag and drop audio files directly from the system file manager onto a track.
2.  Click the **Folder icon** in the toolbar to open the built-in file manager and drag audio onto a track.
3.  Press `Ctrl + F` to open **Quick Search** and select audio to import (the search path matches the current path of the built-in file manager).

#### Audio Editing

  - **Snap to Grid**: Clips snap to the grid by default during moving/trimming; hold `Shift` to temporarily disable snapping.
  - **Trim/Extend**: Drag the left or right edges of a clip to crop or extend it.
  - **Time Stretch**: Hold `Alt` + Left Click and drag the edges to stretch the audio.
  - **Slip-Edit**: Hold `Alt` + Left Click and drag the body of the clip to slide the internal audio content left or right.
  - **Fade In/Out**: Drag the top-left or top-right corners of a clip to adjust fade durations.
  - **Gain (dB)**: Drag the knob at the top-left of a clip (up/down) to adjust gain; the current dB value is displayed at the top-right.
  - **Mute Clip (M)**: Click the `M` button on the top-left to mute a specific clip (muted clips appear gray).
  - **Marquee Select**: Right-click and drag in the empty space of the timeline to select multiple clips.
  - **Copy & Drag**: Hold `Ctrl` while dragging a clip to create a copy at the destination.
  - **Glue**: Right-click a clip and select "Glue" (requires at least 2 clips on the same track).
  - **Split**: Select a clip and press `S` to split it at the playhead position.
  - **Copy/Paste**: Press `Ctrl + C` to copy clips to the internal clipboard. `Ctrl + V` aligns the leftmost start point of the copied clips to the playhead, maintaining relative spacing.

**Note:** Tracks support nesting. You can drag a track under another to make it a sub-track, forming a **Track Group**. Track groups are essential for the tuning process.

### Tempo Map (Variable BPM)

HiFiShifter supports Tempo Maps, allowing multiple tempo change points on the timeline.

  - **Toggle Tempo Track**: Click the **T** button next to the BPM input in the toolbar.
  - **Add Tempo Point**: Double-click on the tempo track to add a new point; it inherits the BPM and time signature of the current position.
  - **Edit Tempo Point**: Double-click a tempo flag to open a dialog for editing BPM (10–300) and time signature.
  - **Delete Tempo Point**: Hover over a flag to see the delete button (the first point cannot be deleted).
  - **Toolbar Sync**: When the tempo track is visible, entering a value in the toolbar BPM box updates/adds a point at the playhead.
  - **Grid Snapping**: In variable tempo mode, the grid and snapping automatically adapt to the bar positions.
  - **Clip Linking**: Updating tempo points automatically adjusts the `playback_rate` of audio slices in the affected area.

### Parameter Panel

The Parameter Panel offers operations similar to VocalShifter to facilitate user tuning.

**Important:** Each track has a special **"C"** button. Audio on a track can only be processed by the tuning algorithms if this button is enabled.

Tuning is performed per **Track Group**. By enabling "C" on the root track, the entire group shares a single algorithm and set of parameter lines. These parameters are applied to every audio slice based on its position.

Different algorithms provide different adjustable parameters, with **Pitch** being the universal parameter.

  - **Analysis**: Upon first opening, HiFiShifter analyzes the pitch of slices.
  - **Visuals**: Solid lines represent the current pitch of the group, dashed lines represent the original global pitch, and colored lines represent the original pitch of individual slices.
  - **Visibility**: Use the "eye" icon next to a panel to keep it visible even when not selected.

### Algorithms

HiFiShifter currently supports three processing algorithms:

#### World

A classic vocoder. Supports **Pitch** editing only.

#### PC-NSF-HIFIGAN

An OpenVPI open-source HiFi-GAN vocoder specialized for singing voices. Supports **Pitch, Breathiness (Aspiration), Tension, Formant, and Volume**.
*Note: Breathiness editing requires enabling the `hnsep` UVR model for separation, which may take time during initial processing. Tension editing requires Breathiness to be enabled.*

#### Vslib

The algorithm library provided by VocalShifter. Supports **Pitch, Pan, Formant, Volume, and Breathiness**.
*Note: Since the official DLL only supports File I/O, processing may take longer compared to the native VocalShifter app.*

## Quick Shortcut Reference

| Action | Shortcut / Mouse |
| :--- | :--- |
| Pan View (Timeline) | Middle Mouse Drag |
| Horizontal Zoom (Timeline) | Mouse Wheel (centered on cursor) |
| Vertical Zoom (Track Height) | Ctrl + Mouse Wheel |
| Vertical Zoom (Parameter Axis) | Ctrl + Mouse Wheel (inside param panel) |
| Play / Pause | Space |
| Play / Stop | Enter |
| Undo / Redo | Ctrl + Z / Ctrl + Y |
| New Project | Ctrl + N |
| Open Project | Ctrl + Shift + O |
| Save | Ctrl + S |
| Save As | Ctrl + Shift + S |
| Export Audio | Ctrl + E |
| Toggle Mode (Select / Draw) | Tab |
| Delete Selected Clip | Delete / Backspace |
| Copy Selected Clips | Ctrl + C |
| Paste at Playhead | Ctrl + V |
| Split Clip | S (at playhead position) |
| New Track | Ctrl + T |
| Quick Search | Ctrl + F |

## Development Environment

This section is for developers. Users may skip this.

### 1\. Clone the Repository

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
```

### 2\. Install Dependencies

Ensure you have the following installed:

  - **Node.js** (v18+ recommended) and npm
  - **Rust toolchain** (see `rust-toolchain.toml`)
  - **Tauri 2 CLI**: `cargo install tauri-cli --version "^2"`

Install frontend dependencies:

```bash
npm --prefix frontend install
```

### 3\. Quick Start (Dev Mode)

```bash
cd backend/src-tauri
cargo tauri dev
```

**Note:** The first compilation will take a significant amount of time.

## Acknowledgments

This project utilizes code or model architectures from the following open-source libraries:

  - [WORLD](https://github.com/mmorise/World) — High-quality speech analysis/synthesis system.
  - [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) — High-quality audio time-stretching library (MIT).
  - [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) — Voice analysis and synthesis library.
  - [SingingVocoders](https://github.com/openvpi/SingingVocoders) — Singing voice vocoders (OpenVPI).
  - [HiFi-GAN](https://github.com/jik876/hifi-gan) — High-fidelity generative adversarial network vocoder.

## License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).