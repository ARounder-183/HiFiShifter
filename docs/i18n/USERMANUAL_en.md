# HiFiShifter User Manual

[简体中文](USERMANUAL.md) | [繁體中文](USERMANUAL_zh-TW.md) | [English](USERMANUAL_en.md) | [日本語](USERMANUAL_ja.md) | [한국어](USERMANUAL_ko.md)

HiFiShifter is a graphical vocal editing and synthesis tool. It supports multi-track audio clip processing and uses various vocoders to achieve pitch correction and parameter adjustment for human voice, integrating splicing and tuning for human VOCALOID production.

## 1. Installation

Download the HiFiShifter installer corresponding to your operating system and architecture. By OS, there are `Windows`, `macOS`, and `Linux`. By architecture, there are `x86_64` and `arm64`.

- For Windows, NSIS installer (`installer`) and portable zip (`portable`) are provided. General users can directly use the installer.  
  If you are a Windows user and do not know the difference between `x86_64` and `arm64`, choose `x86_64`. Only if you clearly understand `arm64` and have a Windows ARM device, you may download the `arm64` version.

- For macOS, an unsigned dmg installer is provided. Since it is not signed, installation requires a few extra steps to allow the app to run.  
  macOS users with M-series chips should install the `arm64` version. Only older Intel users need the `x86_64` version.  
  If you see a "file is damaged" error after double-clicking the dmg, follow these steps:
    1. Run `xattr -cr /Applications/HiFiShifter.app` in Terminal;
    2. Allow the app to run via `System Settings` -> `Privacy & Security` -> click `Open Anyway`.

- For Linux, an AppImage package is provided. You need to go to file `Properties -> Permissions` and check `Allow executing file as program`, then you can run it directly.

**About GPU Acceleration**: HiFiShifter provides multiple GPU acceleration options across platforms:

- **Windows (x86_64 / ARM64)**: DirectML (DirectX 12) — proven and stable, supports NVIDIA / AMD / Intel Arc GPUs
- **macOS (Apple Silicon)**: CoreML + WebGPU (Dawn/Metal) — CoreML leverages the Apple Neural Engine; WebGPU serves as a supplementary backend
- **macOS (Intel)**: CPU inference only (uses the ort-tract alternative backend, no GPU acceleration)
- **Linux (x86_64)**: WebGPU (Dawn/Vulkan) — Dawn accesses the GPU through the Vulkan API; falls back to CPU if no GPU is present
- **Linux (ARM64)**: CPU inference only (no prebuilt WebGPU ONNX Runtime binary for this target)

In the menu `Options → Inference Device`, you can select `Auto`, `CPU`, or `GPU`. Run the benchmark to compare per-device inference latency and pick the fastest option.

**WebView Information**: HiFiShifter is built with the Rust + Tauri framework and requires a WebView component to display its interface.

- **Windows**: Requires Edge WebView2. Windows 10 (version 1803 and later) and Windows 11 have it preinstalled, so no additional action is needed. If you are using an older Windows version or the component is missing, the installer will prompt you to download it automatically. You can also refer to the [Tauri official documentation](https://tauri.app/start/prerequisites/#webview2) for details. General users can simply run the installer without worry.
- **macOS**: WebKit is provided by the system, no extra installation is required.
- **Linux**: Requires WebKitGTK. Most major distributions (e.g., Ubuntu, Fedora, Arch Linux) include it by default. If you see a missing component error, use your package manager to install `webkit2gtk` (e.g., `sudo apt install webkit2gtk`). Refer to your distribution's documentation for specifics.

On Windows, HiFiShifter disables the browser shortcuts that WebView2 normally intercepts (such as `Ctrl + F`, `Ctrl + P`, `F5`, and `Ctrl` with `+` / `-` for page zoom), so those key combinations reach HiFiShifter normally and never interrupt your editing.

**About the VST3 / ARA Plugin**: Besides the standalone app, HiFiShifter is also available as a VST3 (ARA) plugin, which opens the same editing interface directly inside a REAPER project. The plugin is currently a Windows x64 build and has only been verified against REAPER; see the [VST3 / ARA Plugin](#8-vst3--ara-plugin-reaper) chapter for installation and usage.

## 2. Menu

The menu bar consists of six menus from left to right — `File`, `Edit`, `Track`, `View`, `Options` and `Help` — plus a `Language` switcher at the far right, which switches the interface language between English / 简体中文 / 繁體中文 / 日本語 / 한국어.

The `File` menu lets you create, open and save HiFiShifter project files (`New Project`, `Open Project...`, `Recent`, `Save`, `Save As...`), and also import media files (audio or video), import MIDI files, import external projects, export audio, configure auto backup, record, and more. Audio export supports three formats — `wav`, `mp3`, and `flac`. See the [Export Audio](#6-export-audio) section for details.

HiFiShifter project files have the extensions `.hshp` or `.hsp`. Additionally, `Save As` supports saving the project as a plain text `json` file, or packaging the current project together with all used media files into an archive zip `.zip`.

`File → Import Media File` can import common audio formats (`wav`, `mp3`, `flac`, `ogg`, `m4a`, `aac`, etc.) and common video containers (`mp4`, `mov`, `mkv`, `webm`, `avi`, `wmv`, `ts`, `mpg`, etc.). When several media files are selected at once, an import-mode dialog appears where you can choose `Add Across Time` (laid out one after another along the timeline), `Add Across Tracks` (stacked vertically across multiple tracks), or `Add as Takes` (merged as multiple Takes of a single audio clip). If a media file contains multiple audio streams (such as a video with multi-language audio tracks), you can also choose which audio stream to import.

`File → Recapture Missing Media...` re-locates media files in the project after they have been moved or lost.

The automatic backup feature allows you to configure backups for your project files, with two modes: `Backup on save` and `Timed backups`.

- `Backup on save`: When you overwrite the project file via save, the previous project file will automatically be renamed to a backup file with `-bak` appended to the original filename (for example, `.hshp-bak` or `.hsp-bak`). Enabled by default.
- `Timed backups`: Automatically create backup project files at the interval and path you set while you edit the project. Disabled by default.

### Importing a HiFiShifter Project

`File → Import External Projects → Import HiFiShifter Project...` merges all content from another `.hshp` / `.hsp` / `.json` project into the current project without closing it. The same submenu also holds `Import Reaper Project...` (`.rpp`) and `Import VocalShifter Project...` (`.vshp` or `.vsp`).

- `Keep original timeline position`: Imported clips stay at their original timeline positions.
- `Place at playhead`: The imported content is shifted so its earliest clip starts at the current playhead.
- `Import tempo map`: Only available when the current project has no tempo map. After import, the source project's initial BPM, time signature and scale become the current project's baseline.
- All tracks, child tracks, clips, parameter curves and group relationships are assigned fresh IDs, so they never collide with the current project. Audio source files are resolved relative to the imported project file first; missing files still trigger the interactive relink dialog.
- Notes from the imported project are appended to the current project notebook instead of replacing it.

### Cross-Process Copy / Cut / Paste

HiFiShifter's structured copy/paste is stored in the system clipboard, so it supports cross-process copy and paste.

Copy, cut, and paste automatically act on whatever you are currently working with — there is no mode to switch manually:

- When the timeline has focus and clips are selected, `Ctrl + C` / `Ctrl + X` / `Ctrl + V` operate on clips.
- When the Parameter Editor has focus and a parameter curve segment is selected, the same three shortcuts operate on that curve. In that context `Ctrl + X` is `Cut Parameter Frames`: it copies the selected curve to the clipboard and clears it so you can paste it elsewhere. `BackSpace` (initialize) simply resets the selection to its default state and does not touch the clipboard.
- Pasting also inspects the actual clipboard contents, so even if you switch tracks in between, parameter curves are still pasted onto the correct track.

- Select clips in the timeline and press `Ctrl + C` (or right-click `Copy`), then press `Ctrl + V` (or right-click empty track space and select `Paste`) in another process's project to paste the clips along with their parameter curves.
- When you copy clips in HiFiShifter, the data is also serialized to the Reaper clipboard, so you can press `Ctrl + V` in Reaper to paste it. Clips without a usable source file are skipped.
- `Edit → Paste as New Tracks` (`Ctrl + Alt + V`): force-creates new root-track groups using the source hierarchy.

- **Paste Reaper Clipboard Data**: After you copy Items, tracks, or MIDI notes in Reaper, a plain paste in HiFiShifter automatically recognizes and imports the Reaper clipboard data.
    - Item data: Imports as audio clips in HiFiShifter, preserving tuning data from Reaper (overall tuning and pitch envelopes alike).
    - Track data: Imports tracks along with their items as tracks and audio clips in HiFiShifter, preserving track groups.
    - MIDI note data: After exporting note data from other DAWs (Reaper, FL Studio, etc.) to the clipboard as MIDI note data, use the `Select` tool in the Parameter Editor to select a pitch curve segment in HiFiShifter, then you can import the clipboard MIDI note data into that segment. For a detailed introduction to MIDI import, see the [Pitch Reference Clip](#8-pitch-reference-clip) section.

- **Reaper Envelopes**: Whether you import a Reaper project (`.rpp`), paste Reaper clipboard data, or copy content from HiFiShifter back into Reaper, Reaper take envelopes and track envelopes (volume, pan, mute) are carried along and converted to or from the matching HiFiShifter parameter curves. Note that pitch envelopes are only exported when the root track of the corresponding track group has Compose (`C`) enabled, which keeps the exported data consistent with what you actually hear.

- **Paste VocalShifter Clipboard Data**: After you copy parameter curves, audio clips, or tracks in VocalShifter or VocalShifter LE, this function quickly imports the VocalShifter clipboard data into HiFiShifter.
    - Parameter curve data: After selecting a parameter curve segment with the `Select` tool in the Parameter Editor, you can import VocalShifter clipboard parameter curve data into that segment.
    - Audio clip data: Imports as audio clips in HiFiShifter, preserving various parameter curve data.
    - Track data: Imports tracks along with their audio clips into HiFiShifter. Note that HiFiShifter currently cannot distinguish whether your last copied content was an audio clip or a track. If you intend to import a track, before performing the copy track operation in VocalShifter, ensure that no audio clip is selected in the VocalShifter project; otherwise, only the selected audio clips will be imported.

The `Edit` menu gathers the common clipboard and selection operations: `Undo` / `Redo`, `Cut` / `Copy` / `Paste`, `Paste as New Tracks`, and `Paste VocalShifter Clipboard` (Windows builds only). Selection operations include `Select All`, `Deselect`, plus `Add Clip Ranges to Param Selection` / `Remove Clip Ranges from Param Selection` (see the [Track View](#3-track-view) chapter).

The `Track` menu provides `Add Track` (inserted below the currently selected track), `Clone Selected Track` and `Remove Selected Track`.

The `View` menu contains options related to the interface display:

- `Window` / `Layout`: manage the visibility of the dock panels and the layout presets, import/export, reset, etc.
- `Tempo Map`: Show or hide the Tempo Map row (enabled by default; the row is not shown when the project has no Tempo Map data).
- `Show all takes (when room)`: Toggle the expanded display of multi-take clips.
- `Clipboard Preview`: Toggle the parameter editor clipboard preview.
- `Popup Param Values`: Toggle the parameter value popup.
- `Time Display`: Lets you choose the primary/secondary time units of the timeline ruler and open `Timeline Display Settings...`.
- `Theme: Auto / Dark / Light`: Switch the current theme. With `Auto`, HiFiShifter follows your operating system's light/dark appearance and switches automatically (this is the default).
- `Appearance Settings...`: Open the appearance settings panel.
- `Connect to ARA Host...`: Connect to a running ARA host (REAPER) instance to edit the host's audio from the standalone app. Shown in the standalone app only; it opens a floating session panel where you can submit, refresh or disconnect. See the [VST3 / ARA Plugin](#8-vst3--ara-plugin-reaper) chapter for details.
- `Refresh`: Reload runtime information.
- `Clear Waveform Cache`: Clear the cached waveform data; it is regenerated the next time it is displayed.

The `Options` menu allows you to modify various settings of HiFiShifter:

- `Project Stretch Override`: Allows you to modify the current project's stretching algorithm and HiFiGAN mel stretch switch.
- `Global Stretch Default`: Allows you to modify the default global stretching algorithm and HiFiGAN mel stretch switch.
- `Inference Device`: Allows you to set the inference device used for rendering. Currently supports `Auto`, `CPU`, and `GPU`; when `GPU` is selected, you can also pick a specific card from the `GPU Device` list. You can run a benchmark from this menu to test the performance of each device (the benchmark will show specific backends such as GPU (DirectML), GPU (WebGPU), etc.). `GPU` is only available in the corresponding GPU build of HiFiShifter.
- `Background Pre-render`: When enabled, after opening a project or editing parameters, the edited parameters are automatically pre-rendered in the background, and you can play the already-rendered portions even while rendering is still in progress. When disabled, rendering only begins when playback starts, and you must wait for rendering to complete before the timeline plays normally. Enabled by default. Disabling it reduces rendering frequency and saves performance.
- `Sync Edits Across Takes`: When enabled, edits to a take (gain, trim, rate, reverse, loop, etc.) are also applied to all other takes of the same clip (enabled by default).
- `Automatically reload modified media files`: When enabled, media files in the project that have been modified externally are reloaded automatically (enabled by default).
- `Enable loop for new clips`: When enabled, newly imported or created clips have looping enabled by default (enabled by default).
- `Snap/Grid Settings...`: Opens the [Snap / Grid Settings](#snap--grid-settings) dialog (it can also be opened from the `Snap` button's right-click context menu on the timeline toolbar).
- `Search and matching...`: Configures the search and matching rules shared by the file browser and Quick Search (case sensitivity, regular expressions, etc.).
- `Render Cache Manager...`: View and clean up the render cache.
- `Vibrato Preset Manager...`: Opens the [Vibrato Preset](#6-vibrato-presets) manager.
- `Import Channel Handling...`: Sets how the number of channels is handled when importing media files (e.g. converting fake stereo to mono).
- `Pointer & Pen Input...`: Sets the input preferences for trackpads, drawing tablets, styluses, touch screens and other pointer devices.
- `Keyboard Shortcuts...`: Allows you to configure HiFiShifter's keybindings. Several presets are available.

The `Help` menu provides diagnostics and support entries:

- `Open Log Folder`: Opens the folder containing the run log in the system file manager. See the [Logs and Troubleshooting](#9-logs-and-troubleshooting) section.
- `Export Diagnostics...`: Lets you pick a save location, then generates a diagnostics package (system info + all logs + inference-device benchmark results) that you can attach to an issue.
- `About HiFiShifter`: Opens an About dialog showing the project introduction, the current version, and the project homepage link.

## 3. Track View

The general operation logic and shortcuts can be referenced from DAWs like Reaper, VocalShifter, VEGAS Pro. You can customize your shortcut preferences via `Options -> Keyboard Shortcuts...`. The following descriptions are based on default shortcuts.

The track view is one of HiFiShifter's core features, allowing you to crop, splice, and edit audio clips. Its operation logic is largely based on Reaper.

For view navigation, drag the middle mouse button (hold the scroll wheel) to pan. Horizontal/vertical zoom or scrolling can be done by holding modifiers like `Ctrl`, `Alt`, `Shift` while scrolling the mouse wheel. These modifiers can be adjusted in the shortcut settings.

If you move the pointer over a scrollbar at the edge of the view and then scroll the wheel, only that scrollbar's axis scrolls (vertical bar = up/down, horizontal bar = left/right) instead of triggering zoom. Hold the zoom modifier (default `Alt`) at the same time and the wheel zooms that axis instead: the vertical bar zooms track height, the horizontal bar zooms the timeline, both anchored at the pointer.

Common shortcuts:

> **macOS users**: `Ctrl` below corresponds to `Command (⌘)` and `Alt` to `Option (⌥)`.

- `Space`: Play / Pause (does not return to start)
- `Enter`: Play / Stop (returns to start)
- `K`: Toggle the metronome
- `S`: Split
- `G`: Group
- `U`: Ungroup
- `T`: Cycle to the next Take (`Shift + T`: previous)
- `Ctrl + C`: Copy (also writes REAPERMedia data, so it can be pasted directly in REAPER)
- `Ctrl + X`: Cut
- `Ctrl + V`: Paste
- `Ctrl + Alt + V`: Paste as New Tracks
- `Ctrl + Z`: Undo
- `Ctrl + Shift + Z` / `Ctrl + Y`: Redo (macOS: `⌘ + ⇧ + Z` / `⌘ + Y`)
- `Ctrl + A`: Select All

- `Delete`: Delete audio clip
- `-` / `=`: Shift parameter curve down/up for selected clips (hold `Shift` for a large step, `Ctrl` for a fine step)
- Modifier `Alt`: Hold while dragging clip start/end to stretch the clip; drag the middle of the clip to slip-edit (internal content offset)
- Modifier `Shift`: Hold to temporarily toggle snap
- Modifier `Ctrl`: Hold while dragging a clip to copy it
- Modifier `Alt + Shift`: Hold and drag vertically on a clip to shift the pitch of the whole clip; the pitch curve in the Parameter Editor follows in real time so you can see the result while dragging

Double-clicking a clip in the timeline selects all parameter frames within that clip's range and moves the copy/cut focus to the Parameter Editor, which makes it quick to process an entire clip at once.

Hold `Alt` (configurable under `Options -> Keyboard Shortcuts`) and double-click a clip to **add** its range to the Parameter Editor selection (overlapping or touching ranges merge); repeat the same gesture on that clip to **remove** its range again. That makes it quick to assemble several clips' time ranges into one multi-range selection and process them in a single pass. For larger sets, use `Add Range to Parameter Selection` / `Remove Range from Parameter Selection` in the clip context menu, or press `Ctrl + Shift + A` (`Alt` is `Option` and the shortcut is `⌘ + ⇧ + A` on macOS) to add the ranges of all currently selected clips. These operations only act on clips belonging to the **root track group that the Parameter Editor is showing** — the editor displays one root track's parameters at a time, so clips on other tracks are ignored.

The small circle at the top-left of a clip is a volume adjustment knob, the `M` button can mute that clip individually, and the `F` button can open that clip's formant editing menu. The left and right edges of a clip allow adjusting fade-in/fade-out envelope lengths.

Right-click a clip to open the context menu, which includes `Delete`, `Mute` / `Unmute`, `Take` (submenu, includes `Replace`), `Channel Mode` (submenu for swapping left/right channels, converting to mono, etc., plus `Scan for Fake Stereo and Convert`), `Reverse`, `Loop`, `Copy` / `Cut`, `Split at Playhead`, `Normalize`, `Add Range to Parameter Selection`, `Edit Playback Rate...`, `Silence Detection...`, `Convert to Pitch Reference Clip` (a Pitch Reference Clip shows `Update Pitch Reference` instead), `Quick Export`, `Export as MIDI`, `Grouping` (submenu with `Group` / `Ungroup` / `Glue`), and the `Fade In` / `Fade Out` shape button rows at the bottom of the menu. Renaming is not done through this menu — double-click the clip's name area instead (it renames the active Take). If you select multiple clips on the same track, the context menu allows `Glue` to merge them into a single audio clip.

Select multiple clips, then choose `Group` (or press `G`) in the context menu to group them. Similar to Reaper or VEGAS Pro, clips in the same group are linked during edits. Click the chain button at the top-left of a clip to temporarily disable or enable the group's linked editing. Select grouped clips and choose `Ungroup` (or press `U`) to remove them from the group.

### Take

An audio clip can contain multiple Takes (similar to Reaper's Takes), for comparing or switching between different recordings / source material at the same position.

- Adding / managing Takes: right-click a clip and use the `Take` submenu for `Add Take from Media...`, `Duplicate Current Take`, `Rename Take`, `Delete Take`, etc.; when a clip contains multiple Takes, you can also click an entry in the submenu to switch the active Take, or use `Switch to Next Take` / `Switch to Previous Take` (shortcuts `T` / `Shift + T`, which cycle). `Explode Takes into Clips` splits each Take of the clip into separate clips on the timeline. `Replace` also lives in this submenu and swaps the current Take's audio source with a new media file. Each Take row also has its own `Reverse` and `Channel Mode` mini buttons on the right, which act on that single Take.
- Packing Takes across tracks: after selecting multiple clips on different tracks, the context menu's `Pack Clips into Takes` merges them into multiple Takes under a single clip.
- `View -> Show all takes (when room)`: when enabled, multi-take clips are laid out showing all Takes when the track is tall enough; otherwise only the active Take is shown.
- `Options -> Sync Edits Across Takes`: when enabled, edits to any Take (gain, trim, rate, reverse, loop, etc.) are also applied to all other Takes of the same clip.

On the left side of the track view is the track header area, where you can add or delete tracks, adjust track parameters, etc. Right-click a track to clone it; right-click empty track space to paste clips from the clipboard.

Similar to Reaper, HiFiShifter tracks support track groups. Drag one track header onto another in the track header area to create a track group. A track group shares a single parameter panel. In practice, it is recommended to organize by "one voice part per track group".

Track view toolbar buttons (from left to right):

- `Metronome`: Provides a click reference during playback. Left-click to toggle it on/off; right-click opens volume, subdivision, timbre and other settings. See the [Metronome](#metronome) section for details.
- `BPM`: Adjust the global tempo BPM of the project (with a Tempo Map active, this adjusts the BPM of the change point nearest the playhead — see "Tempo Map" below).
- `Time Sig.`: Sets the project time signature.
- `Grid`: Set the grid spacing for the project.
- `Base Scale`: Adjust the global base scale setting for the project, supports custom scales. The scale function is mainly used with `Pitch Snap` and other pitch-related adjustments.
- `Stop` button and `Play / Pause` button: Control playback.
- `Record`: Allows recording on the currently selected track. Right-click to set the recording source and device, or open the detailed recording settings.
- `Undo` / `Redo`: Same as their namesakes in the Edit menu; right-clicking the buttons opens the `Operation History` panel, where you can inspect recent operations and jump to any of them.
- `File Browser`: Open the HiFiShifter file browser window.
- `Notepad`: Open the HiFiShifter notepad window, which records and displays Markdown-formatted text.
- `Auto Crossfade`: Similar to Reaper/VEGAS Pro, when enabled, moving clips that overlap will automatically adjust crossfade envelopes.
- `Split Transition`: Modeled after Reaper/VEGAS Pro split fades, enabled by default. After splitting, it automatically adds a fade-out to the preceding clip and a fade-in to the following clip at the split point, or extends them with overlap to make the transition between clips smoother. Left-click toggles it; right-click opens the detailed settings.
- `Snap`: When enabled, clip adjustments attempt to snap to the grid, other object edges, the playhead and other configured targets. Hold `Shift` to temporarily toggle snap.
- `Ripple Editing (Auto Follow)`: When enabled, certain edits to clips on the track are automatically followed through (ripple). Left-click cycles through the three modes `Off`, `Per Track` and `All Tracks`; right-click lets you pick the mode directly.
- `Zoom at Playhead`: When enabled, horizontal zoom always centers on the playhead, similar to Reaper / VEGAS Pro; when disabled, it centers on the mouse cursor.
- `Auto Scroll`: When enabled, the view automatically scrolls horizontally during playback to follow the playhead.
- `Allow Param Editor to Move Playhead`: When disabled, clicking in the parameter editor will not move the playhead; only clicking the track view or the timecode area of the parameter editor moves the playhead.
- `Allow Timeline Clicks to Switch Track`: Enabled by default. When enabled, clicking a clip or empty area in the timeline switches the current track, and the parameter editor follows the newly selected track. When disabled, only clicking a track header changes the current track.
- `Ignore Grouping`: When enabled, edits to grouped audio clips will globally ignore group-linked editing.

### Snap / Grid Settings

The `Snap` button on the timeline toolbar is the quick toggle for snapping. Left-click toggles the snap master switch; right-click opens the `Snap/Grid Settings...` dialog directly. The same dialog is available from `Options -> Snap/Grid Settings...`. All settings are persisted and restored on the next launch.

Settings:

- `Grid`: show/hide grid lines, choose the grid spacing (normal / dotted / triplet, from `1/1` to `1/64`), and set a minimum pixel spacing so dense grids stay readable. With `Swing` enabled, odd grid lines are shifted by a percentage (0–100%); when `Adjust all clips when changing swing` is checked, existing clips are automatically re-aligned to the new swing grid.
- `Snap Master`: `Enable snapping` is the master switch; `Show snap highlight lines while dragging` (on by default) shows a vertical guide line when a drag hits a snap target; `Snap distance` defines how close (in pixels, default 4) the pointer must be to a target for snapping to trigger; `Snap relative to grid` preserves the clip's original offset relative to the grid instead of snapping to absolute grid positions.
- `Snap Targets / Objects`: independently configure whether `Clips` (audio clips), `Selection`, and `Cursor` snap to `selection / markers / cursor` or to the grid. For example, clips can snap only to the grid while the playhead snaps to both the grid and clip edges.
- `Snap Behavior`: `Snap settings follow grid visibility` stops snapping to the grid when grid lines are hidden; `Snap to grid at any distance` is an aggressive mode that always forces the drag to the grid; `Use independent snap spacing` provides a dedicated snap spacing (separate from the display grid) with its own minimum pixel value.
- `Clip & Special Interactions`: choose whether only clip start/end edges snap or the `snap offset` (content start) also snaps; enable `Snap clips across tracks` and set how many tracks away targets are considered; `Snap razor edits` applies snapping to split operations (`S`).
- `Advanced`: `Snap to project sample rate` provides sample-accurate snapping; `Snap clip edges to source media start/end` pulls trimmed clip edges back to the original source start/end; `Force selections to be multiples of` rounds marquee selections to the selected grid (the `Selection multiple` dropdown chooses how many grid units to round to); `Use the same grid division in arrange view and MIDI editor` keeps the timeline and parameter editor grid precision synchronized.

### Split Transition

`Split Transition` is designed to reduce clicks at split points and is enabled by default. Due to time-stretching algorithms and similar factors, clips can click at the newly created boundary after a split. This feature follows the approach used by Reaper/VEGAS Pro and automatically handles the boundary after every split.

The `Split Transition` toolbar button is located to the right of `Auto Crossfade`. Left-click toggles the feature; right-click opens detailed settings.

- `Fades Only`: After splitting, automatically adds a fade-out of length X to the left clip and a fade-in of length X to the right clip. The two clips do not overlap.
- `Extend & Overlap` (default): After splitting, automatically extends the left clip's tail forward by X and the right clip's head backward by X, creating a 2X-second overlap. The extension keeps the source material at the same timeline position and correctly accounts for playback rate. Extensions are clamped to the clip source's actual length. When `Auto Crossfade` is also enabled, a crossfade is automatically created across this overlap.
- `Transition Length X`: The fade/overlap length used by both modes, 0.01 seconds by default.
- `Transition Length Unit`: Choose `Seconds` (default) or `Percent`. Percent defaults to 1 and is calculated from the combined full length of the two clips after the split; for example, two clips totaling 10 seconds at 1% gives 0.1 seconds.
- `Fade Curve`: Selects the fade curve written to the boundary fades by split transitions. The default is `Keep Original Fade Curve`.
- `Overlap Crossfade`: With `Follow Auto Crossfade`, crossfades are only added to the overlap when `Auto Crossfade` is enabled. With `Always Apply`, crossfades are always added to the overlap.

### Clip Fades & Crossfades

Every clip carries its own fade-in / fade-out envelope, and the interaction model matches REAPER almost exactly: what you edit is not the whole shaded fade area, but the two lines actually visible on screen — the fade envelope line, and the vertical edge line of the fade region.

**Adjusting fade length**

- Drag the fade envelope line (or the vertical edge line of the fade region) at the left / right end of a clip to change the fade-in / fade-out length. To create a fade from zero, drag the small triangle at the top corner of the waveform area, *below* the name bar - that triangle is shown only while the side has no fade yet, and once a fade exists the envelope line itself becomes the handle. The knob and badges on the name bar are clip controls and do not accept fade gestures.
- Clicking a fade control (without dragging) moves the playhead to the corresponding position: the envelope line addresses the inner edge of the fade region (fade-in → its right edge, fade-out → its left edge), the crossfade grip the clicked position, and a clip edge that edge's exact position.
- Hovering over fade controls (envelope line, edge line, crossfade grip) shows a tooltip with the fade-in / fade-out type (curve icon), the length (in the primary / secondary time units from the time display settings), and the curvature.

**Fade shapes and curvature**

HiFiShifter provides the same 7 fade shapes as REAPER: `Linear`, `Fast Start`, `Fast End`, `Fast Start Steep`, `Fast End Steep`, `Slow Start/End (S-Curve)`, and `Slow Start/End Steep`.

- Right-click a fade envelope line or a fade region edge line to open the fade-specific context menu: seven shape buttons (current shape highlighted) on top, and a curvature slider with a mini curve preview below. Curvature ranges from `-1.00` to `+1.00` and applies in real time. The slider supports wheel stepping (hold the `Fine Adjust` modifier for finer steps), and the mini curve itself can be dragged directly.
- Hold the `Fade Curvature` modifier (default `Alt`) and drag the envelope line to "pull" the curve through the pointer position.
- Double-click the envelope line to reset that side's curvature to the default of the current shape; switching shapes also resets the curvature to the new shape's default.
- Hold the `Cycle Fade Shape` modifier (default `Ctrl`) and click the envelope line to cycle through fade shapes.
- The clip context menu also contains `Fade In` / `Fade Out` shape rows for switching fade shapes directly.

**Auto crossfade**

The `Auto Crossfade` toolbar button (enabled by default) behaves like its REAPER / VEGAS Pro counterparts: when moving clips creates an overlap on the same track, a crossfade covering the overlap is applied to both clips automatically.

- Auto-crossfade lengths are stored separately from manual fade lengths: when the overlap disappears, the auto fades are cleared and the previous manual fades are restored; conversely, manually dragging a fade length converts that side into a manual fade.
- When one clip fully contains another, the overlap is not considered a valid crossfade relationship and no auto crossfade is triggered.
- The fade lengths shown on the timeline and in the parameter editor are always the "effective" fades: auto crossfade takes priority over manual fades.

**Editing in the overlap region**

When two clips overlap, the overlap region provides full editing controls for both clips, with the same interactions as regular clips: the left side of the overlap belongs to the later clip (its left edge for trim / extend, plus its fade-in controls), and the right side belongs to the earlier clip (its right edge, plus its fade-out controls). Clip edges take priority, then fade region edge lines, then envelope lines.

Where the two fade curves intersect there is a grip (the crossfade grip):

- Dragging the grip moves the crossfade as a whole while keeping the overlap length unchanged;
- Holding the `Crossfade Reverse Mode` modifier (default `Ctrl`) while dragging moves the two clips' edges in opposite directions, changing the overlap length and scaling the fade lengths proportionally;
- Holding the `Fade Curvature` modifier (default `Alt`) while dragging adjusts both fade curves' curvature at once;
- Right-clicking the grip opens a two-column fade menu (earlier clip's fade-out and later clip's fade-in) for setting both shapes and curvatures independently; double-clicking the grip resets both sides' curvature at once.

All of these modifiers can be reassigned in `Options -> Keyboard Shortcuts...` under the `Modifiers · Fades & Crossfades` group.

### Timeline Time Display

The ruler at the top of the timeline automatically refines its tick labels as you zoom horizontally: at a small zoom level only bars are shown (`1.1`, `2.1`); zooming in progressively refines to half notes (`1.1`, `1.3`), quarter notes (`1.1`, `1.2`), eighth notes (`1.1`, `1.1.500`), and further to 16th/32nd notes. The finest precision is limited by the `Grid` setting.

Four time units are supported:

- `Bar.Beat.Subdivision` (default primary unit): `1.2.500` means bar 1, beat 2, plus 0.5 beat (1000 subdivisions = 1 beat).
- `Bar.Division`: `1.17/32` means the 17th division of bar 1 using a `1/32` grid. Divisions follow the `Grid` setting; triplet grids produce integer division counts (e.g. `1.2/12`), while dotted grids may produce fractional counts (e.g. `1.2/2.6667`).
- `Seconds`: absolute seconds, e.g. `1234.5678`.
- `H:MM:SS.mmm` (default secondary unit): the hour is omitted when zero and milliseconds always use 3 digits (e.g. `4:43.750`, `1:4:43.750`).

Right-click the ruler to choose the primary and secondary time units; the secondary unit can also be set to `Not Used`. The same controls are available in `View -> Time Display` and the `Timeline Display Settings...` dialog. When both units are shown, they appear as two rows separated by a short faint line; when the secondary unit is `Not Used` or identical to the primary unit, only the primary unit is shown, vertically centered.

The `TRACKS` header row on the left side of the track view shows the live playhead time on its right, formatted as `primary / secondary` (only the primary unit when no secondary is used), and refreshes automatically during playback and when the time format changes. The playhead time text is kept at fixed digit alignment (e.g. `1.1.000`, `0.000`) for easy reading.

Moving the mouse over the ruler shows the time at the pointer. The right-click menu also offers `Copy Playhead Time`, which copies the current playhead time as text to the clipboard. `Timeline Display Settings...` additionally lets you adjust the ruler label spacing and toggle the playhead time display in the track header.

### Tempo Map (Tempo / Time Signature / Scale Map)

HiFiShifter supports a project-level Tempo Map that lets you define different BPM, time signatures and scales at different positions of the timeline. A blank project has no Tempo Map data by default; once you add Tempo Map data to the project, the ruler automatically shows an extra Tempo Map row below the time units (separated by a short faint line), with each segment's tempo, time signature and scale shown on the point labels. When a segment's starting label scrolls out of view on the left, a floating label in the same style appears at the far left of the row showing that segment's parameters; it switches smoothly while scrolling horizontally, never overlaps the labels, and offers exactly the same interactions as the fixed labels (double-click to enter inline editing, right-click to open the edit dialog). The parameter editor ruler shows the same Tempo Map row.

- Adding points: right-click the ruler and use `Add Tempo / Time Signature / Scale Change Here...` in the menu to create a point at the clicked position and open the edit dialog. Double-clicking an empty area of the Tempo Map row creates a point right there.
- Editing points: double-click a point label on the Tempo Map row to modify the BPM, time signature and scale. The label turns into an inline text box where you can directly type text such as `120 4/4 - C / Am`. Right-clicking a label opens the `Tempo Map Point` edit window directly. Point labels can also be dragged to adjust their position; the ruler context menu's `Edit This Point...` / `Delete This Point` act on the point under the cursor, and `Clear Tempo Map` empties (or only hides) the whole Tempo Map.
- Follow the previous time signature / scale: each point's time signature can be set to `Follow Previous Time Signature` and its scale to `Follow Previous Scale`.
- Initial point as the project record: the project's global BPM, time signature (numerator and denominator) and scale are recorded at the initial point at position 0, displayed on the Tempo Map row as text such as `120 4/4 - C / Am`.
- Grid and ruler: when a Tempo Map exists, ruler ticks, bar/beat labels and the background grid re-align at every point and are computed per segment according to each segment's tempo and time signature. In addition, the time value is always shown at every Tempo Map change point, so you can read off directly where each change occurs.
- Scale integration: scale changes in the Tempo Map affect pitch snapping (scale mode), scale highlighting, the `Project Scale` option of degree transposition / quantization / mean quantization, and the degree-difference rendering of child tracks.
- Import: when importing MIDI as a Pitch Reference Clip, you can enable `Import as Tempo Map` in the import dialog and separately choose whether to import tempo, time signature and scale; when importing a REAPER project (.rpp), the `Import tempo map (only if the current project has none)` option in the import dialog is checked by default and imports the project-level tempo and time-signature changes.

### Clip Rate and Gain Badges

Every clip shows two badges on the right side of its title bar: `Rate` (e.g. `x1.5`) and `Gain` (e.g. `+3.0dB`). Because of them, the two most common adjustments never require opening a dialog.

- Double-click a badge to type a value in place. Press `Enter` to confirm or `Esc` to cancel; clicking anywhere outside the clip also saves the value.
- The volume knob at the top-left of the clip still works as before: drag it up or down, and double-click it to return to 0 dB.
- The badges hide themselves automatically when the clip is too narrow; increase the track height or zoom in horizontally and they reappear.

Right-clicking the rate badge (or choosing `Edit Playback Rate...` in the clip context menu) opens a lightweight popover for cases where you need BPM conversion or batch editing:

- `Stretch Factor`, `Old BPM`, and `New BPM` are linked and always satisfy "factor = new BPM ÷ old BPM". For example, if the material is 120 BPM and you want 140 BPM, just set `New BPM` to 140 and the factor becomes about 1.167 automatically; editing the factor or the old BPM recalculates the others the same way. Scrolling the mouse wheel on a field steps it by a fixed amount (factor 0.1, BPM 1); hold the `Fine Adjust` modifier (default `Ctrl`, `Command` on macOS) for finer steps (factor 0.01, BPM 0.1).
- The `Duration` field lets you type a target length directly, and the factor is derived from it. Input is very permissive — you do not have to match the displayed format. For example `1.2` (seconds), `0:01.5` (hours:minutes:seconds), `1.2.500` (bar.beat.subdivision), and `1.2/16` (with grid) are all accepted, and a comma can be used as the decimal separator.
- `Auto-adjust clip length to the new rate` (on by default): the clip length changes together with the rate — the familiar "change speed and pitch together" behavior. Turning it off keeps the current length and only changes playback rate and pitch.
- When several clips are selected, a note at the bottom of the popover tells you how many clips will be affected, and the change applies to all of them at once.

### Silence Detection

`Silence Detection` automatically finds the silent parts of your clips and removes them in bulk, saving you from splitting and deleting them one by one. Select one or more clips, right-click, and choose `Silence Detection...`.

The dialog previews the result as soon as it opens: detected silent ranges are highlighted in red on the clips, and the footer reports how many ranges were found and their total length. Changing a parameter re-runs the analysis immediately. Nothing is actually modified until you press `Apply`, and the whole operation can be undone.

- `Method`: `RMS (energy)` (default) judges by average loudness and suits most material; `Peak` looks only at the instantaneous maximum level, which helps with gaps that contain occasional clicks.
- `Threshold (dBFS)`: content below this level counts as silence; the default is `-50`. Higher values mark more material as silent.
- `Adaptive threshold (noise floor + 6 dB)`: when enabled, HiFiShifter estimates the noise floor for you instead of requiring manual tuning.
- `Min Silence (ms)`: quiet stretches shorter than this are left alone; the default is `120`. Lower it to cut more finely.
- `Min Sound (ms, 0 = off)`: sounds shorter than this that are surrounded by silence are removed along with it. Off by default.
- `Padding (ms)`: keeps a little extra inside each cut so you do not clip the start or end of a word; the default is `10`.
- `Cut Fade (ms)`: adds a tiny fade at both sides of each cut to avoid clicks; the default is `5`.
- `Action`:
    - `Cut and close gaps` (default): removes the silence and shifts everything after it forward, making the timeline more compact.
    - `Cut only (keep gaps)`: removes the silence but leaves the original time gap, useful if you want to rearrange things manually afterwards.
    - `Split only`: splits at the silence boundaries without deleting anything, for when you want to handle each piece yourself.
- `Delete fully silent clips`: removes clips that are silent from start to finish (on by default).
- `Detect all takes (union)`: merges the silent ranges of all takes of the same clip before processing, so silence does not reappear after switching takes.

Double-clicking a parameter row in the dialog resets that parameter to its default value.

### Close Gaps

Right-click an empty area of a track and choose `Close Gaps` to move every clip after the clicked position on that track forward, removing the empty space between them so they sit end to end. Content before the clicked position stays where it is. If the track has no clips after the clicked position, the menu item is disabled.

### Metronome

The metronome gives you a steady click reference while playing, which helps when checking rhythm, singing along, or verifying tempo changes.

- Toggling: click the metronome button located to the left of the BPM readout in the main toolbar. The default shortcut is `K`.
- Right-click the metronome button to open its settings:
    - `Volume`: metronome level. Besides dragging, it also responds to the mouse wheel; hold the `Fine Adjust` modifier (default `Ctrl`, `Command` on macOS) for finer steps.
    - `Subdivision`: controls click density. `Follow Grid` (default) clicks at the current grid subdivision, so a `1/8` grid gives one click per half beat, and dotted and triplet grids are followed too; `Beat Only` clicks once per beat according to the BPM; `Bar Start Only` clicks only on the first beat of each bar, which is the sparsest option.
    - `Sound`: choose between `Click` (default), `Woodblock`, and `Beep`.
    - `Accent Downbeats`: when enabled, the first beat of each bar is louder so the meter is easy to hear.
- When the project has a Tempo Map, the metronome follows every BPM change point automatically, so the clicks never drift out of sync with the music.

## 4. File Browser

The file browser allows you to open a specific folder, search and sort audio and video media files within it, and drag them into the HiFiShifter track view. Video files use a purple icon; audio files use a blue icon. Search supports regular expressions. Clicking a media file automatically plays a preview (videos preview their audio track). You can hold `Ctrl` and `Shift` for multi-selection. Left-dragging files adds one or more media files across time into the timeline. Right-dragging files brings up a menu with `Add Across Time` / `Add Across Tracks`. `Add Across Tracks` allows you to add multiple media audio clips vertically across multiple tracks.

When the track view has focus, press `Ctrl + F` to open the Quick Search window. This is a simplified version of the file browser, allowing you to quickly search and preview audio/video media files within a folder (videos preview their audio track) and add them to the timeline.

## 5. Parameter Editor

The parameter editor is one of HiFiShifter's core features, allowing you to edit various parameters of the currently selected track.

To enable parameter editing for a track, you must first press the track's `C` (Compose) button and wait for audio analysis to complete. HiFiShifter uses offline rendering; after each parameter edit, you must wait for the parameters to re-render before auditioning.

The `Sync Timeline View` button (link icon) to the left of the `Parameter Editor` label in the parameter editor header toggles horizontal synchronization with the track view. When enabled (default), the two views share the same horizontal position and zoom in both directions: scrolling or zooming in either view updates the other, and enabling sync aligns the parameter editor to the track view as the reference. Because the track view has a track-header area on its left, the parameter editor automatically compensates for the horizontal offset so grid lines and time-axis ticks line up at the same on-screen positions in both panels. When disabled, the two views zoom and scroll independently.

### 1. Algorithms and Parameters

The current version of HiFiShifter supports three vocal tuning algorithms and their parameters:

- **PC-NSF-HiFiGAN** (shown as `nsf-hifigan` in the UI algorithm list): OpenVPI's open-source hifigan vocoder specialized for singing voices, also HiFiShifter's default algorithm.
    - `Pitch`: Adjust the pitch of the voice.
    - `Formant Shift`: Adjust the formant shift of the voice.
    - `Breath Gain`: Allows adjusting the breath volume of the voice, based on the VR-hnsep model.
    - `Tension`: Adjust the tension of the voice.
    - `Volume`: Adjust the volume of the voice.
    - `Dynamics`: Adjust the loudness of the voice.
    - `Pan`: Adjust the pan of the voice.
Note: both `Breath Gain` and `Tension` require the `Harmonic Separation` switch in the toolbar to be enabled first. Enabling it adds rendering cost (the first pass separates every audio clip and can take a while). While it is off, no separation runs at all: these two parameters are greyed out, their curves stay visible but cannot be edited, and they take no part in synthesis. In addition, `Formant Shift`, `Breath Gain` and `Tension` also require the track group's Compose (`C`) switch to be on, otherwise they are greyed out as well.
- **World** (shown as `world` in the UI algorithm list): Open-source high-quality speech analysis and synthesis algorithm.
    - `Pitch`: Adjust the pitch of the voice.
    - `Volume`: Adjust the volume of the voice.
    - `Dynamics`: Adjust the loudness of the voice.
    - `Pan`: Adjust the pan of the voice.
- **VsLib** (shown as `vslib` in the UI algorithm list): Official voice analysis and synthesis library from VocalShifter. VsLib is only available on Windows x86_64.
    - `Pitch`: Adjust the pitch of the voice.
    - `Formant Shift`: Adjust the formant shift of the voice.
    - `Breathiness`: Adjust the breathiness of the voice.
    - `Volume`: Adjust the volume of the voice.
    - `Dynamics`: Adjust the loudness of the voice.
    - `Pan`: Adjust the pan of the voice.
    - `Synth Mode`: Adjust the synthesis mode algorithm of the voice; some of the above parameters may be ineffective with a specific algorithm.
        - `Mono`: VocalShifter's M algorithm, monophonic instrument algorithm.
        - `Mono (Formant)`: VocalShifter's V algorithm, monophonic vocal algorithm.
        - `Chorus`: VocalShifter's P algorithm, harmony algorithm.

A track can only use one algorithm; if you want to use multiple algorithms, separate them into different tracks.

A track group shares a single set of parameters, with child tracks inheriting parameters from the root track. Additionally, child tracks of a track group have three extra parameters — `Cents Offset`, `Degree Offset` and `Formant Offset` — for conveniently adjusting the current child track's pitch and timbre relative to the root track. Among them, `Degree Offset` uses the project's scale setting as its reference. In the parameter-editor toolbar, child `Cents Offset` / `Degree Offset` are grouped inside the `Pitch` button dropdown, and `Formant Offset` is grouped inside the `Formant Shift` button dropdown. `Dynamics` is grouped inside the `Volume` button dropdown.

After copying a `Pitch` segment using the Select tool, you can paste it onto `Cents Offset` or `Degree Offset`, and HiFiShifter will automatically calculate and apply the appropriate offset.

### 2. Select Tool

The Select tool allows you to select a segment of a parameter curve, drag it, or right-click to open a context menu for parameter adjustments.

Common shortcuts:

- `Ctrl + C`: Copy
- `Ctrl + X`: Cut Parameter Frames (copies the selected curve to the clipboard and clears it, so you can paste it elsewhere)
- `Ctrl + V`: Paste
- `Ctrl + Z`: Undo
- `Ctrl + Shift + Z` / `Ctrl + Y`: Redo (macOS: `⌘ + ⇧ + Z` / `⌘ + Y`)
- `Ctrl + A`: Select All

- `BackSpace`: Initialize (resets the selection to its default state; does not touch the clipboard)
- `-` / `=`: Shift the whole parameter curve of the current clip down/up
- `[` / `]`: Shift the parameter curve down/up within the selection

Of these two shifting shortcuts, `-` / `=` move the entire curve across the clip's range, while `[` / `]` move only the currently selected segment. Both support three step sizes, so you can go from coarse to fine in one pass:

| Step                 | Clip Range                | Selection Range           | Pitch Step             |
| -------------------- | ------------------------- | ------------------------- | ---------------------- |
| Default              | `-` / `=`                 | `[` / `]`                 | ±1 semitone            |
| Large (with `Shift`) | `Shift + -` / `Shift + =` | `Shift + [` / `Shift + ]` | ±12 semitones (octave) |
| Fine (with `Ctrl`)   | `Ctrl + -` / `Ctrl + =`   | `Ctrl + [` / `Ctrl + ]`   | ±1 cent                |

Holding a shortcut down performs one step immediately, then repeats continuously after a short pause, which is handy for nudging a curve into place. Parameters other than pitch use equivalent three-level steps (roughly 2.5% of range by default, 12.5% large, 0.25% fine). All of these can be remapped in the `Parameter Editor` group of `Options -> Keyboard Shortcuts...` (`Ctrl` corresponds to `Command` on macOS).

Left-drag on a selected curve to move it vertically, horizontally, or freely, depending on the `Drag Direction` setting. While left-dragging, press the right button (or the `D` key) to quickly toggle drag direction.

When you drag near the left/right edge of the view, the parameter editor scrolls horizontally automatically (the Select tool only advances while the pointer moves — parking the pointer at the edge does not keep it scrolling). See the [Draw Tool](#3-draw-tool) chapter for details.

Right-drag on a selected curve to adjust its amplitude: drag up to increase amplitude, down to decrease, all the way to fully flattened. For the pitch parameter, amplitude adjustment only strengthens or weakens vibrato and other fine detail — the overall note contour and intervals are preserved, so the pitch is never "lifted" as a whole. The result is previewed live while you drag.

Right-click in the parameter editor to open a context menu with operations such as `Initialize`, `Transpose by Cents...`, `Transpose by Degrees...` (shown for pitch parameters only), `Set To...`, `Average...`, `Smooth...`, `Add Vibrato...`, `Quantize...`, `Mean Quantize...`, etc.; when the parameter is Volume or Dynamics, `Convert to Dynamics` / `Convert to Volume` also appear (a loudness-equivalent conversion, not a simple copy); when the parameter is Pitch, `Save as Pitch Reference Clip` and `Export as MIDI` appear as well.

These operations have default shortcuts: `Transpose by Cents` is `Ctrl + F`, `Transpose by Degrees` is `Ctrl + I`, `Set To` is `Ctrl + 0`, `Average` is `Ctrl + E`, `Smooth` is `Ctrl + M`, `Add Vibrato` is `Ctrl + B`, `Quantize` is `Ctrl + P`, `Mean Quantize` is `Ctrl + Q`; all of them can be remapped in `Options -> Keyboard Shortcuts...`.

`Add Vibrato` opens an **apply dialog**: the left column picks a preset (each row shows that preset's waveform thumbnail) and the right side draws the **real curve from the selection** with the preset applied -- a dashed line for the original curve and a solid line for the result, so you can see at a glance which pitch movement the vibrato rides on. The top-right of the preview has a `Placement` selector that decides what the vibrato oscillates around (default `Start → End`); choosing `Keep curve` overlays the vibrato on the pitch movement that is already there, without changing the original motion. The dialog also carries depth and rate knobs that edit a local copy for this application; `Apply` hands that copy to the edit pipeline and does **not** write the library (press `Save` if you want to keep it). With no selection, the whole curve is selected implicitly first. `Create vibrato preset from selection` at the left of the footer acts on the current selection and turns the vibrato already there into a preset.

Hold `Alt` to enter four-point editing mode for the selected curve. Similar to the feature in VocalShifter, dragging the four points allows you to bend the curve.

Hold `Alt` and drag the edge of the selection area to stretch the parameter curve within the selection.

Hold the Multi-Range Select modifier (default `Ctrl`, `Command` on macOS; configurable under `Options -> Keyboard Shortcuts`, `Param Editor` modifier group) with the Select tool to create **multiple selection ranges**: hold and drag to **add** a range (overlapping or touching ranges merge into one), or hold and click an existing range to **remove** it. The ranges are independent in time — copy, cut, paste, transpose, set-to, average, smooth, quantize, mean-quantize, amplitude adjustment, morph, dragging and edge-stretch all run **per range**, and unselected gaps are never filled, compacted or merged. For example, after copying `0-1s` and `2-3s`, selecting `0-3s` and pasting writes only `0-1s` and `2-3s`, leaving `1-2s` untouched; conversely, after copying `0-3s`, selecting `0-1s` and `2-3s` and pasting puts the first second of the clipboard into the first range and its last second into the second range, while `1-2s` is neither written nor previewed.

### 3. Draw Tool

The Draw tool lets you draw parameter curves by hand. Default shortcut: `F8`.

Left-drag to draw freely or horizontally, depending on the `Drag Direction` setting. While left-dragging, press the right button (or the `D` key) to quickly toggle drag direction.

When you drag near the left/right edge of the view, the view scrolls horizontally automatically, faster the closer you are to the edge, so you can draw in one stroke into places you cannot see. The Draw, Line and Vibrato tools scroll continuously while the drag is held; the Select tool only advances while the pointer moves — parking the pointer at the edge does not keep it scrolling. The vertical direction never auto-scrolls. This behavior needs no setting and cannot be turned off.

Right-drag resets the current curve.

**Right-click** the draw-type tool button in the parameter editor toolbar to switch between the three drawing tools: Draw, Line and Vibrato (the menu also lists each one's shortcut). The button itself shows the tool you are currently on, and left-clicking it activates that tool directly.

The parameter editor has four tool gestures in all: Select (`F7`), Draw (`F8`), Line (`F9`) and Vibrato (`F10`). `Tab` toggles between Select and the draw-type tool you used last. **The tool you are on is remembered** and restored the next time you start the program.

### 4. Line Tool

The Line tool draws a straight line from the start point to the end point. Default shortcut: `F9`.

It *is* the Vibrato tool plus the Straight preset — so every gesture described in the next section (the wheel for depth and rate, `,` / `.` and the mouse side buttons for presets, the two-key resets, the readout bubble) works on the Line tool too. The only difference is which preset a stroke starts from: the Line tool always starts from `Straight`, the Vibrato tool from the preset you selected.

Left-drag to draw a straight line freely or horizontally, depending on the `Drag Direction` setting. While left-dragging, press the right button (or the `D` key) to quickly toggle drag direction.

If you turn the depth up mid-drag the result is no longer a straight line — deliberately so: the Line tool starts from zero vibrato and lets the stroke itself decide how much vibrato it needs. The other way round, pressing both preset-switch keys while using the Vibrato tool (or cycling onto the `Straight` preset) switches you to the Line tool.

Right-drag resets the current curve.

### 5. Vibrato Tool

The Vibrato tool superimposes a vibrato between the start and end points. Default shortcut: `F10`.

Left-drag to draw vibrato freely or horizontally, depending on the `Drag Direction` setting. While left-dragging, press the right button (or the `D` key) to quickly toggle drag direction.

While left-dragging, scroll the wheel to adjust the vibrato **depth**; hold `Alt` while scrolling to adjust the **rate**. Hold the `Param Fine Adjust` modifier (default `Ctrl`) to fine-tune while scrolling. During the drag a single readout bubble follows the pointer: **the parameter value on the first line** (what this stroke is landing on) and **the vibrato state underneath it** (current preset, depth and rate). They share one bubble rather than popping two, which used to leave the second covering the first. The whole bubble is governed by `Options → Show Parameter Popup`. Depth can go **negative**: a negative depth inverts the waveform (it starts by swinging down), and turning the wheel further down does not stick at 0.

**The depth's unit, how much one notch moves, and how far it can go all depend on the parameter type**, because the same depth value means very different things on different parameters:

- **Pitch-like** (pitch, child-track cents / scale-degree offsets, formant shift): depth is in **cents**, 24 cents per notch. The ceiling is the tool's own depth limit (±1200 cents); if the parameter's own range is narrower, that range wins — pushing past it only gets clamped by the write path, flattening the top of the waveform into a straight line rather than making it "more vibrato".
- **Gain-like** (dynamics, volume, breath gain): depth is a **percentage**, 2% per notch. 100% swings the gain across its full amplitude.
- **Other ranged parameters** (pan, tension, breathiness): depth is also a **percentage** — it says "what fraction of this parameter's full swing", 2% per notch, with 100% filling the whole range. So `30%` on pan and `30%` on breathiness are the same order of magnitude, with no need to convert back to each parameter's raw numbers.

One notch is always **1/50 of that parameter's full swing**, so "zero to full" takes roughly 50 notches on every parameter. If a parameter has no descriptor, a built-in fallback range is used, so it never degrades into "the wheel does nothing" or "one notch jumps all the way".

You can switch vibrato presets mid-drag:

- **Mouse side buttons**: forward switches to the next preset, back to the previous one.
- **Keyboard**: `,` for the previous preset and `.` for the next (rebindable in `Options → Keyboard Shortcuts`).

A preset decides the waveform (Sine / Triangle / Saw up / Saw down / Square / Trapezoid / Trill steps / Hand-drawn), the envelope (fade in, swell, fade out) and the rate ramp; depth and rate belong to the gesture itself. Depth and rate tweaks made during a drag never rewrite the preset and are not carried into the next drag — presets change only in the preset manager. **Switching presets, however, is persistent**: pressing a side button or `,` / `.` mid-drag really does change the preset you are using (it is remembered across restarts).

When the cycle lands on the `Straight` preset it **switches to the Line tool** instead (the Line tool is the Vibrato tool plus that preset); cycling out of the Line tool switches back to the Vibrato tool and moves on to the neighbouring preset. Switching tools never touches the vibrato preset you have remembered — come back after drawing a line and your tone is still the one you picked.

Depth and rate are inherited when you switch, **judged separately**: an adjusted depth is kept, an adjusted rate is kept, and whichever one you did not touch takes the new preset's own value — you changed the tone, so the value you never touched should follow the new preset.

A few "both keys" gestures during a drag return things to a known state in one step:

- **Press both preset-switch keys** (`,` and `.` by default), or **press both mouse side buttons**: switches to the Line tool (so you draw a straight line) and clears this gesture's recorded depth and rate adjustments.
- **Press both amplitude keys** (`↑` and `↓` by default), or **press the middle mouse button while the wheel is in amplitude mode** (the frequency modifier `Alt` is not held): puts the depth back to the preset's own value and clears the recorded depth adjustment.
- **Press both frequency keys** (`←` and `→` by default), or **press the middle mouse button while the wheel is in frequency mode** (`Alt` held): puts the rate back to the preset's own value and clears the recorded rate adjustment.

Clearing the record means that value no longer follows your adjustment when you switch presets — it takes the new preset's own value instead.

The drag rate is always measured in Hz, so a preset sounds the same speed however long you drag.

Right-drag resets the current curve.

### 6. Vibrato Presets

A vibrato preset is a reusable set of vibrato parameters. Open the manager from `Options → Vibrato Preset Manager...`, from **right-click → Vibrato preset** on the draw-type tool button in the parameter editor toolbar (a second-level menu listing every available preset, with `Manage presets...` at the bottom), or from `Create vibrato preset from selection` in the footer of the `Add Vibrato` apply dialog. **The default preset is `Natural`** — switching to the Vibrato tool should give you an audible vibrato, and the "no vibrato at all" line is now the Line tool's job. `Straight` still sits first among the built-in presets, but it is **the Line tool's position in the preset sequence**: selecting it in the list switches to the Line tool.

The preset entry shows the preset the **Vibrato tool** is using — it keeps showing that value even while you are on the Line tool, because that is the tone you get back when you switch to the Vibrato tool. The ticked row in the list is the same one.

The **waveform preview** sits across the top of the manager and is deliberately outside every scroll region, so it stays visible while you edit the parameters below it and redraws as you change anything. Its vertical axis is a **fixed** ruler: opening the dialog or selecting a preset fits it once to that preset's amplitude (the wave usually fills about 60% of the box) and it does not move again while you edit. That way the wave's height *is* the depth — read it directly against the `+N / −N` tick labels — and turning the depth up visibly grows the wave instead of rescaling the whole picture (which only made it "shiver" and told you nothing). If you drag past the top of the ruler, press `Fit` on the preview card to re-fit it. The card's play button **auditions** the preset with a synthesized tone -- the same voice as the piano-roll keys, no vocoder involved: one click sounds immediately, a second click stops, and what you hear is exactly the drawn curve including the fade-in, the swell and the irregularity. Built-in presets can be auditioned too.

The preview is editable, not just readable: drag the **small square handles at the left and right edges** to change the fade-in and fade-out durations (the cursor becomes a horizontal resize), and drag on the waveform body to change the start phase (horizontally) and the depth (vertically). Either handle can be dragged all the way to the other end of the vibrato line — a full-length fade-in makes the whole line swell in; when both are pulled to the full length their gains multiply, so the middle dips smoothly. Depth is **1:1 with the canvas** — the ruler is fixed (see above), so "drag the peak to the centre line" is exactly "drag the depth to 0", and a shallow preset moves in small steps per pixel instead of flying off. Hold the `Param Fine Adjust` modifier (default `Ctrl`, `Command` on macOS) to shrink the movement to a fifth for pixel-by-pixel work; the modifier can be pressed or released mid-drag without interrupting the drag (the value never snaps back), it only makes the following movement finer or coarser. Every gesture edits the current draft -- nothing is written back until you press `Save`, and a built-in preset's canvas is read-only just like its form.

`Save` **does not close the window** — you can save a version and keep editing, or move on to another preset. Switching to another preset mid-edit writes the current changes back to the library automatically (the same path as `Save`), so changing presets never loses your work. Only two things ever add a custom preset to the library: pressing `Duplicate as mine`, or editing a built-in preset (see above). Use the `Close` button in the footer (or `Esc`) to exit.

`Export` in the footer saves the selected preset as a JSON file, and `Import` reads one back: duplicates are skipped automatically (importing the same file twice does not double the library), the imported preset is selected afterwards with a line of feedback, and a wrong file -- a theme's JSON, say -- is rejected with a reason. Imported presets always become new custom presets, even when the file carries a built-in preset's name.

The parameters fall into four groups:

- **Waveform**: shape (Sine / Triangle / Saw up / Saw down / Square / Trapezoid / Trill steps) and skew. Skew means different things per shape — the rising fraction for a triangle, the duty cycle for the square family. It is the single most important parameter for making a vibrato sound human, because a real voice does not rise and fall symmetrically. The `Draw...` button next to the shape opens a **single-cycle editor**: drag across it to draw that one cycle cell by cell (it starts from the current shape, which is easier than drawing from nothing), and `Smooth` softens it once per click and **keeps softening while held** (the same feel as keyboard auto-repeat). There are two resets: `Reset to previous` returns to the waveform you entered with (possibly an extracted or drawn table), and `Reset to {shape}` re-samples the table from the shape currently chosen in the dropdown — for starting over after a messy drawing, or turning an extracted waveform into a clean sine. Changing the shape while drawing no longer closes the editor; it applies straight away. The table wraps end to end, so the seam never shows a crease.
- **Depth**: the amount, shown according to the current parameter's type (cents for pitch and cents-like parameters; a **percentage of full swing** for gain parameters such as dynamics and for every other ranged parameter); swell, as a depth multiplier at the start and at the end; centre offset; and irregularity, which jitters phase and depth with deterministic noise so the result wavers rather than sounding mechanical. The dice button beside it rolls a **new wobble pattern** (the seed is a preset field, so the same preset renders, auditions and commits identically every time). At an irregularity of 0 there is no pattern to roll — the noise is multiplied away entirely — so the dice button is disabled and says why.
- **Rate**: measured by frequency in Hz or by a cycle count over the whole region; the rate at the end, for speeding up or slowing down toward the end of a phrase; and landing on a whole cycle so the note returns cleanly to its baseline.
- **Envelope**: fade in and fade out, each with a duration and a curve shape (the duration is capped by the vibrato line itself — pull it to the full length for an all-the-way fade); and the start phase.

Placement is not one of the preset parameters: it is an **apply-time** setting, chosen at the top right of the `Add Vibrato` dialog preview (see above), and decides what the vibrato oscillates around — `Start → End`, `Hold start`, `Hold end`, `Average`, or **Keep curve** (superimposed on the existing curve, preserving the original pitch movement).

A built-in preset itself is read-only (the factory definition can always be restored), but its parameters are yours to turn: **the moment you change anything, the app saves "the factory definition plus this change" as a custom preset** and moves the editing focus onto it - a new row appears in the list right away, named with a number such as `Natural 2`, and the list scrolls down to it. That way "editing a built-in preset creates a custom one" is something you watch happen, rather than something that appears behind your back when you switch away. The built-in preset itself is left untouched. There are two visible ways to **make a preset the current one**: click the dot at the left of its list row (filled = currently in use, hollow = available; hovering spells it out), or press the `Use as current` button at the top of the editor — that one acts on the preset you are editing and turns into `Currently in use`, disabled, once it already is. Double-clicking a list row is a shortcut for the same thing.

Clicking a list entry only selects it for editing and does **not** make it current, so shaping a preset you have not settled on does not change the sound you are about to draw with.

Each row has an **enable / disable** button on its right (the eye icon). A disabled preset disappears from the parameter editor's toolbar list and is **skipped** when you cycle presets with the keyboard or the mouse side buttons mid-drag. Built-in and custom presets can both be disabled — if you only ever use two or three of them, disable the rest and both the toolbar list and the cycling stay short. Disabling only affects *how you pick* a preset: the preset is still editable and can still be the current one (disabling the one you are using is allowed).

**Right-clicking** a row opens that preset's context menu: `Enable / Disable`, `Use as current`, `Rename`, `Move up / Move down`, `Duplicate as mine`, and `Delete` (Delete and Rename are unavailable for built-in presets — duplicate one instead). `Rename` opens an input in place, over the name; Enter or clicking elsewhere commits, `Esc` cancels. Hovering a row shows the project's own tooltip with that preset's description and summary — the summary's first item is the waveform shape (a sampled table reports the closest shape, or `Hand-drawn` when it matches none).

After `New`, `Duplicate as mine`, `Create from selection` or `Import`, the list scrolls to the entry that just appeared — new presets are appended at the end, which falls outside the viewport once the list is long.

The preset list can be **reordered by dragging**: press a row and drag it up or down; it swaps once it passes the neighbouring row's midpoint, and the new order is saved on release. The row being dragged dims and a highlighted line marks the drop position; dragging to the top or bottom edge **auto-scrolls** the list (faster the closer you are to the edge), so a long list can be traversed in one go. Built-in and custom presets form two groups that are ordered independently (they never interleave). Keyboard users get the same adjustment through `Move up / Move down` in the row menu.

**Creating a vibrato preset from a selection**: once you have drawn a vibrato you like, select it and choose `Create vibrato preset from selection` (or the same button in the apply dialog footer). The pitch movement itself is removed, the period is measured by autocorrelation, the remainder is folded into one averaged cycle to become the waveform, and the depth, fades and irregularity are measured, giving you a preset you can keep tuning. The folded waveform is low-passed (only the lowest few harmonics are kept), so the result is a **smooth** curve rather than one that carries the recording's observation noise and looks bumpy. It works on any parameter.

### 7. Pitch Snap

When editing pitch parameters with any tool, Pitch Snap allows you to snap edits to semitones or scale degrees. Hold `Shift` to temporarily toggle snap.

Pitch Snap is off by default. Right-click the Pitch Snap button to open the Pitch Snap Settings menu, where you can adjust the quantization unit and tolerance.

- `Quantize Unit`: Two types: `Semitone` (default) and `Scale`. When set to Scale, the reference scale is the project's current scale.
- `Tolerance`: Adjusts the snap tolerance range (default 0 cents). Edits within the tolerance are not snapped; edits outside the tolerance are snapped to the nearest tolerance edge.

For example, to create vocal harmonies:

1. Confirm and set the project scale.
2. Enable Pitch Snap and set Quantize Unit to `Scale`.
3. Enable Scale Highlight to easily observe the transposition degree.
4. Use the Select tool to drag vertically.

Alternatively, use the `Cents Offset` and `Degree Offset` parameters on child tracks:

1. Confirm and set the project scale.
2. Drag the harmony track's header onto the lead vocal track to form a track group (lead = root, harmony = child).
3. Switch the parameter editor to the `Degree Offset` parameter of the harmony track and draw the desired degree line. Both `Cents Offset` and `Degree Offset` support Pitch Snap, snapping to integer semitones and integer degrees respectively.

This quickly creates harmonies by degree transposition. Similarly, switch to a child track's `Formant Offset` parameter to draw a per-frame formant-shift curve and create timbrally varied harmonies on algorithms that support `Formant Shift` (NSF-HiFiGAN / vslib).

### 8. Pitch Reference Clip

A Pitch Reference Clip on a track is a special type of audio clip that stores a pitch curve on the timeline.

Pitch Reference Clips can be created through the following methods:

- Import MIDI via the `File` menu, by dragging a MIDI file onto the track view, or by dropping a MIDI file directly onto the Parameter Editor while editing the pitch parameter (the drop hint reads `Drop to import MIDI into the pitch parameter`). This opens the MIDI Import dialog.
    - MIDI File: Allows you to select a MIDI file to import. Also supports parsing MIDI data exported to the system clipboard by other DAWs. DAWs confirmed to support system clipboard MIDI data transfer include Reaper and FL Studio.
        - Reaper: In Reaper's MIDI Editor, select notes and copy them to export the selected note data to the system clipboard for HiFiShifter to read. Note that since Reaper's clipboard note data does not include BPM information, when importing, you can use the current project BPM or specify one manually.
        - FL Studio: In FL Studio's Piano Roll, click the small triangle in the top-left corner and select `File` -> `Copy to MIDI Clipboard` to export all notes of the current channel to the system clipboard for HiFiShifter to read.
    - Track Selection: Allows you to select which MIDI tracks to import.
    - Import Target: Choose `Pitch Reference Block` or `Pitch Parameter` (written directly onto the current track's pitch curve).
    - Import Position: Choose `Project Start`, `Playhead`, or `Selection`.
    - Import MIDI BPM as Project BPM: When enabled, imports the MIDI's initial BPM as the project BPM.
    - Note BPM: Configures the BPM mapping for imported notes.
        - MIDI own BPM: Import directly with the MIDI's own BPM without BPM mapping.
        - Current Project BPM: Map note BPM to the current project BPM before importing.
        - Specified BPM: Map note BPM to a manually specified BPM before importing.
    - Multi-track Merge: When enabled, automatically merges all selected tracks and notes, using the highest pitch note as the pitch curve parameter, ultimately importing only 1 Pitch Reference Clip. When disabled, attempts to split tracks and notes so that all notes can be imported as pitch curve parameters, which may result in multiple Pitch Reference Clips stacked vertically.
    - Fill Gaps Between Notes: When enabled, automatically fills the gaps between adjacent notes. There is also a `Close leading gap` option that removes the gap at the beginning of the imported content, before the first note.
    - Import as Tempo Map: When enabled, you can separately choose whether to import the MIDI's tempo, time signature and scale as project Tempo Map points.
- Import MIDI items from tracks via a Reaper project or Reaper clipboard.
- Import MIDI audio clips from tracks via a VocalShifter project or VocalShifter clipboard.
- Right-click a regular audio clip and select `Convert to Pitch Reference Clip` from the context menu. This converts the original pitch curve of that audio clip into the pitch curve of a new Pitch Reference Clip.
- In the Parameter Editor, while editing pitch parameters, use the Select tool to select a region, right-click, and choose `Save as Pitch Reference Clip` from the context menu to save the pitch curve within the selection as a new Pitch Reference Clip.

Pitch Reference Clips have the following common uses:

- Placed on a track, they serve as a general audio clip for other tracks with regular audio clips to reference pitch. On other tracks, the `Reference Track Group` feature in the Parameter Editor can be used to view this track and display its pitch curve.
- When a Pitch Reference Clip is placed on the root track of a track group, it can change the pitch processing logic of that track group, overwriting the original pitch curve of the covered segment with the Pitch Reference Clip's pitch curve. This affects the following scenarios:
    - If a pitch curve segment within the track group has never been edited, its pitch parameters will be directly overwritten by the Pitch Reference Clip's pitch curve, triggering re-rendering of the audio pitch.
    - When using `Initialize`-related functions - for example, right-dragging with the Draw tool in the Parameter Editor - the initialized pitch curve uses the Pitch Reference Clip's pitch curve data rather than the original pitch of the audio clips within the track group's child tracks.
    - If the Pitch Reference Clip is muted, the track group will not reference that Pitch Reference Clip when processing pitch.

Select a Pitch Reference Clip and choose `Update Pitch` from the context menu to update the Pitch Reference Clip with the existing pitch parameters within its range.

### 9. Other Features

Additional convenient features of the parameter editor:

- `Scale Highlight`: Located in the toolbar. Highlights the piano-roll key rows that belong to the project's base scale, making it easy to see the relationship between pitch and mode.
- `Drag Direction`: Located in the toolbar. Controls the movement direction when left-dragging a parameter curve (`Free` / `X Only` / `Y Only`); equivalent to pressing `D` during a drag.
- `Clipboard Preview`: Located in the `View` menu. After copying parameter curve data with the Select tool, the clipboard curve data is previewed live in the selection area for easier paste positioning.
- `Popup Param Values`: Located in the `View` menu. Shows the parameter value when the mouse is near the curve or while drawing/editing.
- `Lock Param Lines`: Located in the toolbar. Controls whether dragging an audio clip on the track also moves its corresponding parameter curves. All parameter editing in HiFiShifter is track-based; if not locked, edited curves will not follow the clip.
- `Smoothness`: Located in the toolbar. Whether to automatically smooth parameter edits and the smoothing strength.
- `Reference Track Group`: Located in the toolbar. When the parameter is `Pitch`, lets you choose other tracks and display pitch curves from other track groups as references in the pitch editor.
- `Import MIDI`: Located in the toolbar. When the parameter is `Pitch`, lets you select a MIDI file and import notes from one or more tracks as a pitch curve; you can also drop a MIDI file directly onto the parameter editor.
- `Secondary parameter overlay`: Every parameter button has an eye icon on its right; when enabled, a related parameter is overlaid as a secondary curve while you edit the current one (e.g. overlaying `Cents Offset` while editing `Pitch`), which is handy for comparison.
- `Volume / Dynamics vertical axis unit toggle`: When the parameter is `Volume` or `Dynamics`, clicking the vertical axis area toggles the display unit between dB and ratio.

### 10. Smoothing

After selecting a region with the Select tool, right-click and choose `Smooth...` (default shortcut `Ctrl + M`) to calm down a jittery parameter curve.

- `Smoothness`: a continuous slider from 0 to 100%; higher means smoother. The dialog's initial value follows the toolbar `Smoothness` setting (which defaults to 0%, i.e. no automatic smoothing).
- Smoothing takes the curve on both sides of the selection into account, so no abrupt step appears at the selection edges. The edge transition width is fixed in time, so it does not widen as the selection grows, and repeated editing does not accumulate distortion outside the selection.
- The `Quantize` and `Mean Quantize` dialogs offer the same `Smoothness` slider, letting you smooth while you quantize pitch.

For a quick flatten, you can also hold the right button inside a selection and drag downward: the lower you drag, the smoother it gets, with live preview.

## 6. Export Audio

After completing all edits, use the `Export Audio` function in the `File` menu to export the HiFiShifter project as an audio file.

Parameters:

- `Output Format`: `WAV` / `MP3` / `FLAC`.
    - `WAV`: uncompressed and lossless, with the widest compatibility and the largest file size. Choose it when you plan to keep editing in other software.
    - `FLAC`: losslessly compressed — identical quality to WAV but a noticeably smaller file. Choose it for long-term lossless archiving.
    - `MP3`: lossy and the smallest, supported by virtually every device and player. Choose it for sharing, uploading, or making rough mixes.
    - Switching format automatically updates the extension of the output filename and of the per-track naming template.
- `Export Type`: `Project` / `Separated Tracks`.
- `Time Range`: `All` / `Custom`. Custom allows setting start and end seconds.
- `Sample Rate`: Set the output sample rate. All three formats support it, but MP3 only accepts fixed steps between 8 kHz and 48 kHz (for example 44100 or 48000 Hz). If you pick a rate MP3 does not support, HiFiShifter silently switches to the nearest supported one and tells you.
- `Bit Depth`: Set the output bit depth. WAV supports 16 / 24 / 32-bit float, and FLAC supports 16 / 24-bit. MP3 has no bit-depth option — it always encodes at an effective 16-bit precision internally, so there is no need to worry about bit depth for MP3.
- `Encoder Settings`: only shown for MP3 and FLAC; used to balance quality against file size.
    - MP3:
        - `Encoding Mode`: `VBR (quality first)` allocates bitrate according to content complexity, giving smaller files at a similar listening experience — a good default; `CBR (constant bitrate)` keeps the bitrate fixed, making file size easier to predict.
        - `Bitrate`: available with `CBR`, from 8 to 320 kbps. Higher means better quality and a larger file.
        - `VBR Quality`: available with `VBR`, from q0 (about 245 kbps) to q9 (about 65 kbps). The default q2 is already very good.
        - `Metadata (ID3 Tags)`: `Title`, `Artist`, `Album`, and `Comment` are written into the MP3 file; leave them blank to skip.
    - FLAC:
        - `Compression Level`: 0 to 8. `0` is the fastest with slightly larger files; `8` gives the smallest files but is the slowest; the default is `5`. Compression level affects file size and encoding time only, never the audio quality, so the default is usually fine.
    - Common:
        - `Dither`: adds a tiny amount of noise when reducing bit depth to improve perceived quality; choose `Off` or `TPDF`. It only applies to integer bit depths (16 / 24-bit); most users can leave it `Off`.
        - `Channels`: `Stereo` keeps the original left/right channels; `Mono Downmix` mixes both channels into mono.
- `Output Folder`: Set the output folder. Supported placeholders:
    - `<ProjectFolder>`: The folder containing the current project. If the project has not been saved, defaults to the `Documents` folder.
    - `<ProjectName>`: The current project's filename without extension.
    - For Project export, the default Output Folder is `<ProjectFolder>`; for Separated export, the default is `<ProjectFolder>/<ProjectName>`.
- `Output File Name`: Set the output filename. Supported placeholders:
    - `<ProjectName>`: The current project's filename without extension.
    - Default is `<ProjectName>.wav`.
- `Separated Track Name Pattern`: Set the naming pattern for separated tracks. Supported placeholders:
    - `<ProjectName>`: The current project's filename without extension.
    - `<ExportIndex>`: Sequential index of the track during export, starting from `0`.
    - `<TrackIndex>`: Internal track index in the project, starting from `0`.
    - `<TrackName>`: The track's name in the project.
    - `<TrackType>`: Track type: `Root` or `Sub`.
    - `<TrackId>`: Internal ID of the track (not recommended for general users).
    - Default pattern is `<ExportIndex>_<TrackName>.wav`.
- `Separated Track Targets`: Select which tracks to export. By default, only non-muted normal tracks and root tracks are selected. Quick buttons above the panel: `Select All`, `Select None`, `Select All Sub Tracks` and `Exclude Muted`.
    - If you check a track that is originally muted, it will be exported regardless of mute state.
    - If you check a root track of a track group, the entire group is exported as a single audio file, and the exported audio excludes data from muted child tracks.
    - If you check a child track, it will be exported regardless of its own or its root track's mute state.
    - The `Exclude Muted` button only affects the targets you have currently checked: it unchecks the muted ones among them and leaves everything you did not check untouched. So you can hand-pick first, then use it to strip out the muted items in one go.

While typing a file path, you can click the `Placeholder` buttons to quickly insert the corresponding text.

All file path strings support time format strings like `%Y-%m-%d-%H-%M-%S`. If you want to include a literal `%` in the output path, use `%%` to escape it.

Select clips and choose `Quick Export` from the context menu to export just those clips from a small window. It also offers `WAV` / `MP3` / `FLAC`, but does not ask for encoder settings — it reuses the settings you saved in the `Export Audio` dialog.

## 7. Recording

HiFiShifter can record directly onto the timeline. Recording also starts timeline playback from the current playhead at the same time (for following the accompaniment or background music); when recording stops, playback stops automatically and the recording is imported into the timeline.

### Record button and shortcut

- The red circular button in the transport area starts and stops recording.
- The default shortcut is `Ctrl + R`, changeable in `Options -> Keyboard Shortcuts...`.
- Right-click the record button to open recording settings quickly.

### Recording settings

Open the recording settings via `File -> Recording...`:

- `Source`: choose one of three capture sources.
- `Input Device`: defaults to `System Default`, or choose any input device (microphone) on this computer.
- `Loopback Device`: when the source is `System Sound (Loopback)`, choose the output device to capture; default is `System Default Output`.
- `Application`: when the source is `Application Audio`, capture only the sound of a specific program; click `Refresh` to re-enumerate the programs currently outputting audio. A restarted program is re-matched automatically by its process name. Windows 10 21H2 (build 20348) and newer prefer the system process-loopback API; older builds automatically fall back to the "mute other sessions" compatibility scheme. On Linux, capture uses PipeWire (`pw-dump` / `pw-cat`). macOS does not support this mode yet.
- `Sample Rate` / `Bit Depth` / `Channels`: sample rate, bit depth and channel count of the output WAV.
- `Input Gain`: pre-recording gain compensation.
- `Countdown`: enter the number of seconds; after clicking record, the countdown runs first, then recording starts.
- `Monitor input while recording`: routes the input signal back to the output device while recording; `Monitor Gain` adjusts the level of the routed signal.
- `Auto-normalize after import`: after stopping and importing, automatically normalize the new clip's peak to 0 dB.
- `Auto-stop at end of selected clips`: automatically stop and import when playback reaches the end of the selected clips.
- `Output Path Template`: supports `<ProjectFolder>`, `<ProjectName>` and time format strings. Default: `<ProjectFolder>/HiFiShifter Record/%Y-%m-%d-%H-%M-%S.wav`.

### Recording workflow and import rules

1. Move the playhead to the desired start position and select the target track.
2. Click the record button. Timeline playback starts from the playhead while capture begins.
3. Click the record button again to stop recording; timeline playback stops with it.
4. If the selected track has no clips within the recording range, the recording is imported directly onto that track. Otherwise a new `Recording` track is created immediately below the selected track, the recording is imported there, and the new track and the new clip are selected automatically.

## 8. VST3 / ARA Plugin (REAPER)

Besides the standalone app, HiFiShifter also ships in a VST3 (ARA) plugin form: load it into REAPER and the same HiFiShifter editing interface opens directly inside the REAPER project. The plugin shares the same editing and rendering logic as the standalone app, but **does not require the standalone app to be started first**, and edits are saved together with the REAPER project.

The plugin is currently a Windows x64 build and has only been verified against REAPER; there is no plugin delivery for Linux / macOS yet.

### Installation and Connection

There are two ways to install the plugin; pick either one:

- **Installer (recommended)**: run `HiFiShifter_v<version>_windows-x86_64-vst3-setup.exe`. The wizard follows the system UI language, lets you choose the VST3 directory, installs the models into the machine-wide shared library (see [Model Files and Sharing](#model-files-and-sharing)), and registers an uninstall entry under Windows "Programs and Features / Installed apps". The installer first confirms that REAPER and its plugin host process have exited.
- **Manual copy**: close REAPER and put the **entire** `HiFiShifter.vst3` folder into a VST3 directory (for example `C:\Program Files\Common Files\VST3`), or add its parent directory to REAPER's VST scan path. Do not copy only the individual files inside the folder — the interface, models and runtime dependencies all live in that folder. This approach writes nothing to the registry; to uninstall, just delete the folder.

Both ways require the WebView2 runtime and the VC++ x64 runtime to be installed.

Access steps (same for both ways):

1. Restart REAPER and let it scan plugins, then add the VST3 plugin HiFiShifter on a track that has a vocal audio item and connect it via the host's ARA mechanism (rather than importing the audio into the standalone app). Trying it on a copy of the project first is recommended.
2. To work on several tracks, add a plugin instance on each track. Opening any one HiFiShifter window lets you edit every track already connected in the same project — these windows are different views of one workspace, not independent editing copies; edits are not shared between different projects.
3. Before using shortcuts, click the plugin window once so keyboard focus enters the interface (the plugin forwards `Ctrl + C` / `Ctrl + X` / `Ctrl + V` / `Ctrl + Z` / `Ctrl + Y`, and the remaining shortcuts are handled by the interface itself).

### Why a Track Group (Folder) Shows No Audio

REAPER manages ARA plug-ins **per track**: it only hands the plug-in the audio on **the one track the plug-in sits on**. So when HiFiShifter is added to the parent track of a track group (folder), the items on the tracks inside the group are visible and can be dragged, but they get **no** audio content — no waveform, and no parameter analysis.

- The correct approach is to **add one HiFiShifter instance per audio track you want to process**, not just one on the group's parent track.
- You do not need to select the items first, and you do not need to "enable ARA" by hand: REAPER has ARA support on by default and starts analysis automatically. The only requirement is that the track actually has an audio item on it.
- The plug-in recognizes this case: it shows a notice above the timeline, and the group's items are drawn as "waiting for REAPER to hand over audio" instead of as a blank clip that looks usable.

> This follows from REAPER managing ARA plug-ins per track; it is not a defect in the plug-in.

### Model Files and Sharing

The plugin needs three models — FCPE, HNSEP and HiFiGAN (about 150 MB in total). The installer puts the models into the **machine-wide shared library** (`%ProgramData%\HiFiShifter\models\<model version>`), which the standalone app reads as well — using both the plugin and the standalone app does not duplicate the 150 MB. The manual-copy version carries its own copy of the models and does not depend on the shared library.

- The model version is derived from the model content: an upgrade lands in a new directory, and reinstalling the same version is skipped.
- Uninstalling the plugin does **not** delete the shared model library (the standalone app uses it too). If you really want it gone, delete `%ProgramData%\HiFiShifter\models` by hand.
- The environment variable `HIFISHIFTER_MODELS_DIR` overrides the model directory and takes precedence over the shared library.

### Division of Labor in Plugin Mode

In plugin mode, **the host handles** audio import, item position / trimming / linear stretching, BPM and fade audio; **HiFiShifter handles** editing of parameter curves such as pitch, breath, tension and formant.

- The `File` menu keeps only `Import Media File...` and a disabled `Save · REAPER`; saving the project, exporting and recording all happen in REAPER.
- `File → Import MIDI...` works in the plugin, but its target is the **pitch curve** (`Import into pitch parameter` in the dialog): the notes are written straight into HiFiShifter's own curve through the same implementation the standalone app uses. `Create pitch reference block` and the item context menu's `Replace MIDI` do not appear in the plugin at all — clips belong to REAPER, and the plugin cannot create a local clip the host would not see.
- Entries taken over by the host — the `Track` menu, the `Record` button, `Options → Inference Device`, etc. — simply do not appear in the plugin rather than sitting there greyed out: the VST window a host gives you is usually narrow, and controls you cannot use should give up their space.
- When the host exposes transport control, the `Stop` and `Play / Pause` buttons control REAPER's playback; the playhead follows the REAPER project's position, and the BPM is decided by REAPER as well. When the host offers no transport control the whole group disappears.
- The toolbar's `BPM` / time-signature fields, the metronome button and the record button do not appear in the plugin: they are host-authoritative readings or host commands, and REAPER's own interface already shows and operates them.
- Fade audio on items is handled by REAPER; the envelope drawn in the interface is HiFiShifter's own schematic. The hover text gives the same three values as the standalone app (shape / length / curvature) and no longer prints the host's `c=` / `S=` readings, which told the user nothing actionable. The curve is not required to match REAPER's point by point.
- The tempo map **is available in the plug-in, but only its scale axis is editable**: the tempo map *is* the storage for time-varying scale, and scale is HiFiShifter's own (REAPER has no project key signature). A change point's `BPM` and time signature are host-authoritative readings and are shown read-only in the dialog; the plug-in also does not auto-match the host tempo map (REAPER hands no tempo / time-signature content to ARA plug-ins — measured on this machine, see F-2 in `probe/ara/README.md`). Scale change points live in the plug-in's own settings.
- Loop source (REAPER's "loop source" flag) is read and written faithfully: the plug-in renders with REAPER's loop semantics, and `Loop` / `Unloop` in the clip context menu are usable.
- Reverse takes are **not processed** by the plug-in: ARA delivers no reversed audio, so the plug-in cannot render correct content. Such clips are still shown (with a reverse marker) and the reverse entry is disabled with a reason — do it in REAPER.
- When a clip has no audio the hint now says **why**: `Waiting for REAPER to finish assigning audio` (the normal in-flight state right after split / trim / paste), `Waiting for REAPER to hand over audio` (this clip will not get audio, e.g. the FX sits on a folder parent track), `Reversed by REAPER; HiFiShifter does not process it`. A split clip inherits its parent's audio immediately, so it no longer sits on a placeholder.
- Clip tooltips show the name only, never a file path.
- The clip context menu is **the same one** as in the standalone app: the Takes submenu, channel mode, loop and the fade-shape rows all appear. Entries the plug-in cannot do are shown disabled with a reason (glue, silence detection) instead of disappearing.

### Editing and Automatic Apply

Parameter editing works exactly as in the standalone app (see the [Parameter Editor](#5-parameter-editor) chapter). After every edit, the plugin renders in the background and applies the result back to the host automatically, with the current progress shown on the right of the status bar:

- `Waiting for host audio`: the host has not handed audio to the plugin yet.
- `Applying automatically...`: an edit is rendering; the first pass on a heavy model may be slow.
- `Applied`: there are no pending edits. Hovering shows the revision count (`Edits {edits} / Audio {audio}`).
- Only fully rendered audio replaces the current output; an unfinished render is never treated as a success.
- On failure it shows `Not applied: {error}`. Follow the hint to troubleshoot first, and do not force a reload repeatedly.
- Once ready, a `Reload from Host` button appears: use it to re-read the host state when models / permissions conflict. **It asks for confirmation while unapplied edits remain**, because reloading replaces the local curves with the host state; it is usually not meant to be your everyday "submit" button.

### Saving and Limitations

- Just save the project in REAPER; the plugin parameters are saved with the project. Closing the plugin window does not lose edits, and the standalone app does not need to stay running.
- **Interface settings are persisted**: the interface language, keybindings, appearance and custom themes, timeline and parameter-editor zoom, and the file browser's last directory all survive quitting REAPER.
- These settings are **shared with the standalone app**: appearance and keybindings adjusted in the standalone app apply in the plugin and vice versa.
- After reopening the project, the plugin must fetch the audio from the host again and finish its background preparation; the status bar first returns to `Waiting for host audio`.
- Before upgrading the plugin, save the project and fully quit REAPER; do not overwrite the loaded plugin folder while the host is running.
- Only forward, **linear** stretching is supported: reverse and non-linear stretching are out of scope, and the interface clearly marks `Reverse playback unsupported`.
- Harmonic Separation (HNSEP) processes the whole region, so very long material can use a lot of memory once `Harmonic Separation` is enabled; HiFiGAN inference, by contrast, runs in chunks.

### Troubleshooting

- The plugin cannot be found in REAPER: make sure you copied the **entire** `HiFiShifter.vst3` folder (not a single file inside it), that its parent directory is in REAPER's VST scan path, and that the WebView2 runtime and VC++ x64 runtime are installed.
- It keeps showing `Waiting for host audio` or cannot analyse: check that the track has valid audio, that the plugin is connected through the host's ARA mode, and the automatic-apply status at the right of the status bar.
- The plugin is on a track group (folder) parent track, items are visible but there is no waveform at all: REAPER manages ARA plug-ins per track and only hands the plug-in the audio on its own track. Add one HiFiShifter instance per audio track you want to process (see [Why a Track Group (Folder) Shows No Audio](#why-a-track-group-folder-shows-no-audio)).
- Parameters did not take effect, or differ after reopening the project: keep a copy of the project and the error text shown in the status bar; do not overwrite the project or force a reload repeatedly.
- An out-of-memory notice appears: it means the harmonic separation did not run this time, and the `Harmonic Separation`-related results are not silently dropped; use shorter material or turn `Harmonic Separation` off first.
- To submit logs or troubleshoot: the plugin log and the standalone app's log live in the **same folder** (the plugin writes `plugin.log`, the app writes `hifishifter.log`; different names, so neither overwrites the other and each rotates on its own). The menu `Help → Open Log Folder` opens that folder for you; if the host environment does not allow it, the path is shown so you can copy it. `Help → Export Diagnostics...` and `Options → Inference Device Benchmark` belong to the standalone app and are not available in the plugin.
- Uninstalling: if you installed via the installer, uninstall HiFiShifter VST3 from Windows "Programs and Features / Installed apps"; if you copied the folder by hand, just delete the `HiFiShifter.vst3` folder. In both cases the settings and the shared model library are kept (see [Model Files and Sharing](#model-files-and-sharing)).

### Connecting to the Host from the Standalone App

If you are used to editing in the standalone app, you can also let it connect to a running REAPER: first connect the plugin in REAPER as described above, then open the standalone app, choose `View → Connect to ARA Host...`, and in the floating session panel (titled `ARA / REAPER`) `Refresh Instances` → select an instance → `Connect`; afterwards you can `Submit to REAPER`, `Refresh Host` or `Disconnect`. The status shows `Not connected` / `Connected` / `Refreshed` / `Submitted` / `Disconnected`, and hovering shows the revision (`Revision {revision} · Model {model}`). When the current project has unsaved changes, connecting or refreshing first asks whether to replace it with the host snapshot.

## 9. Logs and Troubleshooting

HiFiShifter automatically writes its run log to the platform-standard log directory — no command-line flags required. When you run into a problem, attaching the log file(s) to an issue helps a lot with diagnosis:

| OS      | Log directory                                  |
| ------- | ---------------------------------------------- |
| Windows | `%LOCALAPPDATA%\com.arounder.hifishifter\logs` |
| macOS   | `~/Library/Logs/com.arounder.hifishifter`      |
| Linux   | `~/.local/share/com.arounder.hifishifter/logs` |

- Use `Help → Open Log Folder` to jump straight to the log files.
- The plugin (ARA) log shares the app's **folder, format and rotation**, and is named `plugin.log`. Both logs carry the version, date and time, so they can be submitted together.
- Use `Help → Export Diagnostics...` to generate a diagnostics package zip (system info + all logs + inference-device benchmark results) and attach it to your issue.
- Logs rotate automatically by size: 8 MiB per file, with up to 3 historical copies kept (`hifishifter.1.log` ~ `hifishifter.3.log`).
    - Frequently repeating errors / warnings are throttled automatically: a given log site emits at most one message per 10-second window, and suppressed messages are summarized in a `[throttled]` line before the next one.
- Frontend and backend errors are written to the same log file, so a single file tells the whole story in chronological order.

Advanced options (not needed for regular use):

- Launch flag `--log-file=<path>`: write the log to a specific path; `--log-file=-` disables file logging.
- Environment variable `HIFISHIFTER_LOG=debug` (or `trace` / `info` / `warn` / `error`): adjust log verbosity; release builds default to `info`.
- Environment variable `HIFISHIFTER_LOG_DIR`: override the default log directory.
- Run `HiFiShifter --benchmark` in a terminal: run the inference-device benchmark and print the results as JSON.
