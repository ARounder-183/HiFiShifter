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

> **Note**: WSL2 does not expose hardware Vulkan to Linux guests. WebGPU/Dawn can only use Lavapipe (CPU software rendering), which is extremely slow. For GPU acceleration on WSL2, use the Windows native build with DirectML instead.

In the menu `Options → Inference Device`, you can select `Auto`, `CPU`, or `GPU`. Run the benchmark to compare per-device inference latency and pick the fastest option.

**WebView Information**: HiFiShifter is built with the Rust + Tauri framework and requires a WebView component to display its interface.

- **Windows**: Requires Edge WebView2. Windows 10 (version 1803 and later) and Windows 11 have it preinstalled, so no additional action is needed. If you are using an older Windows version or the component is missing, the installer will prompt you to download it automatically. You can also refer to the [Tauri official documentation](https://tauri.app/start/prerequisites/#webview2) for details. General users can simply run the installer without worry.
- **macOS**: WebKit is provided by the system, no extra installation is required.
- **Linux**: Requires WebKitGTK. Most major distributions (e.g., Ubuntu, Fedora, Arch Linux) include it by default. If you see a missing component error, use your package manager to install `webkit2gtk` (e.g., `sudo apt install webkit2gtk`). Refer to your distribution's documentation for specifics.

## 2. Menu

The `File` menu allows you to open and save HiFiShifter project files, as well as import media files (audio or video), import Reaper projects (`.rpp`), import VocalShifter projects (`.vshp` or `.vsp`), import MIDI files, and export audio.

HiFiShifter project files have the extensions `.hshp` or `.hsp`. Additionally, `Save As` supports saving the project as a plain text `json` file, or packaging the current project together with all used media files into an archive zip `.zip`.

`File → Import Media File` accepts common audio formats (`wav`, `mp3`, `flac`, `ogg`, `m4a`, `aac`, etc.) and common video containers (`mp4`, `mov`, `mkv`, `webm`, `avi`, `wmv`, `ts`, `mpg`, etc.). Video files contribute their audio stream only; the picture track is never processed. Video containers and additional audio codecs are decoded through pure-Rust Symphonia with all features plus the FDK AAC / libopus adapters. The container's default audio stream is selected by default; if a video contains multiple audio streams, the File-menu import flow opens an audio-stream picker and extracts the selected stream to `<name>.hifi_audio_<n>.wav` next to the source video before importing it as a regular audio clip. The extracted WAV cache is reused by waveform, playback, pitch-analysis and rendering paths. Source paths referenced by REAPER project files / REAPER clipboard data and VocalShifter project files / VocalShifter clipboard data accept the same video containers and use their audio tracks; unsupported video encodings are skipped and counted in `skipped_files`.

The automatic backup feature allows you to configure backups for your project files, with two modes: `Backup on save` and `Timed backups`.

- `Backup on save`: When you overwrite the project file via save, the previous project file will automatically be renamed to a backup file with `-bak` appended to the original filename (for example, `.hshp-bak` or `.hsp-bak`). Enabled by default.
- `Timed backups`: Automatically create backup project files at the interval and path you set while you edit the project. Disabled by default.

### Importing a HiFiShifter Project

`File → Import HiFiShifter Project...` merges all content from another `.hshp` / `.hsp` / `.json` project into the current project without closing it.

- `Keep original timeline position`: Imported clips stay at their original timeline positions.
- `Place at playhead`: The imported content is shifted so its earliest clip starts at the current playhead.
- `Import tempo map`: Only available when the current project has no tempo map. After import, the source project's initial BPM, time signature and scale become the current project's baseline.
- All tracks, child tracks, clips, parameter curves and group relationships are assigned fresh IDs, so they never collide with the current project. Audio source files are resolved relative to the imported project file first; missing files still trigger the interactive relink dialog.
- Notes from the imported project are appended to the current project notebook instead of replacing it.

### Cross-Process Copy / Cut / Paste

Structured HiFiShifter clipboard operations are now written by the backend directly to the operating-system clipboard. Binary data is stored in a native private clipboard format, while the normal text slot only receives a short human-readable summary such as `HiFiShifter: 3 clip(s) copied...`. Pasting into a regular text box therefore no longer produces base64 garbage. Readers still understand the legacy text envelope. The following operations work between two running HiFiShifter processes:

- Select clips in the timeline and press `Ctrl + C` (or use the context-menu `Copy`), then press `Ctrl + V` (or `Paste` from the empty track-area context menu) in the other process. Clip automation curves are pasted together with the clips.
- Right-click a track header and choose `Copy Track` / `Cut Track`; in the other process press `Ctrl + V` or use the empty track-area context menu to paste the complete track group (child tracks, clips and full parameter curves).
- Parameter-curve copy/paste in the Parameter Editor also uses the backend clipboard and works across processes.
- Select one or more clips and choose `Edit → Copy to Reaper Clipboard` (`Ctrl + Shift + C`). This writes REAPERMedia data containing the audio source, position, length, play rate, reverse, fades, mute and MIDI notes; press `Ctrl + V` in REAPER to paste. Clips without a usable source are skipped.
- Paste behavior: `Ctrl + V` (or `Edit → Paste`) now always behaves like "Paste into Selected Track": it flattens the pasted content onto the currently selected track and never creates extra child tracks based on whether the copied selection covered a complete track group.
- Pasting clips that carry pitch-edit automation automatically enables compose mode on the target root track, invalidates affected render caches, and schedules pitch analysis plus background rendering; no save/reopen is required.
- `Edit → Paste as New Tracks` (`Ctrl + Alt + V`) force-creates new root-track groups using the source hierarchy.


The `Edit` menu contains common track / timeline editing and clipboard import operations; it no longer includes parameter-curve processing items. Operations such as `Initialize`, `Transpose by Cents`, `Transpose by Degrees`, `Set To`, `Average`, `Smooth`, `Add Vibrato`, `Quantize`, and `Mean Quantize` are available only from the Parameter Editor context menu.

- **Paste Reaper Clipboard Data**: After you copy Items, tracks, or MIDI notes in Reaper, this function quickly imports the Reaper clipboard data into HiFiShifter.
    - Item data: Imports as note clips in HiFiShifter, preserving tuning data (both global tuning and pitch envelopes) from Reaper.
    - Track data: Imports tracks along with their items as tracks and audio clips in HiFiShifter, preserving track groups.
    - MIDI note data: After exporting note data from other DAWs (Reaper, FL Studio, etc.) to the clipboard as MIDI note data, use the `Select` tool in the Parameter Editor to select a pitch curve segment in HiFiShifter, then you can import the clipboard MIDI note data into that segment. For a detailed introduction to MIDI import, see the [Pitch Reference Clip](#pitch-reference-clip) section.

- **Paste VocalShifter Clipboard Data**: After you copy parameter curves, audio clips, or tracks in VocalShifter or VocalShifter LE, this function quickly imports the data into HiFiShifter.
    - Parameter curve data: After selecting a parameter curve segment with the `Select` tool in the Parameter Editor, you can import VocalShifter clipboard parameter curve data into that segment.
    - Audio clip data: Imports as note clips in HiFiShifter, preserving various parameter curve data.
    - Track data: Imports tracks along with their audio clips into HiFiShifter. Note that HiFiShifter currently cannot distinguish whether your last copied content was an audio clip or a track. If you intend to import a track, before performing the copy track operation in VocalShifter, ensure that no audio clip is selected in the VocalShifter project; otherwise, only the selected audio clips will be imported.

The `View` menu contains options related to the interface display.

- `Clipboard Preview`: Toggle the parameter editor clipboard preview.
- `Popup Param Values`: Toggle the parameter value popup.
- `Time Display`: Lets you choose the primary/secondary time units of the timeline ruler and open `Timeline Display Settings...`.
- `Theme: Dark / Light`: Switch the current theme.
- `Appearance Settings`: Open the appearance settings window.

The `Options` menu allows you to modify various settings of HiFiShifter.

- `Project Stretch Override`: Allows you to modify the current project's stretching algorithm.
- `Global Stretch Default`: Allows you to modify the default global stretching algorithm.
- `Inference Device`: Allows you to set the inference device used for rendering. Currently supports `Auto`, `CPU`, and `GPU`. You can run a benchmark from this menu to test the performance of each device (the benchmark will show specific backends such as GPU (DirectML), GPU (WebGPU), etc.). `GPU` is only available in the corresponding GPU build of HiFiShifter.
- `Background Pre-render`: When enabled, after opening a project or editing parameters, the edited parameters are automatically pre-rendered in the background, and you can play the already-rendered portions even while rendering is still in progress. When disabled, rendering only begins when playback starts, and you must wait for rendering to complete before the timeline plays normally. Enabled by default. Disabling it reduces rendering frequency and saves performance.
- `Keyboard Shortcuts`: Allows you to configure HiFiShifter's keybindings. Several presets are available.

## 3. Track View

The general operation logic and shortcuts can be referenced from DAWs like Reaper, VocalShifter, VEGAS Pro. You can customize your shortcut preferences via `Options -> Keyboard Shortcuts`. The following descriptions are based on default shortcuts.

The track view is one of HiFiShifter's core features, allowing you to crop, splice, and edit audio clips. Its operation logic is largely based on Reaper.

For view navigation, drag the middle mouse button (hold the scroll wheel) to pan. Horizontal/vertical zoom or scrolling can be done by holding modifiers like `Ctrl`, `Alt`, `Shift` while scrolling the mouse wheel. These modifiers can be adjusted in the shortcut settings.

Common shortcuts:
> **macOS**: `Ctrl` below maps to `Command (⌘)` and `Alt` maps to `Option (⌥)`. Use `⌘ + click` to toggle individual selection, `⇧ + click` for range selection, `⌘ + drag` for copy-drag, and `⌘ + ⇧ + Z` for redo.


- `Space`: Play / Pause (does not return to start)
- `Enter`: Play / Stop (returns to start)
- `S`: Split
- `G`: Group
- `U`: Ungroup
- `Ctrl + C`: Copy
- `Ctrl + V`: Paste
- `Ctrl + Alt + V`: Paste as New Tracks
- `Ctrl + Shift + C`: Copy to Reaper Clipboard
- `Ctrl + Z`: Undo
- `Ctrl + Y`: Redo (`⌘ + ⇧ + Z` on macOS)
- `Ctrl + A`: Select All

- `Delete`: Delete audio clip
- `-` / `=`: Shift parameter curve down/up for selected clips
- Modifier `Alt`: Hold while dragging clip start/end to stretch the clip; drag the middle of the clip to slip-edit (internal content offset)
- Modifier `Shift`: Hold to temporarily toggle snap
- Modifier `Ctrl` (`⌘` on macOS): Hold while dragging a clip to copy it

The small circle at the top-left of a clip is a volume adjustment knob, the `M` button can mute that clip individually, and the `F` button can open that clip's formant editing menu. The left and right edges of a clip allow adjusting fade-in/fade-out envelope lengths.

Right-click a clip to open the context menu, which includes functions like `Reverse`, `Normalize`, `Convert to Pitch Reference Clip`, `Export MIDI`, `Fade Curve Type`. If you select multiple clips on the same track, the context menu allows `Glue` to merge them into a single audio clip.

Select multiple clips, then choose `Group` (or press `G`) in the context menu to group them. Similar to Reaper or VEGAS Pro, clips in the same group are linked during edits. Click the chain button at the top-left of a clip to temporarily disable or enable the group's linked editing. Select grouped clips and choose `Ungroup` (or press `U`) to remove them from the group.

On the left side of the track view is the track header area, where you can add or delete tracks, adjust track parameters, etc. Right-click a track to clone, copy or cut it; right-click empty track space to paste clips or a complete copied track group.

Similar to Reaper, HiFiShifter tracks support track groups. Drag one track header onto another in the track header area to create a track group. A track group shares a single parameter panel. In practice, it is recommended to organize by "one voice part per track group".

Track view toolbar buttons:

- `BPM`: Adjust the global tempo BPM of the project (with a Tempo Map active, this adjusts the nearest point at the playhead — see "Tempo Map" below).
- `Time Sig.`: Set the project time signature (beats per bar) via its numerator field and denominator dropdown (the denominator supports the mouse wheel).
- `Grid`: Set the grid spacing for the project.
- `Base Scale`: Adjust the global base scale setting for the project, supports custom scales. The scale function is mainly used with `Pitch Snap` and other pitch-related adjustments.
- `Stop` and `Play/Pause` buttons: Control playback.
- `File Browser`: Open the HiFiShifter file browser window.
- `Notepad`: Open the HiFiShifter notepad window, which records and displays Markdown-formatted text.
- `Auto Crossfade`: Similar to Reaper/VEGAS Pro, when enabled, moving clips that overlap will automatically adjust crossfade envelopes.
- `Split Transition`: Modeled after Reaper/VEGAS Pro split fades, enabled by default. After splitting, it automatically adds a fade-out to the left clip and a fade-in to the right clip at the split point, or extends both clips into an overlap, to reduce clicks. Left-click toggles it; right-click opens detailed settings.
- `Snap`: When enabled, clip adjustments attempt to snap to the grid, other object edges, the playhead and other configured targets. Hold `Shift` to temporarily toggle snap.
- `Zoom at Playhead`: When enabled, horizontal zoom centers on the playhead; otherwise, centers on the mouse cursor.
- `Auto Scroll`: When enabled, the view automatically scrolls horizontally during playback to follow the playhead.
- `Allow Param Editor to Move Playhead`: When disabled, clicking in the parameter editor will not move the playhead; only clicking the track view or the timecode area of the parameter editor moves the playhead.
- `Allow Timeline Clicks to Switch Track`: Enabled by default. When enabled, clicking a clip or empty area in the timeline switches the current track, and the parameter editor follows the newly selected track. When disabled, only clicking a track header changes the current track.
- `Ignore Grouping`: When enabled, edits to grouped audio clips will globally ignore group-linked editing.

### Snap / Grid Settings

The `Snap` button on the timeline toolbar is the quick toggle for snapping. Left-click toggles the snap master switch; right-click opens the `Snap / Grid Settings` dialog directly. The same dialog is available from `Options -> Snap / Grid Settings`. All settings are persisted and restored on the next launch.

Settings:

- `Grid`: show/hide grid lines, choose the grid spacing (normal / dotted / triplet, from `1/1` to `1/64`), and set a minimum pixel spacing so dense grids stay readable. With `Swing` enabled, odd grid lines are shifted by a percentage (0–100%); when `Adjust all items when changing swing` is checked, existing clips are automatically re-aligned to the new swing grid.
- `Snap Master`: `Enable snapping` is the master switch; `Snap distance` defines how close (in pixels, default 4) the pointer must be to a target for snapping to trigger; `Snap relative to grid` preserves the item's original offset relative to the grid instead of snapping to absolute grid positions.
- `Snap Targets / Objects`: independently configure whether `Media items`, `Selection`, and `Cursor` snap to `Selection / cursor` or to the grid. For example, clips can snap only to the grid while the playhead snaps to both the grid and clip edges.
- `Snap Behavior`: `Snap settings follow grid visibility` stops snapping to the grid when grid lines are hidden; `Snap to grid at any distance` is an aggressive mode that always forces the drag to the grid; `Use independent snap spacing` provides a dedicated snap spacing (separate from the display grid) with its own minimum pixel value.
- `Item & Special Interactions`: choose whether only the item start snaps or the `snap offset` (content start) is also used; enable `Snap across tracks` and set how many tracks away targets are considered; `Snap razor edits` applies snapping to split operations (`S`).
- `Advanced`: `Snap to project sample rate` provides sample-accurate snapping; `Snap media item edges to source media start/end` pulls trimmed clip edges back to the original source bounds; `Force selections to be multiples of` rounds marquee selections to the selected grid; `Use the same grid division in arrange view and MIDI editor` keeps the timeline and parameter editor grid precision synchronized.

HiFiShifter's current data model has no REAPER-style take markers, fixed-lane comp areas, or standalone automation items, so those concepts are intentionally not present in the dialog. All other snapping behaviors are implemented for HiFiShifter's Clip model.

### Split Transition

`Split Transition` is designed to reduce clicks at split points and is enabled by default. Due to time-stretching algorithms and similar factors, clips can click at the newly created boundary after a split. This feature follows the approach used by Reaper/VEGAS Pro and automatically handles the boundary after every split.

The `Split Transition` toolbar button is located to the right of `Auto Crossfade`. Left-click toggles the feature; right-click opens detailed settings.

- `Fades Only`: After splitting, automatically adds a fade-out of length X to the left clip and a fade-in of length X to the right clip. The two clips do not overlap.
- `Extend & Overlap` (default): After splitting, automatically extends the left clip's tail forward by X and the right clip's head backward by X, creating a 2X-second overlap. The extension keeps the source material at the same timeline position and correctly accounts for playback rate. Extensions are clamped to the clip source's actual length. When `Auto Crossfade` is also enabled, a crossfade is automatically created across this overlap.
- `Transition Length X`: The fade/overlap length used by both modes, 0.01 seconds by default.
- `Transition Length Unit`: Choose `Seconds` or `Percent`. Percent defaults to 1% and is calculated from the combined full length of the two clips created by the split; for example, two clips totaling 10 seconds at 1% gives 0.1 seconds.
- `Fade Curve`: Selects the fade curve used by split transitions.
- `Overlap Crossfade`: With `Follow Auto Crossfade`, crossfades are only added to the overlap when `Auto Crossfade` is enabled. With `Always Apply`, crossfades are always added to the overlap.

### Timeline Time Display

The ruler at the top of the timeline automatically refines its tick labels as you zoom horizontally: at a small zoom level only bars are shown (`1.1`, `2.1`); zooming in progressively refines to half notes (`1.1`, `1.3`), quarter notes (`1.1`, `1.2`), eighth notes (`1.1`, `1.1.500`), and further to 16th/32nd notes. The finest precision is limited by the `Grid` setting.

Four time units are supported:

- `Bar.Beat.Subdivision` (default primary unit): `1.2.500` means bar 1, beat 2, plus 0.5 beat (1000 subdivisions = 1 beat). `..` is appended when the value is inexact (e.g. `1.1.333..`).
- `Bar.Division`: `1.17/32` means the 17th division of bar 1 using a `1/32` grid. Divisions follow the `Grid` setting; triplet grids produce integer division counts (e.g. `1.2/12`), while dotted grids may produce fractional counts (e.g. `1.2/2.6667`).
- `Seconds`: absolute seconds, e.g. `1234.5678`; `..` is appended when inexact.
- `H:MM:SS.mmm` (default secondary unit): the hour is omitted when zero and milliseconds always use 3 digits (e.g. `4:43.750`, `1:4:43.750`).

Right-click the ruler to choose the primary and secondary time units; the secondary unit can also be set to `None`. The same controls are available in `View -> Time Display` and the `Timeline Display Settings...` dialog. When both units are shown, they appear as two rows separated by a short faint line; when the secondary unit is `None` or identical to the primary unit, only the primary unit is shown, vertically centered.

The `TRACKS` header row on the left side of the track view shows the live playhead time on its right, formatted as `primary / secondary` (only the primary unit when no secondary is used), and refreshes automatically during playback and when the time format changes. The playhead time text is kept at fixed digit alignment (e.g. `1.1.000`, `0.000`) for easy reading.

Moving the mouse over the ruler shows the time at the pointer. The right-click menu also offers `Copy Playhead Time`, which copies the current playhead time as text to the clipboard. `Timeline Display Settings...` additionally lets you adjust the ruler label spacing and toggle the playhead time display in the track header.

### Tempo Map (Tempo / Time Signature / Scale Map)

HiFiShifter supports a project-level Tempo Map that lets you define different BPM, time signatures and scales at different positions of the timeline. A blank project has no Tempo Map data by default; once you add Tempo Map data to the project, the ruler automatically shows an extra Tempo Map row below the time units (separated by a short faint line), with each segment's tempo, time signature and scale shown on the point labels. When a segment's starting label scrolls out of view on the left, a floating label in the same style appears at the far left of the row showing that segment's parameters; it switches smoothly while scrolling horizontally, never overlaps the labels, and offers exactly the same interactions as the fixed labels (double-click to enter inline editing, right-click to open the edit dialog). The parameter editor ruler shows the same Tempo Map row.

- Adding points: right-click the ruler and use `Add Tempo / Time Signature / Scale Change Here…` to create a point at the clicked position and open the edit dialog (BPM, time signature and scale in the dialog all support mouse-wheel adjustment; while holding the `Fine Adjust` modifier, the BPM wheel steps by 0.1). Double-clicking an empty area of the Tempo Map row creates a point right there — the new point inherits the effective BPM at that position, and its time signature and scale follow the previous ones — and immediately enters that label's inline editing state (type text such as `120 4/4 - C / Am`, press `Enter` to confirm or `Esc` to cancel). The double-click addition and the subsequent inline edit are merged into one edit: a single undo completely reverts the addition; pressing `Esc` cancels the edit and discards the addition (same cancel semantics as the edit dialog).
- Editing points: double-click a point label on the Tempo Map row (including the initial point at 0) to change the BPM (10-960), the time-signature numerator (1-32) / denominator (1/2/4/8/16/32) and the scale. Double-clicking a label turns it into an inline text box where you can type text such as `120 4/4 - C / Am`; press `Enter` or click elsewhere to commit, `Esc` to cancel, and unrecognized input is silently discarded. Right-clicking a label (including the floating label at the far left) opens the "Tempo Map Point" edit dialog directly instead of the ruler context menu (right-clicking the text box opens the same dialog with the values inherited from the text box; confirming it merges both edits into one undo step). While typing you do not have to leave the edit state first: pressing the left button and dragging the pointer out of the text box's bounds (with a small tolerance) is interpreted as "confirm the edit and drag the label" — the edit is committed immediately and the label follows the pointer (like a normal drag; everything is submitted on release as a single undo step). Drags that stay inside the text box always keep the usual text-selection behavior, no matter how long. The initial point (at 0) cannot be dragged.
- Following the previous time signature / scale: each point's time signature can be set to `Follow Previous Time Signature` (via its checkbox) and its scale to `Follow Previous Scale`; both options show the actual effective value next to them. Newly added points follow by default. The flag text strictly follows these rules: with both following, only the BPM is shown (e.g. `120`); with an explicit time signature only, BPM and time signature are shown (e.g. `120 4/4`); with an explicit scale only, BPM and scale are shown (e.g. `120 - C / Am`); with both explicit, everything is shown (e.g. `120 4/4 - C / Am`). Typed text is parsed with exactly the same rules.
- Initial point as the project record: the project's global BPM, time signature (numerator and denominator) and scale are recorded at the initial point, shown on the Tempo Map row as e.g. `120 4/4 - C / Am`. Changing the project scale also updates the initial point's scale; editing the initial point's scale also updates the project scale. The initial point cannot follow a previous time signature.
- Moving / deleting points: any point except the first can be dragged horizontally and uses the same snap targets, threshold and modifier behavior as clips; `Delete This Point` in the right-click menu or the dialog removes a point; `Clear Tempo Map` removes all data and the project falls back to its global BPM / time signature / scale.
- Flag tooltips: hovering a flag on the Tempo Map row (or the floating label at the far left) shows the app's unified custom tooltip. It displays the position in your configured primary / secondary time units, plus the actual BPM, time signature and scale (followed settings show their actual effective values), for example:

  ```text
  Position: 2.1.000 / 0:2.000
  BPM: 120
  Time Signature: 4/4
  Scale: C / Am
  ```

- Visibility and corner buttons: `View -> Tempo Map` (on by default) toggles the Tempo Map row. The row is hidden when the project has no Tempo Map data. The corner rectangles at the top-left of the track view (the "Tracks" header area) and of the parameter editor each have a small button in their bottom-right corner: when there is no Tempo Map, or the Tempo Map is hidden, clicking the button shows the Tempo Map and creates one for the project if it has none (containing only the initial point at 0, i.e. the project record — no other points are added); when a Tempo Map exists and is being shown, the button turns red as a warning, and clicking it opens a confirmation dialog where you can `Clear` (delete all Tempo Map data) or `Hide Only` (same effect as `View -> Tempo Map`; data is kept). Toggling `View -> Tempo Map` automatically updates the button's icon and mode.
- Grid and ruler: when a Tempo Map exists, ruler ticks, bar/beat labels and the background grid re-align at every point and are computed per segment (non-uniform grid), and snapping follows the Tempo Map.
- Toolbar BPM / time signature / base scale: the top-left `Time Sig.` control consists of a numerator field and a denominator dropdown (the denominator supports the mouse wheel). With a Tempo Map active, the toolbar shows the effective values at the playhead; editing BPM, the time-signature numerator/denominator or the base scale updates the nearest point at or before the playhead (no point is created automatically). The base scale now represents that nearest point's scale setting rather than the project-wide scale.
- Scale integration: scale changes in the Tempo Map affect pitch snapping (scale mode), scale highlighting, the `Project Scale` option of degree transposition / quantization / mean quantization, and the degree-difference rendering of child tracks. When the parameter editor selection spans a scale change, the `Project Scale` option label shows "selection is affected by Tempo Map scale"; changing the project scale or a Tempo Map scale automatically invalidates the related render caches and re-renders in the background when `Background Pre-render` is enabled.
- Import: when importing MIDI as pitch reference clips, you can enable `Import as Tempo Map` in the import dialog and separately choose whether to import tempo (on by default), time signature (on by default) and key signature (off by default, since most MIDI files only carry a default C-major key signature, which can be misleading); importing a REAPER project (.rpp) automatically imports its project-level tempo and time-signature changes (tempo changes inside REAPER MIDI items are not imported).

## 4. File Browser

The file browser allows you to open a specific folder, search and sort audio and video media files within it, and drag them into the HiFiShifter track view. Video files use a purple icon; audio files use a blue icon. Search supports regular expressions. Clicking a media file automatically plays a preview (videos preview their audio track). You can hold `Ctrl` and `Shift` for multi-selection. Left-dragging files adds one or more media files across time into the timeline. Right-dragging files brings up a menu with `Add Across Time` / `Add Across Tracks`. `Add Across Tracks` allows you to add multiple media audio clips vertically across multiple tracks.

When the track view has focus, press `Ctrl + F` to open the Quick Search window. This is a simplified version of the file browser, allowing you to quickly search and preview audio/video media files within a folder (videos preview their audio track) and add them to the timeline.

## 5. Parameter Editor

The parameter editor is one of HiFiShifter's core features, allowing you to edit various parameters of the currently selected track.

To enable parameter editing for a track, you must first press the track's `C` (Compose) button and wait for audio analysis to complete. HiFiShifter uses offline rendering; after each parameter edit, you must wait for the parameters to re-render before auditioning.

The `Sync Timeline View` button (link icon) to the left of the `Parameter Editor` label in the parameter editor header toggles horizontal synchronization with the track view. When enabled, the two views share the same horizontal position and zoom in both directions: scrolling or zooming in either view updates the other, and enabling sync aligns the parameter editor to the track view as the reference. Because the track view has a track-header area on its left, the parameter editor automatically compensates for the horizontal offset so grid lines and time-axis ticks line up at the same on-screen positions in both panels. When disabled (default), the two views zoom and scroll independently.

### 1. Algorithms and Parameters

The current version of HiFiShifter supports three vocal tuning algorithms and their parameters:

- **PC-NSF-HiFiGAN**: OpenVPI's open-source hifigan vocoder specialized for singing voices, also HiFiShifter's default algorithm.
    - `Pitch`: Adjust the pitch of the voice.
    - `Breath Gain`: After enabling breath, allows adjusting the breath volume, based on the VR-hnsep model.
    - `Tension`: Adjust the tension of the voice.
    - `Formant Shift`: Adjust formant shift.
    - `Volume`: Adjust the volume.
- **World**: Open-source high-quality speech analysis and synthesis algorithm.
    - `Pitch`: Adjust the pitch of the voice.
- **VsLib**: Official voice analysis and synthesis library from VocalShifter. VsLib is only available on Windows x86_64.
    - `Pitch`: Adjust the pitch.
    - `Volume`: Adjust the volume.
    - `Pan`: Adjust panning.
    - `Formant Shift`: Adjust formant shift.
    - `Breath`: Adjust breathiness.
    - `Synth Mode`: Adjust the synthesis mode; some parameters may be ineffective in certain modes.
        - `Mono`: VocalShifter's M algorithm, monophonic instrument mode.
        - `Mono (Formant)`: VocalShifter's V algorithm, monophonic vocal mode.
        - `Chorus`: VocalShifter's P algorithm, harmony mode.

A track can only use one algorithm; if you want to use multiple algorithms, separate them into different tracks.

A track group shares a single set of parameters, with child tracks inheriting parameters from the root track. Additionally, child tracks have three extra parameters: `Cents Offset`, `Degree Offset` and `Formant Offset`, which conveniently adjust pitch and timbre relative to the root track. The `Degree Offset` uses the project's scale setting as its reference. `Formant Offset` is drawn in cents per frame and accumulates along the root → parent → current-child hierarchy; it is only shown when the track-group algorithm supports `Formant Shift` (NSF-HiFiGAN and VocalShifter / vslib), and editing snaps to 50-cent steps. In the parameter-editor toolbar, child `Cents Offset` / `Degree Offset` are grouped inside the `Pitch` button dropdown, and `Formant Offset` is grouped inside the `Formant Shift` button dropdown.

After copying a `Pitch` segment using the Select tool, you can paste it onto `Cents Offset` or `Degree Offset`, and HiFiShifter will automatically calculate and apply the appropriate offset.

### 2. Select Tool

The Select tool allows you to select a segment of a parameter curve, drag it, or right-click to open a context menu for parameter adjustments.

Common shortcuts:

- `Ctrl + C`: Copy
- `Ctrl + V`: Paste
- `Ctrl + Z`: Undo
- `Ctrl + Y`: Redo (`⌘ + ⇧ + Z` on macOS)
- `Ctrl + A`: Select All

- `BackSpace`: Initialize
- `[` / `]`: Shift parameter curve down/up within the selection

Left-drag on a selected curve to move it vertically, horizontally, or freely, depending on the `Drag Direction` setting. While left-dragging, press the right button to quickly toggle drag direction.

Right-drag on a selected curve to adjust its amplitude: drag up to increase amplitude, down to decrease.

Right-click in the parameter editor to open a context menu with operations such as `Initialize`, `Transpose by Cents`, `Transpose by Degrees`, `Set To`, `Average`, `Smooth`, `Add Vibrato`, `Quantize`, `Mean Quantize`, etc.

Hold `Alt` to enter four-point editing mode for the selected curve. Similar to the feature in VocalShifter, dragging the four points allows you to bend the curve.

Hold `Alt` and drag the edge of the selection area to stretch the parameter curve within the selection.

### 3. Draw Tool

The Draw tool allows you to draw parameter curves.

Left-drag to draw freely or horizontally, depending on the `Drag Direction` setting. While left-dragging, press the right button to quickly toggle drag direction.

Right-drag resets the current curve.

### 4. Line/Vibrato Tool

Right-click the Draw tool button to switch to the Line/Vibrato tool. This tool allows you to draw straight lines or vibrato.

Left-drag to draw a straight line freely or horizontally, depending on the `Drag Direction` setting. While left-dragging, press the right button to quickly toggle drag direction.

While left-dragging, scroll the mouse wheel to superimpose a horizontal sine wave; scrolling adjusts the amplitude. Hold `Alt` while scrolling to adjust frequency. Hold the `Param Fine Adjust` modifier (default `Ctrl`) to fine-tune while scrolling.

Right-drag resets the current curve.

Press `Tab` to cycle through editing tools (Select / Draw-type tools).

### 5. Pitch Snap

When editing pitch parameters with any tool, Pitch Snap allows you to snap edits to semitones or scale degrees. Hold `Shift` to temporarily toggle snap.

Right-click the Pitch Snap button to open the Pitch Snap Settings menu, where you can adjust the quantization unit and tolerance.

- `Quantize Unit`: Two types: `Semitone` and `Scale`. When set to Scale, the reference scale is the project's current scale.
- `Tolerance`: Adjusts the snap tolerance range. Edits within the tolerance are not snapped; edits outside the tolerance are snapped to the nearest tolerance edge.

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

### 6. Pitch Reference Clip

A Pitch Reference Clip on a track is a special type of audio clip that stores a pitch curve on the timeline.

Pitch Reference Clips can be created through the following methods:

- Import MIDI via the `File` menu or by dragging a file. This opens the MIDI Import dialog.
    - MIDI File: Allows you to select a MIDI file to import. Also supports parsing MIDI data exported to the system clipboard by other DAWs. DAWs confirmed to support system clipboard MIDI data transfer include Reaper and FL Studio.
        - Reaper: In Reaper's MIDI Editor, select notes and copy them to export the selected note data to the system clipboard for HiFiShifter to read. Note that since Reaper's clipboard note data does not include BPM information, when importing, you can use the current project BPM or specify one manually.
        - FL Studio: In FL Studio's Piano Roll, click the small triangle in the top-left corner and select `File` -> `Copy to MIDI Clipboard` to export all notes of the current channel to the system clipboard for HiFiShifter to read.
    - Track Selection: Allows you to select which MIDI tracks to import.
    - Import MIDI BPM as Project BPM: When enabled, imports the MIDI's initial BPM as the project BPM. HiFiShifter still does not support variable BPM.
    - Note BPM: Configures the BPM mapping for imported notes.
        - MIDI own BPM: Import directly with the MIDI's own BPM without BPM mapping.
        - Current Project BPM: Map note BPM to the current project BPM before importing.
        - Specified BPM: Map note BPM to a manually specified BPM before importing.
    - Multi-track Merge: When enabled, automatically merges all selected tracks and notes, using the highest pitch note as the pitch curve parameter, ultimately importing only 1 Pitch Reference Clip. When disabled, attempts to split tracks and notes so that all notes can be imported as pitch curve parameters, which may result in multiple Pitch Reference Clips stacked vertically.
    - Fill Gaps Between Notes: When enabled, automatically fills the gaps between adjacent notes.
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

### 7. Other Features

Additional convenient features of the parameter editor:

- `Clipboard Preview`: After copying a parameter curve with the Select tool, the clipboard curve is displayed in real-time within the selection area to help with paste positioning. This option is in the `View` menu and no longer appears in the parameter editor toolbar.
- `Popup Param Values`: Shows parameter values when the mouse is near the curve or during drawing edits. This option is in the `View` menu and no longer appears in the parameter editor toolbar.
- `Lock Param Lines`: When dragging an audio clip on the track, whether to also move its corresponding parameter curves. All parameter editing in HiFiShifter is track-based; if not locked, edited curves will not follow the clip.
- `Smoothness`: Whether to automatically smooth parameter edits and the smoothing strength.
- `Reference Track Group`: When the parameter is `Pitch`, lets you choose other tracks and display pitch curves from other track groups as references in the pitch editor.
- `Import MIDI`: Allows you to select a MIDI file and import notes from one or more tracks as a pitch curve.

## 6. Export Audio

After completing all edits, use the `Export Audio` function in the `File` menu to export the HiFiShifter project as a wav audio file.

Parameters:

- `Export Type`: `Project` / `Separated Tracks`.
- `Time Range`: `All` / `Custom`. Custom allows setting start and end seconds.
- `Sample Rate`: Set the sample rate of the output WAV.
- `Bit Depth`: Set the bit depth of the output WAV.
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
- `Separated Track Targets Panel`: Select which tracks to export. By default, only non-muted normal tracks and root tracks are selected.
    - If you check a track that is originally muted, it will be exported regardless of mute state.
    - If you check a root track of a track group, the entire group is exported as a single audio file, and the exported audio excludes data from muted child tracks.
    - If you check a child track, it will be exported regardless of its own or its root track's mute state.

While typing a file path, you can click the `Placeholder` buttons to quickly insert the corresponding text.

All file path strings support time format strings like `%Y-%m-%d-%H-%M-%S`. If you want to include a literal `%` in the output path, use `%%` to escape it.
## 7. Recording

HiFiShifter can record directly onto the timeline. Recording starts playback from the current playhead so you can sing along with the project's backing audio; when recording stops, playback stops automatically and the take is imported into the timeline.

### Record button and shortcut

- The red circular button in the transport area starts and stops recording.
- The default shortcut is `Ctrl + R` (`⌘ + R` on macOS). You can change it in `Options -> Keybindings`.
- Right-click the record button to open recording settings quickly.

### Recording settings

Open `File -> Recording...` to configure:

- `Source`: choose one of three capture sources.
- `Input Device`: defaults to `System Default`, or choose any input device (microphone) on this computer.
- `Loopback Device`: when the source is `System Sound (Loopback)`, choose the output device to capture (default: `System Default Output`). On Windows this uses a native WASAPI loopback engine that honors the audio engine's silent-buffer flag (`AUDCLNT_BUFFERFLAGS_SILENT`) and writes zeros, so no hiss/rustle is recorded while nothing is playing.
- `Application`: when the source is `Application Audio`, capture only the selected program (e.g. browser or media player); use `Refresh` to re-enumerate programs currently producing audio. If the program restarts (new PID), it is re-matched automatically by its process name. On Windows 10 21H2 (build 20348) and newer the OS process-loopback API is used, with an automatic fallback that mutes other audio sessions on older builds. On Linux capture uses PipeWire (`pw-dump` / `pw-cat`). macOS does not support this mode yet.
- `Sample Rate` / `Bit Depth` / `Channels`: sample rate, bit depth (16 / 24 / 32-bit float) and channel count of the output WAV.
- `Input Gain`: pre-recording gain compensation.
- `Countdown`: enter the number of seconds (unit: seconds, 0–10); after clicking record, capture and playback wait until the countdown finishes.
- `Monitor input while recording`: route the input signal back to the output device while recording, useful when singing with headphones.
- `Auto-normalize after import`: after stopping and importing, normalize the new clip's peak to 0 dB.
- `Auto-stop at end of selected clips`: stop and import automatically when playback reaches the end of the selected clips.
- `Output Path Template`: supports `<ProjectFolder>`, `<ProjectName>` and strftime strings. Default: `<ProjectFolder>/HiFiShifter Record/%Y-%m-%d-%H-%M-%S.wav`. If the file already exists, a numeric suffix is added automatically so old takes are never overwritten.

### Recording workflow and import rules

1. Move the playhead to the desired start position and select the target track.
2. Click the record button (or press `Ctrl + R`; `⌘ + R` on macOS). Timeline playback starts from the playhead while capture begins.
3. Click record again (or press `Ctrl + R`; `⌘ + R` on macOS) to stop. Timeline playback stops at the same time.
4. If the selected track is completely empty within the recording range, the take is imported directly to it. Otherwise a new `Recording` track is created immediately below the selected track, the take is imported there, and the new track and clip are selected automatically. The track name follows the current UI language.
