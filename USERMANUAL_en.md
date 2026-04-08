# HiFiShifter User Manual

[简体中文](USERMANUAL.md) | [繁體中文](USERMANUAL_zh-TW.md) | [English](USERMANUAL_en.md) | [日本語](USERMANUAL_ja.md) | [한국어](USERMANUAL_ko.md)

HiFiShifter is a graphical vocal editing and synthesis tool. It supports multi-track audio clip processing and uses various vocoders to achieve pitch correction and parameter adjustment for human voice, integrating splicing and tuning for human VOCALOID production.

## 1. Installation

Download the HiFiShifter installer corresponding to your operating system and architecture. By OS, there are `Windows`, `macOS`, and `Linux`. By architecture, there are `x86_64` and `arm64`.

- For Windows, NSIS installer (`installer`) and portable zip (`portable`) are provided. General users can directly use the installer.  
  If you are a Windows user and do not know the difference between `x86_64` and `arm64`, choose `x86_64`. Only if you clearly understand `arm64` and have a Windows ARM device, you may download the `arm64` version.

- For macOS, an unsigned dmg installer is provided. Since it is not signed, installation requires a few extra steps to allow the app to run.  
  macOS users with M-series chips should install the `arm64` version. Only older Intel users need the `x86_64` version.

- For Linux, an AppImage package is provided. You need to go to file `Properties -> Permissions` and check `Allow executing file as program`, then you can run it directly.

## 2. Feature Introduction

The general operation logic and shortcuts can be referenced from DAWs like Reaper, VocalShifter, VEGAS Pro. You can customize your shortcut preferences via `View -> Keyboard Shortcuts`. The following descriptions are based on default shortcuts.

### 2.1 Menu

The `File` menu allows you to open and save HiFiShifter project files, as well as import audio files, import Reaper projects (`*.rpp`), import VocalShifter projects (`*.vshp` or `*.vsp`), and export audio.

HiFiShifter project files have the extensions `*.hshp` or `*.hsp`. Additionally, `Save As` supports saving the project as a plain text `json` file, or packaging the current project together with all used media files into an archive zip `*.zip`. Currently, HiFiShifter only supports importing regular audio files, not video files.

The `Edit` menu allows various editing operations. Besides regular track and parameter editing, there are two special items: `Paste Reaper Clipboard Data` and `Paste VocalShifter Clipboard Data`.

- **Paste Reaper Clipboard Data**: After you copy Items, tracks, or MIDI notes in Reaper, this function quickly imports the Reaper clipboard data into HiFiShifter.
    - Item data: Imports as note clips in HiFiShifter, preserving tuning data (both global tuning and pitch envelopes) from Reaper.
    - Track data: Imports tracks along with their items as tracks and audio clips in HiFiShifter, preserving track groups.
    - MIDI note data: After selecting a pitch curve segment with the `Select` tool in the Parameter Editor, you can import Reaper clipboard MIDI note data into that segment. Note that Reaper clipboard MIDI note data does not contain tempo information; HiFiShifter will import it using the project BPM. Therefore, ensure the HiFiShifter project BPM matches your expected BPM before importing.

- **Paste VocalShifter Clipboard Data**: After you copy parameter curves, audio clips, or tracks in VocalShifter or VocalShifter LE, this function quickly imports the data into HiFiShifter.
    - Parameter curve data: After selecting a parameter curve segment with the `Select` tool in the Parameter Editor, you can import VocalShifter clipboard parameter curve data into that segment.
    - Audio clip data: Imports as note clips in HiFiShifter, preserving various parameter curve data.
    - Track data: Imports tracks along with their audio clips into HiFiShifter. Note that HiFiShifter currently cannot distinguish whether your last copied content was an audio clip or a track. If you intend to import a track, before performing the copy track operation in VocalShifter, ensure that no audio clip is selected in the VocalShifter project; otherwise, only the selected audio clips will be imported.

### 2.2 Track View

The track view is one of HiFiShifter's core features, allowing you to crop, splice, and edit audio clips. Its operation logic is largely based on Reaper.

For view navigation, drag the middle mouse button (hold the scroll wheel) to pan. Horizontal/vertical zoom or scrolling can be done by holding modifiers like `Ctrl`, `Alt`, `Shift` while scrolling the mouse wheel. These modifiers can be adjusted in the shortcut settings.

Common shortcuts:

- `Space`: Play / Pause (does not return to start)
- `Enter`: Play / Stop (returns to start)
- `S`: Split
- `Ctrl + C`: Copy
- `Ctrl + V`: Paste
- `Ctrl + Z`: Undo
- `Ctrl + Y`: Redo
- `Ctrl + A`: Select All
- `Ctrl + R`: Deselect
- `Delete`: Delete audio clip
- `-` / `=`: Shift parameter curve down/up for selected clips
- Modifier `Alt`: Hold while dragging clip start/end to stretch the clip; drag the middle of the clip to slip-edit (internal content offset)
- Modifier `Shift`: Hold to temporarily toggle grid snap
- Modifier `Ctrl`: Hold while dragging a clip to copy it

The `M` button at the top-left of a clip can mute that clip individually; the small circle is a volume adjustment knob. The left and right edges of a clip allow adjusting fade-in/fade-out envelope lengths.

Right-click a clip to open the context menu, which includes functions like `Reverse`, `Normalize`, `Fade Curve Type`. If you select multiple clips on the same track, the context menu allows `Glue` to merge them into a single audio clip.

On the left side of the track view is the track header area, where you can add or delete tracks, adjust track parameters, etc. Right-click a track to clone it.

Similar to Reaper, HiFiShifter tracks support track groups. Drag one track header onto another in the track header area to create a track group. A track group shares a single parameter panel. In practice, it is recommended to organize by "one voice part per track group".

Track view toolbar buttons:

- `BPM`: Adjust the global tempo BPM of the project. HiFiShifter currently does not support variable BPM.
- `Beats Per Bar`: Set the number of beats per bar for the project.
- `Grid`: Set the grid spacing for the project.
- `Base Scale`: Adjust the global base scale setting for the project, supports custom scales. The scale function is mainly used with `Pitch Snap` and other pitch-related adjustments.
- `Stop` and `Play/Pause` buttons: Control playback.
- `File Browser`: Open the HiFiShifter file browser window.
- `Auto Crossfade`: Similar to Reaper/VEGAS Pro, when enabled, moving clips that overlap will automatically adjust crossfade envelopes.
- `Grid Snap`: When enabled, all clip adjustments attempt to snap to grid. Hold `Shift` to temporarily toggle snap.
- `Zoom at Playhead`: When enabled, horizontal zoom centers on the playhead; otherwise, centers on the mouse cursor.
- `Allow Param Editor to Move Playhead`: When disabled, clicking in the parameter editor will not move the playhead; only clicking the track view or the timecode area of the parameter editor moves the playhead.
- `Auto Scroll`: When enabled, the view automatically scrolls horizontally during playback to follow the playhead.

### 2.3 File Browser

The file browser allows you to open a specific folder, search and sort audio files within it, and drag them into the HiFiShifter track view. Search supports regular expressions. Clicking an audio file automatically plays a preview. You can hold `Ctrl` and `Shift` for multi-selection. Left-dragging files adds one or more audio files across time into the timeline. Right-dragging files brings up a menu with `Add Across Time` / `Add Across Tracks`. `Add Across Tracks` allows you to add multiple audio clips vertically across multiple tracks.

When the track view has focus, press `Ctrl + F` to open the Quick Search window. This is a simplified version of the file browser, allowing you to quickly search and preview audio files within a folder and add them to the timeline.

### 2.4 Parameter Editor

The parameter editor is one of HiFiShifter's core features, allowing you to edit various parameters of the currently selected track.

To enable parameter editing for a track, you must first press the track's `C` (Compose) button and wait for audio analysis to complete. HiFiShifter uses offline rendering; after each parameter edit, you must wait for the parameters to re-render before auditioning.

#### Algorithms and Parameters

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

A track group shares a single set of parameters, with child tracks inheriting parameters from the root track. Additionally, child tracks have two extra parameters: `Cents Offset` and `Degree Offset`, which conveniently adjust pitch relative to the root track. The `Degree Offset` uses the project's scale setting as its reference.

After copying a `Pitch` segment using the Select tool, you can paste it onto `Cents Offset` or `Degree Offset`, and HiFiShifter will automatically calculate and apply the appropriate offset.

#### Editing Tools

##### Select Tool

The Select tool allows you to select a segment of a parameter curve, drag it, or right-click to open a context menu for parameter adjustments.

Common shortcuts:

- `Ctrl + C`: Copy
- `Ctrl + V`: Paste
- `Ctrl + Z`: Undo
- `Ctrl + Y`: Redo
- `Ctrl + A`: Select All
- `Ctrl + R`: Deselect
- `BackSpace`: Initialize
- `[` / `]`: Shift parameter curve down/up within the selection

Left-drag on a selected curve to move it vertically, horizontally, or freely, depending on the `Drag Direction` setting. While left-dragging, press the right button to quickly toggle drag direction.

Right-drag on a selected curve to adjust its amplitude: drag up to increase amplitude, down to decrease.

Right-click in the parameter editor to open a context menu with operations such as `Initialize`, `Transpose by Cents`, `Transpose by Degrees`, `Set To`, `Average`, `Smooth`, `Add Vibrato`, `Quantize`, `Mean Quantize`, etc.

Hold `Alt` to enter four-point editing mode for the selected curve. Similar to the feature in VocalShifter, dragging the four points allows you to bend the curve.

Hold `Alt` and drag the edge of the selection area to stretch the parameter curve within the selection.

##### Draw Tool

The Draw tool allows you to draw parameter curves.

Left-drag to draw freely or horizontally, depending on the `Drag Direction` setting. While left-dragging, press the right button to quickly toggle drag direction.

Right-drag resets the current curve.

##### Line/Vibrato Tool

Right-click the Draw tool button to switch to the Line/Vibrato tool. This tool allows you to draw straight lines or vibrato.

Left-drag to draw a straight line freely or horizontally, depending on the `Drag Direction` setting. While left-dragging, press the right button to quickly toggle drag direction.

While left-dragging, scroll the mouse wheel to superimpose a horizontal sine wave; scrolling adjusts the amplitude. Hold `Alt` while scrolling to adjust frequency. Hold the `Param Fine Adjust` modifier (default `Ctrl`) to fine-tune while scrolling.

Right-drag resets the current curve.

Press `Tab` to cycle through editing tools (Select / Draw-type tools).

##### Pitch Snap

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

This quickly creates harmonies by degree transposition.

##### Other Features

Additional convenient features of the parameter editor:

- `Clipboard Preview`: After copying a parameter curve with the Select tool, the clipboard curve is displayed in real-time within the selection area to help with paste positioning.
- `Popup Param Values`: Shows parameter values when the mouse is near the curve or during drawing edits.
- `Lock Param Lines`: When dragging an audio clip on the track, whether to also move its corresponding parameter curves. All parameter editing in HiFiShifter is track-based; if not locked, edited curves will not follow the clip.
- `Smoothness`: Whether to automatically smooth parameter edits and the smoothing strength.
- `Import MIDI`: Allows you to select a MIDI file and import notes from one or more tracks as a pitch curve.

### 2.5 Export Audio

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
