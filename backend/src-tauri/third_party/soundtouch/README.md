# SoundTouch Vendor Assets

This directory contains the pinned Windows x64 SoundTouch runtime assets used by HiFiShifter:

- `SoundTouchDLL_x64.dll`
- `SoundTouchDLL_x64.lib`
- `SoundTouchDLL.h`
- `COPYING.TXT`

These files are project-managed vendor assets.

Current source package used for import:

- `soundtouch_dll-2.3.3`

Packaging paths:

- `build.rs` links against `SoundTouchDLL_x64.lib` and copies `SoundTouchDLL_x64.dll` next to the built binary.
- `tauri.conf.json` includes `SoundTouchDLL_x64.dll` in bundle resources.
- `scripts/build/pack-portable.ps1` includes `SoundTouchDLL_x64.dll` in the portable package.
