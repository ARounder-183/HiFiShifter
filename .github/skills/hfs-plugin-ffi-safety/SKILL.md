---
name: hfs-plugin-ffi-safety
description: Use when editing VST3, CLAP, ARA2, Objective-C, GTK, or Win32 host callbacks and real-time audio processing in backend/hifishifter-plugin, hifishifter-vst3, or hifishifter-clap.
---

# HiFiShifter Plugin FFI & Real-Time Audio Safety

In DAW plugin forms (`hifishifter-vst3`, `hifishifter-clap`, `hifishifter-plugin`), an unguarded panic or poisoned lock aborts the user's entire DAW process (`REAPER` / `Cubase` / `Studio One`).

## Pre-Commit Verification Checklist

1. **FFI Panic Guard on Every `extern` Function**:
   - Every `extern "system"`, `extern "C"`, or `extern "C-unwind"` function in production modules **must** contain `guard_ffi(`, `boundary(`, or audio-specific `catch_unwind`.
   - Choose the honest ABI failure return value (`K_NO_INTERFACE` for `queryInterface`, `0` / current count for refcounts, `std::ptr::null_mut()` for pointers, finite zeroed audio buffers for `process`).
2. **Poison Recovery on Every Mutex**:
   - Never call `.lock().unwrap()` or `.read().unwrap()` in production code.
   - Always use `MutexExt::lock_or_recover()` or `.unwrap_or_else(|e| e.into_inner())`.
3. **Real-Time Audio Callback Discipline**:
   - Inside `audio_block` / `audio_process` / CLAP `process`: **no heap allocations**, **no blocking locks**, **no file I/O**, **no logging** (push diagnostics to lock-free RT report ring and drain on the meter/worker thread).
   - Verify zero-allocation invariants with `test_allocator` unit tests (`--test-threads=1`).
4. **Shell Crate Isolation**:
   - Keep `hifishifter-plugin` a pure `rlib` with zero `#[no_mangle]` export symbols.
   - Build and test `hifishifter-clap` with `-p hifishifter-clap` separately from `src-tauri` so Cargo feature unification never links `vslib_x64` into the single-file CLAP binary.
   - Test-only helpers shared with shells (`test_host`, `test_allocator`) must use `#[cfg(any(test, feature = "test-support"))]`.
5. **Run the Gate**:
   - Execute `python3 scripts/checks/check-ffi-guards.py` (must report all extern entries guarded across `hifishifter-plugin`, `hifishifter-vst3`, and `hifishifter-clap`, and `0` baseline violations).
