# Plugin & Host ABI Guide (`hifishifter-plugin`, `hifishifter-vst3`, `hifishifter-clap`)

Read root [`../../AGENTS.md`](../../AGENTS.md), [`../../docs/thread-model.md`](../../docs/thread-model.md), and [`../../docs/host-extension-boundary.md`](../../docs/host-extension-boundary.md) before modifying plugin crates.

## Crate Split

- `hifishifter-plugin` is a **format-neutral `rlib`**. Never add VST3 or CLAP module export symbols (`GetPluginFactory`, `clap_entry`) here — otherwise every format shell exports foreign ABI symbols.
- `hifishifter-vst3` owns the VST3 + ARA2 `cdylib` shell and requires `ARA_SDK_DIR` / `ARA_VST3_SDK_DIR` (`scripts/env/sdk-env.ps1` or `scripts/vst3/unix/build.sh --stage=sdk`).
- `hifishifter-clap` owns the CLAP `cdylib` shell (pure Rust `clap` + bare `ara2-bridge` companion). Build and test it with `-p hifishifter-clap` **outside** any `--workspace` command that includes `HiFiShifter` (`src-tauri`), otherwise Cargo feature unification leaks `vslib_x64` into CLAP's import table and breaks `clap_exports`.

## Real-Time & FFI Rules

1. **Audio Thread (`audio_block.rs` / `vst3.rs::audio_process` / CLAP `process`)**:
   - Zero heap allocations (`test_allocator` enforces this), zero mutex waits, zero file I/O, zero logging.
   - Always wrap in `catch_unwind` degrading to finite silence on panic.
2. **Host Main/UI Thread Callbacks**:
   - Every `extern "system"` / `extern "C"` / `extern "C-unwind"` function must wrap its body in `guard_ffi` or `boundary` (`python3 scripts/checks/check-ffi-guards.py` audits all plugin crates).
   - Always acquire mutexes via `MutexExt::lock_or_recover()`.
3. **Shared Test Fixtures (`test-support` feature)**:
   - `test_host`, `test_allocator`, and `editor::session::tests` helpers exposed to shell crates must be gated with `#[cfg(any(test, feature = "test-support"))]`. Never enable `test-support` in non-dev `[dependencies]`.
4. **Plugin Test Execution**:
   - Run plugin tests with `--test-threads=1` and per-module chunks (`ara::`, `capture::`, `editor::`, `host::`, `render::`) as defined in `scripts/checks/run-rust-gates.ps1` to avoid process-global cache contention.
