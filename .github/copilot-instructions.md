---
applyTo: "**"
---

Please follow the authoritative architecture, safety invariants, coordinate-axis rules, and mandatory local gate commands in [`AGENTS.md`](../AGENTS.md).

Key rules summary:
1. **Run local gates before finishing**: GitHub Actions CI only builds artifacts and runs zero tests. Always run `python3 scripts/checks/run-fast-gates.py` and `python3 scripts/checks/verify-changed.py --run`.
2. **Respect crate & host boundaries**: Keep `hifishifter-kernel` free of Tauri/device I/O; keep `hifishifter-plugin` format-neutral (`GetPluginFactory` lives in `hifishifter-vst3`, `clap_entry` in `hifishifter-clap`); gate host features on `HostCapabilities`, never host name strings.
3. **Follow the Three Iron Laws (`docs/thread-model.md`)**: Guard every FFI/sync-IPC boundary (`guard_ffi` / `guard_command`), recover poisoned locks (`lock_or_recover`, never `.lock().unwrap()`), and propagate structured error codes (never `error.contains(...)`).
4. **Frontend discipline (`frontend/AGENTS.md`)**: Keep `ScrollKernel` as the single viewport source of truth, use `@hs/ui` + `var(--qt-*)` tokens, update all 5 locale dictionaries (`en-US`, `zh-CN`, `zh-TW`, `ja-JP`, `ko-KR`), and include concise file-header comments explaining module responsibility.
