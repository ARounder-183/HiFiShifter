# Claude Code Instructions

Read and follow [`AGENTS.md`](AGENTS.md) as the single authoritative engineering guide for HiFiShifter.

Before committing or finishing any task:
1. Run `python3 scripts/checks/run-fast-gates.py` (all 8 static gates must pass; GitHub Actions CI only builds artifacts and runs zero tests).
2. Run `python3 scripts/checks/verify-changed.py --run` (or targeted crate/frontend tests) to verify every touched subsystem.
3. Consult subdirectory guides (`backend/hifishifter-plugin/AGENTS.md`, `frontend/AGENTS.md`) and `.github/skills/hfs-*` when editing IPC commands, coordinate conversions, FFI boundaries, UI tokens/i18n, or MCP contracts.
