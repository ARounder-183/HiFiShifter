# Frontend Engineering Guide (`frontend/`)

Read root [`../AGENTS.md`](../AGENTS.md) and [`../docs/extension-api.md`](../docs/extension-api.md) first.

## Architecture Overview

- **React 19 + Redux Toolkit (`src/features/session/`)** manages application state, history epochs, interaction locks (`_interactionLockCount`), and host capabilities (`App` vs `Plugin` mode).
- **Render Kernels (Not pure React DOM!)**:
  - Timeline (`src/components/layout/timeline/kernel/`) and Piano Roll (`src/components/layout/pianoRoll/`) use custom high-FPS Canvas/WebGL2 render kernels and `ScrollKernel` (`TimelineViewportAccess`).
  - **Viewport Single Source of Truth**: Never re-attach `ref={scrollRef}` onto JSX elements in `src/components/layout/timeline/` (`tests/timelineViewportGates.test.ts` enforces this). Drive programmatic zoom/scroll through `TimelineViewportAccess` (`viewport.setScrollLeft` / `viewport.setZoomAndScroll` or `hifi:focusTimelineRange`).
- **Dual-Backend IPC (`src/services/invoke.ts`)**:
  - Every frontend method in `invoke.ts` must be dispatched by `backend/src-tauri/src/commands.rs` and/or `backend/hifishifter-plugin/src/editor/commands.rs` (`python3 scripts/checks/check-command-gating.py` enforces 100% coverage).

## Mandatory Conventions

1. **Design System & Theme Tokens**:
   - Prefer `@hs/ui` primitives (`AppButton`, `AppIconButton`, `AppDialog`, `AppContextMenu`, `AppField`, `AppStatusChip`, etc.).
   - Never hardcode hex colors or raw pixel theme values; consume `var(--qt-*)` tokens (`QT_COLOR_TOKENS` names are persisted in user theme JSONs and must never be renamed).
2. **5-Locale Closed Dictionary (`src/i18n/`)**:
   - Every new UI message key added to `en-US.ts` **must** be added simultaneously to `zh-CN.ts`, `zh-TW.ts`, `ja-JP.ts`, and `ko-KR.ts` (enforced by TypeScript union types and `i18nCompleteness` tests).
3. **Logging Discipline**:
   - No raw `console.log` / `console.debug` in production source files (`check-log-verbosity.py` and `tests/logVerbosityGates.test.ts`). Use `src/services/frontendLog.ts`.
4. **Node Version & Verification**:
   - Requires **Node >= 22.12** (`util.styleText` for Vite 8 / Vitest 5 / ESLint 10).
   - Run `npm --prefix frontend run typecheck`, `npm --prefix frontend run lint:budget`, `npm --prefix frontend run format:check`, and `npm --prefix frontend run test` (always keep `cwd` at `frontend/` via `npm --prefix frontend`).
