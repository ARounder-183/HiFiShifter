/**
 * Starts the frontend before `tauri dev` according to the mode:
 * - TAURI_UI_MODE=dev   -> Vite dev server (default)
 * - TAURI_UI_MODE=build -> build first, then serve static assets via Vite preview
 */

import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(__dirname, "../../frontend");
// [The default must be dev] the file header always said "dev (default)", but the
// implementation used to be `|| "build"` — the two disagreed. The consequence was not just
// "one extra build": build mode serves a **production bundle** via `npm run build`, and at
// the time the kernel flag's default followed `import.meta.env.DEV` (false in production),
// so `tauri dev` ran the old rendering implementation and the new kernel was never exercised
// on real hardware (the Windows stutter reports were artifacts of the old implementation;
// see the Phase 3 plan's R8).
//
// [Why both places had to change now] the kernel flag's default has been changed to **on
// regardless of build mode** (see
// `frontend/src/components/layout/timeline/kernel/featureFlag.ts`). Changing only this file
// is not enough: when a user runs with `TAURI_UI_MODE=build`, the old implementation would
// still override the new kernel. Changing both together makes both modes run the same
// implementation.
//
// Set `TAURI_UI_MODE=build` explicitly when a production bundle is needed.
const mode = (process.env.TAURI_UI_MODE || "dev").toLowerCase();

function runCommand(command) {
    return new Promise((resolvePromise, rejectPromise) => {
        const child = spawn(command, {
            cwd: frontendDir,
            stdio: "inherit",
            shell: true,
        });

        child.on("exit", (code, signal) => {
            if (signal) {
                rejectPromise(new Error(`Command terminated by signal: ${signal}`));
                return;
            }
            if (code !== 0) {
                rejectPromise(new Error(`Command failed with exit code ${code}`));
                return;
            }
            resolvePromise();
        });

        child.on("error", (err) => rejectPromise(err));
    });
}

async function main() {
    if (mode !== "dev" && mode !== "build") {
        throw new Error(`Unsupported TAURI_UI_MODE: ${mode}. Expected \"dev\" or \"build\".`);
    }

    if (mode === "build") {
        await runCommand("npm run build");
        await runCommand("npm run preview -- --host 127.0.0.1 --port 5173 --strictPort");
        return;
    }

    await runCommand("npm run dev -- --host 127.0.0.1 --port 5173 --strictPort");
}

main().catch((err) => {
    console.error(`[tauri-before-dev] ${err.message}`);
    process.exit(1);
});
