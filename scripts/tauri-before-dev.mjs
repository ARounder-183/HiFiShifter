/**
 * 在 Tauri dev 启动前根据模式启动前端：
 * - TAURI_UI_MODE=dev   -> Vite 开发服务器（默认）
 * - TAURI_UI_MODE=build -> 先构建再用 Vite preview 提供静态资源
 */

import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(__dirname, "../frontend");
const mode = (process.env.TAURI_UI_MODE || "build").toLowerCase();

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
