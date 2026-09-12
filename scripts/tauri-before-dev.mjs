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
// 【默认值必须是 dev】文件头注释一直写的是 "dev（默认）"，但实现里曾是 `|| "build"`，
// 两者不一致。后果不只是"多跑一次构建"：`build` 模式经 `npm run build` 提供**生产
// 包**，而当时内核开关的默认值跟随 `import.meta.env.DEV`（生产为 false），于是
// `tauri dev` 跑的是旧渲染实现，新内核在真机上根本没被验证到（Windows 上的卡顿报告
// 因此是旧实现的现象，见 Phase 3 计划的 R8）。
//
// 【现在为什么两件事都要改】内核开关的默认值已改为**与构建模式无关地开启**（见
// `frontend/src/components/layout/timeline/kernel/featureFlag.ts`）。仅改这里不够：
// 用户按 `TAURI_UI_MODE=build` 跑时，旧实现仍会盖过新内核。两处一起改，才能让
// 两种模式跑同一套实现。
//
// 需要生产包时显式设 `TAURI_UI_MODE=build`。
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
