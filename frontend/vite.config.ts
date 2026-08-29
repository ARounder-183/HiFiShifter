/**
 * Vite 构建配置
 *
 * 支持多入口（多窗口 Tauri 应用）：
 * - index.html → 主窗口
 * - appearance.html → 外观设置独立窗口
 */

import { resolve } from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
    base: "./",
    plugins: [react()],
    build: {
        rollupOptions: {
            input: {
                main: resolve(__dirname, "index.html"),
                appearance: resolve(__dirname, "appearance.html"),
                waveformTest: resolve(__dirname, "waveform-test.html"),
            },
            output: {
                // 按库分组 vendor chunk：稳定缓存 + 避免超大主 chunk
                manualChunks(id: string) {
                    if (!id.includes("node_modules")) {
                        return undefined;
                    }
                    if (id.includes("@radix-ui/react-icons")) {
                        return "icons";
                    }
                    if (id.includes("@radix-ui")) {
                        return "radix-ui";
                    }
                    if (id.includes("@tauri-apps")) {
                        return "tauri";
                    }
                    if (
                        id.includes("node_modules/react/") ||
                        id.includes("node_modules/react-dom/") ||
                        id.includes("node_modules/scheduler/")
                    ) {
                        return "react";
                    }
                    if (
                        id.includes("@reduxjs") ||
                        id.includes("react-redux") ||
                        id.includes("/redux/") ||
                        id.includes("/immer/") ||
                        id.includes("use-sync-external-store")
                    ) {
                        return "redux";
                    }
                    return "vendor";
                },
            },
        },
        // Tauri 桌面应用：资源从本地磁盘加载，主窗口 chunk 体积对首屏无网络成本。
        // vendor 已按库拆分（react/redux/radix-ui/icons/tauri）；主 chunk 为应用代码本身。
        chunkSizeWarningLimit: 1000,
    },
    server: {
        port: 5173,
        strictPort: true,
        watch: {
            // 避免监听过多文件
            ignored: ["**/node_modules/**", "**/dist/**"],
        },
    },
    // 开发模式下清除缓存，确保总是重新构建
    clearScreen: false,
});
