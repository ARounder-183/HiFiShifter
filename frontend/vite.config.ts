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
            },
        },
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
