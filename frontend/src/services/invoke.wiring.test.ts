import { describe, expect, it } from "vitest";

import { buildTauriArgs } from "./invoke";

/**
 * IPC 布线穷举防回归测试。
 *
 * 【为什么需要】`buildTauriArgs` 是手写的"位置参数 → Tauri 命名参数"映射表，
 * 新增带参命令若忘记登记，invoke 会直接抛 "method not wired yet" —— 而前端
 * 的乐观更新（pending）此时已经生效且 rejected 不回滚，结果是"波形/UI 变了、
 * 后端从未收到调用"的静默状态分叉。这个陷阱已发生过三次（take 命令族、
 * set_clip_take_reversed、set_clip_take_channel_mode）。
 *
 * 本测试扫描前端源码中的全部 `invoke(...)` 调用点，断言每个被调用的命令
 * 都已登记（映射 case 或无参白名单）——新增命令漏登记必红。
 */

describe("invoke wiring", () => {
    it("every invoke call site is wired in buildTauriArgs", () => {
        // 用 vite 的 raw glob 拉全部源码文本（无需 node fs 类型，浏览器环境兼容；
        // 排除本测试与其它测试文件、dev mock）。
        const sources = import.meta.glob("/src/**/*.{ts,tsx}", {
            query: "?raw",
            import: "default",
            eager: true,
        }) as Record<string, string>;

        // 提取 invoke / invoke<T,...>( "cmd" 调用的命令名（跨行 + 泛型兼容）。
        const callPattern = /invoke\s*<[^>]*>?\s*\(\s*["'`]([a-z_0-9]+)["'`]/g;
        const invoked = new Set<string>();
        const fileCount = Object.keys(sources).length;
        for (const [file, text] of Object.entries(sources)) {
            if (/\.test\.(ts|tsx)$/.test(file)) continue;
            for (const match of text.matchAll(callPattern)) {
                invoked.add(match[1]);
            }
        }
        expect(fileCount).toBeGreaterThan(50);
        expect(invoked.size).toBeGreaterThan(80);

        // 判定：只有 { __unwired } 算未登记。返回 undefined / {} 是合法的
        //（"可选单参"命令在无参调用时返回 undefined；无参命令返回 {}）。
        const unwired = [...invoked].filter((cmd) => {
            const mapped = buildTauriArgs(cmd, []) as
                | Record<string, unknown>
                | undefined
                | { __unwired: true };
            return mapped != null && "__unwired" in mapped;
        });
        expect(unwired).toEqual([]);
    });

    it("registers the take channel mode command with positional names", () => {
        // 本次事故的直接回归用例：声道模式命令漏映射导致"只改波形不改渲染"。
        expect(buildTauriArgs("set_clip_take_channel_mode", ["clip-1", "take-2", 3, false])).toEqual(
            { clipId: "clip-1", takeId: "take-2", channelMode: 3, checkpoint: false },
        );
    });
});
