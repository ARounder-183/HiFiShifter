import { describe, expect, it } from "vitest";

// 直接读后端源码文本（`vite/client` 为 `?raw` 提供类型声明，无需 node 类型，
// 与 historyOpLabels.test.ts 的做法一致）。这样后端增删命令时本测试自动跟随，
// 不需要手工维护一份副本。
import backendLibSource from "../../../backend/src-tauri/src/lib.rs?raw";
import invokeSource from "./invoke.ts?raw";

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
 * 本测试做两件事：
 * 1. 扫描前端源码中的全部 `invoke(...)` 调用点，断言每个被调用的命令都已登记
 *   （映射 case 或无参白名单）——新增命令漏登记必红；
 * 2. 断言前端引用的每个命令名（映射 case ∪ 无参白名单 ∪ 调用点）都存在于
 *    后端 `generate_handler!` 注册表 —— 删除后端命令而前端还留着映射时必红
 *   （get_waveform_manifest / get_waveform_tiles_binary 就这样失效过：调用只
 *    会得到后端"未知命令"错误，前端没有任何静态检查能发现）。
 */

/** 从 invoke.ts 源码提取 buildTauriArgs 的全部 `case "cmd"` 标签。 */
function extractSwitchCases(source: string): string[] {
    const commands = new Set<string>();
    for (const match of source.matchAll(/case\s+"([a-z_0-9]+)"/g)) {
        commands.add(match[1]);
    }
    return [...commands];
}

/** 从 invoke.ts 源码提取 NO_ARG_COMMANDS 白名单的全部条目。 */
function extractNoArgCommands(source: string): string[] {
    const block = source.match(/NO_ARG_COMMANDS[\s\S]*?new Set\(\[([\s\S]*?)\]\)/);
    const commands = new Set<string>();
    if (block) {
        for (const match of block[1].matchAll(/"([a-z_0-9]+)"/g)) {
            commands.add(match[1]);
        }
    }
    return [...commands];
}

/** 从后端 lib.rs 的 generate_handler! 块提取全部注册命令。 */
function extractBackendHandlers(source: string): string[] {
    const block = source.match(/generate_handler!\[([\s\S]*?)\]/);
    const commands = new Set<string>();
    if (block) {
        for (const match of block[1].matchAll(/commands::([a-z_0-9]+)/g)) {
            commands.add(match[1]);
        }
    }
    return [...commands];
}

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
        expect(
            buildTauriArgs("set_clip_take_channel_mode", ["clip-1", "take-2", 3, false]),
        ).toEqual({ clipId: "clip-1", takeId: "take-2", channelMode: 3, checkpoint: false });
    });

    it("extracts a plausible command set from both sides", () => {
        // 防"正则失效导致本测试静默通过"：两侧都必须提取到足够多的命令。
        expect(extractSwitchCases(invokeSource).length).toBeGreaterThan(100);
        expect(extractNoArgCommands(invokeSource).length).toBeGreaterThan(40);
        expect(extractBackendHandlers(backendLibSource).length).toBeGreaterThan(150);
        // 抽查两侧的锚点命令，正则失配时立即暴露。
        expect(extractSwitchCases(invokeSource)).toContain("set_clip_take_channel_mode");
        expect(extractNoArgCommands(invokeSource)).toContain("get_ui_settings");
        expect(extractBackendHandlers(backendLibSource)).toContain("save_ui_settings");
    });

    it("every frontend-referenced command exists in the backend handler list", () => {
        const sources = import.meta.glob("/src/**/*.{ts,tsx}", {
            query: "?raw",
            import: "default",
            eager: true,
        }) as Record<string, string>;
        const callPattern = /invoke\s*<[^>]*>?\s*\(\s*["'`]([a-z_0-9]+)["'`]/g;
        const invoked = new Set<string>();
        for (const [file, text] of Object.entries(sources)) {
            if (/\.test\.(ts|tsx)$/.test(file)) continue;
            for (const match of text.matchAll(callPattern)) {
                invoked.add(match[1]);
            }
        }

        const backend = new Set(extractBackendHandlers(backendLibSource));
        const referenced = new Set<string>([
            ...invoked,
            ...extractSwitchCases(invokeSource),
            ...extractNoArgCommands(invokeSource),
        ]);
        // 报告全部失效项而不是第一个，便于一次清理。
        const dead = [...referenced].filter((cmd) => !backend.has(cmd)).sort();
        expect(dead).toEqual([]);
    });
});
