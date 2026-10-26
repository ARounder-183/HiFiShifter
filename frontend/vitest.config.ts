/**
 * Vitest 中央配置。
 *
 * 【为什么需要这个文件】此前测试配置完全依赖 vitest 的隐式默认值 + 各文件
 * 自行 pragma：没有一处能声明"本仓的测试环境策略"。84 个测试文件靠文件头
 * 的 `// @vitest-environment jsdom` pragma 声明 DOM 需求 —— 这个机制本身
 * 是对的（按文件最小化 DOM 依赖），要保留；中央配置负责把**默认环境**钉在
 * node 上，避免有人误以为默认是 jsdom 而漏写 pragma（node 下 DOM 测试会以
 * `document is not defined` 崩掉，方向相反的错误反而难犯）。
 *
 * 【默认 node 的理由】与现状逐字节一致（此前无配置文件时 vitest 的默认即
 * node）。改成全局 jsdom 会让全部文件背上 DOM 依赖与更慢的引导，不在本批
 * 范围。
 *
 * 【为什么 mergeConfig 而不是重写】vite.config.ts 持有扩展别名（@hs/sdk、
 * @hs/ui）与 react 插件；vitest.config.ts 存在时 vitest **不再**读取
 * vite.config.ts，因此必须显式合并进来，否则别名解析与插件链会静默丢失。
 * vite.config.ts 的默认导出是 `defineConfig(({ mode }) => …)` 函数形态，
 * 这里以 serve/test 模式先求值再合并。
 *
 * 【刻意不开的开关】`isolate: false`（约 15s 提速）与 `fsModuleCache` 需要
 * 逐一验证隔离正确性（模块级可变状态、sinon 恢复、诊断计数器），留待后续
 * 专项（见修复方案 T4-6），本文件先只做零行为变化的集中声明。
 */
import { defineConfig, mergeConfig, type ViteUserConfig } from "vitest/config";
import viteConfig from "./vite.config.ts";

// vite.config.ts 的默认导出是 `defineConfig(({ mode }) => …)` 的函数形态；
// 以 serve/test 模式求值（测试不关心 rollup 入口，排除 waveform-test 无副作用）。
// await 兼顾函数可能返回 Promise 的类型面（实际为同步）。
const resolvedViteConfig: ViteUserConfig =
    typeof viteConfig === "function"
        ? await viteConfig({ command: "serve", mode: "test" })
        : viteConfig;

export default mergeConfig(
    resolvedViteConfig,
    defineConfig({
        test: {
            // 默认 node；需要 DOM 的测试文件在文件头写：
            //   // @vitest-environment jsdom
            environment: "node",
            setupFiles: ["./src/test/setup.ts"],
        },
    }),
);
