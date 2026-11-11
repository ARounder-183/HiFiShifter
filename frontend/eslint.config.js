import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier";
import { defineConfig, globalIgnores } from "eslint/config";

/**
 * 本仓没有启用 React Compiler（vite.config.ts / package.json 都没有
 * babel-plugin-react-compiler）。eslint-plugin-react-hooks 7.x 的
 * `flat.recommended` 却把整套编译器规则（refs / immutability /
 * set-state-in-effect / preserve-manual-memoization / purity / ...）都带进来，
 * 对"渲染期同步 ref 镜像""打开对话框时播种草稿"这类本项目刻意的、
 * 编译器之外的模式产生噪声。
 *
 * 因此这里显式只启用经典的两条规则：rules-of-hooks 与 exhaustive-deps。
 * exhaustive-deps 的既有告警已经逐处补齐依赖修到 0；若将来真的接入 React
 * Compiler，必须先把 reactHooks.configs.flat.recommended 加回来并清零其告警
 * （由 tests/reactCompilerGates.test.ts 钉住）。
 */
const classicReactHooks = {
    plugins: { "react-hooks": reactHooks },
    rules: {
        "react-hooks/rules-of-hooks": "error",
        "react-hooks/exhaustive-deps": "error",
    },
};

export default defineConfig([
    globalIgnores(["dist"]),
    {
        files: ["**/*.{ts,tsx}"],
        extends: [
            js.configs.recommended,
            tseslint.configs.recommended,
            classicReactHooks,
            reactRefresh.configs.vite,
        ],
        languageOptions: {
            ecmaVersion: 2020,
            globals: globals.browser,
        },
        rules: {
            "@typescript-eslint/no-unused-vars": [
                "error",
                {
                    argsIgnorePattern: "^_",
                    varsIgnorePattern: "^_",
                    caughtErrorsIgnorePattern: "^_",
                },
            ],
        },
    },
    {
        // 宿主形态与事件路由的唯一出口在 services 层（见 services/hostDetect.ts
        // 与 services/hostEvents.ts 的注释）。此前 14 处监听器各自复制
        // `window.__HFS_PLUGIN_BOOTSTRAP__ ? hostEvents : tauri` 三元式 —— 第 15 个
        // 照抄漏掉守卫就会让插件形态静默收不到事件。这里禁掉 services 之外（与
        // 单测之外）对全局的直接访问，新监听器必须走 hostEvents，形态判定走 hostDetect。
        files: ["**/*.{ts,tsx}"],
        ignores: ["src/services/**", "**/*.test.*", "src/dev/**"],
        rules: {
            "no-restricted-syntax": [
                "error",
                {
                    selector:
                        "MemberExpression[object.name='window'][property.name='__HFS_PLUGIN_BOOTSTRAP__']",
                    message:
                        "宿主形态判定走 services/hostDetect，事件走 services/hostEvents；services 内部除外",
                },
                {
                    selector: "MemberExpression[object.name='window'][property.name='__TAURI__']",
                    message:
                        "独立 App 判定走 services/hostDetect 的 isTauriRuntime()，invoke 走 services/invoke；services 内部除外",
                },
            ],
        },
    },
    {
        // ARA / 插件形态的表面只用本项目的自定义 tooltip 通道（`data-tooltip`，
        // 或最终收敛到它的 `tooltip` prop）。此前这些表面普遍写原生 `title`，
        // 而原生 tooltip 会把多行折成一行、且撞色/不受排版控制
        // （见 `components/dock/dockTooltips.ts:58-61` 的结论）。
        //
        // 【为什么只禁小写标签上的 title】`title` 也有**合法**用途：自定义组件
        // （本项目原语与 Radix 组件）的同名 prop，例如 `AppDialog title=` 是
        // 对话框标题槽，不是悬停提示。而会真的渲染成原生 tooltip 的只有小写
        // 标签（`<span title=` / `<div title=` ...）。规则因此按标签名判定：
        // 正则 `/^[a-z]/` 只命中 DOM 元素，大写开头的组件一律放行 —— 于是
        // 对话框标题槽、`AppStatusChip` 的 title slot、以及 `ui/` 基元内部都不
        // 受影响，而任何新写的 `<xxx title={reason}>` 都会被立刻拦下。
        //
        // 【为什么覆盖这几个目录】它们是用户在宿主与插件里真正会看到的表面
        // （ARA 面板、主菜单栏、停靠栏）。其余目录的原生 title 迁移留到后续
        // 轮次（见 docs/i18n/style-guide.md 的说法），不在本轮扩大范围。
        files: [
            "src/features/ara/**/*.{ts,tsx}",
            "src/components/layout/**/*.{ts,tsx}",
            "src/components/dock/**/*.{ts,tsx}",
        ],
        ignores: ["**/*.test.*"],
        rules: {
            "no-restricted-syntax": [
                "error",
                {
                    selector:
                        "JSXOpeningElement[name.name=/^[a-z]/] > JSXAttribute[name.name='title']",
                    message:
                        "悬停提示请走本项目自定义 tooltip 通道：项目组件用 tooltip prop（AppStatusChip/AppListRow/AppMenuItem），裸 DOM 用 data-tooltip。原生 title 会把多行折成一行、且不受本项目排版与明暗主题控制。",
                },
            ],
        },
    },
    {
        // 根目录的 Node 侧脚本此前不在任何块的 files 里（其余块都是
        // **/*.{ts,tsx}），scripts/*.mjs、eslint.config.js 与 postcss.config.js
        // 等于从不被 lint —— 坏味道只能靠人工发现。补上：
        // - 规则沿用 js.recommended（这些文件没有类型信息，不接 tseslint，
        //   纯 JS 上 TS 规则面只是噪音）；
        // - globals 必须 node+browser 并集：脚本本体在 Node 跑（process /
        //   console），但 playwright 的 page.evaluate 回调字符串在浏览器里
        //   求值，其中合法使用 document / window / rAF —— 只给 node 会让
        //   no-undef 全线报错；
        // - sourceType module（.mjs 与本仓的 ESM 配置文件都是模块语法，
        //   顶层 await 依赖 ecmaVersion 2022+）。
        files: ["scripts/**/*.mjs", "eslint.config.js", "postcss.config.js"],
        extends: [js.configs.recommended],
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: "module",
            globals: { ...globals.node, ...globals.browser },
        },
    },
    eslintConfigPrettier,
]);
