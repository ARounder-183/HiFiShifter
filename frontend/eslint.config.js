import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier";
import { defineConfig, globalIgnores } from "eslint/config";

export default defineConfig([
    globalIgnores(["dist"]),
    {
        files: ["**/*.{ts,tsx}"],
        extends: [
            js.configs.recommended,
            tseslint.configs.recommended,
            reactHooks.configs.flat.recommended,
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
            // React Compiler 系规则对"渲染期直接同步的 ref 镜像"（pxPerSecRef、
            // paramEditorSyncTimelineRef 等，见 useTimelineState 425 行注释）会整体
            // 报错。这是本代码库刻意的取值新鲜度模式：读取方都在事件回调里，渲染期
            // 同步保证 emit 拿到的是本次提交的最新值；改成 effect 镜像会滞后一帧
            // 提交。降为 warn 保留可见性，不作为 CI 阻塞。
            "react-hooks/refs": "warn",
            "react-hooks/immutability": "warn",
            // react-hooks 7.1 把下面两条提升为 recommended 里的 error：
            //   set-state-in-effect          命中"对话框打开时从 props/ref 初始化
            //                                本地状态"这一惯例（35 处，如
            //                                ExportAudioDialog 的 customEndSec、
            //                                AutoBackupDialog 的 notice 复位）。
            //                                这是常见且非错误的写法，改成 key /
            //                                派生状态是一次独立的 UI 重构。
            //   preserve-manual-memoization  React Compiler 无法保留既有手动 memo
            //                                的诊断（13 处），同样需要专门的清理轮次。
            // 两者降为 warn：信号留在 lint 输出里可见，但不阻塞 CI。
            "react-hooks/set-state-in-effect": "warn",
            "react-hooks/preserve-manual-memoization": "warn",
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
    eslintConfigPrettier,
]);
