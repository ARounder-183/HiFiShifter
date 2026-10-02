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
        },
    },
    eslintConfigPrettier,
]);
