/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    theme: {
        extend: {
            colors: {
                qt: {
                    window: "var(--qt-window)",
                    base: "var(--qt-base)",
                    panel: "var(--qt-panel)",
                    surface: "var(--qt-surface)",
                    text: "var(--qt-text)",
                    "text-muted": "var(--qt-text-muted)",
                    highlight: "var(--qt-highlight)",
                    playhead: "var(--qt-playhead)",
                    button: "var(--qt-button)",
                    "button-hover": "var(--qt-button-hover)",
                    border: "var(--qt-border)",
                    hover: "var(--qt-hover)",
                    accent: "var(--qt-accent)",
                    "danger-bg": "var(--qt-danger-bg)",
                    "danger-text": "var(--qt-danger-text)",
                    "danger-border": "var(--qt-danger-border)",
                    "warning-bg": "var(--qt-warning-bg)",
                    "warning-text": "var(--qt-warning-text)",
                    "warning-border": "var(--qt-warning-border)",
                    "success-bg": "var(--qt-success-bg)",
                    "success-text": "var(--qt-success-text)",
                    "success-border": "var(--qt-success-border)",
                    "info-bg": "var(--qt-info-bg)",
                    "info-text": "var(--qt-info-text)",
                    "info-border": "var(--qt-info-border)",
                    "graph-bg": "var(--qt-graph-bg)",
                    "graph-grid-strong": "var(--qt-graph-grid-strong)",
                    "graph-grid-weak": "var(--qt-graph-grid-weak)",
                    "scrollbar-thumb": "var(--qt-scrollbar-thumb)",
                    "scrollbar-thumb-hover": "var(--qt-scrollbar-thumb-hover)",
                    overlay: "var(--qt-overlay)",
                    "meter-rail": "var(--qt-meter-rail)",
                    "meter-well": "var(--qt-meter-well)",
                    "subtle-1": "var(--qt-subtle-1)",
                    "subtle-2": "var(--qt-subtle-2)",
                    "subtle-3": "var(--qt-subtle-3)",
                    "subtle-hover": "var(--qt-subtle-hover)",
                    divider: "var(--qt-divider)",
                },
            },
            fontFamily: {
                sans: ["var(--qt-font-family)"],
            },
            /*
             * 度量令牌的工具类入口（详见 src/index.css 的「度量令牌」注释块）。
             *
             * `px-qt-*` / `rounded-qt-*` / `h-qt-*` / `text-qt-*` / `z-qt-*`
             * 都取 CSS 变量，使「同一语义只有一处取值来源」。新增样式请优先用
             * 这些，而不是 `h-[26px]` / `z-[9999]` 这类字面量。
             *
             */
            spacing: {
                "qt-0": "var(--qt-space-0)",
                "qt-1": "var(--qt-space-1)",
                "qt-2": "var(--qt-space-2)",
                "qt-3": "var(--qt-space-3)",
                "qt-4": "var(--qt-space-4)",
                "qt-5": "var(--qt-space-5)",
                "qt-6": "var(--qt-space-6)",
                "qt-7": "var(--qt-space-7)",
            },
            height: {
                "qt-ctl-sm": "var(--qt-ctl-sm)",
                "qt-ctl-md": "var(--qt-ctl-md)",
                "qt-ctl-lg": "var(--qt-ctl-lg)",
                "qt-bar-title": "var(--qt-bar-title)",
                "qt-bar-compact": "var(--qt-bar-compact)",
                "qt-bar-main": "var(--qt-bar-main)",
                "qt-bar-status": "var(--qt-bar-status)",
            },
            /*
             * 圆角阶梯全部落到 `--qt-radius-*` —— 它们由外观设置的「圆角风格」推导
             * （见 `src/index.css` 的令牌说明）。
             *
             * 【为什么连 Tailwind 的默认档也改】全仓有 126 处裸 `rounded`，另有
             * sm/md/lg/xl 若干，此前都是写死的 rem 值：于是"圆角风格"只影响 Radix
             * 自己的控件，应用自己的表面（对话框、浮窗、菜单、参数编辑器工具栏的
             * 参数胶囊…）一处都不跟随。把默认档也绑到令牌上，存量调用点**不必逐个
             * 迁移**就一起跟随。
             *
             * `full` 保持 9999px：圆/胶囊是**形状**（头像、圆点、开关轨道），不是
             * 风格偏好 —— 跟着"无圆角"变成方块反而是 bug。
             * `lg` 及以上并到同一档：`--qt-radius-lg` 已是"容器级"圆角，再细分三档
             * 只会让作者重新开始挑数字。
             */
            borderRadius: {
                none: "0px",
                sm: "var(--qt-radius-sm)",
                DEFAULT: "var(--qt-radius-sm)",
                md: "var(--qt-radius-md)",
                lg: "var(--qt-radius-lg)",
                xl: "var(--qt-radius-lg)",
                "2xl": "var(--qt-radius-lg)",
                "3xl": "var(--qt-radius-lg)",
                full: "9999px",
                "qt-sm": "var(--qt-radius-sm)",
                "qt-md": "var(--qt-radius-md)",
                "qt-lg": "var(--qt-radius-lg)",
                "qt-pill": "var(--qt-radius-pill)",
            },
            fontSize: {
                "qt-3xs": "var(--qt-fs-3xs)",
                "qt-micro": "var(--qt-fs-micro)",
                "qt-xs": "var(--qt-fs-xs)",
                "qt-sm": "var(--qt-fs-sm)",
                "qt-md": "var(--qt-fs-md)",
                "qt-lg": "var(--qt-fs-lg)",
                "qt-xl": "var(--qt-fs-xl)",
                "qt-2xl": "var(--qt-fs-2xl)",
            },
            zIndex: {
                "qt-transient": "var(--qt-z-transient)",
                "qt-panel": "var(--qt-z-panel)",
                "qt-popover": "var(--qt-z-popover)",
                "qt-menu": "var(--qt-z-menu)",
                "qt-dialog": "var(--qt-z-dialog)",
                "qt-fullscreen": "var(--qt-z-fullscreen)",
            },
        },
    },
    plugins: [],
};
