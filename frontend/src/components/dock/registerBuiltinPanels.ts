/*
 * 内置面板的注册。
 *
 * 【为什么内置面板也走注册表】如果停靠内核里硬编码"时间轴、参数编辑器、文件
 * 浏览器、记事本"，那么将来开放自定义面板 API 就必然要重写内核。让内置面板
 * 与第三方面板走完全相同的注册路径，抽象就不会被架空。
 *
 * 这里只登记**元信息**（标题、尺寸约束、默认落点）；面板组件本身由 `App.tsx`
 * 通过 `setPanelRenderer` 提供，因为它需要大量只有 App 才持有的 props。
 */

import { getPanel, registerPanel } from "../../features/dock/panelRegistry";
import { MAIN_FORM_PARAM_EDITOR, MAIN_FORM_TIMELINE } from "../../features/dock/dockSchema";
import {
    NOTEBOOK_PANEL_MAX_WIDTH,
    NOTEBOOK_PANEL_MIN_WIDTH,
} from "../layout/notebook/notebookSettings";

/** 面板 id 常量（同时是持久化 JSON 里的键，一旦发布不可更改）。 */
export const PANEL_TIMELINE = MAIN_FORM_TIMELINE;
export const PANEL_PARAM_EDITOR = MAIN_FORM_PARAM_EDITOR;
export const PANEL_FILE_BROWSER = "fileBrowser";
export const PANEL_NOTEBOOK = "notebook";
export const PANEL_UNDO_HISTORY = "undoHistory";

let registered = false;

/** 注册全部内置面板。幂等：重复调用只是重放注册。 */
export function registerBuiltinPanels(): void {
    if (registered) return;
    registered = true;

    registerPanel({
        id: PANEL_TIMELINE,
        titleKey: "panel_timeline",
        // 主编辑区面板：不允许被普通面板挤出去，最小高度沿用重构前的 200px。
        preferMain: true,
        singleton: true,
        defaultWidth: 1000,
        defaultHeight: 420,
        minHeight: 200,
        minWidth: 320,
        order: 10,
    });

    registerPanel({
        id: PANEL_PARAM_EDITOR,
        titleKey: "panel_editor",
        preferMain: true,
        // 参数编辑器天然支持多实例：同时看不同片段的音高/音量曲线是常见需求。
        singleton: false,
        defaultWidth: 1000,
        defaultHeight: 320,
        minHeight: 150,
        minWidth: 320,
        order: 20,
    });

    registerPanel({
        id: PANEL_FILE_BROWSER,
        titleKey: "panel_io",
        singleton: true,
        defaultWidth: 320,
        defaultHeight: 480,
        minWidth: 220,
        // 默认贴工作区右边缘、固定 360px：与重构前"右侧栏"的视觉位置一致，
        // 但宽度由用户拖定，且不再与记事本并排挤占空间。
        defaultPlacement: { side: "right", sizePx: 360 },
        order: 30,
    });

    registerPanel({
        id: PANEL_NOTEBOOK,
        titleKey: "notebook",
        singleton: true,
        defaultWidth: 420,
        defaultHeight: 480,
        minWidth: NOTEBOOK_PANEL_MIN_WIDTH,
        // 记事本与文件浏览器同处右侧停靠栏，以标签页共存 —— 打开两者时宽度
        // 不再翻倍，这是"界面拥挤"最直接的解法。
        defaultPlacement: {
            side: "right",
            sizePx: Math.max(360, NOTEBOOK_PANEL_MAX_WIDTH / 2),
            tabWith: PANEL_FILE_BROWSER,
        },
        order: 40,
    });

    registerPanel({
        id: PANEL_UNDO_HISTORY,
        titleKey: "undo_history",
        singleton: true,
        defaultWidth: 420,
        defaultHeight: 420,
        minWidth: 260,
        defaultPlacement: { side: "right", sizePx: 380, tabWith: PANEL_FILE_BROWSER },
        order: 50,
    });
}

/** 面板是否已注册（供菜单构建时跳过未注册项）。 */
export function hasPanel(panelId: string): boolean {
    return getPanel(panelId) !== undefined;
}
