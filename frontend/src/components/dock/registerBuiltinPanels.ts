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
import { NOTEBOOK_PANEL_MIN_WIDTH } from "../layout/notebook/notebookSettings";

/** 面板 id 常量（同时是持久化 JSON 里的键，一旦发布不可更改）。 */
export const PANEL_TIMELINE = MAIN_FORM_TIMELINE;
export const PANEL_PARAM_EDITOR = MAIN_FORM_PARAM_EDITOR;
export const PANEL_FILE_BROWSER = "fileBrowser";
export const PANEL_NOTEBOOK = "notebook";
export const PANEL_UNDO_HISTORY = "undoHistory";
export const PANEL_APPEARANCE = "appearance";

/**
 * 记事本默认浮窗尺寸。
 *
 * 撤销历史的默认落点由它推导（落在记事本左侧），因此抽成常量：改尺寸不会让两个
 * 默认浮窗失配重叠。
 */
const NOTEBOOK_FLOAT_WIDTH = 460;
const NOTEBOOK_FLOAT_HEIGHT = 420;

/** 默认浮窗距视口边缘的边距（与 `openAsFloating.marginPx` 的默认值一致）。 */
const DEFAULT_FLOAT_MARGIN_PX = 24;

/**
 * 注册全部内置面板。
 *
 * 【为什么没有"只注册一次"的私有标志】此前有一个模块级 `registered` 布尔量做早退，
 * 而 `resetPanelRegistryForTests()` 只清注册表、清不掉那个布尔量 —— 于是"先 reset
 * 再注册"的测试会静默地什么都没注册（表现为断言拿到空列表）。那是个陷阱。
 *
 * 现在靠 `registerPanel` 自身的幂等（同 id 覆盖）实现重放：生产路径上本函数在
 * `App.tsx` / `detachedMain.tsx` 的模块作用域各调用一次，重放代价为零。
 */
export function registerBuiltinPanels(): void {
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
        // `fb_title` 是文件浏览器自己的标题键（`panel_io` 是工具栏分组名"I/O"，
        // 用它会让标签条显示成"输入输出"，与面板实际内容不符）。
        titleKey: "fb_title",
        singleton: true,
        defaultWidth: 320,
        defaultHeight: 480,
        minWidth: 220,
        // 默认贴工作区右边缘、固定 360px：与重构前"右侧栏"的视觉位置一致，
        // 但宽度由用户拖定，且不再与记事本并排挤占空间。
        defaultPlacement: { side: "right", sizePx: 360 },
        // 纯 DOM + Redux 面板：重挂载代价低，允许拆到独立窗口（见 detachable 说明）。
        detachable: true,
        order: 30,
    });

    registerPanel({
        id: PANEL_NOTEBOOK,
        titleKey: "common_notebook",
        singleton: true,
        defaultWidth: 420,
        defaultHeight: 480,
        minWidth: NOTEBOOK_PANEL_MIN_WIDTH,
        // 记事本是"随手记"性质的辅助面板：**默认保持关闭**（启动时不该自己冒出来），
        // 用户打开它时希望它浮在手边、而不是挤进布局占一格 —— 因此落在右下角。
        // 声明了 `openAsFloating` 之后 `defaultPlacement` 不会再被用到，故不再声明。
        openAsFloating: {
            width: NOTEBOOK_FLOAT_WIDTH,
            height: NOTEBOOK_FLOAT_HEIGHT,
            anchor: "bottom-right",
        },
        detachable: true,
        order: 40,
    });

    registerPanel({
        id: PANEL_UNDO_HISTORY,
        // 复用既有键：撤销历史面板的标题早就有 i18n 词条，另起一个 `undo_history`
        // 会得到一个查不到的键（表现为菜单里显示 "undefined"）。
        titleKey: "undo_history_title",
        singleton: true,
        defaultWidth: 420,
        defaultHeight: 420,
        minWidth: 260,
        // 与记事本同样的处理：辅助面板，**默认保持关闭**，打开时浮在手边而不是挤进
        // 布局占一格（此前它与文件浏览器并排停靠在右侧栏）。
        // 落点与记事本错开：贴它的**左侧**、底边对齐、间隔一个默认边距 —— 两个默认
        // 浮出的面板同时打开时不会叠在一起。偏移量由记事本的宽度推导（自身已距边缘
        // 一个边距，因此只需再让出"记事本宽 + 一个边距"），改尺寸不会失配。
        openAsFloating: {
            width: 420,
            height: 420,
            anchor: "bottom-right",
            offsetX: -(NOTEBOOK_FLOAT_WIDTH + DEFAULT_FLOAT_MARGIN_PX),
        },
        detachable: true,
        order: 50,
    });

    registerPanel({
        id: PANEL_APPEARANCE,
        // 复用既有菜单词条（「视图 → 外观设置」）。另起 `appearance` 会得到一个
        // 查不到的键 —— 面板标题因此会显示成键名本身（`notebook` 就栽在这上面）。
        titleKey: "menu_appearance_settings",
        singleton: true,
        defaultWidth: 900,
        defaultHeight: 640,
        minWidth: 560,
        minHeight: 420,
        /*
         * 居中浮出，而不是落在某个角。
         *
         * 右下角锚点是给"随手记"性质的辅助面板用的（记事本、撤销历史）—— 它们
         * 出现在手边即可。外观设置不是那种面板：用户打开它是要**专心改一轮**，
         * 接下来一段时间它都是主焦点，正中最合适，也不会盖住时间轴的轨道头。
         */
        openAsFloating: { width: 900, height: 640, anchor: "center" },
        /*
         * 不进「视图 → 窗口」：那个菜单列的是日常切换的工作面板，低频设置入口
         * 混进去只会稀释常用项。入口留在「视图 → 外观设置」（`openPanel`）。
         */
        excludeFromWindowMenu: true,
        /*
         * 不可停靠：拖得动，但停不进去。把设置界面编入工作布局占一格既没有意义
         * （改完就走），也会污染用户精心排好的布局。
         */
        dockable: false,
        // 已经不需要"拆到独立窗口"：它本来就是主窗口里的浮窗，而独立窗口是另一个
        // JS 上下文、必须重新挂载面板 —— 对一个设置界面没有收益。
        detachable: false,
        // 排在最后：它是低频入口，顺序上也不该插进工作面板之间。
        order: 90,
    });
}

/** 面板是否已注册（供菜单构建时跳过未注册项）。 */
export function hasPanel(panelId: string): boolean {
    return getPanel(panelId) !== undefined;
}
