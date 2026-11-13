/**
 * 把自包含内置面板的**组件实现**装配到注册中心。
 *
 * 【为什么与 `registerBuiltinPanels` 分开】那个模块只放元数据（id / 标题键 /
 * 默认尺寸 / 落点），因此可以被 `dockSchema` 依赖而不拖入组件依赖链 ——
 * 布局函数的单测在 node 环境里跑，一旦经由此处 import 面板组件，
 * `localStorage` 之类的 DOM API 就会让纯布局测试崩掉。
 *
 * 本模块只在**渲染入口**调用：主窗口 `App.tsx` 与独立窗口 `detachedMain.tsx`。
 * 独立窗口是另一个 JS 上下文，必须自己装配一次。
 *
 * 文件浏览器 / 记事本 / 撤销历史是自包含的；参数编辑器额外有一个只接
 * `formId` 的卫星包装（`PianoRollPanel` 的状态来自 Redux）。时间轴需要
 * `App.tsx` 提供的约 40 个 props，仍只能走 `setPanelRenderer`。
 */
import { createElement, lazy } from "react";

import { setPanelComponent } from "../../features/dock/panelRegistry";
import { FileBrowserPanel } from "../layout/FileBrowserPanel";
import { UndoHistoryPanel } from "../layout/UndoHistoryPanel";

/*
 * 记事本**必须**懒加载：TipTap + ProseMirror + Turndown 加起来几百 KB，
 * 静态 import 会全部进首屏主包（`App.tsx` 也是为此才用 `lazy`）。
 * 注册中心与渲染侧都支持 lazy 组件，因此这里保持同样的策略。
 */
const NotebookPanel = lazy(() =>
    import("../layout/notebook/NotebookPanel").then((module) => ({
        default: module.NotebookPanel,
    })),
);
import {
    PANEL_FILE_BROWSER,
    PANEL_NOTEBOOK,
    PANEL_PARAM_EDITOR,
    PANEL_UNDO_HISTORY,
} from "./registerBuiltinPanels";

const PianoRollPanel = lazy(() =>
    import("../layout/PianoRollPanel").then((module) => ({ default: module.PianoRollPanel })),
);

/** 卫星窗口里的参数编辑器：只接 formId，与主窗口的 renderer 保持同一组件。 */
function DetachedPianoRollPanel({ formId }: { formId: string }) {
    return createElement(PianoRollPanel, { dockFormId: formId });
}

let attached = false;

/** 幂等：重复调用只生效一次（两个入口都可能调到）。 */
export function attachBuiltinPanelComponents(): void {
    if (attached) return;
    attached = true;
    setPanelComponent(PANEL_FILE_BROWSER, FileBrowserPanel);
    setPanelComponent(PANEL_PARAM_EDITOR, DetachedPianoRollPanel);
    setPanelComponent(PANEL_NOTEBOOK, NotebookPanel);
    setPanelComponent(PANEL_UNDO_HISTORY, UndoHistoryPanel);
}

/** 仅供测试：重置幂等标记。 */
export function resetBuiltinPanelComponentsForTests(): void {
    attached = false;
}
