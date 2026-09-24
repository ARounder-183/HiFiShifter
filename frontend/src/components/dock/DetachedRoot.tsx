/**
 * 独立窗口的根组件：渲染**一个**被拆出去的面板。
 *
 * 【为什么这里直接按 id 映射组件、而不走 `panelRenderer` 注册表】那个注册表是
 * 模块级单例，由主窗口的 `App` 在渲染期写入；独立窗口是另一个 JS 上下文，注册表
 * 是空的。与其在两边维护同一份注册逻辑，不如在这里显式列出"允许被拆出去的面板"
 * —— 它本来就是一份能力清单（与 `PanelDefinition.detachable` 一一对应），写在
 * 一处反而更清楚。
 *
 * 【为什么先等快照】面板读的是 Redux 状态（工程、剪贴板、文件浏览器目录…）。快照
 * 到达前渲染会读到空状态（例如记事本会以为工程没有笔记），因此先显示一行"正在连接
 * 主窗口"，等状态就位再挂载面板 —— 面板的首次渲染就是完整状态。
 */

import { useCallback } from "react";

import { useAppSelector } from "../../app/hooks";
import { FileBrowserPanel } from "../layout/FileBrowserPanel";
import { NotebookPanel } from "../layout/notebook/NotebookPanel";
import { UndoHistoryPanel } from "../layout/UndoHistoryPanel";
import { PANEL_FILE_BROWSER, PANEL_NOTEBOOK, PANEL_UNDO_HISTORY } from "./registerBuiltinPanels";
import { satelliteFormId } from "../../features/dock/detachBridge";

/** 独立窗口承载的面板组件映射（与 `PanelDefinition.detachable` 对应）。 */
function renderDetachedPanel(panelId: string) {
    switch (panelId) {
        case PANEL_FILE_BROWSER:
            return <FileBrowserPanel />;
        case PANEL_NOTEBOOK:
            return <NotebookPanel />;
        case PANEL_UNDO_HISTORY:
            return <UndoHistoryPanel />;
        default:
            return null;
    }
}

export function DetachedRoot() {
    const formId = satelliteFormId();
    const form = useAppSelector((state) =>
        formId ? (state.dock.layout.forms[formId] ?? null) : null,
    );
    /**
     * 快照到达的判定。
     *
     * 【为什么用"布局里有这个窗体"当判据】快照应用后 `dock.layout` 会被整份替换，
     * 其中必然包含正在被拆出的这个窗体（它就是主窗口里那个 floating 窗体）。这比
     * 监听一个专门的"已连接"信号更省事，也不会漏 —— 拿不到布局说明快照还没到。
     *
     * 直接由状态推导（不额外存一份"已连接"布尔量）：多一份状态就多一次渲染，
     * 而且那份状态只能在 effect 里同步，属典型的级联渲染来源。
     */
    const panelId = form?.panelId ?? null;
    const content = panelId ? renderDetachedPanel(panelId) : null;

    const onDragOver = useCallback((event: React.DragEvent) => {
        // 独立窗口不做跨窗口拖放（宿主窗口不同，拖拽数据无法贯通）：显式阻止默认
        // 行为，避免 WebView 把文件拖进来时整页导航。
        event.preventDefault();
    }, []);

    return (
        <div
            className="h-screen w-screen overflow-hidden bg-qt-window text-qt-text"
            onDragOver={onDragOver}
            data-detached-form={formId ?? undefined}
        >
            {content ?? (
                <div className="flex h-full w-full items-center justify-center text-xs text-qt-text-muted">
                    …
                </div>
            )}
        </div>
    );
}
