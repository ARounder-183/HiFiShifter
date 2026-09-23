/*
 * 记事本的错误边界。
 *
 * 【为什么必须有】应用没有任何 ErrorBoundary：渲染期或 effect 体里抛出的异常
 * 会**卸载整个 React 根**，用户看到的是空白窗口，只能强杀进程。记事本是个
 * 重面板（编辑器内核 + 图片解码 + IPC），把它隔离在一个边界里，最坏情况只是
 * 这个面板坏掉，时间轴和参数编辑器照常可用。
 *
 * 边界还负责两件事：
 * - 把异常上报到前端错误日志（与全局兜底同一入口）；
 * - 提供"以 Markdown 源码模式打开"的退路 —— 富文本视图出问题时，源码视图
 *   仍能让人看到并取回自己的笔记。
 */

import { Component, type ErrorInfo, type ReactNode } from "react";

import { useAppDispatch } from "../../../app/hooks";
import { setNotebookMode } from "../../../features/notebook/notebookSlice";
import { closeFormById } from "../../../features/dock/dockApi";
import { PANEL_NOTEBOOK } from "../../dock/registerBuiltinPanels";
import { store } from "../../../app/store";
import { useI18n } from "../../../i18n/I18nProvider";
import { reportFrontendError } from "../../../services/frontendErrorLog";

interface Props {
    children: ReactNode;
    /** 出错后切到源码模式（由外层注入，边界本身不碰 store）。 */
    onOpenSourceMode: () => void;
    onClose: () => void;
    labels: {
        title: string;
        hint: string;
        source: string;
        close: string;
    };
}

interface State {
    error: Error | null;
}

class NotebookErrorBoundaryInner extends Component<Props, State> {
    state: State = { error: null };

    static getDerivedStateFromError(error: Error): State {
        return { error };
    }

    componentDidCatch(error: Error, info: ErrorInfo): void {
        reportFrontendError("notebook_panel_crashed", {
            message: error.message,
            stack: error.stack,
            componentStack: info.componentStack,
        });
    }

    private handleSourceMode = () => {
        this.setState({ error: null });
        this.props.onOpenSourceMode();
    };

    render(): ReactNode {
        const { error } = this.state;
        if (!error) return this.props.children;

        return (
            <div className="flex h-full min-h-0 flex-col items-center justify-center gap-3 bg-qt-window p-4 text-center">
                <div className="text-xs font-medium text-qt-text">{this.props.labels.title}</div>
                <div className="max-w-[280px] text-[11px] text-qt-text-muted">
                    {this.props.labels.hint}
                </div>
                <div className="max-w-[280px] overflow-hidden text-ellipsis whitespace-nowrap text-[10px] text-qt-text-muted opacity-70">
                    {error.message}
                </div>
                <div className="flex gap-2">
                    <button
                        type="button"
                        className="rounded border border-qt-border px-2 py-1 text-[11px] hover:bg-qt-hover"
                        onClick={this.handleSourceMode}
                    >
                        {this.props.labels.source}
                    </button>
                    <button
                        type="button"
                        className="rounded border border-qt-border px-2 py-1 text-[11px] hover:bg-qt-hover"
                        onClick={this.props.onClose}
                    >
                        {this.props.labels.close}
                    </button>
                </div>
            </div>
        );
    }
}

/** 连接 store / i18n 的外壳（类组件里用不了 hook）。 */
export function NotebookErrorBoundary({ children }: { children: ReactNode }) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    return (
        <NotebookErrorBoundaryInner
            onOpenSourceMode={() => dispatch(setNotebookMode("source"))}
            onClose={() => closeFormById(dispatch, store.getState, PANEL_NOTEBOOK)}
            labels={{
                title: t("notebook_crash_title"),
                hint: t("notebook_crash_hint"),
                source: t("notebook_crash_open_source"),
                close: t("close"),
            }}
        >
            {children}
        </NotebookErrorBoundaryInner>
    );
}
