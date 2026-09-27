/**
 * 根级错误边界 —— 兜住面板子树之外抛出的异常。
 *
 * 【为什么必须有】应用此前**没有任何根级 ErrorBoundary**（`App.tsx` 与
 * `NotebookErrorBoundary.tsx` 的注释都各自承认这一点）。停靠层给每个面板包了
 * `PanelErrorBoundary`，因此单个面板崩溃不会拖垮别人 —— 但**面板之外的**渲染期
 * 异常（构建菜单项时、停靠布局归一化时、任何贡献点回调里）会一路冒到 React 根，
 * 卸载整棵树，留下一片空白窗口。用户看到的只有"应用打不开"，没有任何线索。
 *
 * 【为什么放在 Provider 之内、App 之外】边界要能用到主题与语言（界面文案、
 * 配色），因此必须在 `I18nProvider` / `AppThemeProvider` 内侧；但要包住 `App`
 * 整体，因此在外侧。
 *
 * 【为什么不用类组件的 getDerivedStateFromError 就够】React 只提供类组件的
 * `componentDidCatch` / `getDerivedStateFromError`，函数组件无法捕获子树的
 * 渲染异常，因此这里必须是类。
 */
import { Component, type ErrorInfo, type ReactNode } from "react";

import { reportFrontendError } from "../services/frontendErrorLog";

interface AppRootErrorBoundaryProps {
    children: ReactNode;
}

interface AppRootErrorBoundaryState {
    error: Error | null;
}

export class AppRootErrorBoundary extends Component<
    AppRootErrorBoundaryProps,
    AppRootErrorBoundaryState
> {
    state: AppRootErrorBoundaryState = { error: null };

    static getDerivedStateFromError(error: Error): AppRootErrorBoundaryState {
        return { error };
    }

    componentDidCatch(error: Error, info: ErrorInfo): void {
        // 与 `installGlobalErrorReporting` 走同一条上报链路，因此崩溃会进后端日志，
        // 而不是只留在开发者控制台里。
        reportFrontendError(
            `[root] uncaught render error: ${error.message}`,
            `${error.stack ?? ""}\n\ncomponentStack:${info.componentStack ?? ""}`,
        );
    }

    render(): ReactNode {
        const { error } = this.state;
        if (!error) return this.props.children;

        return (
            <div
                role="alert"
                className="flex h-screen w-screen flex-col items-center justify-center gap-4 bg-qt-window p-8 text-center text-qt-text"
            >
                <div className="text-sm font-semibold">{error.message || String(error)}</div>
                {error.stack ? (
                    <pre
                        data-hs-selectable="true"
                        className="max-h-64 w-full max-w-2xl overflow-auto rounded border border-qt-border bg-qt-base p-3 text-left text-qt-xs leading-4 text-qt-text-muted"
                    >
                        {error.stack}
                    </pre>
                ) : null}
                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        className="rounded border border-qt-border px-3 py-1 text-xs hover:bg-qt-hover"
                        onClick={() => {
                            void navigator.clipboard?.writeText(
                                `${error.message}\n\n${error.stack ?? ""}`,
                            );
                        }}
                    >
                        Copy details
                    </button>
                    <button
                        type="button"
                        className="rounded border border-qt-border px-3 py-1 text-xs hover:bg-qt-hover"
                        // 重新加载是这里唯一能真正恢复的手段：出错的组件树无法原地修复，
                        // 而项目数据在后端 / Redux 之外，重载不会丢。
                        onClick={() => window.location.reload()}
                    >
                        Reload
                    </button>
                </div>
            </div>
        );
    }
}
