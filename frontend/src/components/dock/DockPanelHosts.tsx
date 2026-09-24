/*
 * 面板宿主层：把每个"已挂载"窗体的 React 子树渲染一次，仅此一次。
 *
 * 这一层在 DOM 上什么都不占 —— 它渲染的是 `createPortal` 到
 * `panelHostRegistry` 创建的宿主 div，那些 div 由停靠布局层搬来搬去。
 * 因此这里只关心"哪些窗体需要存在"，完全不关心它们显示在哪里。
 *
 * 【何时挂载 / 何时保留】窗体一旦出现就长期挂载（关闭只是把它从布局树上摘掉，
 * 宿主退回停泊区）。这样重开面板时滚动位置、缩放、GL 上下文都还在。只有
 * "从未被打开过"的面板不挂载 —— 避免为一个没打开的窗口白建 WebGL 上下文。
 */

import { Component, type ComponentType, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useEffect, useMemo, useSyncExternalStore } from "react";

import { getPanelRenderer } from "../../features/dock/panelRenderer";
import { acquirePanelHost, notifyPanelHosts } from "./panelHostRegistry";
import {
    getPanel,
    getPanelRegistryVersion,
    subscribePanels,
} from "../../features/dock/panelRegistry";
import type { DockPanelProps } from "../../features/dock/panelRegistry";
import type { DockForm } from "../../features/dock/dockTypes";

/** 一个窗体的挂载点。 */
function PanelMount({ form }: { form: DockForm }) {
    const host = useMemo(() => acquirePanelHost(form.id), [form.id]);
    // 宿主建立后通知停靠层来"认领"它 —— 渲染期不能发这个通知（见
    // `acquirePanelHost` 的说明），因此放在提交后的 effect 里。
    //
    // 【为什么卸载时**不**销毁宿主】`useMemo` 会把宿主元素缓存到 `form.id` 变化
    // 为止，而销毁它（`releasePanelHost` 会把它移出文档并注销）之后，缓存里的
    // 引用仍指向那个已脱离文档的 div —— 面板会被渲染到看不见的地方。开发模式下
    // `StrictMode` 会跑一次"挂载 → 清理 → 再挂载"，必然踩中这个坑；生产环境下
    // 任何导致本组件卸载再挂载的重排也会踩中。
    //
    // 宿主因此是**会话级**的容器：只要窗体还存在于布局里，它就留着（关掉面板时
    // 停泊在视口外）。窗体总数受面板注册表约束，不会无限增长。
    useEffect(() => {
        notifyPanelHosts();
    }, [form.id]);

    const definition = getPanel(form.panelId);
    if (!definition) return null;

    // 渲染函数优先：内置面板的 props 由 `App.tsx` 提供（见 `panelRenderer`），
    // 注册表里的 `component` 只作为没有外部 props 的面板（将来的插件面板）的
    // 后备实现。
    const render = getPanelRenderer(form.panelId);
    const PanelComponent = definition.component as ComponentType<DockPanelProps> | undefined;
    const content = render ? (
        render(form)
    ) : PanelComponent ? (
        <PanelComponent formId={form.id} panelId={form.panelId} props={form.props ?? {}} />
    ) : (
        <MissingPanel panelId={form.panelId} />
    );

    return createPortal(
        <PanelErrorBoundary formId={form.id} panelId={form.panelId}>
            {content}
        </PanelErrorBoundary>,
        host,
    );
}

function MissingPanel({ panelId }: { panelId: string }) {
    return (
        <div className="flex h-full w-full items-center justify-center bg-qt-window p-4 text-center text-xs text-qt-text-muted">
            {panelId}
        </div>
    );
}

/**
 * 单个面板的故障隔离。
 *
 * 面板一旦抛错，React 会卸载整棵子树 —— 包括它自己的宿主，于是停靠层手里
 * 捏着一个已脱离文档的节点，用户看到一片空白且没有任何提示。这里把错误
 * 收在面板内部，给出可读的失败界面与"重试"，其余窗体不受影响。
 */
class PanelErrorBoundary extends Component<
    { formId: string; panelId: string; children: ReactNode },
    { error: Error | null }
> {
    state: { error: Error | null } = { error: null };

    static getDerivedStateFromError(error: Error) {
        return { error };
    }

    componentDidCatch(error: Error) {
        console.error(`[dock] panel "${this.props.panelId}" crashed`, error);
    }

    render() {
        if (!this.state.error) return this.props.children;
        return (
            <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-qt-window p-4 text-center">
                <div className="text-xs text-qt-text">
                    {this.props.panelId} — panel failed to render
                </div>
                <div className="max-w-full truncate text-xs text-qt-text-muted">
                    {this.state.error.message}
                </div>
                <button
                    type="button"
                    className="rounded border border-qt-border bg-qt-panel px-2 py-1 text-xs text-qt-text hover:bg-qt-hover"
                    onClick={() => this.setState({ error: null })}
                >
                    Retry
                </button>
            </div>
        );
    }
}

/**
 * 需要存在的窗体集合。
 *
 * 规则：**曾经可见过**的窗体保持挂载（关闭后仍占一个停泊中的宿主），因此
 * 重开面板零成本、状态全在。从未打开过的面板不挂载 —— 否则启动时就会为
 * 记事本构建富文本编辑器、为未打开的面板创建 WebGL 上下文，白白拖慢冷启动。
 *
 * 这个集合是单调增长的：会话内只增不减。刷新后从头开始。
 */
export function DockPanelHosts({ forms }: { forms: DockForm[] }) {
    // 注册表或宿主集合变化（热更新重放注册、新窗体加入）时重渲染。
    useSyncExternalStore(subscribePanels, getPanelRegistryVersion, getPanelRegistryVersion);

    // "曾经可见"的累积由 `dockSlice.mountedFormIds` 维护（见那里的注释），
    // 这一层只负责把集合里的窗体渲染出来。
    const mounted = forms.filter((form) => getPanel(form.panelId) !== undefined);

    return (
        <>
            {mounted.map((form) => (
                <PanelMount key={form.id} form={form} />
            ))}
        </>
    );
}
