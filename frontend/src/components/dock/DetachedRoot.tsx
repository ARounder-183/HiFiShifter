/**
 * 独立窗口的根组件：渲染**一个**被拆出去的面板。
 *
 * 【为什么这里不能直接按 id 硬编码组件映射】此前这里是一个 `switch (panelId)`，
 * 只认三个内置面板，`default` 返回 `null`。后果是：任何走注册中心
 * `component` 通道的面板（也就是第三方面板唯一可用的通道）即使声明了
 * `detachable`，被拖出成独立窗口后也会**静默渲染成空白** —— 连"面板不可用"
 * 都不显示。而 `PanelDefinition.detachable` 是公开字段，等于对外承诺了一件
 * 做不到的事。
 *
 * 现在改为查注册中心：`getPanel(panelId)?.component`。内置的三个自包含面板
 * 也改走同一条通道（见 `registerBuiltinPanels`），因此这条分支有真实用例。
 *
 * 【为什么独立窗口要自己注册面板】独立窗口是另一个 JS 上下文，`App.tsx` 的
 * 模块级 `registerBuiltinPanels()` 不会执行到。因此 `detachedMain.tsx` 里也要
 * 注册一次；两处调用是幂等的（`registerPanel` 对同 id 覆盖）。
 *
 * 【为什么先等快照】面板读的是 Redux 状态（工程、剪贴板、文件浏览器目录…）。快照
 * 到达前渲染会读到空状态（例如记事本会以为工程没有笔记），因此先显示一行"正在连接
 * 主窗口"，等状态就位再挂载面板 —— 面板的首次渲染就是完整状态。
 */

import { Suspense, useCallback, useEffect, useMemo } from "react";

import { useAppSelector } from "../../app/hooks";
import { translateOutsideReact } from "../../i18n/I18nProvider";
import { getPanel } from "../../features/dock/panelRegistry";
import { satelliteFormId, subscribeRemoteAppearance } from "../../features/dock/detachBridge";
import { collectSubtreeRootIds } from "../../features/dock/dockPanel";
import { isPanelForm, rootOfForm } from "../../features/dock/dockTree";
import { useAppTheme } from "../../theme/AppThemeProvider";
import type { AppearanceSettings } from "../../theme/themeTypes";
import { DockSubRoot } from "./DockSubRoot";
import { DockPanelHosts } from "./DockPanelHosts";
import "./dock.css";

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
    const PanelComponent = panelId ? getPanel(panelId)?.component : undefined;

    /**
     * 继承主窗口的外观（主题 / **自定义字体**）。
     *
     * 【为什么由主窗口下发，而不是自己读 localStorage】外观只存在 localStorage 里。
     * 卫星窗口过去依赖"两个窗口共享同一份存储"这一环境假设，于是当存储分区不同、
     * 或窗口在写入之前就挂载时，字体退回默认值（用户报告"独立窗口没有继承主窗口的
     * 自定义字体"）。现在主窗口在快照里带一次、变更时再推送，卫星不再依赖环境假设；
     * `applySettings` 会同时写入本窗口的存储，后续自读也是对的。
     *
     * `subscribeRemoteAppearance` 在订阅时会立即用已收到的值回调一次，因此无论快照
     * 早于还是晚于本组件挂载，都能应用上。
     */
    const theme = useAppTheme();
    useEffect(() => {
        return subscribeRemoteAppearance((appearance) => {
            if (appearance == null) return;
            theme.applySettings(appearance as AppearanceSettings);
        });
    }, [theme]);

    const onDragOver = useCallback((event: React.DragEvent) => {
        // 独立窗口不做跨窗口拖放（宿主窗口不同，拖拽数据无法贯通）：显式阻止默认
        // 行为，避免 WebView 把文件拖进来时整页导航。
        event.preventDefault();
    }, []);

    // ── 面板分支：整棵子树在本窗口渲染 ─────────────────────────────
    // 面板被拆到独立窗口时，拆出的不是"一个面板组件"而是**一片可停靠区**：
    // 子树里的全部成员都要在本窗口重新挂载（跨 JS 上下文无法搬 DOM —— 可拆性
    // 恰好就是"重挂载代价可接受"的证书，见 `isPanelDetachable`），所以这里
    // 先挂宿主层，再渲染面板自己的布局根。
    const layout = useAppSelector((state) => state.dock.layout);
    const isPanel = isPanelForm(form ?? undefined);
    const subtreeRootIds = useMemo(
        () =>
            isPanel && form?.childRootId ? collectSubtreeRootIds(layout, form.childRootId) : null,
        [isPanel, form, layout],
    );
    const hostedForms = useMemo(() => {
        if (!subtreeRootIds) return [];
        return layout.order
            .map((id) => layout.forms[id])
            .filter((member) => {
                if (!member || isPanelForm(member)) return false;
                const memberRoot = rootOfForm(layout, member.id);
                return memberRoot !== null && subtreeRootIds.has(memberRoot);
            });
    }, [layout, subtreeRootIds]);

    return (
        <div
            className="h-screen w-screen overflow-hidden bg-qt-window text-qt-text"
            onDragOver={onDragOver}
            data-detached-form={formId ?? undefined}
        >
            {/*
             * 三态（普通窗体）：面板未注册（找不到 panelId）→ 显示"不可用"而不是
             * 空白；已注册 → 渲染；快照未到（无 formId）→ 显示"正在连接"。
             * 面板窗体走上面的分支：渲染它自己的整棵布局树。
             */}
            {isPanel && form?.childRootId ? (
                <div className="flex h-full w-full flex-col">
                    <DockPanelHosts forms={hostedForms} />
                    <DockSubRoot rootId={form.childRootId} kind="panel" />
                </div>
            ) : panelId && !PanelComponent ? (
                <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-qt-xs text-qt-text-muted">
                    <span>{translateOutsideReact("panel_unavailable")}</span>
                    <span className="text-qt-text-muted opacity-70">{panelId}</span>
                </div>
            ) : PanelComponent ? (
                // 与停靠宿主一致：注册表里的 component 可能是 lazy 组件
                <Suspense
                    fallback={
                        <div className="flex h-full w-full items-center justify-center text-qt-xs text-qt-text-muted">
                            …
                        </div>
                    }
                >
                    <PanelComponent
                        formId={formId ?? ""}
                        panelId={panelId ?? ""}
                        props={form?.props ?? {}}
                    />
                </Suspense>
            ) : (
                <div className="flex h-full w-full items-center justify-center text-qt-xs text-qt-text-muted">
                    …
                </div>
            )}
        </div>
    );
}
