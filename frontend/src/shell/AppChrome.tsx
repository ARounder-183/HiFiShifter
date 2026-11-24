/**
 * 应用外壳的**唯一**装配点。
 *
 * 【它解决的三个问题】
 *
 * 1. **形态分支只有一处**。此前"这个界面在移动端长什么样"没有任何地方回答 ——
 *    每个组件各自判断。现在只有这里读 `appForm()`。
 * 2. **根容器的度量只有一处**。`100dvh`（而不是 `100vh`）、安全区内边距、
 *    `overflow: hidden`、`relative`（抽屉的定位基准）—— 这些必须同时成立，
 *    分开写就会漏掉一条（典型症状：底部工具栏被手势条盖住 8px）。
 * 3. **抽屉的生命周期与外壳同级**。抽屉要盖住顶栏与底栏，因此它的状态不能住在
 *    底栏里（底栏被抽屉盖住时状态还在，但定位基准错了）。
 *
 * 【为什么 `App.tsx` 只换 render 尾部】它有 4800 行、持有约 40 个对话框状态与 20+
 * 个 effect。抽状态是高风险无收益的重构；抽外壳是低风险高收益 —— 见设计文档 §5.5。
 */
import { useEffect, useMemo, useState, type ReactNode } from "react";

import type { ActionId } from "../features/keybindings/types";
import { appForm } from "../services/hostDetect";
import { registerBackHandler } from "../services/backGesture";
import { MobileActionDrawer, MobileBottomBar, MobileTopBar } from "./MobileChrome";
import { buildModifierChips } from "./mobileActions";
import { useI18n } from "../i18n/I18nProvider";

export interface AppChromeProps {
    /** 桌面顶栏（菜单栏 + 动作栏 + 宿主提示条）。移动形态不渲染。 */
    desktopHeader?: ReactNode;
    /** 桌面状态栏。移动形态不渲染 —— 那里的同一份内容归底栏。 */
    desktopFooter?: ReactNode;

    // ── 移动形态的数据（不是节点）──────────────────────────────────────
    /** 工程名（顶栏标题）。 */
    mobileTitle: string;
    mobileDirty: boolean;
    canUndo: boolean;
    canRedo: boolean;
    isPlaying: boolean;
    onTogglePlay: () => void;
    onStop: () => void;
    /** 执行一条动作。由 `App.tsx` 接到它既有的快捷键派发器上。 */
    onMobileAction: (actionId: ActionId) => void;
    /** 移动形态的状态栏内容（渲染在底栏最后一行）。 */
    mobileStatusBar?: ReactNode;

    children: ReactNode;
}

export function AppChrome({
    desktopHeader,
    desktopFooter,
    mobileTitle,
    mobileDirty,
    canUndo,
    canRedo,
    isPlaying,
    onTogglePlay,
    onStop,
    onMobileAction,
    mobileStatusBar,
    children,
}: AppChromeProps) {
    const form = appForm();
    const isMobile = form === "mobile";
    const { t } = useI18n();
    const [drawerOpen, setDrawerOpen] = useState(false);
    const modifierChips = useMemo(() => buildModifierChips(t), [t]);
    /*
     * 抽屉里的快捷键提示在移动端**不显示**：触摸设备上没有键盘，标出 "Ctrl+Shift+Z"
     * 只会占用宝贵的行宽，并且暗示"这个手势在这里可用"——它不可用。
     * 传一个恒为 null 的函数而不是把参数删掉：`mobileActions` 的签名对桌面调试入口
     * （`?form=mobile`）仍然有意义，那里确实有键盘。
     */
    const shortcutFor = useMemo(() => () => null as string | null, []);

    /*
     * 返回键先关抽屉。
     *
     * 【为什么抽屉要自己注册】Radix 的浮层靠"收到 Escape"关闭（见
     * `services/backGesture`），而抽屉是自绘的全屏层，没有这套机制。不注册的话，
     * 抽屉打开时按返回键会直接退出应用 —— 而用户的本意是"关掉这个列表"。
     */
    useEffect(() => {
        if (!isMobile) return;
        return registerBackHandler(() => {
            if (!drawerOpen) return false;
            setDrawerOpen(false);
            return true;
        });
    }, [drawerOpen, isMobile]);

    return (
        <div
            data-form={form}
            className="hs-surface-root relative flex h-dvh w-full flex-col overflow-hidden bg-qt-window text-qt-text font-sans text-qt-md selection:bg-qt-highlight selection:text-white"
        >
            {isMobile ? (
                <MobileTopBar
                    title={mobileTitle}
                    dirty={mobileDirty}
                    canUndo={canUndo}
                    canRedo={canRedo}
                    onAction={onMobileAction}
                    onOpenDrawer={() => setDrawerOpen(true)}
                />
            ) : (
                desktopHeader
            )}

            {/*
             * 工作区。`relative` 是为了让抽屉的 `absolute inset-0` 以**工作区**为界 ——
             * 抽屉盖住顶栏与底栏是刻意的（见 MobileActionDrawer 的说明），因此抽屉
             * 实际挂在下面那一层，这里只提供定位上下文给面板内的浮层。
             */}
            <div className="relative flex min-h-0 flex-1 flex-col">{children}</div>

            {isMobile ? (
                <MobileBottomBar
                    isPlaying={isPlaying}
                    onTogglePlay={onTogglePlay}
                    onStop={onStop}
                    onAction={onMobileAction}
                    modifierChips={modifierChips}
                    statusBar={mobileStatusBar}
                />
            ) : (
                desktopFooter
            )}

            {isMobile ? (
                <MobileActionDrawer
                    open={drawerOpen}
                    onClose={() => setDrawerOpen(false)}
                    onAction={onMobileAction}
                    shortcutFor={shortcutFor}
                />
            ) : null}
        </div>
    );
}
