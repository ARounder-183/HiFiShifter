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
import { MobileActionDrawer, MobileBottomBar, MobileMenuSheet, MobileTopBar } from "./MobileChrome";
import { buildModifierChips } from "./mobileActions";
import { useI18n } from "../i18n/I18nProvider";

export interface AppChromeProps {
    /** 桌面顶栏（菜单栏 + 动作栏 + 宿主提示条）。移动形态不渲染。 */
    desktopHeader?: ReactNode;
    /**
     * 移动形态的「菜单」内容（桌面菜单栏本体）。
     *
     * 【为什么与 `desktopHeader` 分开】移动端只需要菜单栏 —— 动作栏（ActionBar）
     * 是桌面密度的一整排工具条，在 390px 上既放不下也没有对应交互（它的高频项
     * 已经在移动底栏与动作抽屉里）。把整个 `desktopHeader` 塞进 sheet 会把两件事
     * 混在一起，而且动作栏的键盘监听在触摸设备上没有意义。
     *
     * 【为什么必须由外部传同一个节点】它必须是**同一个 `MenuBar` 实例**：
     * 对话框状态、事件监听、菜单展开状态都住在组件内部，渲染第二份会让同一个
     * 对话框被打开两次。
     */
    mobileMenu?: ReactNode;
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
    mobileMenu,
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
    const [menuOpen, setMenuOpen] = useState(false);
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

    /*
     * 菜单 sheet 的返回键。注册在抽屉之后 —— `registerBackHandler` 是栈，
     * 后注册的先被询问，因此"菜单盖着抽屉"时先关菜单（菜单打开时抽屉已经被关掉，
     * 这条顺序仍然成立：两者互斥，见下面的两个 onOpen）。
     */
    useEffect(() => {
        if (!isMobile) return;
        return registerBackHandler(() => {
            if (!menuOpen) return false;
            setMenuOpen(false);
            return true;
        });
    }, [menuOpen, isMobile]);

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
                    onOpenDrawer={() => {
                        setMenuOpen(false);
                        setDrawerOpen(true);
                    }}
                    onOpenMenu={() => {
                        setDrawerOpen(false);
                        setMenuOpen(true);
                    }}
                />
            ) : (
                desktopHeader
            )}

            {/*
             * 移动端挂载的顶栏：**菜单 sheet**。
             *
             * 【它同时解决了两件事】
             * 1. 对话框的宿主：导入媒体、导出音频、转调/设置音高/平滑/量化这一整套
             *    对话框，实现都住在 `MenuBar` 里：抽屉与面板只派发
             *    `hifi:importMediaFromMenu` / `hifi:openEditDialog` 这两个 DOM 事件，
             *    而**唯一的监听者**是 MenuBar 自己的 `useEffect`。移动形态不渲染
             *    MenuBar，于是这些动作全部变成"点了没反应"—— 没有报错、没有日志。
             * 2. 菜单本身：设置对话框、帮助里的日志与诊断、「视图 → 窗口 / 布局」，
             *    这些**不是动作**（不在 `ACTION_META` 里）的条目此前在移动端根本
             *    没有入口。现在它们就是桌面菜单栏本体（同一个实例，见 `mobileMenu`）。
             *
             * 【为什么关闭时是 display:none 而不是不渲染】`MenuBar` 里的对话框与
             * 事件监听必须常驻 —— 卸载它，上面第 1 条就退回到"点了没反应"。
             * 它的对话框都 portal 到 body，因此父级 `display:none` 不影响它们。
             */}
            {isMobile ? (
                <MobileMenuSheet
                    open={menuOpen}
                    onClose={() => setMenuOpen(false)}
                    menu={mobileMenu ?? desktopHeader}
                />
            ) : null}

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
