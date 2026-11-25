/**
 * 移动形态的外壳：顶栏 + 抽屉 + 底栏 + 修饰键行 + 菜单 sheet。
 *
 * 【为什么是"另一套外壳"而不是给 MenuBar 加移动样式】桌面顶栏是两层菜单（MenuBar +
 * ActionBar），它的存在前提是"有一根鼠标能精确点中 24px 的菜单项"。移动端上：
 * 这两层加起来 64px 高，占掉 844px 屏幕的 7.6%；菜单项需要二级悬停才能展开，
 * 而触摸没有悬停。把它压扁成一行图标并不会变好 —— 135 条动作里能塞进一行的不到
 * 10 条。因此移动端的外壳形态是**抽屉**：一个汉堡按钮换回全部动作，代价是一次点击。
 *
 * 【顶栏有哪些东西】菜单、工程名（用户要知道自己在哪）、撤销、重做、**工具栏**、操作抽屉。
 * 传输控制（播放/停止/录音）在**底栏**——拇指活动区，而且它们是高频操作。
 * 撤销/重做留在顶栏是因为它们与"我刚才那一下"直接相关，必须一眼可见；
 * 其余低频动作一律进抽屉（一次点击的代价换取不占常驻空间）。
 *
 * 【"工具栏"这个入口为什么必须存在】底栏只有传输四件，抽屉只装 `ACTION_META` 里的
 * 动作 —— 两者都**不含**工具栏开关（自动滚屏 / 涟漪 / 吸附 / 拖拽方向 / 参数编辑器
 * 跟随播放头…）。没有它，这些功能在手机上既没有按钮也没有动作：用户报的
 * "移动端仍然无法访问大部分工具栏功能，比如自动滚屏"就是这条断链。
 *
 * 【为什么底栏还有修饰键行】见 \`ModifierRail\`：没有它，slip 编辑 / 复制拖动 /
 * 范围选择在移动端完全不可达。
 *
 * 【为什么还多一个"菜单"按钮】抽屉只装得下 \`ACTION_META\` 里的**动作**。桌面菜单里
 * 另有一大批**不是动作**的条目：设置对话框（外观、快捷键、吸附网格、参数块…）、
 * 帮助里的日志与诊断、以及「视图 → 窗口 / 布局」这一整套窗口管理。它们此前在
 * 移动端**没有任何入口** —— 不是灰的，是不存在。菜单按钮把这些原样接回来：
 * 它渲染的就是同一个 \`MenuBar\`（见 \`AppChrome\`），因此不存在"第二份菜单定义"
 * 与随之而来的漂移。
 */
import {
    Cross2Icon,
    DotFilledIcon,
    HamburgerMenuIcon,
    LightningBoltIcon,
    MagnifyingGlassIcon,
    PauseIcon,
    PlayIcon,
    SliderIcon,
    StopIcon,
} from "@radix-ui/react-icons";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { useAppDispatch, useAppSelector } from "../app/hooks";
import { store } from "../app/store";
import { closeForm } from "../features/dock/dockSlice";
import {
    activeFormId,
    addEmptyPanel,
    resetLayout,
    togglePanelVisible,
} from "../features/dock/dockApi";
import type { ActionId } from "../features/keybindings/types";
import { useI18n } from "../i18n/I18nProvider";
import { AppIconButton } from "../ui/Button";
import { AppConfirmDialog } from "../ui/Dialog";
import { cx } from "../ui/cx";
import { ModifierRail } from "./ModifierRail";
import { buildMobileActionSections, filterMobileSections } from "./mobileActions";
import { buildMobileWindowRows, windowSummary } from "./mobileWindows";
import { menuSelectionClosesSheet } from "./mobileMenuSheet";
import { openDockLayoutSettingsDialog } from "../components/dock/dockLayoutDialogBridge";

export interface MobileChromeProps {
    /** 工程名（顶栏标题）。 */
    title: string;
    /** 是否有未保存的更改。 */
    dirty: boolean;
    canUndo: boolean;
    canRedo: boolean;
    /** 执行一条动作。由 \`App.tsx\` 接上它既有的快捷键派发器。 */
    onAction: (actionId: ActionId) => void;
    isPlaying: boolean;
    onTogglePlay: () => void;
    onStop: () => void;
    /** 状态栏内容（与桌面共用同一个节点，只是放在不同位置）。 */
    statusBar?: ReactNode;
}

/** 顶栏。 */
/**
 * 按住连续执行的动作按钮（撤销 / 重做）。
 *
 * 【为什么不用 `holdRepeat`】那套机制靠 `keyup` 终止，而触摸屏没有 `keyup`：
 * 点一下就会变成「每 50ms 一次，永不停止」—— 用户报的「撤销几乎不可用」
 * 有一半来自这里（一次点击撤掉整个历史）。这里改成**真长按**：
 * `pointerdown` 立即执行一次，按住 400ms 后每 50ms 一次，`pointerup` /
 * `pointercancel` / 移出按钮即停。节奏与桌面 `holdRepeat` 一致，但终止条件
 * 是「手指还在不在」，而不是「有没有键松开」。
 */
const HOLD_REPEAT_INITIAL_MS = 400;
const HOLD_REPEAT_INTERVAL_MS = 50;

function MobileRepeatActionButton({
    icon,
    tooltip,
    disabled,
    onFire,
}: {
    icon: ReactNode;
    tooltip: string;
    disabled: boolean;
    onFire: () => void;
}) {
    const timers = useRef<{ initial: number | null; repeat: number | null }>({
        initial: null,
        repeat: null,
    });
    const stop = useCallback(() => {
        if (timers.current.initial != null) {
            window.clearTimeout(timers.current.initial);
            timers.current.initial = null;
        }
        if (timers.current.repeat != null) {
            window.clearInterval(timers.current.repeat);
            timers.current.repeat = null;
        }
    }, []);
    useEffect(() => stop, [stop]);
    return (
        <span
            className="contents"
            onPointerDown={() => {
                if (disabled) return;
                stop();
                onFire();
                timers.current.initial = window.setTimeout(() => {
                    timers.current.initial = null;
                    timers.current.repeat = window.setInterval(onFire, HOLD_REPEAT_INTERVAL_MS);
                }, HOLD_REPEAT_INITIAL_MS);
            }}
            onPointerUp={stop}
            onPointerCancel={stop}
            onPointerLeave={stop}
        >
            {/* `onClick` 留空：真正的执行在 pointerdown 上 —— 用 onClick 会让
                「按住」的第一次与「抬起」的第二次重复执行同一个动作。 */}
            <AppIconButton icon={icon} tooltip={tooltip} disabled={disabled} onClick={() => {}} />
        </span>
    );
}
export function MobileTopBar({
    title,
    dirty,
    canUndo,
    canRedo,
    onAction,
    onOpenDrawer,
    onOpenMenu,
    onOpenToolbar,
}: {
    title: string;
    dirty: boolean;
    canUndo: boolean;
    canRedo: boolean;
    onAction: (actionId: ActionId) => void;
    onOpenDrawer: () => void;
    onOpenMenu: () => void;
    /** 打开工具栏 sheet（见 `MobileToolbarSheet`）。 */
    onOpenToolbar: () => void;
}) {
    const { t } = useI18n();
    return (
        <div
            className="flex shrink-0 items-center gap-1 border-b border-qt-border bg-qt-window px-1"
            style={{ height: "var(--qt-bar-main)" }}
        >
            {/*
             * 两个入口的分工：
             * - 菜单（☰）= 桌面菜单栏，装的是**对话框与窗口管理**；
             * - 操作（⚡）= 动作抽屉，装的是**可以直接执行的动作**。
             * 顺序固定为"菜单在前"：它与桌面左上角的位置一致，用户在两端的
             * 第一反应都是"去左上角找功能"。
             */}
            <AppIconButton
                icon={<HamburgerMenuIcon />}
                tooltip={t("mobile_menu_title")}
                onClick={onOpenMenu}
            />
            <div className="min-w-0 flex-1">
                <div className="truncate text-qt-md" data-tooltip={title}>
                    {title}
                    {dirty ? <span aria-hidden> *</span> : null}
                </div>
            </div>
            {/* 撤销 / 重做：按下即执行 + 按住连续（见 `MobileRepeatActionButton`）。 */}
            <MobileRepeatActionButton
                icon={<span aria-hidden>↶</span>}
                tooltip={t("kb_edit_undo")}
                disabled={!canUndo}
                onFire={() => onAction("edit.undo")}
            />
            <MobileRepeatActionButton
                icon={<span aria-hidden>↷</span>}
                tooltip={t("kb_edit_redo")}
                disabled={!canRedo}
                onFire={() => onAction("edit.redo")}
            />
            {/*
             * 工具栏（滑块图标）放在"操作"左边：三个入口的分工是
             * **菜单 = 对话框与窗口 / 工具栏 = 开关与模式 / 操作 = 可执行动作**。
             * 顺序按"离手指最远 → 最近"排：工具栏比抽屉更常用（每个开关都是
             * 一次点击，而抽屉里多数动作有快捷键或右键等价物）。
             */}
            <AppIconButton
                icon={<SliderIcon />}
                tooltip={t("mobile_toolbar_title")}
                onClick={onOpenToolbar}
            />
            <AppIconButton
                icon={<LightningBoltIcon />}
                tooltip={t("mobile_actions_title")}
                onClick={onOpenDrawer}
            />
        </div>
    );
}

/** 底栏：传输 + 修饰键行。 */
export function MobileBottomBar({
    isPlaying,
    onTogglePlay,
    onStop,
    onAction,
    modifierChips,
    statusBar,
}: {
    isPlaying: boolean;
    onTogglePlay: () => void;
    onStop: () => void;
    onAction: (actionId: ActionId) => void;
    modifierChips: Array<{ id: ActionId; label: string }>;
    statusBar?: ReactNode;
}) {
    const { t } = useI18n();
    return (
        <div className="flex shrink-0 flex-col border-t border-qt-border bg-qt-window">
            <ModifierRail chips={modifierChips} />
            <div className="flex items-center gap-1 px-1" style={{ height: "var(--qt-bar-main)" }}>
                <AppIconButton
                    icon={isPlaying ? <PauseIcon /> : <PlayIcon />}
                    tooltip={t("kb_playback_toggle")}
                    active={isPlaying}
                    emphasis="accent"
                    onClick={onTogglePlay}
                />
                <AppIconButton
                    icon={<StopIcon />}
                    tooltip={t("kb_playback_stop")}
                    onClick={onStop}
                />
                {/*
                 * 录音按钮用实心圆点而不是 \`RecordIcon\`：Radix 的图标集里没有
                 * "录音"这个语义（\`RecordIcon\` 不存在），而实心圆点是音频软件里
                 * 通用的录音记号。\`intent="danger"\` 让它拿到红色。
                 */}
                <AppIconButton
                    icon={<DotFilledIcon />}
                    tooltip={t("kb_recording_toggle")}
                    intent="danger"
                    onClick={() => onAction("recording.toggle")}
                />
                <div className="min-w-0 flex-1" />
                <AppIconButton
                    icon={<span aria-hidden>⌁</span>}
                    tooltip={t("kb_mode_toggle")}
                    onClick={() => onAction("mode.toggle")}
                />
            </div>
            {statusBar}
        </div>
    );
}

/**
 * 动作抽屉：全屏 sheet，按分组列出全部可用动作 + 窗口开关。
 *
 * 【为什么是全屏而不是侧滑半屏】390px 宽下侧滑面板只剩 300px 给内容，而条目文案
 * （"把选中音频块的范围加入参数编辑器选区"）需要整行宽度。全屏没有这个损失，
 * 而"全屏会挡住工作区"在这里不是问题 —— 用户打开抽屉就是要找东西。
 */
export function MobileActionDrawer({
    open,
    onClose,
    onAction,
    shortcutFor,
}: {
    open: boolean;
    onClose: () => void;
    onAction: (actionId: ActionId) => void;
    shortcutFor: (id: ActionId) => string | null;
}) {
    const { t } = useI18n();
    const [query, setQuery] = useState("");
    const sections = useMemo(() => buildMobileActionSections(t, shortcutFor), [t, shortcutFor]);
    const filtered = useMemo(() => filterMobileSections(sections, query), [sections, query]);

    // 抽屉打开时禁止背景滚动：否则滑动列表会把底下的画布一起拖动。
    useEffect(() => {
        if (!open) return;
        const previous = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        return () => {
            document.body.style.overflow = previous;
        };
    }, [open]);

    // 关闭即清空搜索词：下次打开从完整列表开始（与快捷键对话框同一约定）。
    useEffect(() => {
        if (!open) setQuery("");
    }, [open]);

    if (!open) return null;
    return (
        <div className="hs-mobile-drawer absolute inset-0 z-[var(--qt-z-dialog)] flex flex-col bg-qt-window">
            <div
                className="flex shrink-0 items-center justify-between border-b border-qt-border px-2"
                style={{ height: "var(--qt-bar-main)" }}
            >
                <span className="text-qt-md">{t("mobile_actions_title")}</span>
                <AppIconButton icon={<Cross2Icon />} tooltip={t("close")} onClick={onClose} />
            </div>
            {/*
             * 搜索框用原生 `<input>` 而不是 Radix `TextField`：设计系统的门禁
             * （`ui/araConformanceGates`）对"直用 Radix 表单原语的文件数"设了基线，
             * 而这里需要的只是一个受控文本框 —— 借用 Radix 会让基线多一个文件，
             * 换来的只是它自带的一层样式，与本行的令牌样式重复。
             */}
            <div className="relative shrink-0 px-2 py-1">
                <MagnifyingGlassIcon
                    aria-hidden
                    width={14}
                    height={14}
                    className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-qt-text-muted"
                />
                <input
                    type="text"
                    inputMode="search"
                    enterKeyHint="search"
                    aria-label={t("mobile_search_placeholder")}
                    placeholder={t("mobile_search_placeholder")}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    className="h-qt-ctl-md w-full rounded border border-qt-border bg-qt-base pl-7 pr-2 text-qt-md text-qt-text outline-hidden"
                />
            </div>
            <div className="hs-scroll-gutter min-h-0 flex-1 overflow-y-auto">
                <MobileWindowsSection onNavigate={onClose} />
                {filtered.map((section) => (
                    <div key={section.id} className="border-b border-qt-border/60 py-1">
                        <div className="px-2 py-1 text-qt-micro text-qt-text-muted">
                            {section.title}
                        </div>
                        {section.items.map((item) => (
                            <button
                                key={item.id}
                                type="button"
                                className={cx(
                                    "flex w-full items-center justify-between gap-2 px-2 text-left text-qt-md",
                                    "hover:bg-qt-hover active:bg-qt-highlight/20",
                                )}
                                style={{ minHeight: "var(--qt-menu-item-h)" }}
                                onClick={() => {
                                    onAction(item.id);
                                    onClose();
                                }}
                            >
                                <span className="min-w-0 truncate">{item.label}</span>
                                {item.shortcut ? (
                                    <span className="shrink-0 text-qt-micro text-qt-text-muted">
                                        {item.shortcut}
                                    </span>
                                ) : null}
                            </button>
                        ))}
                    </div>
                ))}
                {filtered.length === 0 ? (
                    <div className="px-2 py-3 text-qt-sm text-qt-text-muted">
                        {t("mobile_search_empty")}
                    </div>
                ) : null}
            </div>
        </div>
    );
}

/**
 * 「窗口」分区：每个面板一个开关 + 窗口级操作。
 *
 * 【为什么它排在动作分区**之前**】用户打开抽屉最常见的原因是"我把东西关掉了，
 * 想找回来"。排在 135 条动作之后等于把恢复入口埋在最底下。
 *
 * 【为什么开关不关闭抽屉】窗口是**可连续操作**的对象（开一个、关一个、再开一个），
 * 每点一次就关掉抽屉会让这套动作变成"点-重开-点-重开"。窗口级命令（重置布局、
 * 布局设置）则相反：它们的结果是全屏级的，继续停在抽屉里只会挡住用户看结果。
 */
function MobileWindowsSection({ onNavigate }: { onNavigate: () => void }) {
    const { t, tf } = useI18n();
    const dispatch = useAppDispatch();
    const layout = useAppSelector((state) => state.dock.layout);
    /*
     * 用 `dockApi.activeFormId` 而不是直接读 `state.dock.activeFormId`：后者只在
     * "聚焦过某个窗体"之后才有值，而活动标签本身（主标签组的 `active`）同样是
     * 一个可关闭的窗体。两处判据分叉的症状是"明明有一个窗口在前台，关闭按钮却是灰的"。
     */
    const currentFormId = useAppSelector((state) => activeFormId(() => state));
    const rows = useMemo(() => buildMobileWindowRows(layout), [layout]);
    const [resetConfirmOpen, setResetConfirmOpen] = useState(false);
    const summary = windowSummary(rows);
    const canCloseCurrent = Boolean(currentFormId && layout.forms[currentFormId]);

    if (rows.length === 0) return null;
    return (
        <div className="border-b border-qt-border/60 py-1" data-mobile-window-section="1">
            <div className="flex items-center justify-between px-2 py-1 text-qt-micro text-qt-text-muted">
                <span>{t("mobile_windows_section")}</span>
                {summary ? <span>{summary}</span> : null}
            </div>
            {rows.map((row) => (
                <button
                    key={row.panelId}
                    type="button"
                    role="switch"
                    aria-checked={row.visible}
                    data-mobile-window-row={row.panelId}
                    className={cx(
                        "flex w-full items-center gap-2 px-2 text-left text-qt-md",
                        "hover:bg-qt-hover active:bg-qt-highlight/20",
                        row.visible ? "text-qt-text" : "text-qt-text-muted",
                    )}
                    style={{ minHeight: "var(--qt-menu-item-h)" }}
                    onClick={() => togglePanelVisible(dispatch, store.getState, row.panelId)}
                >
                    <span aria-hidden className="shrink-0">
                        {row.visible ? "●" : "○"}
                    </span>
                    <span className="min-w-0 flex-1 truncate">{tf(row.titleKey)}</span>
                    <span className="shrink-0 text-qt-micro text-qt-text-muted">
                        {row.visible ? t("mobile_window_shown") : t("mobile_window_hidden")}
                    </span>
                </button>
            ))}
            <div className="mt-1 border-t border-qt-border/60 pt-1">
                <button
                    type="button"
                    disabled={!canCloseCurrent}
                    className={cx(
                        "flex w-full items-center px-2 text-left text-qt-md",
                        canCloseCurrent
                            ? "text-qt-danger-text hover:bg-qt-hover"
                            : "text-qt-text-muted opacity-50",
                    )}
                    style={{ minHeight: "var(--qt-menu-item-h)" }}
                    onClick={() => {
                        const formId = activeFormId(store.getState);
                        if (formId) dispatch(closeForm(formId));
                    }}
                >
                    {t("mobile_window_close_current")}
                </button>
                <button
                    type="button"
                    className="flex w-full items-center px-2 text-left text-qt-md hover:bg-qt-hover"
                    style={{ minHeight: "var(--qt-menu-item-h)" }}
                    onClick={() => {
                        addEmptyPanel(dispatch);
                        onNavigate();
                    }}
                >
                    {tf("layout_new_panel")}
                </button>
                <button
                    type="button"
                    className="flex w-full items-center px-2 text-left text-qt-md hover:bg-qt-hover"
                    style={{ minHeight: "var(--qt-menu-item-h)" }}
                    onClick={() => {
                        openDockLayoutSettingsDialog();
                        onNavigate();
                    }}
                >
                    {tf("layout_settings")}
                </button>
                <button
                    type="button"
                    className="flex w-full items-center px-2 text-left text-qt-md hover:bg-qt-hover"
                    style={{ minHeight: "var(--qt-menu-item-h)" }}
                    onClick={() => setResetConfirmOpen(true)}
                >
                    {tf("layout_reset")}
                </button>
            </div>
            <AppConfirmDialog
                open={resetConfirmOpen}
                onOpenChange={setResetConfirmOpen}
                title={tf("layout_reset_confirm_title")}
                message={tf("layout_reset_confirm_body")}
                confirmLabel={tf("layout_reset")}
                cancelLabel={t("cancel")}
                intent="primary"
                onConfirm={() => {
                    resetLayout(dispatch);
                    setResetConfirmOpen(false);
                    onNavigate();
                }}
            />
        </div>
    );
}

/**
 * 菜单 sheet：把**桌面菜单栏本体**搬到移动端。
 *
 * 【为什么是"搬"而不是"另写一份"】桌面菜单有 6 个一级菜单、几十个二级项与十几套
 * 对话框状态（外观 / 快捷键 / 吸附 / 参数块 / 拉伸算法 / 推理设备 / 日志 / 诊断…）。
 * 在移动端重列一份清单，等于把"每个新功能都要在两处登记"制度化 —— 漏掉的那一处
 * 只会表现为"手机上找不到这个功能"，没有任何报错。这里渲染的是同一个 \`MenuBar\`
 * 实例（\`AppChrome\` 只挂一次），因此两端**永远同步**。
 *
 * 【为什么点完一项就收起】菜单项的结果要么是打开对话框（portal 到 body，与 sheet
 * 无关），要么是切换某个显示项。继续盖着一层全屏 sheet 只会挡住用户刚打开的东西。
 * 子菜单触发器（\`aria-haspopup="menu"\`）是例外：它只是展开下一级，收起 sheet
 * 会让用户再也点不到那一级。
 *
 * 【为什么关闭时是 display:none 而不是不渲染】MenuBar 里住着一批对话框（导入、
 * 导出、转调、诊断…）与它们的事件监听。卸载它，那些功能就变成"点了没反应"。
 */
export function MobileMenuSheet({
    open,
    onClose,
    menu,
}: {
    open: boolean;
    onClose: () => void;
    menu: ReactNode;
}) {
    const { t } = useI18n();

    /*
     * 点完一项就收起 sheet。
     *
     * 【为什么挂在 document 上】菜单内容由 Radix portal 到 body，二级/三级菜单
     * 更是完全不在 sheet 的 DOM 子树里（见 `mobileMenuSheet` 的文件头）。
     * 用捕获阶段是为了先于 Radix 自己的 click 处理拿到事件 —— 我们要的只是
     * "这一下点在了条目上"这个事实，动作本身仍由菜单执行。
     *
     * 【为什么延后一帧】收起 sheet 会把触发器（菜单栏）一起隐藏，而 Radix 在
     * onSelect 之后还要做焦点归还与关闭动画。等这一轮事件处理完再收，避免它在
     * 半关闭状态下找不到可聚焦的触发器。
     */
    useEffect(() => {
        if (!open) return;
        const onClick = (event: MouseEvent) => {
            if (!menuSelectionClosesSheet(event.target)) return;
            window.setTimeout(onClose, 0);
        };
        document.addEventListener("click", onClick, true);
        return () => document.removeEventListener("click", onClick, true);
    }, [open, onClose]);

    return (
        <div
            className={open ? "hs-mobile-menu-sheet" : "hidden"}
            data-mobile-menu-sheet={open ? "open" : "closed"}
            aria-hidden={!open}
        >
            <div
                className="flex shrink-0 items-center justify-between border-b border-qt-border px-2"
                style={{ height: "var(--qt-bar-main)" }}
            >
                <span className="text-qt-md">{t("mobile_menu_title")}</span>
                <AppIconButton icon={<Cross2Icon />} tooltip={t("close")} onClick={onClose} />
            </div>
            <div className="hs-mobile-menu-sheet-body">{menu}</div>
        </div>
    );
}

/**
 * 移动形态的**工具栏 sheet**：桌面那一排工具栏开关的触摸形态。
 *
 * 【为什么是 sheet 而不是"再塞几个按钮进底栏"】工具栏有 12 组控件（捕获、撤销/重做、
 * 文件浏览器、交叉淡化、分割过渡、吸附、涟漪、播放头缩放、自动滚屏、跟随播放头、
 * 点击选轨、忽略编组）。底栏已经装着传输四件与修饰键行，390px 上再塞就是一行点不中
 * 的小图标。sheet 一次点击换回**全部**开关，与菜单 sheet 同一代价模型。
 *
 * 【为什么内容由外部传入】它必须是**同一个 `ToolbarStrip` 实例**（`variant="sheet"`）：
 * `PitchSnapSettingsDialog` / `SnapGridSettingsDialog` / `SplitTransitionSettingsDialog` /
 * `CustomScaleDialog` / `RecordingSettingsDialog` 的开关状态住在那个组件里，
 * 渲染第二份会让同一个对话框有两套状态。与 `MobileMenuSheet` 的说明同一条理由。
 *
 * 【为什么关闭时是 display:none 而不是不渲染】同上：卸载它，那批对话框就退回到
 * "点了没反应"。
 */
export function MobileToolbarSheet({
    open,
    onClose,
    toolbar,
}: {
    open: boolean;
    onClose: () => void;
    toolbar: ReactNode;
}) {
    const { t } = useI18n();
    return (
        <div
            className={open ? "hs-mobile-toolbar-sheet" : "hidden"}
            data-mobile-toolbar-sheet={open ? "open" : "closed"}
            aria-hidden={!open}
        >
            <div
                className="flex shrink-0 items-center justify-between border-b border-qt-border px-2"
                style={{ height: "var(--qt-bar-main)" }}
            >
                <span className="text-qt-md">{t("mobile_toolbar_title")}</span>
                <AppIconButton icon={<Cross2Icon />} tooltip={t("close")} onClick={onClose} />
            </div>
            <div className="hs-mobile-toolbar-sheet-body">{toolbar}</div>
        </div>
    );
}
