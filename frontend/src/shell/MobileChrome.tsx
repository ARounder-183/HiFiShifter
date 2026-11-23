/**
 * 移动形态的外壳：顶栏 + 抽屉 + 底栏 + 修饰键行。
 *
 * 【为什么是"另一套外壳"而不是给 MenuBar 加移动样式】桌面顶栏是两层菜单（MenuBar +
 * ActionBar），它的存在前提是"有一根鼠标能精确点中 24px 的菜单项"。移动端上：
 * 这两层加起来 64px 高，占掉 844px 屏幕的 7.6%；菜单项需要二级悬停才能展开，
 * 而触摸没有悬停。把它压扁成一行图标并不会变好 —— 135 条动作里能塞进一行的不到
 * 10 条。因此移动端的外壳形态是**抽屉**：一个汉堡按钮换回全部动作，代价是一次点击。
 *
 * 【为什么顶栏只放四个东西】工程名（用户要知道自己在哪）、撤销、重做、抽屉。
 * 传输控制（播放/停止/录音）在**底栏**——拇指活动区，而且它们是高频操作。
 *
 * 【为什么底栏还有修饰键行】见 `ModifierRail`：没有它，slip 编辑 / 复制拖动 /
 * 范围选择在移动端完全不可达。
 */
import {
    Cross2Icon,
    DotFilledIcon,
    HamburgerMenuIcon,
    PauseIcon,
    PlayIcon,
    StopIcon,
} from "@radix-ui/react-icons";
import { useEffect, useMemo, type ReactNode } from "react";

import { useI18n } from "../i18n/I18nProvider";
import type { ActionId } from "../features/keybindings/types";
import { AppIconButton } from "../ui/Button";
import { cx } from "../ui/cx";
import { ModifierRail } from "./ModifierRail";
import { buildMobileActionSections } from "./mobileActions";

export interface MobileChromeProps {
    /** 工程名（顶栏标题）。 */
    title: string;
    /** 是否有未保存的更改。 */
    dirty: boolean;
    canUndo: boolean;
    canRedo: boolean;
    /** 执行一条动作。由 `App.tsx` 接上它既有的快捷键派发器。 */
    onAction: (actionId: ActionId) => void;
    isPlaying: boolean;
    onTogglePlay: () => void;
    onStop: () => void;
    /** 状态栏内容（与桌面共用同一个节点，只是放在不同位置）。 */
    statusBar?: ReactNode;
}

/** 顶栏。 */
export function MobileTopBar({
    title,
    dirty,
    canUndo,
    canRedo,
    onAction,
    onOpenDrawer,
}: {
    title: string;
    dirty: boolean;
    canUndo: boolean;
    canRedo: boolean;
    onAction: (actionId: ActionId) => void;
    onOpenDrawer: () => void;
}) {
    const { t } = useI18n();
    return (
        <div
            className="flex shrink-0 items-center gap-1 border-b border-qt-border bg-qt-window px-1"
            style={{ height: "var(--qt-bar-main)" }}
        >
            <AppIconButton
                icon={<HamburgerMenuIcon />}
                tooltip={t("menu_view")}
                onClick={onOpenDrawer}
            />
            <div className="min-w-0 flex-1">
                <div className="truncate text-qt-md" data-tooltip={title}>
                    {title}
                    {dirty ? <span aria-hidden> *</span> : null}
                </div>
            </div>
            <AppIconButton
                icon={<span aria-hidden>↶</span>}
                tooltip={t("kb_edit_undo")}
                disabled={!canUndo}
                onClick={() => onAction("edit.undo")}
            />
            <AppIconButton
                icon={<span aria-hidden>↷</span>}
                tooltip={t("kb_edit_redo")}
                disabled={!canRedo}
                onClick={() => onAction("edit.redo")}
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
            <div
                className="flex items-center gap-1 px-1"
                style={{ height: "var(--qt-bar-main)" }}
            >
                <AppIconButton
                    icon={isPlaying ? <PauseIcon /> : <PlayIcon />}
                    tooltip={t("kb_playback_toggle")}
                    active={isPlaying}
                    emphasis="accent"
                    onClick={onTogglePlay}
                />
                <AppIconButton icon={<StopIcon />} tooltip={t("kb_playback_stop")} onClick={onStop} />
                {/*
                 * 录音按钮用实心圆点而不是 `RecordIcon`：Radix 的图标集里没有
                 * "录音"这个语义（`RecordIcon` 不存在），而实心圆点是音频软件里
                 * 通用的录音记号。`intent="danger"` 让它拿到红色。
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
 * 动作抽屉：全屏 sheet，按分组列出全部可用动作。
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
    const sections = useMemo(() => buildMobileActionSections(t, shortcutFor), [t, shortcutFor]);

    // 抽屉打开时禁止背景滚动：否则滑动列表会把底下的画布一起拖动。
    useEffect(() => {
        if (!open) return;
        const previous = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        return () => {
            document.body.style.overflow = previous;
        };
    }, [open]);

    if (!open) return null;
    return (
        <div className="absolute inset-0 z-[var(--qt-z-dialog)] flex flex-col bg-qt-window">
            <div
                className="flex shrink-0 items-center justify-between border-b border-qt-border px-2"
                style={{ height: "var(--qt-bar-main)" }}
            >
                <span className="text-qt-md">{t("menu_view")}</span>
                <AppIconButton icon={<Cross2Icon />} tooltip={t("close")} onClick={onClose} />
            </div>
            <div className="hs-scroll-gutter min-h-0 flex-1 overflow-y-auto">
                {sections.map((section) => (
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
            </div>
        </div>
    );
}
