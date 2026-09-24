/*
 * 标签右键菜单。
 *
 * 与仓库既有右键菜单同一形态（手搓 `role="menu"` + `fixed` 定位，见
 * `ClipContextMenu` / `EditContextMenu`），不引入 Radix ContextMenu：全应用
 * 的右键菜单样式与关闭语义已经统一，混用两套只会让交互细节分叉。
 */

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useAppDispatch } from "../../app/hooks";
import { renameForm } from "../../features/dock/dockSlice";
import { useI18n } from "../../i18n/I18nProvider";

export interface DockTabMenuProps {
    formId: string;
    x: number;
    y: number;
    onClose: () => void;
    onFloat: () => void;
    onCloseForm: () => void;
    /**
     * 拆到独立窗口 / 从独立窗口收回。
     *
     * `null` 表示当前窗格不支持（面板未声明 `detachable`，或已经是独立窗口且
     * 收回入口由那个窗口自己的标题栏提供）—— 此时不渲染该项，而不是给一个点了
     * 没反应的按钮。
     */
    detachAction?: { labelKey: string; run: () => void } | null;
}

const MENU_WIDTH = 190;
/** 单个菜单项的高度（含内边距）。 */
const MENU_ITEM_PX = 24;
/** 菜单的纵向内边距 + 分隔线。 */
const MENU_CHROME_PX = 18;

export function DockTabMenu({
    formId,
    x,
    y,
    onClose,
    onFloat,
    onCloseForm,
    detachAction,
}: DockTabMenuProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const menuRef = useRef<HTMLDivElement | null>(null);
    const [renaming, setRenaming] = useState(false);
    const [draft, setDraft] = useState("");

    // 视口夹紧：右键点在屏幕右下角时菜单不能跑出可视区。
    //
    // 用**固定估算高度**而不是"先渲染再测量"：后者要在 layout effect 里同步
    // setState（触发级联渲染，React Compiler 会就此告警），而菜单项高度本来就是
    // 确定的常量。估算偏差最多几个像素，视觉上不可见。
    const itemCount = renaming ? 1 : 2 + (detachAction ? 1 : 0);
    const estimatedHeight = itemCount * MENU_ITEM_PX + MENU_CHROME_PX;
    const position = {
        x: Math.min(x, Math.max(0, window.innerWidth - MENU_WIDTH - 4)),
        y: Math.min(y, Math.max(0, window.innerHeight - estimatedHeight - 4)),
    };

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                event.stopPropagation();
                onClose();
            }
        };
        const onPointerDown = (event: PointerEvent) => {
            if (menuRef.current?.contains(event.target as Node)) return;
            onClose();
        };
        window.addEventListener("keydown", onKeyDown, true);
        window.addEventListener("pointerdown", onPointerDown, true);
        return () => {
            window.removeEventListener("keydown", onKeyDown, true);
            window.removeEventListener("pointerdown", onPointerDown, true);
        };
    }, [onClose]);

    return createPortal(
        <div
            ref={menuRef}
            role="menu"
            data-hs-context-menu="1"
            className="fixed z-[9999] min-w-[190px] rounded border border-qt-border bg-qt-window py-1 text-qt-text shadow-lg"
            style={{ left: position.x, top: position.y }}
            onPointerDown={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.preventDefault()}
        >
            {renaming ? (
                <div className="px-2 py-1">
                    <input
                        autoFocus
                        value={draft}
                        onChange={(event) => setDraft(event.target.value)}
                        onKeyDown={(event) => {
                            if (event.key === "Enter") {
                                dispatch(renameForm({ formId, title: draft }));
                                onClose();
                            }
                            if (event.key === "Escape") setRenaming(false);
                        }}
                        onBlur={() => {
                            dispatch(renameForm({ formId, title: draft }));
                            onClose();
                        }}
                        className="w-full rounded border border-qt-border bg-qt-base px-1 py-0.5 text-xs text-qt-text outline-none"
                    />
                </div>
            ) : (
                <MenuItem
                    label={tAny("dock_rename_tab")}
                    onClick={() => {
                        setDraft("");
                        setRenaming(true);
                    }}
                />
            )}
            <MenuItem label={tAny("dock_float")} onClick={onFloat} />
            {detachAction ? (
                <MenuItem
                    label={tAny(detachAction.labelKey)}
                    onClick={() => {
                        detachAction.run();
                        onClose();
                    }}
                />
            ) : null}
            <div className="my-1 border-t border-qt-border" />
            <MenuItem
                label={t("close")}
                danger
                onClick={() => {
                    onCloseForm();
                    onClose();
                }}
            />
        </div>,
        document.body,
    );
}

function MenuItem({
    label,
    onClick,
    danger,
}: {
    label: string;
    onClick: () => void;
    danger?: boolean;
}) {
    return (
        <button
            type="button"
            role="menuitem"
            className={`block w-full px-3 py-1 text-left text-xs ${
                danger
                    ? "hover:bg-qt-danger-bg hover:text-qt-danger-text"
                    : "hover:bg-qt-highlight hover:text-white"
            }`}
            onClick={onClick}
        >
            {label}
        </button>
    );
}
