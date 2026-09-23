/*
 * 标签右键菜单。
 *
 * 与仓库既有右键菜单同一形态（手搓 `role="menu"` + `fixed` 定位，见
 * `ClipContextMenu` / `EditContextMenu`），不引入 Radix ContextMenu：全应用
 * 的右键菜单样式与关闭语义已经统一，混用两套只会让交互细节分叉。
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
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
}

const MENU_WIDTH = 190;

export function DockTabMenu({ formId, x, y, onClose, onFloat, onCloseForm }: DockTabMenuProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const menuRef = useRef<HTMLDivElement | null>(null);
    const [renaming, setRenaming] = useState(false);
    const [draft, setDraft] = useState("");

    // 视口夹紧：右键点在屏幕右下角时菜单不能跑出可视区。
    const [position, setPosition] = useState({ x, y });
    useLayoutEffect(() => {
        const height = menuRef.current?.offsetHeight ?? 160;
        setPosition({
            x: Math.min(x, Math.max(0, window.innerWidth - MENU_WIDTH - 4)),
            y: Math.min(y, Math.max(0, window.innerHeight - height - 4)),
        });
    }, [x, y, renaming]);

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
