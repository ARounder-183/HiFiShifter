import React, { useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../../../i18n/I18nProvider";

const MenuItem: React.FC<{
    label: string;
    disabled?: boolean;
    /** 悬停 / 禁用原因提示（如“点击位置之后没有 Clip”）。 */
    title?: string;
    onClick: () => void;
}> = ({ label, disabled, title, onClick }) => (
    <button
        title={title}
        className={`px-3 py-1.5 text-left w-full text-[12px] transition-colors flex items-center justify-between gap-3 ${
            disabled ? "opacity-40 cursor-default" : "hover:bg-qt-button-hover"
        }`}
        disabled={disabled}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
            e.stopPropagation();
            onClick();
        }}
    >
        <span>{label}</span>
    </button>
);

export const TrackAreaContextMenu: React.FC<{
    x: number;
    y: number;
    canPaste: boolean;
    canSplit: boolean;
    /** 点击位置之后该轨道上是否还有 Clip（关闭间隙的启用条件）。 */
    canCloseGaps: boolean;
    onPaste: () => void;
    onSplit: () => void;
    onCloseGaps: () => void;
    onClose: () => void;
}> = ({ x, y, canPaste, canSplit, canCloseGaps, onPaste, onSplit, onCloseGaps, onClose }) => {
    const { t } = useI18n();
    const menuRef = useRef<HTMLDivElement>(null);

    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw) {
            el.style.left = `${Math.max(0, vw - rect.width)}px`;
        }
        if (rect.bottom > vh) {
            el.style.top = `${Math.max(0, vh - rect.height)}px`;
        }
    }, [x, y]);

    return createPortal(
        <div
            ref={menuRef}
            data-hs-context-menu="1"
            data-hs-floating-menu="1"
            className="fixed z-[999] min-w-[150px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
        >
            <MenuItem
                label={t("menu_paste")}
                disabled={!canPaste}
                onClick={() => {
                    onPaste();
                    onClose();
                }}
            />
            <MenuItem
                label={t("ctx_split_at_playhead")}
                disabled={!canSplit}
                onClick={() => {
                    onSplit();
                    onClose();
                }}
            />
            <MenuItem
                label={t("ctx_close_gaps")}
                disabled={!canCloseGaps}
                title={canCloseGaps ? undefined : t("ctx_close_gaps_disabled")}
                onClick={() => {
                    onCloseGaps();
                    onClose();
                }}
            />
        </div>,
        document.body,
    );
};
