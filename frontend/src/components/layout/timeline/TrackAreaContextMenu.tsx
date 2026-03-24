import React, { useLayoutEffect, useRef } from "react";
import { useI18n } from "../../../i18n/I18nProvider";

const MenuItem: React.FC<{
    label: string;
    disabled?: boolean;
    onClick: () => void;
}> = ({ label, disabled, onClick }) => (
    <button
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
    onPaste: () => void;
    onSplit: () => void;
    onClose: () => void;
}> = ({ x, y, canPaste, canSplit, onPaste, onSplit, onClose }) => {
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

    return (
        <div
            ref={menuRef}
            data-hs-context-menu="1"
            className="fixed z-50 min-w-[150px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
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
        </div>
    );
};
