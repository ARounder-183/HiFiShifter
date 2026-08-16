import { useEffect, useLayoutEffect, useRef } from "react";
import { useI18n } from "../../i18n/I18nProvider";

interface EditContextMenuProps {
    x: number;
    y: number;
    isPitchParam: boolean;
    onClose: () => void;
    onCopy?: () => void;
    onCut?: () => void;
    onPaste?: () => void;
    onSelectAll?: () => void;
    onDeselect?: () => void;
    onInitialize?: () => void;
    onTransposeCents?: () => void;
    onTransposeDegrees?: () => void;
    onSetPitch?: () => void;
    onAverage?: () => void;
    onSmooth?: () => void;
    onAddVibrato?: () => void;
    onQuantize?: () => void;
    onMeanQuantize?: () => void;
    onSaveAsPitchRef?: () => void;
    onExportMidi?: () => void;
}

export function EditContextMenu({
    x,
    y,
    isPitchParam,
    onClose,
    onCopy,
    onCut,
    onPaste,
    onSelectAll,
    onDeselect,
    onInitialize,
    onTransposeCents,
    onTransposeDegrees,
    onSetPitch,
    onAverage,
    onSmooth,
    onAddVibrato,
    onQuantize,
    onMeanQuantize,
    onSaveAsPitchRef,
    onExportMidi,
}: EditContextMenuProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const menuRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClickOutside(e: MouseEvent) {
            if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
                onClose();
            }
        }
        function handleEsc(e: KeyboardEvent) {
            if (e.key === "Escape") onClose();
        }
        document.addEventListener("mousedown", handleClickOutside);
        document.addEventListener("keydown", handleEsc);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
            document.removeEventListener("keydown", handleEsc);
        };
    }, [onClose]);

    // Clamp menu position to viewport edges
    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        let clampedX = x;
        let clampedY = y;
        if (rect.right > vw) clampedX = Math.max(0, vw - rect.width);
        if (rect.bottom > vh) clampedY = Math.max(0, vh - rect.height);
        el.style.left = `${clampedX}px`;
        el.style.top = `${clampedY}px`;
    }, [x, y]);

    const itemClass =
        "px-3 py-1.5 text-left w-full text-[12px] transition-colors cursor-pointer hover:bg-qt-button-hover select-none text-qt-text";
    const sepClass = "my-1 border-t border-qt-border";

    return (
        <div
            ref={menuRef}
            data-hs-context-menu="1"
            className="fixed z-[9999] min-w-[180px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
        >
            <div
                className={itemClass}
                onClick={() => {
                    onCopy?.();
                    onClose();
                }}
            >
                {tAny("menu_copy")}
            </div>
            <div
                className={itemClass}
                onClick={() => {
                    onCut?.();
                    onClose();
                }}
            >
                {tAny("menu_cut")}
            </div>
            <div
                className={itemClass}
                onClick={() => {
                    onPaste?.();
                    onClose();
                }}
            >
                {tAny("menu_paste")}
            </div>
            <div className={sepClass} />
            <div
                className={itemClass}
                onClick={() => {
                    onSelectAll?.();
                    onClose();
                }}
            >
                {tAny("menu_select_all")}
            </div>
            <div
                className={itemClass}
                onClick={() => {
                    onDeselect?.();
                    onClose();
                }}
            >
                {tAny("menu_deselect")}
            </div>
            <div className={sepClass} />
            <div
                className={itemClass}
                onClick={() => {
                    onInitialize?.();
                    onClose();
                }}
            >
                {tAny("menu_initialize")}
            </div>
            {isPitchParam && (
                <>
                    <div className={sepClass} />
                    <div
                        className={itemClass}
                        onClick={() => {
                            onTransposeCents?.();
                            onClose();
                        }}
                    >
                        {tAny("menu_transpose_cents")}
                    </div>
                    <div
                        className={itemClass}
                        onClick={() => {
                            onTransposeDegrees?.();
                            onClose();
                        }}
                    >
                        {tAny("menu_transpose_degrees")}
                    </div>
                </>
            )}
            <div
                className={itemClass}
                onClick={() => {
                    onSetPitch?.();
                    onClose();
                }}
            >
                {isPitchParam ? tAny("menu_set_pitch") : tAny("menu_set_value")}
            </div>
            <div className={sepClass} />
            <div
                className={itemClass}
                onClick={() => {
                    onAverage?.();
                    onClose();
                }}
            >
                {tAny("menu_average")}
            </div>
            <div
                className={itemClass}
                onClick={() => {
                    onSmooth?.();
                    onClose();
                }}
            >
                {tAny("menu_smooth")}
            </div>
            <div
                className={itemClass}
                onClick={() => {
                    onAddVibrato?.();
                    onClose();
                }}
            >
                {tAny("menu_add_vibrato")}
            </div>
            <div
                className={itemClass}
                onClick={() => {
                    onQuantize?.();
                    onClose();
                }}
            >
                {tAny("menu_quantize")}
            </div>
            <div
                className={itemClass}
                onClick={() => {
                    onMeanQuantize?.();
                    onClose();
                }}
            >
                {tAny("menu_mean_quantize")}
            </div>
            {isPitchParam && onSaveAsPitchRef && (
                <>
                    <div className={sepClass} />
                    <div
                        className={itemClass}
                        onClick={() => {
                            onSaveAsPitchRef();
                            onClose();
                        }}
                    >
                        {tAny("menu_save_as_pitch_ref")}
                    </div>
                    {onExportMidi && (
                        <div
                            className={itemClass}
                            onClick={() => {
                                onExportMidi();
                                onClose();
                            }}
                        >
                            {tAny("menu_export_midi")}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
