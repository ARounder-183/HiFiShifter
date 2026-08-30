import { useEffect, useLayoutEffect, useRef } from "react";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppSelector } from "../../app/hooks";
import { selectKeybinding, formatKeybinding } from "../../features/keybindings/keybindingsSlice";
import type { ActionId } from "../../features/keybindings/types";

/**
 * 读取动作当前生效的快捷键文本（跟随用户在快捷键设置中的自定义绑定）。
 * 未绑定（None binding）时返回 undefined，菜单项不显示快捷键。
 */
function useMenuShortcut(actionId: ActionId): string | undefined {
    const kb = useAppSelector((state) => selectKeybinding(state, actionId));
    return formatKeybinding(kb, "") || undefined;
}

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

    // 菜单项右侧的快捷键提示：从快捷键注册表读取当前生效的绑定。
    // 参数编辑器的复制/剪切/粘贴与时间轴共用 clip.* 动作（焦点在参数
    // 编辑器时由参数编辑器接管，见 useKeyboardShortcuts 的焦点分发）。
    const copyShortcut = useMenuShortcut("clip.copy");
    const cutShortcut = useMenuShortcut("clip.cut");
    const pasteShortcut = useMenuShortcut("clip.paste");
    const selectAllShortcut = useMenuShortcut("edit.selectAll");
    const deselectShortcut = useMenuShortcut("edit.deselect");
    const initializeShortcut = useMenuShortcut("edit.initialize");
    const transposeCentsShortcut = useMenuShortcut("edit.transposeCents");
    const transposeDegreesShortcut = useMenuShortcut("edit.transposeDegrees");
    const setPitchShortcut = useMenuShortcut("edit.setPitch");
    const averageShortcut = useMenuShortcut("edit.average");
    const smoothShortcut = useMenuShortcut("edit.smooth");
    const addVibratoShortcut = useMenuShortcut("edit.addVibrato");
    const quantizeShortcut = useMenuShortcut("edit.quantize");
    const meanQuantizeShortcut = useMenuShortcut("edit.meanQuantize");

    useEffect(() => {
        // 在 window 的捕获阶段监听 pointerdown：目标/冒泡阶段的监听会被
        // 时间轴与钢琴卷帘交互的 stopPropagation 吞掉。例如点击 Clip 时
        // ClipItem 的 onPointerDown 会在 React 根容器上 stopPropagation，
        // 事件根本到不了 document —— 这是此前"点击轨道/Clip 菜单不消失"
        // 的根源。捕获阶段在一切目标处理器之前运行，任何 stopPropagation
        // 都无法阻断（与时间轴 Clip 菜单、轨道列表菜单、标尺菜单的关闭
        // 方式一致）。
        function handlePointerDownOutside(e: PointerEvent) {
            if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
                onClose();
            }
        }
        function handleEsc(e: KeyboardEvent) {
            if (e.key === "Escape") onClose();
        }
        window.addEventListener("pointerdown", handlePointerDownOutside, true);
        window.addEventListener("keydown", handleEsc, true);
        return () => {
            window.removeEventListener("pointerdown", handlePointerDownOutside, true);
            window.removeEventListener("keydown", handleEsc, true);
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
        "px-3 py-1.5 text-left w-full text-[12px] transition-colors cursor-pointer hover:bg-qt-button-hover select-none text-qt-text flex items-center justify-between gap-3";
    const shortcutClass = "text-[10px] opacity-50 shrink-0";
    const sepClass = "my-1 border-t border-qt-border";

    const item = (label: string, shortcut: string | undefined, onClick: () => void) => (
        <div className={itemClass} onClick={onClick}>
            <span>{label}</span>
            {shortcut && <span className={shortcutClass}>{shortcut}</span>}
        </div>
    );
    const closeAfter = (action?: () => void) => () => {
        action?.();
        onClose();
    };

    return (
        <div
            ref={menuRef}
            data-hs-context-menu="1"
            className="fixed z-[9999] min-w-[180px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
        >
            {item(tAny("menu_copy"), copyShortcut, closeAfter(onCopy))}
            {item(tAny("menu_cut"), cutShortcut, closeAfter(onCut))}
            {item(tAny("menu_paste"), pasteShortcut, closeAfter(onPaste))}
            <div className={sepClass} />
            {item(tAny("menu_select_all"), selectAllShortcut, closeAfter(onSelectAll))}
            {item(tAny("menu_deselect"), deselectShortcut, closeAfter(onDeselect))}
            <div className={sepClass} />
            {item(tAny("menu_initialize"), initializeShortcut, closeAfter(onInitialize))}
            {isPitchParam && (
                <>
                    <div className={sepClass} />
                    {item(
                        tAny("menu_transpose_cents"),
                        transposeCentsShortcut,
                        closeAfter(onTransposeCents),
                    )}
                    {item(
                        tAny("menu_transpose_degrees"),
                        transposeDegreesShortcut,
                        closeAfter(onTransposeDegrees),
                    )}
                </>
            )}
            {item(
                isPitchParam ? tAny("menu_set_pitch") : tAny("menu_set_value"),
                setPitchShortcut,
                closeAfter(onSetPitch),
            )}
            <div className={sepClass} />
            {item(tAny("menu_average"), averageShortcut, closeAfter(onAverage))}
            {item(tAny("menu_smooth"), smoothShortcut, closeAfter(onSmooth))}
            {item(tAny("menu_add_vibrato"), addVibratoShortcut, closeAfter(onAddVibrato))}
            {item(tAny("menu_quantize"), quantizeShortcut, closeAfter(onQuantize))}
            {item(tAny("menu_mean_quantize"), meanQuantizeShortcut, closeAfter(onMeanQuantize))}
            {isPitchParam && onSaveAsPitchRef && (
                <>
                    <div className={sepClass} />
                    {item(tAny("menu_save_as_pitch_ref"), undefined, closeAfter(onSaveAsPitchRef))}
                    {onExportMidi &&
                        item(tAny("menu_export_midi"), undefined, closeAfter(onExportMidi))}
                </>
            )}
        </div>
    );
}
