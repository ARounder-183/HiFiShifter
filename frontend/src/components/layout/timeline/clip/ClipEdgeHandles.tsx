import React from "react";
import { registerDragAbort } from "../gestureFocusGuard";
import { resolveClipSelectionModifiers } from "../../../../features/keybindings/clipSelectionModifiers";
import { DEFAULT_KEYBINDINGS } from "../../../../features/keybindings/defaultKeybindings";
import type { Keybinding } from "../../../../features/keybindings/types";
import { CLIP_HEADER_HEIGHT, fadeCornerReservePx } from "../constants";

export const ClipEdgeHandles: React.FC<{
    clipId: string;
    /** Clip body 高（px）：与 ClipItem 的淡化角控件同源，决定顶部保留区。 */
    bodyHeightPx: number;
    altPressed: boolean;
    multiSelectedCount: number;
    isInMultiSelectedSet: boolean;
    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
    onCtrlToggleSelect: (clipId: string) => void;
    /** modifier.clipMultiSelectToggle 绑定（按住并点击切换多选） */
    multiSelectToggleKb?: Keybinding;
    /** modifier.clipRangeSelect 绑定（按住并点击范围选择） */
    rangeSelectKb?: Keybinding;
    onShiftRangeSelect: (
        clipId: string,
        anchorClipIdOverride?: string | null,
        targetClientX?: number,
    ) => void;
    rangeSelectAnchorClipId: string | null;
    recordLastClickPosition?: (clientX: number) => void;
    seekFromClientX: (clientX: number, commit: boolean) => void;
    startEditDrag: (
        e: React.PointerEvent,
        clipId: string,
        type: "trim_left" | "trim_right" | "stretch_left" | "stretch_right",
    ) => void;
}> = ({
    clipId,
    bodyHeightPx,
    altPressed,
    multiSelectedCount,
    isInMultiSelectedSet,
    ensureSelected,
    selectClipRemote,
    onCtrlToggleSelect,
    multiSelectToggleKb = DEFAULT_KEYBINDINGS["modifier.clipMultiSelectToggle"],
    rangeSelectKb = DEFAULT_KEYBINDINGS["modifier.clipRangeSelect"],
    onShiftRangeSelect,
    rangeSelectAnchorClipId,
    recordLastClickPosition,
    seekFromClientX,
    startEditDrag,
}) => {
    // 左右边缘的垂直所有权切分：顶部一段让给淡入/淡出角部控件（ClipItem
    // 渲染），裁短/延长只从其下沿开始 —— 几何上互不重叠，任何层叠顺序下都
    // 不会互相抢事件。
    //
    // 保留高度随 Clip 高度自适应（fadeCornerReservePx，body 1/3）：固定值
    // 会让角控件在矮 Clip 上吃掉大半个边缘，裁短手势无从下手。**与 ClipItem
    // 的角控件用同一个 body 基准**：header 之下从 body 顶开始保留 —— 角区
    // [header, header+reserve) 与裁短区 [header+reserve, 底部] 精确拼接。
    const yStyle: React.CSSProperties = {
        top: CLIP_HEADER_HEIGHT + fadeCornerReservePx(bodyHeightPx),
        bottom: 0,
    };

    return (
        <>
            {/* Left/Right edge handles (trim or time-stretch). Extend into the header area. */}
            <div
                className="absolute left-0 w-[10px] z-[60] opacity-0 group-hover:opacity-100 transition-opacity"
                style={{
                    ...yStyle,
                    cursor: altPressed ? "col-resize" : "ew-resize",
                }}
                onPointerDown={(e) => {
                    if (e.button !== 0) return;
                    e.preventDefault();
                    e.stopPropagation();

                    // altPressed tracks the stretch modifier (configurable) — use it
                    // for edit-mode selection (stretch vs trim) and cursor display.
                    // For click-selection bypass, only check the physical Alt key
                    // to avoid breaking Ctrl/Shift selection when those keys are
                    // configured as stretch modifiers.
                    const stretchActive = altPressed;
                    const selectionMods = resolveClipSelectionModifiers({
                        event: e,
                        multiSelectToggleKb,
                        rangeSelectKb,
                    });
                    const doShiftRangeSelect = selectionMods.rangeSelectActive;
                    const shiftRangeAnchorClipId = doShiftRangeSelect
                        ? rangeSelectAnchorClipId
                        : null;
                    const doCtrlToggleOnly = selectionMods.multiSelectToggleActive;
                    const shouldPrimeSelection = selectionMods.shouldPrimeSelection;

                    const startX = e.clientX;
                    const startY = e.clientY;
                    const pointerId = e.pointerId;
                    const targetEl = e.currentTarget as HTMLElement;
                    const mode = stretchActive ? "stretch_left" : "trim_left";
                    // 单击（未拖动）松开时，播放头跳到该边缘的准确位置（左缘）。
                    const seekToEdgeClientX = () => targetEl.getBoundingClientRect().left;
                    let dragStarted = false;

                    const onMove = (ev: PointerEvent) => {
                        if (ev.pointerId !== pointerId || dragStarted) return;
                        const dx = ev.clientX - startX;
                        const dy = ev.clientY - startY;
                        if (dx * dx + dy * dy < 9) return;
                        dragStarted = true;
                        startEditDrag(
                            {
                                button: 0,
                                pointerId,
                                currentTarget: targetEl,
                            } as unknown as React.PointerEvent,
                            clipId,
                            mode,
                        );
                    };

                    // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，blur
                    // 时走与 onEnd 相同的收尾（真正的裁短/拉伸拖拽由 useEditDrag
                    // 自身的失焦守卫收尾并提交；此处只负责点击语义与监听清理）。
                    let finished = false;
                    const finish = () => {
                        if (finished) return;
                        finished = true;
                        unregisterAbort();
                        window.removeEventListener("pointermove", onMove, true);
                        window.removeEventListener("pointerup", onEnd, true);
                        window.removeEventListener("pointercancel", onEnd, true);
                        if (!dragStarted) {
                            if (doCtrlToggleOnly) {
                                onCtrlToggleSelect(clipId);
                                return;
                            }
                            if (doShiftRangeSelect) {
                                onShiftRangeSelect(clipId, shiftRangeAnchorClipId, startX);
                                return;
                            }
                            if (shouldPrimeSelection) {
                                if (!isInMultiSelectedSet || multiSelectedCount > 1) {
                                    ensureSelected(clipId);
                                }
                                selectClipRemote(clipId);
                                recordLastClickPosition?.(e.clientX);
                            }
                            seekFromClientX(seekToEdgeClientX(), true);
                        }
                    };
                    const onEnd = (ev: PointerEvent) => {
                        if (ev.pointerId !== pointerId) return;
                        finish();
                    };
                    const unregisterAbort = registerDragAbort(finish);

                    window.addEventListener("pointermove", onMove, true);
                    window.addEventListener("pointerup", onEnd, true);
                    window.addEventListener("pointercancel", onEnd, true);
                }}
            />
            <div
                className="absolute right-0 w-[10px] z-[60] opacity-0 group-hover:opacity-100 transition-opacity"
                style={{
                    ...yStyle,
                    cursor: altPressed ? "col-resize" : "ew-resize",
                }}
                onPointerDown={(e) => {
                    if (e.button !== 0) return;
                    e.preventDefault();
                    e.stopPropagation();

                    // Same separation as left edge: stretchActive for edit mode,
                    // altKeyDown for click-selection bypass only.
                    const stretchActive = altPressed;
                    const selectionMods = resolveClipSelectionModifiers({
                        event: e,
                        multiSelectToggleKb,
                        rangeSelectKb,
                    });
                    const doShiftRangeSelect = selectionMods.rangeSelectActive;
                    const shiftRangeAnchorClipId = doShiftRangeSelect
                        ? rangeSelectAnchorClipId
                        : null;
                    const doCtrlToggleOnly = selectionMods.multiSelectToggleActive;
                    const shouldPrimeSelection = selectionMods.shouldPrimeSelection;

                    const startX = e.clientX;
                    const startY = e.clientY;
                    const pointerId = e.pointerId;
                    const targetEl = e.currentTarget as HTMLElement;
                    const mode = stretchActive ? "stretch_right" : "trim_right";
                    // 单击（未拖动）松开时，播放头跳到该边缘的准确位置（右缘）。
                    const seekToEdgeClientX = () => targetEl.getBoundingClientRect().right;
                    let dragStarted = false;

                    const onMove = (ev: PointerEvent) => {
                        if (ev.pointerId !== pointerId || dragStarted) return;
                        const dx = ev.clientX - startX;
                        const dy = ev.clientY - startY;
                        if (dx * dx + dy * dy < 9) return;
                        dragStarted = true;
                        startEditDrag(
                            {
                                button: 0,
                                pointerId,
                                currentTarget: targetEl,
                            } as unknown as React.PointerEvent,
                            clipId,
                            mode,
                        );
                    };

                    // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，blur
                    // 时走与 onEnd 相同的收尾（真正的裁短/拉伸拖拽由 useEditDrag
                    // 自身的失焦守卫收尾并提交；此处只负责点击语义与监听清理）。
                    let finished = false;
                    const finish = () => {
                        if (finished) return;
                        finished = true;
                        unregisterAbort();
                        window.removeEventListener("pointermove", onMove, true);
                        window.removeEventListener("pointerup", onEnd, true);
                        window.removeEventListener("pointercancel", onEnd, true);
                        if (!dragStarted) {
                            if (doCtrlToggleOnly) {
                                onCtrlToggleSelect(clipId);
                                return;
                            }
                            if (doShiftRangeSelect) {
                                onShiftRangeSelect(clipId, shiftRangeAnchorClipId, startX);
                                return;
                            }
                            if (shouldPrimeSelection) {
                                if (!isInMultiSelectedSet || multiSelectedCount > 1) {
                                    ensureSelected(clipId);
                                }
                                selectClipRemote(clipId);
                                recordLastClickPosition?.(e.clientX);
                            }
                            seekFromClientX(seekToEdgeClientX(), true);
                        }
                    };
                    const onEnd = (ev: PointerEvent) => {
                        if (ev.pointerId !== pointerId) return;
                        finish();
                    };
                    const unregisterAbort = registerDragAbort(finish);

                    window.addEventListener("pointermove", onMove, true);
                    window.addEventListener("pointerup", onEnd, true);
                    window.addEventListener("pointercancel", onEnd, true);
                }}
            />
        </>
    );
};
