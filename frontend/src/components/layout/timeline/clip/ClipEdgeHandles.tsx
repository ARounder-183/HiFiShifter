import React from "react";
import { isPrimaryModifierDown } from "../../../../utils/platform";

export const ClipEdgeHandles: React.FC<{
    clipId: string;
    altPressed: boolean;
    multiSelectedCount: number;
    isInMultiSelectedSet: boolean;
    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
    onCtrlToggleSelect: (clipId: string) => void;
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
    altPressed,
    multiSelectedCount,
    isInMultiSelectedSet,
    ensureSelected,
    selectClipRemote,
    onCtrlToggleSelect,
    onShiftRangeSelect,
    rangeSelectAnchorClipId,
    recordLastClickPosition,
    seekFromClientX,
    startEditDrag,
}) => {
    const yStyle: React.CSSProperties = {
        top: 0,
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
                    const altKeyDown = Boolean(e.altKey || e.nativeEvent.getModifierState?.("Alt"));
                    const primaryModifierDown = isPrimaryModifierDown(e);
                    const doShiftRangeSelect = e.shiftKey && !altKeyDown && !primaryModifierDown;
                    const shiftRangeAnchorClipId = doShiftRangeSelect
                        ? rangeSelectAnchorClipId
                        : null;
                    const doCtrlToggleOnly = primaryModifierDown && !e.shiftKey && !altKeyDown;
                    const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;

                    const startX = e.clientX;
                    const startY = e.clientY;
                    const pointerId = e.pointerId;
                    const targetEl = e.currentTarget as HTMLElement;
                    const mode = stretchActive ? "stretch_left" : "trim_left";
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

                    const onEnd = (ev: PointerEvent) => {
                        if (ev.pointerId !== pointerId) return;
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
                            seekFromClientX(ev.clientX, true);
                        }
                    };

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
                    const altKeyDown = Boolean(e.altKey || e.nativeEvent.getModifierState?.("Alt"));
                    const primaryModifierDown = isPrimaryModifierDown(e);
                    const doShiftRangeSelect = e.shiftKey && !altKeyDown && !primaryModifierDown;
                    const shiftRangeAnchorClipId = doShiftRangeSelect
                        ? rangeSelectAnchorClipId
                        : null;
                    const doCtrlToggleOnly = primaryModifierDown && !e.shiftKey && !altKeyDown;
                    const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;

                    const startX = e.clientX;
                    const startY = e.clientY;
                    const pointerId = e.pointerId;
                    const targetEl = e.currentTarget as HTMLElement;
                    const mode = stretchActive ? "stretch_right" : "trim_right";
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

                    const onEnd = (ev: PointerEvent) => {
                        if (ev.pointerId !== pointerId) return;
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
                            seekFromClientX(ev.clientX, true);
                        }
                    };

                    window.addEventListener("pointermove", onMove, true);
                    window.addEventListener("pointerup", onEnd, true);
                    window.addEventListener("pointercancel", onEnd, true);
                }}
            />
        </>
    );
};
