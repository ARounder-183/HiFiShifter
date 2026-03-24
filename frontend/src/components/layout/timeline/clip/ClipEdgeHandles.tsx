import React from "react";

export const ClipEdgeHandles: React.FC<{
    clipId: string;
    altPressed: boolean;
    multiSelectedCount: number;
    isInMultiSelectedSet: boolean;
    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
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
                    if (multiSelectedCount === 0 || !isInMultiSelectedSet) {
                        ensureSelected(clipId);
                    }
                    selectClipRemote(clipId);

                    const startX = e.clientX;
                    const startY = e.clientY;
                    const pointerId = e.pointerId;
                    const targetEl = e.currentTarget as HTMLElement;
                    const mode = altPressed ? "stretch_left" : "trim_left";
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
                    if (multiSelectedCount === 0 || !isInMultiSelectedSet) {
                        ensureSelected(clipId);
                    }
                    selectClipRemote(clipId);

                    const startX = e.clientX;
                    const startY = e.clientY;
                    const pointerId = e.pointerId;
                    const targetEl = e.currentTarget as HTMLElement;
                    const mode = altPressed ? "stretch_right" : "trim_right";
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
