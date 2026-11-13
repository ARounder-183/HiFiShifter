import { describe, expect, it, vi } from "vitest";

import { handleCanvasHoverPreview, updateHoverParamValuePreview } from "./canvasPointerMove";

function event(clientX = 20, clientY = 30, pointerType = "mouse") {
    return {
        clientX,
        clientY,
        nativeEvent: { pointerType },
    } as never;
}

describe("hover parameter value preview", () => {
    it("publishes the curve value near the pointer and records hover state", () => {
        const onParamValuePreviewChange = vi.fn();
        const hoverPreviewNearCurveRef = { current: false };
        const lastPointerClientRef = { current: null as { x: number; y: number } | null };
        const deps = {
            getCurveValueNearPointer: vi.fn(() => 60.5),
            hoverPreviewNearCurveRef,
            lastPointerClientRef,
            onParamValuePreviewChange,
            paramValuePopupEnabled: true,
        };
        handleCanvasHoverPreview(deps, event());
        expect(onParamValuePreviewChange).toHaveBeenCalledWith({
            clientX: 20,
            clientY: 30,
            value: 60.5,
        });
        expect(hoverPreviewNearCurveRef.current).toBe(true);
        expect(lastPointerClientRef.current).toEqual({ x: 20, y: 30 });
    });

    it("publishes null when the pointer is away from the curve", () => {
        const onParamValuePreviewChange = vi.fn();
        const hoverPreviewNearCurveRef = { current: true };
        updateHoverParamValuePreview(
            {
                getCurveValueNearPointer: () => null,
                hoverPreviewNearCurveRef,
                lastPointerClientRef: { current: null },
                onParamValuePreviewChange,
                paramValuePopupEnabled: true,
            },
            event(),
        );
        expect(onParamValuePreviewChange).toHaveBeenCalledWith(null);
        expect(hoverPreviewNearCurveRef.current).toBe(false);
    });

    it("does nothing when the popup is disabled", () => {
        const onParamValuePreviewChange = vi.fn();
        const hoverPreviewNearCurveRef = { current: true };
        updateHoverParamValuePreview(
            {
                getCurveValueNearPointer: vi.fn(() => 1),
                hoverPreviewNearCurveRef,
                lastPointerClientRef: { current: null },
                onParamValuePreviewChange,
                paramValuePopupEnabled: false,
            },
            event(),
        );
        expect(onParamValuePreviewChange).not.toHaveBeenCalled();
        expect(hoverPreviewNearCurveRef.current).toBe(true);
    });
});
