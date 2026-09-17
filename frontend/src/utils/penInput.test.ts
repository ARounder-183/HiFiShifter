import { describe, expect, it } from "vitest";

import {
    isEraserButton,
    isSecondaryButtonDown,
    isStylusLike,
    PEN_ERASER_BUTTON,
    PEN_ERASER_BUTTONS_MASK,
    pointerKindOf,
    shouldRejectConcurrentPointer,
    shouldSuppressHoverSideEffects,
} from "./penInput";

describe("pointerKindOf", () => {
    it("classifies known pointer types", () => {
        expect(pointerKindOf("mouse")).toBe("mouse");
        expect(pointerKindOf("pen")).toBe("pen");
        expect(pointerKindOf("touch")).toBe("touch");
    });

    it("falls back to unknown for missing / empty / alien values", () => {
        expect(pointerKindOf("")).toBe("unknown");
        expect(pointerKindOf(undefined)).toBe("unknown");
        expect(pointerKindOf(null)).toBe("unknown");
        expect(pointerKindOf("fingerprint")).toBe("unknown");
    });
});

describe("isStylusLike", () => {
    it("accepts explicit pen events", () => {
        expect(isStylusLike({ pointerType: "pen" })).toBe(true);
    });

    it("rejects explicit mouse and touch events", () => {
        expect(isStylusLike({ pointerType: "mouse" })).toBe(false);
        expect(isStylusLike({ pointerType: "touch" })).toBe(false);
    });

    it("does not treat mouse left-button pressure as a stylus", () => {
        // 鼠标按下时 pressure 恒为 0.5，但 pointerType 明确为 mouse，
        // 不走压力探测分支。
        expect(isStylusLike({ pointerType: "mouse", pressure: 0.5 })).toBe(false);
    });

    it("probes pressure / tilt only when pointerType is missing", () => {
        // 合成事件（无类型）：真实笔带压力特征。
        expect(isStylusLike({ pointerType: "", pressure: 0.7 })).toBe(true);
        expect(isStylusLike({ pointerType: "", tiltX: 0.4 })).toBe(true);
        expect(isStylusLike({ pointerType: "", tiltY: -0.3 })).toBe(true);
        // 无类型也无硬件特征 → 按鼠标（宽松回退，合成事件路径不被拦截）。
        expect(isStylusLike({ pointerType: "" })).toBe(false);
        expect(isStylusLike({})).toBe(false);
        expect(isStylusLike({ pointerType: undefined, pressure: 0 })).toBe(false);
    });

    it("handles synthetic modifier-replay events (no pointerType, no pressure)", () => {
        // usePianoRollInteractions 合成的 pointermove 只有坐标与修饰键。
        expect(isStylusLike({ pointerType: undefined, pressure: undefined })).toBe(false);
    });
});

describe("isEraserButton", () => {
    it("recognizes the pen eraser barrel button", () => {
        expect(isEraserButton(PEN_ERASER_BUTTON, "pen")).toBe(true);
        expect(isEraserButton(PEN_ERASER_BUTTON)).toBe(true);
        expect(isEraserButton(PEN_ERASER_BUTTON, "")).toBe(true);
    });

    it("rejects every other button value", () => {
        expect(isEraserButton(-1, "pen")).toBe(false);
        expect(isEraserButton(0, "pen")).toBe(false);
        expect(isEraserButton(1, "pen")).toBe(false);
        expect(isEraserButton(2, "pen")).toBe(false);
    });

    it("rejects non-finite input explicitly", () => {
        expect(isEraserButton(Number.NaN, "pen")).toBe(false);
        expect(isEraserButton(undefined)).toBe(false);
        expect(isEraserButton(null)).toBe(false);
    });

    it("exposes the buttons bitmask counterpart", () => {
        // Chromium：橡皮接触时 buttons 位 32 置位。
        expect(PEN_ERASER_BUTTONS_MASK & (1 << 5)).toBe(PEN_ERASER_BUTTONS_MASK);
    });
});

describe("isSecondaryButtonDown", () => {
    it("maps right button and pen barrel button to secondary", () => {
        expect(isSecondaryButtonDown(2, "mouse")).toBe(true);
        expect(isSecondaryButtonDown(2, "pen")).toBe(true);
        // 合成事件同样放行（宽松回退）。
        expect(isSecondaryButtonDown(2, "")).toBe(true);
        expect(isSecondaryButtonDown(2)).toBe(true);
    });

    it("never maps touch contact to secondary", () => {
        // 触摸永远不会上报 button 2，防御直通驱动。
        expect(isSecondaryButtonDown(2, "touch")).toBe(false);
    });

    it("rejects non-finite input explicitly", () => {
        expect(isSecondaryButtonDown(Number.NaN)).toBe(false);
        expect(isSecondaryButtonDown(undefined)).toBe(false);
    });
});

describe("shouldSuppressHoverSideEffects", () => {
    it("suppresses for pen and touch", () => {
        expect(shouldSuppressHoverSideEffects({ pointerType: "pen" })).toBe(true);
        expect(shouldSuppressHoverSideEffects({ pointerType: "touch" })).toBe(true);
    });

    it("keeps mouse and synthetic events on the legacy path", () => {
        expect(shouldSuppressHoverSideEffects({ pointerType: "mouse" })).toBe(false);
        expect(shouldSuppressHoverSideEffects({ pointerType: "" })).toBe(false);
        expect(shouldSuppressHoverSideEffects({})).toBe(false);
    });
});

describe("shouldRejectConcurrentPointer", () => {
    it("rejects a touch press while a gesture is active (palm rejection)", () => {
        expect(shouldRejectConcurrentPointer({ pointerType: "touch" }, true)).toBe(true);
    });

    it("lets pen / mouse / unknown take over an active gesture", () => {
        expect(shouldRejectConcurrentPointer({ pointerType: "pen" }, true)).toBe(false);
        expect(shouldRejectConcurrentPointer({ pointerType: "mouse" }, true)).toBe(false);
        expect(shouldRejectConcurrentPointer({ pointerType: "" }, true)).toBe(false);
    });

    it("accepts every pointer when no gesture is active", () => {
        expect(shouldRejectConcurrentPointer({ pointerType: "touch" }, false)).toBe(false);
        expect(shouldRejectConcurrentPointer({ pointerType: "pen" }, false)).toBe(false);
    });
});
