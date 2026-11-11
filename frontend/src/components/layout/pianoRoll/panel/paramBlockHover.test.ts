import { describe, expect, it, vi } from "vitest";
import type { PointerEvent as ReactPointerEvent } from "react";
import { createTimelineAxis } from "../../renderKernel/timelineAxis.js";
import type { PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";
import type { ParamBlock } from "../paramBlocks";
import { handleParamBlockHover, type ParamBlockHoverDeps } from "./paramBlockHover";

type HitRect = ReturnType<PianoRollKernelHost["getParamBlockHitRects"]>[number];

/** 一条参数线：块 a = 0..1s，块 b = 1..2s，两者相接（所以 a 上有过渡手柄）。 */
const HIT_RECTS: readonly HitRect[] = [
    {
        id: "a",
        startSec: 0,
        endSec: 1,
        topPx: 0,
        heightPx: 20,
        averageValue: 60,
        centerYPx: 10,
        selected: true,
        transitionHandle: { sec: 1, leftId: "a", rightId: "b", widthSec: 0.01 },
    },
    {
        id: "b",
        startSec: 1,
        endSec: 2,
        topPx: 0,
        heightPx: 20,
        averageValue: 62,
        centerYPx: 10,
        selected: false,
        transitionHandle: null,
    },
];

function block(id: string, startSec: number, endSec: number): ParamBlock {
    return {
        id,
        rootTrackId: "root",
        kind: "auto",
        members: [],
        startSec,
        endSec,
        averageValue: 60,
        manualValue: null,
    };
}

function pointerEvent(args: {
    readonly clientX: number;
    readonly clientY: number;
    readonly buttons?: number;
}): ReactPointerEvent<HTMLCanvasElement> {
    return {
        clientX: args.clientX,
        clientY: args.clientY,
        buttons: args.buttons ?? 0,
        currentTarget: { getBoundingClientRect: () => ({ left: 0, top: 0 }) },
    } as unknown as ReactPointerEvent<HTMLCanvasElement>;
}

function makeDeps(overrides: Partial<ParamBlockHoverDeps> = {}) {
    const hoveredParamBlockIdRef = { current: null as string | null };
    const hoveredTransitionLeftIdRef = { current: null as string | null };
    const setCanvasCursor = vi.fn();
    const invalidate = vi.fn();
    const host = {
        getAxis: () => createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0 }),
        getParamBlockHitRects: () => HIT_RECTS,
    } as unknown as PianoRollKernelHost;
    const deps: ParamBlockHoverDeps = {
        paramBlockToolActive: true,
        paramBlockHitSlopPx: 4,
        paramBlocksWithValues: [block("a", 0, 1), block("b", 1, 2)],
        hostRef: { current: host },
        hoveredParamBlockIdRef,
        hoveredTransitionLeftIdRef,
        createGestureActive: false,
        getDefaultCanvasCursor: () => "crosshair",
        setCanvasCursor,
        invalidate,
        ...overrides,
    };
    return {
        deps,
        hoveredParamBlockIdRef,
        hoveredTransitionLeftIdRef,
        setCanvasCursor,
        invalidate,
    };
}

describe("handleParamBlockHover", () => {
    it("returns false outside block selection so the generic handler still runs", () => {
        const { deps, setCanvasCursor } = makeDeps({ paramBlockToolActive: false });
        expect(handleParamBlockHover(deps, pointerEvent({ clientX: 50, clientY: 10 }))).toBe(false);
        expect(setCanvasCursor).not.toHaveBeenCalled();
    });

    it("claims blank canvas in block mode and restores the tool default cursor", () => {
        const { deps, hoveredParamBlockIdRef, setCanvasCursor } = makeDeps();
        hoveredParamBlockIdRef.current = "a";
        expect(handleParamBlockHover(deps, pointerEvent({ clientX: 500, clientY: 10 }))).toBe(true);
        expect(setCanvasCursor).toHaveBeenLastCalledWith("crosshair");
        /* 空白必须清掉 hover，否则块会一直带着上一次的高亮。 */
        expect(hoveredParamBlockIdRef.current).toBeNull();
    });

    it("claims the block body and marks it hovered", () => {
        const { deps, hoveredParamBlockIdRef, setCanvasCursor, invalidate } = makeDeps();
        expect(handleParamBlockHover(deps, pointerEvent({ clientX: 50, clientY: 10 }))).toBe(true);
        expect(setCanvasCursor).toHaveBeenLastCalledWith("grab");
        expect(hoveredParamBlockIdRef.current).toBe("a");
        expect(invalidate).toHaveBeenCalled();
    });

    it("claims the block edge and shows the resize cursor", () => {
        const { deps, setCanvasCursor } = makeDeps();
        expect(handleParamBlockHover(deps, pointerEvent({ clientX: 2, clientY: 10 }))).toBe(true);
        expect(setCanvasCursor).toHaveBeenLastCalledWith("ew-resize");
    });

    it("claims the transition handle and keys it by the left block id", () => {
        const { deps, hoveredParamBlockIdRef, hoveredTransitionLeftIdRef, setCanvasCursor } =
            makeDeps();
        /* 手柄圆点在块右边界正上方 6px（outsetY），命中区 ±7px。 */
        expect(handleParamBlockHover(deps, pointerEvent({ clientX: 100, clientY: -6 }))).toBe(true);
        expect(setCanvasCursor).toHaveBeenLastCalledWith("ew-resize");
        expect(hoveredTransitionLeftIdRef.current).toBe("a");
        expect(hoveredParamBlockIdRef.current).toBe("a");
    });

    it("keeps the drag cursor while a gesture is running", () => {
        const grabbing = makeDeps();
        expect(
            handleParamBlockHover(
                grabbing.deps,
                pointerEvent({ clientX: 500, clientY: 10, buttons: 1 }),
            ),
        ).toBe(true);
        expect(grabbing.setCanvasCursor).toHaveBeenLastCalledWith("grabbing");

        const creating = makeDeps({ createGestureActive: true });
        handleParamBlockHover(
            creating.deps,
            pointerEvent({ clientX: 500, clientY: 10, buttons: 1 }),
        );
        expect(creating.setCanvasCursor).toHaveBeenLastCalledWith("crosshair");
    });
});
