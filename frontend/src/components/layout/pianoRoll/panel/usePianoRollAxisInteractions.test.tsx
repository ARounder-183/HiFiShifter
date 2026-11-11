// @vitest-environment jsdom
import { act } from "react";
import { useRef } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePianoRollAxisInteractions } from "./usePianoRollAxisInteractions";

type Preview = { text: string } | null;

function Harness({
    show,
    format,
    onPreview,
}: {
    show: boolean;
    format: (value: number) => string;
    onPreview: (value: Preview) => void;
}) {
    const axisWrapRef = useRef<HTMLDivElement | null>(null);
    const viewSizeRef = useRef({ w: 100, h: 100 });
    const pitchViewRef = useRef({ center: 60, span: 12 });
    return (
        <>
            <div
                ref={axisWrapRef}
                data-axis
                style={{ width: 56, height: 100, position: "relative" }}
            />
            <AxisHook
                axisWrapRef={axisWrapRef}
                viewSizeRef={viewSizeRef}
                pitchViewRef={pitchViewRef}
                show={show}
                format={format}
                onPreview={onPreview}
            />
        </>
    );
}

function AxisHook({
    axisWrapRef,
    viewSizeRef,
    pitchViewRef,
    show,
    format,
    onPreview,
}: {
    axisWrapRef: { current: HTMLDivElement | null };
    viewSizeRef: { current: { w: number; h: number } };
    pitchViewRef: { current: { center: number; span: number } };
    show: boolean;
    format: (value: number) => string;
    onPreview: (value: Preview) => void;
}) {
    usePianoRollAxisInteractions({
        s: { showParamValuePopup: show },
        editParam: "volume",
        axisDirection: "left-bottom",
        axisWrapRef,
        viewSizeRef,
        pitchViewRef,
        dispatch: vi.fn() as never,
        yToValue: (_param, y, h) => 1 - y / h,
        formatParamValuePreview: format,
        setAxisValuePreview: (value) => onPreview(value === null ? null : { text: value.text }),
    });
    return null;
}

describe("usePianoRollAxisInteractions tooltip refresh", () => {
    let container: HTMLDivElement;
    let root: ReturnType<typeof createRoot>;
    let previews: Preview[];

    beforeEach(() => {
        vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) =>
            window.setTimeout(() => cb(performance.now()), 0),
        );
        vi.stubGlobal("cancelAnimationFrame", (id: number) => window.clearTimeout(id));
        container = document.createElement("div");
        document.body.appendChild(container);
        root = createRoot(container);
        previews = [];
    });

    afterEach(async () => {
        await act(async () => root.unmount());
        container.remove();
        vi.unstubAllGlobals();
    });

    async function render(show: boolean, format: (value: number) => string) {
        await act(async () => {
            root.render(
                <Harness
                    show={show}
                    format={format}
                    onPreview={(value) => previews.push(value)}
                />,
            );
        });
        const axis = container.querySelector<HTMLDivElement>("[data-axis]");
        if (axis === null) throw new Error("axis missing");
        vi.spyOn(axis, "getBoundingClientRect").mockReturnValue({
            left: 0,
            top: 0,
            right: 56,
            bottom: 100,
            width: 56,
            height: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
        } as DOMRect);
        return axis;
    }

    it("recomputes immediately when the formatter changes", async () => {
        const axis = await render(true, (value) => `old:${value.toFixed(2)}`);
        await act(async () => {
            axis.dispatchEvent(new PointerEvent("pointermove", { clientX: 20, clientY: 50, bubbles: true }));
            await new Promise((resolve) => setTimeout(resolve, 10));
        });
        expect(previews.at(-1)).toEqual({ text: "old:0.50" });

        previews = [];
        await act(async () => {
            root.render(
                <Harness
                    show
                    format={(value) => `new:${value.toFixed(2)}`}
                    onPreview={(value) => previews.push(value)}
                />,
            );
        });
        expect(previews.at(-1)).toEqual({ text: "new:0.50" });
    });

    it("clears on pointerleave and when the popup is disabled", async () => {
        const axis = await render(true, (value) => value.toFixed(2));
        await act(async () => {
            axis.dispatchEvent(new PointerEvent("pointermove", { clientX: 20, clientY: 50, bubbles: true }));
            await new Promise((resolve) => setTimeout(resolve, 10));
        });
        previews = [];
        await act(async () => {
            axis.dispatchEvent(new PointerEvent("pointerleave", { bubbles: true }));
            await new Promise((resolve) => setTimeout(resolve, 10));
        });
        expect(previews.at(-1)).toBeNull();

        const axis2 = await render(false, (value) => value.toFixed(2));
        previews = [];
        await act(async () => {
            axis2.dispatchEvent(new PointerEvent("pointermove", { clientX: 20, clientY: 50, bubbles: true }));
            await new Promise((resolve) => setTimeout(resolve, 10));
        });
        expect(previews).toEqual([]);
    });
});
