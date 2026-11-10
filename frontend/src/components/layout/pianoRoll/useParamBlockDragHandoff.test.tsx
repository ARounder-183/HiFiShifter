// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createTimelineAxis } from "../renderKernel/timelineAxis";
import { buildCurveLayers } from "./panel/sceneBuilders";
import type { ParamViewSegment } from "./types";
import { useLiveParamEditing } from "./useLiveParamEditing";
import {
    useParamBlockDragHandoff,
    type ParamBlockDragHandoffInputs,
} from "./useParamBlockDragHandoff";

const oldView: ParamViewSegment = {
    key: "v2|track|pitch|0|4|1",
    startFrame: 0,
    stride: 1,
    framePeriodMs: 5,
    orig: [60, 60.1, 59.9, 60],
    edit: [60, 60.1, 59.9, 60],
    referenceKind: "source_curve",
};
const targetView: ParamViewSegment = { ...oldView, edit: oldView.edit.map((value) => value + 4) };
const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    while (mounted.length) await mounted.pop()?.();
    vi.unstubAllGlobals();
});

async function harness() {
    let live!: ReturnType<typeof useLiveParamEditing>;
    const frames = new Map<number, FrameRequestCallback>();
    let frameSerial = 0;
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
        frames.set(++frameSerial, callback);
        return frameSerial;
    });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
    const commit: NonNullable<ParamBlockDragHandoffInputs["commitRef"]["current"]> = {
        blockId: "note",
        gestureToken: 1,
        key: oldView.key,
        startedAt: 0,
        startFrame: 0,
        stride: 1,
        authorityRevisionAtStart: 1,
        authorityRequestSeqAtCommit: 5,
        externalHistoryTokenAtStart: "0:0",
        committed: true,
        committedSegments: [{ startFrame: 0, values: targetView.edit }],
        releaseState: "waiting-for-curve",
    };
    const inputs: Omit<ParamBlockDragHandoffInputs, "clearLiveEditOverride"> = {
        commitRef: { current: commit },
        snapshotRef: { current: null },
        interactionCurveRef: { current: targetView },
        activeGestureTokenRef: { current: 1 },
        releaseRafRef: { current: null },
        paramView: oldView,
        authorityRevision: 1,
        getAuthorityWatermark: () => ({ issuedRequestSeq: 5, appliedRequestSeq: 5 }),
        structureVersion: 0,
        onReleased: vi.fn(),
    };
    function Harness({
        paramView,
        authorityRevision,
        requestSeq,
    }: {
        paramView: ParamViewSegment;
        authorityRevision: number;
        requestSeq: number;
    }) {
        live = useLiveParamEditing({
            rootTrackId: "track",
            editParam: "pitch",
            pitchEnabled: true,
            paramView,
            setParamView: () => {},
            bumpRefreshToken: () => {},
            invalidate: () => {},
        });
        useParamBlockDragHandoff({
            ...inputs,
            paramView,
            authorityRevision,
            getAuthorityWatermark: () => ({
                issuedRequestSeq: requestSeq,
                appliedRequestSeq: requestSeq,
            }),
            clearLiveEditOverride: live.clearLiveEditOverride,
        });
        return null;
    }
    const element = document.createElement("div");
    document.body.append(element);
    const root = createRoot(element);
    mounted.push(async () => {
        await act(async () => root.unmount());
        element.remove();
    });
    const render = (paramView = oldView, authorityRevision = 1, requestSeq = 5) =>
        act(async () => {
            root.render(
                <Harness
                    paramView={paramView}
                    authorityRevision={authorityRevision}
                    requestSeq={requestSeq}
                />,
            );
        });
    const editedLine = (paramView = oldView) =>
        buildCurveLayers(
            {
                clipboardRef: { current: null },
                detectedPitchCurves: [],
                editParam: "pitch",
                liveEditOverrideRef: live.liveEditOverrideRef,
                paramView,
                pitchEnabled: true,
                referencePitchOverlays: [],
                secSpanClipRect: () => null,
                secondaryParamViews: {},
                selectionClipRect: () => null,
                selectionRef: { current: null },
                themeMode: "dark",
                valueToY: (_param, value) => 200 - value,
                viewSizeRef: { current: { w: 800, h: 400 } },
                visibleSecondaryParamIds: [],
                interactionParamViewRef: inputs.interactionCurveRef,
            },
            createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 800 }),
        ).find((layer) => layer.lineWidthPx === 2.6)!.values;
    const paint = () =>
        act(async () => {
            for (const [id, callback] of [...frames]) {
                frames.delete(id);
                callback(0);
            }
        });
    await render();
    return { inputs, commit, render, editedLine, paint, getLive: () => live, frames };
}

describe("block drag release and live curve handoff", () => {
    it("keeps the final pitch curve through an old history reset and authoritative handoff", async () => {
        const h = await harness();
        await act(async () => {
            h.getLive().applyDenseToLiveEdit(oldView, 0, targetView.edit, 0, 3, "draw");
            // A normal write advances history too. Its readback can arrive while
            // uploadFullResCurveSegments is yielding and reset the live mirror.
            h.getLive().resetLiveEditPreview(oldView);
            h.getLive().markLiveEditCommitted();
        });
        expect(h.getLive().liveEditOverrideRef.current?.edit).toEqual(oldView.edit);
        expect(h.editedLine()).toEqual(targetView.edit);
        await h.render(targetView, 2, 6);
        expect(h.editedLine(targetView)).toEqual(targetView.edit);
        await h.paint();
        expect(h.inputs.interactionCurveRef.current).toBeNull();
        expect(h.editedLine(targetView)).toEqual(targetView.edit);
    });

    it("retains the preview until a post-write request contains the actual committed values", async () => {
        const h = await harness();
        await h.render(targetView, 2, 5);
        await h.paint();
        expect(h.inputs.interactionCurveRef.current).toBe(targetView);
        await h.render(oldView, 3, 6);
        await h.paint();
        expect(h.inputs.interactionCurveRef.current).toBe(targetView);
        await h.render(targetView, 4, 7);
        await h.paint();
        expect(h.inputs.interactionCurveRef.current).toBeNull();
    });

    it("cannot clear a newer gesture even if its pointer starts after confirmation", async () => {
        const h = await harness();
        await h.render(targetView, 2, 6);
        h.inputs.activeGestureTokenRef.current = 2;
        await h.paint();
        expect(h.inputs.interactionCurveRef.current).toBe(targetView);
        expect(h.inputs.onReleased).not.toHaveBeenCalled();
    });
});
