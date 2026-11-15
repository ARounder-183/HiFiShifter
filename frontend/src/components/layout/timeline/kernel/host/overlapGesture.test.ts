// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ClipInfo, TrackInfo } from "../../../../../features/session/sessionTypes";
import {
    createTimelineKernelHost,
    type TimelineKernelData,
    type TimelineKernelHost,
} from "./timelineKernelHost";

vi.mock("../../../renderKernel/gl/glContext", () => ({
    createGlCanvas: () => ({ gl: {}, dispose: vi.fn() }),
}));
vi.mock("../../../renderKernel/gl/sdfBoxProgram", () => ({
    createSdfBoxProgram: () => ({ dispose: vi.fn() }),
}));
vi.mock("../../../../../hooks/useDevicePixelRatio", () => ({
    subscribeDevicePixelRatio: () => () => {},
}));

function clip(id: string, startSec: number, lengthSec: number): ClipInfo {
    return {
        id,
        trackId: "track",
        name: id,
        startSec,
        lengthSec,
        color: "blue",
        sourcePath: `${id}.wav`,
        sourceStartSec: 0,
        sourceEndSec: lengthSec,
        gain: 1,
        muted: false,
        playbackRate: 1,
        reversed: false,
        channelMode: 0,
        loopEnabled: false,
        snapOffsetSec: 0,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 0,
        fadeOutShape: 0,
        fadeInDir: 0,
        fadeOutDir: 0,
    };
}

let host: TimelineKernelHost;
let container: HTMLDivElement;
let data: TimelineKernelData;
let testTime = 0;
const onSelectClip = vi.fn((clipId: string) => {
    data = { ...data, selectedClipId: clipId };
});
const onDragPreview = vi.fn();
const onDragCommit = vi.fn();
const onFadePreview = vi.fn();
const onFadeCommit = vi.fn();
const onTrimPreview = vi.fn();
const onTrimCommit = vi.fn();
const onCrossfadeGripPreview = vi.fn();
const onCrossfadeGripCommit = vi.fn();
const onSeekTo = vi.fn();

beforeEach(() => {
    vi.clearAllMocks();
    // Each fresh host is a separate session, not the second press of a prior test's double-click.
    testTime += 1000;
    vi.spyOn(performance, "now").mockReturnValue(testTime);
    // Rendering is unrelated to dispatch: keep queued frames idle, but use real hit/gesture code.
    vi.stubGlobal(
        "requestAnimationFrame",
        vi.fn(() => 1),
    );
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    class Observer {
        observe() {}
        disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", Observer);
    vi.stubGlobal("IntersectionObserver", Observer);

    container = document.createElement("div");
    document.body.appendChild(container);
    vi.spyOn(container, "getBoundingClientRect").mockReturnValue(new DOMRect(0, 0, 1200, 96));
    data = {
        tracks: [{ id: "track" } as TrackInfo],
        clips: [
            { ...clip("short", 1, 2), autoFadeOutSec: 1 },
            { ...clip("long", 2, 4), autoFadeInSec: 1 },
        ],
        projectSec: 12,
        darkMode: true,
        bpm: 120,
        beatsPerBar: 4,
        grid: "1/4",
        primaryTimeUnit: "seconds",
        secondaryTimeUnit: "clock",
        minLabelSpacingPx: 110,
        minGridSpacingPx: 8,
        swingPercent: 0,
        tempoMap: null,
        rowHeight: 96,
        playheadSec: 0,
        keybindings: {
            horizontalZoom: null,
            verticalZoom: null,
            scrollHorizontal: null,
            scrollVertical: null,
            scrollbarZoom: null,
            fadeShapeCycle: null,
            clipMultiSelectToggle: null,
            clipRangeSelect: null,
            clipRangeToParamSelection: null,
            clipStretch: null,
            fadeCurvatureDrag: null,
        },
        playheadZoomEnabled: false,
        initialPxPerSec: 100,
        selectedClipId: null,
        multiSelectedClipIds: [],
        showAllTakes: false,
        activeGroupIds: [],
        disabledGroupIds: [],
    };
    host = createTimelineKernelHost({
        container,
        canvas: document.createElement("canvas"),
        hScrollbarThumb: document.createElement("div"),
        vScrollbarThumb: document.createElement("div"),
        data: () => data,
        interactions: {
            onSelectClip,
            onDragPreview,
            onDragCommit,
            onFadePreview,
            onFadeCommit,
            onTrimPreview,
            onTrimCommit,
            onCrossfadeGripPreview,
            onCrossfadeGripCommit,
            onSeekTo,
        },
    });
});

afterEach(() => {
    host?.dispose();
    container?.remove();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

function pointer(type: string, x: number, y: number, modifiers: Partial<PointerEventInit> = {}) {
    const event = new PointerEvent(type, {
        clientX: x,
        clientY: y,
        button: 0,
        buttons: type === "pointerup" || type === "pointercancel" ? 0 : 1,
        pointerId: 1,
        pointerType: "mouse",
        bubbles: true,
        cancelable: true,
        ...modifiers,
    });
    (type === "pointerdown" ? container : window).dispatchEvent(event);
}

describe("overlap click and control drag dispatch", () => {
    it.each([
        ["in", "long", 225, 1.2],
        ["out", "short", 275, 0.8],
    ] as const)("drags the %s curve without moving a clip", (side, clipId, x, fadeSec) => {
        expect(host.debugHitAt(x, 75)).toMatchObject({ region: `fade-${side}-corner`, clipId });
        pointer("pointerdown", x, 75);
        pointer("pointermove", x + 20, 75);
        pointer("pointerup", x + 20, 75);
        expect(onFadePreview).toHaveBeenCalledExactlyOnceWith(
            expect.objectContaining({
                clipId,
                side,
                fadeSec: expect.closeTo(fadeSec),
            }),
        );
        expect(onFadeCommit).toHaveBeenCalledWith(
            expect.objectContaining({ clipId, side, cancelled: false }),
        );
        expect(onDragPreview).not.toHaveBeenCalled();
        expect(onDragCommit).not.toHaveBeenCalled();
    });

    it("drags the crossfade intersection with both original partners", () => {
        expect(host.debugHitAt(250, 56)).toMatchObject({
            region: "crossfade-grip",
            clipId: "long",
        });
        pointer("pointerdown", 250, 56);
        pointer("pointermove", 270, 56);
        pointer("pointerup", 270, 56);
        expect(onCrossfadeGripPreview).toHaveBeenCalledExactlyOnceWith(
            expect.objectContaining({
                earlierClipId: "short",
                laterClipId: "long",
                deltaSec: expect.closeTo(0.2),
            }),
        );
        expect(onCrossfadeGripCommit).toHaveBeenCalledWith(
            expect.objectContaining({
                earlierClipId: "short",
                laterClipId: "long",
                cancelled: false,
            }),
        );
        expect(onDragPreview).not.toHaveBeenCalled();
        expect(onFadePreview).not.toHaveBeenCalled();
    });

    it.each([
        ["left", "long", 200],
        ["right", "short", 298],
    ] as const)("drags the overlapping %s edge as trim", (edge, clipId, x) => {
        expect(host.debugHitAt(x, 40)).toMatchObject({ region: `${edge}-edge`, clipId });
        pointer("pointerdown", x, 40);
        pointer("pointermove", x + 20, 40);
        pointer("pointerup", x + 20, 40);
        expect(onTrimPreview).toHaveBeenCalledWith(expect.objectContaining({ clipId, edge }));
        expect(onTrimCommit).toHaveBeenCalledWith(
            expect.objectContaining({ clipId, edge, cancelled: false }),
        );
        expect(onDragPreview).not.toHaveBeenCalled();
    });

    it("a plain click with pointer jitter still selects the foreground short clip", () => {
        pointer("pointerdown", 225, 75);
        pointer("pointermove", 228, 75);
        pointer("pointerup", 228, 75);
        expect(data.selectedClipId).toBe("short");
        expect(onSeekTo).toHaveBeenCalledWith(expect.closeTo(2.28));
        expect(onFadePreview).not.toHaveBeenCalled();
        expect(onCrossfadeGripPreview).not.toHaveBeenCalled();
        expect(onDragPreview).not.toHaveBeenCalled();
    });

    it("waits for the overlap drag threshold before editing the control's owner", () => {
        pointer("pointerdown", 225, 75);
        expect(data.selectedClipId).toBe("short");
        pointer("pointermove", 233, 75);
        expect(onFadePreview).not.toHaveBeenCalled();
        pointer("pointermove", 245, 75);
        expect(data.selectedClipId).toBe("long");
        expect(onFadePreview).toHaveBeenCalledWith(expect.objectContaining({ clipId: "long" }));
        pointer("pointerup", 245, 75);
    });

    it("a normal overlapping body still drags the short clip", () => {
        expect(host.debugHitAt(250, 35)).toMatchObject({ region: "body", clipId: "short" });
        pointer("pointerdown", 250, 35);
        pointer("pointermove", 270, 35);
        pointer("pointerup", 270, 35);
        expect(onDragPreview).toHaveBeenCalledWith(expect.objectContaining({ clipId: "short" }));
        expect(onFadePreview).not.toHaveBeenCalled();
        expect(onCrossfadeGripPreview).not.toHaveBeenCalled();
    });

    it("the header keeps the later clip's controls", () => {
        expect(host.debugHitAt(275, 9)).toMatchObject({ region: "header", clipId: "long" });
    });

    it("pointer cancellation rolls back the fade, not a clip move", () => {
        pointer("pointerdown", 225, 75);
        pointer("pointermove", 245, 75);
        pointer("pointercancel", 245, 75);
        expect(onFadeCommit).toHaveBeenCalledWith(
            expect.objectContaining({ clipId: "long", cancelled: true }),
        );
        expect(onDragCommit).not.toHaveBeenCalled();
    });
});
