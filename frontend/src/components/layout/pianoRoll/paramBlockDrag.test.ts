// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { paramsApi } from "../../../services/api";
import {
    beginParamBlockDrag,
    type ParamBlockDragContext,
    type ParamBlockDragStart,
} from "./paramBlockDrag";
import {
    EMPTY_PARAM_BLOCK_STATE,
    paramViewContainsCommittedSegments,
    resolveParamBlockPitchSnapDelta,
    type ParamBlock,
} from "./paramBlocks";
import { resolveParamBlockFrameValue, resolveParamBlockPanelValues } from "./paramBlockFrame";
import type { ParamViewSegment } from "./types";

function deferred() {
    let resolve!: () => void;
    const promise = new Promise<void>((done) => {
        resolve = done;
    });
    return { promise, resolve };
}

function pointer(type: string, clientY: number, modifiers: MouseEventInit = {}) {
    const event = new MouseEvent(type, {
        clientY,
        buttons: type === "pointermove" ? 1 : 0,
        ...modifiers,
    });
    Object.defineProperty(event, "pointerId", { value: 1 });
    window.dispatchEvent(event);
}

function harness(stride = 1) {
    const backend = [60, 60.1, 59.9, 0, 60.2, 60, 59.8, 60.1];
    const pv: ParamViewSegment = {
        key: `v2|track|pitch|0|8|${stride}`,
        framePeriodMs: 5,
        startFrame: 0,
        stride,
        referenceKind: "source_curve",
        orig: backend.filter((_, i) => i % stride === 0),
        edit: backend.filter((_, i) => i % stride === 0),
    };
    const block: ParamBlock = {
        id: "note",
        rootTrackId: "track",
        kind: "auto",
        startSec: 0,
        endSec: 0.04,
        averageValue: 60,
        members: [
            {
                clipId: "clip",
                takeId: "take",
                syllableId: "take:0.000:0.040",
                sourceStartSec: 0,
                sourceEndSec: 0.04,
                startSec: 0,
                endSec: 0.04,
            },
        ],
    };
    const get = vi
        .spyOn(paramsApi, "getParamFrames")
        .mockImplementation(async (_track, _param, start, count, step = 1) => {
            const values = Array.from({ length: count }, (_, i) => backend[start + i * step] ?? 0);
            return {
                ok: true,
                root_track_id: "track",
                param: "pitch",
                frame_period_ms: 5,
                start_frame: start,
                orig: values,
                edit: values,
                reference_kind: "source_curve",
            };
        });
    let writeGate: Promise<void> | undefined;
    const write = vi
        .spyOn(paramsApi, "setParamFrames")
        .mockImplementation(async (_track, _param, start, values) => {
            await writeGate;
            backend.splice(start, values.length, ...values);
            return { ok: true };
        });
    const context: ParamBlockDragContext = {
        paramView: pv,
        rootTrackId: "track",
        editParam: "pitch",
        paramBlocksWithValues: [block],
        paramBlockDragActivityRef: { current: { pointerGestureToken: null, pendingCommits: 0 } },
        paramBlockGestureSerialRef: { current: 0 },
        paramBlockActiveGestureTokenRef: { current: null },
        paramBlockDragCommitRef: { current: null },
        paramBlockDragCommitReleaseRafRef: { current: null },
        paramBlockInteractionCurveRef: { current: null },
        paramBlockDragSnapshotRef: { current: null },
        paramBlockStateRef: { current: EMPTY_PARAM_BLOCK_STATE },
        paramBlockCommitQueueRef: { current: Promise.resolve() },
        paramViewRef: { current: pv },
        liveEditActiveRef: { current: false },
        applyDenseToLiveEdit: vi.fn(),
        resetLiveEditPreview: vi.fn(),
        invalidateParamBlockSpec: vi.fn(),
        requestWaveformRepaint: vi.fn(),
        paintNow: vi.fn(),
        bumpParamBlockStructure: vi.fn(),
        setParamBlockState: vi.fn(),
        persistParamBlockMerges: vi.fn(),
        getAuthorityWatermark: () => ({ issuedRequestSeq: 5, appliedRequestSeq: 5 }),
        paramAuthorityRevision: 1,
        onParamCommitSucceeded: vi.fn(),
        bumpRefreshToken: vi.fn(),
        snapToggleHeldAt: (event) => event.shiftKey,
        resolveValueDelta: (startValue, rawDelta, snapToggleHeld) =>
            resolveParamBlockPitchSnapDelta({ startValue, rawDelta, snapToggleHeld }),
    };
    let activeEnd: (() => void) | undefined;
    const start = (
        overrides: Partial<ParamBlockDragStart> = {},
        contextOverrides: Partial<ParamBlockDragContext> = {},
    ) =>
        beginParamBlockDrag(
            {
                anchorBlockId: "note",
                anchorValue:
                    context.paramBlockDragSnapshotRef.current?.centerValues.get("note") ?? 60,
                clientX: 20,
                clientY: 100,
                contextMenu: false,
                pointerId: 1,
                captureTarget: document.createElement("canvas"),
                selection: ["note"],
                pointerValue: (y) => 60 + (100 - y) / 10,
                setActivePointerGestureEnd: (end) => {
                    activeEnd = end;
                },
                clearActivePointerGestureEnd: () => {
                    activeEnd = undefined;
                },
                ...overrides,
            },
            { ...context, ...contextOverrides },
        );
    return {
        backend,
        pv,
        context,
        get,
        write,
        start,
        end: () => activeEnd?.(),
        gateWrites: (gate: Promise<void> | undefined) => {
            writeGate = gate;
        },
    };
}

async function flush() {
    await vi.runAllTimersAsync();
}

beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) =>
        setTimeout(() => callback(0), 0),
    );
    vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
});
afterEach(() => {
    pointer("pointercancel", 100);
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

describe("block drag pointer and commit lifecycle", () => {
    it("does not acquire the refresh guard or discard a pending transaction on a click", async () => {
        const h = harness();
        h.start();
        pointer("pointerup", 100);
        expect(h.context.liveEditActiveRef?.current).toBe(false);
        expect(h.context.paramBlockDragSnapshotRef.current).toBeNull();
        h.start();
        pointer("pointerup", 80);
        const pending = h.context.paramBlockDragCommitRef.current;
        h.start();
        pointer("pointerup", 100);
        expect(h.context.paramBlockDragCommitRef.current).toBe(pending);
        await flush();
        expect(h.write).toHaveBeenCalledTimes(1);
        expect(h.context.liveEditActiveRef?.current).toBe(false);
    });

    it("releases refresh protection and rolls back a cancelled drag", () => {
        const h = harness();
        h.start();
        pointer("pointermove", 80);
        expect(h.context.liveEditActiveRef?.current).toBe(true);
        expect(h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note")).toBe(62);
        pointer("pointercancel", 80);
        expect(h.context.liveEditActiveRef?.current).toBe(false);
        expect(h.context.paramBlockDragSnapshotRef.current).toBeNull();
        expect(h.write).not.toHaveBeenCalled();
    });

    it("paints the first movement immediately and consumes release coordinates without another move", async () => {
        const h = harness();
        h.start();
        pointer("pointermove", 90);
        expect(h.context.paintNow).toHaveBeenCalledOnce();
        expect(h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note")).toBe(61);
        pointer("pointerup", 60);
        expect(h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note")).toBe(64);
        await flush();
        expect(h.backend[0]).toBe(64);
        expect(h.backend[3]).toBe(0);
    });

    it("uses buttons=0 move coordinates and the configured release modifier", async () => {
        const h = harness();
        vi.spyOn(h.context, "snapToggleHeldAt").mockImplementation((event) => event.ctrlKey);
        h.start();
        pointer("pointermove", 82);
        pointer("pointermove", 75, { buttons: 0, ctrlKey: true });
        expect(h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note")).toBe(62.5);
        await flush();
        expect(h.backend[0]).toBe(62.5);
    });

    it.each([1, 3])(
        "serializes A→B→C while the authoritative window stays stale (stride=%i)",
        async (stride) => {
            const h = harness(stride);
            const gate = deferred();
            h.gateWrites(gate.promise);
            h.start();
            pointer("pointerup", 80);
            const first = h.context.paramBlockDragCommitRef.current;
            await flush();
            h.start();
            pointer("pointerup", 70);
            const second = h.context.paramBlockDragCommitRef.current;
            expect(h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note")).toBe(65);
            expect(h.context.paramBlockInteractionCurveRef.current?.edit[0]).toBe(65);
            gate.resolve();
            await flush();
            expect(h.get).toHaveBeenCalledTimes(2);
            expect(h.write).toHaveBeenCalledTimes(2);
            expect(h.backend[0]).toBe(65);
            expect(h.backend[1]).toBeCloseTo(65.1);
            expect(h.backend[3]).toBe(0);
            expect(h.context.paramBlockDragCommitRef.current).toBe(second);
            expect(first?.committed).toBe(true);
            expect(second?.committed).toBe(true);
            expect(h.context.liveEditActiveRef?.current).toBe(false);
            const sampled = { ...h.pv, edit: h.backend.filter((_, i) => i % stride === 0) };
            expect(
                paramViewContainsCommittedSegments({
                    paramView: sampled,
                    segments: second?.committedSegments ?? [],
                }),
            ).toBe(true);
            expect(
                resolveParamBlockFrameValue({
                    block: { startSec: 0, endSec: 0.04, logicalCenterValue: 65 },
                    samples: { ...sampled, values: sampled.edit, isPitch: true, param: "pitch" },
                }),
            ).toBe(65);
        },
    );

    it("restores an earlier committed handoff when a following gesture is cancelled", async () => {
        const h = harness();
        const gate = deferred();
        h.gateWrites(gate.promise);
        h.start();
        pointer("pointerup", 80);
        const first = h.context.paramBlockDragCommitRef.current;
        await flush();
        h.start();
        pointer("pointermove", 70);
        gate.resolve();
        await flush();
        expect(h.context.liveEditActiveRef?.current).toBe(true);
        pointer("pointercancel", 70);
        expect(h.context.paramBlockDragCommitRef.current).toBe(first);
        expect(first?.committed).toBe(true);
        expect(h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note")).toBe(62);
        expect(h.context.liveEditActiveRef?.current).toBe(false);
    });

    it("does not persist a new center if the curve upload fails and allows the next gesture", async () => {
        const h = harness();
        vi.spyOn(console, "error").mockImplementation(() => {});
        h.write.mockRejectedValueOnce(new Error("write failed"));
        h.start();
        pointer("pointerup", 80);
        await flush();
        expect(h.context.persistParamBlockMerges).not.toHaveBeenCalled();
        expect(h.context.liveEditActiveRef?.current).toBe(false);
        expect(h.context.paramBlockDragCommitRef.current).toBeNull();
        h.start();
        pointer("pointerup", 70);
        await flush();
        expect(h.backend[0]).toBe(63);
        expect(h.context.persistParamBlockMerges).toHaveBeenCalledOnce();
    });

    it.each(["read", "write"])(
        "rolls back a backend %s rejection returned as ok=false",
        async (operation) => {
            const h = harness();
            vi.spyOn(console, "error").mockImplementation(() => {});
            if (operation === "read") {
                const getFrames = h.get.getMockImplementation()!;
                h.get.mockImplementationOnce(async (...args) => ({
                    ...(await getFrames(...args)),
                    ok: false,
                }));
            } else {
                h.write.mockResolvedValueOnce({ ok: false });
            }
            h.start();
            pointer("pointerup", 80);
            await flush();
            expect(h.context.persistParamBlockMerges).not.toHaveBeenCalled();
            expect(h.context.liveEditActiveRef?.current).toBe(false);
            expect(h.context.paramBlockDragCommitRef.current).toBeNull();
            expect(h.backend[0]).toBe(60);
        },
    );

    it("starts every selected block from its rendered center on consecutive drags", async () => {
        const h = harness();
        const original = h.context.paramBlocksWithValues[0];
        const blocks: ParamBlock[] = [
            { ...original, endSec: 0.02, members: [{ ...original.members[0], endSec: 0.02 }] },
            {
                ...original,
                id: "other",
                startSec: 0.02,
                averageValue: 64,
                members: [{ ...original.members[0], startSec: 0.02 }],
            },
        ];
        const gate = deferred();
        h.gateWrites(gate.promise);
        const selection = ["note", "other"];
        h.start(
            {
                selection,
                centerValues: new Map([
                    ["note", 60],
                    ["other", 64],
                ]),
            },
            { paramBlocksWithValues: blocks },
        );
        pointer("pointerup", 80);
        await flush();
        h.start({ selection }, { paramBlocksWithValues: blocks });
        pointer("pointerup", 70);
        expect([...h.context.paramBlockDragSnapshotRef.current!.centerValues]).toEqual([
            ["note", 65],
            ["other", 69],
        ]);
        gate.resolve();
        await flush();
        expect(h.backend[0]).toBe(65);
    });

    it("scales the previous interaction curve while a preceding move is still being committed", async () => {
        const h = harness();
        const gate = deferred();
        h.gateWrites(gate.promise);
        h.start();
        pointer("pointerup", 80);
        await flush();
        h.start({ pitchScale: true });
        pointer("pointermove", 70);
        expect(h.context.paramBlockInteractionCurveRef.current?.edit[0]).toBe(62);
        pointer("pointerup", 70);
        expect(h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note")).toBe(62);
        gate.resolve();
        await flush();
        expect(h.backend[0]).toBe(62);
        expect(h.backend[1]).toBeCloseTo(62 + 0.1 * Math.exp(0.6));
    });

    it("excludes requests issued during the upload from the authority watermark", async () => {
        const h = harness();
        const gate = deferred();
        h.gateWrites(gate.promise);
        const watermark = { issuedRequestSeq: 5, appliedRequestSeq: 5 };
        vi.spyOn(h.context, "getAuthorityWatermark").mockImplementation(() => watermark);
        h.start();
        pointer("pointerup", 80);
        await flush();
        watermark.issuedRequestSeq = 7;
        gate.resolve();
        await flush();
        expect(h.context.paramBlockDragCommitRef.current?.authorityRequestSeqAtCommit).toBe(7);
    });

    it.each(["volume", "dyn", "tension"])(
        "moves the %s curve without replacing the saved pitch center",
        async (param) => {
            const h = harness();
            h.backend.fill(0.5);
            const pv = {
                ...h.pv,
                key: `v2|track|${param}|0|8|1`,
                edit: h.backend.slice(),
                orig: h.backend.slice(),
            };
            const pitchBlock = {
                ...h.context.paramBlocksWithValues[0],
                averageValue: 64,
                manualValue: 64,
            };
            const blocks = resolveParamBlockPanelValues({
                blocks: [pitchBlock],
                param,
                paramView: pv,
                fallbackValue: 0,
            });
            h.context.paramViewRef.current = pv;
            const state = h.context.paramBlockStateRef.current;
            h.start(
                { anchorValue: 0.5, pointerValue: (y) => 0.5 + (100 - y) / 100 },
                {
                    paramView: pv,
                    editParam: param,
                    paramBlocksWithValues: blocks,
                    resolveValueDelta: (_anchor, delta) => delta,
                },
            );
            pointer("pointermove", 80);
            expect(
                h.context.paramBlockDragSnapshotRef.current?.centerValues.get("note"),
            ).toBeCloseTo(0.7);
            expect(h.context.paramBlockInteractionCurveRef.current?.edit[0]).toBeCloseTo(0.7);
            pointer("pointerup", 80);
            await flush();
            expect(h.backend[0]).toBeCloseTo(0.7);
            expect(h.context.paramBlockStateRef.current).toBe(state);
            expect(h.context.persistParamBlockMerges).not.toHaveBeenCalled();
            expect(pitchBlock.manualValue).toBe(64);
            const final = resolveParamBlockPanelValues({
                blocks: [pitchBlock],
                param,
                paramView: { ...pv, edit: h.backend },
                fallbackValue: 0,
            })[0];
            expect(final.averageValue).toBeCloseTo(0.7);
        },
    );
});

it("a root interval without Clip members drags through the atomic curve API", async () => {
    const h = harness();
    const block = {
        ...h.context.paramBlocksWithValues[0],
        members: [],
        lineBlock: { id: "note", startFrame: 0, endFrameExclusive: 8 },
        lineFramePeriodMs: 5,
    };
    h.context.paramBlockStateRef.current = {
        ...EMPTY_PARAM_BLOCK_STATE,
        lineRootTrackId: "track",
        lineLayout: {
            schemaVersion: 1,
            revision: 1,
            framePeriodMs: 5,
            blocks: [block.lineBlock],
            gameBaseline: { revision: 0, segments: [] },
        },
    };
    const atomic = vi.fn(
        async (segments: readonly { startFrame: number; values: readonly number[] }[]) => {
            for (const segment of segments)
                h.backend.splice(segment.startFrame, segment.values.length, ...segment.values);
        },
    );
    h.start({}, { paramBlocksWithValues: [block], commitLineCurve: atomic });
    pointer("pointermove", 80);
    expect(h.context.paramBlockInteractionCurveRef.current?.edit[0]).toBe(62);
    pointer("pointerup", 80);
    await flush();
    expect(atomic).toHaveBeenCalledOnce();
    expect(h.write).not.toHaveBeenCalled();
    expect(h.backend[0]).toBe(62);
    expect(h.context.persistParamBlockMerges).not.toHaveBeenCalled();
});
