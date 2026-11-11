// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { beginParamBlockDrag, canRollbackParamBlockDragCommit } from "./paramBlockDrag";
import type { ParamBlockDragContext } from "./paramBlockDrag";
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlock } from "./paramBlocks";
import type { ParamViewSegment } from "./types";

describe("canRollbackParamBlockDragCommit", () => {
    it("allows rolling back to the pre-commit revision", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 7,
                beforeRevision: 7,
                appliedRevision: null,
            }),
        ).toBe(true);
    });

    it("allows rolling back the gesture's own applied revision", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 8,
                beforeRevision: 7,
                appliedRevision: 8,
            }),
        ).toBe(true);
    });

    it("refuses to overwrite a third-party structural edit", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 9,
                beforeRevision: 7,
                appliedRevision: 8,
            }),
        ).toBe(false);
    });

    it("refuses when no applied revision is known and current moved on", () => {
        expect(
            canRollbackParamBlockDragCommit({
                currentRevision: 9,
                beforeRevision: 7,
                appliedRevision: null,
            }),
        ).toBe(false);
    });
});

/** 与 paramBlockDragCenters.test.ts 同一份块夹具（只保留拖动路径读到的字段）。 */
function block(args: {
    id: string;
    startSec?: number;
    endSec?: number;
    averageValue?: number | null;
}): ParamBlock {
    return {
        id: args.id,
        rootTrackId: "root",
        kind: "auto",
        members: [],
        startSec: args.startSec ?? 0,
        endSec: args.endSec ?? 1,
        averageValue: args.averageValue === undefined ? 60 : args.averageValue,
        manualValue: null,
    };
}

function view(): ParamViewSegment {
    const values = Array.from({ length: 40 }, () => 60);
    return {
        key: "view-1",
        framePeriodMs: 5,
        startFrame: 0,
        stride: 1,
        referenceKind: "source_curve",
        orig: values.slice(),
        edit: values.slice(),
    };
}

/**
 * 拖块读数（问题 1 的核心回归）。
 *
 * 读数必须来自 `projectParamBlockDragDisplayValue(锚点起始值 + 位移)` ——
 * 也就是「块将被拖到的中心值」。断言里的期望值一律手算，不反过来调用该函数：
 * 否则这条测试永远通过，而 bug 正是「读数与块不是同一个投影」。
 */
describe("block drag value readout", () => {
    const cleanups: Array<() => void> = [];
    afterEach(() => {
        for (const dispose of cleanups.splice(0)) dispose();
    });

    function startDrag(args: { readonly popupEnabled: boolean }) {
        const paramView = view();
        const anchor = block({ id: "a", startSec: 0, endSec: 0.2, averageValue: 60 });
        const onPreview = vi.fn();
        const context: ParamBlockDragContext = {
            paramBlockDragActivityRef: {
                current: { pointerGestureToken: null, pendingCommits: 0 },
            },
            paramView,
            rootTrackId: "root",
            editParam: "pitch",
            paramBlocksWithValues: [anchor],
            paramBlockGestureSerialRef: { current: 0 },
            paramBlockActiveGestureTokenRef: { current: null },
            paramBlockDragCommitRef: { current: null },
            paramBlockDragCommitReleaseRafRef: { current: null },
            paramBlockInteractionCurveRef: { current: null },
            paramBlockDragSnapshotRef: { current: null },
            paramBlockStateRef: { current: EMPTY_PARAM_BLOCK_STATE },
            paramBlockEpoch: {
                rootTrackId: "root",
                param: "pitch",
                layoutKey: "layout",
                editRevision: 1,
                externalHistoryToken: "token",
                framePeriodMs: 5,
            },
            curveOwned: false,
            paramViewRef: { current: paramView },
            liveEditActiveRef: { current: false },
            applyDenseToLiveEdit: () => {},
            resetLiveEditPreview: () => {},
            invalidateParamBlockSpec: () => {},
            requestWaveformRepaint: () => {},
            paintNow: () => {},
            bumpParamBlockStructure: () => {},
            setParamBlockState: () => {},
            getAuthorityWatermark: () => ({ issuedRequestSeq: 0, appliedRequestSeq: 0 }),
            paramAuthorityRevision: 0,
            onParamCommitSucceeded: () => {},
            bumpRefreshToken: () => {},
            snapToggleHeldAt: () => false,
            paramValuePopupEnabled: args.popupEnabled,
            onParamValuePreviewChange: onPreview,
            resolveValueDelta: (_startValue, rawDelta) => rawDelta,
        };
        const started = beginParamBlockDrag(
            {
                anchorBlockId: "a",
                clientX: 100,
                clientY: 100,
                contextMenu: false,
                pointerId: 1,
                captureTarget: document.createElement("canvas"),
                setActivePointerGestureEnd: () => {},
                clearActivePointerGestureEnd: () => {},
                /* 纵向值域：往上拖 50px = 值 +50（anchorPointerValue = 0）。 */
                pointerValue: (clientY) => 100 - clientY,
                selection: ["a"],
                anchorValue: 60,
            },
            context,
        );
        cleanups.push(() => window.dispatchEvent(new Event("pointercancel")));
        return { started, onPreview };
    }

    function moveTo(clientY: number) {
        const event = new Event("pointermove") as Event & {
            pointerId: number;
            buttons: number;
            clientX: number;
            clientY: number;
        };
        event.pointerId = 1;
        event.buttons = 1;
        event.clientX = 100;
        event.clientY = clientY;
        window.dispatchEvent(event);
    }

    function nextFrame(): Promise<void> {
        return new Promise((resolve) => requestAnimationFrame(() => resolve()));
    }

    function release() {
        const event = new Event("pointerup") as Event & {
            pointerId: number;
            clientX: number;
            clientY: number;
        };
        event.pointerId = 1;
        event.clientX = 100;
        event.clientY = 50;
        window.dispatchEvent(event);
    }

    it("publishes the value the block will land on, not the value under the pointer", () => {
        const { started, onPreview } = startDrag({ popupEnabled: true });
        expect(started).toBe(true);

        moveTo(50);

        /* 锚点起始 60 + 位移 50 = 110（手算，不调用投影函数）。 */
        expect(onPreview).toHaveBeenLastCalledWith({ clientX: 100, clientY: 50, value: 110 });
    });

    it("publishes the first move immediately and coalesces the rest into one frame", async () => {
        const { onPreview } = startDrag({ popupEnabled: true });
        /* 首个越过死区的移动同步发布（起步零延迟）；同一帧内的后续位移按 rAF 合并。 */
        moveTo(50);
        moveTo(40);
        expect(onPreview).toHaveBeenLastCalledWith({ clientX: 100, clientY: 50, value: 110 });
        await nextFrame();
        expect(onPreview).toHaveBeenLastCalledWith({ clientX: 100, clientY: 40, value: 120 });
    });

    it("clears the readout when the gesture ends", () => {
        const { onPreview } = startDrag({ popupEnabled: true });
        moveTo(50);
        release();
        expect(onPreview).toHaveBeenLastCalledWith(null);
    });

    it("never publishes when the value popup is disabled", () => {
        const { onPreview } = startDrag({ popupEnabled: false });
        moveTo(50);
        release();
        expect(onPreview).not.toHaveBeenCalled();
    });

    it("does not publish before the pointer leaves the dead zone", () => {
        const { onPreview } = startDrag({ popupEnabled: true });
        moveTo(99);
        expect(onPreview).not.toHaveBeenCalled();
    });
});
