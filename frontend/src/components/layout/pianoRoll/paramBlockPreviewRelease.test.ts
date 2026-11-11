/**
 * 预览释放判据（./paramBlockPreviewRelease）自检。
 *
 * 【本测试守护什么】问题 4「松手先回弹旧参数、再生效新参数」的全部路径：
 * 1. 带曲线补丁的提交在**权威可见之前**绝不释放（旧实现按几何相等判断，
 *    pointerup 后一次渲染就成立）；
 * 2. 三个证据（revision 前进 / 水位确认 / 窗口包含片段）缺一不可；
 * 3. 纯布局提交不受曲线门槛影响（它没有曲线可回弹）；
 * 4. 超时必须先对账再释放 —— 判据返回 `"timeout"`，由调用方 flush。
 */
import { describe, expect, it } from "vitest";

import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";
import {
    isParamBlockPreviewGeometrySettled,
    PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS,
    shouldReleaseParamBlockPreview,
} from "./paramBlockPreviewRelease";
import type { ParamBlockEpoch, ParamBlockVisualTransaction } from "./paramBlockVisualTransaction";
import type { ParamViewSegment } from "./types";

const EPOCH: ParamBlockEpoch = {
    rootTrackId: "root",
    param: "pitch",
    layoutKey: "layout",
    editRevision: 3,
    externalHistoryToken: "0:0",
    framePeriodMs: 5,
};

function spec(): PianoRollParamBlockSpec {
    return {
        param: "pitch",
        valueMin: 0,
        valueMax: 127,
        blocks: [],
        borderRgba: [0, 0, 0, 1],
        selectedRgba: [1, 1, 1, 1],
        groupedRgba: [1, 0, 0, 1],
    };
}

/** 提交片段：[0..2] = [64, 64, 64]。 */
const SEGMENTS = [{ startFrame: 0, values: [64, 64, 64] }];

function view(values: readonly number[]): ParamViewSegment {
    return {
        key: "v",
        framePeriodMs: 5,
        startFrame: 0,
        stride: 1,
        referenceKind: "source_curve",
        orig: [...values],
        edit: [...values],
    };
}

function tx(overrides: Partial<ParamBlockVisualTransaction> = {}): ParamBlockVisualTransaction {
    return {
        id: 1,
        epoch: EPOCH,
        kind: "create",
        spec: spec(),
        phase: "awaiting-authority",
        startedAt: 0,
        ownedEditRevision: null,
        authorityRevisionAtStart: 10,
        authorityRequestSeqAtCommit: 5,
        committedSegments: SEGMENTS,
        ...overrides,
    };
}

const BASE = {
    now: 100,
    geometrySettled: true,
    gestureActive: false,
    pendingCommits: 0,
    authorityRevision: 11,
    watermark: { issuedRequestSeq: 6, appliedRequestSeq: 6 },
    paramView: view([64, 64, 64, 60, 60, 60]),
} as const;

describe("shouldReleaseParamBlockPreview", () => {
    it("never releases while the gesture is still running", () => {
        expect(
            shouldReleaseParamBlockPreview({ ...BASE, transaction: tx(), gestureActive: true }),
        ).toBe("wait");
    });

    it("never releases while a commit is still queued", () => {
        expect(
            shouldReleaseParamBlockPreview({ ...BASE, transaction: tx(), pendingCommits: 1 }),
        ).toBe("wait");
    });

    it("★ keeps the preview alive while the authoritative curve is still the old one", () => {
        // 曲线 revision 还没前进（提交未落地）。
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: tx(),
                authorityRevision: 10,
            }),
        ).toBe("wait");
        // revision 前进了，但窗口里仍是旧值。
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: tx(),
                paramView: view([60, 60, 60, 60, 60, 60]),
            }),
        ).toBe("wait");
        // 窗口对了，但水位还没确认（在飞回包可能是提交前发出的）。
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: tx(),
                watermark: { issuedRequestSeq: 6, appliedRequestSeq: 4 },
            }),
        ).toBe("wait");
        // 窗口缺失。
        expect(
            shouldReleaseParamBlockPreview({ ...BASE, transaction: tx(), paramView: null }),
        ).toBe("wait");
    });

    it("releases only when all three pieces of evidence agree", () => {
        expect(shouldReleaseParamBlockPreview({ ...BASE, transaction: tx() })).toBe("release");
    });

    it("★ keeps the preview alive while the committed segments are still unknown", () => {
        // 过渡手柄：pointerup 时宣布"要写曲线"，片段要等异步读基准之后才知道。
        // 这段窗口里几何根本没变 —— 若按几何相等释放，就是"先回弹"。
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: tx({ committedSegments: null }),
            }),
        ).toBe("wait");
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: tx({ committedSegments: null }),
                now: PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS + 1,
            }),
        ).toBe("timeout");
    });

    it("releases a layout-only commit as soon as the geometry is settled", () => {
        const layoutOnly = tx({ committedSegments: undefined });
        expect(
            shouldReleaseParamBlockPreview({ ...BASE, transaction: layoutOnly, paramView: null }),
        ).toBe("release");
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: layoutOnly,
                geometrySettled: false,
            }),
        ).toBe("wait");
    });

    it("waits for the gesture phase to end before considering anything", () => {
        expect(
            shouldReleaseParamBlockPreview({ ...BASE, transaction: tx({ phase: "gesture" }) }),
        ).toBe("wait");
    });

    it("times out instead of locking the preview forever", () => {
        const stale = tx({ startedAt: 0 });
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: stale,
                now: PARAM_BLOCK_PREVIEW_RELEASE_TIMEOUT_MS + 1,
                authorityRevision: 10,
            }),
        ).toBe("timeout");
    });

    it("honours an injected timeout", () => {
        expect(
            shouldReleaseParamBlockPreview({
                ...BASE,
                transaction: tx(),
                now: 50,
                timeoutMs: 10,
                authorityRevision: 10,
            }),
        ).toBe("timeout");
    });
});

describe("isParamBlockPreviewGeometrySettled", () => {
    function specWith(
        blocks: { id: string; startSec: number; endSec: number; selected?: boolean }[],
    ): PianoRollParamBlockSpec {
        return {
            ...spec(),
            blocks: blocks.map((block) => ({
                ...block,
                selected: block.selected ?? false,
                grouped: false,
                topPx: 0,
                heightPx: 10,
            })),
        };
    }

    it("is true once every preview block has landed with the same span", () => {
        expect(
            isParamBlockPreviewGeometrySettled({
                transaction: tx({
                    kind: "resize",
                    spec: specWith([{ id: "a", startSec: 0, endSec: 1 }]),
                }),
                blocks: [{ id: "a", startSec: 0, endSec: 1 }],
            }),
        ).toBe(true);
    });

    it("is false while a block still has the old span", () => {
        expect(
            isParamBlockPreviewGeometrySettled({
                transaction: tx({
                    kind: "resize",
                    spec: specWith([{ id: "a", startSec: 0, endSec: 2 }]),
                }),
                blocks: [{ id: "a", startSec: 0, endSec: 1 }],
            }),
        ).toBe(false);
    });

    it("only looks at the selected block for a create preview", () => {
        expect(
            isParamBlockPreviewGeometrySettled({
                transaction: tx({
                    kind: "create",
                    spec: specWith([
                        { id: "a", startSec: 0, endSec: 1, selected: true },
                        { id: "b", startSec: 5, endSec: 6 },
                    ]),
                }),
                blocks: [{ id: "a", startSec: 0, endSec: 1 }],
            }),
        ).toBe(true);
    });

    it("requires the transition width to match for a transition preview", () => {
        const base = specWith([{ id: "a", startSec: 0, endSec: 1 }]);
        const withHandle = {
            ...base,
            blocks: [
                {
                    ...base.blocks[0],
                    transitionHandle: { leftId: "a", rightId: "b", sec: 1, widthSec: 0.25 },
                },
            ],
        };
        expect(
            isParamBlockPreviewGeometrySettled({
                transaction: tx({ kind: "transition", spec: withHandle }),
                blocks: [{ id: "a", startSec: 0, endSec: 1, transitionSec: 0.25 }],
            }),
        ).toBe(true);
        expect(
            isParamBlockPreviewGeometrySettled({
                transaction: tx({ kind: "transition", spec: withHandle }),
                blocks: [{ id: "a", startSec: 0, endSec: 1, transitionSec: 0.1 }],
            }),
        ).toBe(false);
    });
});
