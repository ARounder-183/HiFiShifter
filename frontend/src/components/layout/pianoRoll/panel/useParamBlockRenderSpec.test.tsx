// @vitest-environment jsdom
/**
 * 渲染规格闸门（./useParamBlockRenderSpec）自检。
 *
 * 【本测试守护什么】`structureTick` 是 memo 的唯一显式开闸。它不变时，父组件
 * 重渲染**不得**重算规格 —— 否则每一帧都会重建整份 spec（宿主每帧重传几何）。
 * 它变了就必须重算，否则命令式写入的几何（拖拽快照 / 伸缩预览）会停在旧值，
 * 用户看到「块不动」。
 *
 * 【它不做什么】不测优先级顺序与几何（见 `paramBlockRenderSpec.test.ts` 与
 * 各手势的用例），只测 memo 闸门本身。
 */
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { ParamBlockSpecColors } from "../paramBlockRenderSpec";
import type { ParamBlockEpoch } from "../paramBlockVisualTransaction";
import type { PianoRollParamBlockSpec } from "../kernel/host/pianoRollKernelData";
import { useParamBlockRenderSpec, type ParamBlockRenderSpecInput } from "./useParamBlockRenderSpec";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const EPOCH: ParamBlockEpoch = {
    rootTrackId: "root",
    param: "pitch",
    layoutKey: "layout",
    structureToken: "structure-empty",
    externalHistoryToken: "0:0",
    framePeriodMs: 5,
};

const COLORS: ParamBlockSpecColors = {
    borderRgba: [0, 0, 0, 1],
    selectedRgba: [1, 1, 1, 1],
    groupedRgba: [1, 0, 0, 1],
};

const CANONICAL_BLOCKS: ParamBlockRenderSpecInput["canonicalBlocks"] = [
    {
        id: "b1",
        rootTrackId: "root",
        kind: "manual",
        startSec: 0,
        endSec: 1,
        averageValue: 60,
        manualValue: null,
        members: [],
    },
];

function stableInputs(): Omit<ParamBlockRenderSpecInput, "structureTick"> {
    return {
        layerVisible: true,
        epoch: EPOCH,
        visualTransactionRef: { current: null },
        resizeRef: { current: null },
        dragSnapshotRef: { current: null },
        activeGestureTokenRef: { current: null },
        dragCommitRef: { current: null },
        valueCacheRef: { current: new Map() },
        latestSpecRef: { current: null },
        canonicalBlocks: CANONICAL_BLOCKS,
        paramView: null,
        selectedIds: new Set<string>(),
        editParam: "pitch",
        colors: COLORS,
        bounds: { min: 0, max: 127 },
        valueToY: (_param, value) => 127 - value,
        framePeriodMs: 5,
        curveOwned: false,
        buildDragSnapshotSpec: () => null,
    };
}

let root: Root | null = null;
const seen: { spec: PianoRollParamBlockSpec | null } = { spec: null };

function Harness({ inputs, tick }: { inputs: ParamBlockRenderSpecInput; tick: number }) {
    const spec = useParamBlockRenderSpec({ ...inputs, structureTick: tick });
    useEffect(() => {
        seen.spec = spec;
    }, [spec]);
    return null;
}

async function render(
    inputs: Omit<ParamBlockRenderSpecInput, "structureTick">,
    tick: number,
): Promise<void> {
    await act(async () => {
        root?.render(<Harness inputs={{ ...inputs, structureTick: tick }} tick={tick} />);
    });
}

describe("useParamBlockRenderSpec", () => {
    afterEach(() => {
        act(() => root?.unmount());
        document.body.innerHTML = "";
        root = null;
        seen.spec = null;
    });

    it("★ structureTick 不变时，父组件重渲染复用同一份 spec 引用", async () => {
        const inputs = stableInputs();
        const host = document.createElement("div");
        document.body.append(host);
        root = createRoot(host);
        await render(inputs, 0);
        const first = seen.spec;
        expect(first).not.toBeNull();
        await render(inputs, 0);
        expect(seen.spec).toBe(first);
    });

    it("★ structureTick 前进时必须重算（命令式写入的几何靠它才看得见）", async () => {
        const inputs = stableInputs();
        const host = document.createElement("div");
        document.body.append(host);
        root = createRoot(host);
        await render(inputs, 0);
        const first = seen.spec;
        await render(inputs, 1);
        expect(seen.spec).not.toBe(first);
    });

    it("★ a block without a sampled value still enters the spec with a fallback", async () => {
        const inputs: Omit<ParamBlockRenderSpecInput, "structureTick"> = {
            ...stableInputs(),
            canonicalBlocks: [
                {
                    id: "cut",
                    rootTrackId: "root",
                    kind: "auto",
                    startSec: 0,
                    endSec: 1,
                    averageValue: null,
                    members: [{ clipId: "clip", startSec: 0, endSec: 1 }],
                    sourceNote: 64,
                    fallbackValue: 64,
                    centerPolicy: "fallback",
                    cutStart: true,
                    sourceKey: "clip|take|0",
                },
            ],
        };
        const host = document.createElement("div");
        document.body.append(host);
        root = createRoot(host);
        await render(inputs, 0);
        expect(seen.spec?.blocks).toHaveLength(1);
        expect(seen.spec?.blocks[0]).toMatchObject({
            id: "cut",
            centerPolicy: "fallback",
            fallbackValue: 64,
            cutStart: true,
            sourceKey: "clip|take|0",
        });
    });
});
