// @vitest-environment jsdom
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { EMPTY_PARAM_BLOCK_STATE } from "../paramBlocks";
import type { ParamBlock, ParamBlockState } from "../paramBlocks";
import type { ParamBlockView } from "../paramBlockTypes";
import type { ParamViewSegment } from "../types";
import { useParamBlockTransitionPersistence } from "./useParamBlockTransitionPersistence";

const mocks = vi.hoisted(() => ({
    fetchFullResCurve: vi.fn(),
    setParamBlockTransition: vi.fn(),
    commitPlan: vi.fn(),
    stampCurveCommit: vi.fn(),
}));

/** 最近一次 plan 的准备结果（由 commitPlan 的 mock 填入）。 */
const prepared: {
    current: {
        patches?: readonly { param: string; startFrame: number; values: readonly number[] }[];
        authority?: { expectedCurveRevision: number } | null;
        checkpoint?: boolean;
    } | null;
} = { current: null };

vi.mock("../selectionEditData", async (importOriginal) => ({
    ...(await importOriginal<typeof import("../selectionEditData")>()),
    fetchFullResCurve: mocks.fetchFullResCurve,
}));
vi.mock("../paramBlocks", async (importOriginal) => ({
    ...(await importOriginal<typeof import("../paramBlocks")>()),
    setParamBlockTransition: mocks.setParamBlockTransition,
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const LEFT: ParamBlock = {
    id: "left",
    rootTrackId: "root",
    kind: "manual",
    startSec: 0,
    endSec: 1,
    members: [],
    averageValue: 60,
};
const RIGHT: ParamBlock = { ...LEFT, id: "right", startSec: 1, endSec: 2, averageValue: 62 };
const VIEW: ParamViewSegment = {
    key: "v2|root|pitch|0|4|1",
    framePeriodMs: 5,
    startFrame: 0,
    stride: 1,
    referenceKind: "source_curve",
    orig: [60],
    edit: [60],
};

const api: {
    persistPitchTransitionCurve:
        | ((
              left: ParamBlock,
              right: ParamBlock,
              widthSec: number,
              options?: { restoreOriginal?: boolean },
          ) => Promise<void>)
        | null;
} = { persistPitchTransitionCurve: null };
let host: HTMLDivElement;
let root: Root;

function Harness() {
    const state: ParamBlockState = EMPTY_PARAM_BLOCK_STATE;
    const hook = useParamBlockTransitionPersistence({
        rootTrackId: "root",
        editParam: "pitch",
        blockPersistence: { commitPlan: mocks.commitPlan },
        paramBlockStateRef: { current: state },
        paramBlockViewRef: { current: {} as ParamBlockView },
        paramBlockSourceClipsRef: { current: [] },
        paramViewRef: { current: VIEW },
        setParamBlockState: vi.fn(),
        setParamView: vi.fn(),
        onParamCommitSucceeded: vi.fn(),
        bumpRefreshToken: vi.fn(),
        invalidate: vi.fn(),
        stampCurveCommit: mocks.stampCurveCommit,
        visualTransactionRef: { current: { ownedEditRevision: null } },
        dragSnapshotRef: { current: null },
        resizeRef: { current: null },
    });
    useEffect(() => {
        api.persistPitchTransitionCurve = hook.persistPitchTransitionCurve;
    }, [hook.persistPitchTransitionCurve]);
    return null;
}

beforeEach(() => {
    mocks.fetchFullResCurve.mockReset();
    mocks.setParamBlockTransition.mockReset();
    mocks.commitPlan.mockReset();
    prepared.current = null;
    mocks.fetchFullResCurve.mockResolvedValue({
        startFrame: 180,
        values: [60, 60, 60, 60],
        authorityRevision: 11,
        geometryRevision: 3,
        geometrySignature: "geo",
    });
    mocks.setParamBlockTransition.mockImplementation((input: { state: ParamBlockState }) => ({
        ok: true,
        state: {
            ...input.state,
            editSet: { ...input.state.editSet, revision: input.state.editSet.revision + 1 },
        },
    }));
    mocks.commitPlan.mockImplementation(async (plan: { prepare: () => Promise<unknown> }) => {
        const result = await plan.prepare();
        prepared.current = result as typeof prepared.current;
        return result;
    });
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

async function render(): Promise<void> {
    await act(async () => {
        root.render(<Harness />);
    });
}

describe("useParamBlockTransitionPersistence", () => {
    it("sends the curve authority with a normal transition commit", async () => {
        await render();
        await act(async () => {
            await api.persistPitchTransitionCurve?.(LEFT, RIGHT, 0.1);
        });
        expect(mocks.commitPlan).toHaveBeenCalledTimes(1);
        expect(mocks.commitPlan.mock.calls[0][0]).toMatchObject({ kind: "transition" });
        const plan = prepared.current;
        expect(plan).not.toBeNull();
        expect(plan?.checkpoint).toBe(true);
        expect((plan?.patches ?? []).length).toBeGreaterThan(0);
        expect(plan?.authority).toEqual({
            expectedCurveRevision: 11,
            expectedGeometryRevision: 3,
            expectedGeometrySignature: "geo",
        });
    });

    it("commits restore as one width=0 transaction rather than layout + upload", async () => {
        await render();
        await act(async () => {
            await api.persistPitchTransitionCurve?.(LEFT, RIGHT, 0.1, {
                restoreOriginal: true,
            });
        });
        expect(mocks.commitPlan).toHaveBeenCalledTimes(1);
        const plan = prepared.current;
        expect((plan?.patches ?? []).length).toBe(1);
        expect(plan?.authority?.expectedCurveRevision).toBe(11);
    });

    /*
     * 【读基准在队列槽内的回归锁】`fetchFullResCurve` 必须发生在
     * `commitPlan` 的 prepare 里（= 槽内），而不是调用前。否则它与写入之间
     * 只要有一次别的写入，`expectedCurveRevision` 就过期了。
     */
    it("★ 读基准发生在 commitPlan 的 prepare 里（槽内），而不是调用之前", async () => {
        await render();
        expect(mocks.fetchFullResCurve).not.toHaveBeenCalled();
        await act(async () => {
            await api.persistPitchTransitionCurve?.(LEFT, RIGHT, 0.1);
        });
        expect(mocks.fetchFullResCurve).toHaveBeenCalledTimes(1);
        // prepare 被调用的时刻 fetch 尚未发生：顺序由 commitPlan 掌握。
        const callOrder = mocks.commitPlan.mock.invocationCallOrder[0];
        const fetchOrder = mocks.fetchFullResCurve.mock.invocationCallOrder[0];
        expect(callOrder).toBeLessThan(fetchOrder);
    });
});
