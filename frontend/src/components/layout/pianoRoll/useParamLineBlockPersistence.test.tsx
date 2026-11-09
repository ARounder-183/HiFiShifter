// @vitest-environment jsdom
import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, test, vi } from "vitest";
import { useParamLineBlockPersistence } from "./useParamLineBlockPersistence";
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlockState } from "./paramBlocks";
import type { ParamLineBlockSet } from "./paramLineBlocks";
import { timelineApi } from "../../../services/api";
vi.mock("../../../services/api", () => ({
    timelineApi: {
        getParamLineBlocks: vi.fn(),
        getParamBlockMerges: vi.fn(),
        setParamLineBlocks: vi.fn(),
    },
}));
vi.mock("../../../services/hostEvents", () => ({ listen: vi.fn(async () => () => {}) }));
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const empty: ParamLineBlockSet = {
    schemaVersion: 1,
    revision: 2,
    framePeriodMs: 5,
    blocks: [],
    gameBaseline: { revision: 0, segments: [] },
};
let api: ReturnType<typeof useParamLineBlockPersistence>;
let state: ParamBlockState;
let updateLocalState: (state: ParamBlockState) => void;
let deferReadback = false;
let root: Root | undefined;
let host: HTMLDivElement;
const onCommitted = vi.fn();
async function mount(initialized = true) {
    vi.mocked(timelineApi.getParamLineBlocks).mockResolvedValue({
        ok: true,
        initialized,
        layout: empty,
        curveRevision: 0,
    });
    vi.mocked(timelineApi.getParamBlockMerges).mockResolvedValue({
        records: [],
        intervalEdits: [],
    });
    vi.mocked(timelineApi.setParamLineBlocks).mockImplementation(async (input) => ({
        ok: true,
        initialized: true,
        layout: { ...empty, revision: input.expectedRevision + 1, blocks: input.blocks },
        curveRevision: 0,
    }));
    function Harness() {
        const [value, setValue] = useState(EMPTY_PARAM_BLOCK_STATE);
        state = value;
        updateLocalState = setValue;
        api = useParamLineBlockPersistence({
            rootTrackId: "root",
            clips: [],
            enabled: true,
            historyKey: "0:0",
            state: value,
            setState: setValue,
            deferReadback: () => deferReadback,
            onCommitted,
        });
        return null;
    }
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
        root!.render(<Harness />);
    });
}
afterEach(async () => {
    await act(async () => root?.unmount());
    host?.remove();
    vi.clearAllMocks();
    deferReadback = false;
});
test("a saved empty layout is authoritative and never reruns legacy migration", async () => {
    await mount();
    expect(api.ready).toBe(true);
    expect(state.lineLayout?.blocks).toEqual([]);
    expect(timelineApi.getParamBlockMerges).not.toHaveBeenCalled();
    expect(timelineApi.setParamLineBlocks).not.toHaveBeenCalled();
    expect(onCommitted).not.toHaveBeenCalled();
});
test("migration writes exactly once and does not create an Undo step", async () => {
    await mount(false);
    expect(api.ready).toBe(true);
    expect(timelineApi.setParamLineBlocks).toHaveBeenCalledOnce();
    expect(vi.mocked(timelineApi.setParamLineBlocks).mock.calls[0][0].checkpoint).toBe(false);
    expect(onCommitted).not.toHaveBeenCalled();
});
test("queued local edits preserve preceding commits instead of replacing them", async () => {
    await mount();
    const one = {
        ...state,
        lineEditBase: empty,
        lineLayout: { ...empty, blocks: [{ id: "a", startFrame: 0, endFrameExclusive: 10 }] },
    };
    const two = {
        ...one,
        lineEditBase: one.lineLayout,
        lineLayout: {
            ...empty,
            blocks: [...one.lineLayout.blocks, { id: "b", startFrame: 10, endFrameExclusive: 20 }],
        },
    };
    await act(async () => {
        await Promise.all([api.commit(one), api.commit(two)]);
    });
    expect(state.lineLayout?.blocks.map((block) => block.id)).toEqual(["a", "b"]);
    expect(vi.mocked(timelineApi.setParamLineBlocks).mock.calls[1][0].expectedRevision).toBe(3);
    expect(onCommitted).toHaveBeenCalledTimes(2);
});

test("a failed block write does not report a committed edit", async () => {
    await mount();
    vi.mocked(timelineApi.setParamLineBlocks).mockRejectedValueOnce(new Error("write failed"));
    await act(async () => {
        await expect(
            api.commit({
                ...state,
                lineEditBase: empty,
                lineLayout: {
                    ...empty,
                    blocks: [{ id: "a", startFrame: 0, endFrameExclusive: 10 }],
                },
            }),
        ).rejects.toThrow("write failed");
    });
    expect(onCommitted).not.toHaveBeenCalled();
});
test("a newer analysis event wins over a late acknowledgement", async () => {
    await mount();
    let finish!: (value: Awaited<ReturnType<typeof timelineApi.setParamLineBlocks>>) => void;
    vi.mocked(timelineApi.setParamLineBlocks).mockImplementationOnce(
        () =>
            new Promise((resolve) => {
                finish = resolve;
            }),
    );
    const local = {
        ...state,
        lineEditBase: empty,
        lineLayout: { ...empty, blocks: [{ id: "a", startFrame: 0, endFrameExclusive: 10 }] },
    };
    let saving: ReturnType<typeof api.commit>;
    await act(async () => {
        saving = api.commit(local);
        await Promise.resolve();
    });
    const newer = {
        ...local.lineLayout,
        revision: 4,
        blocks: [...local.lineLayout.blocks, { id: "game", startFrame: 20, endFrameExclusive: 30 }],
    };
    await act(async () => {
        api.receive("root", newer);
        finish({
            ok: true,
            initialized: true,
            layout: { ...local.lineLayout, revision: 3 },
            curveRevision: 0,
        });
        await saving;
    });
    expect(state.lineLayout).toEqual(newer);
});

test("an older ack cannot replace a newer optimistic creation that is still reading its curve", async () => {
    await mount();
    deferReadback = true;
    const one = {
        ...state,
        lineEditBase: empty,
        lineLayout: { ...empty, blocks: [{ id: "first", startFrame: 0, endFrameExclusive: 10 }] },
    };
    const two = {
        ...one,
        lineEditBase: one.lineLayout,
        lineLayout: {
            ...empty,
            blocks: [
                ...one.lineLayout.blocks,
                { id: "second", startFrame: 20, endFrameExclusive: 30 },
            ],
        },
    };
    await act(async () => {
        updateLocalState(two);
        await api.commit(one);
    });
    expect(state.lineLayout?.blocks.map((b) => b.id)).toEqual(["first", "second"]);
    await act(async () => {
        await api.commit(two);
        deferReadback = false;
        api.flushReadback();
    });
    expect(state.lineLayout?.blocks.map((b) => b.id)).toEqual(["first", "second"]);
    expect(state.lineLayout?.revision).toBe(4);
});
