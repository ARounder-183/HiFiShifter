// @vitest-environment jsdom
import { act, useEffect, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlockState } from "./paramBlocks";
import { useParamBlockPersistence } from "./useParamBlockPersistence";

const mocks = vi.hoisted(() => ({
    getParamBlockView: vi.fn(),
    setParamBlockEdits: vi.fn(),
}));

vi.mock("../../../services/api", () => ({
    timelineApi: {
        getParamBlockView: (root: string) => mocks.getParamBlockView(root),
        setParamBlockEdits: (input: unknown) => mocks.setParamBlockEdits(input),
    },
}));
vi.mock("../../../services/hostEvents", () => ({ listen: async () => () => {} }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function reply(editRevision: number, framePeriodMs = 5) {
    return {
        view: {
            schemaVersion: 1 as const,
            rootTrackId: "root",
            editRevision,
            layoutKey: "layout",
            framePeriodMs,
            blocks: [],
            orphanCount: 0,
            analysisPending: false,
        },
        edits: { schemaVersion: 1 as const, revision: editRevision, blocks: [] },
        curveRevision: 0,
    };
}

function clip(overrides: Partial<ClipInfo> = {}): ClipInfo {
    return {
        id: "clip",
        trackId: "root",
        startSec: 0,
        lengthSec: 2,
        sourceStartSec: 0,
        sourceEndSec: 2,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        activeTakeId: "take",
        takes: [
            {
                id: "take",
                sourceStartSec: 0,
                sourceEndSec: 2,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
            },
        ],
        ...overrides,
    } as ClipInfo;
}

let host: HTMLDivElement;
let root: Root;
/** 观测点：由 effect 写入（渲染期不得改外部变量）。 */
const probe: { state: ParamBlockState } = { state: EMPTY_PARAM_BLOCK_STATE };
const flags = { defer: false };
const props = { clips: [clip()] as readonly ClipInfo[], historyKey: "0:0" };

function Harness(p: { clips: readonly ClipInfo[]; historyKey: string }) {
    const [state, setState] = useState<ParamBlockState>(EMPTY_PARAM_BLOCK_STATE);
    useParamBlockPersistence({
        rootTrackId: "root",
        clips: p.clips,
        enabled: true,
        historyKey: p.historyKey,
        state,
        setState,
        deferReadback: () => flags.defer,
    });
    useEffect(() => {
        probe.state = state;
    }, [state]);
    return null;
}

async function render(): Promise<void> {
    await act(async () => {
        root.render(<Harness clips={props.clips} historyKey={props.historyKey} />);
    });
}

beforeEach(() => {
    mocks.getParamBlockView.mockReset();
    mocks.setParamBlockEdits.mockReset();
    mocks.getParamBlockView.mockResolvedValue(reply(3));
    flags.defer = false;
    props.clips = [clip()];
    props.historyKey = "0:0";
    probe.state = EMPTY_PARAM_BLOCK_STATE;
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

describe("useParamBlockPersistence", () => {
    it("挂载后把权威编辑集与帧周期写进状态", async () => {
        await render();
        expect(probe.state.editSet.revision).toBe(3);
        expect(probe.state.rootTrackId).toBe("root");
        expect(probe.state.framePeriodMs).toBe(5);
    });

    /*
     * 手势期间（Clip 拖动 / 参数块拖动 / 内核手势）不得回读：那时的回包带着
     * 旧几何，落地就会让块在新旧位置之间来回跳。被挡下的回读只登记，等手势
     * 结束后合并成一次请求。
     */
    it("手势期间既不落地也不发请求，手势结束后合并成一次回读", async () => {
        await render();
        const before = mocks.getParamBlockView.mock.calls.length;

        flags.defer = true;
        await render();
        props.historyKey = "1:0";
        mocks.getParamBlockView.mockResolvedValue(reply(4));
        await render();
        expect(mocks.getParamBlockView.mock.calls.length).toBe(before);
        expect(probe.state.editSet.revision).toBe(3);

        flags.defer = false;
        await render();
        expect(mocks.getParamBlockView.mock.calls.length).toBe(before + 1);
        expect(probe.state.editSet.revision).toBe(4);
    });

    it("比本地更旧的编辑集不会覆盖当前状态", async () => {
        await render();
        expect(probe.state.editSet.revision).toBe(3);

        mocks.getParamBlockView.mockResolvedValue(reply(1));
        props.historyKey = "2:0";
        await render();
        expect(probe.state.editSet.revision).toBe(3);
    });

    /*
     * 这是闪烁的根因回归：`historyKey` 只跟撤销深度走。Clip 几何变化不再
     * 触发回读 —— 块视图是 `(clips, editSet, framePeriodMs)` 的派生值，
     * 不需要向任何一方"确认"几何。
     */
    it("Clip 几何变化不触发回读", async () => {
        await render();
        const before = mocks.getParamBlockView.mock.calls.length;

        props.clips = [clip({ startSec: 0.5, lengthSec: 1.5 })];
        await render();
        expect(mocks.getParamBlockView.mock.calls.length).toBe(before);

        props.clips = [clip({ activeTakeId: "other" })];
        await render();
        expect(mocks.getParamBlockView.mock.calls.length).toBe(before);
    });

    it("飞行期间几何变了：回包不落地，改为补一次回读", async () => {
        await render();
        const before = mocks.getParamBlockView.mock.calls.length;

        let release: ((value: unknown) => void) | null = null;
        mocks.getParamBlockView.mockImplementationOnce(
            () =>
                new Promise((resolve) => {
                    release = resolve;
                }),
        );
        mocks.getParamBlockView.mockResolvedValue(reply(7));
        props.historyKey = "3:0";
        await render();
        expect(mocks.getParamBlockView.mock.calls.length).toBe(before + 1);

        // 请求还在飞，几何已经变了。
        props.clips = [clip({ startSec: 1 })];
        await render();
        await act(async () => {
            release?.(reply(7));
        });
        expect(probe.state.editSet.revision).toBe(3);

        // 下一次渲染发现有待回读，补一次并落地。
        await render();
        expect(mocks.getParamBlockView.mock.calls.length).toBe(before + 2);
        expect(probe.state.editSet.revision).toBe(7);
    });
});
