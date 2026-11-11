// @vitest-environment jsdom
import { act, useEffect, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlockState } from "./paramBlocks";
import { useParamBlockPersistence } from "./useParamBlockPersistence";

const mocks = vi.hoisted(() => ({
    getParamBlockView: vi.fn(),
    setParamBlockEdits: vi.fn(),
    viewUpdatedListener: null as
        null | ((event: { payload: { rootTrackId: string; view?: unknown } }) => void),
}));

vi.mock("../../../services/api", () => ({
    timelineApi: {
        getParamBlockView: (root: string) => mocks.getParamBlockView(root),
        setParamBlockEdits: (input: unknown) => mocks.setParamBlockEdits(input),
    },
}));
vi.mock("../../../services/hostEvents", () => ({
    listen: async (
        _event: string,
        listener: (event: { payload: { rootTrackId: string; view?: unknown } }) => void,
    ) => {
        mocks.viewUpdatedListener = listener;
        return () => {
            mocks.viewUpdatedListener = null;
        };
    },
}));

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
        edits: { schemaVersion: 2 as const, revision: editRevision, blocks: [] },
        curveRevision: 0,
    };
}

let host: HTMLDivElement;
let root: Root;
/** 观测点：由 effect 写入（渲染期不得改外部变量）。 */
const probe: { state: ParamBlockState } = { state: EMPTY_PARAM_BLOCK_STATE };
/** 测试用的驱动口（同样由 effect 写入）。 */
const api: {
    persistence: ReturnType<typeof useParamBlockPersistence> | null;
    setState: ((next: ParamBlockState) => void) | null;
} = { persistence: null, setState: null };
const flags = { defer: false };
const props = { historyKey: "0:0" };

function Harness(p: { historyKey: string }) {
    const [state, setState] = useState<ParamBlockState>(EMPTY_PARAM_BLOCK_STATE);
    const persistence = useParamBlockPersistence({
        rootTrackId: "root",
        enabled: true,
        historyKey: p.historyKey,
        state,
        setState,
        deferReadback: () => flags.defer,
    });
    useEffect(() => {
        probe.state = state;
    }, [state]);
    // `persistence` 每次渲染都是新对象 ⇒ 这个 effect 每渲染跑一次，正好让驱动口始终最新。
    useEffect(() => {
        api.persistence = persistence;
        api.setState = setState;
    }, [persistence, setState]);
    return null;
}

async function render(): Promise<void> {
    await act(async () => {
        root.render(<Harness historyKey={props.historyKey} />);
    });
}

/** 模拟一次本地预览：把"内容代次"推到 revision（切分一次会 +3）。 */
function previewTo(revision: number): void {
    act(() => {
        api.setState?.({
            ...probe.state,
            editSet: { schemaVersion: 2, revision, blocks: [] },
        });
    });
}

beforeEach(() => {
    mocks.getParamBlockView.mockReset();
    mocks.setParamBlockEdits.mockReset();
    mocks.viewUpdatedListener = null;
    mocks.getParamBlockView.mockResolvedValue(reply(3));
    flags.defer = false;
    props.historyKey = "0:0";
    probe.state = EMPTY_PARAM_BLOCK_STATE;
    api.persistence = null;
    api.setState = null;
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
     * 手势期间（参数块拖动 / 过渡手柄）不得回读：那时的回包落地会打断正在进行的
     * 本地编辑。被挡下的回读只登记，等手势结束后合并成一次请求。
     */
    it("块手势期间既不落地也不发请求，手势结束后合并成一次回读", async () => {
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

    it("比镜像更旧的编辑集不会覆盖当前状态", async () => {
        await render();
        expect(probe.state.editSet.revision).toBe(3);

        mocks.getParamBlockView.mockResolvedValue(reply(1));
        props.historyKey = "2:0";
        await render();
        expect(probe.state.editSet.revision).toBe(3);
    });

    /*
     * 【撤销切分无效的根因回归】本地编辑集的 revision 是**内容代次**（前端每落一个
     * 块 +1，一次切分 +3），服务端的是**权威代次**（每次提交 +1）。旧实现拿它们比
     * 大小，于是切分之后本地代次永远领先，此后所有权威回写（撤销 / 重做 / 回读）
     * 都被静默丢弃。
     */
    it("本地预览代次领先时，提交回包仍然落地", async () => {
        await render();
        previewTo(6);
        expect(probe.state.editSet.revision).toBe(6);

        mocks.setParamBlockEdits.mockResolvedValue(reply(4));
        await act(async () => {
            await api.persistence?.commit(probe.state);
        });
        expect(probe.state.editSet.revision).toBe(4);
    });

    /*
     * 【代次相等 ≠ 内容相同】一次 `placeUserBlock` 的预览恰好把本地代次推到服务端的
     * 下一个代次（3 → 4，而提交回包也是 4）。若拿 revision 相等当"无需落地"的短路，
     * 服务端那份（已排序 / 规范化的块）就会被跳过，本地继续用预览那一份。
     */
    it("预览代次恰好等于服务端代次时，仍然采用服务端的块", async () => {
        await render();
        previewTo(4);
        expect(probe.state.editSet.revision).toBe(4);
        expect(probe.state.editSet.blocks).toHaveLength(0);

        const serverBlock = {
            id: "ublock",
            geometry: {
                kind: "take" as const,
                members: [
                    {
                        clipId: "clip",
                        takeId: "take",
                        cycle: 0,
                        sourceStartSec: 0,
                        sourceEndSec: 1,
                        label: "manual" as const,
                    },
                ],
            },
            createdSeq: 1,
        };
        mocks.setParamBlockEdits.mockResolvedValue({
            ...reply(4),
            edits: { schemaVersion: 2, revision: 4, blocks: [serverBlock] },
        });
        await act(async () => {
            await api.persistence?.commit(probe.state);
        });
        expect(probe.state.editSet.blocks).toHaveLength(1);
    });

    it("撤销后的回读能盖掉更靠前的本地代次", async () => {
        await render();
        previewTo(6);

        mocks.getParamBlockView.mockResolvedValue(reply(7));
        props.historyKey = "1:0";
        await render();
        expect(probe.state.editSet.revision).toBe(7);
    });

    /*
     * 【CAS 基准过期的一次性自愈】镜像可能落后于服务端（带外写入 / 撤销推进代次而
     * 回读尚未落地）。这时先刷新镜像再原样重试一次；第二次仍冲突才报错。
     */
    it("CAS 冲突后刷新镜像并重试一次", async () => {
        await render();
        mocks.getParamBlockView.mockResolvedValue(reply(9));
        mocks.setParamBlockEdits
            .mockRejectedValueOnce(new Error("Conflict: parameter block edits changed"))
            .mockResolvedValueOnce(reply(10));

        await act(async () => {
            await api.persistence?.commit(probe.state);
        });

        expect(mocks.setParamBlockEdits).toHaveBeenCalledTimes(2);
        const second = mocks.setParamBlockEdits.mock.calls[1][0] as {
            expectedEditRevision: number;
        };
        expect(second.expectedEditRevision).toBe(9);
        expect(probe.state.editSet.revision).toBe(10);
    });

    /*
     * 【带外写入必须回写镜像】`restore_game_param_blocks` 绕过 `commit` 直接推进了
     * 服务端代次。钩子若不知道，下一次提交就会以过期基准被 CAS 拒绝（"还原之后
     * 编辑点了没反应"）。`adopt` 把回包交进同一条落地路径，不需要额外 IPC。
     */
    it("adopt 采用带外写入的回包，下一次提交用新基准", async () => {
        await render();
        act(() => api.persistence?.adopt("root", reply(8)));
        expect(probe.state.editSet.revision).toBe(8);

        mocks.setParamBlockEdits.mockResolvedValue(reply(9));
        await act(async () => {
            await api.persistence?.commit(probe.state);
        });
        const sent = mocks.setParamBlockEdits.mock.calls[0][0] as {
            expectedEditRevision: number;
        };
        expect(sent.expectedEditRevision).toBe(8);
        expect(probe.state.editSet.revision).toBe(9);
    });

    it("forwards curve authority and rejects curve patches without it", async () => {
        await render();
        mocks.setParamBlockEdits.mockResolvedValue(reply(4));
        await act(async () => {
            await api.persistence?.commit(
                probe.state,
                true,
                [{ param: "pitch", startFrame: 0, values: [60, 61] }],
                {
                    expectedCurveRevision: 12,
                    expectedGeometryRevision: 3,
                    expectedGeometrySignature: "sig",
                },
            );
        });
        const sent = mocks.setParamBlockEdits.mock.calls[0][0] as {
            expectedCurveRevision?: number;
            expectedGeometryRevision?: number;
            expectedGeometrySignature?: string;
        };
        expect(sent.expectedCurveRevision).toBe(12);
        expect(sent.expectedGeometryRevision).toBe(3);
        expect(sent.expectedGeometrySignature).toBe("sig");

        expect(() =>
            api.persistence?.commit(probe.state, true, [
                { param: "pitch", startFrame: 0, values: [60] },
            ]),
        ).toThrow("expectedCurveRevision");
    });

    it("replays an in-flight view update after the commit queue drains", async () => {
        await render();
        mocks.getParamBlockView.mockResolvedValue(reply(8));
        let resolveCommit: ((value: unknown) => void) | null = null;
        mocks.setParamBlockEdits.mockImplementation(
            () =>
                new Promise((resolve) => {
                    resolveCommit = resolve;
                }),
        );
        let commitPromise: Promise<unknown> = Promise.resolve();
        await act(async () => {
            commitPromise = api.persistence?.commit(probe.state) ?? Promise.resolve();
            await Promise.resolve();
        });
        // 带外更新在提交进行中到达：必须登记，不能在 finally 后丢失。
        act(() => {
            mocks.viewUpdatedListener?.({
                payload: { rootTrackId: "root", view: reply(8).view },
            });
        });
        await act(async () => {
            resolveCommit?.(reply(4));
            await commitPromise;
        });
        await act(async () => {
            await Promise.resolve();
        });
        expect(mocks.getParamBlockView).toHaveBeenCalled();
        expect(probe.state.editSet.revision).toBe(8);
    });

    it("非冲突错误不重试", async () => {
        await render();
        mocks.setParamBlockEdits.mockRejectedValue(new Error("boom"));
        await act(async () => {
            await api.persistence?.commit(probe.state).catch(() => undefined);
        });
        expect(mocks.setParamBlockEdits).toHaveBeenCalledTimes(1);
    });

    it("★ loads a missing baseline before writing instead of dropping the write", async () => {
        // 回读全部失败 ⇒ 镜像里没有这个根。旧实现此时直接 return，调用方以为写成功了。
        mocks.getParamBlockView.mockRejectedValue(new Error("ipc down"));
        await render();
        mocks.getParamBlockView.mockResolvedValue(reply(7));
        mocks.setParamBlockEdits.mockResolvedValue(reply(8));

        await act(async () => {
            await api.persistence?.commit(probe.state, true);
        });

        expect(mocks.setParamBlockEdits).toHaveBeenCalledTimes(1);
        expect(
            (mocks.setParamBlockEdits.mock.calls[0][0] as { expectedEditRevision: number })
                .expectedEditRevision,
        ).toBe(7);
    });

    it("★ reports a transport failure when the baseline cannot be loaded", async () => {
        mocks.getParamBlockView.mockRejectedValue(new Error("ipc down"));
        await render();

        let caught: unknown = null;
        await act(async () => {
            try {
                await api.persistence?.commit(probe.state, true);
            } catch (error) {
                caught = error;
            }
        });

        expect(caught).toMatchObject({ kind: "transport", code: "not_ready" });
        expect(mocks.setParamBlockEdits).not.toHaveBeenCalled();
    });
});
