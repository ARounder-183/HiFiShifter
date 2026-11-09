import {
    useCallback,
    useEffect,
    useLayoutEffect,
    useRef,
    useState,
    type Dispatch,
    type SetStateAction,
} from "react";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { timelineApi } from "../../../services/api";
import { listen } from "../../../services/hostEvents";
import { backendDetail } from "../../../services/invoke";
import { toParamBlockSourceClip, type ParamBlockSourceClip } from "./paramBlockProjection";
import {
    normalizeParamBlockEditSet,
    type ParamBlockEditSet,
    type ParamBlockView,
} from "./paramBlockTypes";
import type { ParamBlockState } from "./paramBlocks";

/**
 * 请求内一致性签名：覆盖所有会改变投影结果的 Clip/Take 几何与分析输入。
 *
 * 【它不再进 `historyKey`】块几何是派生值（见 `ParamBlockState` 的注释），
 * clip 变了不需要回读编辑集 —— 回读只服务"历史位置变了"。把布局拼进
 * `historyKey` 会让一次 Clip 拖动每帧发一轮 `get_param_block_view`，而那时的
 * 回包带着旧几何，正是闪烁的来源。
 *
 * 现在它只回答一个问题：**这次请求飞行期间，几何变了吗？**
 *
 * 不包含 `editSet.revision`：编辑 reply 已经更新状态；把它放进键会让每次
 * 编辑都多一轮 `get_param_block_view`。
 */
export function paramBlockClipLayoutKey(clips: readonly ParamBlockSourceClip[]): string {
    return clips
        .map((clip) =>
            [
                clip.id,
                clip.trackId,
                clip.startSec,
                clip.lengthSec,
                clip.sourceStartSec ?? 0,
                clip.sourceEndSec ?? 0,
                clip.playbackRate ?? 1,
                clip.reversed ? 1 : 0,
                clip.loopEnabled ? 1 : 0,
                clip.contentDurationSec ?? 0,
                clip.activeTakeId ?? "",
                clip.takes
                    ?.map((take) =>
                        [
                            take.id,
                            take.sourceStartSec,
                            take.sourceEndSec,
                            take.playbackRate,
                            take.reversed ? 1 : 0,
                            take.loopEnabled ? 1 : 0,
                            take.durationSec ?? 0,
                            take.durationFrames ?? 0,
                            take.sourceSampleRate ?? 0,
                            take.syllableSpansSec
                                ?.map((span) => [span.startSec, span.endSec].join(","))
                                .join("|") ?? "",
                            take.midiNoteData
                                ?.map((note) => [note.startSec, note.endSec, note.note].join(","))
                                .join("|") ?? "",
                        ].join(":"),
                    )
                    .join("|") ?? "",
                clip.syllableSpansSec
                    ?.map((span) => [span.startSec, span.endSec].join(","))
                    .join("|") ?? "",
                clip.midiNoteData
                    ?.map((note) => [note.startSec, note.endSec, note.note].join(","))
                    .join("|") ?? "",
            ].join(":"),
        )
        .join(";");
}

/** 一次回读的权威结果：编辑集 + 帧周期。**没有几何** —— 几何永远是派生的。 */
interface ParamBlockSnapshot {
    readonly edits: ParamBlockEditSet;
    readonly framePeriodMs: number;
}

function framePeriodOf(view: ParamBlockView | undefined, fallback: number | undefined): number {
    const value = view?.framePeriodMs;
    if (typeof value === "number" && Number.isFinite(value) && value > 0.1) return value;
    return typeof fallback === "number" && Number.isFinite(fallback) && fallback > 0.1
        ? fallback
        : 5;
}

/**
 * 参数块编辑集的持久化。
 *
 * 【职责收窄】它现在只负责"编辑集"这一件事：读（`get_param_block_view`）、
 * 写（`set_param_block_edits`）、以及把手势期间的异步回包挡在门外。
 * 块几何不在它的管辖范围内 —— 那是 `composeParamBlockView` 的纯函数输出。
 *
 * 【为什么回读仍然存在】视图可以派生，**编辑集不能**：撤销 / 重做 / 分析回填 /
 * 宿主侧改动都只存在于后端。所以回读只是不再参与几何，而不是被删掉。
 */
export function useParamBlockPersistence(args: {
    rootTrackId: string | null;
    clips: readonly ClipInfo[];
    enabled: boolean;
    /** 历史位置变化键（撤销 / 重做深度）。**不含** clip 几何。 */
    historyKey: string;
    state: ParamBlockState;
    setState: Dispatch<SetStateAction<ParamBlockState>>;
    deferReadback?: () => boolean;
    onCommitted?: () => void;
    onError?: (detail: string) => void;
}) {
    const [readyRoot, setReadyRoot] = useState<string | null>(null);
    const latest = useRef(args);
    useLayoutEffect(() => {
        latest.current = args;
    });
    /** 每个根的最近一次权威结果；提交队列用它做 `expectedEditRevision`。 */
    const server = useRef(new Map<string, ParamBlockSnapshot>());
    const queues = useRef(new Map<string, Promise<unknown>>());
    const pending = useRef(new Map<string, number>());
    /** 被 deferral 挡下的回读：按根合并，手势结束后合成一次请求。 */
    const refetchPending = useRef(new Set<string>());

    const publishError = useCallback((error: unknown) => {
        latest.current.onError?.(backendDetail(error) ?? String(error));
    }, []);

    /** 落地到 React 状态；本地编辑集更新时不倒退。 */
    const writeState = useCallback((root: string, snapshot: ParamBlockSnapshot) => {
        if (latest.current.rootTrackId !== root) return;
        latest.current.setState((state) =>
            state.editSet.revision > snapshot.edits.revision
                ? state
                : {
                      ...state,
                      rootTrackId: root,
                      editSet: snapshot.edits,
                      framePeriodMs: snapshot.framePeriodMs,
                  },
        );
    }, []);

    /**
     * 回包落地。
     *
     * 闸门只有一道：**不得比本地编辑集更旧**。几何不需要校验 —— 状态里
     * 没有几何，块一律按当前 clips 现算，所以再旧的回包也不可能把块的位置
     * 打回去。旧实现在这里无条件写 `view`，那正是"松手后闪一下回到旧位置"。
     */
    const applyReply = useCallback(
        (
            root: string,
            reply: { readonly view?: ParamBlockView; readonly edits: ParamBlockEditSet },
        ) => {
            const edits = normalizeParamBlockEditSet(reply.edits);
            const known = server.current.get(root);
            if (known !== undefined && known.edits.revision > edits.revision) return;
            const snapshot: ParamBlockSnapshot = {
                edits,
                framePeriodMs: framePeriodOf(reply.view, known?.framePeriodMs),
            };
            server.current.set(root, snapshot);
            if (latest.current.rootTrackId !== root) return;
            if ((pending.current.get(root) ?? 0) > 0) return;
            if (latest.current.deferReadback?.()) {
                refetchPending.current.add(root);
                return;
            }
            writeState(root, snapshot);
        },
        [writeState],
    );

    const currentLayoutKey = useCallback(
        () => paramBlockClipLayoutKey(latest.current.clips.map(toParamBlockSourceClip)),
        [],
    );

    /** 读一次权威编辑集。手势期间只登记，不发请求。 */
    const readback = useCallback(
        async (root: string) => {
            if ((pending.current.get(root) ?? 0) > 0) return;
            if (latest.current.deferReadback?.()) {
                refetchPending.current.add(root);
                return;
            }
            const requestLayout = currentLayoutKey();
            try {
                const reply = await timelineApi.getParamBlockView(root);
                if (latest.current.rootTrackId !== root) return;
                if (currentLayoutKey() !== requestLayout) {
                    /*
                     * 飞行期间几何变了：这份回包的 `view` 已经对不上当前 clips。
                     * `edits` 与几何无关，但既然几何还在动，就说明用户还没停手 ——
                     * 补一次回读（deferral 会把它排到手势结束），而不是把权威丢掉。
                     */
                    refetchPending.current.add(root);
                    return;
                }
                applyReply(root, reply);
            } catch (error) {
                publishError(error);
            }
        },
        [applyReply, currentLayoutKey, publishError],
    );

    useEffect(() => {
        const root = args.rootTrackId;
        if (!root || !args.enabled) return;
        let disposed = false;
        void timelineApi
            .getParamBlockView(root)
            .then((reply) => {
                if (disposed) return;
                applyReply(root, reply);
                setReadyRoot(root);
            })
            .catch((error) => {
                if (!disposed) publishError(error);
            });
        const subscription = listen<{ rootTrackId: string; view?: ParamBlockView }>(
            "param_block_view_updated",
            ({ payload }) => {
                if (disposed || payload?.rootTrackId !== root) return;
                // 载荷可能来自更早的协议版本（键名 `layout`）；没有 view 就只补一次回读。
                if ((pending.current.get(root) ?? 0) > 0) return;
                const revision = payload.view?.editRevision;
                const known = server.current.get(root)?.edits.revision;
                if (revision !== undefined && known !== undefined && revision <= known) {
                    // 自己提交的回声：编辑集已经在手，不必再问一遍。
                    return;
                }
                void readback(root);
            },
        ).catch(() => () => {});
        return () => {
            disposed = true;
            void subscription.then((off) => off());
        };
    }, [args.rootTrackId, args.enabled, applyReply, publishError, readback]);

    /** 历史位置变了（撤销 / 重做 / 提交落库）才回读编辑集。 */
    useEffect(() => {
        const root = args.rootTrackId;
        if (!root || !args.enabled) return;
        void readback(root);
    }, [args.rootTrackId, args.enabled, args.historyKey, readback]);

    /**
     * 手势结束后的合并回读。
     *
     * 没有依赖数组是刻意的：`deferReadback()` 读的是 ref，它变假不会触发渲染，
     * 所以只能在"下一次渲染"时检查。检查本身是纯内存判断，被挡下时不发 IPC。
     */
    useEffect(() => {
        const root = latest.current.rootTrackId;
        if (root === null) return;
        if (!refetchPending.current.delete(root)) return;
        void readback(root);
    });

    const commit = useCallback(
        (
            state: ParamBlockState,
            checkpoint = true,
            curvePatches: readonly {
                param: string;
                startFrame: number;
                values: readonly number[];
            }[] = [],
            expectedCurveRevision?: number,
            expectedGeometryRevision?: number,
            expectedGeometrySignature?: string,
        ) => {
            const root = state.rootTrackId ?? latest.current.rootTrackId;
            const known = root === null ? undefined : server.current.get(root);
            /*
             * 【为什么必须已有基线】写入是**整集覆盖**。还没读到过这个根的编辑集时
             * 提交本地空集，会把用户已有的块静默抹掉 —— 这是唯一一个能丢数据的
             * 入口，宁可不提交。就绪判据因此从"有没有 view"换成"读没读过"。
             */
            if (!root || known === undefined) return Promise.resolve(undefined);
            const target = normalizeParamBlockEditSet(state.editSet);
            pending.current.set(root, (pending.current.get(root) ?? 0) + 1);
            const execute = async (): Promise<ParamBlockSnapshot> => {
                const current = server.current.get(root);
                const reply = await timelineApi.setParamBlockEdits({
                    rootTrackId: root,
                    expectedEditRevision: current?.edits.revision ?? target.revision,
                    edits: target.blocks,
                    checkpoint,
                    curvePatches,
                    expectedCurveRevision,
                    expectedGeometryRevision,
                    expectedGeometrySignature,
                });
                const snapshot: ParamBlockSnapshot = {
                    edits: normalizeParamBlockEditSet(reply.edits ?? target),
                    framePeriodMs: framePeriodOf(
                        reply.view,
                        current?.framePeriodMs ?? state.framePeriodMs,
                    ),
                };
                server.current.set(root, snapshot);
                return snapshot;
            };
            const job = (queues.current.get(root) ?? Promise.resolve())
                .catch(() => undefined)
                .then(execute);
            const complete = job
                .then((snapshot) => {
                    if (snapshot) latest.current.onCommitted?.();
                    if (
                        snapshot &&
                        latest.current.rootTrackId === root &&
                        !latest.current.deferReadback?.() &&
                        pending.current.get(root) === 1
                    ) {
                        writeState(root, snapshot);
                    }
                    return snapshot;
                })
                .catch((error) => {
                    publishError(error);
                    if (latest.current.rootTrackId === root && pending.current.get(root) === 1) {
                        const current = server.current.get(root);
                        if (current) writeState(root, current);
                    }
                    throw error;
                })
                .finally(() =>
                    pending.current.set(root, Math.max(0, (pending.current.get(root) ?? 1) - 1)),
                );
            queues.current.set(
                root,
                complete.catch(() => undefined),
            );
            return complete;
        },
        [publishError, writeState],
    );

    const persist = useCallback(
        (state: ParamBlockState, checkpoint = true) =>
            commit(state, checkpoint).catch(() => undefined),
        [commit],
    );

    /**
     * 强制把最近一次权威结果写回状态（忽略手势 deferral）。
     *
     * 只给"刚提交完、需要立刻让派生视图吃上新编辑集"的路径用；正常回读走
     * `readback`，它会尊重手势锁。
     */
    const flushReadback = useCallback(() => {
        const root = latest.current.rootTrackId;
        const current = root === null ? undefined : server.current.get(root);
        if (root === null || current === undefined) return;
        writeState(root, current);
    }, [writeState]);

    return {
        ready: readyRoot === args.rootTrackId && args.state.rootTrackId === args.rootTrackId,
        persist,
        commit,
        /**
         * 后端侧改动（分析回填 / 宿主编辑）后的收敛入口。
         *
         * 载荷里的 `view` 不再被采纳 —— 几何是派生的；这里只需要把编辑集重新读一次。
         */
        receive: (root: string) => {
            void readback(root);
        },
        flushReadback,
    };
}
