import {
    useCallback,
    useEffect,
    useLayoutEffect,
    useRef,
    useState,
    type Dispatch,
    type SetStateAction,
} from "react";
import { timelineApi } from "../../../services/api";
import { listen } from "../../../services/hostEvents";
import { backendDetail } from "../../../services/invoke";
import {
    normalizeParamBlockEditSet,
    normalizeParamBlockEditSetWithDiagnostics,
    type ParamBlockEditSet,
    type ParamBlockView,
} from "./paramBlockTypes";
import type { ParamBlockState } from "./paramBlocks";
import { trackParamBlockCommit, type ParamBlockCommitKind } from "./paramBlockCommitCoordinator";
import {
    assertCurveCommitAuthority,
    type ParamCurveCommitAuthority,
} from "./paramBlockCommitPolicy";

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

/** 后端拒绝整集替换时的原文（见 `ParamBlockEditSet::replace`）。 */
const EDIT_SET_CONFLICT = "parameter block edits changed";

function isEditSetConflict(error: unknown): boolean {
    return (backendDetail(error) ?? String(error)).includes(EDIT_SET_CONFLICT);
}

/**
 * 参数块编辑集的持久化。
 *
 * 【职责收窄】它只负责"编辑集"这一件事：读（`get_param_block_view`）、写
 * （`set_param_block_edits`）、以及把手势期间的异步回包挡在门外。块几何不在
 * 它的管辖范围内 —— 那是 `composeParamBlockView` 的纯函数输出，所以本钩子
 * **不需要** `clips`。
 *
 * 【权威镜像】`server` 是**唯一**的权威代次来源。写路径的 CAS 基准
 * （`expectedEditRevision`）读的就是它，因此：
 *
 * - 任何**不经过 `commit`** 的后端写入（`restore_game_param_blocks` 等）
 *   必须调 `adopt` 把回包交回来，否则下一次提交的基准是过期的；
 * - 本地编辑集的 `revision` 是**内容代次**（前端每落一个块 +1），服务端的是
 *   **权威代次**（每次提交 +1）—— 两者**不可比**。唯一的陈旧判据是
 *   {@link adoptSnapshot} 里"服务端 vs 服务端"的单调闸。
 *
 * 【为什么回读仍然存在】视图可以派生，**编辑集不能**：撤销 / 重做 / 分析回填 /
 * 宿主侧改动都只存在于后端。
 */
export function useParamBlockPersistence(args: {
    rootTrackId: string | null;
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

    /**
     * 落地到 React 状态。
     *
     * 【为什么没有"本地更新就不倒退"的判据】本地编辑集的 `revision` 是**内容
     * 代次**（每落一个块 +1），服务端的是**权威代次**（每次提交 +1）—— 一次切分会让
     * 前者一次 +3（物化 + 左右两半）。拿它们比较，切分之后本地代次永远跑在前面，
     * 于是撤销 / 重做 / 回读的权威结果**全部被丢弃**（用户看到"撤销无法撤销切分"）。
     *
     * 唯一有效的陈旧判据在 {@link adoptSnapshot}：两侧都是服务端代次。落地的是已经
     * 过那道闸的结果，无条件写入。
     *
     * 唯一的短路是**对象同一性**（`state.editSet === snapshot.edits`）：只有"把手上
     * 这一份再写一次"（`flushReadback`）才会命中，返回原对象省掉一次无意义的重渲染。
     *
     * 【为什么不能拿 revision 相等当短路】本地预览的代次也可能恰好等于服务端的下一个
     * 代次（一次 `placeUserBlock` 的编辑就会这样），那时按 revision 短路会把**服务端
     * 的块**跳过去，本地继续用预览那一份 —— 又是一次"看起来一样、其实没落地"。
     */
    const writeState = useCallback((root: string, snapshot: ParamBlockSnapshot) => {
        if (latest.current.rootTrackId !== root) return;
        latest.current.setState((state) =>
            state.editSet === snapshot.edits && state.framePeriodMs === snapshot.framePeriodMs
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
     * 规范化 + 单调闸 + 写镜像。
     *
     * 闸门只比**服务端代次**：迟到的回包（revision 更旧）不得覆盖更新的权威。
     * 返回 `null` 表示这份回包被丢弃。
     */
    const adoptSnapshot = useCallback(
        (
            root: string,
            reply: { readonly view?: ParamBlockView; readonly edits: ParamBlockEditSet },
        ): ParamBlockSnapshot | null => {
            const normalized = normalizeParamBlockEditSetWithDiagnostics(reply.edits);
            if (normalized.diagnostics.length > 0) {
                console.warn(
                    "[pianoRoll] dropped malformed parameter blocks:",
                    normalized.diagnostics.join(", "),
                );
            }
            const edits = normalized.edits;
            const known = server.current.get(root);
            if (known !== undefined && known.edits.revision > edits.revision) return null;
            const snapshot: ParamBlockSnapshot = {
                edits,
                framePeriodMs: framePeriodOf(reply.view, known?.framePeriodMs),
            };
            server.current.set(root, snapshot);
            return snapshot;
        },
        [],
    );

    /** 把手势 / 提交中的落地推到它们结束之后（只登记，不丢权威）。 */
    const deferLanding = useCallback((root: string): boolean => {
        if ((pending.current.get(root) ?? 0) > 0) {
            // 提交期间落地的回包不能直接丢弃：登记一次合并回读，等队列排空后
            // 用最新权威重置镜像，否则下一次 CAS 基准会过期。
            refetchPending.current.add(root);
            return true;
        }
        if (latest.current.deferReadback?.()) {
            refetchPending.current.add(root);
            return true;
        }
        return false;
    }, []);

    /**
     * 回包落地：镜像 + 状态。
     *
     * 镜像**永远**更新（它就是权威本身，与手势无关）；状态则要等手势 / 提交结束。
     * 旧实现在这里无条件写 `view`，那正是"松手后闪一下回到旧位置"的来源；
     * 现在状态里根本没有几何。
     */
    const applyReply = useCallback(
        (
            root: string,
            reply: { readonly view?: ParamBlockView; readonly edits: ParamBlockEditSet },
        ) => {
            const snapshot = adoptSnapshot(root, reply);
            if (snapshot === null) return;
            if (latest.current.rootTrackId !== root) return;
            if (deferLanding(root)) return;
            writeState(root, snapshot);
        },
        [adoptSnapshot, deferLanding, writeState],
    );

    /** 读一次权威编辑集。手势期间只登记，不发请求。 */
    const readback = useCallback(
        async (root: string) => {
            if ((pending.current.get(root) ?? 0) > 0) return;
            if (latest.current.deferReadback?.()) {
                refetchPending.current.add(root);
                return;
            }
            try {
                const reply = await timelineApi.getParamBlockView(root);
                if (latest.current.rootTrackId !== root) return;
                applyReply(root, reply);
            } catch (error) {
                publishError(error);
            }
        },
        [applyReply, publishError],
    );

    /** 只刷新镜像（给提交冲突后的重试用）：不受手势 / 提交中守卫影响。 */
    const refreshAuthority = useCallback(
        async (root: string): Promise<ParamBlockSnapshot | null> => {
            try {
                const reply = await timelineApi.getParamBlockView(root);
                return adoptSnapshot(root, reply);
            } catch (error) {
                publishError(error);
                return null;
            }
        },
        [adoptSnapshot, publishError],
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
                if ((pending.current.get(root) ?? 0) > 0) {
                    // 提交中的带外更新：先登记，队列排空后再回读。直接 return
                    // 会让服务端代次前进而镜像停在旧值，下一次提交必被 CAS 拒绝。
                    refetchPending.current.add(root);
                    return;
                }
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
        if ((pending.current.get(root) ?? 0) > 0) {
            // A history change during an in-flight commit must be replayed
            // after the queue drains, not silently dropped.
            refetchPending.current.add(root);
            return;
        }
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
            authority?: ParamCurveCommitAuthority | null,
            kind: ParamBlockCommitKind = "layout",
        ) => {
            /*
             * 有曲线补丁却没有权威 revision，后端必定返回 `Conflict: parameter curve
             * changed`。把这条契约放在队列入口，任何新调用点在开发期就会炸出
             * 明确堆栈，而不是变成用户看到的“参数块操作失败”。
             */
            assertCurveCommitAuthority({ hasPatches: curvePatches.length > 0, authority });
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
                    expectedCurveRevision: authority?.expectedCurveRevision,
                    expectedGeometryRevision: authority?.expectedGeometryRevision,
                    expectedGeometrySignature: authority?.expectedGeometrySignature,
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
            /*
             * 【冲突重试一次】CAS 基准是镜像里的服务端代次。镜像过期只有两种成因：
             * 带外写入（`restore_game_param_blocks`）与撤销 / 重做推进代次而回读
             * 尚未落地。两者都不代表"别人改了内容"，所以先刷新镜像再原样重试一次；
             * 第二次仍冲突就说明真的有人在并发改，交给调用方报错（整集覆盖本来就是
             * "最后写入者胜"，重试不会放大任何风险）。
             */
            const executeWithRetry = async (): Promise<ParamBlockSnapshot> => {
                try {
                    return await execute();
                } catch (error) {
                    if (!isEditSetConflict(error)) throw error;
                    if ((await refreshAuthority(root)) === null) throw error;
                    return execute();
                }
            };
            const job = (queues.current.get(root) ?? Promise.resolve())
                .catch(() => undefined)
                .then(executeWithRetry);
            const complete = trackParamBlockCommit(kind, root, job)
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
        [publishError, refreshAuthority, writeState],
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
        /**
         * 采用一份来自**其它命令**的权威回包（`restore_game_param_blocks` 等）。
         *
         * 【为什么必须调】写入的 CAS 基准是镜像里的服务端代次；带外写入让服务端代次
         * 前进却不告诉镜像，下一次提交就会以过期基准被拒（用户看到"点了没反应"）。
         * 这里把回包直接交进同一条落地路径，不需要多一次 IPC。
         */
        adopt: (
            root: string,
            reply: { readonly view?: ParamBlockView; readonly edits: ParamBlockEditSet },
        ) => {
            applyReply(root, reply);
        },
        flushReadback,
    };
}
