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
import {
    classifyParamBlockFailure,
    canRebaseParamBlockCommit,
    ParamBlockCommitError,
    paramBlockFailureFeedback,
    type ParamBlockFailureFeedback,
} from "./paramBlockCommitError";
import { validateParamBlockEditSet } from "./paramBlockValidate";
import { uploadFullResCurveSegments } from "./selectionEditData";
import type {
    ParamBlockCurvePatch,
    ParamBlockWritePlan,
    ParamBlockWritePreparation,
} from "./paramBlockWritePlan";
import { frontendLog } from "../../../services/frontendLog";

export type { ParamBlockCurvePatch, ParamBlockWritePlan, ParamBlockWritePreparation };

/**
 * 一次回读 / 一次写入的权威结果：编辑集 + 帧周期 + 后端曲线代次。
 *
 * 【为什么带着 curveRevision】它是 `set_param_block_edits` 的 CAS 单位，也是
 * 「这次写入有没有生效」的唯一可比证据（预览释放判据要用它，见
 * `paramBlockPreviewRelease`）。本地计数器与它不可比 —— 那是「过渡手柄每次都要
 * 等满 2s 超时」的成因。
 */
export interface ParamBlockSnapshot {
    readonly edits: ParamBlockEditSet;
    readonly framePeriodMs: number;
    /** 后端 `param_curve_revision`；0 = 还没落地过任何回包。 */
    readonly curveRevision: number;
}

/** 冲突后最多再准备 / 重试几次。与既有「重试一次」语义一致。 */
const PARAM_BLOCK_REBASE_ATTEMPTS = 1;

function framePeriodOf(view: ParamBlockView | undefined, fallback: number | undefined): number {
    const value = view?.framePeriodMs;
    if (typeof value === "number" && Number.isFinite(value) && value > 0.1) return value;
    return typeof fallback === "number" && Number.isFinite(fallback) && fallback > 0.1
        ? fallback
        : 5;
}

/**
 * 预检：非法编辑集**永远不离开浏览器**。
 *
 * 【为什么放在这里而不是调用点】`commit` 是唯一的写入入口，把预检放在入口意味着
 * 任何新调用点都自动受保护；放在调用点则迟早有人漏写（后端 `Invalid:` 拒绝，
 * 用户看到"随机失败"）。
 */
function assertParamBlockEditSetValid(edits: ParamBlockEditSet): void {
    const result = validateParamBlockEditSet(edits);
    if (result.ok) return;
    throw new ParamBlockCommitError({
        kind: "rejected",
        code: result.code,
        detail: result.detail,
    });
}

interface ParamBlockReply {
    readonly view?: ParamBlockView;
    readonly edits: ParamBlockEditSet;
    readonly curveRevision?: number;
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
 * - 任何**不经过 `commitPlan`** 的后端写入（`restore_game_param_blocks` 等）
 *   必须调 `adopt` 把回包交回来，否则下一次提交的基准是过期的；
 * - 本地编辑集的 `revision` 是**内容代次**（前端每落一个块 +1），服务端的是
 *   **权威代次**（每次提交 +1）—— 两者**不可比**。唯一的陈旧判据是
 *   {@link adoptSnapshot} 里"服务端 vs 服务端"的单调闸。
 *
 * 【为什么回读仍然存在】视图可以派生，**编辑集不能**：撤销 / 重做 / 分析回填 /
 * 宿主侧改动都只存在于后端。
 *
 * 【单一写队列】一个根只有一条队列（`queues`），「读基准 → 算补丁 → 写」整体在
 * 队列槽内。旧实现有**两条**互不相识的队列（这里一条、面板的
 * `paramBlockCommitQueueRef` 一条）共用同一个后端 CAS，交错时后写者必然过期 ——
 * 那是「各种编辑场景下仍有概率报参数块操作失败」的结构性根因。
 *
 * 【违反边界的症状】
 * - 在队列之外读基准再提交：冲突变成常态，用户看到"操作失败"；
 * - 在队列槽内再排一次队：自己等自己，写入永久挂起（拖块 + 新建会死锁）；
 * - 让调用点各自上报失败：单槽反馈被通用红条覆盖，真实原因被吞掉。
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
    /**
     * 写入失败。**只**用于"用户发起的那一笔没写成"。
     *
     * 【为什么与回读分开】回读失败是后台噪声（下一次回读会自愈），把它按
     * danger 弹 15 秒会让用户以为自己的编辑丢了。两者共用同一个回调正是旧实现
     * "偶尔弹一条吓人的红条、但编辑其实成功了"的来源。
     */
    onWriteError?: (failure: ParamBlockFailureFeedback, error: ParamBlockCommitError) => void;
    /** 回读 / 刷新失败；默认静默（只写一条诊断）。 */
    onReadError?: (error: ParamBlockCommitError) => void;
}) {
    const [readyRoot, setReadyRoot] = useState<string | null>(null);
    const latest = useRef(args);
    useLayoutEffect(() => {
        latest.current = args;
    });
    /** 每个根的最近一次权威结果；提交队列用它做 `expectedEditRevision`。 */
    const server = useRef(new Map<string, ParamBlockSnapshot>());
    /** 每个根**唯一**的写队列。 */
    const queues = useRef(new Map<string, Promise<unknown>>());
    const pending = useRef(new Map<string, number>());
    /** 被 deferral 挡下的回读：按根合并，手势结束后合成一次请求。 */
    const refetchPending = useRef(new Set<string>());

    const publishReadError = useCallback((error: unknown) => {
        const failure = classifyParamBlockFailure(error);
        if (latest.current.onReadError) latest.current.onReadError(failure);
        else
            frontendLog(
                "warn",
                "[pianoRoll] param block read failed:",
                failure.detail ?? failure.code,
            );
    }, []);

    const publishWriteError = useCallback((error: unknown) => {
        const failure = classifyParamBlockFailure(error);
        latest.current.onWriteError?.(paramBlockFailureFeedback(failure), failure);
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
    const adoptSnapshot = useCallback((root: string, reply: ParamBlockReply) => {
        const normalized = normalizeParamBlockEditSetWithDiagnostics(reply.edits);
        if (normalized.diagnostics.length > 0) {
            frontendLog(
                "warn",
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
            curveRevision:
                typeof reply.curveRevision === "number" && Number.isFinite(reply.curveRevision)
                    ? reply.curveRevision
                    : (known?.curveRevision ?? 0),
        };
        server.current.set(root, snapshot);
        return snapshot;
    }, []);

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
        (root: string, reply: ParamBlockReply) => {
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
                publishReadError(error);
            }
        },
        [applyReply, publishReadError],
    );

    /** 只刷新镜像（给提交冲突后的重基用）：不受手势 / 提交中守卫影响。 */
    const refreshAuthority = useCallback(
        async (root: string): Promise<ParamBlockSnapshot | null> => {
            try {
                const reply = await timelineApi.getParamBlockView(root);
                return adoptSnapshot(root, reply);
            } catch (error) {
                publishReadError(error);
                return null;
            }
        },
        [adoptSnapshot, publishReadError],
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
                if (!disposed) publishReadError(error);
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
    }, [args.rootTrackId, args.enabled, applyReply, publishReadError, readback]);

    /** 历史位置变了（撤销 / 重做 / 提交落库）才回读编辑集。 */
    /*
     * 图层关闭时回收三张按根累积的 Map。长时间运行 + 频繁切工程会让它们按
     * `rootTrackId` 无限增长；关掉图层意味着「不再关心这些根的结果」。
     * `queues` 里可能有未完成的 Promise —— 只 clear 不 cancel，语义与既有的
     * 「回读失败自愈」一致：结果落地时若根已变，写入会被 rootTrackId 检查挡下。
     */
    useEffect(() => {
        if (args.enabled) return;
        server.current.clear();
        queues.current.clear();
        pending.current.clear();
        refetchPending.current.clear();
    }, [args.enabled]);
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

    /** 一次 IPC。CAS 基准取镜像里的服务端代次。 */
    const executeWrite = useCallback(
        async (
            root: string,
            target: ParamBlockEditSet,
            prepared: ParamBlockWritePreparation,
            patches: readonly ParamBlockCurvePatch[],
        ): Promise<ParamBlockSnapshot> => {
            const current = server.current.get(root);
            const reply = await timelineApi.setParamBlockEdits({
                rootTrackId: root,
                expectedEditRevision: current?.edits.revision ?? target.revision,
                edits: target.blocks,
                checkpoint: prepared.checkpoint ?? true,
                curvePatches: patches,
                expectedCurveRevision: prepared.authority?.expectedCurveRevision,
                expectedGeometryRevision: prepared.authority?.expectedGeometryRevision,
                expectedGeometrySignature: prepared.authority?.expectedGeometrySignature,
            });
            const snapshot: ParamBlockSnapshot = {
                edits: normalizeParamBlockEditSet(reply.edits ?? target),
                framePeriodMs: framePeriodOf(
                    reply.view,
                    current?.framePeriodMs ?? prepared.state.framePeriodMs,
                ),
                curveRevision:
                    typeof reply.curveRevision === "number" && Number.isFinite(reply.curveRevision)
                        ? reply.curveRevision
                        : (current?.curveRevision ?? 0),
            };
            server.current.set(root, snapshot);
            return snapshot;
        },
        [],
    );

    /**
     * 队列槽内的一次完整写入：准备 → 预检 → 一次 IPC →（冲突则重基再试）。
     *
     * 【重基而不是重放】`canRebaseParamBlockCommit` 放开了 `curve` / `geometry` 冲突，
     * 前提正是 `prepare()` 会重新读基准并按同一个意图重算补丁。旧实现只允许
     * `edit_set` 原样重放，于是另外两类冲突一律变成用户可见的失败。
     *
     * 【仍然排除 transport】传输失败时上一次写入**可能已经落地**，重试会把一次
     * 用户操作变成两次写入（历史里出现两步）。
     */
    const runPlan = useCallback(
        async (root: string, plan: ParamBlockWritePlan): Promise<ParamBlockSnapshot> => {
            let lastFailure: ParamBlockCommitError | null = null;
            for (let attempt = 0; attempt <= PARAM_BLOCK_REBASE_ATTEMPTS; attempt += 1) {
                const prepared = await plan.prepare();
                /*
                 * 降级路径：只写曲线（无 CAS）。它没有编辑集可写，但仍然走同一个
                 * 队列槽 —— 否则它会推进 `param_curve_revision`，暗算别人的基准。
                 */
                if (prepared.curveOnly !== undefined) {
                    await uploadFullResCurveSegments({
                        trackId: root,
                        param: prepared.curveOnly.param,
                        segments: prepared.curveOnly.segments.map((segment) => ({
                            startFrame: segment.startFrame,
                            values: segment.values.slice(),
                        })),
                    });
                    const mirror = server.current.get(root);
                    if (mirror === undefined) {
                        throw new ParamBlockCommitError({
                            kind: "transport",
                            code: "not_ready",
                            detail: "parameter block edit set is not loaded",
                        });
                    }
                    return mirror;
                }
                const target = normalizeParamBlockEditSet(prepared.state.editSet);
                // ★ 预检：非法编辑集不排队、不发 IPC（见 assertParamBlockEditSetValid）。
                assertParamBlockEditSetValid(target);
                const patches = prepared.patches ?? [];
                assertCurveCommitAuthority({
                    hasPatches: patches.length > 0,
                    authority: prepared.authority,
                });
                try {
                    return await executeWrite(root, target, prepared, patches);
                } catch (error) {
                    const failure = classifyParamBlockFailure(error);
                    lastFailure = failure;
                    if (!canRebaseParamBlockCommit(failure)) throw failure;
                    if (attempt === PARAM_BLOCK_REBASE_ATTEMPTS) throw failure;
                    // 重基：刷新权威镜像，再让 prepare() 重新读基准。
                    if ((await refreshAuthority(root)) === null) throw failure;
                }
            }
            throw (
                lastFailure ?? new ParamBlockCommitError({ kind: "transport", code: "transport" })
            );
        },
        [executeWrite, refreshAuthority],
    );

    /**
     * **唯一**的写入入口。
     *
     * 顺序：确认已有基线 → 占位 pending → 排进该根的队列 → 槽内准备并写入 →
     * 落地 / 分类上报 → 释放占位。
     */
    const commitPlan = useCallback(
        (plan: ParamBlockWritePlan): Promise<ParamBlockSnapshot | undefined> => {
            const root = plan.rootTrackId ?? latest.current.rootTrackId;
            if (root === null) return Promise.resolve(undefined);

            const start = (): Promise<ParamBlockSnapshot | undefined> => {
                pending.current.set(root, (pending.current.get(root) ?? 0) + 1);
                const job = (queues.current.get(root) ?? Promise.resolve())
                    .catch(() => undefined)
                    .then(() => runPlan(root, plan));
                const complete = trackParamBlockCommit(plan.kind, root, job)
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
                        // 写入失败：分类上报（rejected / conflict / transport），并回权威。
                        publishWriteError(error);
                        if (
                            latest.current.rootTrackId === root &&
                            pending.current.get(root) === 1
                        ) {
                            const current = server.current.get(root);
                            if (current) writeState(root, current);
                        }
                        throw error;
                    })
                    .finally(() =>
                        pending.current.set(
                            root,
                            Math.max(0, (pending.current.get(root) ?? 1) - 1),
                        ),
                    );
                queues.current.set(
                    root,
                    complete.catch(() => undefined),
                );
                return complete;
            };

            /*
             * 【为什么必须已有基线】写入是**整集覆盖**。还没读到过这个根的编辑集时
             * 提交本地空集，会把用户已有的块静默抹掉 —— 这是唯一一个能丢数据的
             * 入口，宁可不提交。就绪判据因此从"有没有 view"换成"读没读过"。
             *
             * 【为什么不再静默返回】旧实现直接 `return`，调用方以为写成功了 ——
             * 用户的编辑就这样消失且没有任何提示。现在先补一次回读；仍失败就抛
             * `transport`（可重试、会被分类反馈）。
             */
            if (server.current.get(root) === undefined) {
                return refreshAuthority(root).then((refreshed) => {
                    if (refreshed === null) {
                        throw new ParamBlockCommitError({
                            kind: "transport",
                            code: "not_ready",
                            detail: "parameter block edit set is not loaded",
                        });
                    }
                    return start();
                });
            }
            return start();
        },
        [publishWriteError, refreshAuthority, runPlan, writeState],
    );

    /**
     * 一次性写入（状态与补丁在调用时就已算好）。
     *
     * 【为什么保留这个签名】分割 / 编组 / 删除 / 过渡 / 幅度这些命令的补丁是
     * 纯本地推导的，没有「读基准」这一步，重基也只是重发同一份意图。把它们都改成
     * `ParamBlockWritePlan` 只会增加噪音；真正的队列仍然只有一条（`commitPlan`）。
     */
    const commit = useCallback(
        (
            state: ParamBlockState,
            checkpoint = true,
            curvePatches: readonly ParamBlockCurvePatch[] = [],
            authority?: ParamCurveCommitAuthority | null,
            kind: ParamBlockCommitKind = "layout",
        ) => {
            /*
             * 有曲线补丁却没有权威 revision，后端必定返回 `Conflict: parameter curve
             * changed`。把这条契约放在队列入口，任何新调用点在开发期就会炸出
             * 明确堆栈，而不是变成用户看到的“参数块操作失败”。
             *
             * 这里是**同步**抛（不是 rejected promise）：契约违反必须在调用点就炸。
             */
            assertCurveCommitAuthority({ hasPatches: curvePatches.length > 0, authority });
            return commitPlan({
                kind,
                prepare: async () => ({ state, checkpoint, patches: curvePatches, authority }),
            });
        },
        [commitPlan],
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

    /**
     * 「写不进去就当没写过」的入口（分割 / 编组 / 删除 / 还原）。
     *
     * 【为什么要回滚】本地状态已经领先（`setParamBlockState` 是同步的），写失败时
     * 若不处理，用户看到的是「块已经变了，但重启后没了」。这里用权威覆盖本地，
     * 把「看起来成功」压到一个 IPC 往返之内 —— 分类上报仍然由写入层负责。
     */
    const persist = useCallback(
        (state: ParamBlockState, checkpoint = true) =>
            commit(state, checkpoint).catch(() => {
                flushReadback();
                return undefined;
            }),
        [commit, flushReadback],
    );

    return {
        ready: readyRoot === args.rootTrackId && args.state.rootTrackId === args.rootTrackId,
        persist,
        commit,
        /** 唯一的写入入口（队列内读基准 + 冲突重基）。 */
        commitPlan,
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
        adopt: (root: string, reply: ParamBlockReply) => {
            applyReply(root, reply);
        },
        flushReadback,
    };
}
