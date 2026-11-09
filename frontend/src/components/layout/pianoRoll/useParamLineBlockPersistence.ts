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
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlockState } from "./paramBlocks";
import { migrateParamLineBlocks } from "./paramLineBlockMigration";
import { rebaseParamLineBlocks } from "./paramLineBlockPatch";
import type { ParamLineBlockSet } from "./paramLineBlocks";

export function useParamLineBlockPersistence(args: {
    rootTrackId: string | null;
    clips: readonly ClipInfo[];
    enabled: boolean;
    historyKey: string;
    state: ParamBlockState;
    setState: Dispatch<SetStateAction<ParamBlockState>>;
    deferReadback?: () => boolean;
    onCommitted?: () => void;
}) {
    const [readyRoot, setReadyRoot] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const latest = useRef(args);
    useLayoutEffect(() => {
        latest.current = args;
    });
    const server = useRef(new Map<string, ParamLineBlockSet>());
    const optimistic = useRef(new Map<string, ParamLineBlockSet>());
    const queues = useRef(new Map<string, Promise<unknown>>());
    const pending = useRef(new Map<string, number>());
    const receive = useCallback((root: string, layout: ParamLineBlockSet) => {
        const previous = server.current.get(root);
        if (previous && previous.revision > layout.revision) return;
        server.current.set(root, layout);
        if (
            latest.current.rootTrackId === root &&
            !(pending.current.get(root) ?? 0) &&
            !latest.current.deferReadback?.()
        ) {
            optimistic.current.set(root, layout);
            latest.current.setState({
                ...EMPTY_PARAM_BLOCK_STATE,
                lineLayout: layout,
                lineRootTrackId: root,
            });
        }
    }, []);

    useEffect(() => {
        const root = args.rootTrackId;
        if (!root || !args.enabled) return;
        let disposed = false;
        const initialClips = args.clips;
        void (async () => {
            let reply = await timelineApi.getParamLineBlocks(root);
            if (!reply.initialized) {
                const legacy = await timelineApi.getParamBlockMerges(root);
                const layout = migrateParamLineBlocks({
                    root,
                    clips: initialClips,
                    framePeriodMs: reply.layout.framePeriodMs,
                    records: legacy.records,
                    intervalEdits: legacy.intervalEdits,
                });
                try {
                    reply = await timelineApi.setParamLineBlocks({
                        rootTrackId: root,
                        expectedRevision: reply.layout.revision,
                        blocks: layout.blocks,
                        gameBaseline: layout.gameBaseline.segments,
                        checkpoint: false,
                    });
                } catch (error) {
                    if (!backendDetail(error)?.includes("Conflict:")) throw error;
                    reply = await timelineApi.getParamLineBlocks(root);
                }
            }
            const newer = server.current.get(root);
            if (newer && newer.revision > reply.layout.revision)
                reply = { ...reply, layout: newer };
            server.current.set(root, reply.layout);
            if (!disposed) {
                optimistic.current.set(root, reply.layout);
                latest.current.setState({
                    ...EMPTY_PARAM_BLOCK_STATE,
                    lineLayout: reply.layout,
                    lineRootTrackId: root,
                });
                setReadyRoot(root);
                setError(null);
            }
        })().catch((error) => {
            if (!disposed) setError(backendDetail(error) ?? String(error));
        });
        const subscription = listen<{ rootTrackId: string; layout: ParamLineBlockSet }>(
            "param_line_blocks_updated",
            ({ payload }) => {
                if (disposed || payload.rootTrackId !== root) return;
                const previous = server.current.get(root);
                if (previous && previous.revision > payload.layout.revision) return;
                server.current.set(root, payload.layout);
                if (!(pending.current.get(root) ?? 0) && !latest.current.deferReadback?.()) {
                    optimistic.current.set(root, payload.layout);
                    latest.current.setState({
                        ...EMPTY_PARAM_BLOCK_STATE,
                        lineLayout: payload.layout,
                        lineRootTrackId: root,
                    });
                }
            },
        ).catch(() => () => {});
        return () => {
            disposed = true;
            void subscription.then((off) => off());
        };
    }, [args.rootTrackId, args.enabled]);

    useEffect(() => {
        const root = args.rootTrackId;
        if (
            !root ||
            !args.enabled ||
            (pending.current.get(root) ?? 0) > 0 ||
            latest.current.deferReadback?.()
        )
            return;
        let disposed = false;
        void timelineApi
            .getParamLineBlocks(root)
            .then((reply) => {
                if (disposed || !reply.initialized || (pending.current.get(root) ?? 0) > 0) return;
                receive(root, reply.layout);
            })
            .catch(() => undefined);
        return () => {
            disposed = true;
        };
    }, [args.rootTrackId, args.enabled, args.historyKey, receive]);

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
            const root = state.lineRootTrackId ?? latest.current.rootTrackId;
            const target = state.lineLayout;
            if (!root || !target) return Promise.resolve();
            const basis = state.lineEditBase ?? server.current.get(root) ?? target;
            optimistic.current.set(root, target);
            pending.current.set(root, (pending.current.get(root) ?? 0) + 1);
            const execute = async () => {
                for (let attempt = 0; attempt < 3; attempt++) {
                    const current =
                        server.current.get(root) ??
                        (await timelineApi.getParamLineBlocks(root)).layout;
                    if (curvePatches.length) {
                        const currentById = new Map(
                            current.blocks.map((block) => [block.id, block]),
                        );
                        for (const block of basis.blocks) {
                            if (
                                !curvePatches.some(
                                    (patch) =>
                                        patch.startFrame < block.endFrameExclusive &&
                                        patch.startFrame + patch.values.length > block.startFrame,
                                )
                            )
                                continue;
                            const actual = currentById.get(block.id);
                            if (
                                !actual ||
                                actual.startFrame !== block.startFrame ||
                                actual.endFrameExclusive !== block.endFrameExclusive
                            )
                                throw new Error(
                                    "Conflict: parameter block geometry changed during gesture",
                                );
                        }
                    }
                    const blocks = rebaseParamLineBlocks(
                        current.blocks,
                        basis.blocks,
                        target.blocks,
                    );
                    const baseline = rebaseParamLineBlocks(
                        current.gameBaseline.segments,
                        basis.gameBaseline.segments,
                        target.gameBaseline.segments,
                    );
                    try {
                        const reply = await timelineApi.setParamLineBlocks({
                            rootTrackId: root,
                            expectedRevision: current.revision,
                            blocks,
                            gameBaseline: baseline,
                            checkpoint,
                            curvePatches,
                            expectedCurveRevision,
                            expectedGeometryRevision,
                            expectedGeometrySignature,
                        });
                        const newer = server.current.get(root);
                        const acknowledged =
                            newer && newer.revision > reply.layout.revision ? newer : reply.layout;
                        server.current.set(root, acknowledged);
                        setError(null);
                        return acknowledged;
                    } catch (error) {
                        if (
                            attempt === 2 ||
                            !backendDetail(error)?.includes("Conflict: parameter blocks")
                        )
                            throw error;
                        server.current.set(
                            root,
                            (await timelineApi.getParamLineBlocks(root)).layout,
                        );
                    }
                }
            };
            const job = (queues.current.get(root) ?? Promise.resolve())
                .catch(() => undefined)
                .then(execute);
            const complete = job
                .then((layout) => {
                    if (layout) latest.current.onCommitted?.();
                    if (
                        layout &&
                        latest.current.rootTrackId === root &&
                        !latest.current.deferReadback?.() &&
                        pending.current.get(root) === 1
                    ) {
                        optimistic.current.set(root, layout);
                        latest.current.setState({
                            ...EMPTY_PARAM_BLOCK_STATE,
                            lineLayout: layout,
                            lineRootTrackId: root,
                        });
                    }
                    return layout;
                })
                .catch((error) => {
                    setError(backendDetail(error) ?? String(error));
                    const current = server.current.get(root);
                    if (
                        current &&
                        latest.current.rootTrackId === root &&
                        !latest.current.deferReadback?.() &&
                        pending.current.get(root) === 1
                    )
                        latest.current.setState({
                            ...EMPTY_PARAM_BLOCK_STATE,
                            lineRootTrackId: root,
                            lineLayout: current,
                        });
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
        [],
    );
    const persist = useCallback(
        (state: ParamBlockState, checkpoint = true) =>
            commit(state, checkpoint).catch(() => undefined),
        [commit],
    );
    const flushReadback = useCallback(() => {
        const root = latest.current.rootTrackId;
        const layout = root ? server.current.get(root) : undefined;
        if (root && layout) receive(root, layout);
    }, [receive]);

    return {
        ready: readyRoot === args.rootTrackId && args.state.lineRootTrackId === args.rootTrackId,
        error,
        persist,
        commit,
        receive,
        flushReadback,
    };
}
