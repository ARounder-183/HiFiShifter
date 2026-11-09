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
import type { ParamBlockSourceClip } from "./paramBlockProjection";
import {
    EMPTY_PARAM_BLOCK_EDIT_SET,
    type ParamBlockEditSet,
    type ParamBlockView,
} from "./paramBlockTypes";
import { EMPTY_PARAM_BLOCK_STATE, type ParamBlockState } from "./paramBlocks";

function toSourceClip(clip: ClipInfo): ParamBlockSourceClip {
    return {
        id: clip.id,
        trackId: clip.trackId,
        startSec: clip.startSec,
        lengthSec: clip.lengthSec,
        sourceStartSec: clip.sourceStartSec,
        sourceEndSec: clip.sourceEndSec,
        playbackRate: clip.playbackRate,
        reversed: clip.reversed,
        loopEnabled: clip.loopEnabled,
        contentDurationSec: clip.durationSec ?? null,
        activeTakeId: clip.activeTakeId,
        takes: clip.takes?.map((take) => ({
            id: take.id,
            sourceStartSec: take.sourceStartSec,
            sourceEndSec: take.sourceEndSec,
            playbackRate: take.playbackRate,
            reversed: take.reversed,
            loopEnabled: take.loopEnabled,
            durationSec: take.durationSec,
            durationFrames: take.durationFrames,
            sourceSampleRate: take.sourceSampleRate,
            syllableSpansSec: take.syllableSpansSec,
            midiNoteData: take.midiNoteData,
        })),
        syllableSpansSec: clip.syllableSpansSec,
        midiNoteData: clip.midiNoteData,
    };
}

export function useParamBlockPersistence(args: {
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
    const server = useRef(new Map<string, { view: ParamBlockView; edits: ParamBlockEditSet }>());
    const queues = useRef(new Map<string, Promise<unknown>>());
    const pending = useRef(new Map<string, number>());

    const applyReply = useCallback(
        (root: string, reply: { view: ParamBlockView; edits: ParamBlockEditSet }) => {
            server.current.set(root, reply);
            if (
                latest.current.rootTrackId === root &&
                !(pending.current.get(root) ?? 0) &&
                !latest.current.deferReadback?.()
            ) {
                latest.current.setState({
                    ...EMPTY_PARAM_BLOCK_STATE,
                    rootTrackId: root,
                    clips: latest.current.clips.map(toSourceClip),
                    editSet: reply.edits,
                    view: reply.view,
                });
            }
        },
        [],
    );

    useEffect(() => {
        const root = args.rootTrackId;
        if (!root || !args.enabled) return;
        let disposed = false;
        void timelineApi
            .getParamBlockView(root)
            .then((reply) => {
                if (disposed) return;
                const normalized = {
                    view: reply.view,
                    edits: reply.edits ?? EMPTY_PARAM_BLOCK_EDIT_SET,
                };
                server.current.set(root, normalized);
                latest.current.setState({
                    ...EMPTY_PARAM_BLOCK_STATE,
                    rootTrackId: root,
                    clips: latest.current.clips.map(toSourceClip),
                    editSet: normalized.edits,
                    view: normalized.view,
                });
                setReadyRoot(root);
                setError(null);
            })
            .catch((error) => {
                if (!disposed) setError(backendDetail(error) ?? String(error));
            });
        const subscription = listen<{ rootTrackId: string; view: ParamBlockView }>(
            "param_block_view_updated",
            ({ payload }) => {
                if (disposed || payload.rootTrackId !== root) return;
                const current = server.current.get(root);
                if (current && current.view.editRevision > payload.view.editRevision) return;
                const next = {
                    view: payload.view,
                    edits: current?.edits ?? EMPTY_PARAM_BLOCK_EDIT_SET,
                };
                server.current.set(root, next);
                if (!(pending.current.get(root) ?? 0) && !latest.current.deferReadback?.()) {
                    latest.current.setState((state) => ({
                        ...state,
                        rootTrackId: root,
                        clips: latest.current.clips.map(toSourceClip),
                        editSet: next.edits,
                        view: next.view,
                    }));
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
        if (!root || !args.enabled) return;
        latest.current.setState((state) => ({
            ...state,
            rootTrackId: root,
            clips: latest.current.clips.map(toSourceClip),
        }));
    }, [args.clips, args.rootTrackId, args.enabled]);

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
            .getParamBlockView(root)
            .then((reply) => {
                if (disposed || (pending.current.get(root) ?? 0) > 0) return;
                applyReply(root, {
                    view: reply.view,
                    edits: reply.edits ?? EMPTY_PARAM_BLOCK_EDIT_SET,
                });
            })
            .catch(() => undefined);
        return () => {
            disposed = true;
        };
    }, [args.rootTrackId, args.enabled, args.historyKey, applyReply]);

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
            if (!root || state.view === undefined) return Promise.resolve(undefined);
            const target = state.editSet;
            pending.current.set(root, (pending.current.get(root) ?? 0) + 1);
            const execute = async () => {
                const current = server.current.get(root);
                const expected = current?.edits.revision ?? target.revision;
                const reply = await timelineApi.setParamBlockEdits({
                    rootTrackId: root,
                    expectedEditRevision: expected,
                    edits: target.blocks,
                    checkpoint,
                    curvePatches,
                    expectedCurveRevision,
                    expectedGeometryRevision,
                    expectedGeometrySignature,
                });
                const next = { view: reply.view, edits: reply.edits ?? target };
                server.current.set(root, next);
                setError(null);
                return next;
            };
            const job = (queues.current.get(root) ?? Promise.resolve())
                .catch(() => undefined)
                .then(execute);
            const complete = job
                .then((next) => {
                    if (next) latest.current.onCommitted?.();
                    if (
                        next &&
                        latest.current.rootTrackId === root &&
                        !latest.current.deferReadback?.() &&
                        pending.current.get(root) === 1
                    ) {
                        latest.current.setState((state) => ({
                            ...state,
                            editSet: next.edits,
                            view: next.view,
                        }));
                    }
                    return next;
                })
                .catch((error) => {
                    setError(backendDetail(error) ?? String(error));
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
        const current = root ? server.current.get(root) : undefined;
        if (root && current) applyReply(root, current);
    }, [applyReply]);

    return {
        ready: readyRoot === args.rootTrackId && args.state.rootTrackId === args.rootTrackId,
        error,
        persist,
        commit,
        receive: (root: string, view: ParamBlockView, edits?: ParamBlockEditSet) =>
            applyReply(root, {
                view,
                edits: edits ?? server.current.get(root)?.edits ?? EMPTY_PARAM_BLOCK_EDIT_SET,
            }),
        flushReadback,
    };
}
