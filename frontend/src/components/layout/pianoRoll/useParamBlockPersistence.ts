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
import { composeParamBlockView } from "./paramBlockComposition";
import type { ParamBlockSourceClip } from "./paramBlockProjection";
import {
    EMPTY_PARAM_BLOCK_EDIT_SET,
    normalizeParamBlockEditSet,
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

/**
 * 视图刷新键：覆盖所有会改变投影结果的 Clip/Take 几何与分析输入。
 *
 * 不包含 editSet.revision：编辑 reply 已经更新状态；把它放进键会让每次
 * 编辑都多一轮 get_param_block_view。
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

export function useParamBlockPersistence(args: {
    rootTrackId: string | null;
    clips: readonly ClipInfo[];
    enabled: boolean;
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
    const server = useRef(new Map<string, { view: ParamBlockView; edits: ParamBlockEditSet }>());
    const queues = useRef(new Map<string, Promise<unknown>>());
    const pending = useRef(new Map<string, number>());

    const publishError = useCallback((error: unknown) => {
        latest.current.onError?.(backendDetail(error) ?? String(error));
    }, []);

    const applyReply = useCallback(
        (root: string, reply: { view: ParamBlockView; edits: ParamBlockEditSet }) => {
            const normalized = {
                view: reply.view,
                edits: normalizeParamBlockEditSet(reply.edits),
            };
            server.current.set(root, normalized);
            if (
                latest.current.rootTrackId === root &&
                !(pending.current.get(root) ?? 0) &&
                !latest.current.deferReadback?.()
            ) {
                latest.current.setState({
                    ...EMPTY_PARAM_BLOCK_STATE,
                    rootTrackId: root,
                    clips: latest.current.clips.map(toSourceClip),
                    editSet: normalized.edits,
                    view: normalized.view,
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
                    edits: normalizeParamBlockEditSet(reply.edits),
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
            })
            .catch((error) => {
                if (!disposed) publishError(error);
            });
        const subscription = listen<{ rootTrackId: string; view: ParamBlockView }>(
            "param_block_view_updated",
            ({ payload }) => {
                if (disposed || payload.rootTrackId !== root) return;
                const current = server.current.get(root);
                if (current && current.view.editRevision > payload.view.editRevision) return;
                const next = {
                    view: payload.view,
                    edits: normalizeParamBlockEditSet(current?.edits),
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
    }, [args.rootTrackId, args.enabled, publishError]);

    /** 本地即时重投影：Clip 拖动 / 分析变化时先让块跟随，权威 reply 随后覆盖。 */
    useEffect(() => {
        const root = args.rootTrackId;
        if (
            !root ||
            !args.enabled ||
            (pending.current.get(root) ?? 0) > 0 ||
            latest.current.deferReadback?.()
        )
            return;
        const sourceClips = args.clips.map(toSourceClip);
        const current = latest.current.state;
        const edits = normalizeParamBlockEditSet(current.editSet);
        const framePeriodMs = current.view?.framePeriodMs ?? 5;
        const nextView = composeParamBlockView(root, sourceClips, edits, framePeriodMs);
        if (
            current.view !== undefined &&
            current.view.layoutKey === nextView.layoutKey &&
            current.view.editRevision === nextView.editRevision
        ) {
            return;
        }
        latest.current.setState((state) => ({
            ...state,
            rootTrackId: root,
            clips: sourceClips,
            editSet: edits,
            view: nextView,
        }));
    }, [args.clips, args.rootTrackId, args.enabled]);

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
                    edits: normalizeParamBlockEditSet(reply.edits),
                });
            })
            .catch((error) => {
                if (!disposed) publishError(error);
            });
        return () => {
            disposed = true;
        };
    }, [args.rootTrackId, args.enabled, args.historyKey, applyReply, publishError]);

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
            const target = normalizeParamBlockEditSet(state.editSet);
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
                const next = {
                    view: reply.view,
                    edits: normalizeParamBlockEditSet(reply.edits ?? target),
                };
                server.current.set(root, next);
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
                    publishError(error);
                    if (latest.current.rootTrackId === root && pending.current.get(root) === 1) {
                        const current = server.current.get(root);
                        if (current) {
                            latest.current.setState((state) => ({
                                ...state,
                                rootTrackId: root,
                                clips: latest.current.clips.map(toSourceClip),
                                editSet: normalizeParamBlockEditSet(current.edits),
                                view: current.view,
                            }));
                        }
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
        [publishError],
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
        persist,
        commit,
        receive: (root: string, view: ParamBlockView, edits?: ParamBlockEditSet) =>
            applyReply(root, {
                view,
                edits: normalizeParamBlockEditSet(
                    edits ?? server.current.get(root)?.edits ?? EMPTY_PARAM_BLOCK_EDIT_SET,
                ),
            }),
        flushReadback,
    };
}
