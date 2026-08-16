import { useRef } from "react";
import { batch } from "react-redux";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import {
    bumpParamsEpoch,
    checkpointHistory,
    moveClipStart,
    setClipFades,
    setClipGain,
    setClipLength,
    setClipPlaybackRate,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipSourceRange,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import { applyAutoCrossfade } from "./autoCrossfade";
import { clamp } from "../math";
import { advanceFineAxisDrag, type FineAxisDragState } from "../fineAxisDrag";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import { paramsApi } from "../../../../services/api";
import { webApi } from "../../../../services/webviewApi";
import {
    buildStretchGroupState,
    computeStretchGroupUpdate,
    scaleClipFadesForStretch,
    type StretchGroupState,
} from "./stretchGroup";
import { applyBulkFadeValue, applyBulkGainDeltaDb, getBulkEditableClipIds } from "./bulkClipEdit";
import { expandClipIdsWithGroups } from "./useGroupExpansion";
import { buildBulkClipStateUpdates } from "./bulkClipRemotePayloads";

const CLIP_GAIN_DRAG_DB_PER_PX = 0.25;

export function resolveStretchParamTypes(
    pitchEditUserModified: boolean | null | undefined,
): Array<"pitch" | "tension"> {
    // 未手动编辑的 pitch 曲线由后端根据 clip 几何自动重建，
    // 前端若再次映射会造成“二次拉伸”。
    if (pitchEditUserModified === false) {
        return ["tension"];
    }
    return ["pitch", "tension"];
}

/**
 * 拉伸后对参数线进行时域映射（拉伸或压缩）。
 * 将旧范围 [oldStartSec, oldStartSec+oldLengthSec] 内的参数值，
 * 线性重映射到新范围 [newStartSec, newStartSec+newLengthSec]，
 * 并将不再被音频块覆盖的旧帧恢复为原始值。
 */
async function stretchLinkedParams(
    trackId: string,
    oldStartSec: number,
    oldLengthSec: number,
    newStartSec: number,
    newLengthSec: number,
): Promise<void> {
    if (
        Math.abs(oldLengthSec - newLengthSec) < 1e-6 &&
        Math.abs(oldStartSec - newStartSec) < 1e-6
    ) {
        return;
    }

    // 获取帧周期（通过最小量探针请求）。
    // 同时读取 pitch_edit_user_modified 以决定是否应手动映射 pitch。
    const probe = await paramsApi.getParamFrames(trackId, "pitch", 0, 1, 1);
    if (!probe?.ok) return;
    const fp = Math.max(1, Number(probe.frame_period_ms) || 5);
    const stretchParams = resolveStretchParamTypes(probe.pitch_edit_user_modified);

    const oldStartFrame = Math.round((oldStartSec * 1000) / fp);
    const oldEndFrame = Math.round(((oldStartSec + oldLengthSec) * 1000) / fp);
    const oldFrameCount = Math.max(1, oldEndFrame - oldStartFrame);

    const newStartFrame = Math.round((newStartSec * 1000) / fp);
    const newEndFrame = Math.round(((newStartSec + newLengthSec) * 1000) / fp);
    const newFrameCount = Math.max(1, newEndFrame - newStartFrame);

    for (const paramType of stretchParams) {
        const res = await paramsApi.getParamFrames(
            trackId,
            paramType,
            oldStartFrame,
            oldFrameCount,
            1,
        );
        if (!res?.ok) continue;
        const oldValues = (res.edit ?? []).map((v) => Number(v) || 0);
        if (oldValues.length === 0) continue;

        // 线性插值时域映射：用旧帧值填充新帧
        const newValues = new Array<number>(newFrameCount);
        const oldMaxIdx = oldValues.length - 1;
        const newMaxIdx = newFrameCount > 1 ? newFrameCount - 1 : 1;
        const ratio = oldMaxIdx / newMaxIdx;

        for (let i = 0; i < newFrameCount; i++) {
            const oldIdxF = i * ratio;
            const lo = oldIdxF | 0;
            const hi = lo < oldMaxIdx ? lo + 1 : oldMaxIdx;
            const frac = oldIdxF - lo;
            const loVal = oldValues[lo] ?? 0;
            const hiVal = oldValues[hi] ?? 0;
            if (paramType === "pitch") {
                // pitch=0 表示无效（无声）帧，保留 0
                if (loVal === 0 && hiVal === 0) {
                    newValues[i] = 0;
                } else if (loVal === 0) {
                    newValues[i] = 0;
                } else if (hiVal === 0) {
                    newValues[i] = frac < 0.5 ? loVal : 0;
                } else {
                    newValues[i] = loVal + (hiVal - loVal) * frac;
                }
            } else {
                newValues[i] = loVal + (hiVal - loVal) * frac;
            }
        }

        // 将重映射后的值写入新范围
        await paramsApi.setParamFrames(trackId, paramType, newStartFrame, newValues, false);

        // 恢复旧范围中不再被新音频块覆盖的帧（还原到原始值）
        const newRangeMax = newStartFrame + newFrameCount - 1;
        const oldRangeMax = oldStartFrame + oldFrameCount - 1;

        if (oldStartFrame < newStartFrame) {
            const clearLen = newStartFrame - oldStartFrame;
            void paramsApi.restoreParamFrames(trackId, paramType, oldStartFrame, clearLen, false);
        }
        if (oldRangeMax > newRangeMax) {
            const clearFrom = newRangeMax + 1;
            const clearLen = oldRangeMax - newRangeMax;
            void paramsApi.restoreParamFrames(trackId, paramType, clearFrom, clearLen, false);
        }
    }
}

type StretchRangeMapping = {
    oldStartSec: number;
    oldLengthSec: number;
    newStartSec: number;
    newLengthSec: number;
};

function buildMappedParamValues(
    oldValues: number[],
    paramType: "pitch" | "tension",
    newFrameCount: number,
): number[] {
    const newValues = new Array<number>(newFrameCount);
    const oldMaxIdx = oldValues.length - 1;
    const newMaxIdx = newFrameCount > 1 ? newFrameCount - 1 : 1;
    const ratio = oldMaxIdx / newMaxIdx;

    for (let i = 0; i < newFrameCount; i++) {
        const oldIdxF = i * ratio;
        const lo = oldIdxF | 0;
        const hi = lo < oldMaxIdx ? lo + 1 : oldMaxIdx;
        const frac = oldIdxF - lo;
        const loVal = oldValues[lo] ?? 0;
        const hiVal = oldValues[hi] ?? 0;
        if (paramType === "pitch") {
            if (loVal === 0 && hiVal === 0) {
                newValues[i] = 0;
            } else if (loVal === 0) {
                newValues[i] = 0;
            } else if (hiVal === 0) {
                newValues[i] = frac < 0.5 ? loVal : 0;
            } else {
                newValues[i] = loVal + (hiVal - loVal) * frac;
            }
        } else {
            newValues[i] = loVal + (hiVal - loVal) * frac;
        }
    }

    return newValues;
}

function subtractIntervals(
    range: { start: number; end: number },
    excluded: Array<{ start: number; end: number }>,
): Array<{ start: number; end: number }> {
    const sorted = excluded
        .filter((item) => item.end >= range.start && item.start <= range.end)
        .sort((a, b) => a.start - b.start);
    const result: Array<{ start: number; end: number }> = [];
    let cursor = range.start;

    for (const item of sorted) {
        if (cursor < item.start) {
            result.push({
                start: cursor,
                end: Math.min(item.start - 1, range.end),
            });
        }
        cursor = Math.max(cursor, item.end + 1);
        if (cursor > range.end) break;
    }

    if (cursor <= range.end) {
        result.push({ start: cursor, end: range.end });
    }

    return result;
}

/**
 * Stretch parameter lines for several clips on the same track as one batch.
 *
 * Per-clip independent writes were racing with per-clip restores: the old
 * range of one clip can overlap the new range of a neighbouring clip, so a
 * restore could erase freshly written mapped values. This version writes all
 * new ranges first, then only restores old-range parts that are not covered
 * by any new range on that track.
 */
async function stretchTrackLinkedParams(
    trackId: string,
    mappings: StretchRangeMapping[],
): Promise<void> {
    if (mappings.length === 0) return;

    const probe = await paramsApi.getParamFrames(trackId, "pitch", 0, 1, 1);
    if (!probe?.ok) return;
    const fp = Math.max(1, Number(probe.frame_period_ms) || 5);
    const stretchParams = resolveStretchParamTypes(probe.pitch_edit_user_modified);

    const frameMappings = mappings.map((mapping) => {
        const oldStartFrame = Math.round((mapping.oldStartSec * 1000) / fp);
        const oldEndFrame = Math.round(((mapping.oldStartSec + mapping.oldLengthSec) * 1000) / fp);
        const oldFrameCount = Math.max(1, oldEndFrame - oldStartFrame);

        const newStartFrame = Math.round((mapping.newStartSec * 1000) / fp);
        const newEndFrame = Math.round(((mapping.newStartSec + mapping.newLengthSec) * 1000) / fp);
        const newFrameCount = Math.max(1, newEndFrame - newStartFrame);

        return { oldStartFrame, oldFrameCount, newStartFrame, newFrameCount };
    });

    const newRanges = frameMappings.map((mapping) => ({
        start: mapping.newStartFrame,
        end: mapping.newStartFrame + mapping.newFrameCount - 1,
    }));

    for (const paramType of stretchParams) {
        const fetched = await Promise.all(
            frameMappings.map(async (mapping) => {
                const res = await paramsApi.getParamFrames(
                    trackId,
                    paramType,
                    mapping.oldStartFrame,
                    mapping.oldFrameCount,
                    1,
                );
                if (!res?.ok) return null;
                return {
                    mapping,
                    values: (res.edit ?? []).map((value) => Number(value) || 0),
                };
            }),
        );

        // Write every new range first so no restore can clobber a fresh write.
        for (const item of fetched) {
            if (!item || item.values.length === 0) continue;
            const newValues = buildMappedParamValues(
                item.values,
                paramType,
                item.mapping.newFrameCount,
            );
            await paramsApi.setParamFrames(
                trackId,
                paramType,
                item.mapping.newStartFrame,
                newValues,
                false,
            );
        }

        // Restore only old-range parts not covered by any new range.
        for (const mapping of frameMappings) {
            const oldEndFrame = mapping.oldStartFrame + mapping.oldFrameCount - 1;
            const restoreSegments = subtractIntervals(
                { start: mapping.oldStartFrame, end: oldEndFrame },
                newRanges,
            );
            for (const segment of restoreSegments) {
                await paramsApi.restoreParamFrames(
                    trackId,
                    paramType,
                    segment.start,
                    segment.end - segment.start + 1,
                    false,
                );
            }
        }
    }
}

export type EditDragType =
    | "trim_left"
    | "trim_right"
    | "stretch_left"
    | "stretch_right"
    | "fade_in"
    | "fade_out"
    | "gain";

export type EditDragState = {
    type: EditDragType;
    pointerId: number;
    clipId: string;
    basestartSec: number;
    baselengthSec: number;
    basePlaybackRate: number;
    baseSourceStartSec: number;
    baseSourceEndSec: number;
    basefadeInSec: number;
    basefadeOutSec: number;
    baseGain: number;
    sourceBeats: number | null;
    rightEdgeBeat: number;
    baseReversed: boolean;
    baseDurationFrames: number | null;
    baseSourceSampleRate: number | null;
    baseDurationSec: number | null;
    stretchGroup: StretchGroupState | null;
    selectedClipIds: string[];
    baseGainById: Record<string, number>;
    /** Per-clip base state for multi-clip trim operations */
    baseByClipId: Record<
        string,
        {
            startSec: number;
            lengthSec: number;
            playbackRate: number;
            sourceStartSec: number;
            sourceEndSec: number;
            reversed: boolean;
            durationFrames: number | null;
            sourceSampleRate: number | null;
            durationSec: number | null;
        }
    >;
};

export function useEditDrag(deps: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    dispatch: AppDispatch;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    snapTimeline: (
        sec: number,
        object: "mediaItem",
        opts?: {
            originSec?: number;
            anchorTrackId?: string | null;
            excludeClipIds?: ReadonlySet<string>;
        },
    ) => number;
    beatFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
    /** modifier.clipNoSnap 绑定 */
    noSnapKb: Keybinding;
    /** 吸附全局开关 */
    snapEnabled: boolean;
    /** 忽略编组 */
    ignoreGrouping: boolean;
    /** modifier.paramFineAdjust 绑定 */
    paramFineAdjustKb: Keybinding;
}) {
    const {
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        snapTimeline,
        beatFromClientX,
        noSnapKb,
        snapEnabled,
        ignoreGrouping,
        paramFineAdjustKb,
    } = deps;

    const editDragRef = useRef<EditDragState | null>(null);
    // 用于节流向后端发送 clip 状态更新，避免拖动时频繁覆盖与后端同步引起闪烁
    const lastRemoteSentRef = useRef<Record<string, number>>({});

    function startEditDrag(e: React.PointerEvent, clipId: string, type: EditDragType) {
        if (e.button !== 0) return;
        const clip = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!clip) return;
        const scroller = scrollRef.current;
        if (!scroller) return;
        const rightEdgeBeat = clip.startSec + clip.lengthSec;

        // Resolve which clips to operate on.
        // Trim / stretch / slip expand to all selected + their group members.
        // Gain / fades only apply to multi-selected clips (no group expansion).
        const initialIds = getBulkEditableClipIds({
            activeClipId: clipId,
            multiSelectedClipIds,
            multiSelectedSet,
        });
        const supportsGroupExpansion =
            !ignoreGrouping && type !== "fade_in" && type !== "fade_out" && type !== "gain";
        const selectedClipIds = supportsGroupExpansion
            ? expandClipIdsWithGroups(
                  initialIds,
                  sessionRef.current.clips,
                  false,
                  sessionRef.current.disabledGroupIds,
              )
            : initialIds;
        const baseGainById = Object.fromEntries(
            selectedClipIds.map((id) => {
                const selectedClip = sessionRef.current.clips.find((entry) => entry.id === id);
                return [id, Number(selectedClip?.gain ?? 1) || 1];
            }),
        ) as Record<string, number>;
        const stretchGroup =
            type === "stretch_left" || type === "stretch_right"
                ? buildStretchGroupState({
                      clips: sessionRef.current.clips,
                      selectedClipIds,
                      anchorClipId: clipId,
                      edge: type,
                  })
                : null;

        dispatch(checkpointHistory());
        dispatch(beginInteraction());

        // Gain drag sends throttled backend preview updates while dragging. Open the
        // backend undo group up front so the single checkpoint is the pre-drag value;
        // otherwise the final bulk write would checkpoint after the previews already
        // changed the backend, and undo would bounce back to the post-drag value.
        const gainUndoGroupPromise = type === "gain" ? webApi.beginUndoGroup() : null;

        editDragRef.current = {
            type,
            pointerId: e.pointerId,
            clipId,
            basestartSec: clip.startSec,
            baselengthSec: clip.lengthSec,
            basePlaybackRate: Number(clip.playbackRate ?? 1) || 1,
            baseSourceStartSec: clip.sourceStartSec,
            baseSourceEndSec: clip.sourceEndSec,
            basefadeInSec: clip.fadeInSec,
            basefadeOutSec: clip.fadeOutSec,
            baseGain: clip.gain,
            sourceBeats: null,
            rightEdgeBeat,
            baseReversed: !!clip.reversed,
            baseDurationFrames: clip.durationFrames ?? null,
            baseSourceSampleRate: clip.sourceSampleRate ?? null,
            baseDurationSec: clip.durationSec ?? null,
            stretchGroup,
            selectedClipIds,
            baseGainById,
            baseByClipId: Object.fromEntries(
                selectedClipIds.map((id) => {
                    const c =
                        id === clipId ? clip : sessionRef.current.clips.find((x) => x.id === id);
                    return [
                        id,
                        {
                            startSec: c?.startSec ?? 0,
                            lengthSec: c?.lengthSec ?? 0,
                            playbackRate: Number(c?.playbackRate ?? 1) || 1,
                            sourceStartSec: c?.sourceStartSec ?? 0,
                            sourceEndSec: c?.sourceEndSec ?? 0,
                            reversed: !!c?.reversed,
                            durationFrames: c?.durationFrames ?? null,
                            sourceSampleRate: c?.sourceSampleRate ?? null,
                            durationSec: c?.durationSec ?? null,
                        },
                    ];
                }),
            ),
        };

        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);

        let ticking = false;
        let latestEvent: PointerEvent | null = null;
        let accumulatedGainDeltaDb = 0;
        let gainFineAxisState: FineAxisDragState | null = null;
        let gainDragStartClientY: number | null = null;
        let remotePreviewChain: Promise<unknown> = Promise.resolve();

        const finishGainUndoGroup = async () => {
            if (!gainUndoGroupPromise) return;
            try {
                await remotePreviewChain;
                await gainUndoGroupPromise;
            } finally {
                await webApi.endUndoGroup();
            }
        };

        function onMove(ev: PointerEvent) {
            latestEvent = ev;
            if (ticking) return;
            ticking = true;

            requestAnimationFrame(() => {
                ticking = false;
                if (!latestEvent) return;
                const currentEv = latestEvent;

                const drag = editDragRef.current;
                const el = scrollRef.current;
                if (!drag || drag.pointerId !== e.pointerId || !el) return;
                const b = el.getBoundingClientRect();
                let beat = beatFromClientX(currentEv.clientX, b, el.scrollLeft);
                const shouldSnap =
                    drag.type === "trim_left" ||
                    drag.type === "trim_right" ||
                    drag.type === "stretch_left" ||
                    drag.type === "stretch_right";
                const noSnapActive = isModifierActive(noSnapKb, currentEv);
                const effectiveSnap = snapEnabled && !noSnapActive;
                if (shouldSnap && effectiveSnap) {
                    const leftEdge = drag.type === "trim_right" || drag.type === "stretch_right";
                    beat = snapTimeline(beat, "mediaItem", {
                        originSec: leftEdge ? drag.rightEdgeBeat : drag.basestartSec,
                        anchorTrackId: sessionRef.current.clips.find((c) => c.id === drag.clipId)
                            ?.trackId,
                        excludeClipIds: new Set(drag.selectedClipIds),
                    });
                }

                const minLen = 0.0;
                if (drag.type === "fade_in") {
                    const raw = beat - drag.basestartSec;
                    const next = clamp(raw, 0, Math.max(0, drag.baselengthSec));
                    const fadeUpdates = applyBulkFadeValue({
                        clipIds: drag.selectedClipIds,
                        clipsById: new Map(
                            sessionRef.current.clips.map((clip) => [clip.id, clip] as const),
                        ),
                        target: "fadeInSec",
                        nextValue: next,
                    });
                    batch(() => {
                        for (const update of fadeUpdates) {
                            dispatch(setClipFades(update));
                        }
                    });
                    try {
                        if (drag.selectedClipIds.length === 1) {
                            const now = Date.now();
                            const last = lastRemoteSentRef.current[drag.clipId] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[drag.clipId] = now;
                                void dispatch(
                                    setClipStateRemote({ clipId: drag.clipId, fadeInSec: next }),
                                );
                            }
                        }
                    } catch {
                        // Best-effort remote preview update; ignore transient failures.
                    }
                    return;
                }
                if (drag.type === "fade_out") {
                    const raw = drag.rightEdgeBeat - beat;
                    const next = clamp(raw, 0, Math.max(0, drag.baselengthSec));
                    const fadeUpdates = applyBulkFadeValue({
                        clipIds: drag.selectedClipIds,
                        clipsById: new Map(
                            sessionRef.current.clips.map((clip) => [clip.id, clip] as const),
                        ),
                        target: "fadeOutSec",
                        nextValue: next,
                    });
                    batch(() => {
                        for (const update of fadeUpdates) {
                            dispatch(setClipFades(update));
                        }
                    });
                    try {
                        if (drag.selectedClipIds.length === 1) {
                            const now = Date.now();
                            const last = lastRemoteSentRef.current[drag.clipId] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[drag.clipId] = now;
                                void dispatch(
                                    setClipStateRemote({ clipId: drag.clipId, fadeOutSec: next }),
                                );
                            }
                        }
                    } catch {
                        // Best-effort remote preview update; ignore transient failures.
                    }
                    return;
                }
                if (drag.type === "gain") {
                    if (gainDragStartClientY == null || !gainFineAxisState) {
                        gainDragStartClientY = currentEv.clientY;
                        gainFineAxisState = {
                            raw: currentEv.clientY,
                            adjusted: currentEv.clientY,
                            fineActive: isModifierActive(paramFineAdjustKb, currentEv),
                        };
                    }
                    const adjustedY = advanceFineAxisDrag(
                        gainFineAxisState,
                        currentEv.clientY,
                        isModifierActive(paramFineAdjustKb, currentEv),
                    );
                    const deltaY = gainDragStartClientY - adjustedY;
                    accumulatedGainDeltaDb = deltaY * CLIP_GAIN_DRAG_DB_PER_PX;
                    const gainUpdates = applyBulkGainDeltaDb({
                        clipIds: drag.selectedClipIds,
                        clipsById: new Map(
                            drag.selectedClipIds.map((id) => [
                                id,
                                { gain: drag.baseGainById[id] ?? 1 },
                            ]),
                        ),
                        deltaDb: accumulatedGainDeltaDb,
                        minDb: -12,
                        maxDb: 12,
                    });
                    batch(() => {
                        for (const update of gainUpdates) {
                            dispatch(setClipGain(update));
                        }
                    });
                    try {
                        if (drag.selectedClipIds.length === 1) {
                            const now = Date.now();
                            const last = lastRemoteSentRef.current[drag.clipId] || 0;
                            const nextGain = gainUpdates[0]?.gain;
                            if (nextGain != null && now - last > 200) {
                                lastRemoteSentRef.current[drag.clipId] = now;
                                const remoteUpdate = () =>
                                    webApi.setClipState({
                                        clipId: drag.clipId,
                                        gain: nextGain,
                                        checkpoint: false,
                                    });
                                const nextRemotePreview = gainUndoGroupPromise
                                    ? gainUndoGroupPromise.then(remoteUpdate)
                                    : Promise.resolve().then(remoteUpdate);
                                remotePreviewChain = remotePreviewChain
                                    .then(() => nextRemotePreview)
                                    .catch(() => undefined);
                            }
                        }
                    } catch {
                        // Best-effort remote preview update; ignore transient failures.
                    }
                    return;
                }

                if (
                    drag.stretchGroup &&
                    (drag.type === "stretch_left" || drag.type === "stretch_right")
                ) {
                    const stretchGroup = drag.stretchGroup;
                    const update = computeStretchGroupUpdate({
                        group: stretchGroup,
                        edge: drag.type,
                        pointerSec: beat,
                    });
                    batch(() => {
                        for (const clipId of stretchGroup.clipIds) {
                            const next = update.byId[clipId];
                            if (!next) continue;
                            dispatch(
                                moveClipStart({
                                    clipId,
                                    startSec: next.startSec,
                                }),
                            );
                            dispatch(
                                setClipLength({
                                    clipId,
                                    lengthSec: next.lengthSec,
                                }),
                            );
                            dispatch(
                                setClipPlaybackRate({
                                    clipId,
                                    playbackRate: next.playbackRate,
                                }),
                            );
                            dispatch(
                                setClipFades({
                                    clipId,
                                    fadeInSec: next.fadeInSec,
                                    fadeOutSec: next.fadeOutSec,
                                }),
                            );
                        }
                    });
                    return;
                }

                if (drag.type === "trim_left") {
                    const minLen = 0.0;
                    const anchorBase = drag.baseByClipId[drag.clipId];
                    if (!anchorBase) return;
                    const anchorRight = anchorBase.startSec + anchorBase.lengthSec;
                    const desiredStart = clamp(beat, 0, anchorRight - minLen);
                    const desiredDelta = desiredStart - anchorBase.startSec;

                    // Find the most constrained delta across all group members
                    let limitedDelta = desiredDelta;
                    for (const id of drag.selectedClipIds) {
                        const base = drag.baseByClipId[id];
                        if (!base) continue;
                        const rate = base.playbackRate > 0 ? base.playbackRate : 1;
                        if (base.reversed) {
                            const maxDelta = (base.sourceEndSec - base.sourceStartSec) / rate;
                            limitedDelta = Math.min(limitedDelta, maxDelta);
                        } else {
                            limitedDelta = Math.min(limitedDelta, base.lengthSec - minLen);
                            const minAllowed = -base.sourceStartSec / rate;
                            limitedDelta = Math.max(limitedDelta, minAllowed);
                        }
                    }

                    batch(() => {
                        for (const id of drag.selectedClipIds) {
                            const base = drag.baseByClipId[id];
                            if (!base) continue;
                            const rate = base.playbackRate > 0 ? base.playbackRate : 1;
                            if (base.reversed) {
                                const sourceDuration = (() => {
                                    if (
                                        base.durationFrames &&
                                        base.sourceSampleRate &&
                                        base.sourceSampleRate > 0
                                    ) {
                                        return base.durationFrames / base.sourceSampleRate;
                                    }
                                    return base.durationSec || 0;
                                })();
                                let nextTrimEnd = base.sourceEndSec - limitedDelta * rate;
                                nextTrimEnd = Math.max(base.sourceStartSec, nextTrimEnd);
                                if (sourceDuration > 0)
                                    nextTrimEnd = Math.min(nextTrimEnd, sourceDuration);
                                const actualDeltaTrim = base.sourceEndSec - nextTrimEnd;
                                const actualDeltaTimeline = actualDeltaTrim / rate;
                                const nextStart = base.startSec + actualDeltaTimeline;
                                const nextLen = clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                );
                                dispatch(moveClipStart({ clipId: id, startSec: nextStart }));
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                dispatch(
                                    setClipSourceRange({ clipId: id, sourceEndSec: nextTrimEnd }),
                                );
                            } else {
                                let nextTrimStart = base.sourceStartSec + limitedDelta * rate;
                                nextTrimStart = Math.max(0, nextTrimStart);
                                const actualDeltaTrim = nextTrimStart - base.sourceStartSec;
                                const actualDeltaTimeline = actualDeltaTrim / rate;
                                const nextStart = base.startSec + actualDeltaTimeline;
                                const nextLen = clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                );
                                dispatch(moveClipStart({ clipId: id, startSec: nextStart }));
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                dispatch(
                                    setClipSourceRange({
                                        clipId: id,
                                        sourceStartSec: nextTrimStart,
                                    }),
                                );
                            }
                        }
                    });
                    return;
                }

                if (drag.type === "stretch_left") {
                    const desiredStart = clamp(beat, 0, drag.rightEdgeBeat - minLen);
                    const rawLen = clamp(drag.rightEdgeBeat - desiredStart, minLen, 10_000);
                    const baseLen = Math.max(1e-6, Number(drag.baselengthSec) || 0);
                    const baseRate =
                        drag.basePlaybackRate > 0 && Number.isFinite(drag.basePlaybackRate)
                            ? drag.basePlaybackRate
                            : 1;
                    const nextRate = clamp((baseRate * baseLen) / Math.max(1e-6, rawLen), 0.1, 10);
                    const correctedLen = (baseRate * baseLen) / nextRate;
                    const nextStart = drag.rightEdgeBeat - correctedLen;
                    const scaledFades = scaleClipFadesForStretch({
                        baseFadeInSec: drag.basefadeInSec,
                        baseFadeOutSec: drag.basefadeOutSec,
                        baseLengthSec: drag.baselengthSec,
                        nextLengthSec: correctedLen,
                    });
                    dispatch(moveClipStart({ clipId: drag.clipId, startSec: nextStart }));
                    dispatch(setClipLength({ clipId: drag.clipId, lengthSec: correctedLen }));
                    dispatch(setClipPlaybackRate({ clipId: drag.clipId, playbackRate: nextRate }));
                    dispatch(
                        setClipFades({
                            clipId: drag.clipId,
                            fadeInSec: scaledFades.fadeInSec,
                            fadeOutSec: scaledFades.fadeOutSec,
                        }),
                    );
                    return;
                }

                if (drag.type === "trim_right") {
                    const minLen = 0.0;
                    const anchorBase = drag.baseByClipId[drag.clipId];
                    if (!anchorBase) return;

                    const desiredRight = clamp(beat, anchorBase.startSec + minLen, 10_000);
                    const desiredLen = desiredRight - anchorBase.startSec;
                    const nextLen = clamp(desiredLen, minLen, 10_000);
                    const desiredDeltaTimeline = nextLen - anchorBase.lengthSec;

                    // Find the most constrained delta across all group members
                    let limitedDelta = desiredDeltaTimeline;
                    for (const id of drag.selectedClipIds) {
                        const base = drag.baseByClipId[id];
                        if (!base) continue;
                        const rate = base.playbackRate > 0 ? base.playbackRate : 1;
                        const sourceDuration = (() => {
                            if (
                                base.durationFrames &&
                                base.sourceSampleRate &&
                                base.sourceSampleRate > 0
                            ) {
                                return base.durationFrames / base.sourceSampleRate;
                            }
                            return base.durationSec || 0;
                        })();
                        if (base.reversed) {
                            const maxSourceLen = base.sourceEndSec;
                            const maxTimelineLen = maxSourceLen / rate;
                            limitedDelta = Math.min(limitedDelta, maxTimelineLen - base.lengthSec);
                            limitedDelta = Math.max(limitedDelta, -base.lengthSec + minLen);
                        } else {
                            const maxSourceLen =
                                sourceDuration > 0
                                    ? sourceDuration - base.sourceStartSec
                                    : Number.POSITIVE_INFINITY;
                            const maxTimelineLen = maxSourceLen / rate;
                            limitedDelta = Math.min(limitedDelta, maxTimelineLen - base.lengthSec);
                            limitedDelta = Math.max(limitedDelta, -base.lengthSec + minLen);
                        }
                    }

                    batch(() => {
                        for (const id of drag.selectedClipIds) {
                            const base = drag.baseByClipId[id];
                            if (!base) continue;
                            const rate = base.playbackRate > 0 ? base.playbackRate : 1;
                            const sourceDuration = (() => {
                                if (
                                    base.durationFrames &&
                                    base.sourceSampleRate &&
                                    base.sourceSampleRate > 0
                                ) {
                                    return base.durationFrames / base.sourceSampleRate;
                                }
                                return base.durationSec || 0;
                            })();
                            if (base.reversed) {
                                let nextTrimStart = base.sourceStartSec - limitedDelta * rate;
                                nextTrimStart = Math.max(0, nextTrimStart);
                                nextTrimStart = Math.min(nextTrimStart, base.sourceEndSec);
                                const actualSourceLen = base.sourceEndSec - nextTrimStart;
                                const maxTimelineLen = actualSourceLen / rate;
                                const finalLen =
                                    maxTimelineLen > 0
                                        ? Math.min(base.lengthSec + limitedDelta, maxTimelineLen)
                                        : base.lengthSec + limitedDelta;
                                dispatch(setClipLength({ clipId: id, lengthSec: finalLen }));
                                dispatch(
                                    setClipSourceRange({
                                        clipId: id,
                                        sourceStartSec: nextTrimStart,
                                    }),
                                );
                            } else {
                                let nextTrimEnd = base.sourceEndSec + limitedDelta * rate;
                                nextTrimEnd = Math.max(0, nextTrimEnd);
                                if (sourceDuration > 0)
                                    nextTrimEnd = Math.min(nextTrimEnd, sourceDuration);
                                const actualSourceLen = nextTrimEnd - base.sourceStartSec;
                                const maxTimelineLen = actualSourceLen / rate;
                                const finalLen =
                                    maxTimelineLen > 0
                                        ? Math.min(base.lengthSec + limitedDelta, maxTimelineLen)
                                        : base.lengthSec + limitedDelta;
                                dispatch(setClipLength({ clipId: id, lengthSec: finalLen }));
                                dispatch(
                                    setClipSourceRange({ clipId: id, sourceEndSec: nextTrimEnd }),
                                );
                            }
                        }
                    });
                    return;
                }

                if (drag.type === "stretch_right") {
                    const desiredRight = clamp(beat, drag.basestartSec + minLen, 10_000);
                    const rawLen = clamp(desiredRight - drag.basestartSec, minLen, 10_000);
                    const baseLen = Math.max(1e-6, Number(drag.baselengthSec) || 0);
                    const baseRate =
                        drag.basePlaybackRate > 0 && Number.isFinite(drag.basePlaybackRate)
                            ? drag.basePlaybackRate
                            : 1;
                    const nextRate = clamp((baseRate * baseLen) / Math.max(1e-6, rawLen), 0.1, 10);
                    const correctedLen = (baseRate * baseLen) / nextRate;
                    const scaledFades = scaleClipFadesForStretch({
                        baseFadeInSec: drag.basefadeInSec,
                        baseFadeOutSec: drag.basefadeOutSec,
                        baseLengthSec: drag.baselengthSec,
                        nextLengthSec: correctedLen,
                    });
                    dispatch(setClipLength({ clipId: drag.clipId, lengthSec: correctedLen }));
                    dispatch(setClipPlaybackRate({ clipId: drag.clipId, playbackRate: nextRate }));
                    dispatch(
                        setClipFades({
                            clipId: drag.clipId,
                            fadeInSec: scaledFades.fadeInSec,
                            fadeOutSec: scaledFades.fadeOutSec,
                        }),
                    );
                }
            });
        }

        function end() {
            const drag = editDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            editDragRef.current = null;

            const isGroupStretch =
                drag.stretchGroup != null &&
                (drag.type === "stretch_left" || drag.type === "stretch_right");

            const clipNow = sessionRef.current.clips.find((c) => c.id === drag.clipId);
            if (!isGroupStretch && !clipNow) {
                if (gainUndoGroupPromise) {
                    void finishGainUndoGroup().catch(() => undefined);
                }
                dispatch(endInteraction());
                return;
            }
            const singleClipNow = clipNow ?? null;

            // 保存拉伸后的播放速率，persist 后重新应用（两阶段更新策略）
            let reapplyRates: Array<{ clipId: string; rate: number }> | null = null;

            const isMultiClipEdit = drag.selectedClipIds.length > 1;
            const autoCrossfadeClipIds =
                isMultiClipEdit || (isGroupStretch && drag.stretchGroup)
                    ? (drag.stretchGroup?.clipIds ?? drag.selectedClipIds)
                    : [drag.clipId];
            const shouldApplyAutoCrossfade =
                sessionRef.current.autoCrossfadeEnabled &&
                (drag.type === "trim_left" ||
                    drag.type === "trim_right" ||
                    drag.type === "stretch_left" ||
                    drag.type === "stretch_right");

            const runInsideUndoGroup = async (task: () => Promise<void>): Promise<void> => {
                await webApi.beginUndoGroup();
                try {
                    await task();
                } finally {
                    await webApi.endUndoGroup();
                }
            };

            const runWithOptionalAutoCrossfade = async (
                task: () => Promise<void>,
            ): Promise<void> => {
                if (!shouldApplyAutoCrossfade) {
                    await task();
                    return;
                }

                await runInsideUndoGroup(async () => {
                    await task();
                    await applyAutoCrossfade(sessionRef.current, autoCrossfadeClipIds, dispatch);
                });
            };

            // 交互锁在最终持久化请求完成后才释放，
            // 避免 endInteraction() 到 fulfilled 之间的窗口内，
            // 其他 in-flight thunk 的旧快照覆盖前端乐观更新导致闪烁。

            let persistPromise: Promise<unknown> | null = null;
            if (isGroupStretch && drag.stretchGroup) {
                const stretchPatches = drag.stretchGroup.clipIds
                    .map((id) => {
                        const now = sessionRef.current.clips.find((c) => c.id === id);
                        if (!now) return null;
                        return {
                            clipId: id,
                            startSec: now.startSec,
                            lengthSec: now.lengthSec,
                            playbackRate: now.playbackRate,
                            fadeInSec: now.fadeInSec,
                            fadeOutSec: now.fadeOutSec,
                        };
                    })
                    .filter(
                        (
                            patch,
                        ): patch is {
                            clipId: string;
                            startSec: number;
                            lengthSec: number;
                            playbackRate: number;
                            fadeInSec: number;
                            fadeOutSec: number;
                        } => patch != null,
                    );

                if (stretchPatches.length > 0) {
                    reapplyRates = stretchPatches
                        .filter((p) => p.playbackRate !== 1)
                        .map((p) => ({ clipId: p.clipId, rate: p.playbackRate }));
                    persistPromise = runInsideUndoGroup(async () => {
                        const stretchPersistPromises = stretchPatches.map((patch) =>
                            dispatch(
                                setClipStateRemote({
                                    clipId: patch.clipId,
                                    startSec: patch.startSec,
                                    lengthSec: patch.lengthSec,
                                    playbackRate: patch.playbackRate,
                                    fadeInSec: patch.fadeInSec,
                                    fadeOutSec: patch.fadeOutSec,
                                    checkpoint: false,
                                }),
                            ).unwrap(),
                        );
                        await Promise.allSettled(stretchPersistPromises);

                        if (shouldApplyAutoCrossfade) {
                            await applyAutoCrossfade(
                                sessionRef.current,
                                autoCrossfadeClipIds,
                                dispatch,
                            );
                        }
                    });
                }
            } else if (drag.type === "trim_left" && singleClipNow) {
                if (drag.selectedClipIds.length > 1) {
                    const trimPatches = drag.selectedClipIds
                        .map((id) => {
                            const now = sessionRef.current.clips.find((c) => c.id === id);
                            if (!now) return null;
                            return {
                                clipId: id,
                                startSec: now.startSec,
                                lengthSec: now.lengthSec,
                                reversed: now.reversed,
                                sourceStartSec: now.sourceStartSec,
                                sourceEndSec: now.sourceEndSec,
                            };
                        })
                        .filter((p) => p != null);
                    if (trimPatches.length > 0) {
                        persistPromise = runInsideUndoGroup(async () => {
                            const promises = trimPatches.map((patch) => {
                                const src = patch.reversed
                                    ? { sourceEndSec: patch.sourceEndSec }
                                    : { sourceStartSec: patch.sourceStartSec };
                                return dispatch(
                                    setClipStateRemote({
                                        clipId: patch.clipId,
                                        startSec: patch.startSec,
                                        lengthSec: patch.lengthSec,
                                        ...src,
                                        checkpoint: false,
                                    }),
                                ).unwrap();
                            });
                            await Promise.allSettled(promises);
                            if (shouldApplyAutoCrossfade) {
                                await applyAutoCrossfade(
                                    sessionRef.current,
                                    autoCrossfadeClipIds,
                                    dispatch,
                                );
                            }
                        });
                    }
                } else {
                    const sourceRangePatch = singleClipNow.reversed
                        ? { sourceEndSec: singleClipNow.sourceEndSec }
                        : { sourceStartSec: singleClipNow.sourceStartSec };
                    if (shouldApplyAutoCrossfade) {
                        persistPromise = runWithOptionalAutoCrossfade(async () => {
                            await dispatch(
                                setClipStateRemote({
                                    clipId: drag.clipId,
                                    startSec: singleClipNow.startSec,
                                    lengthSec: singleClipNow.lengthSec,
                                    ...sourceRangePatch,
                                    checkpoint: false,
                                }),
                            ).unwrap();
                        });
                    } else {
                        persistPromise = dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                startSec: singleClipNow.startSec,
                                lengthSec: singleClipNow.lengthSec,
                                ...sourceRangePatch,
                            }),
                        ).unwrap();
                    }
                }
            } else if (drag.type === "trim_right" && singleClipNow) {
                if (drag.selectedClipIds.length > 1) {
                    const trimPatches = drag.selectedClipIds
                        .map((id) => {
                            const now = sessionRef.current.clips.find((c) => c.id === id);
                            if (!now) return null;
                            return {
                                clipId: id,
                                lengthSec: now.lengthSec,
                                reversed: now.reversed,
                                sourceStartSec: now.sourceStartSec,
                                sourceEndSec: now.sourceEndSec,
                            };
                        })
                        .filter((p) => p != null);
                    if (trimPatches.length > 0) {
                        persistPromise = runInsideUndoGroup(async () => {
                            const promises = trimPatches.map((patch) => {
                                const src = patch.reversed
                                    ? { sourceStartSec: patch.sourceStartSec }
                                    : { sourceEndSec: patch.sourceEndSec };
                                return dispatch(
                                    setClipStateRemote({
                                        clipId: patch.clipId,
                                        lengthSec: patch.lengthSec,
                                        ...src,
                                        checkpoint: false,
                                    }),
                                ).unwrap();
                            });
                            await Promise.allSettled(promises);
                            if (shouldApplyAutoCrossfade) {
                                await applyAutoCrossfade(
                                    sessionRef.current,
                                    autoCrossfadeClipIds,
                                    dispatch,
                                );
                            }
                        });
                    }
                } else {
                    const sourceRangePatch = singleClipNow.reversed
                        ? { sourceStartSec: singleClipNow.sourceStartSec }
                        : { sourceEndSec: singleClipNow.sourceEndSec };
                    if (shouldApplyAutoCrossfade) {
                        persistPromise = runWithOptionalAutoCrossfade(async () => {
                            await dispatch(
                                setClipStateRemote({
                                    clipId: drag.clipId,
                                    lengthSec: singleClipNow.lengthSec,
                                    ...sourceRangePatch,
                                    checkpoint: false,
                                }),
                            ).unwrap();
                        });
                    } else {
                        persistPromise = dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                lengthSec: singleClipNow.lengthSec,
                                ...sourceRangePatch,
                            }),
                        ).unwrap();
                    }
                }
            } else if (drag.type === "stretch_left" && singleClipNow) {
                if (shouldApplyAutoCrossfade) {
                    persistPromise = runWithOptionalAutoCrossfade(async () => {
                        await dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                startSec: singleClipNow.startSec,
                                lengthSec: singleClipNow.lengthSec,
                                playbackRate: singleClipNow.playbackRate,
                                fadeInSec: singleClipNow.fadeInSec,
                                fadeOutSec: singleClipNow.fadeOutSec,
                                checkpoint: false,
                            }),
                        ).unwrap();
                    });
                } else {
                    persistPromise = dispatch(
                        setClipStateRemote({
                            clipId: drag.clipId,
                            startSec: singleClipNow.startSec,
                            lengthSec: singleClipNow.lengthSec,
                            playbackRate: singleClipNow.playbackRate,
                            fadeInSec: singleClipNow.fadeInSec,
                            fadeOutSec: singleClipNow.fadeOutSec,
                        }),
                    ).unwrap();
                }
                if (singleClipNow.playbackRate !== 1) {
                    reapplyRates = [{ clipId: drag.clipId, rate: singleClipNow.playbackRate }];
                }
            } else if (drag.type === "stretch_right" && singleClipNow) {
                if (shouldApplyAutoCrossfade) {
                    persistPromise = runWithOptionalAutoCrossfade(async () => {
                        await dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                lengthSec: singleClipNow.lengthSec,
                                playbackRate: singleClipNow.playbackRate,
                                fadeInSec: singleClipNow.fadeInSec,
                                fadeOutSec: singleClipNow.fadeOutSec,
                                checkpoint: false,
                            }),
                        ).unwrap();
                    });
                } else {
                    persistPromise = dispatch(
                        setClipStateRemote({
                            clipId: drag.clipId,
                            lengthSec: singleClipNow.lengthSec,
                            playbackRate: singleClipNow.playbackRate,
                            fadeInSec: singleClipNow.fadeInSec,
                            fadeOutSec: singleClipNow.fadeOutSec,
                        }),
                    ).unwrap();
                }
                if (singleClipNow.playbackRate !== 1) {
                    reapplyRates = [{ clipId: drag.clipId, rate: singleClipNow.playbackRate }];
                }
            } else if (drag.type === "fade_in" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        return [clipId, { fadeInSec: nextClip?.fadeInSec ?? 0 }] as const;
                    }),
                );
                persistPromise = dispatch(
                    setClipsStateBulkRemote({
                        updates: buildBulkClipStateUpdates({
                            clipIds: drag.selectedClipIds,
                            changesById,
                        }),
                    }),
                ).unwrap();
            } else if (drag.type === "fade_out" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        return [clipId, { fadeOutSec: nextClip?.fadeOutSec ?? 0 }] as const;
                    }),
                );
                persistPromise = dispatch(
                    setClipsStateBulkRemote({
                        updates: buildBulkClipStateUpdates({
                            clipIds: drag.selectedClipIds,
                            changesById,
                        }),
                    }),
                ).unwrap();
            } else if (drag.type === "gain" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        return [clipId, { gain: nextClip?.gain ?? 1 }] as const;
                    }),
                );
                const persistBulkGain = () =>
                    remotePreviewChain.then(() =>
                        dispatch(
                            setClipsStateBulkRemote({
                                updates: buildBulkClipStateUpdates({
                                    clipIds: drag.selectedClipIds,
                                    changesById,
                                }),
                            }),
                        ).unwrap(),
                    );
                persistPromise = gainUndoGroupPromise
                    ? gainUndoGroupPromise.then(persistBulkGain)
                    : persistBulkGain();
            }

            // 两阶段播放速率更新：后端响应后重新应用前端计算的值
            if (reapplyRates && reapplyRates.length > 0 && persistPromise) {
                void persistPromise.then(() => {
                    for (const { clipId, rate } of reapplyRates!) {
                        dispatch(setClipPlaybackRate({ clipId, playbackRate: rate }));
                    }
                });
            }

            // 拉伸后同步参数线：当"锁定参数线"启用时，将旧范围内的参数值时域映射到新范围
            const isStretch = drag.type === "stretch_left" || drag.type === "stretch_right";
            // The clip persistence already bumps paramsEpoch, but at that point
            // stretchLinkedParams has not written the new curves yet. Bump again
            // after the mapping finishes so the parameter editor fetches fresh data.
            if (isStretch && sessionRef.current.lockParamLinesEnabled && drag.stretchGroup) {
                const mappingsByRootTrack = new Map<string, StretchRangeMapping[]>();
                for (const id of drag.stretchGroup.clipIds) {
                    const initial = drag.stretchGroup?.initialById[id];
                    const now = sessionRef.current.clips.find((c) => c.id === id);
                    if (!initial || !now?.trackId) continue;
                    const rootTrackId = resolveRootTrackId(sessionRef.current.tracks, now.trackId);
                    if (!rootTrackId) continue;
                    const trackMappings = mappingsByRootTrack.get(rootTrackId) ?? [];
                    trackMappings.push({
                        oldStartSec: initial.startSec,
                        oldLengthSec: initial.lengthSec,
                        newStartSec: now.startSec,
                        newLengthSec: now.lengthSec,
                    });
                    mappingsByRootTrack.set(rootTrackId, trackMappings);
                }
                const stretchTasks = Array.from(mappingsByRootTrack, ([trackId, mappings]) =>
                    stretchTrackLinkedParams(trackId, mappings),
                );
                void Promise.resolve(persistPromise)
                    .then(() => Promise.allSettled(stretchTasks))
                    .finally(() => dispatch(bumpParamsEpoch()));
            } else if (
                isStretch &&
                sessionRef.current.lockParamLinesEnabled &&
                singleClipNow?.trackId
            ) {
                const stretchTrackId = singleClipNow.trackId;
                const oldStartSec = drag.basestartSec;
                const oldLengthSec = drag.baselengthSec;
                const newStartSec = singleClipNow.startSec;
                const newLengthSec = singleClipNow.lengthSec;
                void Promise.resolve(persistPromise)
                    .then(() =>
                        stretchLinkedParams(
                            stretchTrackId,
                            oldStartSec,
                            oldLengthSec,
                            newStartSec,
                            newLengthSec,
                        ),
                    )
                    .finally(() => dispatch(bumpParamsEpoch()));
            }

            // 在所有持久化请求完成后释放交互锁
            void Promise.resolve(persistPromise).finally(async () => {
                if (gainUndoGroupPromise) {
                    try {
                        await finishGainUndoGroup();
                    } catch {
                        // Best-effort undo-group cleanup.
                    }
                }
                dispatch(endInteraction());
            });

            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return { editDragRef, startEditDrag };
}
