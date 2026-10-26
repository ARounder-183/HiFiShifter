// 时间轴 MIDI 导入 / 替换 / 导出与 Compose 联动。拆自 TimelinePanel。
import {
    setTrackStateRemote,
    importMidiAsClip,
    replaceMidiClipDataRemote,
} from "../../../../features/session/sessionSlice";
import { webApi } from "../../../../services/webviewApi";
import { coreApi } from "../../../../services/api/core";
import { paramsApi } from "../../../../services/api/params";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import { SCALE_NOTES } from "../../../../utils/musicalScales";
import type { TimelineSessionSlice } from "./useTimelineState";
import type { AppDispatch } from "../../../../app/store";
import React from "react";

type ReplaceMidiDialog = { open: boolean; clipId: string | null; midiPath: string | null };

type Deps = {
    dispatch: AppDispatch;
    s: TimelineSessionSlice;
    sessionRef: React.MutableRefObject<TimelineSessionSlice>;
    midiClipTrackId: string | null;
    midiClipStartSec: number;
    replaceMidiDialog: ReplaceMidiDialog;
    setReplaceMidiDialog: React.Dispatch<React.SetStateAction<ReplaceMidiDialog>>;
};

export function useTimelineMidiActions({
    dispatch,
    s,
    sessionRef,
    midiClipTrackId,
    midiClipStartSec,
    replaceMidiDialog,
    setReplaceMidiDialog,
}: Deps) {
    const handleMidiClipImport = React.useCallback(
        (result: {
            trackIndices: number[];
            notesCount: number;
            midiPath: string;
            fillGaps: boolean;
            multiTrackMerge?: boolean;
            noteBpmMode?: string;
            specifiedBpm?: number;
            importBpmAsProject?: boolean;
            clipboardGuid?: string;
            closeLeadingGap?: boolean;
            importAsTempoMap?: boolean;
            importTempo?: boolean;
            importTimeSignature?: boolean;
            importKeySignature?: boolean;
        }) => {
            void dispatch(
                importMidiAsClip({
                    midiPath: result.midiPath,
                    trackIndices: result.trackIndices,
                    trackId: midiClipTrackId,
                    startSec: midiClipStartSec,
                    fillGaps: result.fillGaps || undefined,
                    multiTrackMerge: result.multiTrackMerge,
                    noteBpmMode: result.noteBpmMode,
                    specifiedBpm: result.specifiedBpm,
                    importBpmAsProject: result.importBpmAsProject,
                    clipboardGuid: result.clipboardGuid,
                    closeLeadingGap: result.closeLeadingGap,
                    importAsTempoMap: result.importAsTempoMap,
                    importTempo: result.importTempo,
                    importTimeSignature: result.importTimeSignature,
                    importKeySignature: result.importKeySignature,
                }),
            );
        },
        [dispatch, midiClipTrackId, midiClipStartSec],
    );

    // ── Replace MIDI ──
    const handleReplaceMidiImport = React.useCallback(
        (result: {
            trackIndices: number[];
            notesCount: number;
            midiPath: string;
            fillGaps: boolean;
            multiTrackMerge?: boolean;
            noteBpmMode?: string;
            specifiedBpm?: number;
            importBpmAsProject?: boolean;
            closeLeadingGap?: boolean;
        }) => {
            const clipId = replaceMidiDialog.clipId;
            if (!clipId) return;
            void dispatch(
                replaceMidiClipDataRemote({
                    clipId,
                    midiPath: result.midiPath,
                    trackIndices: result.trackIndices,
                    fillGaps: result.fillGaps || undefined,
                    noteBpmMode: result.noteBpmMode,
                    specifiedBpm: result.specifiedBpm,
                    importMidiBpmAsProject: result.importBpmAsProject,
                    closeLeadingGap: result.closeLeadingGap,
                }),
            );
            setReplaceMidiDialog({ open: false, clipId: null, midiPath: null });
        },
        [dispatch, replaceMidiDialog.clipId, setReplaceMidiDialog],
    );

    const openReplaceMidiForClip = React.useCallback(
        async (clipId: string) => {
            const picked = await webApi.openMidiDialog();
            if (!picked.ok || picked.canceled || !picked.path) return;
            setReplaceMidiDialog({ open: true, clipId, midiPath: picked.path });
        },
        [setReplaceMidiDialog],
    );

    const midiClipRootTrackComposeEnabled = React.useMemo(() => {
        if (!midiClipTrackId) return true;
        const rootId = resolveRootTrackId(s.tracks, midiClipTrackId);
        if (!rootId) return true;
        const rootTrack = s.tracks.find((t) => t.id === rootId);
        return rootTrack?.composeEnabled ?? true;
    }, [midiClipTrackId, s.tracks]);

    const handleRequestEnableCompose = React.useCallback(() => {
        if (!midiClipTrackId) return;
        const rootId = resolveRootTrackId(s.tracks, midiClipTrackId);
        if (!rootId) return;
        dispatch(
            setTrackStateRemote({
                trackId: rootId,
                composeEnabled: true,
            }),
        );
    }, [dispatch, midiClipTrackId, s.tracks]);

    const handleExportMidi = React.useCallback(
        async (clipIds: string[]) => {
            const saveResult = await coreApi.pickMidiOutputPath();
            if (!saveResult.ok || saveResult.canceled || !saveResult.path) return;

            const s = sessionRef.current;
            const clipsMap = new Map(s.clips.map((c) => [c.id, c]));
            const trackMap = new Map(s.tracks.map((t) => [t.id, t]));

            const entries: Array<{
                trackId: string;
                rootTrackId: string;
                name: string;
                startSec: number;
                endSec: number;
                clipId?: string;
            }> = [];
            const seenComposeRoots = new Set<string>();

            for (const id of clipIds) {
                const clip = clipsMap.get(id);
                if (!clip) continue;
                const rootId = resolveRootTrackId(s.tracks, clip.trackId);
                if (!rootId) continue;
                const rootTrack = trackMap.get(rootId);
                const isComposeEnabled = rootTrack?.composeEnabled ?? false;

                if (isComposeEnabled) {
                    // Compose 轨道：按 rootTrackId 去重（共享 track 级音高数据）
                    if (seenComposeRoots.has(rootId)) continue;
                    seenComposeRoots.add(rootId);
                }

                const track = trackMap.get(clip.trackId);
                entries.push({
                    trackId: clip.trackId,
                    rootTrackId: rootId,
                    name: track?.name ?? clip.name,
                    startSec: clip.startSec,
                    endSec: clip.startSec + clip.lengthSec,
                    ...(isComposeEnabled ? {} : { clipId: clip.id }),
                });
            }

            if (entries.length === 0) return;

            const scaleNotes =
                SCALE_NOTES[(s.project?.baseScale as keyof typeof SCALE_NOTES) ?? "C"] ??
                SCALE_NOTES.C;

            await paramsApi.exportPitchToMidi({
                outputPath: saveResult.path,
                tracks: entries,
                bpm: s.bpm,
                beatsPerBar: s.project?.beatsPerBar ?? 4,
                baseScale: s.project?.baseScale ?? "C",
                projectScaleNotes: scaleNotes,
            });
        },
        [sessionRef],
    );

    return {
        handleMidiClipImport,
        handleReplaceMidiImport,
        openReplaceMidiForClip,
        midiClipRootTrackComposeEnabled,
        handleRequestEnableCompose,
        handleExportMidi,
    };
}
