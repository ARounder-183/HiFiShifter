/**
 * 参考音高、颤音提取与 MIDI 导出：几条"把当前编辑结果交出去"的命令。
 *
 * 【为什么单独成文件】它们的共同点是**读一大片当前状态、产出一样东西**（一条参考
 * 曲线 / 一个预设 / 一个 MIDI 文件），而不是改状态或摆布局。与 `editOperations` 的
 * 区别是：那边写参数帧，这边只读。
 *
 * 【依赖怎么拿】捕获值由模型打包传入（`Pick<ModelDeps, …>`，AST 扫描得出），
 * 函数体原样搬过来。
 *
 * 【返回值为什么必须显式标注】模型类型与本文件的入参类型之间不能成环，否则
 * TS 会把整条链退化成 `any`（详见 `modelDeps.ts`）。标注返回值即切断环。
 */
import {
    persistUiSettings,
    setActiveVibratoPreset,
    createClipsRemote,
    addTrackRemote,
} from "../../../../features/session/sessionSlice";
import { paramsApi } from "../../../../services/api/params";
import { coreApi } from "../../../../services/api/core";
import type { ParamFramesPayload } from "../../../../types/api";
import { SCALE_NOTES } from "../../../../utils/musicalScales";
import { frameRangeEnd, selectionToFrameRanges } from "../paramSelection";
import { isDynParam } from "../paramRanges";
import { framesToTime } from "../utils";
import { extractVibratoPreset } from "../../../../features/vibrato/vibratoExtract";
import type { VibratoPreset } from "../../../../features/vibrato/vibratoTypes";
import { upsertVibratoPreset } from "../../../../features/session/sessionSlice";
import { VIBRATO_APPLY_PREVIEW_FRAMES } from "./panelHelpers";

import type { ModelDeps } from "./modelDeps";

export async function handleSaveAsPitchRef(
    deps: Pick<
        ModelDeps,
        "dispatch" | "paramView" | "rootTrackId" | "s" | "selectAllParamRange" | "selectionRef"
    >,
): Promise<void> {
    const { dispatch, paramView, rootTrackId, s, selectAllParamRange, selectionRef } = deps;

    if (!rootTrackId) return;
    if (!selectionRef.current || selectionRef.current.length === 0) {
        selectAllParamRange();
    }
    const sel = selectionRef.current;
    if (!sel || sel.length === 0) return;

    const fp = paramView?.framePeriodMs ?? 5;
    const selFrameRanges = selectionToFrameRanges(sel);
    if (selFrameRanges.length === 0) return;

    // 逐段取 pitch → MIDI 音符事件（与旧单选区同一转换，保留浮点音高）
    const clipTemplates: Array<{
        startSec: number;
        lengthSec: number;
        midiNoteData: Array<{
            startSec: number;
            endSec: number;
            note: number;
            velocity: number;
            channel: number;
        }>;
    }> = [];
    for (const range of selFrameRanges) {
        const res = await paramsApi.getParamFrames(
            rootTrackId,
            "pitch",
            range.startFrame,
            range.frameCount,
            1,
        );
        if (!res?.ok || !res.edit) continue;
        const pitchValues: number[] = (res.edit as number[]).map((v) => Number(v) || 0);
        if (pitchValues.length === 0) continue;

        const startSec = (range.startFrame * fp) / 1000;
        const lengthSec = Math.max(0.01, (pitchValues.length * fp) / 1000);

        // Convert pitch values (semitones) to MIDI note events
        // 保留原始浮点音高值，不进行半音量化
        const fpSec = fp / 1000;
        const midiNoteData: Array<{
            startSec: number;
            endSec: number;
            note: number;
            velocity: number;
            channel: number;
        }> = [];
        let segStartFrame = 0;
        let currentNote = pitchValues[0];
        for (let i = 1; i < pitchValues.length; i++) {
            const note = pitchValues[i];
            if (Math.abs(note - currentNote) > 0.001) {
                midiNoteData.push({
                    startSec: segStartFrame * fpSec,
                    endSec: i * fpSec,
                    note: currentNote,
                    velocity: 100,
                    channel: 0,
                });
                segStartFrame = i;
                currentNote = note;
            }
        }
        midiNoteData.push({
            startSec: segStartFrame * fpSec,
            endSec: pitchValues.length * fpSec,
            note: currentNote,
            velocity: 100,
            channel: 0,
        });
        clipTemplates.push({ startSec, lengthSec, midiNoteData });
    }
    if (clipTemplates.length === 0) return;

    // Determine target track: try the track above the currently selected track.
    // If no track above exists, or the above track has overlapping clips
    // in the import time range, create a new track above the current track.
    const orderedTrackIds = s.tracks.map((t) => t.id);
    const trackIndexById: Record<string, number> = {};
    orderedTrackIds.forEach((id, idx) => {
        trackIndexById[id] = idx;
    });

    const currentIdx = s.selectedTrackId ? (trackIndexById[s.selectedTrackId] ?? -1) : -1;
    let targetTrackId: string | null = null;

    if (currentIdx > 0) {
        const aboveTrackId = orderedTrackIds[currentIdx - 1];

        const hasOverlap = s.clips.some(
            (c) =>
                c.trackId === aboveTrackId &&
                clipTemplates.some(
                    (template) =>
                        c.startSec < template.startSec + template.lengthSec &&
                        c.startSec + c.lengthSec > template.startSec,
                ),
        );
        if (!hasOverlap) {
            const currentTrack = s.tracks.find((t) => t.id === s.selectedTrackId);
            const aboveTrack = s.tracks.find((t) => t.id === aboveTrackId);
            if (
                currentTrack &&
                aboveTrack &&
                currentTrack.depth != null &&
                aboveTrack.depth != null &&
                currentTrack.depth >= aboveTrack.depth
            ) {
                targetTrackId = aboveTrackId;
            }
        }
    }

    if (!targetTrackId) {
        // Create a new track above the current track
        const currentTrack = s.tracks.find((t) => t.id === s.selectedTrackId);
        const newTrackPayload: Record<string, unknown> = {
            name: undefined,
            parentTrackId: currentTrack?.parentId ?? null,
        };
        if (currentIdx >= 0) {
            newTrackPayload.index = currentIdx;
        }
        const result = await dispatch(
            addTrackRemote(newTrackPayload as { name?: string; parentTrackId?: string | null }),
        ).unwrap();
        const added = result as {
            selected_track_id?: string;
            tracks?: Array<{ id: string }>;
        };
        targetTrackId =
            added.selected_track_id ?? added.tracks?.[added.tracks.length - 1]?.id ?? null;
    }

    if (!targetTrackId) return;

    await dispatch(
        createClipsRemote({
            templates: clipTemplates.map((template) => ({
                trackId: targetTrackId as string,
                name: "Pitch Ref",
                startSec: template.startSec,
                lengthSec: template.lengthSec,
                midiNoteData: template.midiNoteData,
                midiFillGaps: true,
            })),
        }),
    );
}

export async function handleExtractVibratoPreset(
    deps: Pick<
        ModelDeps,
        | "currentParamRange"
        | "dispatch"
        | "editParam"
        | "paramView"
        | "rootTrackId"
        | "selectAllParamRange"
        | "selectionRef"
        | "t"
    >,
): Promise<VibratoPreset | null> {
    const {
        currentParamRange,
        dispatch,
        editParam,
        paramView,
        rootTrackId,
        selectAllParamRange,
        selectionRef,
        t,
    } = deps;

    if (!rootTrackId) return null;
    if (!selectionRef.current || selectionRef.current.length === 0) {
        selectAllParamRange();
    }
    const ranges = selectionRef.current ? selectionToFrameRanges(selectionRef.current) : [];
    const range = ranges[0];
    if (!range || range.frameCount < 8) return null;

    const fp = paramView?.framePeriodMs ?? 5;
    const res = await paramsApi.getParamFrames(
        rootTrackId,
        editParam,
        range.startFrame,
        range.frameCount,
        1,
        true,
        isDynParam(editParam),
    );
    if (!res?.ok) return null;
    const payload = res as ParamFramesPayload;
    const values = (payload.edit ?? []).map((value) => Number(value) || 0);
    const fpMs = Number(payload.frame_period_ms ?? fp) || fp;
    const extracted = extractVibratoPreset({
        values,
        framePeriodMs: fpMs,
        param: editParam,
        range: currentParamRange,
    });
    if (!extracted.ok) return null;

    const preset = {
        ...extracted.preset,
        name: t("vibrato_from_selection"),
    };
    dispatch(upsertVibratoPreset(preset));
    dispatch(setActiveVibratoPreset(preset.id));
    void dispatch(persistUiSettings());
    return preset;
}

export async function handleExportMidiFromEditor(
    deps: Pick<
        ModelDeps,
        "paramView" | "rootTrackId" | "s" | "selectAllParamRange" | "selectionRef"
    >,
): Promise<void> {
    const { paramView, rootTrackId, s, selectAllParamRange, selectionRef } = deps;

    if (!rootTrackId) return;
    // 无选区时先隐式全选（作用域 = 整条参数曲线），与菜单里其它以选区为
    // 作用域的操作一致。
    if (!selectionRef.current || selectionRef.current.length === 0) {
        selectAllParamRange();
    }
    const sel = selectionRef.current;
    if (!sel || sel.length === 0) return;

    const saveResult = await coreApi.pickMidiOutputPath();
    if (!saveResult.ok || saveResult.canceled || !saveResult.path) return;

    const selectedTrack = s.tracks.find((t) => t.id === s.selectedTrackId);
    const trackName = selectedTrack?.name ?? "Track";
    const scaleNotes =
        SCALE_NOTES[(s.project?.baseScale as keyof typeof SCALE_NOTES) ?? "C"] ?? SCALE_NOTES.C;

    await paramsApi.exportPitchToMidi({
        outputPath: saveResult.path,
        tracks: sel.map((range) => {
            // 半开帧区间 → 秒（右端是最后一帧的右缘）。
            const fp = paramView?.framePeriodMs ?? 5;
            const startSec = framesToTime(range.startFrame, fp);
            const endSec = Math.max(startSec + 0.01, framesToTime(frameRangeEnd(range), fp));
            return {
                trackId: s.selectedTrackId ?? rootTrackId,
                rootTrackId,
                name: trackName,
                startSec,
                endSec,
            };
        }),
        bpm: s.bpm,
        beatsPerBar: s.project?.beatsPerBar ?? 4,
        baseScale: s.project?.baseScale ?? "C",
        projectScaleNotes: scaleNotes,
    });
}

export async function loadVibratoApplyOriginal(
    deps: Pick<ModelDeps, "editParam" | "paramView" | "rootTrackId" | "selectionRef">,
): Promise<{ values: number[]; framePeriodMs: number } | null> {
    const { editParam, paramView, rootTrackId, selectionRef } = deps;

    if (!rootTrackId) return null;
    const ranges = selectionRef.current ? selectionToFrameRanges(selectionRef.current) : [];
    const range = ranges[0] ?? { startFrame: 0, frameCount: VIBRATO_APPLY_PREVIEW_FRAMES };
    const frameCount = Math.min(range.frameCount, VIBRATO_APPLY_PREVIEW_FRAMES);
    if (frameCount < 2) return null;

    const fp = paramView?.framePeriodMs ?? 5;
    const res = await paramsApi.getParamFrames(
        rootTrackId,
        editParam,
        range.startFrame,
        frameCount,
        1,
        true,
        isDynParam(editParam),
    );
    if (!res?.ok) return null;
    const payload = res as ParamFramesPayload;
    const values = (payload.edit ?? []).map((value) => Number(value) || 0);
    if (values.length < 2) return null;
    const framePeriodMs = Number(payload.frame_period_ms ?? fp) || fp;
    return { values, framePeriodMs };
}

export function handleApplyVibrato(
    deps: Pick<ModelDeps, "handleEditOp">,
    preset: VibratoPreset,
): void {
    const { handleEditOp } = deps;

    void handleEditOp("addVibrato", { preset });
}
