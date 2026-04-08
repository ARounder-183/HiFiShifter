/**
 * Helpers for computing child-track pitch-offset paste conversions.
 *
 * This module centralizes the logic used by both the PianoRoll panel and the
 * piano-roll interactions hook to convert a clipboard pitch curve into the
 * per-child-track pitch-offset parameter values (either cents or scale degrees).
 *
 * Exported functions:
 * - `resolveChildLineage(tracks, trackId)` - compute lineage from root's child
 *    down to the target track (excluding the root track itself).
 * - `buildChildOffsetPasteValues(args)` - given clipboard pitch frames and
 *    project state, compute the child-offset param values matching the project's
 *    internal representation.
 */

import type { ScaleLike } from "../../../utils/musicalScales";
import type { ParamFramesPayload } from "../../../types/api";

import { transposePitchByScaleSteps } from "../../../utils/musicalScales";
import {
    buildChildPitchOffsetCentsParam,
    buildChildPitchOffsetDegreesParam,
    CHILD_PITCH_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_DEGREES_RANGE,
} from "./childPitchOffsetParams";
import { clamp } from "../timeline";

interface ChildOffsetParamsApi {
    getParamFrames: (
        trackId: string,
        param: string,
        startFrame: number,
        frameCount: number,
        stride?: number,
    ) => Promise<ParamFramesPayload>;
}

export function resolveChildLineage(
    tracks: Array<{ id: string; parentId?: string | null }>,
    trackId: string,
): string[] {
    const byId = new Map(tracks.map((track) => [track.id, track] as const));
    const out: string[] = [];
    let cursor: string | null = trackId;
    let safety = 0;
    while (cursor && safety < tracks.length + 2) {
        const node = byId.get(cursor);
        if (!node || !node.parentId) break;
        out.push(node.id);
        cursor = node.parentId ?? null;
        safety += 1;
    }
    out.reverse();
    return out;
}

export async function buildChildOffsetPasteValues(args: {
    tracks: Array<{ id: string; parentId?: string | null }>;
    rootTrackId: string | null;
    targetTrackId: string;
    startFrame: number;
    frameCount: number;
    clipboardPitch: number[];
    mode: "cents" | "degrees";
    paramsApi: ChildOffsetParamsApi;
    pitchDeltaToDegreeSteps: (basePitch: number, targetPitch: number, scale: ScaleLike) => number;
    projectScale: ScaleLike | undefined;
}): Promise<number[] | null> {
    const {
        tracks,
        rootTrackId,
        targetTrackId,
        startFrame,
        frameCount,
        clipboardPitch,
        mode,
        paramsApi,
        pitchDeltaToDegreeSteps,
        projectScale,
    } = args;

    if (!rootTrackId || !projectScale) return null;

    const rootPitchPayload = await paramsApi.getParamFrames(
        rootTrackId,
        "pitch",
        startFrame,
        frameCount,
        1,
    );
    if (!rootPitchPayload?.ok) return null;

    const lineage = resolveChildLineage(tracks, targetTrackId);
    if (lineage.length === 0) return null;

    const curvePromises = lineage.flatMap((trackId) => [
        paramsApi.getParamFrames(
            rootTrackId,
            buildChildPitchOffsetCentsParam(trackId),
            startFrame,
            frameCount,
            1,
        ),
        paramsApi.getParamFrames(
            rootTrackId,
            buildChildPitchOffsetDegreesParam(trackId),
            startFrame,
            frameCount,
            1,
        ),
    ]);
    const curvePayloads = await Promise.all(curvePromises);

    const rootEdit = Array.isArray(rootPitchPayload.edit)
        ? rootPitchPayload.edit.map((v) => Number(v) || 0)
        : [];

    const curvesByTrack = new Map<string, { cents: number[]; degrees: number[] }>();
    for (let i = 0; i < lineage.length; i += 1) {
        const trackId = lineage[i];
        const centsPayload = curvePayloads[i * 2];
        const degreesPayload = curvePayloads[i * 2 + 1];
        curvesByTrack.set(trackId, {
            cents:
                centsPayload?.ok && Array.isArray(centsPayload.edit)
                    ? centsPayload.edit.map((v) => Number(v) || 0)
                    : [],
            degrees:
                degreesPayload?.ok && Array.isArray(degreesPayload.edit)
                    ? degreesPayload.edit.map((v) => Number(v) || 0)
                    : [],
        });
    }

    const out: number[] = new Array(frameCount).fill(0);

    for (let i = 0; i < frameCount; i += 1) {
        let tempPitch = Number(rootEdit[i] ?? 0);
        if (!(Number.isFinite(tempPitch) && tempPitch > 0)) {
            out[i] = 0;
            continue;
        }

        for (const trackId of lineage) {
            const curves = curvesByTrack.get(trackId);
            if (!curves) continue;

            const applyDegrees = !(trackId === targetTrackId && mode === "degrees");
            const applyCents = !(trackId === targetTrackId && mode === "cents");

            if (applyDegrees) {
                const degreeSteps = Number(curves.degrees[i] ?? 0);
                if (Math.abs(degreeSteps) > 1e-9) {
                    tempPitch = transposePitchByScaleSteps(tempPitch, degreeSteps, projectScale);
                }
            }

            if (applyCents) {
                const cents = Number(curves.cents[i] ?? 0);
                if (Math.abs(cents) > 1e-9) {
                    tempPitch += cents / 100;
                }
            }
        }

        const targetPitch = Number(clipboardPitch[i] ?? 0);
        if (!(Number.isFinite(targetPitch) && targetPitch > 0)) {
            out[i] = 0;
            continue;
        }

        if (mode === "cents") {
            const cents = (targetPitch - tempPitch) * 100;
            out[i] = clamp(
                cents,
                CHILD_PITCH_OFFSET_CENTS_RANGE.min,
                CHILD_PITCH_OFFSET_CENTS_RANGE.max,
            );
        } else {
            const degreeSteps = pitchDeltaToDegreeSteps(tempPitch, targetPitch, projectScale);
            out[i] = clamp(
                degreeSteps,
                CHILD_PITCH_OFFSET_DEGREES_RANGE.min,
                CHILD_PITCH_OFFSET_DEGREES_RANGE.max,
            );
        }
    }

    return out;
}
