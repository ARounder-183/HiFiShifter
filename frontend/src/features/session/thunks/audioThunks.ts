import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import { requestMissingFileReplacement } from "./missingFilePrompt";

async function resolveMissingFilesInteractively(
    timeline: any,
    missingFiles: string[] | undefined,
) {
    let latestTimeline = timeline;
    const uniquePaths = Array.from(
        new Set((missingFiles ?? []).filter((p) => typeof p === "string" && p.trim().length > 0)),
    );

    for (const missingPath of uniquePaths) {
        const shouldPick = await requestMissingFileReplacement(missingPath);
        if (!shouldPick) continue;

        const picked = await webApi.openAudioDialog();
        if (!picked.ok || picked.canceled || !picked.path) continue;

        const targetClipIds = (latestTimeline?.clips ?? [])
            .filter((clip: any) => clip?.source_path === missingPath)
            .map((clip: any) => clip.id)
            .filter((id: unknown): id is string => typeof id === "string");

        if (targetClipIds.length === 0) continue;

        const replaced = await webApi.replaceClipSource({
            clipIds: targetClipIds,
            newSourcePath: picked.path,
            replaceSameSource: true,
        });
        if (replaced?.ok) {
            latestTimeline = replaced;
        }
    }

    return latestTimeline;
}

export const processAudio = createAsyncThunk(
    "session/processAudio",
    async (audioPath: string) => {
        return webApi.processAudio(audioPath);
    },
);

export const pickOutputPath = createAsyncThunk(
    "session/pickOutputPath",
    async (_, { rejectWithValue }) => {
        const picked = await webApi.pickOutputPath();
        if (!picked.ok) {
            return rejectWithValue("pick_output_path_failed");
        }
        return picked;
    },
);

export const applyPitchShift = createAsyncThunk(
    "session/applyPitchShift",
    async (semitones: number) => {
        return webApi.setPitchShift(semitones);
    },
);

export const synthesizeAudio = createAsyncThunk(
    "session/synthesizeAudio",
    async () => {
        return webApi.synthesize();
    },
);

export const exportAudio = createAsyncThunk(
    "session/exportAudio",
    async (outputPath: string) => {
        return webApi.saveSynthesized(outputPath);
    },
);

export const exportSeparated = createAsyncThunk(
    "session/exportSeparated",
    async (outputDir: string) => {
        return webApi.saveSeparated(outputDir);
    },
);

export const pasteVocalShifterClipboard = createAsyncThunk(
    "session/pasteVocalShifterClipboard",
    async (
        arg:
            | {
                  selectionStartFrame?: number;
                  selectionMaxFrames?: number;
                  activeParam?: string;
              }
            | undefined,
        { rejectWithValue },
    ) => {
        let result = await webApi.pasteVocalShifterClipboard(
            arg?.selectionStartFrame,
            arg?.selectionMaxFrames,
            arg?.activeParam,
        );
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "paste_vocalshifter_clipboard_failed");
        }
        result = await resolveMissingFilesInteractively(
            result,
            (result as any)?.missing_files,
        );
        return result;
    },
);

export const pasteReaperClipboard = createAsyncThunk(
    "session/pasteReaperClipboard",
    async (
        arg: { selectionStartFrame?: number; selectionMaxFrames?: number } | undefined,
        { rejectWithValue },
    ) => {
        let result = await webApi.pasteReaperClipboard(
            arg?.selectionStartFrame,
            arg?.selectionMaxFrames,
        );
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "paste_reaper_clipboard_failed");
        }
        result = await resolveMissingFilesInteractively(
            result,
            (result as any)?.missing_files,
        );
        return {
            ok: true,
            timeline: result,
            skippedFiles: result.skipped_files as string[] | undefined,
        } as const;
    },
);
