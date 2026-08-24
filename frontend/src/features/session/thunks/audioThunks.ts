import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import type { AdvancedExportRequest } from "../../../services/api/core";
import type { SessionState } from "../sessionSlice";
import { computePasteEndSec, type PasteEndClipLike } from "../pastePlayhead";

/** 粘贴产生 Clip 后：光标跳到所有新 Clip 最靠右的结束位置并同步后端 transport。 */
async function syncPastePlayheadToEnd(
    clips: PasteEndClipLike[] | undefined,
    newClipIds: string[],
): Promise<number | null> {
    if (newClipIds.length === 0) return null;
    const pasteEndSec = computePasteEndSec(clips, newClipIds);
    if (pasteEndSec !== null) {
        try {
            await webApi.setTransport({ playheadSec: pasteEndSec });
        } catch {
            // transport 同步失败不应让粘贴本身报错。
        }
    }
    // 视图聚焦（若新光标在画面外则水平滚动）由 reducer 记录的
    // pendingPlayheadRevealSec 驱动，在状态与 DOM 提交后执行。
    return pasteEndSec;
}
export const processAudio = createAsyncThunk("session/processAudio", async (audioPath: string) => {
    return webApi.processAudio(audioPath);
});

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

export const synthesizeAudio = createAsyncThunk("session/synthesizeAudio", async () => {
    return webApi.synthesize();
});

export const exportAudio = createAsyncThunk("session/exportAudio", async (outputPath: string) => {
    return webApi.saveSynthesized(outputPath);
});

export const exportSeparated = createAsyncThunk(
    "session/exportSeparated",
    async (outputDir: string) => {
        return webApi.saveSeparated(outputDir);
    },
);

export const exportAudioAdvanced = createAsyncThunk(
    "session/exportAudioAdvanced",
    async (request: AdvancedExportRequest) => {
        return webApi.exportAudioAdvanced(request);
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
        { rejectWithValue, getState },
    ) => {
        const result = await webApi.pasteVocalShifterClipboard(
            arg?.selectionStartFrame,
            arg?.selectionMaxFrames,
            arg?.activeParam,
        );
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "paste_vocalshifter_clipboard_failed");
        }
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const clips = (result as { clips?: Array<{ id?: string }> }).clips ?? [];
        const newClipIds = clips
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        const pasteEndSec = await syncPastePlayheadToEnd(
            clips as PasteEndClipLike[],
            newClipIds,
        );
        return { ...result, newClipIds, pasteEndSec };
    },
);

export const pasteReaperClipboard = createAsyncThunk(
    "session/pasteReaperClipboard",
    async (
        arg: { selectionStartFrame?: number; selectionMaxFrames?: number } | undefined,
        { rejectWithValue, getState },
    ) => {
        const result = await webApi.pasteReaperClipboard(
            arg?.selectionStartFrame,
            arg?.selectionMaxFrames,
        );
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "paste_reaper_clipboard_failed");
        }
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const clips = (result as { clips?: Array<{ id?: string }> }).clips ?? [];
        const newClipIds = clips
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        const pasteEndSec = await syncPastePlayheadToEnd(
            clips as PasteEndClipLike[],
            newClipIds,
        );
        return {
            ok: true,
            timeline: result,
            skippedFiles: result.skipped_files as string[] | undefined,
            newClipIds,
            pasteEndSec,
        } as const;
    },
);
