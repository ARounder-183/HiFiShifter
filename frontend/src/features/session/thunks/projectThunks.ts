import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import type { SessionState } from "../sessionSlice";

export const undoRemote = createAsyncThunk("session/undoRemote", async () => {
    return webApi.undoTimeline();
});

export const redoRemote = createAsyncThunk("session/redoRemote", async () => {
    return webApi.redoTimeline();
});

export const newProjectRemote = createAsyncThunk("session/newProjectRemote", async () => {
    return webApi.newProject();
});

export const openProjectFromDialog = createAsyncThunk(
    "session/openProjectFromDialog",
    async (_, { rejectWithValue }) => {
        const picked = await webApi.openProjectDialog();
        if (!picked.ok) return rejectWithValue("open_project_dialog_failed");
        if (picked.canceled || !picked.path) {
            return { ok: true, canceled: true } as const;
        }
        const timeline = await webApi.openProject(picked.path);
        return { ok: true, canceled: false, timeline } as const;
    },
);

export const openProjectFromPath = createAsyncThunk(
    "session/openProjectFromPath",
    async (projectPath: string) => {
        const timeline = await webApi.openProject(projectPath);
        return timeline;
    },
);

export const saveProjectRemote = createAsyncThunk(
    "session/saveProjectRemote",
    async (_, { rejectWithValue, getState }) => {
        const state = getState() as any;
        const hasPath = Boolean(state?.session?.project?.path);
        const notesMarkdown = String(state?.session?.project?.notesMarkdown ?? "");

        const res = hasPath
            ? await webApi.saveProject(notesMarkdown)
            : await webApi.saveProjectAs(notesMarkdown);
        if (!res || res.ok === false) {
            return rejectWithValue(res?.error ?? "save_project_failed");
        }
        return res as any;
    },
);

export const saveProjectAsRemote = createAsyncThunk(
    "session/saveProjectAsRemote",
    async (_, { rejectWithValue, getState }) => {
        const state = getState() as any;
        const notesMarkdown = String(state?.session?.project?.notesMarkdown ?? "");
        const res = await webApi.saveProjectAs(notesMarkdown);
        if (!res || res.ok === false) {
            return rejectWithValue(res?.error ?? "save_project_as_failed");
        }
        return res as any;
    },
);

export const setProjectBaseScaleRemote = createAsyncThunk(
    "session/setProjectBaseScaleRemote",
    async (baseScale: string, { rejectWithValue }) => {
        const res = await webApi.setProjectBaseScale(baseScale);
        if (!res || res.ok === false) {
            return rejectWithValue("set_project_base_scale_failed");
        }
        return res;
    },
);

export const setProjectCustomScaleRemote = createAsyncThunk(
    "session/setProjectCustomScaleRemote",
    async (customScale: { id: string; name: string; notes: number[] }, { rejectWithValue }) => {
        const res = await webApi.setProjectCustomScale(customScale);
        if (!res || res.ok === false) {
            return rejectWithValue("set_project_custom_scale_failed");
        }
        return res;
    },
);

export const setProjectTimelineSettingsRemote = createAsyncThunk(
    "session/setProjectTimelineSettingsRemote",
    async (
        payload: { beatsPerBar: number; timeSignatureDenominator?: number; gridSize: string },
        { rejectWithValue },
    ) => {
        const res = await webApi.setProjectTimelineSettings(
            payload.beatsPerBar,
            payload.timeSignatureDenominator ?? 4,
            payload.gridSize,
        );
        if (!res || res.ok === false) {
            return rejectWithValue("set_project_timeline_settings_failed");
        }
        return res;
    },
);

export const setProjectStretchSettingsRemote = createAsyncThunk(
    "session/setProjectStretchSettingsRemote",
    async (
        payload: {
            stretchAlgorithmOverride?: "linear" | "signalsmith" | "soundtouch" | null;
            hifiganMelStretchOverride?: boolean | null;
        },
        { rejectWithValue },
    ) => {
        const res = await webApi.setProjectStretchSettings(payload);
        if (!res || res.ok === false) {
            return rejectWithValue("set_project_stretch_settings_failed");
        }
        return res;
    },
);

export const openVocalShifterFromDialog = createAsyncThunk(
    "session/openVocalShifterFromDialog",
    async (_, { rejectWithValue, getState }) => {
        const picked = await webApi.openVocalShifterDialog();
        if (!picked.ok) return rejectWithValue("open_vocalshifter_dialog_failed");
        if (picked.canceled || !picked.path) {
            return { ok: true, canceled: true } as const;
        }
        const result = await webApi.importVocalShifterProject(picked.path);
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "import_vocalshifter_failed");
        }
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const clips = (result as { clips?: Array<{ id?: string }> }).clips ?? [];
        const newClipIds = clips
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        return {
            ok: true,
            canceled: false,
            timeline: result,
            skippedFiles: result.skipped_files as string[] | undefined,
            newClipIds,
        } as const;
    },
);

export const openVocalShifterFromPath = createAsyncThunk(
    "session/openVocalShifterFromPath",
    async (vspPath: string, { rejectWithValue, getState }) => {
        const result = await webApi.importVocalShifterProject(vspPath);
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "import_vocalshifter_failed");
        }
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const clips = (result as { clips?: Array<{ id?: string }> }).clips ?? [];
        const newClipIds = clips
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        return {
            ok: true,
            canceled: false,
            timeline: result,
            skippedFiles: result.skipped_files as string[] | undefined,
            newClipIds,
        } as const;
    },
);

export const pickProjectToImport = createAsyncThunk(
    "session/pickProjectToImport",
    async (_, { rejectWithValue }) => {
        const picked = await webApi.importProjectDialog();
        if (!picked.ok) return rejectWithValue("import_project_dialog_failed");
        if (picked.canceled || !picked.path) {
            return { ok: true, canceled: true } as const;
        }
        return { ok: true, canceled: false, path: picked.path } as const;
    },
);

export const importProjectFromPath = createAsyncThunk(
    "session/importProjectFromPath",
    async (
        payload: {
            projectPath: string;
            placeAtPlayhead?: boolean;
            importTempoMap?: boolean;
        },
        { rejectWithValue, getState },
    ) => {
        const result = await webApi.importProject(payload);
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "import_project_failed");
        }
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const newClipIds = ((result as { clips?: Array<{ id?: string }> }).clips ?? [])
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        return {
            ok: true,
            canceled: false,
            timeline: result,
            newClipIds,
            sourceProject: (result as { sourceProject?: string }).sourceProject,
            importedTrackCount: (result as { importedTrackCount?: number }).importedTrackCount,
            importedClipCount: (result as { importedClipCount?: number }).importedClipCount,
            tempoMapImported: (result as { tempoMapImported?: boolean }).tempoMapImported,
            tempoMapSkipped: (result as { tempoMapSkipped?: boolean }).tempoMapSkipped,
        } as const;
    },
);

export const openReaperFromDialog = createAsyncThunk(
    "session/openReaperFromDialog",
    async (_, { rejectWithValue, getState }) => {
        const picked = await webApi.openReaperDialog();
        if (!picked.ok) return rejectWithValue("open_reaper_dialog_failed");
        if (picked.canceled || !picked.path) {
            return { ok: true, canceled: true } as const;
        }
        const result = await webApi.importReaperProject(picked.path);
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "import_reaper_failed");
        }
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const clips = (result as { clips?: Array<{ id?: string }> }).clips ?? [];
        const newClipIds = clips
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        return {
            ok: true,
            canceled: false,
            timeline: result,
            skippedFiles: result.skipped_files as string[] | undefined,
            newClipIds,
        } as const;
    },
);

export const openReaperFromPath = createAsyncThunk(
    "session/openReaperFromPath",
    async (rppPath: string, { rejectWithValue, getState }) => {
        const result = await webApi.importReaperProject(rppPath);
        if (!result?.ok) {
            return rejectWithValue(result?.error ?? "import_reaper_failed");
        }
        const beforeClipIds = new Set(
            (getState() as { session: SessionState }).session.clips.map((c) => c.id),
        );
        const clips = (result as { clips?: Array<{ id?: string }> }).clips ?? [];
        const newClipIds = clips
            .map((c) => c.id)
            .filter((id): id is string => !!id && !beforeClipIds.has(id));
        return {
            ok: true,
            canceled: false,
            timeline: result,
            skippedFiles: result.skipped_files as string[] | undefined,
            newClipIds,
        } as const;
    },
);
