import type { TimelineResult } from "../../types/api";
import type { StretchAlgorithmOption } from "./settings";

import { invoke } from "../invoke";

export interface AutoBackupSettings {
    saveOnSaveEnabled: boolean;
    timedBackupEnabled: boolean;
    timedBackupIntervalSec: number;
    timedBackupPathTemplate: string;
}

export const projectApi = {
    consumeStartupProjectPath: () =>
        invoke<{ ok: boolean; path?: string | null }>("consume_startup_project_path"),

    // Project meta
    getProjectMeta: () =>
        invoke<{
            name: string;
            path?: string | null;
            dirty: boolean;
            recent: string[];
            notes_markdown?: string;
            base_scale?: string;
            use_custom_scale?: boolean;
            custom_scale?: {
                id: string;
                name: string;
                notes: number[];
            } | null;
            beats_per_bar?: number;
            grid_size?: string;
            stretch_algorithm_override?: StretchAlgorithmOption | null;
            hifigan_mel_stretch_override?: boolean | null;
        }>("get_project_meta"),

    setProjectBaseScale: (baseScale: string) =>
        invoke<{
            ok: boolean;
            project?: {
                base_scale?: string;
                use_custom_scale?: boolean;
                custom_scale?: {
                    id: string;
                    name: string;
                    notes: number[];
                } | null;
            };
        }>("set_project_base_scale", baseScale),

    setProjectCustomScale: (customScale: { id: string; name: string; notes: number[] }) =>
        invoke<{ ok: boolean; project?: { custom_scale?: unknown; use_custom_scale?: boolean } }>(
            "set_project_custom_scale",
            customScale,
        ),

    setProjectTimelineSettings: (
        beatsPerBar: number,
        timeSignatureDenominator: number,
        gridSize: string,
    ) =>
        invoke<{
            ok: boolean;
            project?: {
                beats_per_bar?: number;
                time_signature_denominator?: number;
                grid_size?: string;
                dirty?: boolean;
            };
        }>("set_project_timeline_settings", beatsPerBar, timeSignatureDenominator, gridSize),

    setProjectStretchSettings: (payload: {
        stretchAlgorithmOverride?: StretchAlgorithmOption | null;
        hifiganMelStretchOverride?: boolean | null;
    }) =>
        invoke<{
            ok: boolean;
            project?: {
                stretch_algorithm_override?: StretchAlgorithmOption | null;
                hifigan_mel_stretch_override?: boolean | null;
                dirty?: boolean;
            };
        }>(
            "set_project_stretch_settings",
            payload.stretchAlgorithmOverride ?? null,
            payload.hifiganMelStretchOverride ?? null,
        ),

    newProject: () => invoke<TimelineResult>("new_project"),

    openProjectDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("open_project_dialog"),

    openProject: (projectPath: string) => invoke<TimelineResult>("open_project", projectPath),

    saveProject: (notesMarkdown?: string) => invoke<any>("save_project", notesMarkdown),

    saveProjectAs: (notesMarkdown?: string) => invoke<any>("save_project_as", notesMarkdown),

    getAutoBackupSettings: () => invoke<AutoBackupSettings>("get_auto_backup_settings"),

    saveAutoBackupSettings: (settings: AutoBackupSettings) =>
        invoke<{ ok: boolean; settings?: AutoBackupSettings }>(
            "save_auto_backup_settings",
            settings,
        ),

    runTimedAutoBackup: (pathTemplate: string) =>
        invoke<{ ok: boolean; path?: string; formatFallbackApplied?: boolean; error?: string }>(
            "run_timed_auto_backup",
            pathTemplate,
        ),

    openVocalShifterDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("open_vocalshifter_dialog"),

    importVocalShifterProject: (vspPath: string) =>
        invoke<TimelineResult & { error?: string; skipped_files?: string[] }>(
            "import_vocalshifter_project",
            vspPath,
        ),

    openReaperDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("open_reaper_dialog"),

    importReaperProject: (rppPath: string) =>
        invoke<TimelineResult & { error?: string; skipped_files?: string[] }>(
            "import_reaper_project",
            rppPath,
        ),
};
