import type { TimelineResult } from "../../types/api";

import { invoke } from "../invoke";

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
            base_scale?: string;
            use_custom_scale?: boolean;
            custom_scale?: {
                id: string;
                name: string;
                notes: number[];
            } | null;
            beats_per_bar?: number;
            grid_size?: string;
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

    setProjectTimelineSettings: (beatsPerBar: number, gridSize: string) =>
        invoke<{
            ok: boolean;
            project?: { beats_per_bar?: number; grid_size?: string; dirty?: boolean };
        }>("set_project_timeline_settings", beatsPerBar, gridSize),

    newProject: () => invoke<TimelineResult>("new_project"),

    openProjectDialog: () =>
        invoke<{ ok: boolean; canceled?: boolean; path?: string }>("open_project_dialog"),

    openProject: (projectPath: string) => invoke<TimelineResult>("open_project", projectPath),

    saveProject: () => invoke<any>("save_project"),

    saveProjectAs: () => invoke<any>("save_project_as"),

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
