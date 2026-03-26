import { invoke } from "../invoke";

export interface UiSettings {
    autoCrossfade: boolean;
    gridSnap: boolean;
    gridSize?: string;
    pitchSnap: boolean;
    pitchSnapUnit: string;
    pitchSnapScale?: string;
    pitchSnapToleranceCents?: number;
    scaleHighlightMode?: string;
    playheadZoom: boolean;
    autoScroll: boolean;
    paramEditorSeekPlayhead?: boolean;
    showClipboardPreview: boolean;
    showParamValuePopup?: boolean;
    lockParamLines?: boolean;
    selectDragDirection?: string;
    drawDragDirection?: string;
    lineVibratoDragDirection?: string;
    smoothnessPercent?: number;
    customScalePresets?: Array<{
        id: string;
        name: string;
        notes: number[];
    }>;
}

export const settingsApi = {
    getUiSettings: () => invoke<UiSettings>("get_ui_settings"),
    saveUiSettings: (settings: UiSettings) =>
        invoke<{ ok: boolean }>("save_ui_settings", { settings }),
};
