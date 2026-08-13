import { invoke } from "../invoke";

export type StretchAlgorithmOption = "linear" | "signalsmith" | "soundtouch";

export interface UiSettings {
    autoCrossfade: boolean;
    splitTransitionEnabled?: boolean;
    splitTransitionMode?: "fade" | "overlap";
    splitTransitionDurationUnit?: "seconds" | "percent";
    splitTransitionDurationSec?: number;
    splitTransitionDurationPercent?: number;
    splitTransitionCurve?: string;
    splitTransitionOverlapCrossfade?: "auto" | "always";
    gridSnap: boolean;
    gridSize?: string;
    primaryTimeUnit?: string;
    secondaryTimeUnit?: string;
    rulerLabelSpacingPx?: number;
    showPlayheadTimeInTrackHeader?: boolean;
    paramEditorSyncTimeline?: boolean;
    paramEditorTimelineClickSelectTrack?: boolean;
    pitchSnap: boolean;
    pitchSnapUnit: string;
    pitchSnapScale?: string;
    pitchSnapToleranceCents?: number;
    scaleHighlightMode?: string;
    ignoreGrouping?: boolean;
    playheadZoom: boolean;
    autoScroll: boolean;
    paramEditorSeekPlayhead?: boolean;
    showClipboardPreview: boolean;
    showParamValuePopup?: boolean;
    lockParamLines?: boolean;
    quickSearchAutoNormalize?: boolean;
    visibleReferenceRootTrackIds?: string[];
    defaultStretchAlgorithm?: StretchAlgorithmOption;
    defaultHifiganMelStretch?: boolean;
    selectDragDirection?: string;
    drawDragDirection?: string;
    lineVibratoDragDirection?: string;
    smoothnessPercent?: number;
    midiImportPosition?: string;
    midiFillGaps?: boolean;
    midiMultiTrackMerge?: boolean;
    midiImportBpmAsProject?: boolean;
    midiNoteBpmMode?: string;
    midiSpecifiedBpm?: number;
    midiCloseLeadingGap?: boolean;
    midiImportTargetMenu?: string;
    midiImportTargetDragDrop?: string;
    midiImportTargetReaperClipboard?: string;
    midiImportTargetParamEditor?: string;
    ortEp?: string;
    gpuDeviceId?: number;
    ortDeviceId?: number | null;
    autoBackgroundRender?: boolean;
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
