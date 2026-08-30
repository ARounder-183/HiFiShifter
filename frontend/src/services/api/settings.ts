import { invoke } from "../invoke";
import type { TimelineSnapSettings } from "../../features/session/sessionTypes";

export type StretchAlgorithmOption = "linear" | "signalsmith" | "soundtouch";

export interface UiSettings {
    autoCrossfade: boolean;
    /** 空间足够时显示 Clip 内全部 Take 波形。 */
    showAllTakes?: boolean;
    splitTransitionEnabled?: boolean;
    splitTransitionMode?: "fade" | "overlap";
    splitTransitionDurationUnit?: "seconds" | "percent";
    splitTransitionDurationSec?: number;
    splitTransitionDurationPercent?: number;
    splitTransitionCurve?: string;
    splitTransitionOverlapCrossfade?: "auto" | "always";
    snapEnabled: boolean;
    /** 旧版吸附开关字段名（读取兼容）。 */
    gridSnap?: boolean;
    gridSize?: string;
    timelineSnap?: TimelineSnapSettings;
    /** Tempo Map 标尺行可见性（默认开启）。 */
    tempoMapVisible?: boolean;
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
    /** 波纹编辑（自动跟进）模式：off / track / all（对应 REAPER Ripple Editing）。 */
    rippleMode?: "off" | "track" | "all";
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
    /** 旧版边缘平滑字段名（读取兼容）。 */
    edgeSmoothnessPercent?: number;
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
    midiImportAsTempoMap?: boolean;
    midiImportTempoMapTempo?: boolean;
    midiImportTempoMapTimeSignature?: boolean;
    midiImportTempoMapKeySignature?: boolean;
    ortEp?: string;
    gpuDeviceId?: number;
    ortDeviceId?: number | null;
    autoBackgroundRender?: boolean;
    /** 自动重新加载已修改的媒体文件（默认开启）。 */
    autoReloadModifiedMedia?: boolean;
    /** 为新的音频块启用循环（Loop / 循环源，默认开启；仅影响新建 Clip）。 */
    loopNewClips?: boolean;
    /** 同步编辑所有 Take：内容级编辑同步到同一 Clip 的全部 Take。 */
    syncEditsAcrossTakes?: boolean;
    customScalePresets?: Array<{
        id: string;
        name: string;
        notes: number[];
    }>;
}

export const settingsApi = {
    getUiSettings: () => invoke<UiSettings>("get_ui_settings"),
    saveUiSettings: (settings: Partial<UiSettings>) =>
        invoke<{ ok: boolean }>("save_ui_settings", { settings }),
};
