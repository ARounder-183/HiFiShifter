import { invoke } from "../invoke";
import type { NotebookSettings } from "../../components/layout/notebook/notebookSettings";
import type { DockPersistedSettings } from "../../features/dock/dockSettings";
import type { TimelineSnapSettings } from "../../features/session/sessionTypes";

export type StretchAlgorithmOption = "linear" | "signalsmith" | "soundtouch";

/** 渲染缓存写入模式：即时异步 / 退出时批量 / 仅保存工程时。 */
export type RenderCacheWriteMode = "immediate" | "onExit" | "manual";
/** 渲染缓存位置：应用缓存目录 / 自定义目录。 */
export type RenderCacheLocation = "system" | "custom";

/**
 * 渲染缓存设置（持久化到 app_config.json 的 `ui.renderCache`）。
 *
 * 渲染缓存把整 Clip 的合成结果按内容哈希落盘，使"重新打开工程"不再重新
 * 合成未变更的片段。关闭总开关后行为与旧版本完全一致。
 */
export interface RenderCacheSettings {
    /** 总开关（默认开启）。 */
    enabled: boolean;
    /** 磁盘占用上限（MB；0 = 不限制）。 */
    maxSizeMb: number;
    /** 超过 N 天未写入自动清理（0 = 不限龄）。 */
    maxAgeDays: number;
    /** 小于该时长的片段不落盘（秒）。 */
    minClipSecs: number;
    /** 单条缓存上限（MB；0 = 不限制）。 */
    maxEntryMb: number;
    writeMode: RenderCacheWriteMode;
    location: RenderCacheLocation;
    /** 自定义缓存目录（location = "custom" 时生效）。 */
    customDir: string | null;
    /** 读取时校验完整性（默认开启）。 */
    verifyChecksum: boolean;
    /** 可用磁盘空间低于该值（MB）时暂停写入（0 = 不检查）。 */
    minFreeDiskMb: number;
    /** 打开工程后显示命中统计（默认开启）。 */
    showHitStats: boolean;
}

/** 渲染缓存的出厂默认值（与后端 `config::RenderCacheSettings::default` 对齐）。 */
export const DEFAULT_RENDER_CACHE_SETTINGS: RenderCacheSettings = {
    enabled: true,
    maxSizeMb: 4096,
    maxAgeDays: 90,
    minClipSecs: 0.5,
    maxEntryMb: 512,
    writeMode: "immediate",
    location: "system",
    customDir: null,
    verifyChecksum: true,
    minFreeDiskMb: 512,
    showHitStats: true,
};

/** 规范化渲染缓存设置（钳制越界值、回退非法枚举），保存前调用。 */
export function normalizeRenderCacheSettings(input: RenderCacheSettings): RenderCacheSettings {
    const clampInt = (value: number, min: number, max: number, fallback: number) => {
        if (!Number.isFinite(value)) return fallback;
        return Math.min(max, Math.max(min, Math.round(value)));
    };
    const minClipSecs = Number.isFinite(input.minClipSecs)
        ? Math.min(60, Math.max(0, input.minClipSecs))
        : DEFAULT_RENDER_CACHE_SETTINGS.minClipSecs;
    const customDir = (input.customDir ?? "").trim();

    return {
        enabled: Boolean(input.enabled),
        maxSizeMb: clampInt(
            input.maxSizeMb,
            0,
            1024 * 1024,
            DEFAULT_RENDER_CACHE_SETTINGS.maxSizeMb,
        ),
        maxAgeDays: clampInt(input.maxAgeDays, 0, 3650, DEFAULT_RENDER_CACHE_SETTINGS.maxAgeDays),
        minClipSecs,
        maxEntryMb: clampInt(
            input.maxEntryMb,
            0,
            64 * 1024,
            DEFAULT_RENDER_CACHE_SETTINGS.maxEntryMb,
        ),
        writeMode: (["immediate", "onExit", "manual"] as const).includes(input.writeMode)
            ? input.writeMode
            : DEFAULT_RENDER_CACHE_SETTINGS.writeMode,
        location: input.location === "custom" ? "custom" : "system",
        customDir: customDir.length > 0 ? customDir : null,
        verifyChecksum: Boolean(input.verifyChecksum),
        minFreeDiskMb: clampInt(
            input.minFreeDiskMb,
            0,
            1024 * 1024,
            DEFAULT_RENDER_CACHE_SETTINGS.minFreeDiskMb,
        ),
        showHitStats: Boolean(input.showHitStats),
    };
}

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
    /**
     * 纵轴标尺的展示单位（参数 id → `"ratio"` | `"db"`）。
     *
     * 只对音量 / 动态有效（`1× = 0 dB`）：同一个线性幅值既能读成倍率也能读成 dB。
     * 缺项 / 未知键由前端 `normalizeParamAxisUnits` 过滤（见 sessionSlice）。
     */
    paramAxisUnits?: Record<string, string>;
    lockParamLines?: boolean;
    metronomeEnabled?: boolean;
    metronomeGain?: number;
    metronomeMode?: string;
    metronomeAccent?: boolean;
    metronomeSound?: string;
    silenceDetectOptions?: {
        method?: string;
        thresholdDb?: number;
        adaptive?: boolean;
        minSilenceMs?: number;
        minSoundMs?: number;
        paddingMs?: number;
        cutFadeMs?: number;
        action?: string;
        deleteSilentClips?: boolean;
        syncAllTakes?: boolean;
    };
    quickSearchAutoNormalize?: boolean;
    /**
     * **新建工程**默认是否保存 UNDO 操作记录数据（默认开启）。
     *
     * 保存时是否写出由工程级开关（`project.saveUndoHistory`）决定；打开工程
     * 时总是尝试读取伴生文件。
     */
    saveUndoHistoryByDefault?: boolean;
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
    /** 渲染缓存：把渲染结果落盘，重新打开工程时直接复用。 */
    renderCache?: RenderCacheSettings;
    /**
     * 记事本（Notebook）设置。可缺省 —— 旧配置文件没有这一项，
     * 前端用 `normalizeNotebookSettings` 补默认值（见
     * `components/layout/notebook/notebookSettings.ts`）。
     */
    notebook?: NotebookSettings;
    /**
     * 停靠窗体系统设置与布局。可缺省 —— 旧配置文件没有这一项，前端用
     * `normalizeDockSettings` / `normalizeDockLayout` 补默认值（见
     * `features/dock/dockSettings.ts` 与 `features/dock/dockSchema.ts`）。
     *
     * 行为选项与整份布局同处一个对象：后端 `save_ui_settings` 是"读-改-写整个
     * 配置文件"，一次写入同时落盘两者可以少一轮文件往返，也少一个并发窗口。
     */
    dock?: DockPersistedSettings;
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