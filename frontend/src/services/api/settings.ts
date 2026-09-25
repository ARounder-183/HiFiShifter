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
    /** 导入媒体时的声道处理策略（假立体声 → 单声道）。 */
    channelImportPolicy?: ChannelImportPolicy;
    customScalePresets?: Array<{
        id: string;
        name: string;
        notes: number[];
    }>;
}

/** 导入声道处理策略的总模式。 */
export type ChannelImportMode = "smart" | "alwaysMono" | "off";

/**
 * 导入媒体时的声道处理策略（持久化到 app_config.json 的 `ui.channelImportPolicy`）。
 *
 * 背景：真立体声源在渲染时会把整条处理器链按声道跑两遍，耗时翻倍。大量
 * "人力"素材实际是单声道内容被混流成双声道（两声道逐样本相同），折叠为
 * 单声道后听感不变、耗时减半。
 *
 * 作用范围仅限"无声道权威来源"的新建 Take：媒体导入、VocalShifter 导入、
 * 以及 v4 及更早工程的 Take 升级。REAPER 导入导出自带 CHANMODE，不受影响。
 */
export interface ChannelImportPolicy {
    /** `smart`（智能判定，默认）/ `alwaysMono` / `off`。 */
    mode: ChannelImportMode;
    /** 抽样窗口时长（秒）。 */
    windowSec: number;
    /** 抽样窗口数（0 = 不限，扫描整个消费区间）。 */
    windowCount: number;
    /**
     * 逐样本绝对差容差（覆盖有损编码的量化噪声）。
     *
     * 默认 1e-3（约 -60 dBFS）：有损编码的左右差异常在这个量级。判定是
     * "任一样本超出即真立体声"，所以偏松只会折叠**几乎就是单声道**的素材
     *（差异小于 -60 dB 本就听不出声像），不会把真立体声折错。
     */
    tolerance: number;
    /** 转换目标模式：2 = 混合为单声道（默认）/ 3 = 仅左 / 4 = 仅右。 */
    monoTargetMode: number;
}

/** 导入声道策略的出厂默认值（与后端 `config::ChannelImportPolicy::default` 对齐）。 */
export const DEFAULT_CHANNEL_IMPORT_POLICY: ChannelImportPolicy = {
    mode: "smart",
    windowSec: 0.25,
    windowCount: 12,
    tolerance: 1e-3,
    monoTargetMode: 2,
};

/**
 * 容差上限（百分比）。`tolerance` 是"允许的逐样本最大绝对差"，以满幅为 1；
 * 后端把它钳到 `[0, 0.1]`，因此百分比上限为 10。
 */
export const TOLERANCE_PERCENT_MAX = 10;

/**
 * 容差 → 百分比（界面展示单位）。
 *
 * 百分比是容差最自然的读法：0.1% 即满幅的千分之一（≈ -60 dBFS），
 * `0` 表示逐样本完全相等。界面上让用户直接填百分比，避免在 `1e-3`
 * 这种科学计数法里数零。
 *
 * 展示值取 6 位小数：足以表达 [0, 10]% 内任何有意义的精度，同时消掉
 * `1e-6 × 100 = 0.00009999999999999999` 这类二进制表示残渣。
 */
export function toleranceToPercent(tolerance: number): number {
    const percent = (Number.isFinite(tolerance) ? tolerance : 0) * 100;
    return Math.round(percent * 1e6) / 1e6;
}

/** 百分比 → 容差（写回策略）。越界值由 `normalizeChannelImportPolicy` 收口。 */
export function percentToTolerance(percent: number): number {
    return (Number.isFinite(percent) ? percent : 0) / 100;
}

/**
 * 规范化导入声道策略（钳制越界值、回退非法枚举），保存前调用。
 * 与后端 `ChannelImportPolicy::normalized` 保持同口径。
 */
export function normalizeChannelImportPolicy(input: ChannelImportPolicy): ChannelImportPolicy {
    const clampNumber = (value: number, min: number, max: number, fallback: number) => {
        if (!Number.isFinite(value)) return fallback;
        return Math.min(max, Math.max(min, value));
    };
    const mode: ChannelImportMode = (["smart", "alwaysMono", "off"] as const).includes(input.mode)
        ? input.mode
        : DEFAULT_CHANNEL_IMPORT_POLICY.mode;
    return {
        mode,
        windowSec: clampNumber(
            input.windowSec,
            0.05,
            5,
            DEFAULT_CHANNEL_IMPORT_POLICY.windowSec,
        ),
        windowCount: Math.min(
            256,
            Math.max(
                0,
                Number.isFinite(input.windowCount)
                    ? Math.round(input.windowCount)
                    : DEFAULT_CHANNEL_IMPORT_POLICY.windowCount,
            ),
        ),
        tolerance: clampNumber(
            input.tolerance,
            0,
            0.1,
            DEFAULT_CHANNEL_IMPORT_POLICY.tolerance,
        ),
        monoTargetMode: [2, 3, 4].includes(input.monoTargetMode)
            ? input.monoTargetMode
            : DEFAULT_CHANNEL_IMPORT_POLICY.monoTargetMode,
    };
}

export const settingsApi = {
    getUiSettings: () => invoke<UiSettings>("get_ui_settings"),
    saveUiSettings: (settings: Partial<UiSettings>) =>
        invoke<{ ok: boolean }>("save_ui_settings", { settings }),
};
