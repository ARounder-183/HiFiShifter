import { invoke } from "../invoke";
import type { NotebookSettings } from "../../components/layout/notebook/notebookSettings";
import type { SearchSettings } from "../../features/search/searchSettings";
import type { DockPersistedSettings } from "../../features/dock/dockSettings";
import type { TimelineSnapSettings } from "../../features/session/sessionTypes";
import type { VibratoPreset } from "../../features/vibrato/vibratoTypes";

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
    /** 小于该时长的片段不落盘（秒；0 = 不设下限，默认）。 */
    minClipSecs: number;
    /** 小于该大小的条目不落盘（KB；0 = 不限制）。 */
    minEntryKb: number;
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
    /**
     * 导出音频时复用渲染缓存（默认开启）。
     *
     * 关闭后导出总是自行渲染（与引入复用之前一致）；缓存仍照常服务预览与播放。
     */
    exportReuseEnabled: boolean;
    showHitStats: boolean;
}

/** 渲染缓存的出厂默认值（与后端 `config::RenderCacheSettings::default` 对齐）。 */
export const DEFAULT_RENDER_CACHE_SETTINGS: RenderCacheSettings = {
    enabled: true,
    maxSizeMb: 4096,
    maxAgeDays: 90,
    // 0 = 不设时长下限。时长与渲染成本弱相关、与存储成本强相关，用它当准入
    // 闸门会剔除性价比最高的短片段（见后端 `min_clip_secs` 的实测数据）。
    minClipSecs: 0,
    minEntryKb: 4,
    maxEntryMb: 512,
    // 出厂默认「仅保存工程时写入」。渲染产物动辄几百 MB，而"渲染完就写盘"会把每一次
    // 试听 / 微调都变成一次写入 —— 用户当时并没有要求留档。要更早留档的人在设置里改。
    writeMode: "manual",
    location: "system",
    customDir: null,
    verifyChecksum: true,
    minFreeDiskMb: 512,
    showHitStats: true,
    exportReuseEnabled: true,
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
    const minEntryKb = clampInt(
        input.minEntryKb,
        0,
        64 * 1024,
        DEFAULT_RENDER_CACHE_SETTINGS.minEntryKb,
    );
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
        minEntryKb,
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
        // 缺省视为开启（旧配置里没有这个字段）。
        exportReuseEnabled: input.exportReuseEnabled !== false,
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
    /** 搜索匹配设置（转写 / 宽严 / 各语言子开关）。缺省由前端归一化补默认值。 */
    search?: SearchSettings;
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
    /** 指针设备（触控板 / 数位板 / 触控笔 / 触摸）的输入偏好。 */
    penInput?: PenInputSettings;
    customScalePresets?: Array<{
        id: string;
        name: string;
        notes: number[];
    }>;
    /**
     * 用户自定义颤音预设。
     *
     * 元素类型复用 `features/vibrato` 里的 `VibratoPreset`（而非就地写一份
     * 结构相同的匿名类型）：它同时被生成内核、预设编辑器与切片消费，抄一份
     * 出来必然漂移。名字字段一律 camelCase，与后端
     * `vibrato::VibratoPreset` 的 `rename_all = "camelCase"` 对应。
     */
    vibratoPresets?: VibratoPreset[];
    /** 当前活动颤音预设的 id（系统预设的 `builtin.*` 也合法）。 */
    activeVibratoPresetId?: string;
    /**
     * 「摆放方式」：添加颤音时颤音围绕哪条曲线摆（本机记忆）。
     *
     * 它不是预设的一部分：用户先定摆放方式、再挑预设（见 `BaselineMode`）。
     * 未知取值在读取时回落默认，因此不需要迁移。
     */
    vibratoBaseline?: string;
    /**
     * 被停用的颤音预设 id（系统与用户预设共用一份名单）。
     *
     * 停用只影响本机的工具栏列表与拖拽中的循环切换，因此不进预设文件、只进设置。
     */
    disabledVibratoPresetIds?: string[];
    /**
     * 系统预设的自定义顺序（id 列表）。空 / 缺席 = 出厂顺序。
     *
     * 系统预设在代码里，要允许用户排序就只能把顺序记在设置里；缺项与无效项在读取时
     * 兜底，因此新增出厂预设不需要迁移。
     */
    builtinVibratoPresetOrder?: string[];
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
     * 默认 1e-2（1%，约 -40 dBFS）：有损编码——尤其 mp3 joint stereo 的 M/S
     * 量化残留——解码后左右本就带着这个量级的差异，容差过严会漏判大量"内容
     * 其实一致"的素材。判定是"任一样本超出即真立体声"，所以偏松只会折叠
     * **几乎就是单声道**的素材（差异小于 -40 dB 本就听不出声像），不会把真
     * 立体声折错。
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
    tolerance: 1e-2,
    monoTargetMode: 2,
};

/**
 * 容差上限（百分比）。`tolerance` 是"允许的逐样本最大绝对差"，以满幅为 1；
 * 后端把它钳到 `[0, 1]`，因此百分比上限为 100。
 */
export const TOLERANCE_PERCENT_MAX = 100;

/**
 * 容差 → 百分比（界面展示单位）。
 *
 * 百分比是容差最自然的读法：1% 即满幅的百分之一（默认，≈ -40 dBFS），
 * `0` 表示逐样本完全相等。界面上让用户直接填百分比，避免在 `1e-3`
 * 这种科学计数法里数零。
 *
 * 展示值取 6 位小数：足以表达 [0, 100]% 内任何有意义的精度，同时消掉
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
        windowSec: clampNumber(input.windowSec, 0.05, 5, DEFAULT_CHANNEL_IMPORT_POLICY.windowSec),
        windowCount: Math.min(
            256,
            Math.max(
                0,
                Number.isFinite(input.windowCount)
                    ? Math.round(input.windowCount)
                    : DEFAULT_CHANNEL_IMPORT_POLICY.windowCount,
            ),
        ),
        tolerance: clampNumber(input.tolerance, 0, 1, DEFAULT_CHANNEL_IMPORT_POLICY.tolerance),
        monoTargetMode: [2, 3, 4].includes(input.monoTargetMode)
            ? input.monoTargetMode
            : DEFAULT_CHANNEL_IMPORT_POLICY.monoTargetMode,
    };
}

/**
 * 指针设备的显式声明；`auto` = 保持既有启发式判定。
 *
 * 类型本身定义在 `utils/inputProfile.ts`（"有哪些设备"的唯一出处），这里只做转出，
 * 让设置模块的调用方不必多引一个模块。
 */
export type { PointerDeviceDeclaration } from "../../utils/inputProfile";

import type { PointerDeviceDeclaration } from "../../utils/inputProfile";

/** 接触读数的显示时机。 */
export type ContactReadoutMode = "off" | "touchOnly" | "always";

/**
 * 指针设备（触控板 / 数位板 / 触控笔 / 触摸）的输入偏好
 * （持久化到 app_config.json 的 `ui.penInput`）。
 *
 * 背景：手势层原先只按"鼠标 + 键盘修饰键"设计 —— 精细调整必须另一只手按 Ctrl、
 * 次级手势只能靠右键、连续调节只有滚轮。这在触控板（没有侧键 / 中键）、数位笔
 * （另一只手扶着板子）、触屏（根本没有修饰键）上分别是难用、不实用、不存在。
 *
 * 本块集中这些设备的偏好，使它们**可关**：任何一项关掉后，行为退回"和鼠标一样"。
 * 能力判定（有没有压力通道）不在这里，而在 `utils/inputProfile.ts`。
 */
export interface PenInputSettings {
    /**
     * 显式设备声明。
     *
     * 【为什么需要人工声明】Web 平台在这件事上没有可靠信号：`WheelEvent` 不带
     * `pointerType`，触控板在指针层就是 `"mouse"`。`auto` 走既有启发式，行为不变；
     * 显式设定后跳过猜测。项目里已有同类先例（键位预设的 `touchpad`）。
     */
    device: PointerDeviceDeclaration;
    /** 压感是否作为连续的强度通道（拖拽位移倍率 / 画笔权重）。 */
    pressureEnabled: boolean;
    /** 死区：低于此压力视为"未用力"。 */
    pressureDeadZone: number;
    /** 物理下界。 */
    pressureFloor: number;
    /** 物理上界（自动标定只向上抬，见 `utils/pressureCurve.ts`）。 */
    pressureCeiling: number;
    /** 输出倍率下界。 */
    pressureMinGain: number;
    /** 输出倍率上界。 */
    pressureMaxGain: number;
    /** 响应指数：> 1 让轻压段更细腻。 */
    pressureGamma: number;
    /**
     * 倾斜是否参与映射。
     *
     * 【为什么默认关】倾斜是三维输入里最不可靠的一轴：大量设备不报，且握笔姿势
     * 一变值就漂。留白比绑一个会漂移的语义安全。
     */
    tiltEnabled: boolean;
    /** 触控板捏合是否接管缩放（否则捏合被全局吞掉、什么也不做）。 */
    trackpadPinchZoom: boolean;
    /** 触摸的"前置精细斜坡"（前 12px 按 0.35 倍走）。 */
    touchPrecisionRamp: boolean;
    /** 接触读数浮标：关闭 / 仅触摸 / 总是。 */
    contactReadout: ContactReadoutMode;
}

/** 指针设备偏好的出厂默认值（与后端 `config::PenInputSettings::default` 对齐）。 */
export const DEFAULT_PEN_INPUT_SETTINGS: PenInputSettings = {
    device: "auto",
    // 默认开：无压感设备（鼠标 / 触摸）会自动退化，因此开着不会误伤。
    pressureEnabled: true,
    pressureDeadZone: 0.06,
    pressureFloor: 0.05,
    pressureCeiling: 0.9,
    pressureMinGain: 0.25,
    pressureMaxGain: 1.6,
    pressureGamma: 1.6,
    tiltEnabled: false,
    trackpadPinchZoom: true,
    touchPrecisionRamp: true,
    contactReadout: "touchOnly",
};

/**
 * 规范化指针设备偏好（钳制越界值、回退非法枚举），保存前调用。
 * 与后端 `PenInputSettings::normalized` 保持同口径。
 */
export function normalizePenInputSettings(input: PenInputSettings): PenInputSettings {
    const clampNumber = (value: number, min: number, max: number, fallback: number) => {
        if (!Number.isFinite(value)) return fallback;
        return Math.min(max, Math.max(min, value));
    };
    const device: PointerDeviceDeclaration = (
        ["auto", "mouse", "trackpad", "pen", "touch"] as const
    ).includes(input.device)
        ? input.device
        : DEFAULT_PEN_INPUT_SETTINGS.device;
    const contactReadout: ContactReadoutMode = (
        ["off", "touchOnly", "always"] as const
    ).includes(input.contactReadout)
        ? input.contactReadout
        : DEFAULT_PEN_INPUT_SETTINGS.contactReadout;
    // 死区与上界必须留出可用的跨度，否则映射会退化成一条水平线。
    const deadZone = clampNumber(
        input.pressureDeadZone,
        0,
        0.5,
        DEFAULT_PEN_INPUT_SETTINGS.pressureDeadZone,
    );
    const ceiling = clampNumber(
        input.pressureCeiling,
        deadZone + 0.1,
        4,
        DEFAULT_PEN_INPUT_SETTINGS.pressureCeiling,
    );
    const minGain = clampNumber(
        input.pressureMinGain,
        0.02,
        4,
        DEFAULT_PEN_INPUT_SETTINGS.pressureMinGain,
    );
    const maxGain = clampNumber(
        input.pressureMaxGain,
        minGain,
        8,
        DEFAULT_PEN_INPUT_SETTINGS.pressureMaxGain,
    );
    return {
        device,
        pressureEnabled: Boolean(input.pressureEnabled),
        pressureDeadZone: deadZone,
        pressureFloor: clampNumber(
            input.pressureFloor,
            0,
            deadZone,
            DEFAULT_PEN_INPUT_SETTINGS.pressureFloor,
        ),
        pressureCeiling: ceiling,
        pressureMinGain: minGain,
        pressureMaxGain: maxGain,
        pressureGamma: clampNumber(
            input.pressureGamma,
            0.2,
            4,
            DEFAULT_PEN_INPUT_SETTINGS.pressureGamma,
        ),
        tiltEnabled: Boolean(input.tiltEnabled),
        trackpadPinchZoom: Boolean(input.trackpadPinchZoom),
        touchPrecisionRamp: Boolean(input.touchPrecisionRamp),
        contactReadout,
    };
}

/**
 * 保存串行链：把每笔 `save_ui_settings` 排在上一笔完成之后。
 *
 * 后端对 `save_ui_settings` 做"读整个配置文件 → 合并本次字段 → 写回"，且调用
 * 之间没有锁。前端到处都是 fire-and-forget 的**部分**保存（一个开关一次调用），
 * 若两笔并发，后一笔的读取可能发生在前一笔写入之前，前一笔的字段就会被
 * 覆盖丢失。串行化后每笔合并都能看到前一笔的落盘结果（调用方保持
 * fire-and-forget 语义不变，仅提交顺序被保留为 dispatch 顺序）。
 */
let saveChain: Promise<unknown> = Promise.resolve();

export const settingsApi = {
    getUiSettings: () => invoke<UiSettings>("get_ui_settings"),
    saveUiSettings: (settings: Partial<UiSettings>) => {
        const run = saveChain.then(() => invoke<{ ok: boolean }>("save_ui_settings", { settings }));
        // 失败不能断链：这一笔照常向调用方抛错，但队列本身继续消化后续保存。
        saveChain = run.catch(() => undefined);
        return run;
    },
};
