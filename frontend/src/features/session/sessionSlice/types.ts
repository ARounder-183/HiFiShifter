// Session 状态类型。拆自 sessionSlice，供 reducer / helper / thunk 共享。
import type { HistoryRecordSummary, HostAudioPayload } from "../../../types/api";
import type {
    ClipInfo,
    ClipFormantAnalysisState,
    ClipTemplate,
    DrawToolMode,
    DragDirection,
    DrawDragDirection,
    EditParam,
    GridSize,
    PitchSnapUnit,
    SplitTransitionCurveType,
    TimelineSnapSettings,
    TimeUnit,
    TimeUnitChoice,
    SilenceDetectSettings,
    TrackMeterInfo,
    ToolMode,
    ToolModeGroup,
    TrackInfo,
} from "../sessionTypes";
import type { SearchSettings } from "../../search/searchSettings";
import type { FileBrowserViewOptions } from "../../fileBrowser/fileBrowserViewOptions";
import type { FolderImportOptions } from "../../fileBrowser/folderImportOptions";
import type {
    ChannelImportPolicy,
    PenInputSettings,
    RenderCacheSettings,
} from "../../../services/api/settings";
import type { ParamAxisUnits } from "../../../components/layout/pianoRoll/paramAxisUnits";
import type { CustomScalePreset } from "../../../utils/customScales";
import type { BaselineMode, VibratoPreset } from "../../vibrato/vibratoTypes";

export type { FadeCurveType } from "../sessionTypes";
export type {
    ClipInfo,
    ClipTemplate,
    DrawToolMode,
    DragDirection,
    DrawDragDirection,
    EditParam,
    GridSize,
    ToolMode,
    ToolModeGroup,
    TrackInfo,
};

export type ClipColor = ClipInfo["color"];
export type StretchAlgorithmOption = "linear" | "signalsmith" | "soundtouch";
type ClipFormantToolWindowState = {
    open: boolean;
    clipId: string | null;
    x: number;
    y: number;
    hasMoved: boolean;
};

/**
 * 参数编辑器选区快照（**帧**单位：`[startFrame, frameCount]`）。
 *
 * 与选区的内部单位、后端 `get/set_param_frames` 的契约同形（见 `paramSelection.ts`）。
 * 帧栅格是工程级常量，因此这份快照与 BPM 无关。
 */
export type ParamSelectionSnapshot = Array<{ startFrame: number; frameCount: number }>;

export interface SessionState {
    toolMode: ToolMode;
    toolModeGroup: ToolModeGroup;
    drawToolMode: DrawToolMode;
    editParam: EditParam;
    bpm: number;
    beats: number;
    projectSec: number;
    grid: GridSize;
    primaryTimeUnit: TimeUnit;
    secondaryTimeUnit: TimeUnitChoice;
    rulerLabelSpacingPx: number;
    showPlayheadTimeInTrackHeader: boolean;
    paramEditorSyncTimeline: boolean;

    autoCrossfadeEnabled: boolean;
    /** 空间足够时显示 Clip 内全部 Take 波形。 */
    showAllTakes: boolean;
    /** 同步编辑所有 Take：内容级编辑同步到同一 Clip 的全部 Take。 */
    syncEditsAcrossTakes: boolean;
    /** 为新的音频块启用循环（Loop / 循环源，默认开启；仅影响新建 Clip）。 */
    loopNewClipsEnabled: boolean;
    /** 分割过渡 */
    splitTransitionEnabled: boolean;
    splitTransitionMode: "fade" | "overlap";
    splitTransitionDurationUnit: "seconds" | "percent";
    splitTransitionDurationSec: number;
    splitTransitionDurationPercent: number;
    /** 淡化曲线（新版 REAPER 预设；"keep" = 分割后保留原 Clip 曲线，默认）。 */
    splitTransitionCurve: SplitTransitionCurveType;
    splitTransitionOverlapCrossfade: "auto" | "always";
    /** 吸附总开关（兼容旧字段 gridSnapEnabled）。 */
    snapEnabled: boolean;
    /** 完整的时间轴吸附/网格设置（snapEnabled 仅作为旧字段镜像 enabled）。 */
    timelineSnap: TimelineSnapSettings;
    /**
     * Tempo Map 数据（null = 无 Tempo Map，使用工程全局 BPM/拍号/音阶）。
     * 变化点按秒锚定；0 位置点始终存在。
     */
    tempoMap: import("../../../utils/tempoMap").TempoMap | null;
    /** Tempo Map 标尺行可见性（视图菜单开关，默认开启）。 */
    tempoMapVisible: boolean;
    /** 音高吸附 */
    pitchSnapEnabled: boolean;
    pitchSnapUnit: PitchSnapUnit;
    /** 音高吸附容差（分）用于微调吸附强度 */
    pitchSnapToleranceCents: number;
    /** 基准音阶键名，如 "C" "Db" 等 */
    pitchSnapScale: import("../../../utils/musicalScales").ScaleKey;
    /** 音阶高亮模式：始终 / 关闭 */
    scaleHighlightMode: "always" | "off";
    /** 播放头缩放 */
    playheadZoomEnabled: boolean;
    /** 自动滚动（播放时跟随播放头） */
    autoScrollEnabled: boolean;
    /** 忽略编组（启用后编组同步编辑操作不生效） */
    ignoreGrouping: boolean;
    /**
     * 波纹编辑（自动跟进）模式：
     * - `"off"`：关闭（默认）。
     * - `"track"`：仅被编辑的轨道上的后续剪辑一起跟进。
     * - `"all"`：所有轨道上位于编辑点之后的剪辑一起跟进。
     */
    rippleMode: "off" | "track" | "all";
    /** 被禁用的编组 ID 列表（禁用后该编组内同步编辑不生效） */
    disabledGroupIds: string[];
    /** 允许参数编辑器点击时调整播放头 */
    paramEditorSeekPlayheadEnabled: boolean;
    /** 允许时间轴上的点击操作自动切换当前轨道 */
    paramEditorTimelineClickSelectTrackEnabled: boolean;
    /** 剪贴板预览（在参数编辑器选区内显示剪贴板曲线预览） */
    showClipboardPreview: boolean;
    /** 参数线附近显示参数值浮窗 */
    showParamValuePopup: boolean;
    /**
     * 纵轴标尺的展示单位（参数 id → `"ratio"` | `"db"`）。
     *
     * 只对音量 / 动态这类**线性幅值倍率**参数有意义（`1× = 0 dB`）：同一个值既能
     * 读成倍率也能读成 dB，用户需要按习惯切换读法。缺项 = 倍率（默认）。
     * 换算与解析见 `components/layout/pianoRoll/paramAxisUnits`。
     */
    paramAxisUnits: ParamAxisUnits;
    /** 参数编辑器（选择工具）拖动方向限制 */
    selectDragDirection: DragDirection;
    /** 参数编辑器（绘制工具）拖动方向限制 */
    drawDragDirection: DrawDragDirection;
    /** 参数编辑器（直线工具与颤音工具共用）拖动方向限制 */
    lineVibratoDragDirection: DrawDragDirection;

    /** 参数编辑器选区拖拽时的边缘平滑度（0-100%） */
    edgeSmoothnessPercent: number;

    /** 在粘贴/创建时是否锁定参数线以应用 linked params */
    lockParamLinesEnabled: boolean;

    /** 节拍器：是否启用（跟随网格标尺发出节拍器声音） */
    metronomeEnabled: boolean;
    /** 节拍器音量 0..1 */
    metronomeGain: number;
    /** 节拍器细分模式：grid（跟随网格标尺）/ beat（仅每拍）/ bar（仅小节首） */
    metronomeMode: "grid" | "beat" | "bar";
    /** 节拍器音色（程序化合成预设） */
    metronomeSound: "click" | "woodblock" | "beep";
    /** 节拍器是否强调小节首（重音音色） */
    metronomeAccent: boolean;

    /** 静音检测预览：clipId → 检测到的静音区间（时间线秒），由设置对话框管理生命周期 */
    silencePreviewSegments: Record<string, Array<[number, number]>> | null;
    /** 最近一次静音预览请求 id（乱序守卫） */
    _silencePreviewRequestId: string | null;
    /** 静音检测对话框的上次使用参数（持久化到 UiSettings） */
    silenceDetectOptions: SilenceDetectSettings;
    /** 搜索匹配设置（转写 / 宽严 / 各语言子开关，持久化到 UiSettings.search） */
    searchSettings: SearchSettings;
    /**
     * 文件浏览器视图选项（排序 / 密度 / 隐藏文件 / 详情列 …，持久化到
     * UiSettings.fileBrowser）。
     *
     * 【为什么在 session 而不在 fileBrowser 切片】它随 `app_config.json` 一起备份
     * 与迁移，与 `searchSettings` 同类；放进 fileBrowser 切片会与
     * `persistUiSettings` 的大 payload 分家，保存路径变成两条。
     */
    fileBrowserView: FileBrowserViewOptions;
    /**
     * 目录导入的选项（排布方式 / 递归 / 建轨道组）。
     *
     * 【为什么放在这里而不是 fileBrowser 切片】与 `fileBrowserView` 同一条理由：
     * 它随 `app_config.json` 一起备份与迁移，保存路径也只有一条。
     *
     * 【为什么"记住上次的选择"是必要的】它是**对话框的预填**：目录导入一律弹窗
     * （见 `FolderImportHost`），每次都从上次的选择起手，用户多数时候只需按回车。
     */
    folderImportOptions: FolderImportOptions;
    /**
     * 文件浏览器的常用位置（用户固定的目录）。
     *
     * 【为什么与视图选项分开存】前者是"视图怎么画"（受白名单约束的枚举），
     * 这是"常去哪里"（任意路径）。变更原因与校验规则都不同，混在一个对象里会让
     * `normalizeFileBrowserViewOptions` 同时管两件不相干的事。
     */
    fileBrowserFavorites: string[];
    /**
     * 「搜索与匹配设置」对话框是否打开。
     *
     * 【为什么放在切片里】三个地方都要能把它打开：选项菜单、文件浏览器与快速搜索
     * 各自的「文A」右键菜单。各自持一份 `useState` 就会出现三个窗口实例 ——
     * 对话框是单例，开关状态也只能有一份。
     */
    searchSettingsDialogOpen: boolean;
    /** 快速搜索放置音频时自动规格化 */
    quickSearchAutoNormalizeEnabled: boolean;
    /**
     * **新建工程**默认是否保存 UNDO 操作记录数据（全局设置的初值，默认关闭）。
     *
     * 保存时是否写出 UNDO 由工程级开关（`project.saveUndoHistory`）决定；
     * 打开工程时后端总是尝试读取伴生文件（静默失败即无历史）。
     */
    saveUndoHistoryByDefault: boolean;
    /** PianoRoll 中显示的其他 root track 参考线 */
    visibleReferenceRootTrackIds: string[];
    /** 全局默认外部拉伸算法 */
    defaultStretchAlgorithm: StretchAlgorithmOption;
    /** 全局默认 HiFiGAN mel stretch 开关 */
    defaultHifiganMelStretch: boolean;
    ortEp: string;
    ortDeviceKey: string | null;
    /**
     * vslib 是否可用（启动时经 `get_vslib_status` 探测）。
     *
     * `null` = 尚未取到后端状态 —— 算法列表按"不可用"处理（隐藏 vslib），
     * 详见 `features/tracks/pitchAlgoOptions.ts` 的说明。
     */
    vslibAvailable: boolean | null;
    /** 后台预渲染：编辑后立即在后台渲染，无需等待播放触发 */
    autoBackgroundRender: boolean;
    /** 渲染缓存设置：把合成结果落盘，重新打开工程时直接复用。 */
    renderCache: RenderCacheSettings;
    /** 导入媒体时的声道处理策略（假立体声 → 单声道）。 */
    channelImportPolicy: ChannelImportPolicy;
    /** 指针设备（触控板 / 数位板 / 触控笔 / 触摸）的输入偏好。 */
    penInput: PenInputSettings;
    /**
     * 在途的导入任务数（>0 = 正在导入）。
     *
     * 只用于状态栏的**延迟**等待提示：导入变快之后绝大多数情况都不会亮起，
     * 它存在的意义是慢盘 / 网络盘 / 超大文件时不至于让界面看起来毫无反应。
     * 因此这里只记"有没有"，不记进度 —— 导入本身没有可汇报的粒度。
     */
    importInFlight: number;
    /**
     * 后端"播放/预渲染"状态镜像（`playback_rendering_state` 事件）——只镜像
     * 播放轮询 reducer 需要的 active/target 两个原始值字段（变化低频，且
     * 订阅全量 session 的组件以 shallowEqual 比较，原始值不会因高频 progress
     * 事件破坏相等性）；进度百分比由 App 本地状态展示。
     *
     * target="original"（阻塞式前台预渲染）active 期间，引擎尚未真正进入
     * playing，`get_playback_state` 返回的是上一场播放/seek 的陈旧传输态 ——
     * 播放轮询 reducer 据此拒采（见 syncPlaybackState.fulfilled），否则渲染
     * 完成后迟到的"未播放"采样会把光标拽回播放起点并掐灭轮询。
     */
    playbackRenderingActive: boolean;
    playbackRenderingTarget: string | null;
    /**
     * 阻塞式前台预渲染（target="original"）的独立活跃镜像。
     *
     * 与 playbackRenderingActive（任意渲染，驱动状态栏）分离的原因：前台与
     * 后台渲染线程会并发发射 playback_rendering_state 事件，单一镜像会被
     * 互相覆盖 —— 后台渲染的 active=false 完成事件会提前关闭前台预渲染的
     * 拒采窗口，让渲染期间派发的陈旧轮询响应溜进 reducer（光标跳变）。
     * 由 App.tsx 的事件监听按 target 分 ref 跟踪后一并派发。
     */
    playbackBlockingRenderActive: boolean;

    // Monotonic bump token for invalidating parameter curve caches.
    // - Not included in undo/redo snapshots.
    // - Should be bumped on any timeline/undo/redo operation that may affect param rendering.
    paramsEpoch: number;
    /** 单调递增，任何 clip 的 playbackRate 变更时 +1，强制 UI 刷新 */
    playbackRateVersion: number;

    playheadSec: number;
    /**
     * playheadSec 最近一次由**本地写入**（拖拽 / seek / 粘贴）设置时的
     * `performance.now()`；0 表示从未。
     *
     * 【为什么需要它】拖拽播放头时前端连续写 playheadSec，而 4~5 Hz 的
     * `get_playback_state` 轮询同时在读**宿主**的编辑光标 —— 那个光标要等
     * `set_transport` 回写之后才会跟上。于是拖拽期间每一个轮询都会把光标
     * 拽回旧位置，下一帧拖拽又把它拉回鼠标处：用户看到的就是"跳回去再跳回来"。
     * 本地写入之后的一小段窗口内，停止态的轮询不再覆盖光标。
     */
    playheadLocalAtMs: number;
    /**
     * playheadSec 最近一次由**播放轮询**写入时的 performance.now()（毫秒，
     * 与 RAF/视觉插值同一时钟源；0 表示从未由轮询写入）。
     *
     * 轮询载荷的 base+position 是"后端执行命令那一刻"的引擎位置；reducer
     * 会把它按"派发→处理"的往返时延外推到处理时刻（真实可听位置）再写入，
     * 并同时记录该时刻。useVisualPlayhead 据此把视觉插值锚定在采样时刻：
     * 采样值与锚定时刻严格配对，IPC 往返时延及其抖动不再造成视觉光标的
     * 系统性滞后与回跳（后台预渲染负载下时延可达数十～数百毫秒）。
     * seek/stop/粘贴等非轮询写入不更新该时间戳 —— 陈旧的时间戳让
     * useVisualPlayhead 走硬复位分支，不会被外推放大。
     */
    playheadSampledAtMs: number;
    /**
     * 待执行的“聚焦播放光标”请求（粘贴后跳转光标时设置）。
     *
     * 粘贴可能显著扩充工程全长（水平可滚动范围随之扩大）。若在 fulfilled
     * reducer 提交之前就尝试滚动，dynamicProjectSec / DOM 内容宽度都还是
     * 旧值，滚动会被旧上限钳制，导致光标无法进入画面。因此这里只记录
     * 目标位置，由 TimelinePanel 的 useLayoutEffect 在新状态与 DOM 均提交
     * 后再执行滚动并回写 null。
     */
    pendingPlayheadRevealSec: number | null;
    tracks: TrackInfo[];
    trackMeters: Record<string, TrackMeterInfo>;
    clips: ClipInfo[];
    selectedTrackId: string | null;
    selectedClipId: string | null;
    /** 多选 clip 的 id 列表（框选 / 主修饰键 + 点击） */
    multiSelectedClipIds: string[];
    /**
     * 多选集合是否来自**用户显式选择**。
     *
     * 【为什么需要它】集合有两种来源，用户意图完全不同：
     * - `true`（框选 / 主修饰键点击 / Shift 范围选择 / 全选）="我要操作这几个"，
     *   拖动其中任一成员时**整组一起移动**是正确行为；
     * - `false`（多文件导入 / 打开工程 / 粘贴 / 分割 / 复制拖拽）=只是"把新产生的
     *   东西设为当前选中"，**不表示**想整组拖动。
     *
     * 时间轴内核据此决定拖拽起手是否收敛选区：不区分来源时，导入后同轨相邻的
     * 多个 clip 都在集合里，用户随手抓一个拖，右邻就跟着走——这正是
     * 「拖一个 clip，右边的 clip 也跟着动」的根因（多选描边仅 2px，看不出来）。
     */
    multiSelectionIntentional: boolean;
    /** 参数编辑器当前是否存在参数线选区（PianoRollPanel selectionUi 同步；
     *  复制/剪切路由依据之一，见 focusRouting.resolveCopyCutRoute） */
    paramSelectionActive: boolean;
    /** 最近被触碰的选区上下文：轨道选中变化（换轨 = 参数编辑器重定向）与
     *  参数线选区变化记 "param"，Clip 选中变化记 "clips"。两者并存时由它
     *  仲裁复制/剪切的归属。 */
    selectionContext: "param" | "clips" | null;
    clipPitchRanges: Record<string, { min: number; max: number }>;

    /**
     * 后端推送的 per-clip 音高检测结果（MIDI 曲线）。
     * key: clip_id
     * value: { curveStartSec, midiCurve, framePeriodMs }
     */
    clipPitchCurves: Record<
        string,
        {
            /** MIDI 曲线第 0 帧对应的 timeline 绝对时间（秒） */
            curveStartSec: number;
            midiCurve: number[];
            framePeriodMs: number;
        }
    >;
    clipFormantStatus: Record<string, "ready" | "rebuilding" | "failed">;
    /** Clip 源共振峰分析结果（共振峰工具窗口可视化用） */
    clipFormantAnalysis: Record<string, ClipFormantAnalysisState>;
    clipFormantToolWindow: ClipFormantToolWindowState;
    modelDir: string;
    audioPath: string;
    outputPath: string;
    pitchShift: number;
    playbackClipId: string | null;
    playbackAnchorSec: number;

    runtime: {
        device: string;
        modelLoaded: boolean;
        audioLoaded: boolean;
        hasSynthesized: boolean;
        isPlaying: boolean;
        /** "传输层原地等待渲染"：isPlaying=true 但引擎位置冻结（后台渲染
         * 进行中遇到未渲染窗口——起播位置落在其中，或播放途中推进到其
         * 开头——保持播放态静音等待，就绪后自动开始/继续播放）。轮询据此
         * 跳过外推、冻结播放光标；视觉层的 isTransportAdvancing 代理经由
         * 播放启动时的 positionSec 归零自然为 false，无需额外接线。 */
        playbackWaitingForRender: boolean;
        playbackTarget: string | null;
        playbackPositionSec: number;
        playbackDurationSec: number;
        gpuBackend: string;
    };

    selectedTrackSummary: {
        trackId: string | null;
        clipCount: number;
        waveformPreview: number[];
        pitchRange: { min: number; max: number };
    };

    customScalePresets: CustomScalePreset[];

    /**
     * 用户自定义颤音预设。系统预设在代码里（`features/vibrato/systemPresets.ts`），
     * 不在这份列表里 —— 两者合成"可用预设列表"的逻辑见
     * `features/vibrato/vibratoPresetList.ts`。
     */
    vibratoPresets: VibratoPreset[];
    /** 当前活动颤音预设的 id（系统预设的 `builtin.*` 也合法）。 */
    activeVibratoPresetId: string;
    /**
     * 被停用的颤音预设 id（系统与用户预设共用一份名单）。
     *
     * 【为什么单独存一份名单，而不是给预设加 `enabled` 字段】系统预设在代码里、
     * 用户预设在设置里，两者都要能被停用；而预设对象还会被导出成预设文件，往里塞
     * 一个"本机是否启用"的字段会让文件携带与它无关的本地状态。停用只影响本机的
     * 工具栏列表与循环切换，因此单独记一份 id 名单最干净。
     */
    disabledVibratoPresetIds: string[];
    /**
     * 「摆放方式」：添加颤音时，颤音围绕哪条曲线摆。
     *
     * 【为什么在设置里，而不是预设字段】它回答的是"这一次把颤音挂到哪条线上"，
     * 与"颤音长什么样"无关：用户先定摆放方式、再挑预设，换预设不该把它换掉；
     * 预设库、拖拽工具、预设文件也都用不到它（见 `BaselineMode` 的说明）。
     * 于是它作为**本机记忆**存在设置里，下次打开「添加颤音」还是上次选的那个。
     */
    vibratoBaseline: BaselineMode;
    /**
     * 系统预设的自定义顺序（id 列表）。
     *
     * 【为什么单独存】系统预设在代码里（`systemPresets.ts` 的出厂顺序），要允许用户
     * 排序就只能把顺序记在设置里。空数组 = 用出厂顺序；缺项（新版本新增的出厂预设）
     * 与无效项（旧版本删过的）由 `effectiveBuiltinPresetOrder` 兜底，因此不需要迁移。
     */
    builtinVibratoPresetOrder: string[];

    project: {
        name: string;
        path: string | null;
        dirty: boolean;
        recent: string[];
        notesMarkdown: string;
        baseScale: import("../../../utils/musicalScales").ScaleKey;
        useCustomScale: boolean;
        customScale: CustomScalePreset | null;
        beatsPerBar: number;
        /** 工程基准拍号分母（1/2/4/8/16/32）。 */
        timeSignatureDenominator: number;
        gridSize: GridSize;
        stretchAlgorithmOverride: StretchAlgorithmOption | null;
        hifiganMelStretchOverride: boolean | null;
        /**
         * 保存本工程时是否一并写出 UNDO 操作记录数据（工程级开关，随工程
         * 文件持久化）。打开工程时总是尝试读取 UNDO 数据，与本开关无关。
         */
        saveUndoHistory: boolean;
    };

    busy: boolean;
    status: string;
    /**
     * 状态行的**补充说明**（后端原文），渲染为状态文字的 ToolTip。
     *
     * 【为什么需要它】"音频块编辑被拒绝"这句话本身零信息：到底是宿主还没追上、
     * 是宿主拒绝了某个字段，还是校验失败？用户报障时只能截一句无信息的话，
     * 开发者只能靠日志猜。原文放进 ToolTip 之后，截图本身就能定位。
     * 只在下一条状态覆盖 `status` 时一起清掉，避免旧原因挂在新的状态上。
     */
    statusDetail: string | null;
    error?: string;
    lastResult?: unknown;
    vocalShifterSkippedFilesDialog: string[] | null;
    reaperSkippedFilesDialog: string[] | null;

    /**
     * 保存/另存为目标位置已存在版本不一致的工程文件时，弹出的覆盖确认对话框。
     * 为 null 表示没有待确认的覆盖。
     */
    saveVersionConflictDialog: {
        path: string;
        existingVersion: number;
        currentVersion: number;
        existingIsNewer: boolean;
    } | null;

    /**
     * 交互锁计数器：当用户正在进行连续操作（拖动、滑动等）时 > 0。
     * 在锁定期间，连续操作类 thunk 的 fulfilled handler 将跳过 applyTimelineState()，
     * 避免后端返回的过期快照覆盖前端乐观更新导致的闪烁。
     */
    _interactionLockCount: number;
    /** 插件自动刷新只接纳同一交互代次的响应，不能覆盖在途宿主写入。 */
    _pluginTimelineEpoch: number;
    _pluginTimelineFetchEpochs: Record<string, number>;
    _pluginClipEditRequests: Record<string, boolean>;

    /**
     * 最近一次 undo/redo 请求的 requestId。快速连续撤销/重做会产生多个
     * in-flight thunk；fulfilled/rejected 到达时与该字段比对，丢弃过期
     * 响应，防止旧快照以 force 覆盖新状态（与 seekPlayhead 的乱序防护同理）。
     */
    _latestHistoryOpRequestId: string | null;

    /**
     * 撤销栈深度镜像（后端权威：`history_state` 事件 / 撤销·重做响应同步）。
     * 0 = 无可撤销 —— 菜单项置灰、快捷键静默失败都以此为准。
     */
    historyUndoDepth: number;

    /** 重做栈深度镜像。0 = 无可重做。 */
    historyRedoDepth: number;

    /**
     * 「操作记录」条目（后端权威：`history_state` 事件同步）。
     *
     * 下标即历史位置：`records[historyUndoDepth]` 是当前状态；其后的条目是
     * 可重做的部分；`records[0]` 是初始状态行。
     */
    historyRecords: HistoryRecordSummary[];

    /**
     * 待参数编辑器消费的选区恢复请求（撤销/重做命中拉伸步骤时登记）。
     * requestId 单调递增，消费方按 id 幂等应用一次即可（无需回写清除）。
     */
    pendingParamSelectionRestore: {
        requestId: number;
        selection: ParamSelectionSnapshot | null;
    } | null;

    /** 选区恢复请求序号（单调递增）。 */
    _paramSelectionRestoreSeq: number;

    /**
     * 宿主音频读数（插件模式；`null` = 未知或不适用）。
     *
     * 【为什么必须进 Redux 而不是就地轮询】它是**时间线内容**的解释：同一批片段
     * 有没有音频，取决于宿主有没有把 region 交给插件。放在会话状态里，`HostAudioNotice`
     * 与片段渲染都读同一份读数，不会出现"提示条说有音频、片段却没有"的错位。
     *
     * 独立 App 从不产生该字段，因此缺省 `null` 即"不显示任何宿主音频提示"。
     */
    hostAudio: HostAudioPayload | null;

    /**
     * 最近一次**编辑类**请求的 requestId（setClipState / bulk / move 族）。
     * 编辑 thunk 的 fulfilled 会 force-apply 后端全量快照；多选批量等场景
     * 若旧请求的响应迟到（或后端返回中间状态），会覆盖更新的乐观状态、
     * 造成"闪回原状 / 部分 Clip 还原"。pending 记录、fulfilled 比对，
     * 过期响应直接丢弃 —— 最新一次编辑的响应（含此前所有已提交变更）
     * 总是最后一个生效，撤/重做等权威操作不受影响。
     */
    _latestEditRequestId: string | null;

    /**
     * 传输层纪元（单调递增）。播放头所有权在 播放/暂停/停止/seek 之间
     * 交接的瞬间递增；30Hz 播放轮询在派发时携带当时的纪元，fulfilled 时
     * 与当前纪元比对，不一致的响应（跨传输操作的迟到/乱序快照）直接丢弃。
     *
     * 背景：后端预渲染（ONNX 推理）会令 IPC 往返延迟至数十～数百毫秒，
     * 停止→播放的间隙采样（引擎已归零 base/position）或停止前的播放采样
     * 迟到后按旧语义应用，会把 isPlaying 复活/掐灭并把光标改写成 0 或旧
     * 播放位置 —— 即反复播放/暂停/停止时播放光标跳变的主要根源。与
     * `_latestHistoryOpRequestId` / seekPlayhead 的乱序防护同理。
     */
    _transportEpoch: number;
    /** 插件seek写入未结束时不接纳同代但仍在旧位置的宿主轮询。 */
    _pluginSeekRequestId: string | null;

    /**
     * 最近一次 stop/pause 命令**开始瞬间**的真实播放状态（由
     * `stopAudioPlayback.pending` 在乐观置 false 之前捕获）。
     * `stopAudioPlayback` thunk 据此判定 wasPlaying（Stop 是否恢复锚点），
     * 避免任何基于 playhead/anchor 差值的误判恢复（跳变）。
     */
    _stopInterruptedPlayback: boolean;
}
