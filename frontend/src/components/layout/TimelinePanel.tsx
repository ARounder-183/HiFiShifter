/**
 * TimelinePanel — Timeline 面板 UI 组件（精简后）
 *
 * 所有业务逻辑已拆分至 4 个 hook：
 * - useTimelineState        → state / ref / viewport / scroll / 坐标转换
 * - useTimelineDragDrop     → Tauri 原生拖放 + 文件浏览器面板自定义拖拽
 * - useTimelineClipActions  → Clip 多选 + 操作回调
 * - useTimelineEventHandlers→ 全局事件监听
 *
 * 此文件只保留：JSX 渲染 + 胶水 + 拖拽 hooks 桥接
 *
 * 【播放头桥接】`TimelineTransportBridge` 逐帧写标尺播放头（React 声明式元素），
 * 并调用 `kernelHostRef.invalidatePlayhead()` 请求内核重绘**轨道区**播放头——
 * 后者只在宿主 `draw()` 内被写，而内核渲染循环是纯脏标记驱动的，镜像变化不会
 * 自动标脏（不请求就永不重绘）。
 *
 * 【渲染路径】轨道区**恒由** `TimelineKernelView`（WebGL2 渲染内核）承载。
 * 旧实现（原生滚动 + Canvas2D 的 `TimelineScrollArea` / `TimelineSurface` /
 * `TrackLane` / `ClipItem` 等）已随"唯一路径"改造删除，此处**没有运行期二选一、
 * 也没有可回退的第二条路径**。
 *
 * 【失败处理】WebGL2 不可用时内核视图回报失败（`handleKernelUnavailable`），本面板改
 * 渲染 `KernelUnavailableNotice` —— 一个可自助排障的界面（原因 + 排查清单 + 可复制的
 * 诊断信息）。实测 WebGL2 在纯 CPU 环境（含无 GPU）可用（ANGLE + SwiftShader），只有
 * 显式禁止软件光栅一类配置才会失败。
 *
 * 【历史背景（勿删）】内核曾是 opt-in 且默认值跟随 `import.meta.env.DEV`，导致打包后
 * **静默退回旧渲染器**（Phase 3 计划 R8）。结论是"绝不静默降级"：失败必须显式告知。
 *
 * @see docs/superpowers/specs/2026-09-13-timeline-single-path-design.md
 */
import React, { useMemo, Profiler } from "react";
import {
    isHostGeometryReadOnly,
    canEditFadeShape,
    canEditFadeCurvature,
    canImportAsTakes,
} from "../../services/hostCapabilities";
import { Flex } from "@radix-ui/themes";
import { AppDialog } from "../../ui/Dialog";
import { AppContextMenu } from "../../ui/Menu";
import { useMenuShortcut } from "../../ui/useMenuShortcut";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { useAppSelector } from "../../app/hooks";
import { DockGutter } from "../dock/DockGutter";
import { DEFAULT_GUTTER_SIZES, GUTTER_LIMITS } from "../../features/dock/dockSchema";
import { setGutterSize } from "../../features/dock/dockSlice";
import { isModifierActive } from "../../features/keybindings/keybindingsSlice";
import { isClipSplittableAtSec, resolveSplitTargetsAtSec } from "./timeline/splitTargets";
import { normalizedTrackColorCss } from "./timeline/runtime/timelineCanvasStyle";
import { defaultFadeDirFor, FADE_PRESETS } from "./timeline/reaperFade";
import { hostFadeDisplayShape } from "./timeline/hostFadeDisplay";
import { FadeContextMenuHost } from "./timeline/FadeContextMenuHost";
import { createPortal } from "react-dom";
import {
    closeClipFormantToolWindow,
    setClipFormantToolWindowPosition,
    setClipMuted,
    importAudioAtPosition,
    importMultipleAudioAtPosition,
    setClipStateRemote,
    setClipsStateBulkRemote,
    scanAndConvertFakeStereoRemote,
    setClipFades,
    setClipActiveTakeRemote,
    glueClipsRemote,
    convertClipsToPitchReferenceRemote,
    updatePitchReferenceRemote,
    removeClipsRemote,
    closeTrackGapsRemote,
} from "../../features/session/sessionSlice";
import { batch } from "react-redux";
import { NEW_TRACK_SENTINEL } from "./timeline/constants";
import { getBulkEditableClipIds } from "./timeline/hooks/bulkClipEdit";
import { collectFadeContextClips } from "./timeline/clipFadeContext";
import { emitExternalFileAction } from "../../features/session/projectOpenEvents";
import { webApi } from "../../services/webviewApi";
import { useClipPitchDrag } from "./timeline/hooks/useClipPitchDrag";
import { AppTooltipBubble } from "../AppTooltip";
import { formatPitchDragCents } from "./timeline/clipPitchDrag";
import { QuickClipExportDialog } from "./QuickClipExportDialog";
import { SilenceDetectionDialog } from "./timeline/SilenceDetectionDialog";
import { MidiTrackSelectDialog } from "./MidiTrackSelectDialog";
import { ClipContextMenu, TrackAreaContextMenu, TimeRuler, TrackList } from "./timeline";
import { timeRulerHeightPx } from "./timeline/rulerHeight";
import { requestResetFadeCurvature } from "./timeline/fadeContextMenuBus";
import { clearSnapHighlights } from "../../utils/snapHighlight";
import { TimelineKernelView } from "./timeline/kernel/TimelineKernelView";
import { isKernelAvailable } from "./timeline/kernel/kernelAvailability";
import { KernelUnavailableNotice } from "./timeline/kernel/KernelUnavailableNotice";
import type { TimelineKernelHost } from "./timeline/kernel/host/timelineKernelHost";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";
import { resolveTimelineScrollRange } from "./timeline/runtime/timelineScrollRange";
import { useTimelineState } from "./timeline/hooks/useTimelineState";
import { useTimelineKernelSeekDrag } from "./timeline/hooks/useTimelineKernelSeekDrag";
import { useTimelineKernelTrim } from "./timeline/hooks/useTimelineKernelTrim";
import { useTimelineKernelFade } from "./timeline/hooks/useTimelineKernelFade";
import { useTimelineKernelSelection } from "./timeline/hooks/useTimelineKernelSelection";
import { useTimelineKernelGainCrossfade } from "./timeline/hooks/useTimelineKernelGainCrossfade";
import { useTimelineTrackActions } from "./timeline/hooks/useTimelineTrackActions";
import { useTimelineRuler } from "./timeline/hooks/useTimelineRuler";
import { useTimelineMidiActions } from "./timeline/hooks/useTimelineMidiActions";
import { useTimelineFileDrop } from "./timeline/hooks/useTimelineFileDrop";
import { useTimelineViewportController } from "./timeline/hooks/useTimelineViewportController";
import { TimelineTransportBridge } from "./timeline/TimelineTransportBridge";
import type { TimelinePanelProps } from "./timeline/TimelinePanelProps";
import { useTimelineDragDrop } from "./timeline/hooks/useTimelineDragDrop";
import { useTimelineClipActions } from "./timeline/hooks/useTimelineClipActions";
import { useTimelineEventHandlers } from "./timeline/hooks/useTimelineEventHandlers";
import { expandClipIdsWithGroups } from "./timeline/hooks/useGroupExpansion";
import { useDebouncedPersist } from "../../hooks/useDebouncedPersist";
import { ClipRateEditorDialog } from "./timeline/ClipRateEditorDialog";
import { resolveQuickExportClipIds } from "./timeline/quickExportSelection";
import { type ClipFormantMorph } from "../../features/session/sessionTypes";
import { ClipFormantToolWindow } from "./timeline/clip/ClipFormantToolWindow";
import { timelineViewportBus } from "../../utils/timelineViewportBus";

export const TimelinePanel: React.FC<TimelinePanelProps> = ({
    midiClipDialogOpen,
    midiClipPath,
    midiClipStartSec,
    midiClipTrackId,
    fillGaps,
    multiTrackMerge,
    importBpmAsProject,
    noteBpmMode,
    specifiedBpm,
    onMidiClipDialogOpenChange,
    onMidiClipPathChange,
    onMidiClipStartSecChange,
    onMidiClipTrackIdChange,
    onFillGapsChange,
    onMultiTrackMergeChange,
    onImportBpmAsProjectChange,
    onNoteBpmModeChange,
    onSpecifiedBpmChange,
    midiClipClipboardGuid,
    importPosition,
    onImportPositionChange,
    closeLeadingGap,
    onCloseLeadingGapChange,
    midiDialogSource,
    onMidiDialogSourceChange,
    importTargetMenu,
    onImportTargetMenuChange,
    importTargetDragDrop,
    onImportTargetDragDropChange,
    importTempoMapEnabled,
    onImportTempoMapEnabledChange,
    importTempoMapTempo,
    onImportTempoMapTempoChange,
    importTempoMapTimeSignature,
    onImportTempoMapTimeSignatureChange,
    importTempoMapKeySignature,
    onImportTempoMapKeySignatureChange,
}) => {
    const importTarget = midiDialogSource === "dragDrop" ? importTargetDragDrop : importTargetMenu;
    const onImportTargetChange =
        midiDialogSource === "dragDrop" ? onImportTargetDragDropChange : onImportTargetMenuChange;
    const { t, tf, tVars } = useI18n();
    // 轨道头宽度是**布局状态**的一部分（用户调过的尺寸必须随布局持久化），
    // 取代了原先写死的 `w-64`。
    const trackHeaderWidthPx = useAppSelector(
        (state) => state.dock.layout.gutters.timelineTrackHeaderPx,
    );
    const trackHeaderRef = React.useRef<HTMLDivElement | null>(null);
    const ignoreGrouping = useAppSelector((state) => state.session.ignoreGrouping);
    const disabledGroupIds = useAppSelector((state) => state.session.disabledGroupIds);
    /**
     * 静音检测预览区段（内核模式用）。
     *
     * 由静音检测对话框实时写入；内核把它画成半透明红色覆盖层。`TimelineSessionSlice`
     * 不含该字段，因此在这里单独选择（与旧实现 `ClipItem` 的取值方式一致）。
     */
    const silencePreviewSegments = useAppSelector((state) => state.session.silencePreviewSegments);
    // 时间轴 scroller 水平滚动条的占用高度（offsetHeight - clientHeight）。
    // 轨道头底部按此留出同高占位（bottomGutterHeightPx），保证轨道头与
    // 时间轴区域的竖直滚动范围严格一致。
    const [horizontalScrollbarGutterPx, setHorizontalScrollbarGutterPx] = React.useState(0);
    // 【已删除：timelineViewportHeightPx】它只为旧实现的 `TimelineSurface` 的
    // `playheadHeightPx` 供数（把播放光标延伸到时间轴可视区底部）。内核路径的
    // 播放头是内核容器（视口元素）的直接子节点、恒 `top-0 bottom-0`，高度天然
    // 等于可视区高度，无需测量——留着这个 state 只会是永不读取的死状态。
    const [quickExportDialog, setQuickExportDialog] = React.useState<{
        open: boolean;
        clipIds: string[];
    }>({ open: false, clipIds: [] });
    const [silenceDialogIds, setSilenceDialogIds] = React.useState<string[] | null>(null);

    const [replaceMidiDialog, setReplaceMidiDialog] = React.useState<{
        open: boolean;
        clipId: string | null;
        midiPath: string | null;
    }>({ open: false, clipId: null, midiPath: null });
    const [timeDisplaySettingsOpen, setTimeDisplaySettingsOpen] = React.useState(false);

    // 文件浏览器拖入 HiFiShifter 工程（hshp/hsp）时的「打开工程 / 导入工程」菜单。
    const [projectActionMenu, setProjectActionMenu] = React.useState<{
        x: number;
        y: number;
        path: string;
    } | null>(null);
    // 该菜单里「打开工程」的快捷键提示（与菜单栏「文件 → 打开工程」同一个动作）。
    // 取值是原始字符串，因此只有在用户真的改了绑定时才会让本面板重渲染。
    const openProjectShortcut = useMenuShortcut("project.open");

    // ── 1. State / refs / viewport / scroll / 坐标转换 ──────
    /**
     * 内核宿主句柄（仅内核模式下非空）。
     *
     * 用途：把「轨道头滚动」这类外部意图转发给内核（`setScrollTop`），以及让
     * 标尺点击 seek / 拖入落点 / 参数编辑器同步等需要「当前视口」的逻辑在两种
     * 模式下共用一份取值。
     *
     * 特殊说明：必须在 `useTimelineState()` **之前**声明——该 hook 需要它来
     * 落地「参数编辑器视图同步」（内核模式没有原生 scroller 可写）。
     */
    const kernelHostRef = React.useRef<TimelineKernelHost | null>(null);

    /**
     * 内核运行期失败的原因（`null` = 未失败）。
     *
     * 【为什么需要】WebGL2 不可用（老驱动 / 远程桌面 / GPU 黑名单 / 上下文数超限）
     * 是预期内的环境差异。内核是**唯一**渲染路径，没有回退可走，因此必须让用户能
     * 自助排障——失败界面（`KernelUnavailableNotice`）要展示具体原因。
     *
     * 【为什么用 state 而不是 ref】它决定渲染内核还是失败界面，必须触发重渲染。
     *
     * 【为什么置位后不再尝试】失败原因（无 GL）在会话内不会自愈，而每次重渲染都重挂
     * 一次内核会反复失败并刷日志。要恢复只需刷新页面。
     */
    const [kernelUnavailableReason, setKernelUnavailableReason] = React.useState<string | null>(
        null,
    );

    /**
     * `TimelineKernelView` 回报内核不可用：记录原因并告警。
     *
     * 流程：记录原因（触发重渲染，切到失败界面）→ 控制台告警（带原因，供诊断）。
     *
     * 特殊说明 1：**没有回退**。旧实现（原生滚动 + Canvas2D）已随阶段 2/3 删除，
     * 因此这里不会、也无法切回任何第二套渲染实现；唯一出路是失败界面 + 重启。
     *
     * 特殊说明 2：用 `useCallback` 保持引用稳定——它经 props 传给内核视图，而视图把
     * 它放进「回调镜像」ref；引用抖动虽不会重建宿主，但稳定引用更省心。
     *
     * @param reason 失败原因（来自内核创建的 catch）。
     */
    const handleKernelUnavailable = React.useCallback((reason: string) => {
        console.warn(`[TimelinePanel] 时间轴内核不可用（${reason}），当前没有回退渲染路径`);
        setKernelUnavailableReason(reason);
    }, []);

    /**
     * 「clip 左键按下拦截」句柄（内核在启动自己的手势之前调用）。
     *
     * 【为什么用 ref 中转】拦截需要复用旧实现的 `useClipPitchDrag`（它在面板里
     * 于渲染流程的**后段**才实例化），而内核交互对象在**前段**构建。用 ref 中转
     * 既避免把 hook 调用提前（改动面大），也不产生 TDZ。
     *
     * @returns true = 面板已接管，内核不得再启动自己的手势。
     */
    const kernelClipInterceptRef = React.useRef<
        | ((args: {
              clipId: string;
              clientX: number;
              clientY: number;
              pointerId: number;
              modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
              container: HTMLElement;
          }) => boolean)
        | null
    >(null);
    const state = useTimelineState({ kernelHostRef });
    // 视觉插值播放头的共享读取点：bridge 的 onFrame 每帧写入（与绘制同源），
    // 缩放锚点与内核宿主（经 `playheadSec` getter）读取同一值——播放中缩放不得
    // 以 33Hz 轮询的 store 滞后值锚定，否则播放头会跳变 δ·Δpx（δ = 轮询间隔内的
    // 插值领先量）。
    const visualPlayheadSecRef = React.useRef(0);
    const getVisualPlayheadSec = React.useCallback(() => visualPlayheadSecRef.current, []);
    /**
     * 请求时间轴内核重绘播放头。
     *
     * 【为什么走 hostRef 而不是 React state】播放头每帧都可能移动，走 React 会
     * 把 60fps 的更新灌进渲染；这里只做一次命令式标脏。
     *
     * 特殊说明：**不是** `invalidateScene()`——那个会置 `sceneDirty` 并在下一帧
     * 重建全部 GPU 几何，播放中每帧重建等于自毁性能（见宿主接口说明）。
     */
    const handleRequestPlayheadRepaint = React.useCallback(() => {
        kernelHostRef.current?.invalidatePlayhead();
    }, []);
    // 面板卸载时清空吸附竖线高亮（拖拽手势异常中断的兜底）。
    React.useEffect(() => {
        return () => {
            clearSnapHighlights();
        };
    }, []);

    // 播放开始（键盘快捷键 / 播放按钮 / 远端传输皆可触发）＝拖拽视觉语境终止：
    // 按住鼠标拖拽期间开始播放时，吸附竖线高亮必须立即消失，而不是冻结在原地
    // 直到松手才被手势结束逻辑清理。
    const timelineRuntimeIsPlaying = useAppSelector((state) => state.session.runtime.isPlaying);
    React.useEffect(() => {
        if (timelineRuntimeIsPlaying) {
            clearSnapHighlights();
        }
    }, [timelineRuntimeIsPlaying]);
    const {
        dispatch,
        s,
        sessionRef,
        scrollRef,
        trackListScrollRef,
        rulerContentRef,
        rulerPlayheadLineRef,
        rulerPlayheadHeadRef,
        dropPreviewRef,
        lastClickedClipIdRef,
        pxPerSecRef,
        viewportWidthRef,
        rowHeightRef,
        pxPerSec,
        setPxPerSec,
        viewportWidth,
        setViewportWidth,
        rowHeight,
        setRowHeight,
        trackVolumeUi,

        setTrackVolumeUi,
        sameSourceConfirmOpen,
        setSameSourceConfirmOpen,
        sameSourceConfirmResolverRef,
        pxPerBeat,
        contentWidth,
        contentHeight,
        dynamicProjectSec,
        timelineTicks,
        scrollLeft,
        rulerScrollLeft,
        verticalZoomKb,
        paramFineAdjustKb,
        pitchDragKb,
        copyDragKb,
        dropPreview,
        setDropPreview,
        pendingDropDurationPathRef,
        syncScrollLeft,
        setScrollLeftState,
        beatFromClientX,
        rowTopForTrackId,
        ensureDropPreviewDuration,
        getDropPreviewWidthPx,
        snapTimeline,
        setPlayheadFromClientX,
        keyboardZoomPendingRef,
    } = state;

    /**
     * 模式无关的视口访问器。
     *
     * 内核模式下旧滚动容器不挂载（`scrollRef.current === null`），凡是「读/写视口」
     * 的逻辑（拖入落点、自动滚屏、键盘缩放、聚焦光标、参数编辑器同步）都必须经它，
     * 而不是各自判断模式。
     *
     * 特殊说明：引用必须**稳定**——下游 hook 把它作为 effect 依赖，随渲染重建会
     * 导致事件监听反复重注册。因此惰性创建一次并缓存。
     */
    const {
        viewportAccess,
        resolveTrackIdAtClientY,
        handleKernelScrollLeftCommit,
        handleKernelZoomRequest,
        handleRulerWheel,
    } = useTimelineViewportController({
        state,
        s,
        dispatch,
        kernelHostRef,
        setHorizontalScrollbarGutterPx,
        visualPlayheadSecRef,
    });

    const {
        timeContext,
        timeValueFormatCtx,
        projectScale,
        handleTempoMapChange,
        handleTempoMapCommit,
        handlePrimaryUnitChange,
        handleSecondaryUnitChange,
        handleRulerOpenSettings,
        handleRulerCopyPlayheadTime,
        handleRulerMouseDown,
    } = useTimelineRuler({
        state,
        s,
        dispatch,
        kernelHostRef,
        setTimeDisplaySettingsOpen,
    });
    const lastClickedClientXRef = React.useRef<number | null>(null);

    // ── 时间轴缩放与行高的持久化 ─────────────────────────────
    //
    // 旧实现由 `TimelineScrollArea`（原生滚动容器）负责写这两个键；"内核唯一路径"
    // 改造删掉该组件时**没有把写入点接回来**，于是读取侧（`useTimelineState` 初始化
    // 读 `hifishifter.pxPerSec` / `hifishifter.rowHeight`）一直在，但再没有人写
    // ——表现为每次重启都回到默认缩放与行高。这里接回同一对键名与取值语义。
    //
    // 用防抖 hook 而不是直接 `localStorage.setItem`：滚轮缩放是高频事件，同步落盘
    // 会与同帧的渲染 / 重绘挤在一起（见 `useDebouncedPersist` 文件头）。
    useDebouncedPersist("hifishifter.pxPerSec", pxPerSec);
    useDebouncedPersist("hifishifter.rowHeight", rowHeight);

    // ── 2. Clip 多选 + 操作回调 ─────────────────────────────
    const clipActions = useTimelineClipActions({
        sessionRef,
        scrollRef,
        // 模式无关视口：范围选择的「点击位置 → 工程秒」换算需要它（内核模式下
        // 原生 scroller 不存在，缺了它会退化成用 clip 起点近似）。
        viewport: viewportAccess,
        lastClickedClipIdRef,
        lastClickedClientXRef,
        pxPerSec,
        pxPerBeat,
        ignoreGrouping,
        disabledGroupIds,
        dispatch,
        sameSourceConfirmResolverRef,
        setSameSourceConfirmOpen,
        setPlayheadFromClientX,
    });
    const {
        multiSelectedClipIds,
        multiSelectedSet,
        setMultiSelectedClipIds,
        contextMenu,
        setContextMenu,
        trackAreaMenu,
        setTrackAreaMenu,
        importModeMenu,
        setImportModeMenu,
        clipboardAvailable,
        copyClips,
        cutClips,
        groupClips,
        ungroupClips,
        toggleGroupDisabled,
        normalizeClips,
        replaceClipSources,
        splitClipIdsAtPlayhead,
        splitSelectedAtPlayhead,
        pasteClipsAtPlayhead,
        clearContextMenu,
        toggleTrackLaneClipMuted,
        commitTrackLaneRename,
        commitTrackLaneGain,
        commitTrackLaneRate,
    } = clipActions;
    // 右键播放速率角标 → 高级编辑浮层（BPM 换算）的目标 Clip 与锚点。
    const [rateEditorClipId, setRateEditorClipId] = React.useState<string | null>(null);
    const [rateEditorPosition, setRateEditorPosition] = React.useState<{
        x: number;
        y: number;
    } | null>(null);
    const openRateBadgeMenu = React.useCallback(
        (clipId: string, screenX: number, screenY: number) => {
            setRateEditorClipId(clipId);
            setRateEditorPosition({ x: screenX, y: screenY });
        },
        [],
    );

    // 角标行内编辑提交：按字段路由到速率（自动调整时长）/增益提交。
    // 传给 React.memo 化的 TrackList / TrackLane 的回调必须引用稳定，
    // 否则每次 TimelinePanel 渲染（播放头提交值、滚动、修饰键）都会击穿 memo。
    const handleToggleGroupDisabled = React.useCallback(
        (groupId: string) => {
            toggleGroupDisabled(groupId);
        },
        [toggleGroupDisabled],
    );
    const commitTrackLaneFormantMorph = React.useCallback(
        (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => {
            void dispatch(
                setClipStateRemote({
                    clipId,
                    formantMorph: value,
                    checkpoint,
                }),
            );
        },
        [dispatch],
    );
    const activateTrackLaneTake = React.useCallback(
        (clipId: string, takeId: string) => {
            void dispatch(setClipActiveTakeRemote({ clipId, takeId }));
        },
        [dispatch],
    );
    /**
     * 单侧循环到下一个形状并重置默认曲率。
     *
     * 多选：重点 clip 属于选区时应用到全部选中（与淡变菜单同一判定），
     * 每个 clip 从**自身**当前形状循环前进；一次 bulk 提交 = 单撤销步。
     *
     * @param checkpoint 单独调用默认 true（一笔后端写入 = 一个撤销步）；
     *                   交叉点双列循环时传 false 并由调用方开 undo group
     *                   合并为单步。
     */
    const cycleOneFade = React.useCallback(
        (clipId: string, side: "in" | "out", checkpoint = true) => {
            // 形状循环写的是宿主形状轴：legacy 写 `C_FADE*SHAPE`，continuous 写
            // `(curvature, S)` 预设对。轴版本读不出来时不做（也不静默退化成别的编辑）。
            if (!canEditFadeShape()) return;
            const targets = getBulkEditableClipIds({
                activeClipId: clipId,
                multiSelectedClipIds,
                multiSelectedSet,
            });
            const updates: Array<{
                clipId: string;
                fadeInShape?: number;
                fadeInDir?: number;
                fadeOutShape?: number;
                fadeOutDir?: number;
            }> = [];
            for (const targetId of targets) {
                const clip = sessionRef.current.clips.find((entry) => entry.id === targetId);
                if (!clip) continue;
                // 【为什么不能直接用 `clip.fadeInShape`】宿主是 7.81+ 时那个字段是
                // `C_FADE*SHAPE` 的**派生**读数：非预设曲线上它可能是 -1（多对一，
                // 见 `fade_axes.rs`），照着循环会从"线性"重新开始而不是从相邻预设。
                // `hostFadeDisplayShape` 是"画布实际画的那一族"的唯一判定，
                // 与浮标、右键菜单用的是同一个。
                const currentShape = hostFadeDisplayShape(
                    clip.hostFades,
                    side === "out",
                    Number(side === "in" ? clip.fadeInShape : clip.fadeOutShape) || 0,
                );
                const index = FADE_PRESETS.findIndex((preset) => preset.shape === currentShape);
                const nextPreset =
                    FADE_PRESETS[(index + 1 + FADE_PRESETS.length) % FADE_PRESETS.length];
                const nextDir = defaultFadeDirFor(nextPreset.shape, side === "out");
                dispatch(
                    setClipFades({
                        clipId: targetId,
                        ...(side === "in"
                            ? { fadeInShape: nextPreset.shape, fadeInDir: nextDir }
                            : { fadeOutShape: nextPreset.shape, fadeOutDir: nextDir }),
                    }),
                );
                updates.push({
                    clipId: targetId,
                    ...(side === "in"
                        ? { fadeInShape: nextPreset.shape, fadeInDir: nextDir }
                        : { fadeOutShape: nextPreset.shape, fadeOutDir: nextDir }),
                });
            }
            if (updates.length > 0) {
                void dispatch(setClipsStateBulkRemote({ updates, checkpoint }));
            }
        },
        [dispatch, sessionRef, multiSelectedClipIds, multiSelectedSet],
    );

    // Ctrl+点击循环切换：普通包络线只切该线；交叉点抓手同时切换两侧
    // （前者淡出 + 后者淡入）。
    const handleFadeShapeCycleClick = React.useCallback(
        (clipId: string, side: "in" | "out") => {
            cycleOneFade(clipId, side);
        },
        [cycleOneFade],
    );
    const handleCrossfadeCycleClick = React.useCallback(
        (sides: Array<{ clipId: string; isOut: boolean }>) => {
            // 交叉点双列循环 = 一次手势：开 undo group 把两侧循环合并为
            // 单个撤销步（否则两笔 checkpoint:true 会变成两个撤销步）。
            void (async () => {
                await webApi.beginUndoGroup("edit_clip");
                try {
                    for (const side of sides) {
                        cycleOneFade(side.clipId, side.isOut ? "out" : "in", false);
                    }
                } finally {
                    await webApi.endUndoGroup();
                }
            })().catch(() => undefined);
        },
        [cycleOneFade],
    );
    const activeFormantToolClip = React.useMemo(
        () =>
            s.clipFormantToolWindow.clipId
                ? (s.clips.find((clip) => clip.id === s.clipFormantToolWindow.clipId) ?? null)
                : null,
        [s.clipFormantToolWindow.clipId, s.clips],
    );

    // ── MIDI clip drag-drop handler ──────────────────────
    const {
        handleMidiClipImport,
        handleReplaceMidiImport,
        openReplaceMidiForClip,
        midiClipRootTrackComposeEnabled,
        handleRequestEnableCompose,
        handleExportMidi,
    } = useTimelineMidiActions({
        dispatch,
        s,
        sessionRef,
        midiClipTrackId,
        midiClipStartSec,
        replaceMidiDialog,
        setReplaceMidiDialog,
    });
    const { tauriDraggedPathRef, tauriLastDropPathRef, tauriDropHandledAtRef } =
        useTimelineDragDrop({
            dispatch,
            viewport: viewportAccess,
            sessionRef,
            pxPerSecRef,
            rowHeightRef,
            dropPreviewRef,
            pendingDropDurationPathRef,
            beatFromClientX,
            snapTimeline,
            trackIdFromClientY: resolveTrackIdAtClientY,
            rowTopForTrackId,
            setDropPreview,
            ensureDropPreviewDuration,
            getDropPreviewWidthPx,
            setImportModeMenu,
            setProjectActionMenu,
            pxPerSec,
            rowHeight,
            onMidiDrop: (payload) => {
                onMidiDialogSourceChange("dragDrop");
                onMidiClipPathChange(payload.midiPath);
                onMidiClipStartSecChange(payload.startSec);
                onMidiClipTrackIdChange(payload.trackId);
                onMidiClipDialogOpenChange(true);
            },
        });

    /**
     * 拖入几何用的水平滚动量（模式无关）。
     *
     * 内核模式下时间轴是**自绘滚动**，DOM 容器的 scrollLeft 恒为 0——真值在内核
     * 宿主里。旧模式仍取容器自身的 scrollLeft。
     *
     * @param el 事件目标容器（旧模式下即滚动容器）。
     * @returns 当前水平滚动量（CSS px）。
     */
    const {
        dragScrollLeftOf,
        handleKernelSeek,
        handleKernelSeekTo,
        handleKernelSelectClip,
        handleKernelDragPreview,
        handleKernelDragCommit,
        kernelDropToNewTrack,
        kernelGhost,
        beginKernelGestureInteraction,
        endKernelGestureInteraction,
    } = useTimelineKernelSeekDrag({ state, clipActions, s, dispatch, kernelHostRef });

    const {
        handleKernelTrimPreview,
        handleKernelTrimCommit,
        handleKernelSnapOffsetPreview,
        handleKernelSnapOffsetCommit,
    } = useTimelineKernelTrim({
        state,
        s,
        dispatch,
        multiSelectedClipIds,
        timeValueFormatCtx,
        tVars,
        beginKernelGestureInteraction,
        endKernelGestureInteraction,
    });

    const {
        handleKernelClipHover,
        handleKernelFadePreview,
        handleKernelFadeCommit,
        fadeTooltipAnchor,
        publishFadeGripInfo,
    } = useTimelineKernelFade({
        state,
        dispatch,
        multiSelectedClipIds,
        timeValueFormatCtx,
        t,
        tVars,
        beginKernelGestureInteraction,
        endKernelGestureInteraction,
    });

    const {
        handleKernelBoxSelectPreview,
        handleKernelBoxSelectCommit,
        handleKernelBoxSelectToParamSelection,
        handleKernelContextMenu,
        handleKernelFadeContextMenu,
        handleKernelFadeHover,
        handleKernelActivateTake,
        handleKernelClipParamSelectionGesture,
        handleKernelToggleClipMute,
        handleKernelOpenClipFormant,
        handleKernelRateBadgeMenu,
    } = useTimelineKernelSelection({
        sessionRef,
        multiSelectedClipIds,
        setMultiSelectedClipIds,
        dispatch,
        clearContextMenu,
        setContextMenu,
        setTrackAreaMenu,
        fadeTooltipAnchor,
        timeValueFormatCtx,
        t,
        timelineRuntimeIsPlaying,
        activateTrackLaneTake,
        toggleTrackLaneClipMuted,
        setRateEditorClipId,
        setRateEditorPosition,
    });

    const {
        handleKernelGainDragPreview,
        handleKernelGainDragCommit,
        handleKernelGainReset,
        handleKernelRenameClipStart,
        handleKernelBadgeEditStart,
        kernelInlineEditProp,
        handleKernelCrossfadeGripPreview,
        handleKernelCrossfadeGripCommit,
    } = useTimelineKernelGainCrossfade({
        state,
        s,
        dispatch,
        multiSelectedClipIds,
        paramFineAdjustKb,
        t,
        commitTrackLaneGain,
        commitTrackLaneRename,
        commitTrackLaneRate,
        publishFadeGripInfo,
        fadeTooltipAnchor,
        beginKernelGestureInteraction,
        endKernelGestureInteraction,
    });
    const kernelInteractions = React.useMemo(
        () => ({
            geometryReadOnly: isHostGeometryReadOnly(),
            // 【为什么拆成两个标志】形状与曲率同门槛（都要宿主版本分得清轴语义），
            // 但分开表达让"哪件事被关掉"一目了然；长度永远可写，不受这两个影响。
            fadeShapeReadOnly: !canEditFadeShape(),
            fadeCurvatureReadOnly: !canEditFadeCurvature(),
            onSeek: handleKernelSeek,
            onSeekTo: handleKernelSeekTo,
            onSelectClip: handleKernelSelectClip,
            onClipParamSelectionGesture: handleKernelClipParamSelectionGesture,
            onToggleClipMute: handleKernelToggleClipMute,
            onToggleGroupDisabled: handleToggleGroupDisabled,
            onOpenClipFormant: handleKernelOpenClipFormant,
            onRateBadgeMenu: handleKernelRateBadgeMenu,
            onRenameClipStart: handleKernelRenameClipStart,
            onBadgeEditStart: handleKernelBadgeEditStart,
            // 拦截经 ref 中转：面板的接管实现（音高拖拽）在渲染后段才实例化。
            onClipPointerDownIntercept: (args: {
                clipId: string;
                clientX: number;
                clientY: number;
                pointerId: number;
                modifiers: {
                    ctrlKey: boolean;
                    shiftKey: boolean;
                    altKey: boolean;
                    metaKey: boolean;
                };
                container: HTMLElement;
            }) => kernelClipInterceptRef.current?.(args) ?? false,
            onGainDragPreview: handleKernelGainDragPreview,
            onGainDragCommit: handleKernelGainDragCommit,
            onGainReset: handleKernelGainReset,
            onCrossfadeGripPreview: handleKernelCrossfadeGripPreview,
            onCrossfadeGripCommit: handleKernelCrossfadeGripCommit,
            onFadeShapeCycle: handleFadeShapeCycleClick,
            onCrossfadeCycle: handleCrossfadeCycleClick,
            // 重置曲率走既有总线（旧实现同样经它派发）：消费者在淡变相关的 hook 里，
            // 这条契约与渲染模式无关。内核只给「哪些侧」，请求包络由这里组装。
            onResetFadeCurvature: (sides: Array<{ clipId: string; isOut: boolean }>) => {
                if (canEditFadeCurvature()) requestResetFadeCurvature({ sides });
            },
            onDragPreview: handleKernelDragPreview,
            onDragCommit: handleKernelDragCommit,
            onTrimPreview: handleKernelTrimPreview,
            onTrimCommit: handleKernelTrimCommit,
            onFadePreview: handleKernelFadePreview,
            onFadeCommit: handleKernelFadeCommit,
            onSnapOffsetPreview: handleKernelSnapOffsetPreview,
            onSnapOffsetCommit: handleKernelSnapOffsetCommit,
            onBoxSelectPreview: handleKernelBoxSelectPreview,
            onBoxSelectCommit: handleKernelBoxSelectCommit,
            onBoxSelectToParamSelection: handleKernelBoxSelectToParamSelection,
            onContextMenu: handleKernelContextMenu,
            // 【为什么永远提供】此前轴版本读不出来时这里给 `undefined`，于是淡变右键
            // 菜单**整个消失**、降级到通用 clip 菜单 —— 用户连"重置曲率/看形状"的入口
            // 都找不到。菜单内部已经按能力降级（形状行不可选时给说明文字，曲率滑杆
            // 照常），所以"给菜单"和"能不能写"是两件事，不该混成一个闸门。
            onFadeContextMenu: handleKernelFadeContextMenu,
            onFadeHover: handleKernelFadeHover,
            onClipHover: handleKernelClipHover,
            onActivateTake: handleKernelActivateTake,
        }),
        [
            handleKernelSeek,
            handleKernelSeekTo,
            handleKernelSelectClip,
            handleKernelDragPreview,
            handleKernelDragCommit,
            handleKernelTrimPreview,
            handleKernelTrimCommit,
            handleKernelFadePreview,
            handleKernelFadeCommit,
            handleKernelSnapOffsetPreview,
            handleKernelSnapOffsetCommit,
            handleKernelBoxSelectPreview,
            handleKernelBoxSelectCommit,
            handleKernelBoxSelectToParamSelection,
            handleKernelContextMenu,
            handleKernelFadeContextMenu,
            handleKernelFadeHover,
            handleKernelClipHover,
            handleKernelActivateTake,
            handleKernelRateBadgeMenu,
            // 下面这些回调内联在 `kernelInteractions` 里（或经局部函数转发），
            // 原先同样漏在依赖数组外：`useMemo` 会继续持有**首帧闭包**，表现为
            // 「改了设置 / 选择后，内核手势仍按旧值执行」。依赖数组必须与
            // useMemo 体内引用的回调一一对应。
            handleKernelClipParamSelectionGesture,
            handleKernelToggleClipMute,
            handleToggleGroupDisabled,
            handleKernelOpenClipFormant,
            handleKernelRenameClipStart,
            handleKernelBadgeEditStart,
            handleKernelGainDragPreview,
            handleKernelGainDragCommit,
            handleKernelGainReset,
            handleKernelCrossfadeGripPreview,
            handleKernelCrossfadeGripCommit,
            handleFadeShapeCycleClick,
            handleCrossfadeCycleClick,
        ],
    );

    // ── 4. 全局事件监听 ─────────────────────────────────────
    useTimelineEventHandlers({
        dispatch,
        sessionRef,
        getPlayheadSec: getVisualPlayheadSec,
        scrollRef,
        viewport: viewportAccess,
        trackListScrollRef,
        pxPerSecRef,
        viewportWidthRef,
        keyboardZoomPendingRef,
        pxPerSec,
        setPxPerSec,
        commitScrollLeftState: setScrollLeftState,
        rowHeight,
        setMultiSelectedClipIds,
        copyClips,
        cutClips,
        pasteClipsAtPlayhead,
        splitSelectedAtPlayhead,
        normalizeClips,
        groupClips,
        ungroupClips,
        contextMenu,
        trackAreaMenu,
        setContextMenu,
        setTrackAreaMenu,
        syncScrollLeft,
    });

    // ── 5. 拖拽 hooks 桥接 ──────────────────────────────────
    const formatClipPitchDragTooltip = React.useCallback(
        (cents: number) =>
            t("clip_pitch_drag_tooltip").replace("{delta}", formatPitchDragCents(cents)),
        [t],
    );
    const { startClipPitchDrag, pitchDragTooltip } = useClipPitchDrag({
        sessionRef,
        dispatch,
        fineAdjustKb: paramFineAdjustKb,
        formatDragTooltip: formatClipPitchDragTooltip,
    });

    /**
     * 内核 clip 左键按下拦截：`Alt + Shift`（`modifier.clipPitchDrag`）按住时
     * 把这次手势**整体交给旧实现的 `useClipPitchDrag`**。
     *
     * 【为什么委托而不是重写】该手势是一台带异步状态机的完整实现：先取基准参数帧、
     * 逐帧节流写后端预览、undo group 惰性开启、收尾提交 / 回滚、tooltip 发布。
     * 内核侧重写会产生第二份音高语义；委托则天然与旧实现一致（tooltip 浮层也已
     * 由面板渲染，与渲染模式无关）。
     *
     * 特殊说明：旧 hook 需要「React 合成事件」形态的入参（只用 button / pointerId /
     * clientX / clientY / preventDefault / stopPropagation / currentTarget），因此这里
     * 构造一个最小鸭子类型对象（与旧 `ClipHeader` 调用 `startEditDrag` 的手法一致）。
     */
    const handleKernelClipPointerDownIntercept = React.useCallback(
        (args: {
            clipId: string;
            clientX: number;
            clientY: number;
            pointerId: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
            container: HTMLElement;
        }): boolean => {
            if (!isModifierActive(pitchDragKb, args.modifiers)) return false;
            const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) return false;
            startClipPitchDrag(
                {
                    button: 0,
                    pointerId: args.pointerId,
                    clientX: args.clientX,
                    clientY: args.clientY,
                    preventDefault: () => undefined,
                    stopPropagation: () => undefined,
                    currentTarget: args.container,
                } as unknown as React.PointerEvent<HTMLDivElement>,
                args.clipId,
            );
            return true;
        },
        [pitchDragKb, sessionRef, startClipPitchDrag],
    );
    kernelClipInterceptRef.current = handleKernelClipPointerDownIntercept;

    const {
        handleSelectTrack,
        handleRemoveTrack,
        handleMoveTrack,
        handleToggleTrackMute,
        handleToggleTrackSolo,
        handleToggleTrackCompose,
        handleTrackVolumeUiChange,
        handleTrackVolumeCommit,
        handleAddTrack,
        handleTrackColorChange,
        handleTrackAlgoChange,
        handleTrackNameChange,
        handleDuplicateTrack,
        handleDuplicateTrackTo,
        handleCreateTrackBelow,
        handleTrackListScrollTopChange,
        kernelActiveGroupIds,
    } = useTimelineTrackActions({
        dispatch,
        sessionRef,
        setTrackVolumeUi,
        kernelHostRef,
        s,
        multiSelectedClipIds,
        disabledGroupIds,
    });
    // 主题切换 → darkMode prop 变化 → clip 体画布同帧按新主题重绘配色。
    const { mode: themeMode } = useAppTheme();
    const darkMode = themeMode === "dark";
    const timelineScrollRange = useMemo(
        () => resolveTimelineScrollRange({ contentWidth, viewportWidth }),
        [contentWidth, viewportWidth],
    );
    // ═════════════════════════════════════════════════════════
    // JSX 渲染
    // ═════════════════════════════════════════════════════════

    // scrollLeft 按 REACT_SCROLL_STEP_PX 量化提交，React 渲染期用 `timelineAxis`
    // 算出的位置最多滞后一个步长。轨道区播放头由内核自绘，其重绘请求由
    // TimelineTransportBridge 的 onFrame 逐帧发起（那里读的是视觉插值真值）；
    // 这里不需要再纠正任何 DOM。

    /**
     * 标尺播放头元素（竖线 / 倒三角）的挂载回调。
     *
     * 【为什么挂载时要请求一帧】两者的位置只由内核在帧提交里写，而元素可能在
     * **播放头静止**时被重建（面板重挂载 / 视图切换）：新节点没有任何 `left`，
     * 等价于 `left: auto` ⇒ 落在静态位置（左缘 = 工程起始处）。此时若没有任何东西
     * 弄脏帧（暂停且不滚动），内核不提交，元素就**停在工程起始处不动**。元素一出现
     * 就请求一帧，写入器随即按当前视口定位它（去重键含元素身份，见
     * `createPlayheadElementWriter`）。
     */
    const attachRulerPlayheadLine = React.useCallback(
        (element: HTMLDivElement | null) => {
            rulerPlayheadLineRef.current = element;
            if (element !== null) kernelHostRef.current?.invalidatePlayhead();
        },
        [rulerPlayheadLineRef],
    );
    const attachRulerPlayheadHead = React.useCallback(
        (element: HTMLDivElement | null) => {
            rulerPlayheadHeadRef.current = element;
            if (element !== null) kernelHostRef.current?.invalidatePlayhead();
        },
        [rulerPlayheadHeadRef],
    );

    /**
     * 标尺节点（内核模式与旧模式共用同一实例）。
     *
     * 标尺是重交互、低频变化的 DOM 子树（刻度标签 / Tempo Map 旗帜拖拽 / 内联编辑 /
     * 右键菜单），搬进 canvas 等于整体重写且收益极低。让两种渲染模式共用它，视觉与
     * 交互天然与旧实现一致；内核只把「跟随水平滚动」的部分收敛为 rAF 内一次
     * transform 写入（见 kernel host 的 syncDom）。
     */
    const timeRulerNode = (
        // playheadSec 传提交值（而非渲染期读 ref）：`positionPlayheadFromProps={false}`
        // 时 React **不写**这两个元素的位置，视觉插值完全由内核逐帧命令式驱动
        // （含倒三角，见 `TimelineKernelDomSync.rulerPlayheadHead`）；`playheadSec`
        // 只作为其它消费方（如内联编辑）的提交值来源。
        //
        // 标尺不消费实时滚动位置：刻度与可见范围都按量化的 `rulerScrollLeft` 生成
        // （缓冲已保证覆盖视口），这样滚动期间 `TimeRulerMarks` 的 memo 不会失效，
        // 整棵刻度子树不必每帧重渲染。内核模式下水平滚动由内核在 rAF 内直接写
        // 内容层 transform（不经 React），这条量化约定依然成立。
        <TimeRuler
            // `scrollLeft` 仍是量化 state：只服务随 React 重渲染的低频几何
            // （Tempo Map 可见段 / 浮动标签）。鼠标交互换算一律走 `scrollLeftLive`
            // 读取内核逐帧真值 —— 用 state 会把 ToolTip/点击落点算偏一个量化残差。
            scrollLeft={scrollLeft}
            scrollLeftLive={state.getScrollLeftLive}
            tickWindowAnchorPx={rulerScrollLeft}
            ticks={timelineTicks}
            pxPerSec={pxPerSec}
            viewportWidth={viewportWidth}
            playheadSec={s.playheadSec}
            positionPlayheadFromProps={false}
            onRulerWheel={handleRulerWheel}
            playheadLineRef={attachRulerPlayheadLine}
            playheadHeadRef={attachRulerPlayheadHead}
            contentRef={rulerContentRef}
            timeContext={timeContext}
            primaryUnit={s.primaryTimeUnit}
            secondaryUnit={s.secondaryTimeUnit}
            onPrimaryUnitChange={handlePrimaryUnitChange}
            onSecondaryUnitChange={handleSecondaryUnitChange}
            onOpenSettings={handleRulerOpenSettings}
            onCopyPlayheadTime={handleRulerCopyPlayheadTime}
            t={tf}
            tempoMap={s.tempoMap}
            tempoMapVisible={s.tempoMapVisible}
            projectSec={dynamicProjectSec}
            grid={s.grid}
            snapEnabled={s.snapEnabled}
            timelineSnap={s.timelineSnap}
            projectScale={projectScale}
            projectScaleName={
                s.project.useCustomScale ? (s.project.customScale?.name ?? undefined) : undefined
            }
            fallbackDenominator={s.project.timeSignatureDenominator}
            customScalePresets={s.customScalePresets}
            onTempoMapChange={handleTempoMapChange}
            onTempoMapCommit={handleTempoMapCommit}
            subscribeViewport={timelineViewportBus.subscribe}
            onMouseDown={handleRulerMouseDown}
        />
    );

    const { handleTimelineDragOver, handleTimelineDrop } = useTimelineFileDrop({
        tauriDraggedPathRef,
        tauriLastDropPathRef,
        tauriDropHandledAtRef,
        beatFromClientX,
        dragScrollLeftOf,
        resolveTrackIdAtClientY,
        dropPreview,
        setDropPreview,
        ensureDropPreviewDuration,
        dispatch,
        onMidiClipPathChange,
        onMidiClipStartSecChange,
        onMidiClipTrackIdChange,
        onMidiClipDialogOpenChange,
    });
    return (
        <Profiler
            id="TimelinePanel"
            onRender={(_id, _phase, actualDuration) => {
                // dev 帧率探针：经 globalThis 挂钩上报 React 提交耗时。
                // 未启用探针时只有一次属性查找，零成本。
                (
                    globalThis as unknown as {
                        __hfsFrameProfiler?: { recordReact(ms: number): void };
                    }
                ).__hfsFrameProfiler?.recordReact(actualDuration);
            }}
        >
            <Flex
                className="h-full w-full bg-qt-graph-bg overflow-hidden"
                // 编辑表面声明：文档级 pointerdown/focusin 捕获据此把本面板
                // 解析为「timeline」表面（轨道列在其内部以 trackHeader 就近
                // 覆盖，见 focusSurface.ts / data-hs-surface）。
                data-hs-surface="timeline"
                onPointerDownCapture={(e) => {
                    // 点击轨道背景等非输入区域时，主动失焦当前聚焦的输入框，
                    // 让名称/速率/增益/轨道增益等行内编辑器走各自的 onBlur
                    // 提交路径（背景 pointerdown 会 preventDefault，焦点不
                    // 转移时 blur 不会自然触发，输入框将无法退出）。
                    //
                    // ★ 失焦延迟到 rAF：若点击落在编辑中的 Clip 内部，ClipItem
                    //   捕获阶段的 controller.commit() 必须先获胜——同步 blur
                    //   会经 onBlur 抢先提交一次，controller 再提交一次，造成
                    //   双重提交。rAF 时编辑器已卸载/失焦，重复提交自然消失。
                    //   浮层面板（data-hs-floating-menu）不参与失焦逻辑。
                    const target = e.target as HTMLElement | null;
                    if (
                        target?.closest?.(
                            "input,textarea,select,[contenteditable='true'],[data-hs-floating-menu]",
                        )
                    ) {
                        return;
                    }
                    const active = document.activeElement as HTMLElement | null;
                    if (active && active.tagName === "INPUT") {
                        requestAnimationFrame(() => {
                            if (document.activeElement === active) {
                                active.blur();
                            }
                        });
                    }
                }}
            >
                {/*
                 * 轨道头宽度由布局状态驱动（`gutters.timelineTrackHeaderPx`），
                 * 取代原先写死的 `w-64`。外层 div 持有宽度：拖动时只改它的
                 * 内联样式，TrackList 自身填满即可，避免 memo 组件因宽度 prop
                 * 变化而重渲染。
                 */}
                <div
                    ref={trackHeaderRef}
                    className="shrink-0 flex min-h-0"
                    style={{ width: trackHeaderWidthPx }}
                >
                    <TrackList
                        t={t}
                        tracks={s.tracks}
                        trackMeters={s.trackMeters}
                        selectedTrackId={s.selectedTrackId}
                        rowHeight={rowHeight}
                        setRowHeight={setRowHeight}
                        verticalZoomKb={verticalZoomKb}
                        paramFineAdjustKb={paramFineAdjustKb}
                        trackVolumeUi={trackVolumeUi}
                        listScrollRef={trackListScrollRef}
                        onSelectTrack={handleSelectTrack}
                        onRemoveTrack={handleRemoveTrack}
                        onMoveTrack={handleMoveTrack}
                        copyDragKb={copyDragKb}
                        onDuplicateTrackTo={handleDuplicateTrackTo}
                        onToggleMute={handleToggleTrackMute}
                        onToggleSolo={handleToggleTrackSolo}
                        onToggleCompose={handleToggleTrackCompose}
                        onVolumeUiChange={handleTrackVolumeUiChange}
                        onVolumeCommit={handleTrackVolumeCommit}
                        onAddTrack={handleAddTrack}
                        onTrackColorChange={handleTrackColorChange}
                        onAlgoChange={handleTrackAlgoChange}
                        onTrackNameChange={handleTrackNameChange}
                        onDuplicateTrack={handleDuplicateTrack}
                        onCreateTrackBelow={handleCreateTrackBelow}
                        onScrollTopChange={handleTrackListScrollTopChange}
                        headerHeight={timeRulerHeightPx(
                            Boolean(
                                s.tempoMap && s.tempoMap.points.length > 0 && s.tempoMapVisible,
                            ),
                        )}
                        bottomGutterHeightPx={horizontalScrollbarGutterPx}
                    />
                </div>
                <DockGutter
                    dir="col"
                    value={trackHeaderWidthPx}
                    min={GUTTER_LIMITS.timelineTrackHeaderPx.min}
                    max={GUTTER_LIMITS.timelineTrackHeaderPx.max}
                    label={t("aria_resize_track_header")}
                    onLive={(px) => {
                        const element = trackHeaderRef.current;
                        if (element) element.style.width = `${px}px`;
                    }}
                    onCommit={(px) => dispatch(setGutterSize({ key: "timelineTrackHeaderPx", px }))}
                    onReset={() =>
                        dispatch(
                            setGutterSize({
                                key: "timelineTrackHeaderPx",
                                px: DEFAULT_GUTTER_SIZES.timelineTrackHeaderPx,
                            }),
                        )
                    }
                />

                {/* Timeline View (Right) */}
                <Flex direction="column" className="flex-1 relative overflow-hidden bg-qt-graph-bg">
                    {/* 新渲染内核（自绘滚动 + 单 WebGL2）渲染**轨道区**。
                        标尺（`timeRulerNode`）与左侧轨道头保留 DOM：它们重交互、低频
                        变化，搬进 canvas 等于重写 Tempo 旗帜拖拽与电平表，收益极低；
                        内核只把「跟随视口」的部分收敛为 rAF 内一次 transform / scrollTop
                        写入。轨道区（网格 / clip / 波形 / 交互）才是滚动瓶颈，由内核自绘。

                        内核是**唯一**渲染路径：旧实现（原生滚动 + Canvas2D 的
                        `TimelineScrollArea` / `TimelineSurface` / `TrackLane` 等）已随
                        阶段 2/3 删除，此处没有运行期二选一，也没有可回退的第二条路径。
                        WebGL2 不可用时内核视图会回报失败，此处改渲染可自助排障的
                        失败界面（见 `KernelUnavailableNotice` 与
                        `timeline/kernel/kernelAvailability`）——没有回退渲染路径。 */}
                    {isKernelAvailable(kernelUnavailableReason !== null) ? (
                        <>
                            {timeRulerNode}
                            <TimelineKernelView
                                rowHeight={rowHeight}
                                onRowHeightChange={setRowHeight}
                                initialPxPerSec={pxPerSec}
                                onPxPerSecChange={setPxPerSec}
                                onZoomRequest={handleKernelZoomRequest}
                                onScrollLeftCommit={handleKernelScrollLeftCommit}
                                onScrollLeftFrame={state.syncScrollLeftFrame}
                                onViewportWidthChange={setViewportWidth}
                                onUnavailable={handleKernelUnavailable}
                                getPlayheadSec={getVisualPlayheadSec}
                                rulerContentRef={rulerContentRef}
                                trackListScrollerRef={trackListScrollRef}
                                rulerPlayheadLineRef={rulerPlayheadLineRef}
                                rulerPlayheadHeadRef={rulerPlayheadHeadRef}
                                hostRef={kernelHostRef}
                                interactions={kernelInteractions}
                                activeGroupIds={kernelActiveGroupIds}
                                disabledGroupIds={disabledGroupIds}
                                silenceSegmentsByClipId={silencePreviewSegments ?? undefined}
                                showAllTakes={s.showAllTakes}
                                inlineEdit={kernelInlineEditProp}
                                snapHighlight={{
                                    pxPerSec,
                                    contentWidth: timelineScrollRange.paddedContentWidth,
                                    contentHeight,
                                }}
                                ghost={
                                    kernelGhost === null
                                        ? undefined
                                        : {
                                              // ghost 的轨道着色与真实 clip 同源（旧实现
                                              // 用 color-mix + 归一化轨道色）。
                                              items: kernelGhost.map((item) => {
                                                  const trackColor =
                                                      s.tracks.find((t) => t.id === item.trackId)
                                                          ?.color || undefined;
                                                  const tint =
                                                      trackColor === undefined
                                                          ? null
                                                          : normalizedTrackColorCss(
                                                                trackColor,
                                                                darkMode,
                                                            );
                                                  return {
                                                      ...item,
                                                      headerBackground:
                                                          tint === null
                                                              ? "var(--qt-clip-bg)"
                                                              : `color-mix(in oklab, var(--qt-clip-bg) 40%, ${tint} 60%)`,
                                                      bodyBackground:
                                                          tint === null
                                                              ? "var(--qt-clip-bg)"
                                                              : `color-mix(in oklab, var(--qt-clip-bg) 45%, ${tint} 55%)`,
                                                  };
                                              }),
                                              contentWidth: timelineScrollRange.paddedContentWidth,
                                              contentHeight,
                                          }
                                }
                                dropPreview={
                                    // 【不能因为 `trackId === null` 就不渲染】`null` 表示
                                    // 落在全部轨道之下（将新建轨道）：旧实现用
                                    // `rowTopForTrackId(null) = tracks.length × rowHeight`
                                    // 把预览画在**新轨道那一行**，用户能看到素材预览。
                                    // 面板这里曾直接传 `undefined`，于是拖到下方完全没有
                                    // 预览（`contentHeight` 已经为哨兵行留了高度，见
                                    // `useTimelineState` 的 `dropExtraRows`）。
                                    dropPreview === null
                                        ? undefined
                                        : {
                                              leftPx: Math.max(0, dropPreview.startSec * pxPerSec),
                                              widthPx: Math.max(
                                                  1,
                                                  pxPerSec *
                                                      (dropPreview.durationSec > 0
                                                          ? dropPreview.durationSec
                                                          : 1),
                                              ),
                                              trackId: dropPreview.trackId,
                                              fileName: dropPreview.fileName,
                                              contentWidth: timelineScrollRange.paddedContentWidth,
                                              contentHeight,
                                          }
                                }
                                onDragOver={handleTimelineDragOver}
                                onDrop={handleTimelineDrop}
                                // 拖出容器即清掉落点预览。旧实现（`TimelineScrollArea`
                                // 的 `onDragLeave`）同源：只看 `relatedTarget` 是否仍在
                                // 容器内——子元素之间移动也会触发 `dragleave`，不加这层
                                // 判断会把预览抖掉。缺了它，把文件拖出时间轴后预览会一直
                                // 挂着（没有任何后续事件会清它）。
                                onDragLeave={(event) => {
                                    const related = event.relatedTarget as Node | null;
                                    if (related !== null && event.currentTarget.contains(related)) {
                                        return;
                                    }
                                    setDropPreview(null);
                                }}
                                // 预览内层元素的 ref：`useTimelineDragDrop` 拖动期间
                                // 直接写它的 style 移动预览（不 setState）。不接上时
                                // 同一条轨道内移动指针预览不动（见该 prop 的说明）。
                                dropPreviewItemRef={dropPreviewRef}
                                newTrackDrop={
                                    kernelDropToNewTrack
                                        ? {
                                              // 幽灵行里的 clip 取 Redux 里的乐观位置
                                              // （预览已把参与者写到哨兵轨上，含吸附结果）。
                                              items: s.clips
                                                  .filter(
                                                      (clip) => clip.trackId === NEW_TRACK_SENTINEL,
                                                  )
                                                  .map((clip) => ({
                                                      key: clip.id,
                                                      leftPx: Math.max(0, clip.startSec * pxPerSec),
                                                      widthPx: Math.max(
                                                          1,
                                                          clip.lengthSec * pxPerSec,
                                                      ),
                                                  })),
                                              contentWidth: timelineScrollRange.paddedContentWidth,
                                          }
                                        : undefined
                                }
                            />
                        </>
                    ) : (
                        <KernelUnavailableNotice reason={kernelUnavailableReason ?? "未知原因"} />
                    )}

                    {/* 共振峰工具窗口：`fixed` 定位（视口坐标），与渲染模式无关。
                        原先它随旧子树一起被包在内核开关内，内核模式下右键
                        「共振峰变形」后窗口不出现——这里移到开关之外，两种模式共用。 */}
                    {s.clipFormantToolWindow.open && activeFormantToolClip ? (
                        <ClipFormantToolWindow
                            clip={activeFormantToolClip}
                            status={s.clipFormantStatus[activeFormantToolClip.id] ?? "ready"}
                            x={s.clipFormantToolWindow.x}
                            y={s.clipFormantToolWindow.y}
                            onCommit={commitTrackLaneFormantMorph}
                            onMove={(x, y) => dispatch(setClipFormantToolWindowPosition({ x, y }))}
                            onClose={() => dispatch(closeClipFormantToolWindow())}
                        />
                    ) : null}

                    {/*
                      导入模式选择菜单 / 工程文件拖放菜单。

                      这两处此前是手写模态（`fixed inset-0` + 绝对定位卡片）：没有 Esc、
                      没有焦点管理、不抑制全局快捷键，菜单项的悬停/焦点样式也与全应用
                      其它菜单不同。它们的内容是**扁平项列表**，正好是 `AppContextMenu`
                      的形态 —— 迁过去即得到位置夹紧、方向键、Home/End、Esc、外部点击
                      关闭这一整套，且与剪辑右键菜单视觉一致。
                    */}
                    {importModeMenu ? (
                        <AppContextMenu
                            x={importModeMenu.x}
                            y={importModeMenu.y}
                            floating
                            ariaLabel={t("menu_import_media")}
                            onClose={() => setImportModeMenu(null)}
                            items={[
                                {
                                    key: "across-time",
                                    label: t("import_across_time"),
                                    onSelect: () => {
                                        const m = importModeMenu;
                                        if (m.audioPaths.length === 1) {
                                            void dispatch(
                                                importAudioAtPosition({
                                                    audioPath: m.audioPaths[0],
                                                    trackId: m.trackId,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        } else {
                                            void dispatch(
                                                importMultipleAudioAtPosition({
                                                    audioPaths: m.audioPaths,
                                                    mode: "across-time",
                                                    trackId: m.trackId,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        }
                                    },
                                },
                                {
                                    key: "across-tracks",
                                    label: t("import_across_tracks"),
                                    onSelect: () => {
                                        const m = importModeMenu;
                                        if (m.audioPaths.length === 1) {
                                            void dispatch(
                                                importAudioAtPosition({
                                                    audioPath: m.audioPaths[0],
                                                    trackId: null,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        } else {
                                            void dispatch(
                                                importMultipleAudioAtPosition({
                                                    audioPaths: m.audioPaths,
                                                    mode: "across-tracks",
                                                    trackId: m.trackId,
                                                    startSec: m.startSec,
                                                }),
                                            );
                                        }
                                    },
                                },
                                // 【插件里为什么整项不出现】多 take 的**新建**没有宿主
                                // 对应（ARA 侧只能读宿主已有的 take）；与 MenuBar 的
                                // 模式选择同一道闸，免得这里留下一个点了没反应的入口。
                                ...(canImportAsTakes()
                                    ? [
                                          {
                                              key: "as-takes",
                                              label: t("import_as_takes"),
                                              onSelect: () => {
                                                  const m = importModeMenu;
                                                  void dispatch(
                                                      importMultipleAudioAtPosition({
                                                          audioPaths: m.audioPaths,
                                                          mode: "as-takes",
                                                          trackId: m.trackId,
                                                          startSec: m.startSec,
                                                      }),
                                                  );
                                              },
                                          },
                                      ]
                                    : []),
                            ]}
                        />
                    ) : null}

                    {projectActionMenu ? (
                        <AppContextMenu
                            x={projectActionMenu.x}
                            y={projectActionMenu.y}
                            floating
                            ariaLabel={tf("import_project_dialog_title")}
                            onClose={() => setProjectActionMenu(null)}
                            items={[
                                {
                                    key: "open-project",
                                    label: t("menu_open_project"),
                                    shortcut: openProjectShortcut,
                                    onSelect: () =>
                                        emitExternalFileAction(
                                            "openProject",
                                            projectActionMenu.path,
                                        ),
                                },
                                {
                                    key: "import-project",
                                    label: tf("import_project_dialog_title"),
                                    onSelect: () =>
                                        window.dispatchEvent(
                                            new CustomEvent("hifi:importProjectPick", {
                                                detail: { path: projectActionMenu.path },
                                            }),
                                        ),
                                },
                            ]}
                        />
                    ) : null}

                    <FadeContextMenuHost />
                    {contextMenu
                        ? (() => {
                              const ctxClip = s.clips.find((c) => c.id === contextMenu.clipId);
                              if (!ctxClip) return null;

                              const selectedIds = resolveQuickExportClipIds({
                                  contextClipId: contextMenu.clipId,
                                  multiSelectedClipIds,
                              });
                              const selectedClips = s.clips.filter((c) =>
                                  selectedIds.includes(c.id),
                              );

                              // 右键位置的工程时间：**必须**用与渲染模式无关的视口
                              // 访问器。旧实现读原生 scroller 的 bounds + scrollLeft；
                              // 内核模式下 `scrollRef.current` 为 null，原写法会静默退化
                              // 成 `ctxClip.startSec`，于是"指针下有哪些重叠的淡变 clip"
                              // 这份候选集会与旧实现不同（菜单里少/多出条目）。
                              const _ctxBounds = viewportAccess.getRect();
                              const contextTimeSec =
                                  _ctxBounds !== null
                                      ? beatFromClientX(
                                            contextMenu.x,
                                            _ctxBounds,
                                            viewportAccess.getScrollLeft(),
                                        )
                                      : ctxClip.startSec;

                              const overlappingFadeClips = collectFadeContextClips({
                                  allClips: s.clips,
                                  contextClip: ctxClip,
                                  contextTimeSec,
                                  explicitOverlappingClipIds: contextMenu.overlappingClipIds,
                              });

                              const currentPlayheadSec = Math.max(
                                  0,
                                  Number(s.playheadSec ?? 0) || 0,
                              );
                              // 用与执行端同一个开区间谓词：闭区间会让"播放头恰好
                              // 落在 Clip 边缘"时菜单亮着、点下去却什么都不发生。
                              const playheadInClip = isClipSplittableAtSec(
                                  ctxClip,
                                  currentPlayheadSec,
                              );

                              // 菜单自己 portal 到 `document.body`（见 ClipContextMenu），
                              // 这里不再套一层 —— 表面归组件所有。
                              return (
                                  <ClipContextMenu
                                      x={contextMenu.x}
                                      y={contextMenu.y}
                                      clip={ctxClip}
                                      selectedClips={selectedClips}
                                      overlappingClips={overlappingFadeClips}
                                      playheadInClip={playheadInClip}
                                      canSplitSelected={selectedClips.some((c) =>
                                          isClipSplittableAtSec(c, currentPlayheadSec),
                                      )}
                                      onClose={() => setContextMenu(null)}
                                      onDelete={(ids) => {
                                          setContextMenu(null);
                                          setMultiSelectedClipIds([]);
                                          void dispatch(removeClipsRemote(ids));
                                      }}
                                      onMute={(ids, muted) => {
                                          // 批量走 bulk 通道：单次 IPC + 单个撤销步
                                          //（逐个 setClipStateRemote 会产生 N 次
                                          // IPC/N 步撤销）。乐观更新先行。
                                          for (const id of ids) {
                                              dispatch(
                                                  setClipMuted({
                                                      clipId: id,
                                                      muted,
                                                  }),
                                              );
                                          }
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates: ids.map((id) => ({
                                                      clipId: id,
                                                      muted,
                                                  })),
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                      onCopy={(ids) => {
                                          const s = sessionRef.current;
                                          const expandedIds = expandClipIdsWithGroups(
                                              ids,
                                              s.clips,
                                              s.ignoreGrouping,
                                              s.disabledGroupIds,
                                          );
                                          void copyClips(expandedIds);
                                      }}
                                      onCut={(ids) => {
                                          const s = sessionRef.current;
                                          const expandedIds = expandClipIdsWithGroups(
                                              ids,
                                              s.clips,
                                              s.ignoreGrouping,
                                              s.disabledGroupIds,
                                          );
                                          setContextMenu(null);
                                          cutClips(expandedIds);
                                      }}
                                      onReplace={(ids) => {
                                          void replaceClipSources(ids);
                                      }}
                                      onReplaceMidi={(ids) => {
                                          if (ids.length > 0) {
                                              void openReplaceMidiForClip(ids[0]);
                                          }
                                      }}
                                      onQuickExport={(ids) => {
                                          setQuickExportDialog({
                                              open: true,
                                              clipIds: ids,
                                          });
                                      }}
                                      onSplit={(clipIds) => {
                                          setContextMenu(null);
                                          splitClipIdsAtPlayhead(clipIds);
                                      }}
                                      onGroup={(ids) => {
                                          setContextMenu(null);
                                          groupClips(ids);
                                      }}
                                      onUngroup={(ids) => {
                                          setContextMenu(null);
                                          ungroupClips(ids);
                                      }}
                                      onGlue={(ids) => {
                                          setContextMenu(null);
                                          if (ids.length >= 2) {
                                              void dispatch(glueClipsRemote(ids));
                                              setMultiSelectedClipIds([]);
                                          }
                                      }}
                                      onConvertToPitchRef={(ids) => {
                                          setContextMenu(null);
                                          void dispatch(convertClipsToPitchReferenceRemote(ids));
                                          setMultiSelectedClipIds([]);
                                      }}
                                      onUpdatePitchRef={(ids) => {
                                          setContextMenu(null);
                                          void dispatch(updatePitchReferenceRemote(ids));
                                          setMultiSelectedClipIds([]);
                                      }}
                                      onExportMidi={(ids) => {
                                          setContextMenu(null);
                                          void handleExportMidi(ids);
                                      }}
                                      onAddToParamSelection={(ids) => {
                                          // 批量入口：把所选音频块的时间范围并入
                                          // 参数编辑器选区（隐藏菜单由 PianoRollPanel
                                          // 消费，按当前根轨道组过滤，见 handleEditOp）。
                                          setContextMenu(null);
                                          window.dispatchEvent(
                                              new CustomEvent("hifi:editOp", {
                                                  detail: {
                                                      op: "addClipsToParamSelection",
                                                      clipIds: ids,
                                                  },
                                              }),
                                          );
                                      }}
                                      onFadeShapeChange={(clipIds, target, shape) => {
                                          // 切换形状必须重置曲率（REAPER 语义：各形状的
                                          // 默认曲率由形状自身定义，见 reaperFade 的
                                          // DEFAULT_FADE_DIR_BY_SHAPE / defaultFadeDirFor）。
                                          const dir = defaultFadeDirFor(shape, target === "out");
                                          const patch =
                                              target === "in"
                                                  ? { fadeInShape: shape, fadeInDir: dir }
                                                  : { fadeOutShape: shape, fadeOutDir: dir };
                                          // 乐观更新（本地 reducer）逐条应用；
                                          // 持久化走 bulk 通道：单次 IPC + 单个撤销步。
                                          batch(() => {
                                              for (const clipId of clipIds) {
                                                  dispatch(setClipFades({ clipId, ...patch }));
                                              }
                                          });
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates: clipIds.map((clipId) => ({
                                                      clipId,
                                                      ...patch,
                                                  })),
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                      onSilenceDetection={(ids) => setSilenceDialogIds(ids)}
                                      onNormalize={normalizeClips}
                                      onEditRate={openRateBadgeMenu}
                                      onToggleReverse={(ids, reversed) => {
                                          // 批量走 bulk 通道：单次 IPC + 单个撤销步
                                          //（逐个 setClipStateRemote 会产生 N 次 IPC/N 步撤销）。
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates: ids.map((id) => ({
                                                      clipId: id,
                                                      reversed,
                                                  })),
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                      onToggleLoop={(ids, loopEnabled) => {
                                          const session = sessionRef.current;
                                          const updates = ids.map((id) => {
                                              const clip = session.clips.find(
                                                  (entry) => entry.id === id,
                                              );
                                              const update: {
                                                  clipId: string;
                                                  loopEnabled: boolean;
                                                  sourceEndSec?: number;
                                              } = { clipId: id, loopEnabled };
                                              // 关闭循环的瞬间：非 Loop 正放 Clip 按
                                              // 派生窗口模型归一 source_end
                                              //（= 起点+长度×速率）。循环期间锚点被
                                              // 回绕/窗口被保持，直接关掉会把陈旧
                                              // 窗口带入非 Loop 状态 —— 静音区冻结、
                                              // 音频错位都源于此。
                                              // 与后端 clip_effective_source_end_sec
                                              // 一致：不按 midiNoteData 排除 —— 音高
                                              // 参考块等无源媒体 Clip 的音高曲线
                                              //（trim_and_resample_midi）同样使用派生
                                              // 窗口，存储值也必须一并归一。
                                              if (!loopEnabled && clip && !clip.reversed) {
                                                  const rate =
                                                      Number(clip.playbackRate) > 0
                                                          ? Number(clip.playbackRate)
                                                          : 1;
                                                  update.sourceEndSec =
                                                      (Number(clip.sourceStartSec) || 0) +
                                                      Math.max(0, clip.lengthSec) * rate;
                                              }
                                              return update;
                                          });
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates,
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                      onSetChannelMode={(ids, mode) => {
                                          // 批量走 bulk 通道：单次 IPC + 单个撤销步。
                                          // 作用范围（active take / 全部 take）跟随全局的
                                          // "同步编辑所有 Take"设置。
                                          void dispatch(
                                              setClipsStateBulkRemote({
                                                  updates: ids.map((id) => ({
                                                      clipId: id,
                                                      channelMode: mode,
                                                  })),
                                                  checkpoint: true,
                                              }),
                                          );
                                      }}
                                      onScanFakeStereo={(ids) => {
                                          // 后端整批只打一个撤销步，因此前端不做
                                          // 逐条乐观更新；结果摘要写进状态行。
                                          void dispatch(
                                              scanAndConvertFakeStereoRemote({
                                                  clipIds: ids,
                                              }),
                                          );
                                      }}
                                  />
                              );
                          })()
                        : null}

                    {trackAreaMenu
                        ? createPortal(
                              <TrackAreaContextMenu
                                  x={trackAreaMenu.x}
                                  y={trackAreaMenu.y}
                                  canPaste={clipboardAvailable}
                                  // 与 `splitSelectedAtPlayhead` 共用同一个解析：
                                  // 有选区时看选区，**无选区**时看播放头处有没有
                                  // 可分割的 Clip —— 否则这条菜单项会在"能切但
                                  // 没选东西"时被误判为不可用。
                                  //
                                  // 这里按**未吸附**的播放头判断（与改动前的谓词
                                  // 同口径）：razor 吸附只在执行时发生，把整条吸附
                                  // 管线搬进渲染只为消除"播放头偏离网格半格且跨过
                                  // Clip 边缘"这一种罕见情形的可用性误差，不划算。
                                  canSplit={
                                      resolveSplitTargetsAtSec({
                                          clips: s.clips,
                                          splitSec: Math.max(
                                              0,
                                              Number(sessionRef.current.playheadSec ?? 0) || 0,
                                          ),
                                          selectedIds:
                                              multiSelectedClipIds.length > 0
                                                  ? multiSelectedClipIds
                                                  : s.selectedClipId
                                                    ? [s.selectedClipId]
                                                    : [],
                                      }).ids.length > 0
                                  }
                                  canCloseGaps={s.clips.some(
                                      (c) =>
                                          c.trackId === trackAreaMenu.trackId &&
                                          c.startSec > trackAreaMenu.timeSec + 1e-9,
                                  )}
                                  onCloseGaps={() => {
                                      void dispatch(
                                          closeTrackGapsRemote({
                                              trackId: trackAreaMenu.trackId,
                                              fromSec: trackAreaMenu.timeSec,
                                          }),
                                      );
                                  }}
                                  onPaste={pasteClipsAtPlayhead}
                                  onSplit={splitSelectedAtPlayhead}
                                  onClose={() => setTrackAreaMenu(null)}
                              />,
                              document.body,
                          )
                        : null}

                    <SilenceDetectionDialog
                        open={silenceDialogIds != null}
                        clipIds={silenceDialogIds ?? []}
                        onOpenChange={(open) => {
                            if (!open) setSilenceDialogIds(null);
                        }}
                    />
                    <QuickClipExportDialog
                        open={quickExportDialog.open}
                        clipIds={quickExportDialog.clipIds}
                        onOpenChange={(open) =>
                            setQuickExportDialog((prev) =>
                                open ? prev : { open: false, clipIds: [] },
                            )
                        }
                    />

                    <MidiTrackSelectDialog
                        open={midiClipDialogOpen}
                        onOpenChange={onMidiClipDialogOpenChange}
                        midiPath={midiClipPath}
                        importTarget={importTarget}
                        onImportTargetChange={onImportTargetChange}
                        clipboardGuid={midiClipClipboardGuid ?? null}
                        rootTrackComposeEnabled={midiClipRootTrackComposeEnabled}
                        onRequestEnableCompose={handleRequestEnableCompose}
                        onImportAsClip={handleMidiClipImport}
                        importPosition={importPosition}
                        onImportPositionChange={onImportPositionChange}
                        fillGaps={fillGaps}
                        onFillGapsChange={onFillGapsChange}
                        multiTrackMerge={multiTrackMerge}
                        onMultiTrackMergeChange={onMultiTrackMergeChange}
                        projectBpm={s.bpm}
                        importBpmAsProject={importBpmAsProject}
                        onImportBpmAsProjectChange={onImportBpmAsProjectChange}
                        noteBpmMode={noteBpmMode}
                        onNoteBpmModeChange={onNoteBpmModeChange}
                        specifiedBpm={specifiedBpm}
                        onSpecifiedBpmChange={onSpecifiedBpmChange}
                        closeLeadingGap={closeLeadingGap}
                        onCloseLeadingGapChange={onCloseLeadingGapChange}
                        importTempoMapEnabled={importTempoMapEnabled}
                        onImportTempoMapEnabledChange={onImportTempoMapEnabledChange}
                        importTempoMapTempo={importTempoMapTempo}
                        onImportTempoMapTempoChange={onImportTempoMapTempoChange}
                        importTempoMapTimeSignature={importTempoMapTimeSignature}
                        onImportTempoMapTimeSignatureChange={onImportTempoMapTimeSignatureChange}
                        importTempoMapKeySignature={importTempoMapKeySignature}
                        onImportTempoMapKeySignatureChange={onImportTempoMapKeySignatureChange}
                    />

                    <MidiTrackSelectDialog
                        open={replaceMidiDialog.open}
                        onOpenChange={(open) => {
                            if (!open)
                                setReplaceMidiDialog({
                                    open: false,
                                    clipId: null,
                                    midiPath: null,
                                });
                        }}
                        midiPath={replaceMidiDialog.midiPath}
                        mode="replaceMidi"
                        onImportAsClip={handleReplaceMidiImport}
                        fillGaps={fillGaps}
                        onFillGapsChange={onFillGapsChange}
                        projectBpm={s.bpm}
                        importBpmAsProject={importBpmAsProject}
                        onImportBpmAsProjectChange={onImportBpmAsProjectChange}
                        noteBpmMode={noteBpmMode}
                        onNoteBpmModeChange={onNoteBpmModeChange}
                        specifiedBpm={specifiedBpm}
                        onSpecifiedBpmChange={onSpecifiedBpmChange}
                        closeLeadingGap={closeLeadingGap}
                        onCloseLeadingGapChange={onCloseLeadingGapChange}
                    />

                    <AppDialog
                        open={sameSourceConfirmOpen}
                        onOpenChange={(open) => {
                            setSameSourceConfirmOpen(open);
                            if (!open && sameSourceConfirmResolverRef.current) {
                                sameSourceConfirmResolverRef.current(false);
                                sameSourceConfirmResolverRef.current = null;
                            }
                        }}
                        title={t("ctx_replace")}
                        message={t("clip_replace_same_source_confirm")}
                        size="sm"
                        actions={[
                            {
                                id: "cancel",
                                label: t("cancel"),
                                onClick: () => {
                                    setSameSourceConfirmOpen(false);
                                    if (sameSourceConfirmResolverRef.current) {
                                        sameSourceConfirmResolverRef.current(false);
                                        sameSourceConfirmResolverRef.current = null;
                                    }
                                },
                            },
                            {
                                id: "ok",
                                label: t("ok"),
                                intent: "primary",
                                onClick: () => {
                                    setSameSourceConfirmOpen(false);
                                    if (sameSourceConfirmResolverRef.current) {
                                        sameSourceConfirmResolverRef.current(true);
                                        sameSourceConfirmResolverRef.current = null;
                                    }
                                },
                            },
                        ]}
                    />

                    <TimelineTransportBridge
                        pxPerSecRef={pxPerSecRef}
                        viewport={viewportAccess}
                        visualPlayheadRef={visualPlayheadSecRef}
                        syncScrollLeft={syncScrollLeft}
                        autoScrollEnabled={s.autoScrollEnabled}
                        projectSec={dynamicProjectSec}
                        requestPlayheadRepaint={handleRequestPlayheadRepaint}
                    />

                    {/* 右键播放速率角标 → 高级编辑（倍率 + BPM 换算，批量应用） */}
                    <ClipRateEditorDialog
                        open={rateEditorClipId != null && rateEditorPosition != null}
                        clip={
                            rateEditorClipId
                                ? (s.clips.find((entry) => entry.id === rateEditorClipId) ?? null)
                                : null
                        }
                        tempoMap={s.tempoMap}
                        position={rateEditorPosition}
                        projectBpm={s.bpm}
                        targetCount={
                            rateEditorClipId != null &&
                            multiSelectedClipIds.length > 0 &&
                            multiSelectedSet.has(rateEditorClipId)
                                ? multiSelectedClipIds.length
                                : 1
                        }
                        formatCtx={timeValueFormatCtx}
                        onApply={(rate, adjustLength, durationSec) => {
                            if (rateEditorClipId != null) {
                                commitTrackLaneRate(rateEditorClipId, {
                                    rate,
                                    durationSec: durationSec ?? undefined,
                                    autoLength: adjustLength,
                                });
                            }
                        }}
                        onOpenChange={(o) => {
                            if (!o) setRateEditorClipId(null);
                        }}
                    />

                    <TimelineDisplaySettingsDialog
                        open={timeDisplaySettingsOpen}
                        onOpenChange={setTimeDisplaySettingsOpen}
                    />

                    {/* 音高拖拽悬浮 ToolTips：跟随指针展示 Clip 范围内音高变化量 */}
                    <AppTooltipBubble
                        text={pitchDragTooltip?.text ?? null}
                        position={pitchDragTooltip?.position ?? null}
                    />
                </Flex>
            </Flex>
        </Profiler>
    );
};
