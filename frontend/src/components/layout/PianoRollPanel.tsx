import { PitchSnapSettingsDialog } from "./PitchSnapSettingsDialog";
import React, {
    type CSSProperties,
    useCallback,
    useEffect,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import { flushSync } from "react-dom";
import { Flex, Text, Button, Select, Box, IconButton, DropdownMenu } from "@radix-ui/themes";
import {
    ChevronDownIcon,
    CursorArrowIcon,
    EyeOpenIcon,
    EyeClosedIcon,
    Link2Icon,
    LinkBreak2Icon,
    Pencil1Icon,
    CheckIcon,
} from "@radix-ui/react-icons";

import { shallowEqual } from "react-redux";
import { useAppDispatch, useAppSelector, useAppStore } from "../../app/hooks";
import { resolveSyncOffsetForms } from "../../features/dock/dockSchema";
import { PANEL_PARAM_EDITOR, PANEL_TIMELINE } from "../dock/registerBuiltinPanels";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    setEditParam,
    setEdgeSmoothnessPercent,
    setTrackStateRemote,
    togglePitchSnap,
    setPitchSnapUnit,
    setScaleHighlightMode,
    toggleLockParamLines,
    cycleDragDirection,
    setToolMode,
    persistUiSettings,
    toggleParamAxisUnit,
    setParamEditorSyncTimeline,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setVisibleReferenceRootTrackIds,
    toggleVisibleReferenceRootTrackId,
    createClipsRemote,
    addTrackRemote,
    importMidiAsClip,
    setTempoMap,
    setParamSelectionActive,
} from "../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../features/session/trackUtils";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { getWaveformColors } from "../../theme/waveformColors";
import type { ProcessorParamDescriptor } from "../../types/api";
import { paramsApi } from "../../services/api/params";
import { coreApi } from "../../services/api/core";
import { webApi } from "../../services/webviewApi";
import type { ParamFramesPayload } from "../../types/api";
import {
    degreeInputToScaleSteps,
    isScaleKey,
    resolveScaleNotes,
    SCALE_NOTES,
    snapToScale,
    snapToSemitone,
    transposePitchByScaleSteps,
} from "../../utils/musicalScales";
import {
    measureTimelineViewportOffsetPx,
    timelineViewportSync,
    timelineViewportNativeToState,
    timelineViewportStateToNative,
} from "../../utils/timelineViewportSync";
import { isModifierActive, isNoneBinding } from "../../features/keybindings/keybindingsSlice";
import { useNonPassiveWheel } from "../../utils/useNonPassiveWheel";
import { getActiveSurface, setActiveSurfaceExplicit } from "../../features/uiFocus/focusSurface";
import { findFirstExternalPathAction } from "./timeline/dnd";
import { shiftPitchValue } from "./timeline/clipPitchDrag";
import type { ScaleLike } from "../../utils/musicalScales";
import {
    pasteReaperClipboard,
    pasteVocalShifterClipboard,
} from "../../features/session/thunks/audioThunks";

import {
    BackgroundGrid,
    DEFAULT_PX_PER_SEC,
    MAX_PX_PER_SEC,
    MIN_PX_PER_SEC,
    TimeRuler,
    clamp,
    formatCursorTime,
} from "./timeline";
import { timeRulerHeightPx } from "./timeline/rulerHeight";
import { TempoMapCornerButton } from "./timeline/TempoMapCornerButton";
import { invokeGridRedrawHandler } from "./timeline/gridRedrawBridge";
import { isMirrorEcho } from "./timeline/scrollEcho";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeline";
import type { TempoMap } from "../../utils/tempoMap";
import { effectiveScaleAtSec, buildScaleSegments } from "../../utils/tempoMap";
import { setTempoMapRemote } from "../../features/session/thunks/tempoMapThunks";
import { publishPianoRollSelection } from "../../utils/pianoRollSelectionBus";
import {
    resolveHorizontalWheelZoom,
    resolveTimelineScrollRange,
} from "./timeline/runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "./timeline/runtime/timelineZoomBounds";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";

import {
    AXIS_W,
    PARAM_EDITOR_BOTTOM_BAR_PX,
    PITCH_MAX_MIDI,
    PITCH_MIN_MIDI,
} from "./pianoRoll/constants";
import { drawPianoRoll } from "./pianoRoll/render";
import type { DetectedPitchCurve, ReferencePitchOverlay } from "./pianoRoll/render";
import type { MainCanvasSignature } from "./pianoRoll/mainCanvasSignature";
import {
    buildReferencePitchStrokeColor,
    cleanupVisibleReferenceRootTrackIds,
    listReferenceRootTracks,
} from "./pianoRoll/referenceRootTracks";
import { buildReferenceRootTrackTriggerElement } from "./pianoRoll/referenceRootTrackTrigger";
import {
    averageSelectionValues,
    smoothSelectionValues,
    smoothContextPadFrames,
} from "./pianoRoll/selectionTransforms";
import {
    applySelectionEditOverRanges,
    type SelectionEditExtension,
} from "./pianoRoll/selectionEditApply";
import {
    addFrameRange,
    addFrameRanges,
    frameRangeEnd,
    frameRangeEndCut,
    frameRangeStartCut,
    normalizeSelection,
    selectionBoundingSpan,
    selectionFromFrames,
    selectionToFrameRanges,
    subtractFrameRange,
    toggleFrameRange,
    type FrameRange,
    type ParamSelection,
} from "./pianoRoll/paramSelection";
import {
    clipboardPreviewSpans,
    mapClipboardToTargetRanges,
    pasteTargetSelectionFromClipboard,
    toParamClipboardPayload,
    type ParamClipboardData,
    type ParamClipboardSegment,
} from "./pianoRoll/paramClipboardMapping";
import { uploadFullResCurveSegments } from "./pianoRoll/selectionEditData";
import { planParamConversion } from "./pianoRoll/paramConversion";
import { editablePitchValue } from "./pianoRoll/paramSmoothing";
import {
    DYN_DEFAULT_VIEW,
    DYN_FOLLOW_ORIG,
    DYN_VALUE_MAX,
    dynMultiplicativeFactor,
    isDynParam,
    VOLUME_DEFAULT_VIEW,
    restoreDynSentinels,
} from "./pianoRoll/paramRanges";
import { usePianoRollData } from "./pianoRoll/usePianoRollData";
import { useClipsPeaksForPianoRoll } from "./pianoRoll/useClipsPeaksForPianoRoll";
import { PianoRollWaveformSurface } from "./pianoRoll/PianoRollWaveformSurface";
import { makeLoudnessAmplitudeMap } from "./pianoRoll/PianoRollWaveformSurface";
import { clampParamWriteValue } from "./pianoRoll/paramRanges";
import {
    formatDbReadout,
    resolveParamAxisUnit,
    supportsParamAxisUnit,
} from "./pianoRoll/paramAxisUnits";
import { framesToTime, midiToLabel, timeToFrame } from "./pianoRoll/utils";
import { useLoudnessCurves } from "./pianoRoll/useLoudnessCurves";
import {
    createLiveOverrideReader,
    type LiveOverrideReader,
} from "./pianoRoll/liveLoudnessOverride";
import { pianoRollViewportBus } from "./pianoRoll/pianoRollViewportBus";
import { createRenderLoop, type RenderLoop } from "./renderKernel/renderLoop.js";
import { buildTimelineTicks } from "./timeline/runtime/buildTimelineTicks.js";
import {
    createTimelineAxis,
    playheadLineLeftPx,
    viewportEndSec,
    viewportStartSec,
} from "./renderKernel/timelineAxis.js";
import { usePianoRollInteractions } from "./pianoRoll/usePianoRollInteractions";
import { useLiveParamEditing } from "./pianoRoll/useLiveParamEditing";
import { getParamShiftStep, parseParamShiftMagnitude } from "./pianoRoll/paramShiftStep";
import {
    beginSelectionParamEdit,
    endSelectionParamEdit,
} from "../../features/session/selectionEditInFlight";
import {
    buildChildPitchOffsetCentsParam,
    buildChildPitchOffsetDegreesParam,
    buildChildFormantOffsetCentsParam,
    childPitchOffsetValueToDisplay,
    CHILD_PITCH_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_DEGREES_RANGE,
    CHILD_FORMANT_OFFSET_CENTS_RANGE,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetParam,
    parseChildPitchOffsetParam,
} from "./pianoRoll/childPitchOffsetParams";
import { buildChildOffsetPasteValues as buildChildOffsetPasteValuesHelper } from "./pianoRoll/childPitchOffsetPaste";
import { readSystemClipboardObject, writeSystemClipboardObject } from "../../utils/systemClipboard";
import { getParamEditorWheelAction } from "./pianoRoll/wheelGesture";
import type { Keybinding } from "../../features/keybindings/types";
import { pianoKeySound } from "../../utils/PianoKeySound";
import { computeAutoFollowScrollLeft } from "../../utils/autoFollowScroll";
import { readDevicePixelRatio } from "../../utils/devicePixelLine";
import { useVisualPlayhead } from "../../hooks/useVisualPlayhead";
import {
    getVisibleSecondaryParamIds,
    toggleSecondaryParamVisibility,
} from "./pianoRoll/secondaryOverlaySelection";
import type {
    ParamMorphOverlay,
    ParamName,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "./pianoRoll/types";
import {
    formatKeybinding,
    selectKeybinding,
    selectMergedKeybindings,
} from "../../features/keybindings/keybindingsSlice";

import { usePianoRollStatusUpdate } from "../../contexts/PianoRollStatusContext";
import { MidiTrackSelectDialog } from "./MidiTrackSelectDialog";
import { settingsApi } from "../../services/api/settings";
import { EditContextMenu } from "../editDialogs/EditContextMenu";
import { resolveScrollableProjectSec } from "../../features/session/projectBoundary";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { parseCustomScaleToken } from "../../utils/scaleSelection";
import {
    centerFromVerticalScrollTop,
    verticalScrollTopFromCenter,
} from "./pianoRoll/verticalScrollMapping";
import {
    resolveClipboardPreviewColor,
    resolveDetectedCurveColors,
    resolveSecondaryCurveColor,
} from "./pianoRoll/colors";
import { resolveSecondaryOverlayValues } from "./pianoRoll/secondaryOverlaySelection";
import { getFixedDashPattern } from "./pianoRoll/render";
import type { PianoRollCurveLayer } from "./pianoRoll/kernel/host/pianoRollKernelData";
import type { TimelineAxis } from "./renderKernel/timelineAxis";
import { secToViewportPx } from "./renderKernel/timelineAxis";
import {
    createPianoRollKernelHost,
    type PianoRollKernelHost,
} from "./pianoRoll/kernel/host/pianoRollKernelHost";
import type {
    MutablePianoRollKernelData,
    PianoRollGridSpec,
    PianoRollSelectionBandSpec,
} from "./pianoRoll/kernel/host/pianoRollKernelData";
import { PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX } from "./pianoRoll/kernel/scroll/verticalValueScroll";
import { resolvePanelRenderViewport } from "./pianoRoll/kernel/viewportSource";
import { normalizeCssColor, resolvePianoRollColors } from "./pianoRoll/colors";
import { parseRgbaColor } from "./timeline/runtime/timelineClipGlRenderer";

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX = PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX;

/**
 * **不参与「无选区时先全选」** 的操作（见 `handleEditOp` 开头的隐式全选）。
 *
 * 参数编辑器里的操作绝大多数以**参数选区**为作用域（复制/剪切、初始化、
 * 各类对话框编辑、音量↔动态互转、另存为音高参考、导出 MIDI…）：没有选区时它们
 * 原本静默什么都不做，用户得先自己"全选"再点一次。这些一律先全选再执行。
 *
 * 只有三类例外：
 * - **选区自身的命令**：`selectAll` / `deselect` —— 它们就是在操作选区，
 *   "先全选再执行"会把"取消选择"变成"全选"；
 * - **作用域来自剪辑选择**的命令：`selectClipParamRange` /
 *   `addClipsToParamSelection` / `removeClipsFromParamSelection` —— 它们按被选中的
 *   剪辑（或其起止时间）改写参数选区，与"当前有没有参数选区"无关；
 * - **`paste`** —— 粘贴的作用对象完全由**剪贴板**决定：无选区时按"播放光标作为
 *   复制起点 + 剪贴板自己的段布局"推导选区（见 `pasteTargetSelectionFromClipboard`）。
 *   在这里全选会把数据摊到整条曲线上（起点跑到工程开头、断层被拉长）。
 */
const SELECTION_SCOPE_EXEMPT_OPS: ReadonlySet<string> = new Set([
    "selectAll",
    "deselect",
    "selectClipParamRange",
    "addClipsToParamSelection",
    "removeClipsFromParamSelection",
    "paste",
]);

/**
 * 音频块的时间范围（秒）→ 选区**帧边界**（半开 `[startBound, endBound)`）。
 *
 * 起点向下取整、终点向上取整（沿用改造前 `floor(起点) / ceil(终点)` 的约定）：
 * 块两端**部分覆盖**的帧也算在范围内，不会在边界处悄悄丢掉一帧。
 *
 * 【为什么需要这一层】音频块是秒制（`startSec` / `lengthSec`，与 BPM 无关），
 * 选区是帧制；两者只在"按块选参数范围"这类入口处交界，故集中在此一处换算。
 *
 * 注意结果走**数据路径**（`selectionFromFrames` / `addFrameRanges`），不经过
 * "切点"量化：块的起止是绝对时间，不该被"最近中点"再挪半帧。
 */
function clipTimeRangeToFrameBounds(
    clip: { startSec: number; lengthSec: number },
    framePeriodMs: number,
): { startBound: number; endBound: number } {
    const fp = Math.max(1e-6, framePeriodMs);
    const startBound = Math.max(0, timeToFrame(clip.startSec, fp));
    const endBound = Math.max(
        startBound,
        Math.ceil(((clip.startSec + clip.lengthSec) * 1000) / fp),
    );
    return { startBound, endBound };
}

/**
 * 本面板在共享视口中的来源标识。
 *
 * 与时间轴的 `TIMELINE_SYNC_ORIGIN` 成对：双方发布时登记来源，订阅回调据此
 * 跳过"自己造成的广播"（否则时间轴会把自己的值再应用一遍并回退）。
 */
const PIANO_ROLL_SYNC_ORIGIN = "pianoRoll";

/**
 * 把 `getFixedDashPattern` 的返回值收窄为元组。
 *
 * 特殊说明：该函数返回 `number[]`（历史签名，也被 Canvas2D 的 `setLineDash` 消费），
 * 而曲线图层描述符要求 `[dash, gap]` 二元组。这里做一次显式收窄而不是改它的签名——
 * 改签名会波及 Canvas2D 路径，而那条路径是已充分验证的。长度不符时返回 null
 * （视为实线），宁可退化也不产出非法图案。
 *
 * @param pattern `[dash, gap]`。
 * @returns 二元组；长度不符时为 null。
 */
function toDashTuple(pattern: readonly number[]): [number, number] | null {
    return pattern.length >= 2 ? [pattern[0], pattern[1]] : null;
}

/**
 * 判断数值是否值得写入 DOM（跨过容差）。
 *
 * 【为什么要取反写法】`previous` 为 NaN（从未写入）时 `Math.abs(next - NaN) <= eps`
 * 恒为 false，若正着写会得到「无需写入」，表现为首帧完全不写。取反让 NaN 稳定走
 * 「需要写入」分支（与内核宿主同一写法）。
 *
 * @param next 本次要写入的值。
 * @param previous 上一次写入的值（NaN = 从未写入）。
 * @param epsilon 视为「无变化」的最大差值。
 * @returns 需要写入时为 true。
 */
function shouldWriteNumber(next: number, previous: number, epsilon = 0.01): boolean {
    return !(Math.abs(next - previous) <= epsilon);
}

/** 播放头 RGBA 的解析缓存（键 = 解析后的 CSS 颜色字符串）。 */
const playheadRgbaCache = new Map<string, [number, number, number, number]>();

/**
 * 解析参数编辑器播放头的 GL 颜色（与上方标尺同源）。
 *
 * 流程：按主题取配色表里的 `playheadLine`（它是指向 `--qt-playhead` 的 CSS 变量，
 * 与时间轴标尺同一来源）→ 经 `normalizeCssColor` 借浏览器解析成 `rgb()/rgba()`
 * → `parseRgbaColor` 转 0..1 浮点。
 *
 * 【为什么按「解析后的字符串」做缓存】本函数在每帧的绘制提交路径上被调用，而
 * `normalizeCssColor` 会挂 DOM 探针读 `getComputedStyle`——裸调等于把样式重算拖进
 * 每一帧（与 `resolveThemeColor` 的缓存理由相同）。主题或自定义主题色一变，解析
 * 结果字符串就变，键随之变化、缓存自然失效；反之稳态下每帧命中缓存、零重算。
 *
 * 特殊说明 1：GL 不认 `var(...)`，所以**必须**先归一化；`parseRgbaColor` 只认
 * `rgb()/rgba()`，直接喂变量会被解析成不透明洋红。
 *
 * 特殊说明 2（**为什么不能省掉这一步、让宿主兜底**）宿主在缺省时用一个硬编码
 * `[0,0,0,0.2]`——那是迁移期的占位值，与 `--qt-playhead` 毫无关系。参数编辑器的
 * Canvas2D 播放头已被 `skipPlayhead: true` 永久跳过，GL 是唯一绘制者，因此"不喂"
 * 就等于"画布与标尺两个颜色"（用户报告："播放线在底下和上方标尺的颜色不一致"）。
 *
 * 特殊说明 3：`themeMode` 参与键（经配色表间接体现），但这里额外接收它只是为了
 * 让调用方语义清晰——缓存键用的是解析结果，本身已包含主题信息。
 *
 * @param themeMode 当前主题模式（决定取哪套配色表）。
 * @returns 归一化后的 RGBA（0..1 浮点）。
 */
function resolvePlayheadRgba(themeMode: "dark" | "light"): [number, number, number, number] {
    const css = resolvePianoRollColors(themeMode === "dark").playheadLine;
    const normalized = normalizeCssColor(css);
    const cached = playheadRgbaCache.get(normalized);
    if (cached !== undefined) return cached;
    const parsed = parseRgbaColor(normalized);
    const rgba: [number, number, number, number] = parsed.every((v) => Number.isFinite(v))
        ? [parsed[0], parsed[1], parsed[2], parsed[3]]
        : // 解析失败时退回与标尺同色相的实色（`--qt-playhead` 的深色默认值），
          // 而不是透明/洋红：宁可颜色略有偏差，也不要播放头看不见。
          [0xf0 / 255, 0x5a / 255, 0x5a / 255, 1];
    playheadRgbaCache.set(normalized, rgba);
    return rgba;
}

/**
 * 参数编辑器工具栏的参数显示顺序排名（数值越小越靠左）。
 * - 「音高」为核心参数，固定在最左侧（在 JSX 中单独渲染，不在此排序）；
 * - 「音量/声像」是所有算法的共通参数，固定在最右侧；
 * - 「动态」紧挨音量（两者共用一个药丸呈现，排序只需保证相邻）；
 * - 中间参数随算法不同而变化。
 */
function getParamToolbarRank(paramId: string, algo: string | undefined | null): number {
    switch (algo) {
        case "nsf_hifigan_onnx":
            // 音高、共振峰、气声音量、张力、音量、动态、声像
            switch (paramId) {
                case "formant_shift_cents":
                    return 10;
                case "breath_gain":
                    return 20;
                case "hifigan_tension":
                    return 30;
                case "volume":
                    return 90;
                case "dyn":
                    return 91;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
        case "vslib":
            // 音高、共振峰、气声强度、音量、动态、声像
            switch (paramId) {
                case "formant_shift_cents":
                    return 10;
                case "breathiness":
                    return 20;
                case "volume":
                    return 90;
                case "dyn":
                    return 91;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
        default:
            // world / 其它：仅保证音量/动态/声像在右侧，其余保持后端顺序
            switch (paramId) {
                case "volume":
                    return 90;
                case "dyn":
                    return 91;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
    }
}

function sameStringArray(a: string[], b: string[]) {
    if (a.length !== b.length) return false;
    return a.every((value, index) => value === b[index]);
}

/**
 * “气声/气流”图标：三道向右上方倾斜流动的曲线，表示风/气流（类似 Material “Air” 图标），
 * 避免被误认为汉堡菜单；关闭（off）时气流变淡并叠加一条斜杠。
 * 用于代替冗长的“气声开启/气声关闭”文本。
 */
const BreathAirIcon: React.FC<{ off?: boolean }> = ({ off = false }) => (
    <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ display: "block" }}
    >
        {/* 三道右倾的流动曲线 = 风/气流 */}
        <path
            d="M2.2 3.6C4.8 1.9 8.2 2.5 11 5.2"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            opacity={off ? 0.4 : 1}
        />
        <path
            d="M1.4 6.8C4.6 5.1 8.1 5.9 11.4 8.9"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            opacity={off ? 0.4 : 1}
        />
        <path
            d="M1.9 10.1C4.7 9.3 7.5 9.5 9.9 11.7"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            opacity={off ? 0.4 : 1}
        />
        {off ? (
            <path d="M3 11L11 3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        ) : null}
    </svg>
);

type ParamToolbarPillProps = {
    /** 参数按钮上的简短标签（如 PIT / 共振峰） */
    label: string;
    /** 完整参数名（ToolTip） */
    labelTooltip?: string;
    /** 是否为主参数（激活）：整个药丸整体高亮（统一用全局强调色） */
    active: boolean;
    /** 点击标签：选中该参数 */
    onSelect: () => void;
    /** 眼睛状态：main=主参数（仅展示“睁开”，不响应点击）；on/off=副曲线叠加可见/隐藏 */
    eyeMode: "main" | "on" | "off";
    /** 点击眼睛（非 main 时调用；main 时不响应，以保持排版稳定） */
    onToggleEye?: () => void;
    /** 眼睛 ToolTip（两行：状态 + 点击动作） */
    eyeTooltip?: string;
    /** 眼睛的无障碍标签（简短，如“显示/隐藏副参数叠加曲线”） */
    eyeLabel?: string;
    /** 可选尾部片段（如气声开关）：渲染在参数名之后、子参数下拉之前 */
    trailing?: React.ReactNode;
    /** 可选片段：子参数下拉菜单的触发按钮（已含 param-pill__seg 样式类） */
    dropdown?: React.ReactNode;
};

/**
 * 参数编辑器工具栏的“参数分组药丸”：眼睛 → 参数名 →（气声开关/子参数下拉）。
 * 各片段共享一块连续背景；激活参数统一铺全局强调色（--accent-9），
 * 片段间用细分隔线区分；悬停时只高亮当前片段，提示其独立可点击。
 */
const ParamToolbarPill: React.FC<ParamToolbarPillProps> = ({
    label,
    labelTooltip,
    active,
    onSelect,
    eyeMode,
    onToggleEye,
    eyeTooltip,
    eyeLabel,
    trailing,
    dropdown,
}) => {
    const eyeInert = eyeMode === "main";
    const eyeIcon = eyeMode === "off" ? <EyeClosedIcon /> : <EyeOpenIcon />;
    return (
        <div
            className="param-pill"
            data-active={active ? "true" : undefined}
            data-eye-off={eyeMode === "off" ? "true" : undefined}
        >
            <button
                type="button"
                tabIndex={-1}
                className={
                    eyeInert
                        ? "param-pill__seg param-pill__seg--eye param-pill__seg--inert"
                        : "param-pill__seg param-pill__seg--eye"
                }
                data-tooltip={eyeInert ? undefined : eyeTooltip}
                aria-label={eyeInert ? label : (eyeLabel ?? eyeTooltip)}
                onClick={(e) => {
                    e.stopPropagation();
                    if (!eyeInert) onToggleEye?.();
                }}
            >
                {eyeIcon}
            </button>
            <button
                type="button"
                className="param-pill__seg param-pill__seg--label"
                data-tooltip={labelTooltip}
                onClick={onSelect}
            >
                {label}
            </button>
            {trailing}
            {dropdown}
        </div>
    );
};

/**
 * 「一个药丸 + 下拉切换两个参数」的通用按钮。
 *
 * 两个使用场景：
 * - **共振峰**：根参数（整个轨道组）与子参数（当前子轨道的偏移）。只有在
 *   选中子轨道时才有第二个选项，否则退化为普通药丸。
 * - **音量 / 动态**：两个同量纲的混音级参数。两者**始终**可选，因此
 *   `alwaysShowDropdown = true`。
 *
 * 下拉的样式与交互（RadioGroup + ChevronDown 段）与「音高」参数组完全一致，
 * 保持工具栏里三组参数的观感统一。
 */
type ParamGroupButtonProps = {
    rootParamId: string;
    /** 按钮上的简短标签（如 FRM / 共振峰） */
    rootLabel: string;
    /** 下拉菜单中“根参数”选项的详细说明（如 Formant Shift (Track Group)） */
    rootMenuLabel?: string;
    rootTooltip?: string;
    childParamId: string | null;
    /** 按钮上的简短子参数标签（如 共振峰差） */
    childLabel: string;
    /** 下拉菜单中“子参数”选项的详细说明（如 Formant Offset (Current Sub-track)） */
    childMenuLabel?: string;
    rootActive: boolean;
    childActive: boolean;
    secondaryVisible: boolean;
    /** 眼睛的无障碍标签（简短动作说明） */
    hideSecondaryLabel: string;
    showSecondaryLabel: string;
    /** 眼睛 ToolTip（两行：状态 + 点击动作） */
    hideSecondaryTooltip: string;
    showSecondaryTooltip: string;
    onSelectRoot: () => void;
    onSelectChild: () => void;
    onToggleSecondary: () => void;
    /**
     * `true` = 即使没有子参数也显示下拉（音量/动态这类"恒有两个选项"的组）。
     * 缺省 `false`（共振峰语义：只有子轨道才有第二个选项）。
     */
    alwaysShowDropdown?: boolean;
};

const ParamGroupButton: React.FC<ParamGroupButtonProps> = ({
    rootParamId,
    rootLabel,
    rootMenuLabel,
    rootTooltip,
    childParamId,
    childLabel,
    childMenuLabel,
    rootActive,
    childActive,
    secondaryVisible,
    hideSecondaryLabel,
    showSecondaryLabel,
    hideSecondaryTooltip,
    showSecondaryTooltip,
    onSelectRoot,
    onSelectChild,
    onToggleSecondary,
    alwaysShowDropdown = false,
}) => {
    const eyeMode: "main" | "on" | "off" =
        rootActive || childActive ? "main" : secondaryVisible ? "on" : "off";

    if (!childParamId && !alwaysShowDropdown) {
        return (
            <ParamToolbarPill
                label={rootLabel}
                labelTooltip={rootTooltip}
                active={rootActive}
                onSelect={onSelectRoot}
                eyeMode={eyeMode}
                onToggleEye={onToggleSecondary}
                eyeTooltip={secondaryVisible ? showSecondaryTooltip : hideSecondaryTooltip}
                eyeLabel={secondaryVisible ? showSecondaryLabel : hideSecondaryLabel}
            />
        );
    }

    return (
        <DropdownMenu.Root>
            <ParamToolbarPill
                label={childActive ? childLabel : rootLabel}
                labelTooltip={childActive ? (childMenuLabel ?? childLabel) : rootTooltip}
                active={rootActive || childActive}
                onSelect={onSelectRoot}
                eyeMode={eyeMode}
                onToggleEye={onToggleSecondary}
                eyeTooltip={secondaryVisible ? showSecondaryTooltip : hideSecondaryTooltip}
                eyeLabel={secondaryVisible ? showSecondaryLabel : hideSecondaryLabel}
                dropdown={
                    <DropdownMenu.Trigger
                        className="param-pill__seg param-pill__seg--chev"
                        data-tooltip={childActive ? (childMenuLabel ?? childLabel) : rootTooltip}
                        tabIndex={-1}
                    >
                        <ChevronDownIcon width="12" height="12" />
                    </DropdownMenu.Trigger>
                }
            />
            <DropdownMenu.Content variant="soft" color="gray">
                <DropdownMenu.RadioGroup
                    value={
                        rootActive
                            ? rootParamId
                            : childActive && childParamId
                              ? childParamId
                              : undefined
                    }
                    onValueChange={(value) => {
                        if (value === rootParamId) {
                            onSelectRoot();
                        } else if (value === childParamId) {
                            onSelectChild();
                        }
                    }}
                >
                    <DropdownMenu.RadioItem value={rootParamId}>
                        {rootMenuLabel ?? rootLabel}
                    </DropdownMenu.RadioItem>
                    {childParamId ? (
                        <DropdownMenu.RadioItem value={childParamId}>
                            {childMenuLabel ?? childLabel}
                        </DropdownMenu.RadioItem>
                    ) : null}
                </DropdownMenu.RadioGroup>
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
};

export const PianoRollPanel: React.FC<{
    /**
     * 本窗体在停靠布局里的 id（由 `setPanelRenderer` 注入）。
     *
     * 用于判断"我是否与时间轴上下堆叠" —— 同步偏移只在堆叠时有意义。缺省时
     * 按面板 id 回退查找，保证单独渲染（测试、独立窗口）也能工作。
     */
    dockFormId?: string;
}> = ({ dockFormId }) => {
    const dispatch = useAppDispatch();
    // 事件监听器内同步读取 session（如 selectClipParamRange 的 Clip 查找），
    // 避免闭包快照滞后。
    const store = useAppStore();
    const rafRef = useRef<number | null>(null);
    const visualPlayheadSecRef = useRef(0);
    // 视觉插值播放头读取器（与绘制同源）：缩放锚点必须使用它，不能用 33Hz
    // 轮询的 store 滞后值——播放中缩放以滞后值锚定会让播放头跳变 δ·Δpx。
    const getVisualPlayheadSec = useCallback(() => visualPlayheadSecRef.current, []);
    const rulerPlayheadLineRef = useRef<HTMLDivElement | null>(null);
    const rulerPlayheadHeadRef = useRef<HTMLDivElement | null>(null);
    /**
     * 内核宿主的稳定引用。
     *
     * 【为什么声明在这里】`invalidate`（紧随其后）要把标脏转交宿主，
     * 因此必须在它之前声明。放在下方原来的位置会让 `invalidate` 的闭包引用一个
     * 尚未初始化的 `const`（TDZ）——首次调用即抛 ReferenceError。
     */
    const hostRef = useRef<PianoRollKernelHost | null>(null);
    const drawRef = useRef<() => void>(() => {});
    /**
     * 请求重绘（所有会改变画面的数据/状态变更都经此入口）。
     *
     * 【为什么要转交宿主】曲线层在 GL 上，画面由**两个**渲染循环驱动：
     * 面板自己的 rAF（Canvas2D 细节层）与内核宿主的 rAF（GL 层）。只调度前者会让
     * GL 层永远停留在旧内容上——典型症状是"曲线数据到了但屏幕上不出现，直到滚动
     * 一下才显示"。因此标脏一律交给宿主：宿主的帧提交里会回调 `onFrame`
     * → `applyScrollLayers` → `drawRef.current()`，Canvas2D 与 GL 同帧一起刷新。
     *
     * 特殊说明：`drawRef.current()` 自身不调用 `invalidate()`，所以这条链不会自激。
     * 宿主尚未创建（挂载前 / 卸载后）时退回面板自己的 rAF，避免丢失首次绘制。
     */
    const invalidate = useCallback(() => {
        const host = hostRef.current;
        if (host != null) {
            host.invalidate();
            return;
        }
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            drawRef.current();
        });
    }, []);

    /**
     * 标尺播放头元素（竖线 / 倒三角）的挂载回调。
     *
     * 【为什么挂载时要请求一帧】位置只由内核在帧提交里写，而元素可能在**播放头
     * 静止**时被重建（面板重挂载 / 视图切换）：新节点没有任何 `left`，等价于
     * `left: auto` ⇒ 落在静态位置（左缘 = 工程起始处）。此时若没有任何东西弄脏帧
     * （暂停且不滚动），内核不会提交，元素就**停在工程起始处不动**。元素一出现就
     * 请求一帧，写入器随即按当前视口定位它（去重键含元素身份，见
     * `createPlayheadElementWriter`）。
     */
    const attachRulerPlayheadLine = useCallback((element: HTMLDivElement | null) => {
        rulerPlayheadLineRef.current = element;
        if (element !== null) hostRef.current?.invalidate();
    }, []);
    const attachRulerPlayheadHead = useCallback((element: HTMLDivElement | null) => {
        rulerPlayheadHeadRef.current = element;
        if (element !== null) hostRef.current?.invalidate();
    }, []);
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const s = useAppSelector((state: RootState) => state.session, shallowEqual);
    const effectiveProjectScale = useMemo<ScaleLike>(
        () =>
            s.project.useCustomScale && s.project.customScale
                ? s.project.customScale.notes
                : s.project.baseScale,
        [s.project.baseScale, s.project.customScale, s.project.useCustomScale],
    );
    /**
     * 某秒位置生效的“工程音阶”（受 Tempo Map 音阶变化点影响）。
     * 无 Tempo Map 音阶覆盖时即为工程音阶。
     */
    const projectScaleAtSec = useCallback(
        (sec: number): ScaleLike | undefined =>
            effectiveScaleAtSec(s.tempoMap, sec, effectiveProjectScale),
        [s.tempoMap, effectiveProjectScale],
    );
    /**
     * 将音阶 token 解析为 ScaleLike。
     * `__project__` 在提供 `atSec` 时按该时刻的 Tempo Map 生效音阶解析，
     * 否则使用工程音阶（用于全局场景）。
     */
    const resolveScaleFromToken = useCallback(
        (scaleToken: string, atSec?: number): ScaleLike => {
            if (scaleToken === "__project__") {
                return atSec != null ? (projectScaleAtSec(atSec) ?? "C") : effectiveProjectScale;
            }

            const customScaleId = parseCustomScaleToken(scaleToken);
            if (customScaleId) {
                const preset = s.customScalePresets.find((item) => item.id === customScaleId);
                if (preset) {
                    return preset.notes;
                }
            }

            return isScaleKey(scaleToken) ? scaleToken : "C";
        },
        [effectiveProjectScale, projectScaleAtSec, s.customScalePresets],
    );
    const editParam = s.editParam as ParamName;
    /**
     * `editParam` 的 ref 镜像。
     *
     * 【为什么需要】宿主在**挂载时创建一次**（长生命周期），其回调（`onFrame` /
     * `onScrollTopFrame`）里无法直接读到最新的 `editParam`——闭包捕获的是挂载时的
     * 值，切换参数后回调仍按旧参数换算视口，会出现"切了参数但竖向位置按上一个参数
     * 映射"的错位。ref 每次渲染刷新，回调现读即最新。
     */
    const editParamRef = useRef<ParamName>(editParam);
    editParamRef.current = editParam;
    // pitchSnapOpen 已在顶部工具栏 JSX 内声明和使用，无需重复声明
    // pianoRoll.copy/cut/paste 的复制/剪切/粘贴已由全局路由统一派发到
    // handleEditOp，本地不再需要键位匹配（见 useKeybindings/focusRouting）。
    const prVerticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoRollVerticalZoom"),
    );
    const horizontalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.horizontalZoom"),
    );
    const scrollHorizontalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollHorizontal"),
    );
    const scrollVerticalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollVertical"),
    );
    const scrollbarZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollbarZoom"),
    );
    const pianoKeysVerticalScrollKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoKeysVerticalScroll"),
    );
    const pianoKeysVerticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoKeysVerticalZoom"),
    );
    const paramMorphKb = useAppSelector((state) => selectKeybinding(state, "modifier.paramMorph"));
    // 多选区修饰键（默认 ⌘/Ctrl）：按住拖动追加一段选区，按住点击取消该段
    const paramMultiSelectKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramMultiSelect"),
    );
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    // 边缘平滑度滑块的滚轮步进：React 17+ 的根容器 wheel 监听是 passive，
    // JSX onWheel 里的 preventDefault 无效（伴随干预警告），必须走原生
    // 非 passive 监听（与主画布滚轮路径同模式）。
    const edgeSmoothnessWheelRef = useNonPassiveWheel<HTMLInputElement>((e) => {
        e.preventDefault();
        const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
        const step = fine ? 1 : 5;
        const dir = e.deltaY < 0 ? 1 : -1;
        const next = clamp(Math.round(s.edgeSmoothnessPercent) + dir * step, 0, 100);
        dispatch(setEdgeSmoothnessPercent(next));
        void dispatch(persistUiSettings());
    });
    // 参数选区边缘拉伸的修饰键。与时间轴 clip 边缘的 `modifier.clipStretch`
    // **分离**（两个表面各自独立改绑），只用于选择工具下的选区边缘拉伸。
    const stretchKb = useAppSelector((state) => selectKeybinding(state, "modifier.paramStretch"));
    const vibratoAmplitudeAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.vibratoAmplitudeAdjust"),
    );
    const vibratoFrequencyAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.vibratoFrequencyAdjust"),
    );
    const vibratoDragAmplitudeIncreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragAmplitudeIncrease"),
    );
    const vibratoDragAmplitudeDecreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragAmplitudeDecrease"),
    );
    const vibratoDragFrequencyIncreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragFrequencyIncrease"),
    );
    const vibratoDragFrequencyDecreaseKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.vibratoDragFrequencyDecrease"),
    );
    // 拖动方向循环切换键：拖拽进行中按下可即时切换本次拖拽方向（触控板替代右键）。
    const cycleDragDirectionKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.cycleDragDirection"),
    );
    const mergedKeybindings = useAppSelector(selectMergedKeybindings);
    // 是否按住切换吸附的修饰键（临时切换吸附时用于高亮显示）
    const [snapToggleHeld, setSnapToggleHeld] = useState(false);
    // 仅在参数编辑实际操作期间（选择拖拽/绘制）参与临时吸附视觉切换
    const [snapGestureActive, setSnapGestureActive] = useState(false);
    const [hoveredReferenceRootTrackId, setHoveredReferenceRootTrackId] = useState<string | null>(
        null,
    );

    useEffect(() => {
        const kb = mergedKeybindings["modifier.clipNoSnap"];
        if (!kb) return;
        const onKey = (e: KeyboardEvent) => {
            const active = isModifierActive(kb, e);
            setSnapToggleHeld(active);
        };
        window.addEventListener("keydown", onKey as EventListener);
        window.addEventListener("keyup", onKey as EventListener);
        // also track blur to clear state
        const onBlur = () => setSnapToggleHeld(false);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKey as EventListener);
            window.removeEventListener("keyup", onKey as EventListener);
            window.removeEventListener("blur", onBlur);
        };
    }, [mergedKeybindings]);
    const { mode: themeMode, fontFamily } = useAppTheme();
    const waveformColors = useMemo(() => getWaveformColors(themeMode, "piano-roll"), [themeMode]);

    const effectivePitchSnapVisual =
        snapGestureActive && snapToggleHeld ? !s.pitchSnapEnabled : s.pitchSnapEnabled;

    // MIDI 导入弹窗状态
    const [midiDialogOpen, setMidiDialogOpen] = useState(false);
    const [midiPath, setMidiPath] = useState<string | null>(null);
    const [clipboardGuid, setClipboardGuid] = useState<string | null>(null);
    // 导入位置选项（持久化到软件设置）
    const [importPosition, setImportPosition] = useState<string>("selection");
    // 填补空隙选项（持久化到软件设置）
    const [fillGaps, setFillGaps] = useState<boolean>(false);
    // BPM 选项（持久化到软件设置）
    const [importBpmAsProject, setImportBpmAsProject] = useState(false);
    const [noteBpmMode, setNoteBpmMode] = useState<string>("midi");
    const [specifiedBpm, setSpecifiedBpm] = useState<number>(120);
    const [multiTrackMerge, setMultiTrackMerge] = useState<boolean>(true);
    const [closeLeadingGap, setCloseLeadingGap] = useState<boolean>(true);
    const [importTempoMapEnabled, setImportTempoMapEnabled] = useState(false);
    const [importTempoMapTempo, setImportTempoMapTempo] = useState(true);
    const [importTempoMapTimeSignature, setImportTempoMapTimeSignature] = useState(true);
    const [importTempoMapKeySignature, setImportTempoMapKeySignature] = useState(false);
    const [importTargetReaperClipboard, setImportTargetReaperClipboard] =
        useState<string>("pitchParam");
    const [importTargetParamEditor, setImportTargetParamEditor] = useState<string>("pitchParam");
    const midiDialogSourceRef = useRef<"reaperClipboard" | "paramEditor">("paramEditor");
    // 启动时从设置加载
    useEffect(() => {
        settingsApi.getUiSettings().then((s) => {
            if (s?.midiImportPosition) {
                setImportPosition(s.midiImportPosition);
            }
            if (s?.midiFillGaps != null) {
                setFillGaps(s.midiFillGaps);
            }
            if (s?.midiImportBpmAsProject != null) {
                setImportBpmAsProject(s.midiImportBpmAsProject);
            }
            if (s?.midiNoteBpmMode != null) {
                setNoteBpmMode(s.midiNoteBpmMode);
            }
            if (s?.midiSpecifiedBpm != null) {
                setSpecifiedBpm(s.midiSpecifiedBpm);
            }
            if (s?.midiMultiTrackMerge != null) {
                setMultiTrackMerge(s.midiMultiTrackMerge);
            }
            if (s?.midiCloseLeadingGap != null) {
                setCloseLeadingGap(s.midiCloseLeadingGap);
            }
            if (s?.midiImportAsTempoMap != null) {
                setImportTempoMapEnabled(Boolean(s.midiImportAsTempoMap));
            }
            if (s?.midiImportTempoMapTempo != null) {
                setImportTempoMapTempo(Boolean(s.midiImportTempoMapTempo));
            }
            if (s?.midiImportTempoMapTimeSignature != null) {
                setImportTempoMapTimeSignature(Boolean(s.midiImportTempoMapTimeSignature));
            }
            if (s?.midiImportTempoMapKeySignature != null) {
                setImportTempoMapKeySignature(Boolean(s.midiImportTempoMapKeySignature));
            }
            if (s?.midiImportTargetReaperClipboard != null) {
                setImportTargetReaperClipboard(s.midiImportTargetReaperClipboard);
            }
            if (s?.midiImportTargetParamEditor != null) {
                setImportTargetParamEditor(s.midiImportTargetParamEditor);
            }
        });
    }, []);
    // 记录打开弹窗时的选区（拍数，多段），用于后续计算帧偏移。
    // 注意：MIDI 导入的「选区约束」只有单个时间窗接口，因此用包围区间
    // （首段起点 → 末段终点）—— 见 midiSelArgs。
    const [midiDialogSelection, setMidiDialogSelection] = useState<ParamSelection | null>(null);

    // 右键编辑菜单状态
    const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);
    const [drawToolMenuOpen, setDrawToolMenuOpen] = useState(false);
    const drawToolMenuRef = useRef<HTMLDivElement | null>(null);
    const [pitchSnapMenuOpen, setPitchSnapMenuOpen] = useState(false);
    const pitchSnapMenuRef = useRef<HTMLDivElement | null>(null);
    const [paramValuePreview, setParamValuePreview] = useState<{
        clientX: number;
        clientY: number;
        value: number;
        displayText?: string;
    } | null>(null);
    /**
     * 纵轴标尺的悬浮读数（`弹出展示参数` 在左轴上的形态）。
     *
     * 【为什么与 `paramValuePreview` 分开】那个浮窗挂**曲线画布**的坐标系
     * （`canvasRef` 的 rect），本浮窗挂轴列（`axisWrapRef` 的 rect）。两者坐标系
     * 不同，混用会让浮窗横向偏出 56px 宽的轴列。
     */
    const [axisValuePreview, setAxisValuePreview] = useState<{
        clientX: number;
        clientY: number;
        text: string;
    } | null>(null);

    /**
     * 当前参数的纵轴展示单位（音量 / 动态支持倍率 ↔ dB 切换）。
     *
     * 一个值同时喂给三处：GL 刻度标签（经 `buildGridSpec`）、参数线悬浮浮窗
     * （`formatParamValuePreview`）、纵轴标尺悬浮浮窗。三者必须同源，否则同一位
     * 置会出现"刻度写 −6、浮窗写 0.501"的自相矛盾读数。
     */
    const editParamAxisUnit = resolveParamAxisUnit(s.paramAxisUnits, editParam);
    /** 当前参数的左轴是否可切换展示单位（决定光标与角标）。 */
    const axisUnitToggleAvailable = supportsParamAxisUnit(editParam);

    const formatParamValuePreview = useCallback(
        (value: number): string => {
            if (!Number.isFinite(value)) return "";
            if (editParam === "pitch") {
                const rounded = Math.round(value);
                const pitchClass = ((rounded % 12) + 12) % 12;
                const octave = Math.floor(rounded / 12) - 1;
                const noteName = `${NOTE_NAMES_SHARP[pitchClass]}${octave}`;
                const cents = Math.round((value - rounded) * 100);
                const signedCents = cents >= 0 ? `+${cents}` : `${cents}`;
                return `${noteName}${signedCents}`;
            }
            if (isChildPitchOffsetDegreesParam(editParam)) {
                const display = childPitchOffsetValueToDisplay(editParam, value);
                if (Math.abs(display) >= 100) return display.toFixed(1);
                if (Math.abs(display) >= 10) return display.toFixed(2);
                return display.toFixed(3);
            }
            // 音量 / 动态切到 dB 读法：1× = 0 dB（见 paramAxisUnits）。带 `dB` 后缀
            // 与倍率读数区分——浮窗有空间写清楚单位，用户才不会把 −6 误读成倍率。
            if (editParamAxisUnit === "db" && supportsParamAxisUnit(editParam)) {
                return formatDbReadout(value);
            }
            if (Math.abs(value) >= 100) return value.toFixed(1);
            if (Math.abs(value) >= 10) return value.toFixed(2);
            return value.toFixed(3);
        },
        [editParam, editParamAxisUnit],
    );

    const currentDrawTool = s.drawToolMode === "line" ? "vibrato" : s.drawToolMode;
    const drawToolButtonTitle =
        currentDrawTool === "vibrato" ? tAny("vibrato_draw_tool") : tAny("draw_tool");
    const activeDragDirection =
        s.toolMode === "select"
            ? s.selectDragDirection
            : currentDrawTool === "draw"
              ? s.drawDragDirection
              : s.lineVibratoDragDirection;
    const activeDragDirectionTool =
        s.toolMode === "select"
            ? ("select" as const)
            : currentDrawTool === "draw"
              ? ("draw" as const)
              : ("vibrato" as const);

    useEffect(() => {
        if (!drawToolMenuOpen && !pitchSnapMenuOpen) return;
        const onPointerDown = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (drawToolMenuRef.current?.contains(target)) return;
            if (pitchSnapMenuRef.current?.contains(target)) return;
            setDrawToolMenuOpen(false);
            setPitchSnapMenuOpen(false);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setDrawToolMenuOpen(false);
                setPitchSnapMenuOpen(false);
            }
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [drawToolMenuOpen, pitchSnapMenuOpen]);

    /** 打开“导入到参数编辑器”的 MIDI 导入对话框（编辑器按钮 / 拖放到编辑器内共用）。
     *  midiPath 为 null 时由用户在文件选择器中挑选文件；非 null 时直接导入该文件。 */
    const openParamEditorMidiImport = useCallback(
        (midiPath: string | null) => {
            midiDialogSourceRef.current = "paramEditor";
            // 快照当前选区（拍为单位）
            const sel = selectionRef.current;
            setMidiDialogSelection(sel ? sel.map((range) => ({ ...range })) : null);
            // 快照当前的 editParam 和 toolMode，保证异步加载轨道期间 selectionAvailable 不变
            midiDialogOpenParamsRef.current = {
                editParam: s.editParam,
                toolMode: s.toolMode,
            };
            setMidiPath(midiPath);
            setClipboardGuid(null);
            setMidiDialogOpen(true);
        },
        [s.editParam, s.toolMode],
    );

    const handleOpenMidiDialog = useCallback(() => {
        openParamEditorMidiImport(null);
    }, [openParamEditorMidiImport]);

    // ── MIDI 拖放到参数编辑器（文件浏览器拖拽 + Tauri 系统文件拖放）────────
    // 与“导入 MIDI”按钮同属参数编辑器场景：导入目标默认 Pitch Param，并持久化
    // 到 midiImportTargetParamEditor（与按钮共用同一设置项）。
    const paramEditorRef = useRef<HTMLDivElement | null>(null);
    const [paramEditorMidiDragOver, setParamEditorMidiDragOver] = useState(false);

    const isPointOverParamEditor = useCallback((clientX: number, clientY: number) => {
        const el = paramEditorRef.current;
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        return (
            clientX >= rect.left &&
            clientX <= rect.right &&
            clientY >= rect.top &&
            clientY <= rect.bottom
        );
    }, []);

    useEffect(() => {
        // 从拖拽载荷中取第一个 MIDI 文件路径（无则 null）。
        const firstMidiPath = (
            paths: string[] | null | undefined,
            primary: string | null | undefined,
        ): string | null => {
            const all = Array.isArray(paths) && paths.length > 0 ? paths : primary ? [primary] : [];
            const found = findFirstExternalPathAction(all);
            return found && found.kind === "importMidi" ? found.path : null;
        };

        // 文件浏览器面板的自定义拖拽事件（无 leave/end 事件，drop 即结束）。
        const onHifiFileDrag = (e: Event) => {
            const detail = (e as CustomEvent).detail as {
                type?: string;
                filePath?: string;
                filePaths?: string[];
                clientX?: number;
                clientY?: number;
            } | null;
            if (!detail) return;
            const clientX = Number(detail.clientX);
            const clientY = Number(detail.clientY);
            const x = Number.isFinite(clientX) ? clientX : undefined;
            const y = Number.isFinite(clientY) ? clientY : undefined;
            const over = x !== undefined && y !== undefined && isPointOverParamEditor(x, y);
            if (detail.type === "start" || detail.type === "move") {
                setParamEditorMidiDragOver(
                    Boolean(over && firstMidiPath(detail.filePaths, detail.filePath)),
                );
                return;
            }
            if (detail.type === "drop") {
                const midiPath = firstMidiPath(detail.filePaths, detail.filePath);
                setParamEditorMidiDragOver(false);
                if (over && midiPath) {
                    openParamEditorMidiImport(midiPath);
                }
            }
        };
        window.addEventListener("hifi-file-drag", onHifiFileDrag);

        // Tauri 系统文件拖放：与时间轴的 useTimelineDragDrop 各自独立监听，
        // 按坐标区域互斥处理（落在参数编辑器内的 MIDI 才在此导入）。
        // ⚠️ 必须先持有 Window 实例再调用方法：onDragDropEvent 依赖 `this`
        // （内部 this.listen），直接取方法引用会丢失绑定并静默失效。
        let disposed = false;
        let unlisten: null | (() => void) = null;
        void import("@tauri-apps/api/window")
            .then((mod) => mod.getCurrentWindow())
            .then((win) =>
                win.onDragDropEvent((event: unknown) => {
                    if (disposed) return;
                    const payload = (
                        event && typeof event === "object" && "payload" in event
                            ? (event as { payload?: unknown }).payload
                            : event
                    ) as
                        | {
                              type?: string;
                              event?: string;
                              paths?: string[];
                              position?: { x?: number; y?: number };
                              pos?: { x?: number; y?: number };
                              cursorPosition?: { x?: number; y?: number };
                          }
                        | undefined;
                    if (!payload) return;
                    const type = String(payload.type ?? payload.event ?? "");
                    const paths: string[] = Array.isArray(payload.paths) ? payload.paths : [];
                    const pos = (payload.position ?? payload.pos ?? payload.cursorPosition) as
                        | { x?: number; y?: number }
                        | undefined;
                    const dpr = window.devicePixelRatio || 1;
                    const clientX = typeof pos?.x === "number" ? pos.x / dpr : undefined;
                    const clientY = typeof pos?.y === "number" ? pos.y / dpr : undefined;
                    const over =
                        clientX !== undefined &&
                        clientY !== undefined &&
                        isPointOverParamEditor(clientX, clientY);
                    if (type === "enter" || type === "over") {
                        setParamEditorMidiDragOver(Boolean(over && firstMidiPath(paths, null)));
                        return;
                    }
                    if (type === "leave") {
                        setParamEditorMidiDragOver(false);
                        return;
                    }
                    if (type === "drop") {
                        const midiPath = firstMidiPath(paths, null);
                        setParamEditorMidiDragOver(false);
                        if (over && midiPath) {
                            openParamEditorMidiImport(midiPath);
                        }
                    }
                }),
            )
            .then((fn) => {
                if (disposed) {
                    // 卸载竞态兜底：注册完成后组件已卸载，立即解绑。
                    fn();
                    return;
                }
                unlisten = fn;
            })
            .catch((err) => {
                console.warn(
                    "[param-editor-midi-drop] Failed to attach Tauri drag-drop listener",
                    err,
                );
            });

        return () => {
            disposed = true;
            window.removeEventListener("hifi-file-drag", onHifiFileDrag);
            if (unlisten) unlisten();
        };
    }, [isPointOverParamEditor, openParamEditorMidiImport]);

    const effectiveSelectedTrackId = useMemo(() => {
        if (s.selectedTrackId) return s.selectedTrackId;
        const clipId = s.selectedClipId;
        if (!clipId) return null;
        const clip = s.clips.find((c) => c.id === clipId);
        return clip?.trackId ?? null;
    }, [s.selectedTrackId, s.selectedClipId, s.clips]);

    const selectedTrack = useMemo(() => {
        if (!effectiveSelectedTrackId) return null;
        return s.tracks.find((track) => track.id === effectiveSelectedTrackId) ?? null;
    }, [effectiveSelectedTrackId, s.tracks]);

    const selectedIsChildTrack = Boolean(selectedTrack?.parentId);

    const childPitchOffsetCentsParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        return buildChildPitchOffsetCentsParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack]);

    const childPitchOffsetDegreesParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        return buildChildPitchOffsetDegreesParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack]);

    // 可滚域时长必须与时间轴内核**同源**：两处各取一个来源时，内容宽 / 可滚上限
    // 会分叉，共享 scrollLeft 的同步会被浏览器钳制、两边永久错位（缺陷 7）。
    // 详见 `resolveScrollableProjectSec` 的说明。
    const dynamicProjectSec = useMemo(
        () => resolveScrollableProjectSec(s.projectSec, s.clips),
        [s.projectSec, s.clips],
    );
    const [scrollLeft, setScrollLeft] = useState(0);
    const [pxPerSec, setPxPerSec] = useState(() => {
        const stored = Number(localStorage.getItem("hifishifter.paramPxPerSec"));
        return Number.isFinite(stored) && stored > 0
            ? Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, stored))
            : DEFAULT_PX_PER_SEC;
    });
    // 渲染时根 ?BPM 换算 pxPerBeat：pxPerBeat = pxPerSec × (60 / bpm)
    const pxPerBeat = pxPerSec * (60 / Math.max(1e-6, s.bpm));
    const scrollLeftRef = useRef(scrollLeft);
    const pxPerBeatRef = useRef(pxPerBeat);
    const pxPerSecRef = useRef(pxPerSec);
    // 同步开关的 ref 镜像：rAF 原子提交与宿主帧回调读取最新值，避免陈旧闭包。
    const paramEditorSyncTimelineRef = useRef(s.paramEditorSyncTimeline);
    // 渲染期把 state 同步进 ref。**缩放**照旧无条件同步（它是原子提交的，state 就是
    // 真值）；**横向位置**只在宿主尚未挂载时才同步。
    //
    // 【为什么位置必须有条件（这是一个真实缺陷的根因）】横向位置的真值在**内核**，
    // 而 React state 只是 256px 步长的**量化提交**（见宿主 `SCROLL_COMMIT_STEP_PX`）：
    // 滚轮 / 拖 thumb / 触摸 / 自动滚屏只改内核与原生镜像，state 会滞后最多 255px
    // （最后一次不足一步的位移永不提交）。内核挂载后 `scrollLeftRef` 由 `onFrame`
    // 每帧对齐内核真值；若无条件用 state 回写，每一次 React 重渲都会把滞后值**灌回**
    // ref，于是"渲染按内核、交互按 refs"两套视口并存——框选出的选区因此与鼠标划过
    // 的区域相差同一距离（用户报告"实际产生的选区与光标划定的区域不一致"，且先做
    // 一次缩放后滚动才出现：缩放经 flushSync 原子对齐 state，滚动又让它重新滞后；
    // 再缩放一次又对齐，于是"缩放后恢复正常"）。
    //
    // 宿主未挂载时（挂载期）没有内核真值，ref 必须由 state 供给，此时回写是唯一
    // 正确行为。缩放 / pxPerBeat 与滚动无关（由 zomm 的原子提交直接写入 ref），
    // 保持无条件同步以免影响既有缩放路径。
    if (hostRef.current === null) {
        if (scrollLeftRef.current !== scrollLeft) scrollLeftRef.current = scrollLeft;
    }
    if (pxPerBeatRef.current !== pxPerBeat) pxPerBeatRef.current = pxPerBeat;
    if (pxPerSecRef.current !== pxPerSec) pxPerSecRef.current = pxPerSec;
    paramEditorSyncTimelineRef.current = s.paramEditorSyncTimeline;
    const timelineSyncApplyingRef = useRef(false);
    const timelineOffsetRef = useRef(0);
    const [timelineOffsetPx, setTimelineOffsetPx] = useState(0);

    /**
     * 同步偏移是否适用 —— 判据是**两个面板都可见**。
     *
     * 【为什么不是"必须上下堆叠"】偏移 = 轨道区左缘 − 参数编辑器绘制区左缘，把参数
     * 编辑器的内容按它平移后，同一时刻会落在**同一个屏幕 x** 上。只要两个面板同时
     * 可见，这个对齐就有意义（上下相邻是主场景，并排或一个浮在另一个之上同样成立：
     * 偏移可正可负，负值由 `minScrollLeft = -offset` 兜住）。
     *
     * 早期实现把第一个参数写成了 `dockFormId`（本面板自己的窗体 id），两个参数于是
     * 是同一个窗体 —— 判定必然为假、偏移被强制为 0，"同步时间轴视图"的像素对齐因此
     * **整个失效**。这里改为分别取时间轴与本面板的窗体 id，并只要求"都可见"。
     */
    const dockLayout = useAppSelector((state) => state.dock.layout);
    const syncOffsetApplicable = useMemo(
        () =>
            resolveSyncOffsetForms(dockLayout, PANEL_TIMELINE, PANEL_PARAM_EDITOR, dockFormId) !==
            null,
        [dockFormId, dockLayout],
    );

    // 待落地的同步视口：**只记缩放**。位置在落地时直接取共享视口的当前值（权威且最新），
    // 不再捕获快照——见下方落地 effect 的说明（捕获值 + 比对 React state 的老做法会在
    // state 被同期写入点覆盖时静默取消落地，造成随机错位）。
    const pendingParamSyncViewportRef = useRef<{ pxPerSec: number } | null>(null);
    const horizontalZoomPendingRef = useRef<{
        nextScale: number;
        nextScrollLeft: number;
    } | null>(null);
    const horizontalZoomChainRef = useRef<{
        nextPxPerSec: number;
        nextScrollLeft: number;
    } | null>(null);
    // 水平缩放提交的 rAF 合并：一帧内多次滚轮/快捷键缩放只做一次原子提交。
    const zoomRafRef = useRef<number | null>(null);

    // 测量轨道时间线区与参数编辑器画布区之间的全局水平偏移，
    // 用于同步时把参数编辑器的绘制坐标与轨道视图按同一屏幕位置对齐。
    //
    // 【为什么必须容错 + 重试（这是一个真实缺陷的根因）】
    // 偏移是两个视口元素左缘之差，而**两个面板的挂载顺序不固定**：时间轴的内核容器
    // 可能晚于参数编辑器出现（面板按需挂载、WebGL 初始化、加载顺序）。元素缺失时
    // `measureTimelineViewportOffsetPx()` 返回 `null`（旧实现返回 0，与"真的对齐"不可
    // 区分）——若把 0 当成测量值，参数编辑器就会**丢掉整段同步位移**，错位量恰好等于
    // 偏移本身（= 轨道头宽度 − 键盘列宽度 ≈ 轨道头区域宽度）；更糟的是那一刻观察器也
    // 没绑上时间轴元素，之后再没有重测时机，只有恰好一次布局尺寸变化才会恢复——表现
    // 为**随机**错位（实测复现：偏移恒为 0，两面板相差整整 200px，并持续存在）。
    //
    // 因此这里：测不到就保持上一次的有效值，并在后续帧重试（时间轴元素可能刚出现）；
    // 每次重测都重新绑定观察器，元素后出现时也能补上。
    useLayoutEffect(() => {
        // 时间轴不可见（或本面板不可见）：偏移没有意义，固定为 0 并停止重试/观察。
        if (!syncOffsetApplicable) {
            timelineOffsetRef.current = 0;
            setTimelineOffsetPx((prev) => (prev === 0 ? prev : 0));
            return;
        }

        /** 视口元素缺失时的重试上限（帧）；超过后退回低频轮询，避免长期每帧空转。 */
        const RETRY_FRAMES = 120;
        /** 低频轮询间隔（ms）。 */
        const RETRY_INTERVAL_MS = 500;

        let observer: ResizeObserver | null = null;
        let frame: number | null = null;
        let retryTimer: ReturnType<typeof setInterval> | null = null;
        let stopped = false;

        const stopRetry = () => {
            if (frame !== null) {
                cancelAnimationFrame(frame);
                frame = null;
            }
            if (retryTimer !== null) {
                clearInterval(retryTimer);
                retryTimer = null;
            }
        };

        /** 把两边的视口元素（重新）挂到观察器上：元素可能后于本面板出现。 */
        const observeViewports = () => {
            if (typeof ResizeObserver === "undefined") return;
            observer = observer ?? new ResizeObserver(() => measureAndApply());
            observer.disconnect();
            const piano = scrollerRef.current;
            if (piano) observer.observe(piano);
            const track = document.querySelector<HTMLElement>("[data-timeline-scroller]");
            if (track) observer.observe(track);
        };

        /** 采用一次有效测量（并停止重试）。 */
        const applyMeasured = (value: number) => {
            stopRetry();
            timelineOffsetRef.current = value;
            setTimelineOffsetPx((prev) => (Math.abs(prev - value) < 0.5 ? prev : value));
            observeViewports();
        };

        const scheduleRetry = () => {
            if (stopped || frame !== null || retryTimer !== null) return;
            let attempts = 0;
            const tick = () => {
                frame = null;
                if (stopped) return;
                const next = measureTimelineViewportOffsetPx();
                if (next !== null) {
                    applyMeasured(next);
                    return;
                }
                observeViewports();
                attempts += 1;
                if (attempts >= RETRY_FRAMES) {
                    retryTimer = setInterval(measureAndApply, RETRY_INTERVAL_MS);
                } else {
                    frame = requestAnimationFrame(tick);
                }
            };
            frame = requestAnimationFrame(tick);
        };

        function measureAndApply(): void {
            if (stopped) return;
            const next = measureTimelineViewportOffsetPx();
            if (next !== null) {
                applyMeasured(next);
                return;
            }
            // 尚不可测（视口元素未挂载）：保持上一次的有效偏移并重试——**绝不能**当成 0。
            observeViewports();
            scheduleRetry();
        }

        measureAndApply();
        if (typeof ResizeObserver === "undefined") {
            window.addEventListener("resize", measureAndApply);
        }
        return () => {
            stopped = true;
            stopRetry();
            observer?.disconnect();
            window.removeEventListener("resize", measureAndApply);
        };
        // 【为什么要依赖布局】停靠重排会把面板的 DOM 宿主搬到别处，而**尺寸可能
        // 完全不变** —— 那样 ResizeObserver 不会触发，本 effect 若只挂载时跑一次，
        // 偏移就永远停在旧布局测出的值（实测：两面板相差数百像素，且只在恰好发生
        // 一次尺寸变化时才自愈）。因此布局一变就重测。
    }, [syncOffsetApplicable, dockLayout]);

    // BPM 变化时，按比例调 ?scrollLeft，保持视口中心点的秒数不 ?
    // scrollLeft_new = scrollLeft_old × (bpm_old / bpm_new)
    const prevBpmRef = useRef(s.bpm);
    useEffect(() => {
        const prevBpm = prevBpmRef.current;
        prevBpmRef.current = s.bpm;
        if (s.paramEditorSyncTimeline) return;
        if (Math.abs(prevBpm - s.bpm) < 1e-9) return;
        const ratio = prevBpm / Math.max(1e-6, s.bpm);
        const newScrollLeft = scrollLeftRef.current * ratio;
        // 先按绘制坐标把位置交给内核（它负责换算原生坐标并镜像回写），再同步面板
        // state。宿主尚未创建（未挂载）时 `applyHorizontalScrollPosition` 是空操作，
        // 仍同步 state，与迁移前无宿主时的收尾一致。
        applyHorizontalScrollPosition(newScrollLeft);
        scrollLeftRef.current = newScrollLeft;
        lastScrollLeftRef.current = newScrollLeft;
        setScrollLeft(newScrollLeft);
    }, [s.bpm, s.paramEditorSyncTimeline]);

    useEffect(() => {
        const timer = setTimeout(() => {
            localStorage.setItem("hifishifter.paramPxPerSec", String(pxPerSec));
        }, 500);
        return () => clearTimeout(timer);
    }, [pxPerSec]);

    // 同步开关（双向交互）：订阅共享视口并应用到本面板。
    // 原生滚动位置 = 共享视口值（轨道坐标）；绘制坐标 = 原生 - 左右偏移。
    useLayoutEffect(() => {
        if (!s.paramEditorSyncTimeline) return;
        horizontalZoomPendingRef.current = null;
        horizontalZoomChainRef.current = null;
        const applyViewport = () => {
            // 未播种（时间轴尚未把当前位置写入共享视口）时拒绝应用：
            // 模块默认 {0,150} 只是占位，提前应用会把一帧错误缩放/位置
            // 画出来再被纠正（启动"一闪"）。时间轴在挂载/切换的 layout
            // effect（首帧绘制前）播种，播种后 emit 会驱动本订阅应用。
            if (!timelineViewportSync.isSeeded()) return;
            // 【跳过自己发布的广播】参数编辑器既发布又订阅，把刚发布的值再应用一遍
            // 没有意义，却会在缩放事务中间（pending 尚未落地）把位置/缩放拉回上一份
            // 快照——表现就是「水平缩放时波形/参数线抽搐一帧」。时间轴侧早就有同一条
            // 来源判定（`getOrigin() === TIMELINE_SYNC_ORIGIN`），这里补齐对称的一半。
            if (timelineViewportSync.getOrigin() === PIANO_ROLL_SYNC_ORIGIN) return;
            const store = timelineViewportSync.get();
            const offset = timelineOffsetRef.current;
            const drawingScrollLeft = timelineViewportNativeToState(store.scrollLeft, offset);
            // 纯滚动（pxPerSec 未变）：在同一个事件帧内同步落地——原生
            // scroller、标尺/网格层（applyScrollLayers）与轨道视图同帧提交，
            // 两个面板严丝合缝。state 仅作事后对齐（React 在绘制前提交）。
            //
            // 【判据必须取内核真值，不能取渲染期 ref】`pxPerSecRef` 在渲染期就被同步成
            // state 的新值，而并发渲染可能"渲染了但被丢弃"：用 ref 判定会把一次**缩放**
            // 广播误判成纯滚动（于是只应用位置、不应用缩放）；反过来也会把纯滚动误判成
            // 缩放而走延迟落地。内核才是这台面板的视口真值（与 `resolvePanelRenderViewport`
            // 同一约定）。
            const kernelPxPerSec = hostRef.current?.getViewport().pxPerSec ?? pxPerSecRef.current;
            const scroller = scrollerRef.current;
            if (scroller && Math.abs(store.pxPerSec - kernelPxPerSec) <= 1e-9) {
                timelineSyncApplyingRef.current = true;
                pxPerSecRef.current = store.pxPerSec;
                scrollLeftRef.current = drawingScrollLeft;
                lastScrollLeftRef.current = drawingScrollLeft;
                // 同步落地走统一载体（宿主；它会换算原生坐标并镜像回写），
                // 并在**同一任务**里提交各图层（DOM / Canvas2D / GL）——否则 GL
                // 侧的曲线 / 播放头会比标尺、网格慢一帧到几帧（见 `paintNow`）。
                commitViewportNow(drawingScrollLeft);
                setScrollLeft(drawingScrollLeft);
                timelineSyncApplyingRef.current = false;
                return;
            }
            // 缩放（pxPerSec 变化）：内容宽度必须先按新 pxPerSec 重排，维持
            // “先提交 state，再由 layout effect 落地”的既有路径。
            timelineSyncApplyingRef.current = true;
            pendingParamSyncViewportRef.current = { pxPerSec: store.pxPerSec };
            setScrollLeft(drawingScrollLeft);
            setPxPerSec(store.pxPerSec);
            timelineSyncApplyingRef.current = false;
        };
        const unsubscribe = timelineViewportSync.subscribe(applyViewport);
        // 启用瞬间以轨道视图当前值为基准：原生位置对齐共享视口。
        applyViewport();
        const scroller = scrollerRef.current;
        return () => {
            unsubscribe();
            pendingParamSyncViewportRef.current = null;
            horizontalZoomPendingRef.current = null;
            horizontalZoomChainRef.current = null;
            // 禁用时移除偏移补偿：把位置还原为绘制坐标。
            if (scroller) {
                const next = Math.max(0, scrollLeftRef.current);
                scrollLeftRef.current = next;
                lastScrollLeftRef.current = next;
                commitViewportNow(next);
                setScrollLeft(next);
            }
        };
    }, [s.paramEditorSyncTimeline]);

    // 布局偏移变化时，同一共享视口对应的绘制坐标也会变化。
    // 重新按当前偏移计算状态，让同步对齐始终使用最新几何位置。
    useEffect(() => {
        if (!s.paramEditorSyncTimeline) return;
        const store = timelineViewportSync.get();
        const drawingScrollLeft = timelineViewportNativeToState(
            store.scrollLeft,
            timelineOffsetRef.current,
        );
        timelineSyncApplyingRef.current = true;
        pendingParamSyncViewportRef.current = { pxPerSec: store.pxPerSec };
        setScrollLeft(drawingScrollLeft);
        timelineSyncApplyingRef.current = false;
    }, [timelineOffsetPx, s.paramEditorSyncTimeline]);

    // 同步视口必须等内容宽度按新 pxPerSec 更新后再落到 DOM。
    // 否则设置 scroller.scrollLeft 时会被浏览器钳回旧的最大滚动位置，
    // 形成“缩放已变、滚动没变”的水平漂移。
    //
    // 【为什么位置取"共享视口的当前值"而不是捕获时的快照，且不再比对 React state】
    // 这条落地路径原先要求 `React scrollLeft state` 恰好等于 pending 的目标值，否则
    // **直接 return 把 pending 搁置**。而 state 是异步提交的：同期可能被其它写入点
    // （量化提交的 rAF、纯滚动广播、以及参数编辑器自己发布的回声）覆盖成更旧的值，
    // 于是这次落地被静默取消——面板停在旧位置、缩放却已生效，两个面板从此错开一段
    // 距离（用户报告：「参数编辑器的偏移量有误」，且**随机**出现）。位置本身在共享
    // 视口里就是权威且最新的，直接取它即可；只要该视口的**缩放**与本次 pending 一致
    // （说明这是"同一个缩放代"的落地），落地就是对的。缩放若已被更新的广播取代，
    // 那份广播会写下新的 pending，由它落地。
    useLayoutEffect(() => {
        const pending = pendingParamSyncViewportRef.current;
        if (!pending || !s.paramEditorSyncTimeline) return;
        if (Math.abs(pxPerSec - pending.pxPerSec) > 1e-9) return;
        if (Math.abs(timelineOffsetPx - timelineOffsetRef.current) > 0.5) return;

        const store = timelineViewportSync.get();
        // 更新的缩放已在路上：等它自己的 pending（否则会用旧缩放的位置落地）。
        if (Math.abs(store.pxPerSec - pending.pxPerSec) > 1e-9) return;

        pendingParamSyncViewportRef.current = null;
        const scroller = scrollerRef.current;
        if (!scroller) return;

        const offset = timelineOffsetRef.current;
        const drawingScrollLeft = timelineViewportNativeToState(store.scrollLeft, offset);
        timelineSyncApplyingRef.current = true;
        pxPerSecRef.current = pending.pxPerSec;
        pxPerBeatRef.current = pending.pxPerSec * (60 / Math.max(1e-6, s.bpm));
        scrollLeftRef.current = drawingScrollLeft;
        // `store.scrollLeft` 是共享视口的原生值；换算成绘制坐标后走统一载体
        // （宿主会再加回偏移，落到与共享视口相同的位置），并在**同一任务**里把
        // DOM / Canvas2D / GL 一起提交（见 `paintNow`）。
        //
        // 【这里不能再调 syncScrollLeft】它是「读原生 scroller → 采纳为真值」的
        // 路径，而此时原生镜像还停留在**上一帧的旧值**（本函数的写入要等内核下一帧才
        // 回写），于是会把刚提交的目标位置又覆盖回旧值——表现为「同步从 1200 拨回 0
        // 时参数编辑器不动」。
        // 缩放与位置一起落进内核再绘制（见 `commitViewportNow`）：否则同帧绘制会用
        // 内核的旧缩放画新位置，而 DOM 侧标尺/网格已按新缩放排好——一帧内两套缩放。
        commitViewportNow(drawingScrollLeft, pending.pxPerSec);
        timelineSyncApplyingRef.current = false;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pxPerSec, scrollLeft, s.paramEditorSyncTimeline, timelineOffsetPx]);

    useLayoutEffect(() => {
        const pending = horizontalZoomPendingRef.current;
        if (!pending) return;
        // 【先消费再判定】请求一旦"过期"就必须丢弃，不能留在 ref 里等未来某次渲染
        // 的 state 凑巧等于它的缩放——那会用一次**旧缩放的位置**落地（旧快照的
        // `nextScrollLeft`），并在同步模式下把它推回共享视口：两个面板一起跳到旧位置。
        // 因此这里无条件清空，只有"当前 state 正是这次请求的缩放"才真正落地。
        horizontalZoomPendingRef.current = null;
        if (Math.abs(pending.nextScale - pxPerSec) > 1e-9) return;
        horizontalZoomChainRef.current = null;
        const scroller = scrollerRef.current;
        if (!scroller) return;

        const syncEnabled = s.paramEditorSyncTimeline;
        const offset = syncEnabled ? timelineOffsetRef.current : 0;
        const native = pending.nextScrollLeft;
        const next = timelineViewportNativeToState(native, offset);
        // 缩放落地：缩放 + 位置一次原子写入内核，再同任务提交各图层（见 `commitViewportNow`）。
        commitViewportNow(next, pending.nextScale);
        if (lastScrollLeftRef.current !== next) {
            lastScrollLeftRef.current = next;
            scrollLeftRef.current = next;
        }
        // 同步模式下手动缩放后必须把新的 pxPerSec 写回共享视口；即使滚动位置
        // 没有变化（例如光标位于左侧同步空白区时锚定在工程起点，next 仍为 -offset），
        // 也要广播缩放，否则轨道视图不会跟着缩放。
        //
        // 【广播的是"落地值"而不是"请求值"】`applyHorizontalScrollPosition` 已经
        // 把目标交给内核（内核按当前边界钳制），所以这里回读一次：共享视口里只能是
        // 两个面板都到得了的位置。广播请求值（可能越界）会让轨道视图先按自己的上限
        // 钳回来，两个面板在工程结尾处错开一个差量。
        if (syncEnabled && !timelineSyncApplyingRef.current) {
            const appliedDrawing = hostRef.current?.getViewport().scrollLeft ?? next;
            timelineViewportSync.setViewport(
                {
                    scrollLeft: timelineViewportStateToNative(appliedDrawing, offset),
                    pxPerSec,
                },
                PIANO_ROLL_SYNC_ORIGIN,
            );
        }
        // 原生滚动位置的钳制校正由宿主的镜像回写负责（它每帧都会把原生位置对齐真值）
        // ——不要再写原生 scroller，否则会与内核真值打架。
        setScrollLeft(next);
    }, [pxPerSec, s.paramEditorSyncTimeline]);

    const zoomTimelineStateRef = useRef({
        playheadSec: s.playheadSec,
        projectSec: dynamicProjectSec,
    });
    useLayoutEffect(() => {
        zoomTimelineStateRef.current = {
            playheadSec: s.playheadSec,
            projectSec: dynamicProjectSec,
        };
    });

    // 指定缩放下的内容宽（口径与轨道视图一致：工程宽向上取整）。缩放提交时要用
    // **目标**缩放的内容宽来钳位置——用当前缩放算出的上限会在放大时把位置压低。
    // 定义在缩放入口之前：`useCallback` 的依赖数组在 render 期求值，写在后面会命中 TDZ。
    const contentWidthAt = useCallback(
        (scale: number) => Math.max(1, Math.ceil(dynamicProjectSec * scale)),
        [dynamicProjectSec],
    );

    const queueHorizontalZoom = useCallback(
        (nextPxPerSec: number, nextNativeScrollLeft: number) => {
            // 事件内只记录缩放意图，**绝不**在提交前改写 refs/state：任何
            // “ref 先行”都会让夹缝中的 rAF 绘制读到“新缩放 + 旧滚动”的
            // 混合投影——参数线/原始音高线/参考线等所有线条整体抽搐一帧。
            horizontalZoomPendingRef.current = {
                nextScale: nextPxPerSec,
                nextScrollLeft: nextNativeScrollLeft,
            };
            // 「一帧一次原子提交」：rAF 合并同帧内的连续缩放事件；
            // flushSync 把两个 state 在同一批提交中落地，
            // DOM 内容宽度按新缩放重排后，layout effect 在同一提交内写原生
            // scrollLeft 并同帧重绘标尺/网格/画布/波形（applyScrollLayers）。
            // refs 只随 render 写入、与 state 同帧移动——绘制前所有图层拿到
            // 同一对投影值，不存在任何混合帧窗口。
            if (zoomRafRef.current == null) {
                zoomRafRef.current = requestAnimationFrame(() => {
                    zoomRafRef.current = null;
                    const pending = horizontalZoomPendingRef.current;
                    if (!pending) return;
                    const offset = paramEditorSyncTimelineRef.current
                        ? timelineOffsetRef.current
                        : 0;
                    const drawingScrollLeft = timelineViewportNativeToState(
                        pending.nextScrollLeft,
                        offset,
                    );
                    flushSync(() => {
                        setPxPerSec(pending.nextScale);
                        setScrollLeft(drawingScrollLeft);
                    });
                });
            }
        },
        [setPxPerSec, setScrollLeft],
    );

    const handleHorizontalZoom = useCallback(
        (nextPxPerSec: number, nextScrollLeft: number) => {
            // 计算结果为绘制坐标；同步时需换算回原生（轨道）坐标再交给 layout effect。
            const nativeNextScrollLeft = timelineViewportStateToNative(
                nextScrollLeft,
                s.paramEditorSyncTimeline ? timelineOffsetRef.current : 0,
            );
            // 【为什么必须在这里钳一次】共享位置（原生坐标）必须落在两个面板**公共**
            // 的可滚范围内（= `resolveTimelineScrollRange` 的上限，按**目标**缩放的
            // 工程宽算，与轨道视图同一份口径）。不钳的话，缩放锚点可以把位置算到范围
            // 之外：参数编辑器会广播一个自己都到不了的值，轨道视图收到后按自己的上限
            // 钳回来 —— 两个面板在工程结尾处错开，且位置互相追赶（抖动）。
            const range = resolveTimelineScrollRange({
                contentWidth: contentWidthAt(nextPxPerSec),
                viewportWidth: viewSizeRef.current.w,
            });
            const clampedNativeScrollLeft = Math.min(
                range.maxScrollLeft,
                Math.max(range.minScrollLeft, nativeNextScrollLeft),
            );
            queueHorizontalZoom(nextPxPerSec, clampedNativeScrollLeft);
        },
        [s.paramEditorSyncTimeline, contentWidthAt, queueHorizontalZoom],
    );

    // 工具栏/快捷键聚焦缩放（hifi:zoomTimelineFocus）。放在 handleHorizontalZoom
    // 定义之后，避免 render 期求值 deps 时命中 const 的 TDZ。
    useEffect(() => {
        function onZoomFocused(e: Event) {
            const { projectSec } = zoomTimelineStateRef.current;
            const inPianoRoll = getActiveSurface() === "pianoRoll";
            if (!inPianoRoll) return;

            const factor = Number((e as CustomEvent<{ factor?: number }>).detail?.factor ?? 1);
            if (!Number.isFinite(factor) || factor <= 0) return;

            const scroller = scrollerRef.current;
            if (!scroller) return;

            const syncEnabled = s.paramEditorSyncTimeline;
            const zoom = resolveHorizontalWheelZoom({
                factor,
                basePxPerSec: pxPerSecRef.current,
                // 与滚轮路径同一契约：base 一律用绘制坐标（scrollLeftRef），
                // 并显式传入同步模式的负向最小滚动与锚点偏移；结果同样交给
                // handleHorizontalZoom 换算回原生坐标——两个缩放入口的坐标
                // 空间保持一致，避免同步模式下基准坐标空间错位。
                baseScrollLeft: scrollLeftRef.current,
                totalSec: projectSec,
                viewportWidth: scroller.clientWidth,
                playheadZoomEnabled: true,
                playheadSec: getVisualPlayheadSec(),
                anchorScreenX: 0,
                minPxPerSec: resolveTimelineMinPxPerSec({
                    baseMinPxPerSec: MIN_PX_PER_SEC,
                    projectSec,
                    viewportWidthPx: scroller.clientWidth,
                }),
                maxPxPerSec: MAX_PX_PER_SEC,
                minScrollLeft: syncEnabled ? -timelineOffsetRef.current : 0,
                anchorOffsetPx: syncEnabled ? timelineOffsetRef.current : 0,
            });
            if (!zoom) return;

            handleHorizontalZoom(zoom.nextPxPerSec, zoom.nextScrollLeft);
        }

        window.addEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
        return () =>
            window.removeEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
    }, [getVisualPlayheadSec, s.paramEditorSyncTimeline, handleHorizontalZoom]);
    // 副参数独立显示开关，默认全部关闭
    const [secondaryParamVisible, setSecondaryParamVisible] = useState<
        Partial<Record<ParamName, boolean>>
    >({});

    const toggleSecondaryParam = useCallback((param: ParamName) => {
        setSecondaryParamVisible((prev) => toggleSecondaryParamVisibility(prev, param));
    }, []);

    const pitchViewRef = useRef<ValueViewport>({
        center: 72,
        span: 24,
    });
    const setPitchView = useCallback(
        (next: ValueViewport) => {
            pitchViewRef.current = next;
            syncVerticalScrollbarForViewport("pitch", next);
            // 值域视口是**竖轴的手势提交点**：滚动条 thumb 已同步写下 DOM，曲线 /
            // 音高键 / 数值轴（GL）必须在**同一任务**里提交，否则竖轴平移/缩放时
            // 这些线比滚动条慢一帧到几帧（见 `paintNow`）。
            paintNow();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const paramViewsRef = useRef<Record<string, ValueViewport>>({});
    const setParamViewport = useCallback(
        (param: string, next: ValueViewport) => {
            paramViewsRef.current = { ...paramViewsRef.current, [param]: next };
            syncVerticalScrollbarForViewport(param as ParamName, next);
            // 同 `setPitchView`：竖轴的手势提交点，DOM 与 GL 必须同任务。
            paintNow();
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const rootTrackId = useMemo(() => {
        return resolveRootTrackId(s.tracks, effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, s.tracks]);

    const rootTrack = useMemo(() => {
        if (!rootTrackId) return null;
        return s.tracks.find((tr) => tr.id === rootTrackId) ?? null;
    }, [s.tracks, rootTrackId]);

    const childFormantOffsetParam = useMemo(() => {
        if (!effectiveSelectedTrackId || !selectedIsChildTrack) return null;
        const algo = rootTrack?.pitchAnalysisAlgo;
        if (algo !== "nsf_hifigan_onnx" && algo !== "vslib") return null;
        return buildChildFormantOffsetCentsParam(effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, selectedIsChildTrack, rootTrack?.pitchAnalysisAlgo]);

    const pitchGroupActive =
        editParam === "pitch" ||
        editParam === childPitchOffsetCentsParam ||
        editParam === childPitchOffsetDegreesParam;
    // 工具栏上的音高组按钮使用简写（非中文语系为 PIT），完整名称放入 ToolTip。
    const pitchGroupLabel =
        editParam === childPitchOffsetCentsParam
            ? t("child_pitch_mode_cents")
            : editParam === childPitchOffsetDegreesParam
              ? t("child_pitch_mode_degrees")
              : t("param_btn_pitch");
    const pitchGroupTooltip =
        editParam === childPitchOffsetCentsParam
            ? t("child_pitch_offset_cents_label")
            : editParam === childPitchOffsetDegreesParam
              ? t("child_pitch_offset_degrees_label")
              : t("pitch");

    // 声码器参数描述符（由 algo 动态定制面板）
    const [processorParams, setProcessorParams] = useState<ProcessorParamDescriptor[]>([]);
    const processorParamsRef = useRef<ProcessorParamDescriptor[]>([]);
    const [processorStaticParams, setProcessorStaticParams] = useState<ProcessorParamDescriptor[]>(
        [],
    );
    const [processorStaticValues, setProcessorStaticValues] = useState<Record<string, number>>({});

    // 工具栏参数按钮按“音高 → 中间参数（随算法变化）→ 音量/声像”的顺序排列
    const orderedProcessorParams = useMemo(() => {
        const algo = rootTrack?.pitchAnalysisAlgo;
        return [...processorParams].sort(
            (a, b) => getParamToolbarRank(a.id, algo) - getParamToolbarRank(b.id, algo),
        );
    }, [processorParams, rootTrack?.pitchAnalysisAlgo]);
    const currentParamRange = useMemo(() => {
        if (editParam === "pitch") {
            return { min: 24, max: 108 };
        }
        if (isChildPitchOffsetCentsParam(editParam)) {
            return {
                min: CHILD_PITCH_OFFSET_CENTS_RANGE.min,
                max: CHILD_PITCH_OFFSET_CENTS_RANGE.max,
            };
        }
        if (isChildPitchOffsetDegreesParam(editParam)) {
            return {
                min: CHILD_PITCH_OFFSET_DEGREES_RANGE.min,
                max: CHILD_PITCH_OFFSET_DEGREES_RANGE.max,
            };
        }
        if (isChildFormantOffsetCentsParam(editParam)) {
            return {
                min: CHILD_FORMANT_OFFSET_CENTS_RANGE.min,
                max: CHILD_FORMANT_OFFSET_CENTS_RANGE.max,
            };
        }
        const desc = processorParamsRef.current.find((d) => d.id === editParam);
        if (desc?.kind.type === "automation_curve") {
            return {
                min: desc.kind.min_value,
                max: desc.kind.max_value,
            };
        }
        return undefined;
    }, [editParam]);

    const currentParamDefaultValue = useMemo(() => {
        if (editParam === "pitch") return 60;
        if (
            isChildPitchOffsetCentsParam(editParam) ||
            isChildPitchOffsetDegreesParam(editParam) ||
            isChildFormantOffsetCentsParam(editParam)
        ) {
            return 0;
        }
        const desc = processorParamsRef.current.find((d) => d.id === editParam);
        if (desc?.kind.type === "automation_curve") {
            return Number(desc.kind.default_value) || 0;
        }
        if (editParam === "volume" || editParam === "dyn" || editParam === "dyn_edit") {
            return 1;
        }
        return 0;
    }, [editParam]);

    const currentParamQuantizeUnit = useMemo(() => {
        if (isChildPitchOffsetCentsParam(editParam)) return 100;
        if (isChildPitchOffsetDegreesParam(editParam)) return 0.5;
        if (isChildFormantOffsetCentsParam(editParam)) return 50;
        if (editParam === "volume" || isDynParam(editParam)) {
            return 0.05;
        }
        if (editParam === "formant_shift_cents") return 50;
        if (editParam === "breath_gain" || editParam === "hifigan_tension") {
            return 0.05;
        }
        if (editParam === "pan") return 0.1;
        if (editParam === "breathiness") return 250;
        const span = Math.abs((currentParamRange?.max ?? 1) - (currentParamRange?.min ?? 0));
        if (span <= 0) return 0.01;
        return Math.max(0.01, span / 20);
    }, [editParam, currentParamRange]);

    useEffect(() => {
        if (!isChildPitchOffsetParam(editParam)) return;
        if (paramViewsRef.current[editParam]) return;
        const range = isChildPitchOffsetCentsParam(editParam)
            ? CHILD_PITCH_OFFSET_CENTS_RANGE
            : isChildPitchOffsetDegreesParam(editParam)
              ? CHILD_PITCH_OFFSET_DEGREES_RANGE
              : CHILD_FORMANT_OFFSET_CENTS_RANGE;
        paramViewsRef.current = {
            ...paramViewsRef.current,
            [editParam]: {
                center: (range.min + range.max) / 2,
                span: range.max - range.min,
            },
        };
        invalidate();
    }, [editParam, invalidate]);

    // 当 algo 变化时，重新抓取参数描述符
    useEffect(() => {
        const algo = rootTrack?.pitchAnalysisAlgo ?? "nsf_hifigan_onnx";
        let cancelled = false;
        paramsApi
            .getProcessorParams(algo)
            .then((params) => {
                if (cancelled) return;
                // 只保留 AutomationCurve 类型（可以绘制曲线的）
                const curvable = params.filter((p) => p.kind.type === "automation_curve");
                const staticParams = params.filter((p) => p.kind.type === "static_enum");
                processorParamsRef.current = curvable;
                setProcessorParams(curvable);
                setProcessorStaticParams(staticParams);
                // 初始化还没有视口的参数 (优化，直接读写 Ref)
                const nextViews = { ...paramViewsRef.current };
                let viewsChanged = false;
                for (const p of curvable) {
                    if (!nextViews[p.id] && p.kind.type === "automation_curve") {
                        const { min_value, max_value, default_value } = p.kind;
                        const span = max_value - min_value;
                        nextViews[p.id] = {
                            center: default_value,
                            span: span > 0 ? span : 1,
                        };
                        viewsChanged = true;
                    }
                }
                if (viewsChanged) {
                    paramViewsRef.current = nextViews;
                    invalidate(); // 数据有初始化，通知画布重绘
                }

                if (!rootTrackId || staticParams.length === 0) {
                    setProcessorStaticValues({});
                    return;
                }

                Promise.all(
                    staticParams.map((param) => paramsApi.getStaticParam(rootTrackId, param.id)),
                )
                    .then((values) => {
                        if (cancelled) return;
                        const nextValues: Record<string, number> = {};
                        for (const item of values) {
                            if (item.ok) {
                                nextValues[item.param] = item.value;
                            }
                        }
                        setProcessorStaticValues(nextValues);
                    })
                    .catch(() => {
                        if (!cancelled) {
                            setProcessorStaticValues({});
                        }
                    });
            })
            .catch(() => {
                if (!cancelled) {
                    processorParamsRef.current = [];
                    setProcessorParams([]);
                    setProcessorStaticParams([]);
                    setProcessorStaticValues({});
                }
            });
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rootTrack?.pitchAnalysisAlgo, rootTrackId]);

    const handleStaticParamChange = useCallback(
        async (paramId: string, value: number) => {
            if (!rootTrackId) return;
            try {
                const result = await paramsApi.setStaticParam(rootTrackId, paramId, value, true);
                if (result.ok) {
                    setProcessorStaticValues((prev) => ({
                        ...prev,
                        [paramId]: value,
                    }));
                }
            } catch (err) {
                // 静态参数调整是滑块 onValueChange 的 fire-and-forget 调用，
                // IPC 失败必须兜底，否则 unhandledrejection 且滑块值不一致。
                console.error("[setStaticParam] failed:", err);
            }
        },
        [rootTrackId],
    );

    const getProcessorParamLabel = useCallback(
        (param: ProcessorParamDescriptor) => {
            switch (param.id) {
                case "breath_enabled":
                    return t("breath_mode_label");
                case "breath_gain":
                    return t("breath_gain_label");
                case "hifigan_tension":
                    return t("hifigan_tension_label");
                case "formant_shift_cents":
                    return t("formant_shift_label");
                case "hifigan_volume":
                case "volume":
                    return t("volume_label");
                case "dyn":
                case "dyn_edit":
                    return t("dyn_label");
                case "synth_mode":
                    return t("vslib_synth_mode_label");
                case "pan":
                    return t("pan_label");
                case "breathiness":
                    return t("vslib_breathiness_label");
                default:
                    return param.display_name;
            }
        },
        [t],
    );

    // 工具栏参数按钮的简写标签（非中文语系用三个大写字形的缩写，如 PIT/BRE/VOL）。
    // 全称仍通过 getProcessorParamLabel() 提供，用于 ToolTip。
    const getProcessorParamShortLabel = useCallback(
        (param: ProcessorParamDescriptor) => {
            switch (param.id) {
                case "breath_enabled":
                case "breath_gain":
                    return t("param_btn_breath");
                case "hifigan_tension":
                case "tension":
                    return t("param_btn_tension");
                case "formant_shift_cents":
                    return t("param_btn_formant");
                case "hifigan_volume":
                case "volume":
                case "vslib_volume":
                    return t("param_btn_volume");
                case "dyn":
                case "dyn_edit":
                    return t("param_btn_dyn");
                case "pan":
                    return t("param_btn_pan");
                case "breathiness":
                    return t("param_btn_breathiness");
                default:
                    return getProcessorParamLabel(param);
            }
        },
        [getProcessorParamLabel, t],
    );

    const getStaticOptionLabel = useCallback(
        (paramId: string, label: string, value: number) => {
            if (paramId === "breath_enabled") {
                if (value === 0) return t("switch_off");
                if (value === 1) return t("switch_on");
            }
            if (paramId === "synth_mode") {
                if (value === 0) return t("vslib_synth_mode_mono");
                if (value === 1) return t("vslib_synth_mode_mono_formant");
                if (value === 2) return t("vslib_synth_mode_chorus");
            }
            return label;
        },
        [t],
    );

    // 当 processorParams 变化时，若 editParam 不在可用集合内，自动回退到 pitch
    useEffect(() => {
        const available = new Set([
            "pitch",
            ...processorParams.map((p) => p.id),
            ...(childPitchOffsetCentsParam ? [childPitchOffsetCentsParam] : []),
            ...(childPitchOffsetDegreesParam ? [childPitchOffsetDegreesParam] : []),
            ...(childFormantOffsetParam ? [childFormantOffsetParam] : []),
        ]);
        if (isChildPitchOffsetParam(editParam)) {
            if (!selectedIsChildTrack || !effectiveSelectedTrackId) {
                dispatch(setEditParam("pitch"));
                return;
            }
            if (isChildPitchOffsetCentsParam(editParam)) {
                const expected = buildChildPitchOffsetCentsParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
            if (isChildPitchOffsetDegreesParam(editParam)) {
                const expected = buildChildPitchOffsetDegreesParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
            if (isChildFormantOffsetCentsParam(editParam)) {
                const expected = buildChildFormantOffsetCentsParam(effectiveSelectedTrackId);
                if (editParam !== expected) {
                    dispatch(setEditParam(expected));
                    return;
                }
            }
        }

        if (!available.has(editParam)) {
            dispatch(setEditParam("pitch"));
        }
    }, [
        processorParams,
        editParam,
        dispatch,
        childPitchOffsetCentsParam,
        childPitchOffsetDegreesParam,
        childFormantOffsetParam,
        effectiveSelectedTrackId,
        selectedIsChildTrack,
    ]);

    // 收集轨道组内所有 trackId（root + 递归所有子轨道）
    const groupTrackIds = useMemo(() => {
        const ids = new Set<string>();
        if (!rootTrackId) return ids;
        ids.add(rootTrackId);
        const frontier = [rootTrackId];
        let idx = 0;
        while (idx < frontier.length) {
            const cur = frontier[idx++];
            const track = s.tracks.find((t) => t.id === cur);
            if (track?.childTrackIds) {
                for (const childId of track.childTrackIds) {
                    if (!ids.has(childId)) {
                        ids.add(childId);
                        frontier.push(childId);
                    }
                }
            }
        }
        return ids;
    }, [rootTrackId, s.tracks]);

    const referenceRootTrackOptions = useMemo(
        () =>
            listReferenceRootTracks({
                tracks: s.tracks,
                currentRootTrackId: rootTrackId,
            }),
        [rootTrackId, s.tracks],
    );

    const visibleReferenceRootTrackIds = useMemo(
        () =>
            cleanupVisibleReferenceRootTrackIds({
                tracks: s.tracks,
                currentRootTrackId: rootTrackId,
                visibleReferenceRootTrackIds: s.visibleReferenceRootTrackIds,
            }),
        [rootTrackId, s.tracks, s.visibleReferenceRootTrackIds],
    );

    useEffect(() => {
        if (sameStringArray(visibleReferenceRootTrackIds, s.visibleReferenceRootTrackIds)) {
            return;
        }
        dispatch(setVisibleReferenceRootTrackIds(visibleReferenceRootTrackIds));
        void dispatch(persistUiSettings());
    }, [dispatch, s.visibleReferenceRootTrackIds, visibleReferenceRootTrackIds]);

    useEffect(() => {
        if (
            hoveredReferenceRootTrackId &&
            !visibleReferenceRootTrackIds.includes(hoveredReferenceRootTrackId)
        ) {
            setHoveredReferenceRootTrackId(null);
        }
    }, [hoveredReferenceRootTrackId, visibleReferenceRootTrackIds]);

    const pitchHardDisableReason = useMemo(() => {
        if (editParam !== "pitch") return null;
        if (!rootTrack) return null;
        if (!rootTrack.composeEnabled) return t("pitch_requires_compose");
        if (rootTrack.pitchAnalysisAlgo === "none") return t("pitch_requires_algo");
        return null;
    }, [editParam, rootTrack, t]);

    const childPitchHardDisableReason = useMemo(() => {
        if (!isChildPitchOffsetParam(editParam)) return null;
        if (!rootTrack) return null;
        if (!rootTrack.composeEnabled) return t("pitch_requires_compose");
        return null;
    }, [editParam, rootTrack, t]);

    const pitchEnabled =
        editParam === "pitch"
            ? pitchHardDisableReason == null
            : isChildPitchOffsetParam(editParam)
              ? childPitchHardDisableReason == null
              : true;

    const visibleSecondaryParamIds = useMemo(() => {
        // 依赖 state 版 processorParams 而非 ref：切换算法后 ref 已更新，但
        // 若 editParam/开关不变，memo 不重算会让副参数列表继续持有旧算法
        // 的参数 id（持续对已不存在的参数发起取数，眼睛开关与叠加层不一致）。
        return getVisibleSecondaryParamIds({
            editParam,
            processorParamIds: processorParams.map((p) => p.id as ParamName),
            secondaryParamVisible,
        });
    }, [editParam, secondaryParamVisible, processorParams]);

    const updateVisibleReferenceRootTrackIds = useCallback(
        (nextTrackIds: string[]) => {
            dispatch(setVisibleReferenceRootTrackIds(nextTrackIds));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );

    // 【已删除 `secPerBeat = 60 / bpm`】参数编辑器的选区、剪贴板映射、交互换算、
    // 取数窗口全部改为**帧制**（工程级常量栅格，见 `paramSelection.ts`），BPM 只
    // 剩下一个消费者：网格与标尺（`buildTimelineTicks` 直接读 `s.bpm` + Tempo Map）。
    // 因此这里不再需要"每拍秒数"，改 BPM 也不会再牵动选区或触发曲线重取。
    const contentWidth = contentWidthAt(pxPerSec);

    const scrollerRef = useRef<HTMLDivElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisWrapRef = useRef<HTMLDivElement | null>(null);
    const lastScrollLeftRef = useRef<number | null>(null);
    const scrollStateRafRef = useRef<number | null>(null);
    /**
     * **上一次镜像写入 scroller 的原生偏移**（两轴各自记录）。
     *
     * 用途只有一个：判定原生 `scroll` 事件是不是镜像回写造成的**回声**
     * （见 `timeline/scrollEcho` 的 `isMirrorEcho`）。判据必须拿"我上次写下的值"做
     * 基准，而不是拿内核的当前值——连续滚动 / 缩放时事件报的是**上一帧**镜像的值，
     * 与内核此刻已差一整帧位移，用它判会稳定地把回声误收成用户输入，进而把量化误差
     * 推回共享视口，让时间轴跟着抽动。
     *
     * 初值 `NaN` = 从未写过 → 一律判"不是回声"（宁可多采纳一次也不吞掉首帧前的输入）。
     */
    const lastMirroredScrollLeftRef = useRef(Number.NaN);
    const lastMirroredScrollTopRef = useRef(Number.NaN);

    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const gridLayerRef = useRef<HTMLDivElement | null>(null);

    // ── 渲染内核（滚动 / 视口所有权）────────────────────────────────────
    //
    // 【所有权在谁手里】内核是唯一渲染路径，宿主持有滚动真值，**原生 scroller 只是
    // 被动镜像**——宿主每帧把真值写回它，使尚未迁移的输入代码（中键拖拽、框选自动
    // 滚动等直接读写 `scroller.scrollLeft` 的地方）读到的仍是同一份位置，行为不变。
    // 各图层由宿主的帧回调 `applyScrollLayers` 驱动（取代此前的 `scroll` 事件）。
    //
    /**
     * 内核数据镜像（每次 render 更新字段，宿主每帧现读）。
     *
     * 特殊说明：显式标注为 `PianoRollKernelData` 而不是让它靠初始值推断——推断出的
     * 窄类型会让后续写入 `grid`（可选字段）报错，也会漏掉字段名拼写错误。
     */
    const kernelDataRef = useRef<MutablePianoRollKernelData>({
        projectSec: 1,
        valueDomain: { min: 0, max: 1, span: 1 },
        grid: null,
        overlay: null,
        selectionBand: null,
    });
    /** 自绘滚动条的 thumb（恒挂载）。 */
    const hScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    /** 自绘滚动条的**轨道**（承接「点空白翻页」，恒挂载）。 */
    const hScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    /** GL 静态层画布：网格 / 键盘几何，恒挂载（它是这些图层的唯一绘制者）。 */
    const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 键盘轴 GL 画布（阶段 2/3）：键盘几何 + 音名标签。独立画布：轴列不随横向滚动移动。 */
    const glAxisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 动态叠加层 GL 画布（阶段 2/3）：播放头，位于曲线层之上，恒挂载。 */
    const glOverlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

    function pitchDeltaToDegreeSteps(
        basePitch: number,
        targetPitch: number,
        scale: ScaleLike,
    ): number {
        if (!Number.isFinite(basePitch) || !Number.isFinite(targetPitch)) {
            return 0;
        }
        if (Math.abs(targetPitch - basePitch) <= 1e-9) return 0;

        const minStep: number = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
        const maxStep: number = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
        const minPitch = transposePitchByScaleSteps(basePitch, minStep, scale);
        const maxPitch = transposePitchByScaleSteps(basePitch, maxStep, scale);
        const lowPitch = Math.min(minPitch, maxPitch);
        const highPitch = Math.max(minPitch, maxPitch);
        if (targetPitch <= lowPitch) {
            return minPitch <= maxPitch ? minStep : maxStep;
        }
        if (targetPitch >= highPitch) {
            return minPitch <= maxPitch ? maxStep : minStep;
        }

        let left = minStep;
        let right = maxStep;
        const ascending = minPitch <= maxPitch;
        for (let i = 0; i < 24; i += 1) {
            const mid = (left + right) / 2;
            const midPitch = transposePitchByScaleSteps(basePitch, mid, scale);
            if (midPitch < targetPitch === ascending) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return (left + right) / 2;
    }

    const viewSizeRef = useRef({ w: 1, h: 1 });
    const [viewSize, setViewSize] = useState({ w: 1, h: 1 });
    const [timeDisplaySettingsOpen, setTimeDisplaySettingsOpen] = useState(false);
    // 参数编辑器的内容绘制在 sticky 视口层中，滚动范围由后面的 spacer 提供。
    // 两个子元素按垂直方向堆叠，因此 scrollWidth 取二者宽度最大值；
    // 想让原生最大滚动位置为「工程宽」，spacer 需再加一个视口宽。
    //
    // 【不变量：与轨道视图同一份可滚范围】这里复用轨道视图的同一个函数
    // （`resolveTimelineScrollRange`），两者都传自己的视口宽——于是两个面板的
    // 最大滚动位置**逐值相等**（都 = 内容宽）。
    //
    // 【为什么同步偏移不进这里（用户报告的「工程结尾右侧空白长度不同」）】spacer 的
    // 宽度决定原生最大滚动位置。此前同步开启时会再加一个 `timelineOffsetPx`，参数
    // 编辑器因此比轨道视图多出一段「只有自己能滚」的空白：滚到最右时轨道视图停在
    // 内容宽处（工程结尾落在轨道区左缘），参数编辑器却能再滚一个偏移量（工程结尾落到
    // 轨道区左缘以左），两个面板不再对齐。偏移是**纯投影量**，只改内容画在屏幕上的
    // 位置（见内核宿主的 `horizontalOffsetPx`），不参与可滚范围。
    const timelineScrollRange = useMemo(
        () => resolveTimelineScrollRange({ contentWidth, viewportWidth: viewSize.w }),
        [contentWidth, viewSize.w],
    );
    const paddedContentWidth = timelineScrollRange.paddedContentWidth;

    useLayoutEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;
        const ro = new ResizeObserver(() => {
            const w = Math.max(1, Math.floor(el.clientWidth));
            const h = Math.max(1, Math.floor(el.clientHeight));
            viewSizeRef.current = { w, h };
            setViewSize({ w, h });
            // 视口尺寸（含首次测量）必须立即同步到参数编辑器总线：总线
            // 初始窗口是 1px 占位，PianoRollWaveformSurface 按总线 axis
            // 裁剪（窗口外 clip 直接跳过、不发起取数）——fresh 应用在
            // 首次滚动/缩放之前总线永远不会被 emit，波形因此无法显示。
            pianoRollViewportBus.emit(scrollLeftRef.current, pxPerSecRef.current, w);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // The ruler is React-rendered, but the main graph is canvas-rendered.
    // Ensure playhead changes (seek / playback) trigger a redraw.
    useEffect(() => {
        invalidate();
    }, [s.playheadSec, invalidate]);

    /**
     * 标尺播放头线的**首帧与缩放后**定位。
     *
     * 【为什么位置不再交给 React 渲染】见 `TimeRulerPlayhead.positionFromProps`：
     * React 只持有 30Hz 的已提交播放头，而画布/GL 上的播放头用的是 60Hz 插值
     * 位置，两者同时写 `style.left` 会让标尺上的线落在旧位置 —— 用户看到的就是
     * "标尺的播放头与参数编辑器里的播放头分离"。
     *
     * 这里用 layout effect（绘制前执行）补上首帧与缩放后的位置，播放/滚动期间的
     * 逐帧位置由 `useVisualPlayhead` 的 onFrame 与 `applyScrollLayers` 负责。
     */
    useLayoutEffect(() => {
        // 缩放取**内核真值**（与 GL 播放头、标尺平移同一口径）：面板的 React
        // `pxPerSec` 是请求值，内核可能因视口宽度变化而钳制它，用请求值定位会与
        // 主体播放头差一个 `播放头秒数 × 缩放差`。宿主未就绪时退回面板值。
        const view = resolvePanelRenderViewport({
            kernelView: hostRef.current?.getViewport() ?? null,
            refPxPerSec: pxPerSecRef.current,
            refScrollLeftPx: scrollLeftRef.current,
        });
        const leftPx = playheadLineLeftPx(
            createTimelineAxis({
                pxPerSec: view.pxPerSec,
                scrollLeftPx: view.scrollLeftPx,
                viewportWidthPx: viewSizeRef.current.w,
                dpr: readDevicePixelRatio(),
            }),
            visualPlayheadSecRef.current,
        );
        if (rulerPlayheadLineRef.current) {
            rulerPlayheadLineRef.current.style.left = `${leftPx}px`;
        }
        if (rulerPlayheadHeadRef.current) {
            rulerPlayheadHeadRef.current.style.left = `${leftPx}px`;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pxPerSec]);

    // 原地等待渲染（位置冻结）期间不得推进视觉插值（见 TimelinePanel 同名注释）。
    const isTransportAdvancing =
        s.runtime.isPlaying &&
        !s.runtime.playbackWaitingForRender &&
        s.runtime.playbackPositionSec > 1e-4;

    useVisualPlayhead({
        syncedPlayheadSec: s.playheadSec,
        syncedAtMs: s.playheadSampledAtMs,
        isTransportAdvancing,
        onFrame: useCallback(
            (visualPlayheadSec: number) => {
                visualPlayheadSecRef.current = visualPlayheadSec;
                // 标尺播放头线**不在这里写**：它由内核在帧提交里按内核视口写，
                // 与 GL 主体播放头同源同帧（见 `sync.rulerPlayheadLine` 的说明）。
                // 这里曾用面板的 `pxPerSecRef` 写同一条线 —— 面板缩放与内核缩放
                // 一旦不一致（面板宽度变化时内核会重新钳制缩放），两条线就会相差
                // `播放头秒数 × 缩放差`，也就是用户报告的"标尺线与主体线分离"。
                if (!s.paramEditorSyncTimeline && s.autoScrollEnabled && s.runtime.isPlaying) {
                    const scroller = scrollerRef.current;
                    if (scroller) {
                        const next = computeAutoFollowScrollLeft({
                            playheadSec: visualPlayheadSec,
                            pxPerSec: pxPerSecRef.current,
                            viewportWidth: scroller.clientWidth,
                            contentWidth,
                        });
                        if (Math.abs(scroller.scrollLeft - next) > 0.5) {
                            scroller.scrollLeft = next;
                            lastMirroredScrollLeftRef.current = next;
                            syncScrollLeft(scroller);
                        }
                    }
                }
                invalidate();
            },
            // syncScrollLeft reads the latest scroll state through refs and is called
            // imperatively; including the plain render-scope function would defeat memoization.
            // eslint-disable-next-line react-hooks/exhaustive-deps
            [
                contentWidth,
                invalidate,
                s.paramEditorSyncTimeline,
                s.autoScrollEnabled,
                s.runtime.isPlaying,
            ],
        ),
    });

    useEffect(() => {
        return () => {
            if (scrollStateRafRef.current != null) {
                cancelAnimationFrame(scrollStateRafRef.current);
                scrollStateRafRef.current = null;
            }
            if (zoomRafRef.current != null) {
                cancelAnimationFrame(zoomRafRef.current);
                zoomRafRef.current = null;
            }
        };
    }, []);

    function applyScrollLayers(next: number) {
        if (rulerContentRef.current) {
            rulerContentRef.current.style.transform = `translateX(${-next}px)`;
        }

        if (gridLayerRef.current) {
            invokeGridRedrawHandler(gridLayerRef.current, next);
        }

        // 同步绘制：滚动事件在绘制前触发，画布必须与标尺/网格（上方已同步
        // 落地）在同一帧内提交，否则滚动中会出现画布与网格的分层漂移。
        drawRef.current();
        // 波形面走同一条同步链：不能在 React state（rAF）提交后再画。
        //
        // 缩放一并取内核真值（与 `drawRef` 同一理由，见 `resolvePanelRenderViewport`）：
        // 渲染期 ref 可能提前于内核变化，波形与其余图层就会差一个缩放。
        const emitPxPerSec = resolvePanelRenderViewport({
            kernelView: hostRef.current?.getViewport() ?? null,
            refPxPerSec: pxPerSecRef.current,
            refScrollLeftPx: scrollLeftRef.current,
        }).pxPerSec;
        pianoRollViewportBus.emit(next, emitPxPerSec, viewSizeRef.current.w);
        // 标尺播放头线：**宿主模式下由内核写**（同一次帧提交、同一份内核视口，
        // 与 GL 主体播放头逐设备像素一致）。这里只在无宿主时兜底 —— 两个写者写同一
        // 个 `style.left` 是"最后一次写入者获胜"，一旦口径不同就会分离。
        if (hostRef.current === null) {
            const playheadLeftPx = playheadLineLeftPx(
                createTimelineAxis({
                    pxPerSec: emitPxPerSec,
                    scrollLeftPx: next,
                    viewportWidthPx: viewSizeRef.current.w,
                    dpr: readDevicePixelRatio(),
                }),
                visualPlayheadSecRef.current,
            );
            if (rulerPlayheadLineRef.current) {
                rulerPlayheadLineRef.current.style.left = `${playheadLeftPx}px`;
            }
            if (rulerPlayheadHeadRef.current) {
                rulerPlayheadHeadRef.current.style.left = `${playheadLeftPx}px`;
            }
        }
    }

    /**
     * 把「绘制坐标」的水平位置提交到当前滚动载体。
     *
     * 【为什么要单独一个函数】面板里有若干**权威写入**点（时间轴同步、键盘缩放、
     * 缩放事务落地、关闭同步时还原位置）：它们算出目标位置后直接落到滚动载体
     * （宿主，内部换算原生坐标并镜像回写）。把换算收在一处，避免每调用点各写一份。
     *
     * 特殊说明：滚轮 / 自动滚屏等**增量**路径不走这里——它们仍写原生 scroller，
     * 再由 `syncScrollLeft` 采纳进内核（见该函数注释）。宿主尚未创建（挂载期）时
     * 退回直接写原生 scroller，与调用方原本的语义一致。
     *
     * @param drawingScrollLeft 目标水平位置（绘制坐标）。
     */
    function applyHorizontalScrollPosition(drawingScrollLeft: number): void {
        const host = hostRef.current;
        if (host) {
            host.setScrollLeft(drawingScrollLeft);
            return;
        }
        const scroller = scrollerRef.current;
        if (!scroller) return;
        const offset = paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0;
        const native = timelineViewportStateToNative(drawingScrollLeft, offset);
        scroller.scrollLeft = native;
        lastMirroredScrollLeftRef.current = native;
    }

    /**
     * **立即**提交一帧（有宿主时同任务提交，否则退回标脏 rAF）。
     *
     * 【为什么需要它：DOM / Canvas2D 与 GL 必须落在同一个任务里】
     * 面板的权威写入点（缩放落地、共享视口应用、值域平移/缩放）会同步写 DOM /
     * Canvas2D。若 GL 侧（参数线 / 原始音高线 / 播放头）等下一帧才跟上，同一份视口
     * 就被两套图层分帧呈现——用户看到的就是"这些线迟缓几帧渲染"。
     *
     * 实测（Chrome，参数编辑器内滚轮缩小，按帧记录两个绘制路径的 scrollLeft）：
     * 面板在 t=6384 写下 DOM/Canvas2D，GL 到 t=6446 才用同一个值重绘（React 提交 +
     * rAF 重新排队，慢一帧到数帧）。改用本方法后两者同任务提交，手感与旧实现
     * （原生滚动 + 在事件里同步重绘）一致。
     *
     * 与 `invalidate()` 的分工：数据变更（编辑结果、主题、音阶）仍走 rAF 合并；
     * **用户手势的视口提交**走本方法。
     */
    function paintNow(): void {
        const host = hostRef.current;
        if (host) {
            host.paintNow();
            return;
        }
        invalidate();
    }

    /**
     * 视口提交的统一入口：内核写入 + 各图层**同任务**提交。
     *
     * 流程：内核写入（含钳制与标脏）→ `paintNow()`（同一任务的 GL + DOM + Canvas2D 提交）。
     *
     * 【缩放必须与位置**一起**写入内核（否则画布层会抽搐一帧）】
     * 只写位置时，内核仍是**旧缩放**：本函数紧接着的同帧绘制（波形 / 参数线 / 播放头 /
     * 选区都在 GL 或 Canvas2D 上）就会用旧缩放画出新位置，而 DOM 侧的标尺 / 网格（由
     * React 的 `pxPerSec` 驱动）已经是新缩放——一屏之内两套缩放，下一帧才被纠正。
     * 实测（Chrome，参数编辑器内滚轮缩放，日志打在绘制前）：`kernelPps=150 targetPps=165`
     * ——画布按 150 画、DOM 按 165 画。传 `pxPerSecAtCommit` 后内核先原子换到新缩放，
     * 同帧绘制即与 DOM 一致（`kernelPps=165 targetPps=165`）。
     *
     * 特殊说明：宿主尚未创建（挂载期）时退回 `applyScrollLayers` 直接写 DOM，与各调用点
     * 原本的兜底语义一致。
     *
     * @param drawingScrollLeft 目标水平位置（绘制坐标）。
     * @param pxPerSecAtCommit 本次提交应生效的缩放；缺省表示缩放不变（纯滚动）。
     */
    function commitViewportNow(drawingScrollLeft: number, pxPerSecAtCommit?: number): void {
        const host = hostRef.current;
        if (host) {
            if (pxPerSecAtCommit !== undefined && Number.isFinite(pxPerSecAtCommit)) {
                // 缩放 + 位置**一次原子写入**：内核用新缩放算上限并钳制，随后同帧绘制
                // 用的就是新缩放（见上方说明）。
                host.setViewport({ pxPerSec: pxPerSecAtCommit, scrollLeft: drawingScrollLeft });
            } else {
                applyHorizontalScrollPosition(drawingScrollLeft);
            }
            host.paintNow();
            return;
        }
        applyHorizontalScrollPosition(drawingScrollLeft);
        applyScrollLayers(drawingScrollLeft);
    }

    /**
     * 采纳一次水平滚动位置变化（原生 scroller → 真值）。
     *
     * 流程：原生坐标 → 绘制坐标 → 同步开关时推送共享视口 → 通知各图层 → 量化提交 state。
     *
     * 【本函数的语义（重要）】内核是唯一渲染路径，它是**采纳点**：尚未迁移的输入
     * 代码（滚轮、中键平移、自动滚屏）仍然直接写 `scroller.scrollLeft`，这里把该值
     * 收进内核（`host.setScrollLeft`），由内核完成钳制、镜像回写与帧提交。这样：
     * - 所有既有手势**逐条保持可用**，无需在本任务里重写输入路径（Task 7 才迁移）；
     * - 钳制与渲染真值收敛到内核一处，不会出现两套边界；
     * - 收敛性：内核回写的值与内核当前值相同 → 不产生状态变化 → 不再触发帧，
     *   因此「写原生 → 采纳 → 镜像回写 → 再触发 onScroll」不会形成循环。
     *
     * 特殊说明：原生 scroller 的 `scroll` 事件是**纯镜像回声**，不再经本函数
     * （见 `onScrollerScroll`）；本函数只由显式写入原生的输入路径调用。
     *
     * @param scroller 原生滚动容器。
     */
    function syncScrollLeft(scroller: HTMLDivElement) {
        const syncEnabled = s.paramEditorSyncTimeline;
        const offset = syncEnabled ? timelineOffsetRef.current : 0;
        const next = timelineViewportNativeToState(scroller.scrollLeft, offset);
        if (lastScrollLeftRef.current != null && lastScrollLeftRef.current === next) {
            return;
        }
        lastScrollLeftRef.current = next;
        scrollLeftRef.current = next;
        if (syncEnabled && !timelineSyncApplyingRef.current) {
            // 【必须用 next（采纳值）而不是 scroller.scrollLeft（原生 DOM）】
            //
            // 原生 scroller 只是**镜像**：真值在 ScrollKernel，由宿主在帧
            // 提交时回写。直接读 DOM 会拿到**尚未回写**的旧值，于是把旧位置当成
            // "用户滚动"推回共享视口——时间轴收到后应用旧值，位置出现回退。
            //
            // 实测（拖时间轴带动参数编辑器时）：共享视口序列 `10 → 20 → pianoRoll
            // 推回 10`，时间轴内核随之从 20 退回 10。连续拖拽时每三帧回退一次
            // （增量呈 `+30, +10, -10` 循环），即用户报告的"阶梯感/被吸附感"。
            //
            // `next` 是刚由原生位置换算出的绘制坐标，再换算回原生即得权威值；
            // 与 `onUserScrollLeft` 的口径一致（后者用的是内核的绘制坐标）。
            timelineViewportSync.setViewport(
                {
                    scrollLeft: timelineViewportStateToNative(next, offset),
                    pxPerSec: pxPerSecRef.current,
                },
                PIANO_ROLL_SYNC_ORIGIN,
            );
        }
        // 交给内核（它会按新边界钳制、镜像回写并提交各图层）。
        //
        // 【为什么在内核写入后立刻 `paintNow`】本函数由**原生滚动事件**驱动（触摸拖拽、
        // 触控板惯性、中键平移、框选自动滚屏）。原生滚动发生在浏览器的渲染步骤里，而
        // 滚轮/拖拽任务里排队的 rAF 要等**下一帧**才跑——这会让可见内容（标尺、网格、
        // 主画布、波形、曲线、播放头）比原生滚动慢一帧。旧实现是在滚动事件里同步重绘，
        // 因此没有这一帧差；这里用 `paintNow()`（同任务提交）恢复到同一时序。
        const host = hostRef.current;
        if (host) {
            host.setScrollLeft(next);
            host.paintNow();
            return;
        }
        applyScrollLayers(next);
        if (scrollStateRafRef.current == null) {
            scrollStateRafRef.current = requestAnimationFrame(() => {
                scrollStateRafRef.current = null;
                setScrollLeft(scrollLeftRef.current);
            });
        }
    }

    // ── 内核宿主：创建 / 销毁 ────────────────────────────────────────
    //
    // 【为什么只在挂载时创建】宿主持有滚动位置与手势状态，是长生命周期运行时
    // 对象；随渲染重建会让滚动位置静默归零（与 `ScrollKernel` 的生命周期约束
    // 一致）。数据经 `kernelDataRef` 流入，尺寸经宿主自己的 ResizeObserver 更新。
    //
    // 【两套坐标系（本任务最容易出错的地方）】
    // 「同步时间轴视图」开启时，参数编辑器的内容层要整体右移一个偏移量（`offset`，
    // 实测 200px），两边的网格线才能在屏幕上对齐。于是：
    // - **原生坐标**：`[0, 内容宽]`——与轨道视图**同一格式、同一范围**（同步模式下
    //   它就是两个面板共享的那个位置，见 `horizontalOffsetPx` 的「关键不变量」）；
    // - **绘制坐标**：`[−offset, 内容宽 − offset]`——含负值，各图层（标尺/网格/画布/
    //   波形）用。偏移只是投影：它把内容整体右移，不扩大可滚范围。
    //
    // 内核的位置字段恒被钳到 `[0, max]`（无法表示负值），所以内核持有**原生坐标**，
    // 本面板消费的 `axis.scrollLeftPx` 是**绘制坐标**。换算只在宿主边界发生
    // （见宿主 `horizontalOffsetPx`），面板这一侧不再自行换算，避免出现第三份口径。
    useEffect(() => {
        const container = scrollerRef.current;
        if (!container) return;
        const hThumb = hScrollbarThumbRef.current;
        const vThumb = vScrollbarThumbRef.current;
        if (!hThumb || !vThumb) return;

        const host = createPianoRollKernelHost({
            container,
            hScrollbarThumb: hThumb,
            vScrollbarThumb: vThumb,
            hScrollbarTrack: hScrollbarTrackRef.current ?? undefined,
            vScrollbarTrack: vScrollbarTrackRef.current ?? undefined,
            data: () => kernelDataRef.current,
            initialPxPerSec: pxPerSecRef.current,
            glCanvas: glCanvasRef.current,
            glAxisCanvas: glAxisCanvasRef.current,
            glOverlayCanvas: glOverlayCanvasRef.current,
            axisWidthPx: AXIS_W,
            // GL 场景层恒开启：它是唯一渲染路径（见下方 `skip*` 的「已知并接受的
            // 限制」说明与设计文档 §2.4）。
            glSceneEnabled: true,
            // 偏移经 ref 读取（同步开关与布局偏移都在运行时变化，闭包捕获会读到挂载时的旧值）。
            horizontalOffsetPx: () =>
                paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0,
            // 全部传 **getter**（见 `PianoRollKernelDomSync` 的说明）：这些元素可能
            // 在宿主创建之后才挂载（标尺随视图出现、停靠重排后 DOM 重建）。按值捕获
            // 会让宿主永久持有 null 或已脱离文档的节点，对应写入从此静默失效。
            sync: {
                rulerContent: () => rulerContentRef.current,
                gridLayer: () => gridLayerRef.current,
                // 标尺播放头线与三角由内核在同一次帧提交里写（与 GL 播放头同源同帧），
                // 避免"面板按自己的缩放写、GL 按内核缩放画"造成的水平分离。
                rulerPlayheadLine: () => rulerPlayheadLineRef.current,
                rulerPlayheadHead: () => rulerPlayheadHeadRef.current,
            },
            // 帧提交：宿主已完成滚动条几何与标尺 / 网格的 DOM 写入，这里只做
            // 「画布 + 波形 + 播放头」三项的提交。复用 `applyScrollLayers`，
            // 保证各图层与宿主同帧刷新。
            onFrame: (axis) => {
                const drawing = axis.scrollLeftPx;
                scrollLeftRef.current = drawing;
                applyScrollLayers(drawing);
                // 【为什么必须同步 lastScrollLeftRef】镜像回写会让原生 scroller 触发
                // `scroll` 事件。把"上一次已知位置"同步成内核真值后，即便有代码经
                // `syncScrollLeft` 采纳，`next` 也与之相等 → 立即早退，**不会**把值
                // 推回共享视口（否则镜像回写会被误当成用户滚动，把时间轴也推着走）。
                lastScrollLeftRef.current = drawing;
                // 镜像回写：让尚未迁移的输入代码（中键拖拽 / 框选自动滚动）读到的
                // 原生位置始终等于内核真值。仅在真正不一致时写，避免每帧触发样式重算。
                const native = timelineViewportStateToNative(
                    drawing,
                    paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0,
                );
                if (shouldWriteNumber(container.scrollLeft, native, 0.5)) {
                    container.scrollLeft = native;
                    // 记下"我写下的值"：它是判定原生 `scroll` 事件是否为回声的唯一基准
                    // （见 `isMirrorEcho` 与镜像 ref 的说明）。
                    lastMirroredScrollLeftRef.current = native;
                }
                if (shouldWriteNumber(container.scrollTop, axis.scrollTopPx, 0.5)) {
                    container.scrollTop = axis.scrollTopPx;
                    lastMirroredScrollTopRef.current = axis.scrollTopPx;
                }
            },
            onScrollTopFrame: (scrollTopPx) => {
                // 竖向逐帧上报：把内核真值**正向**写回面板的值域视口 ref。
                //
                // 【为什么必须有这条通道】`pitchViewRef` / `paramViewsRef` 是竖向滚动
                // 的使用者（曲线投影、命中测试、`valueToY` 都读它们），但竖向真值在
                // 内核且此前**没有回写通道**——面板只能靠原生 `scroll` 事件反向回写
                // 来"猜"，而那条路正是镜像回声的来源（实测每帧把内核回退 ~7.5px，
                // 用户报告为"上下拖经常拖不动 / 有吸附感"）。
                //
                // 改用正向通道后回声不再承担任何职责，可以在 `onScrollerScroll` 里
                // 安全忽略。宿主保证本回调排在 `onFrame`（绘制）之前，因此画面与数据
                // 同帧对齐、不会慢一拍。
                //
                // 只做赋值，不进 React（与横向 `onScrollLeftFrame` 同一约定）。
                const param = editParamRef.current;
                const current = getCurrentViewportForScrollbar(param);
                const bounds = getParamValueBoundsForScrollbar(param);
                const center = centerFromVerticalScrollTop({
                    min: bounds.min,
                    max: bounds.max,
                    span: current.span,
                    scrollTop: scrollTopPx,
                    scrollRangePx: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
                });
                if (param === "pitch") {
                    pitchViewRef.current = { center, span: current.span };
                } else {
                    paramViewsRef.current = {
                        ...paramViewsRef.current,
                        [param]: { center, span: current.span },
                    };
                }
            },
            onScrollLeftCommit: () => {
                // 量化提交：标尺的刻度范围由 React 按视口计算，不同步就会出现
                // 「滚动后刻度消失」。注意 `px` 是**绘制坐标**（宿主对外统一口径）。
                if (scrollStateRafRef.current == null) {
                    scrollStateRafRef.current = requestAnimationFrame(() => {
                        scrollStateRafRef.current = null;
                        setScrollLeft(scrollLeftRef.current);
                    });
                }
            },
            onUserScrollLeft: (drawingScrollLeft) => {
                // 用户手势（拖 thumb / 点轨道翻页）：需要把新位置推给共享视口，
                // 否则同步模式下滚动时间轴不会跟随。
                //
                // 【为什么不走 syncScrollLeft】那条路径依赖 `scroll` 事件，而内核的
                // 镜像是同帧程序化写入的，事件到达时无法再区分"用户滚动"与"自身回写"。
                // 由宿主主动上报手势来源是唯一可靠判据（见 `onUserScrollLeft` 说明）。
                scrollLeftRef.current = drawingScrollLeft;
                lastScrollLeftRef.current = drawingScrollLeft;
                const offset = paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0;
                if (paramEditorSyncTimelineRef.current && !timelineSyncApplyingRef.current) {
                    timelineViewportSync.setViewport(
                        {
                            scrollLeft: timelineViewportStateToNative(drawingScrollLeft, offset),
                            pxPerSec: pxPerSecRef.current,
                        },
                        PIANO_ROLL_SYNC_ORIGIN,
                    );
                }
            },
        });
        hostRef.current = host;
        // dev-only 调试出口：浏览器里读取内核视口真值（滚动 / 值域中心 / 滚动条
        // 几何），用于核对「滚动条 thumb 与实际可滚范围是否一致」这类问题。
        if (import.meta.env.DEV) {
            (
                window as unknown as { __hsPianoRollKernel?: PianoRollKernelHost }
            ).__hsPianoRollKernel = host;
        }
        return () => {
            hostRef.current = null;
            if (import.meta.env.DEV) {
                delete (window as unknown as { __hsPianoRollKernel?: PianoRollKernelHost })
                    .__hsPianoRollKernel;
            }
            host.dispose();
        };
        // 挂载时创建一次；数据经 kernelDataRef 流入（见上方注释）。
        //
        // 依赖项为空是刻意的：宿主一旦重建，滚动位置与手势状态会静默归零（见
        // `ScrollKernel` 的生命周期约束）。回调里的 `editParamRef` 与
        // `getCurrentViewportForScrollbar` / `getParamValueBoundsForScrollbar`
        // 都只读 ref（不读渲染期 state），因此闭包捕获的旧引用仍能取到最新值。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // 内核数据镜像：每次 render 更新字段（宿主每帧现读，不触发重建）。
    //
    // 特殊说明 1：`projectSec` 必须是**秒**而不是 `contentWidth`（像素）——宿主内部
    // 会自行乘以 pxPerSec 得到内容宽，传像素会让内容宽度被放大 pxPerSec 倍，
    // 横向滚动条 thumb 缩成一条线。
    //
    // 特殊说明 2：值域（`min/max/span`）必须与面板当前视口同源。宿主用它把像素位置
    // 换算成值域中心；若停留在占位值，竖向换算的比例就错了——表现为钢琴键盘整体
    // 偏移一个八度、竖向滚动条 thumb 位置也不对（曾实际发生）。
    //
    // 特殊说明 3：`span` 用**当前视口**的跨度（而非值域全长）。竖向像素位置 ↔ 值域
    // 中心的映射依赖 span，用错会让滚到同一位置对应的中心值不同。

    /**
     * 构建 GL 场景层的网格输入（阶段 2）。
     *
     * 流程：由 `editParam` 判定网格种类 → 取值域边界与当前视口 → 解析该主题的两条
     * 网格线颜色（以及钢琴背景 / 音阶高亮颜色）为数值 RGBA → 绑定 `valueToY` →
     * pitch 下再补键盘颜色与音阶音级。
     *
     * 特殊说明 1：`valueToY` 传的是**面板自己的**那个函数（只绑定 `editParam`），
     * 与 Canvas2D 路径共用同一份投影——这正是两种渲染模式网格不会错位的原因。
     *
     * 特殊说明 2：颜色经 `parseRgbaColor` 解析为 0..1 浮点，因为 GL 上传统一用浮点；
     * 在渲染热路径上反复解析 CSS 字符串是纯浪费，故在镜像更新（低频）时做。
     * 解析失败（拿到 NaN）时退回**不透明黑**——宁可颜色不对也不要上传 NaN，
     * NaN 会让整个实例属性失效、整层消失。
     *
     * 特殊说明 3（音阶高亮）：条件与旧 Canvas2D 分支**逐字对齐**——`always` 模式且
     * 存在工程音阶时才给音级集合（`off` / 无音阶时给空数组，等价于不高亮）。
     *
     * 【已知限制：Tempo Map 分段音阶未迁移】旧 Canvas2D 路径还支持 Tempo Map 的
     * **分段音阶**（`buildScaleSegments`：不同时间段用不同音阶，按段画不同 x 范围）。
     * GL 网格几何是**视口坐标且不含时间轴**——它不知道自己在哪个时间段上，无法表达
     * 分段。要迁移需要把网格层做成时间相关（或把高亮拆成独立的时间感知图层），
     * 超出本次改动范围。因此这里只实现**单一工程音阶**的均匀路径：有 Tempo Map
     * 音阶变化的工程，高亮会按工程音阶（而非各段音阶）绘制。
     *
     * 特殊说明 4（钢琴背景）：黑键行背景带**恒开**、无开关、无持久化设置；它只是
     * 纹理，与音阶高亮（语义）互不影响，后者在 GL 实例顺序上压在它之上。
     *
     * @returns 网格输入；GL 关闭或无法解析时返回 null（宿主按"没有网格"处理）。
     */
    function buildGridSpec(): PianoRollGridSpec | null {
        // 显式标注为字面量联合：不加标注时 TS 会把嵌套三元推断成 `string`，
        // 导致返回值无法赋给 `PianoRollGridSpec["kind"]`。
        //
        // 【fallback 分支为什么必须有】内核模式下轴画布整张归 GL（`skipAxisCanvas`
        // 恒为 true，见下方 drawPianoRoll 调用点），这里返回 null 就意味着该参数
        // 的左侧刻度与网格**没有任何绘制者** —— 整列空白。曾表现为：切到音量 /
        // 声像 / 张力 / 气声等一般数值参数时左侧刻度完全消失。凡自动化曲线参数
        // 一律给出 fallback kind（GL 的 resolveAxisKind / 签名层早已支持），
        // 只有非曲线参数（理论上一条都不该走到这）才维持 null。
        const descForKind = processorParamsRef.current.find((d) => d.id === editParam);
        const isAutomationCurve = descForKind?.kind.type === "automation_curve";
        const kind: PianoRollGridSpec["kind"] | null =
            editParam === "pitch"
                ? "pitch"
                : isChildPitchOffsetCentsParam(editParam)
                  ? "cents"
                  : isChildPitchOffsetDegreesParam(editParam)
                    ? "degrees"
                    : isChildFormantOffsetCentsParam(editParam)
                      ? "formantCents"
                      : // 动态面板用倍率刻度（1.0× / 0.5× / 0.25×…），
                        // 与纵轴标签的 dB 换算表配合，见 axisMarkInstances。
                        isDynParam(editParam)
                        ? "level"
                        : isAutomationCurve
                          ? "fallback"
                          : null;
        if (kind === null) return null;

        const bounds = getParamValueBoundsForScrollbar(editParam);
        const view = clampViewport(editParam, getCurrentViewportForScrollbar(editParam));
        const colors = resolvePianoRollColors(themeMode === "dark");
        const finite = (c: readonly number[]) => c.every((v) => Number.isFinite(v));
        /**
         * 解析 CSS 颜色为数值 RGBA；解析失败（NaN）时退回**不透明黑**。
         *
         * 特殊说明：宁可颜色不对也不要上传 NaN——NaN 会让整个实例属性失效、
         * 整层几何消失（比"颜色错"严重得多，且更难归因）。
         */
        const toRgba = (css: string): [number, number, number, number] => {
            // 必须先归一化：`parseRgbaColor` 只认 rgb()/rgba()，hex 会被解析成
            // **不透明洋红**（故意设计，用于暴露漏解析）。本配色表的 whiteKey /
            // blackKey 正是 hex，不归一化会让整个键盘变洋红（曾实际发生）。
            const parsed = parseRgbaColor(normalizeCssColor(css));
            return finite(parsed) ? parsed : [0, 0, 0, 1];
        };

        const base = {
            kind,
            view: { center: view.center, span: view.span },
            absMin: bounds.min,
            absMax: bounds.max,
            valueToY: (value: number, heightPx: number) => valueToY(editParam, value, heightPx),
            strongRgba: toRgba(colors.pitchGridC),
            weakRgba: toRgba(colors.pitchGridOther),
            // 文字与刻度线（阶段 2 Task 5）：GL 侧据此渲染轴标签与刻度。
            paramName: editParam,
            // 纵轴展示单位（音量 / 动态的倍率 ↔ dB）。只对支持切换的参数下发，
            // 其余参数保持缺省 —— 签名里的 `axisUnit ?? ""` 因此不会给非切换参数
            // 附加一个无意义的常量。
            ...(supportsParamAxisUnit(editParam) ? { axisUnit: editParamAxisUnit } : {}),
            // 与传给 drawPianoRoll 的 fontFamily 同一个值（第 3268 行），
            // 保证两种渲染模式的字形完全一致。
            fontFamily,
            tensionLabelRgba: toRgba(colors.tensionLabel),
            tensionLineRgba: toRgba(colors.tensionLine),
        };

        // 键盘轴颜色只在音高参数下提供：非 pitch 时 GL 层据此判定"没有键盘"
        // 并清空几何（否则切到别的参数后键盘会残留在画布上）。
        if (kind !== "pitch") return base;

        // 音阶高亮：条件与旧 Canvas2D 分支逐字对齐——只在 `always` 且有工程音阶时
        // 高亮。两条路径：
        // - **分段**（有 Tempo Map 且换过音阶）：每段按自己的音阶、画自己那段 x 范围。
        //   段的时间范围在这里**投影成视口 x** 再交给 GL 层（它是纯视口坐标、不含
        //   时间轴，见 `buildPitchGridInstances` 的说明）。
        // - **均匀**（默认）：整宽一条线，只高亮工程音阶的音级。
        //
        // 范围取「可见区间 ± 5s」并按 0.02s 量化（与旧实现同一取法）：量化让滚动
        // 在 0.02s 内不改变段边界，减少几何重建；±5s 保证视口边缘的段完整。
        const highlightActive = s.scaleHighlightMode === "always";
        const segStartSec = Math.round(Math.max(0, viewportStartSec(prAxis) - 5) / 0.02) * 0.02;
        const segEndSec = Math.round((viewportEndSec(prAxis) + 5) / 0.02) * 0.02;
        const scaleSegments = highlightActive
            ? (buildScaleSegments(s.tempoMap, effectiveProjectScale, segStartSec, segEndSec) ?? [])
                  .map((segment) => ({
                      startSec: segment.startSec,
                      endSec: segment.endSec,
                      // 段没有自己的音阶（`null`）时退回工程音阶——与旧实现
                      // `buildScaleSegments` 的 `current ?? projectScale` 同一口径。
                      notes: resolveScaleNotes(segment.scale ?? effectiveProjectScale),
                  }))
                  .filter(
                      (segment) => segment.notes.length > 0 && segment.endSec > segment.startSec,
                  )
            : [];
        // 分段存在时不再走高亮的均匀路径（两者互斥，分段优先）。
        const scaleNotes =
            highlightActive && scaleSegments.length === 0
                ? resolveScaleNotes(effectiveProjectScale)
                : [];

        return {
            ...base,
            // 钢琴背景（恒开）：黑键行背景带。GL 侧保证带子在所有网格线之前发射，
            // 音阶强调线再压在其上。
            blackKeyRowBandRgba: toRgba(colors.blackKeyRowBand),
            scaleNotes,
            scaleSegments,
            scaleHighlightRgba: toRgba(colors.scaleHighlight),
            whiteKeyRgba: toRgba(colors.whiteKey),
            blackKeyRgba: toRgba(colors.blackKey),
            blackKeyGradientRgba: toRgba(colors.blackKeyGradient),
            cSeparatorRgba: toRgba(colors.cSeparator),
            keySeparatorRgba: toRgba(colors.keySeparator),
            axisBorderRgba: toRgba(colors.axisBorder),
            cLabelRgba: toRgba(colors.cLabel),
            whiteKeyLabelRgba: toRgba(colors.whiteKeyLabel),
            blackKeyLabelRgba: toRgba(colors.blackKeyLabel),
        };
    }

    /**
     * 内核数据镜像：项目时长 / 值域 / 网格输入。
     *
     * 特殊说明 1：网格输入必须每次构建——它是 GL 静态层的唯一来源（Canvas2D 侧的
     * 音高网格分支已整体删除，网格只有 GL 一个绘制者）。颜色解析会创建 DOM 探针
     * （`normalizeCssColor`），但这是唯一路径，没有可省的余地。
     *
     * 特殊说明 2（**依赖项必须覆盖 buildGridSpec 读取的全部输入**）：本 effect 是
     * 镜像里 `grid` 的**唯一**写入点，因此 `buildGridSpec` 读到的每一个输入都必须
     * 在依赖里出现，否则该输入变化时镜像不刷新、签名不变、GL 几何不重建——症状
     * 正是"点了按钮但画面纹丝不动"。因此除视口 / 参数外，还必须含：
     * - `themeMode`：决定两套配色（切主题后背景带与网格线都要换色）；
     * - `s.scaleHighlightMode`：决定音阶高亮是否产生音级集合（缺陷 #6）；
     * - `effectiveProjectScale`：换调 / 换音阶后高亮的音级集合变化。
     *
     * 实测（补依赖之前）：点击「音阶高亮」按钮后 Redux 状态确实翻转（按钮 variant
     * `ghost → solid`），但 GL 画布的像素**完全不变**——本 effect 没重跑，镜像里的
     * `grid` 还是旧对象（连带旧签名），几何自然不重建。补上依赖后同样操作能画出
     * 琥珀色强调线（像素级实测：变化行数 = 音阶音级数 × 每线设备像素行数，且再点
     * 一次回到逐像素相同）。
     */
    useLayoutEffect(() => {
        const bounds = getParamValueBoundsForScrollbar(editParam);
        const view = clampViewport(editParam, getCurrentViewportForScrollbar(editParam));
        kernelDataRef.current.projectSec = dynamicProjectSec;
        kernelDataRef.current.valueDomain = {
            min: bounds.min,
            max: bounds.max,
            span: view.span,
        };
        kernelDataRef.current.grid = buildGridSpec();
        // pxPerSec 由面板解析后写入内核（阶段 1 不迁移缩放，见宿主文件头）。
        hostRef.current?.setPxPerSec(pxPerSec);
        // 同步偏移会改变**水平上限**（原生域 = 内容宽 + 偏移），必须按新边界重钳。
        hostRef.current?.reclamp();
        hostRef.current?.invalidate();
        // `pitchViewRef` / `paramViewsRef` 是 ref（变更不触发渲染），故依赖项里无法列出；
        // 值域跨度变化由下面的显式同步点负责（`syncVerticalScrollbarForViewport`）。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        dynamicProjectSec,
        pxPerSec,
        editParam,
        processorParams,
        s.paramEditorSyncTimeline,
        timelineOffsetPx,
        // 见特殊说明 2：buildGridSpec 读到的主题 / 高亮 / 音阶都必须在此列出。
        themeMode,
        s.scaleHighlightMode,
        effectiveProjectScale,
        // 纵轴展示单位（倍率 / dB）：同样是 buildGridSpec 的输入，漏了就是
        // "点了切换但刻度纹丝不动"（签名不变 → GL 几何与文字都不重建）。
        editParamAxisUnit,
        // Tempo Map 分段音阶（缺陷 #6）：换 Tempo Map 会换分段。
        s.tempoMap,
    ]);

    // 渲染期刷新 syncScrollLeft 引用（其函数体随每次渲染重建）：供只注册一次的
    // 原生 `wheel` 监听器调用，避免闭包捕获陈旧实现。
    const syncScrollLeftRef = useRef(syncScrollLeft);
    syncScrollLeftRef.current = syncScrollLeft;

    // 【已删除：每帧对账自愈循环】
    // 该循环的存在前提是「原生 scroller 是滚动/缩放的唯一事实源」。内核是唯一渲染
    // 路径后真值源反转（内核持有、原生是被动镜像），用原生值反向对账会变成两个方向
    // 的自愈互相打架：内核刚写镜像，循环又把镜像读回来当作"权威"重发一次。内核自己
    // 每帧提交、且镜像由内核回写，故该职责整体消失。

    const valueToY = useCallback((param: ParamName, v: number, h: number): number => {
        const H = Math.max(1, h);
        if (param === "pitch") {
            const absMin = PITCH_MIN_MIDI;
            const absMax = PITCH_MAX_MIDI;
            const view = pitchViewRef.current;
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
            return (1 - t) * H;
        }

        if (isChildPitchOffsetCentsParam(param)) {
            const absMin = CHILD_PITCH_OFFSET_CENTS_RANGE.min;
            const absMax = CHILD_PITCH_OFFSET_CENTS_RANGE.max;
            const view = paramViewsRef.current[param] ?? {
                center: (absMin + absMax) / 2,
                span: absMax - absMin,
            };
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
            return (1 - t) * H;
        }

        if (isChildPitchOffsetDegreesParam(param)) {
            const absMin = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
            const absMax = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
            const view = paramViewsRef.current[param] ?? {
                center: (absMin + absMax) / 2,
                span: absMax - absMin,
            };
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
            return (1 - t) * H;
        }

        const desc = processorParamsRef.current.find((d) => d.id === param);
        const absMin = desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
        const absMax = desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
        const view = paramViewsRef.current[param] ?? {
            center: (absMin + absMax) / 2,
            span: absMax - absMin || 1,
        };
        const span = clamp(view.span, 1e-6, absMax - absMin || 1);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
        return (1 - t) * H;
    }, []);

    const yToViewportT = useCallback((y: number, h: number): number => {
        const H = Math.max(1, h);
        return clamp(y / H, 0, 1);
    }, []);

    const yToValue = useCallback((param: ParamName, y: number, h: number): number => {
        const H = Math.max(1, h);
        const t = 1 - clamp(y / H, 0, 1);
        if (param === "pitch") {
            const absMin = PITCH_MIN_MIDI;
            const absMax = PITCH_MAX_MIDI;
            const view = pitchViewRef.current;
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            return clamp(min + t * span, absMin, absMax);
        }

        if (isChildPitchOffsetCentsParam(param)) {
            const absMin = CHILD_PITCH_OFFSET_CENTS_RANGE.min;
            const absMax = CHILD_PITCH_OFFSET_CENTS_RANGE.max;
            const view = paramViewsRef.current[param] ?? {
                center: (absMin + absMax) / 2,
                span: absMax - absMin,
            };
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            return clamp(min + t * span, absMin, absMax);
        }

        if (isChildPitchOffsetDegreesParam(param)) {
            const absMin = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
            const absMax = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
            const view = paramViewsRef.current[param] ?? {
                center: (absMin + absMax) / 2,
                span: absMax - absMin,
            };
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            return clamp(min + t * span, absMin, absMax);
        }

        const desc = processorParamsRef.current.find((d) => d.id === param);
        const absMin = desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
        const absMax = desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
        const view = paramViewsRef.current[param] ?? {
            center: (absMin + absMax) / 2,
            span: absMax - absMin || 1,
        };
        const span = clamp(view.span, 1e-6, absMax - absMin || 1);
        const min = clamp(view.center - span / 2, absMin, absMax - span);
        return clamp(min + t * span, absMin, absMax);
    }, []);

    function clampViewport(param: ParamName, v: ValueViewport): ValueViewport {
        if (param === "pitch") {
            const absMin = PITCH_MIN_MIDI;
            const absMax = PITCH_MAX_MIDI;
            const span = clamp(v.span, 6, absMax - absMin);
            const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
            return { center, span };
        }
        if (isChildPitchOffsetCentsParam(param)) {
            const absMin = CHILD_PITCH_OFFSET_CENTS_RANGE.min;
            const absMax = CHILD_PITCH_OFFSET_CENTS_RANGE.max;
            const span = clamp(v.span, 100, absMax - absMin);
            const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
            return { center, span };
        }
        if (isChildPitchOffsetDegreesParam(param)) {
            const absMin = CHILD_PITCH_OFFSET_DEGREES_RANGE.min;
            const absMax = CHILD_PITCH_OFFSET_DEGREES_RANGE.max;
            const span = clamp(v.span, 1, absMax - absMin);
            const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
            return { center, span };
        }
        const desc = processorParamsRef.current.find((d) => d.id === param);
        const absMin = desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
        const absMax = desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
        const range = Math.max(1e-6, absMax - absMin);
        const span = clamp(v.span, range * 0.05, range);
        const center = clamp(v.center, absMin + span / 2, absMax - span / 2);
        return { center, span };
    }

    function getParamValueBoundsForScrollbar(param: ParamName): { min: number; max: number } {
        if (param === "pitch") {
            return { min: PITCH_MIN_MIDI, max: PITCH_MAX_MIDI };
        }
        if (isChildPitchOffsetCentsParam(param)) {
            return {
                min: CHILD_PITCH_OFFSET_CENTS_RANGE.min,
                max: CHILD_PITCH_OFFSET_CENTS_RANGE.max,
            };
        }
        if (isChildPitchOffsetDegreesParam(param)) {
            return {
                min: CHILD_PITCH_OFFSET_DEGREES_RANGE.min,
                max: CHILD_PITCH_OFFSET_DEGREES_RANGE.max,
            };
        }

        const desc = processorParamsRef.current.find((d) => d.id === param);
        if (desc?.kind.type === "automation_curve") {
            return {
                min: desc.kind.min_value,
                max: desc.kind.max_value,
            };
        }
        // `dyn_edit` 是动态的历史别名（后端描述符只有 "dyn"）。若在这里退化成
        // {0,1}，动态面板的值域会从 0..4 塌成 0..1，网格与轴线全部按错误值域
        // 绘制（表现为"动态显示成了音量的标尺"的另一种形态）。它与 dyn 同值域。
        if (isDynParam(param)) {
            return { min: 0, max: DYN_VALUE_MAX };
        }
        return { min: 0, max: 1 };
    }

    function getCurrentViewportForScrollbar(param: ParamName): ValueViewport {
        if (param === "pitch") {
            return pitchViewRef.current;
        }

        const bounds = getParamValueBoundsForScrollbar(param);
        return (
            paramViewsRef.current[param] ??
            // 默认视口按参数语义分流（见 paramRanges 的两个常量说明）：
            // - dyn：0..1.25，0 dB 在 80% 高度 —— 面板主体留给 −∞..0 dB 的
            //   编辑区间（目标电平几乎不会超过 0 dB）；
            // - volume：0..2，1.0 在中线（>1 的提升是常态操作）。
            // 自定义过的视口（paramViewsRef 已有记录）原样尊重。
            (isDynParam(param)
                ? { center: DYN_DEFAULT_VIEW.center, span: DYN_DEFAULT_VIEW.span }
                : param === "volume"
                  ? { center: VOLUME_DEFAULT_VIEW.center, span: VOLUME_DEFAULT_VIEW.span }
                  : {
                        center: (bounds.min + bounds.max) / 2,
                        span: Math.max(1e-6, bounds.max - bounds.min),
                    })
        );
    }

    /**
     * 把值域视口同步到竖向滚动位置（值域 → 像素）。
     *
     * 流程：钳制视口 → 更新内核数据镜像的值域 → 把 `center` 交给滚动载体。
     *
     * 特殊说明 1：滚动载体是内核宿主。像素域恒为 0..1600，映射由
     * `verticalScrollTopFromCenter` 单一来源保证。宿主内部自带去重，故无需在这里
     * 再判一次差值。宿主尚未创建时（挂载期）退回原生 scroller，语义一致。
     *
     * 特殊说明 2（**必须先刷新镜像的 span**）：竖向缩放（钢琴键区 alt/ctrl+滚轮）
     * 只改 `span`，而视口存在 **ref** 里——ref 变更不触发 React 渲染，于是「内核数据
     * 镜像」那个 effect 不会重跑，内核读到的 `valueDomain.span` 仍是旧值。此时
     * `setValueCenter` 会按**旧 span** 反算像素位置，结果完全错位（实测：缩放后
     * nativeTop 直接掉到 0 且再也动不了）。因此这里在提交前就地刷新镜像。
     *
     * @param param 参数名（决定值域边界）。
     * @param view 目标值域视口。
     */
    function syncVerticalScrollbarForViewport(param: ParamName, view: ValueViewport): void {
        const clampedView = clampViewport(param, view);
        const bounds = getParamValueBoundsForScrollbar(param);

        const host = hostRef.current;
        if (host) {
            // 就地刷新值域镜像（见特殊说明 2）。
            kernelDataRef.current.valueDomain = {
                min: bounds.min,
                max: bounds.max,
                span: clampedView.span,
            };
            host.setValueCenter(clampedView.center);
            return;
        }

        const scroller = scrollerRef.current;
        if (!scroller) return;

        const nextTop = verticalScrollTopFromCenter({
            min: bounds.min,
            max: bounds.max,
            span: clampedView.span,
            center: clampedView.center,
            scrollRangePx: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
        });

        if (Math.abs(scroller.scrollTop - nextTop) > 0.75) {
            scroller.scrollTop = nextTop;
            lastMirroredScrollTopRef.current = nextTop;
        }
    }

    /**
     * 【已删除：`applyViewportFromVerticalScrollbar`（像素 → 值域）】
     *
     * 它的唯一调用者是原生 `scroll` 事件里的竖向对账分支，而那段的成立前提是
     * 「原生 scroller 是竖向事实源」。内核是唯一渲染路径后竖向由内核拥有，值域视口
     * 由宿主的 `onScrollTopFrame` **正向**回写（见上方内核宿主创建处的说明），
     * 反向采纳路径失去存在理由，故随分支一并删除。值域 → 像素方向仍由
     * `syncVerticalScrollbarForViewport` 提供。
     */

    /** 多选区（升序、互不相交、相邻已合并的**帧**区间列表；null = 无选区） */
    const selectionRef = useRef<ParamSelection | null>(null);
    // 记录打开 MIDI 弹窗时的 editParam / toolMode 快照，避免异步加载轨道期间 Redux 状态变化导致 selectionAvailable 跳变
    const midiDialogOpenParamsRef = useRef<{
        editParam: string;
        toolMode: string;
    }>({ editParam: "pitch", toolMode: "select" });
    const [selectionUi, setSelectionUi] = useState<ParamSelection | null>(null);
    // 参数线选区存在性入仓（session.paramSelectionActive）：复制/剪切按
    // "当前选中了什么"路由时以此判定参数侧（selectionRef 的每次变更都
    // 成对经过 setSelectionUi，见 usePianoRollInteractions）。拖拽期间的
    // 重复派发值不变，immer 判定无修改直接跳过。
    useEffect(() => {
        dispatch(setParamSelectionActive(selectionUi !== null));
    }, [dispatch, selectionUi]);
    // 撤销/重做/跳转恢复参数编辑器选区：请求由后端载荷驱动 —— 只有「边缘拉伸」
    // 这类同时改变选区的步骤才在后端记了快照（见 state.rs 的
    // `HistoryRecord::param_selection`），撤销/重做时随载荷带回；其余操作不带该
    // 字段，因此**不会**动用户手动调整过的选区。requestId 单调，按 id 幂等应用一次。
    const appliedParamSelectionRestoreRef = useRef(0);
    useEffect(() => {
        const request = s.pendingParamSelectionRestore;
        if (!request || request.requestId === appliedParamSelectionRestoreRef.current) return;
        appliedParamSelectionRestoreRef.current = request.requestId;
        const next = normalizeSelection(request.selection);
        selectionRef.current = next;
        setSelectionUi(next);
        invalidate();
    }, [invalidate, s.pendingParamSelectionRestore]);
    const [paramMorphOverlays, setParamMorphOverlays] = useState<ParamMorphOverlay[] | null>(null);
    const [canvasCursor, setCanvasCursor] = useState<CSSProperties["cursor"]>(
        s.toolMode === "select" ? "default" : "crosshair",
    );

    const strokeRef = useRef<{
        mode: StrokeMode;
        pointerId: number;
        param: ParamName;
        points: StrokePoint[];
    } | null>(null);

    const panRef = useRef<{
        pointerId: number;
        startClientX: number;
        startClientY: number;
        startScrollLeft: number;
        startView: ValueViewport;
        startRectH: number;
    } | null>(null);

    /**
     * 参数线剪贴板（多选区）：每段带**相对复制起点**的帧偏移，断层以偏移空洞
     * 的形式保留 —— 粘贴/预览按 paramClipboardMapping 的唯一映射规则求交。
     */
    const clipboardRef = useRef<ParamClipboardData | null>(null);

    // 将 PianoRoll 加载状态同步到全局 Context（供 status bar 使用）
    const updatePianoRollStatus = usePianoRollStatusUpdate();

    // 用于通知 usePianoRollData 当前是否处于 live 编辑状态（pointer down 期间 ?true） ?
    // pitch_orig_updated 事件到达时若 ?true，则延迟曲线刷新 ?pointer-up 后执行 ?
    const liveEditActiveRef = useRef(false);

    const {
        paramView,
        setParamView,
        secondaryParamViews,
        referencePitchViews,
        bumpRefreshToken,
        refreshToken,
        refreshNow,
        refreshSecondaryNow,
        notifyLiveEditEnded,
        isLoading,
    } = usePianoRollData({
        editParam,
        secondaryParamIds: visibleSecondaryParamIds,
        referenceRootTrackIds: visibleReferenceRootTrackIds,
        pitchEnabled,
        paramsEpoch: (s as unknown as { paramsEpoch?: number }).paramsEpoch ?? 0,
        rootTrackId,
        selectedTrackId: effectiveSelectedTrackId,
        scrollLeft,
        // 视口窗口只需要「秒 ↔ 像素」这一个系数。此前传的是 `pxPerBeat` +
        // `secPerBeat`（两者相乘才等于它），于是**改 BPM 会让取数 effect 重跑**、
        // 整条曲线白取一次；换成 pxPerSec 后 BPM 与取数彻底解耦。
        pxPerSec,
        viewWidth: viewSize.w,
        viewSizeRef,
        scrollLeftRef,
        pxPerSecRef,
        invalidate,
        liveEditActiveRef,
    });

    /**
     * 波形「可听结果」映射所需的响度自动化快照（整条工程的 volume 曲线 +
     * 动态目标/原声基线）。
     *
     * 【为什么独立于 paramView】波形的形变与"当前编辑哪个参数"无关 —— 用户在
     * **任何**参数面板都要实时看到音频波形（画一笔音量/动态曲线波形立刻跟着
     * 动）。paramView 只覆盖当前参数且随视口窗口化，撑不起这个语义；快照按
     * 整工程自适应 stride 拉取，滚动/缩放零重取（见 useLoudnessCurves）。
     */
    const loudnessFpMs = paramView?.framePeriodMs ?? 5;
    const loudnessProjectFrames = Math.max(1, Math.ceil((dynamicProjectSec * 1000) / loudnessFpMs));
    const {
        snapshot: loudnessSnapshot,
        analysisPending: loudnessAnalysisPending,
        snapshotFetchSeq: loudnessSnapshotFetchSeq,
        getLatestFetchSeq: getLatestLoudnessFetchSeq,
    } = useLoudnessCurves({
        rootTrackId,
        projectFrames: loudnessProjectFrames,
        framePeriodMs: loudnessFpMs,
        paramsEpoch: (s as unknown as { paramsEpoch?: number }).paramsEpoch ?? 0,
        refreshToken,
    });

    /**
     * live 覆盖读取器（解析按覆盖对象身份缓存，见 `createLiveOverrideReader`）。
     *
     * 【为什么必须缓存】`readLiveOverrideFor` 在波形几何重建的热路径上被反复
     * 调用 —— 几何层按列内增益切片调用幅度映射，一次重建可达数万次（实测
     * 2112 列 × 16 切片 × 2 = 6.8 万次）。若每次询问都 `split` key 再新建视图
     * 对象，仅字符串切分就要吃掉 ~26ms/帧 —— 用户报告的「编辑音量/动态时
     * 卡顿」。读取器把稳态压缩成一次身份比较，零分配。
     */
    const liveOverrideReaderRef = useRef<LiveOverrideReader | null>(null);
    if (liveOverrideReaderRef.current === null) {
        liveOverrideReaderRef.current = createLiveOverrideReader();
    }

    /**
     * 把 live 覆盖读成 LoudnessLiveCurve（按 key 中的参数 id 与窗口对齐）。
     *
     * 键解析与视图对象由 `createLiveOverrideReader` 按覆盖对象身份缓存。
     */
    const readLiveOverrideFor = useCallback((param: "volume" | "dyn") => {
        return liveOverrideReaderRef.current?.read(param, liveEditOverrideRef.current) ?? null;
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 两个 ref 都是稳定引用（reader 在下方惰性创建、live 覆盖由 useLiveParamEditing 持有并就地更新）；加入依赖不会让回调更正确，反而会在每次渲染换掉引用、使幅度映射的几何缓存键失效
    }, []);

    /**
     * 波形响度映射的修订号（ref，**不触发 React 渲染**）。
     *
     * 绘制中的 live 覆盖写在 ref 上，幅度映射内部用延迟取值读它，因此函数引用
     * 不变也能产出不同结果 —— 几何缓存若只按引用比较会误判"没变"，
     * 绘制中的曲线就画不上去。这里用 ref 累加计数、映射通过 `revision()`
     * 惰性读取：若改成 state，拖动时每次 pointermove 都会重渲染整块面板。
     */
    const loudnessWaveformRevisionRef = useRef(0);

    /**
     * 波形重绘的**帧内合并**调度（与曲线层同用 `renderKernel/renderLoop` 语义）。
     *
     * 【为什么需要】一次全量重建要重算两千余列 × 16 切片的包络并上传数百 KB
     * 顶点（实测典型窗口 ~2.7ms）。而绘制中的 live 覆盖在**同一帧内可能被更新
     * 多次**：手绘工具会把一个 `pointermove` 的 coalesced 采样全部展开处理
     * （`usePianoRollInteractions` 的 `flushPendingMoves`），高刷鼠标 / 笔一帧
     * 可积 2~8 个采样。此前每个采样都**同步**强制提交一次全量重建 —— 单帧
     * 8~30ms 的几何工作，正是用户报告的"编辑音量/动态时卡顿"。
     *
     * 改为标脏 + rAF 合并后一帧至多重建一次，且与曲线层（面板自己的
     * `invalidate()` → 宿主 renderLoop）落在**同一帧**，两层不再错帧。
     *
     * 【为什么不用同帧 `flush()`】滚动路径要求同帧提交，是因为要与原生滚动的
     * DOM 内容层对齐（见 PianoRollWaveformSurface / viewportBus 的契约）。绘制
     * 参数曲线的路径没有随滚动移动的 DOM 层，与曲线层同帧即可。
     */
    const waveformRepaintLoopRef = useRef<RenderLoop | null>(null);
    if (waveformRepaintLoopRef.current === null) {
        waveformRepaintLoopRef.current = createRenderLoop({
            draw: () => pianoRollViewportBus.invalidate(),
        });
    }
    useEffect(() => {
        const loop = waveformRepaintLoopRef.current;
        loop?.start();
        return () => loop?.stop();
    }, []);

    /**
     * 强制重绘参数面板波形（**仅当绘制中的 live 覆盖会改变波形时**）。
     *
     * 必要性：绘制中的响度曲线只写在 `liveEditOverrideRef`（ref 变更不触发
     * React 渲染），而波形面是 memo 组件 + 几何缓存，既收不到 ref 变更、
     * 也不会因 props 未变而重绘。故显式强制重绘一次：总线以同一份投影 force
     * commit，波形图层命中重绘并因修订号变化而重建几何（其余图层内容未变，
     * 开销可忽略）。
     *
     * 【为什么必须按参数过滤】波形画的是「可听结果」
     * （`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`，见
     * `makeLoudnessAmplitudeMap`）—— 它**不依赖音高 / 共振峰 / 齿度**等参数。
     * 而波形面的几何重建在总线驱动下是**每次全量**（`WaveformSurface.draw`：
     * 总线驱动的 `canReuse` 恒为 false，见该函数的说明），一次重建要重算
     * 两千余列 × 16 切片的包络。绘制音高时逐帧触发它纯属浪费 —— 一次重建
     * 在参数面板的典型窗口下约 2.7ms（响度映射），叠加曲线自身重绘后会
     * 明显抬高指针帧成本。因此这里用当前 live 覆盖的参数 id 做闸门：
     * 只有编辑 volume / dyn 时才推进修订号并请求重绘。
     *
     * 【与曲线绘制的关系】曲线（选区、绘制中的参数线）由面板自己的
     * `invalidate()` → 宿主帧提交驱动，**不经过本函数**；因此本闸门只影响波形
     * 面，绘制音高时曲线仍然逐帧更新。
     */
    const requestWaveformRepaint = useCallback(() => {
        // 闸门：只有编辑 volume / dyn 时波形才可能变（波形画的是可听结果，
        // 与音高 / 共振峰等参数无关）。判定复用读取器的解析缓存，零分配。
        if (!liveOverrideReaderRef.current?.affectsWaveform(liveEditOverrideRef.current)) {
            return;
        }
        // 修订号在**请求时**推进（不是绘制时）：同一帧内其它路径触发的绘制也要
        // 看到"上次重建之后 live 覆盖变过"，否则几何缓存会误判为可复用。
        loudnessWaveformRevisionRef.current += 1;
        // 绘制请求帧内合并：同一帧内多次请求只重绘一次（见 waveformRepaintLoop）。
        waveformRepaintLoopRef.current?.invalidate();
        // eslint-disable-next-line react-hooks/exhaustive-deps -- liveOverrideReaderRef / waveformRepaintLoopRef 是稳定引用（惰性创建）；live 覆盖经 ref 读取，故本回调必须保持引用稳定（它进入 usePianoRollInteractions 的依赖链）
    }, []);

    /**
     * 参数面板波形的幅度映射（**所有参数统一**）。
     *
     * 把源波形画成**应用响度自动化之后的可听结果**
     * （`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`）：
     * - 编辑任何参数时画一笔音量/动态曲线，波形立刻跟着起伏；
     * - 切换参数时波形保持稳定（映射不依赖 editParam），只有曲线叠加层变化；
     * - 音量↔动态互转前后波形逐像素不变 —— 等效性最直观的佐证。
     *
     * 【引用稳定性】该值参与 `WaveformSurface` 的几何缓存键，因此必须 memo：
     * 只在快照对象真正变化时换引用。绘制中的 live 覆盖不改变本对象
     * （延迟取值 + 修订号），由 `requestWaveformRepaint` 驱动重建。
     *
     * 【恒等快化】volume 恒 1 且动态无基线（未使用响度自动化的工程）时
     * `identity = true` → 不挂映射，与时间线及既有行为逐像素一致。
     */
    const pianoRollAmplitudeMap = useMemo(() => {
        if (!loudnessSnapshot || loudnessSnapshot.identity) return undefined;
        if (loudnessSnapshot.volume.length === 0) return undefined;
        return makeLoudnessAmplitudeMap(
            loudnessSnapshot,
            {
                volume: () => readLiveOverrideFor("volume"),
                dyn: () => readLiveOverrideFor("dyn"),
            },
            () => loudnessWaveformRevisionRef.current,
        );
    }, [loudnessSnapshot, readLiveOverrideFor]);

    const refreshSecondaryNowRef = useRef(refreshSecondaryNow);
    useEffect(() => {
        refreshSecondaryNowRef.current = refreshSecondaryNow;
    }, [refreshSecondaryNow]);

    useEffect(() => {
        if (!rootTrackId) {
            invalidate();
            return;
        }
        if (visibleSecondaryParamIds.length > 0 || visibleReferenceRootTrackIds.length > 0) {
            void refreshSecondaryNowRef.current();
            return;
        }
        invalidate();
    }, [invalidate, rootTrackId, visibleReferenceRootTrackIds, visibleSecondaryParamIds]);

    const handleMidiImported = useCallback(() => {
        refreshNow();
    }, [refreshNow]);

    const handleImportAsClip = useCallback(
        (result: {
            trackIndices: number[];
            notesCount: number;
            midiPath: string;
            fillGaps: boolean;
            multiTrackMerge?: boolean;
            noteBpmMode?: string;
            specifiedBpm?: number;
            importBpmAsProject?: boolean;
            importAsTempoMap?: boolean;
            importTempo?: boolean;
            importTimeSignature?: boolean;
            importKeySignature?: boolean;
            clipboardGuid?: string;
            closeLeadingGap?: boolean;
        }) => {
            void dispatch(
                importMidiAsClip({
                    midiPath: result.midiPath,
                    trackIndices: result.trackIndices,
                    trackId: s.selectedTrackId,
                    startSec: s.playheadSec,
                    fillGaps: result.fillGaps || undefined,
                    multiTrackMerge: result.multiTrackMerge,
                    noteBpmMode: result.noteBpmMode,
                    specifiedBpm: result.specifiedBpm,
                    importBpmAsProject: result.importBpmAsProject,
                    clipboardGuid: result.clipboardGuid,
                    closeLeadingGap: result.closeLeadingGap,
                    importAsTempoMap: result.importAsTempoMap,
                    importTempo: result.importTempo,
                    importTimeSignature: result.importTimeSignature,
                    importKeySignature: result.importKeySignature,
                }),
            );
        },
        [dispatch, s.selectedTrackId, s.playheadSec],
    );

    // 导入位置变更时持久化保存
    const handleImportPositionChange = useCallback((position: string) => {
        setImportPosition(position);
        void settingsApi.saveUiSettings({ midiImportPosition: position });
    }, []);

    // 填补空隙选项变更时持久化保存
    const handleFillGapsChange = useCallback((value: boolean) => {
        setFillGaps(value);
        void settingsApi.saveUiSettings({ midiFillGaps: value });
    }, []);

    // BPM 选项变更时持久化保存
    const handleImportBpmAsProjectChange = useCallback((v: boolean) => {
        setImportBpmAsProject(v);
        void settingsApi.saveUiSettings({ midiImportBpmAsProject: v });
    }, []);

    const handleNoteBpmModeChange = useCallback((v: string) => {
        setNoteBpmMode(v);
        void settingsApi.saveUiSettings({ midiNoteBpmMode: v });
    }, []);

    const handleSpecifiedBpmChange = useCallback((v: number) => {
        setSpecifiedBpm(v);
        void settingsApi.saveUiSettings({ midiSpecifiedBpm: v });
    }, []);

    const handleMultiTrackMergeChange = useCallback((v: boolean) => {
        setMultiTrackMerge(v);
        void settingsApi.saveUiSettings({ midiMultiTrackMerge: v });
    }, []);

    const handleCloseLeadingGapChange = useCallback((v: boolean) => {
        setCloseLeadingGap(v);
        void settingsApi.saveUiSettings({ midiCloseLeadingGap: v });
    }, []);

    const handleImportTempoMapEnabledChange = useCallback((v: boolean) => {
        setImportTempoMapEnabled(v);
        void settingsApi.saveUiSettings({ midiImportAsTempoMap: v });
    }, []);
    const handleImportTempoMapTempoChange = useCallback((v: boolean) => {
        setImportTempoMapTempo(v);
        void settingsApi.saveUiSettings({ midiImportTempoMapTempo: v });
    }, []);
    const handleImportTempoMapTimeSignatureChange = useCallback((v: boolean) => {
        setImportTempoMapTimeSignature(v);
        void settingsApi.saveUiSettings({ midiImportTempoMapTimeSignature: v });
    }, []);
    const handleImportTempoMapKeySignatureChange = useCallback((v: boolean) => {
        setImportTempoMapKeySignature(v);
        void settingsApi.saveUiSettings({ midiImportTempoMapKeySignature: v });
    }, []);

    const handleImportTargetChange = useCallback((v: string) => {
        if (midiDialogSourceRef.current === "reaperClipboard") {
            setImportTargetReaperClipboard(v);
            void settingsApi.saveUiSettings({ midiImportTargetReaperClipboard: v });
        } else {
            setImportTargetParamEditor(v);
            void settingsApi.saveUiSettings({ midiImportTargetParamEditor: v });
        }
    }, []);

    const handleRequestEnableCompose = useCallback(() => {
        const rtId = rootTrackId;
        if (!rtId) return;
        dispatch(
            setTrackStateRemote({
                trackId: rtId,
                composeEnabled: true,
            }),
        );
    }, [dispatch, rootTrackId]);

    // 计算 MIDI 导入的多选区帧约束：与参数编辑器选区逐段一致，导入只写入
    // 落在任一段内的帧（断层保持原值），对齐基准为首段起点。
    const midiSelRanges = useMemo(() => {
        if (!midiDialogSelection) return undefined;
        const ranges = selectionToFrameRanges(midiDialogSelection);
        return ranges.length > 0 ? ranges : undefined;
    }, [midiDialogSelection]);

    // selection 导入模式是否可用（基于弹窗打开时的快照，避免异步加载轨道时状态变化）
    const midiSelectionAvailable = useMemo(() => {
        if (!midiDialogSelection) return false;
        const p = midiDialogOpenParamsRef.current;
        return p.editParam === "pitch" && p.toolMode === "select";
    }, [midiDialogSelection]);

    // 将当前选区发布到总线，供 MenuBar 等判断“工程音阶”是否受 Tempo Map 影响。
    // 多选区发布**包围区间**（首段起点 → 末段终点）：该提示只关心"选区覆盖的
    // 时间跨度里有没有音阶变化点"，包围区间是它的保守上界。
    useEffect(() => {
        const sel = selectionUi;
        const bounding = selectionBoundingSpan(sel);
        if (!bounding) {
            publishPianoRollSelection(null);
            return;
        }
        // 选区已经是帧制：直接发布，不再有任何换算（此前要经 `secPerBeat` 转一道）。
        publishPianoRollSelection({
            startFrame: Math.max(0, bounding.startFrame),
            frameCount: Math.max(1, bounding.frameCount),
            framePeriodMs: paramView?.framePeriodMs ?? 5,
        });
        return () => {
            publishPianoRollSelection(null);
        };
    }, [selectionUi, paramView?.framePeriodMs]);

    // 获取当前 track 下的所 ?clips，用 ?per-clip 波形叠加绘制
    // 获取轨道组内所有 clips（包含 root 轨道及所有子轨道的 clip）
    const trackClips = useMemo(
        () => s.clips.filter((c) => groupTrackIds.has(c.trackId)),
        [s.clips, groupTrackIds],
    );

    /**
     * 参数编辑器的统一坐标投影（渲染期）。
     *
     * 用 React state 的 scrollLeft / pxPerSec 构造：本面板所有**渲染期**的时间↔
     * 像素换算都走它。滚动热路径另用 ref 构造 `drawAxis`（见 drawRef），因为
     * 滚动时 ref 同步更新而 state 滞后一帧。
     *
     * 注意：`scrollLeft` 是**绘制坐标**（由 timelineViewportNativeToState 转换
     * 得到），与 axis 的契约一致；DOM 原生坐标只在量测边界转换。
     */
    const prAxis = useMemo(
        () =>
            createTimelineAxis({
                pxPerSec,
                scrollLeftPx: scrollLeft,
                viewportWidthPx: viewSize.w,
                dpr: window.devicePixelRatio || 1,
            }),
        [pxPerSec, scrollLeft, viewSize.w],
    );

    // 可见区域的 sec 范围：仅用于数据窗口选择（clip peaks 取窗），
    // 像素换算一律经 prAxis，不得再由这里除回秒。
    const visibleStartSec = viewportStartSec(prAxis);
    const visibleEndSec = viewportEndSec(prAxis);

    // Per-clip 波形 peaks（替代原来的 mix 波形）
    const clipPeaks = useClipsPeaksForPianoRoll({
        clips: trackClips,
        visibleStartSec,
        visibleEndSec,
    });
    // Data and viewport changes should always trigger a canvas redraw.
    // usePianoRollData() may call invalidate() before these refs update,
    // so we schedule a follow-up redraw after React commits state.
    // clipPeaks 已经通过 useMemo 稳定化，只在数据真正变化时才产生新引用。
    useEffect(() => {
        invalidate();
    }, [clipPeaks, paramView, secondaryParamViews, pxPerBeat, viewSize.w, viewSize.h, invalidate]);

    useEffect(() => {
        invalidate();
    }, [editParam, visibleSecondaryParamIds, themeMode, invalidate]);

    // 检测音高曲线更新时触发重绘（必须在 detectedPitchCurves 声明之后 ?
    // useEffect 已移 ?detectedPitchCurves useMemo 定义之后，见下方 ?

    const paramViewRef = useRef<import("./pianoRoll/types").ParamViewSegment | null>(null);
    useEffect(() => {
        paramViewRef.current = paramView;
    }, [paramView]);

    const {
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        clearCommittedLiveEditOverride,
        markLiveEditCommitted,
        commitStroke: commitStrokeBase,
    } = useLiveParamEditing({
        rootTrackId,
        editParam,
        pitchEnabled,
        paramView,
        setParamView,
        bumpRefreshToken,
        invalidate,
    });

    /**
     * 提交时记下的"已发出的最大取数序号"（见 `useLoudnessCurves` 的
     * `getLatestFetchSeq` 与下方收尾 effect）。
     */
    const committedSettleSeqRef = useRef(0);

    /**
     * **参数写入成功**后调用：让波形在快照追上之前继续显示刚提交的值。
     *
     * 【做两件事】① 记下当前的取数序号水位（只有序号更大的快照才可能是提交之后
     * 取的）；② 把 live 覆盖层标记为"已提交"—— **保留**其值而不是立刻撤下，
     * 避免波形的幅度因子退回旧快照（"松手闪回旧波形"）。
     *
     * 【谁调用】面板的 `commitStroke` 包装层，以及 `usePianoRollInteractions` 里
     * 四条**自己发起回写**的提交路径（选区拖拽 / 右键拖拽 / 直线拖拽 / morph
     * 应用）—— 它们不经过 `commitStroke`，此前都是立刻把覆盖层置空，正是闪屏的
     * 来源。失败路径仍走硬清除（见各自的 catch）。
     */
    const onParamCommitSucceeded = useCallback(() => {
        committedSettleSeqRef.current = getLatestLoudnessFetchSeq();
        markLiveEditCommitted();
    }, [getLatestLoudnessFetchSeq, markLiveEditCommitted]);

    /**
     * **提交之后取得**的响度快照到位 ⇒ 撤下"已提交"的 live 覆盖层。
     *
     * 【为什么要等】提交成功后覆盖层不立刻撤下（见 `LiveEditOverride.committed`）：
     * 它的值就是刚提交的曲线，而整工程快照还要走一趟 IPC。若此时撤下，波形的幅度
     * 因子退回**旧快照**，于是"先跳回旧波形、再恢复新波形"（用户报告的松手闪屏）。
     *
     * 【为什么用取数序号而不是"快照换了引用"】提交发生时可能有一次**提交之前就
     * 已发出**的在飞取数（例如原声基线分析完成触发的刷新），它的数据里没有本次
     * 编辑。只看"引用变了"会在这份陈旧快照到达时提前收尾、照样闪一下。比较取数
     * 序号则可以排除它：那种请求的序号 ≤ 提交时的最大值。
     *
     * 【为什么不会挡住撤销】撤销/重做经 `paramsEpoch`、切轨经 `rootTrackId`，
     * 都会重新取数并带来更大的序号，因此覆盖层的存活期最多到"下一个提交后的快照
     * 到达"，不会长期遮挡后续状态。
     */
    useEffect(() => {
        if (loudnessSnapshotFetchSeq > committedSettleSeqRef.current) {
            clearCommittedLiveEditOverride();
        }
    }, [loudnessSnapshotFetchSeq, clearCommittedLiveEditOverride]);

    // Clip 音高拖拽（修饰键 + 波形垂直拖拽）的实时预览桥：拖拽侧以节流
    // 后端预览写入修改 pitch 参数线；这里把同一音分偏移实时应用到本机
    // live 覆盖层，画布立即渲染拖拽结果 —— 与选择工具在选区内垂直拖拽的
    // 实时编辑同一渲染路径。仅当前参数为"音高"时生效（其余参数无法展示
    // 音高曲线；选区设定仍由 selectClipParamRange 完成）。
    // 提交事件把最终偏移就地写入 paramView 并清除覆盖层，与拖拽侧的
    // epoch 重取数无缝衔接（同键同值，无闪跳）。
    useEffect(() => {
        function readDragWindow(e: Event): {
            cents: number;
            startFrame: number;
            frameCount: number;
        } | null {
            const detail = (
                e as CustomEvent<{
                    cents?: number;
                    startFrame?: number;
                    frameCount?: number;
                }>
            ).detail;
            if (!detail) return null;
            const cents = Number(detail.cents ?? 0);
            const startFrame = Math.max(0, Math.floor(Number(detail.startFrame ?? 0)));
            const frameCount = Math.max(1, Math.floor(Number(detail.frameCount ?? 1)));
            if (!Number.isFinite(cents) || !Number.isFinite(startFrame)) return null;
            return { cents, startFrame, frameCount };
        }
        function applyPitchDragPreview(e: Event) {
            const drag = readDragWindow(e);
            if (!drag || !paramView || editParam !== "pitch" || !pitchEnabled) return;
            ensureLiveEditBase(paramView);
            const override = liveEditOverrideRef.current;
            if (!override || override.key !== paramView.key) return;
            const deltaSemitones = drag.cents / 100;
            const windowEndFrame = drag.startFrame + drag.frameCount;
            // 以 paramView 的原始帧为基准重复推导（预览事件幂等，不叠加），
            // 再走**统一的 live 写入入口**：它负责值域钳制（与后端同构）、版本号
            // 推进与区间记账。此前这里直接改数组 —— 既不钳制（音高拖到 127 以上
            // 时预览值与后端存下的值不一致），也不推进版本号（主画布签名与波形
            // 几何缓存都看不到变化）。
            const dense = new Array<number>(drag.frameCount);
            for (let i = 0; i < override.edit.length; i += 1) {
                const frame = paramView.startFrame + i * paramView.stride;
                if (frame < drag.startFrame || frame >= windowEndFrame) continue;
                dense[frame - drag.startFrame] = shiftPitchValue(
                    paramView.edit[i] ?? 0,
                    deltaSemitones,
                );
            }
            // 步长 > 1 时 dense 会有空洞：写入侧 `dense[j] ?? edit[i]` 会保留原值。
            applyDenseToLiveEdit(
                paramView,
                drag.startFrame,
                dense,
                drag.startFrame,
                windowEndFrame - 1,
                "draw",
            );
            invalidate();
        }
        function commitPitchDragPreview(e: Event) {
            const drag = readDragWindow(e);
            if (!drag || !paramView || editParam !== "pitch" || !pitchEnabled) return;
            const deltaSemitones = drag.cents / 100;
            const windowEndFrame = drag.startFrame + drag.frameCount;
            const nextEdit = paramView.edit.slice();
            for (let i = 0; i < nextEdit.length; i += 1) {
                const frame = paramView.startFrame + i * paramView.stride;
                if (frame < drag.startFrame || frame >= windowEndFrame) continue;
                // 与后端写入值域同构地钳制，保证"本地曲线"与"随后取回的曲线"一致。
                nextEdit[i] = clampParamWriteValue(
                    "pitch",
                    shiftPitchValue(paramView.edit[i] ?? 0, deltaSemitones),
                );
            }
            setParamView({ ...paramView, edit: nextEdit });
            liveEditOverrideRef.current = null;
            invalidate();
        }
        window.addEventListener("hifi:pitchDragPreview", applyPitchDragPreview);
        window.addEventListener("hifi:pitchDragCommit", commitPitchDragPreview);
        return () => {
            window.removeEventListener("hifi:pitchDragPreview", applyPitchDragPreview);
            window.removeEventListener("hifi:pitchDragCommit", commitPitchDragPreview);
        };
    }, [
        paramView,
        editParam,
        pitchEnabled,
        ensureLiveEditBase,
        liveEditOverrideRef,
        setParamView,
        invalidate,
        applyDenseToLiveEdit,
    ]);

    // 包装 commitStroke：在 pointer-up 提交笔画后，清除 liveEditActive 状态，
    // 并触发可能被延迟 ?pitch_orig_updated 曲线刷新 ?
    const commitStroke: typeof commitStrokeBase = useCallback(
        async (points, mode) => {
            // 收尾分工：**水位**记在这里（必须在发起写入之前 —— 提交内部会 bump
            // 刷新令牌、立刻发起取数）；**标记已提交**由 commitStrokeBase 在成功
            // 路径完成（失败则硬清除覆盖层）。两者合起来即 onParamCommitSucceeded
            // 的语义，见其说明。
            committedSettleSeqRef.current = getLatestLoudnessFetchSeq();
            await commitStrokeBase(points, mode);
            liveEditActiveRef.current = false;
            notifyLiveEditEnded();
        },
        [commitStrokeBase, getLatestLoudnessFetchSeq, notifyLiveEditEnded],
    );

    // 从 store 中的 clipPitchCurves 转换为 DetectedPitchCurve[] 供 drawPianoRoll 使用。
    // 仅在 pitch 模式下且轨道 Compose 开启时显示，其他情况下传空数组以避免不必要的计算。
    const detectedPitchCurves = useMemo((): DetectedPitchCurve[] => {
        if (editParam !== "pitch") return [];
        if (!rootTrack?.composeEnabled) return [];
        return Object.entries(s.clipPitchCurves)
            .filter(([clipId]) => {
                // 只保留属于当前轨道组内的 clip，显示 root 及所有子轨道的 detected curve
                const clip = s.clips.find((cl) => cl.id === clipId);
                return clip && groupTrackIds.has(clip.trackId) && !clip.muted;
            })
            .map(([, c]) => ({
                curveStartSec: c.curveStartSec,
                midiCurve: c.midiCurve,
                framePeriodMs: c.framePeriodMs,
            }));
    }, [editParam, rootTrack, s.clipPitchCurves, s.clips, groupTrackIds]);

    const referencePitchOverlays = useMemo((): ReferencePitchOverlay[] => {
        if (editParam !== "pitch") return [];
        return visibleReferenceRootTrackIds
            .map((trackId) => {
                const paramViewForTrack = referencePitchViews[trackId];
                if (!paramViewForTrack) return null;
                const totalPoints = Math.max(
                    paramViewForTrack.orig.length,
                    paramViewForTrack.edit.length,
                );
                if (totalPoints < 2) return null;
                const track = s.tracks.find((item) => item.id === trackId);
                return {
                    rootTrackId: trackId,
                    strokeColor: buildReferencePitchStrokeColor(
                        track?.color ?? null,
                        hoveredReferenceRootTrackId === trackId,
                    ),
                    highlighted: hoveredReferenceRootTrackId === trackId,
                    paramView: paramViewForTrack,
                };
            })
            .filter((item): item is ReferencePitchOverlay => item != null);
    }, [
        editParam,
        hoveredReferenceRootTrackId,
        referencePitchViews,
        s.tracks,
        visibleReferenceRootTrackIds,
    ]);

    // 检测音高曲线更新时触发重绘
    useEffect(() => {
        invalidate();
    }, [detectedPitchCurves, invalidate]);

    useEffect(() => {
        invalidate();
    }, [invalidate, referencePitchOverlays]);

    // Ensure pitch-snap related changes immediately redraw
    useEffect(() => {
        invalidate();
    }, [
        s.pitchSnapEnabled,
        s.pitchSnapUnit,
        effectiveProjectScale,
        s.scaleHighlightMode,
        s.tempoMap,
        snapToggleHeld,
        invalidate,
    ]);

    // 剪贴板预览开关变化时立即重绘
    useEffect(() => {
        invalidate();
    }, [s.showClipboardPreview, invalidate]);

    // 【已删除：scaleSegments 帧间缓存】
    //
    // 它存在的唯一目的是给 `drawPianoRoll` 的 `scaleSegments` 入参供数（Tempo Map
    // 分段音阶高亮）。该入参随 Canvas2D 音高网格分支一起删除（分支因 `skipGrid`
    // 恒为 true 而不可达），缓存随之失去唯一读者。
    //
    // 【分段音阶高亮的去向】GL 网格几何是视口坐标、不含时间轴，无法表达"不同时间段
    // 用不同音阶"，因此**分段高亮未迁移**；现在只支持单一工程音阶。见
    // `buildGridSpec` 的说明。`s.tempoMap` 仍被自动吸附与标尺使用，未失去读者。

    /**
     * 组装曲线图层描述符（阶段 3：曲线 GL）。
     *
     * 流程：按**绘制顺序**产出 7 类图层 —— 参考线 → 检测曲线 → 副参数 → 原始曲线
     * → 编辑曲线 → 选区高亮 → 剪贴板预览（后者在上）。顺序与 `render.ts` 的
     * Canvas2D 调用顺序**逐项对应**，因为 GL 靠数组顺序决定层叠。
     *
     * 特殊说明 1：每条图层自带 `valueToY`。副参数的值域与主参数不同（音分 / 度数 /
     * 共振峰），必须用各自的投影，否则副参数曲线会被画到错误高度——而且错得"合理"
     * （仍在视口内），极难归因。
     *
     * 特殊说明 2：虚线图案一律经 `getFixedDashPattern`（按 dpr 量化），与 Canvas2D
     * 路径同一个函数，杜绝疏密分叉。
     *
     * 特殊说明 3：`values` 传引用而不是副本——采样值不随滚动变化，每帧复制的成本
     * 在长曲线上很可观。
     *
     * 特殊说明 4：两个需要裁剪的图层（选区高亮 / 剪贴板预览）把选区矩形同时作为
     * `clipRect`（GL 侧走 scissor）与坐标边界。
     *
     * @param axis 当前投影（每帧由 drawRef 构造）。
     * @returns 曲线图层列表（按绘制顺序）；GL 关闭时不会被调用。
     */
    function buildCurveLayers(axis: TimelineAxis): PianoRollCurveLayer[] {
        const layers: PianoRollCurveLayer[] = [];
        const isDark = themeMode === "dark";
        // 与 Canvas2D 路径同一份配色表（`render.ts` 也调 `resolvePianoRollColors`）。
        const colors = resolvePianoRollColors(isDark);
        const h = viewSizeRef.current.h;
        const project = (param: ParamName, value: number) => valueToY(param, value, h);

        // ── ① 参考线（pitch 模式）────────────────────────────────────
        if (editParam === "pitch") {
            for (const overlay of referencePitchOverlays) {
                const values = resolveSecondaryOverlayValues({
                    orig: overlay.paramView.orig,
                    edit: overlay.paramView.edit,
                });
                if (values.length < 2) continue;
                layers.push({
                    values,
                    param: "pitch",
                    startFrame: overlay.paramView.startFrame,
                    stride: overlay.paramView.stride,
                    framePeriodMs: overlay.paramView.framePeriodMs,
                    lineWidthPx: overlay.highlighted ? 3.2 : 2.6,
                    rgba: parseRgbaColor(normalizeCssColor(overlay.strokeColor)),
                    dash: null,
                    projection: "curve",
                    valueToY: (v) => project("pitch", v),
                });
            }
        }

        // ── ② 检测曲线（pitch 模式，按 clip 循环调色）─────────────────
        //
        // 投影模式必须是 `"detected"`（不是 `"curve"`）：
        // - 检测曲线自带**绝对起始秒** `curveStartSec`，塞进 `startFrame` 会把它丢掉，
        //   曲线整体平移到时间轴原点；
        // - 检测曲线的 `midi <= 0` 表示**无声帧**，必须跳过，否则相邻有声点之间会
        //   拉出一条贯穿底部的垂直尖刺（GL 迁移后实际出现过）。
        // 两条语义都与 `drawCurveTimed` 不同，见 `projectDetectedCurvePoints` 说明。
        if (editParam === "pitch") {
            const palette = resolveDetectedCurveColors(isDark);
            detectedPitchCurves.forEach((curve, ci) => {
                if (!curve.midiCurve || curve.midiCurve.length < 2) return;
                layers.push({
                    values: curve.midiCurve,
                    param: "pitch",
                    startFrame: 0,
                    stride: 1,
                    framePeriodMs: curve.framePeriodMs,
                    lineWidthPx: 2,
                    rgba: parseRgbaColor(normalizeCssColor(palette[ci % palette.length])),
                    dash: null,
                    projection: "detected",
                    curveStartSec: curve.curveStartSec,
                    valueToY: (v) => project("pitch", v),
                });
            });
        }

        // ── ③ 副参数曲线 ────────────────────────────────────────────
        if (pitchEnabled && visibleSecondaryParamIds.length > 0) {
            visibleSecondaryParamIds.forEach((paramId, index) => {
                const pv = secondaryParamViews[paramId];
                if (!pv || Math.max(pv.orig.length, pv.edit.length) < 2) return;
                const values = resolveSecondaryOverlayValues({ orig: pv.orig, edit: pv.edit });
                layers.push({
                    values,
                    param: paramId,
                    startFrame: pv.startFrame,
                    stride: pv.stride,
                    framePeriodMs: pv.framePeriodMs,
                    lineWidthPx: 2,
                    rgba: parseRgbaColor(
                        normalizeCssColor(resolveSecondaryCurveColor(isDark, paramId, index)),
                    ),
                    dash: null,
                    projection: "curve",
                    valueToY: (v) => valueToY(paramId as ParamName, v, h),
                });
            });
        }

        // ── ④⑤⑥ 主参数曲线（原始 / 编辑 / 选区高亮）─────────────────
        const pv = pitchEnabled ? paramView : null;
        if (pv) {
            const editValues =
                liveEditOverrideRef.current && liveEditOverrideRef.current.key === pv.key
                    ? liveEditOverrideRef.current.edit
                    : pv.edit;

            if (pv.orig.length >= 2) {
                layers.push({
                    values: pv.orig,
                    param: editParam,
                    startFrame: pv.startFrame,
                    stride: pv.stride,
                    framePeriodMs: pv.framePeriodMs,
                    lineWidthPx: 1.8,
                    rgba: parseRgbaColor(normalizeCssColor(colors.origCurve)),
                    dash: toDashTuple(getFixedDashPattern(6, 6)),
                    projection: "curve",
                    valueToY: (v) => project(editParam, v),
                });
            }

            if (editValues.length >= 2) {
                layers.push({
                    values: editValues,
                    param: editParam,
                    startFrame: pv.startFrame,
                    stride: pv.stride,
                    framePeriodMs: pv.framePeriodMs,
                    lineWidthPx: 2.6,
                    rgba: parseRgbaColor(normalizeCssColor(colors.editCurve)),
                    dash: null,
                    projection: "curve",
                    valueToY: (v) => project(editParam, v),
                });
            }

            // 选区高亮：裁剪到选区矩形（与 Canvas2D 的 ctx.clip 对应）。
            // 多段选区：每段推一层（同一曲线、各自的 clipRect）——曲线被画多次、
            // 每次只露出该段窗口，视觉等价于 Canvas2D 的「多矩形并集 clip」。
            const selection = selectionRef.current;
            if (selection && selection.length > 0 && editValues.length >= 2) {
                for (const range of selection) {
                    const clip = selectionClipRect(axis, range);
                    if (!clip || clip.w <= 0) continue;
                    layers.push({
                        values: editValues,
                        param: editParam,
                        startFrame: pv.startFrame,
                        stride: pv.stride,
                        framePeriodMs: pv.framePeriodMs,
                        lineWidthPx: 3.6,
                        rgba: parseRgbaColor(normalizeCssColor(colors.selectionCurve)),
                        dash: null,
                        projection: "curve",
                        clipRect: clip,
                        valueToY: (v) => project(editParam, v),
                    });
                }
            }

            // ── ⑦ 剪贴板预览（不同的投影语义：从落点起点按原始帧距排布）──
            // 与 Canvas2D 路径（render.ts）共用 paramClipboardMapping 的唯一映射
            // 规则：选区（帧）→ 帧区间 → 逐段落点 span，每段推一层。预览画的
            // 就是粘贴会落下的数据，断层两侧的截断与 Canvas2D 路径完全一致。
            // 时间换算用目标帧周期（粘贴按帧号落盘，用剪贴板帧周期会让预览与
            // 结果错位，见 clipboardPreviewSpans 说明）。
            const preview = clipboardRef.current;
            if (preview && selection && selection.length > 0 && preview.param === editParam) {
                const frameRanges = selectionToFrameRanges(selection);
                const spans = clipboardPreviewSpans({
                    targetRanges: frameRanges,
                    clipboard: preview,
                    targetFramePeriodMs: pv.framePeriodMs,
                });
                for (const span of spans) {
                    if (span.values.length === 0) continue;
                    const spanEndSec =
                        span.startSec + (span.values.length * span.framePeriodMs) / 1000;
                    const clip = secSpanClipRect(axis, span.startSec, spanEndSec);
                    if (!clip || clip.w <= 0) continue;
                    layers.push({
                        values: span.values,
                        param: editParam,
                        startFrame: 0,
                        stride: 1,
                        framePeriodMs: span.framePeriodMs,
                        lineWidthPx: 2,
                        rgba: parseRgbaColor(
                            normalizeCssColor(resolveClipboardPreviewColor(isDark)),
                        ),
                        dash: toDashTuple(getFixedDashPattern(4, 4)),
                        projection: "clipboard",
                        clipStartSec: span.startSec,
                        clipEndSec: spanEndSec,
                        clipRect: clip,
                        valueToY: (v) => project(editParam, v),
                    });
                }
            }
        }

        return layers;
    }

    /**
     * 把选区（拍单位）组装为 GL 场景层的选区块镜像。
     *
     * 【为什么在这里换算帧 → 秒】宿主只认统一投影（`TimelineAxis` 以秒为单位），
     * 而选区数据是帧。帧 → 秒只依赖工程级帧周期（与 BPM 无关）；放到宿主侧就是
     * 第三份口径。因此本函数只做"业务单位 → 宿主的秒"，不改任何几何。
     *
     * 【选区带 == 被圈住的采样点】选区边界是**切点**（两帧中间，见
     * `paramSelection.snapCut`）：第 k 帧的采样点画在 `framesToTime(k)`，它的领地
     * 是左右各半帧，因此带画在 `[startFrame - 0.5, startFrame + frameCount - 0.5]`。
     * 于是"框住第 2、3 帧"的两条边界正好落在第 1/2 帧之间与第 3/4 帧之间。
     *
     * 【为什么空选区返回 null 而不是空数组】两者对宿主是同一件事（清空实例），
     * 但 null 让"没有选区"与"有选区但都在视口外"在调试时仍然可区分。
     *
     * @param selection 选区（多区间，帧）；null / 空表示无选区。
     * @returns 选区块镜像；无选区时 null。
     */
    function buildSelectionBandSpec(
        selection: ParamSelection | null,
    ): PianoRollSelectionBandSpec | null {
        if (!selection || selection.length === 0) return null;
        const framePeriodMs = paramView?.framePeriodMs ?? 5;
        const colors = resolvePianoRollColors(themeMode === "dark");
        return {
            spansSec: selection.map((range) => ({
                startSec: framesToTime(frameRangeStartCut(range), framePeriodMs),
                endSec: framesToTime(frameRangeEndCut(range), framePeriodMs),
            })),
            fillRgba: parseRgbaColor(normalizeCssColor(colors.selectionBand)),
            borderRgba: parseRgbaColor(normalizeCssColor(colors.selectionBorder)),
        };
    }

    /**
     * 把秒区间换算为视口坐标的裁剪矩形。
     *
     * @param axis 当前投影。
     * @param startSec 区间起点（工程秒）。
     * @param endSec 区间终点（工程秒）。
     * @returns 裁剪矩形；宽度为 0 时返回 null。
     */
    function secSpanClipRect(
        axis: TimelineAxis,
        startSec: number,
        endSec: number,
    ): { x: number; y: number; w: number; h: number } | null {
        const x0 = secToViewportPx(axis, startSec);
        const x1 = secToViewportPx(axis, endSec);
        const w = x1 - x0;
        if (!(w > 0)) return null;
        return { x: x0, y: 0, w, h: viewSizeRef.current.h };
    }

    /**
     * 把单段选区（帧）换算为视口坐标的裁剪矩形。
     *
     * 与选区带同口径：边界取**切点**（两帧中间）。
     *
     * @param axis 当前投影。
     * @param range 单段选区（半开帧区间）。
     * @returns 裁剪矩形；宽度为 0 时返回 null。
     */
    function selectionClipRect(
        axis: TimelineAxis,
        range: FrameRange,
    ): { x: number; y: number; w: number; h: number } | null {
        const framePeriodMs = paramView?.framePeriodMs ?? 5;
        return secSpanClipRect(
            axis,
            framesToTime(frameRangeStartCut(range), framePeriodMs),
            framesToTime(frameRangeEndCut(range), framePeriodMs),
        );
    }

    // Keep draw function always up-to-date (invalidate() is stable and calls drawRef.current()).
    drawRef.current = () => {
        // 滚动热路径的投影：**必须以内核视口为准**（`resolvePanelRenderViewport`）。
        //
        // 【为什么不能用渲染期的 refs】`pxPerSecRef` / `scrollLeftRef` 在渲染期就被同步成
        // React state 的新值，而并发渲染允许"渲染但尚未提交"（被更高优先级更新打断并丢弃）
        // ——那一瞬间 ref 已是新值、内核与 DOM 还是旧值。用 refs 当投影源会让面板的
        // Canvas2D 与曲线**可见段选择**落在新视口、宿主 GL（网格 / 曲线 / 播放头）与 DOM
        // 落在旧视口：同一屏两套视口，表现为"这些线偏移了"。时间轴的 `livePxPerSec` 是同一
        // 约定的先例（逐帧发布一律取内核真值）。
        const kernelView = hostRef.current?.getViewport() ?? null;
        const viewport = resolvePanelRenderViewport({
            kernelView,
            refPxPerSec: pxPerSecRef.current,
            refScrollLeftPx: scrollLeftRef.current,
        });
        const drawAxis = createTimelineAxis({
            pxPerSec: viewport.pxPerSec,
            scrollLeftPx: viewport.scrollLeftPx,
            viewportWidthPx: viewSizeRef.current.w,
            dpr: window.devicePixelRatio || 1,
        });
        // 曲线图层（阶段 3）：每帧重建描述符列表。
        //
        // 【为什么要每帧构建】滚动/缩放会改变可见段与 `axis`，而 GL 侧要在绘制时
        // 才投影；描述符里的 `values` 引用与视口无关（数据没变），因此这里的成本
        // 只是一次浅层数组构建，不复制采样值。
        //
        // 【必须是唯一来源】曲线已归 GL（Canvas2D 侧 `skipCurves` 恒为 true），
        // 置 null 等于曲线无人绘制。
        kernelDataRef.current.curves = buildCurveLayers(drawAxis);

        // 叠加层镜像（播放头）必须**每帧**更新：播放头用的是插值的
        // 视觉值 `visualPlayheadSecRef`，它不由 React 渲染驱动（见下方注释），
        // 因此不能在 render 期写入镜像——那样播放头会停在旧的提交值上。
        // 选区块不在这里，而是走 `selectionBand` 字段（它是曲线**之下**的一层，
        // 由 GL 场景层在曲线之前绘制，见 `PianoRollSelectionBandSpec` 的层序说明）。
        //
        // 【颜色必须显式喂进去，不能靠宿主兜底】宿主在 `playheadRgba` 缺省时用
        // 一个硬编码兜底色（`[0,0,0,0.2]`），而参数编辑器的播放头在 Canvas2D 侧的
        // 绘制已被 `skipPlayhead: true` 永久跳过——也就是说 GL 这里是**唯一**绘制者，
        // 不喂颜色等于画布播放头与上方标尺（`--qt-playhead`）各是一套颜色。
        // 用户报告："播放线在底下和上方标尺的颜色不一致"。
        //
        // 取值经 `normalizeCssColor` 归一化：GL 不认 `var(...)`，必须借浏览器把
        // CSS 变量解析成 `rgb()/rgba()` 再交给 `parseRgbaColor`。归一化会触发样式
        // 重算，但本块是每帧执行的——因此按解析后的字符串做一次模块级缓存，
        // 主题切换时字符串本身会变，缓存自然失效（键就是字符串）。
        kernelDataRef.current.overlay = {
            playheadSec: visualPlayheadSecRef.current,
            playheadRgba: resolvePlayheadRgba(themeMode),
        };
        // 选区块镜像（每帧刷新）：选区是**拍**为单位，宿主只认秒，因此这里先换算。
        // 空选区时喂 null，宿主据此清空本帧的实例（否则旧带子会留在画布上）。
        kernelDataRef.current.selectionBand = buildSelectionBandSpec(selectionRef.current);
        /**
         * 主画布上绘制的中央提示文字（"音高被硬禁用"的原因）。
         *
         * 【为什么提取成局部变量】它同时是**签名项**与 `drawPianoRoll` 的入参。
         * 就地写两遍表达式（此前签名里根本没有它）正是本文件反复警告的
         * "签名里写了 A、实际喂给绘制的是 B"的漂移来源：一旦两处写法分叉，
         * 文字就会停在旧值或漏绘。提取一次、两处共用，漂移在结构上不可能发生。
         *
         * 【为什么非 pitch 参数用 childPitchHardDisableReason】子音高偏移参数
         * （cents / degrees / formant）走的是 `childPitchHardDisableReason`，
         * 与 `pitchEnabled` 的判定分支保持一一对应（见上方 `pitchEnabled`）。
         *
         * 【动态基线分析中】dyn 面板的虚线基线依赖后台电平分析；未就绪时基线是
         * 静默的 1.0 平线，用户无从知道"稍后会变"。就绪事件（dyn_orig_updated）
         * 会刷新快照并清掉本提示（useLoudnessCurves 的 analysisPending）。
         */
        const overlayText = !pitchEnabled
            ? editParam === "pitch"
                ? pitchHardDisableReason
                : childPitchHardDisableReason
            : isDynParam(editParam) && loudnessAnalysisPending
              ? t("dyn_analysis_pending")
              : null;

        /**
         * 主画布的内容签名（阶段 2 Task 6）。
         *
         * 【必须包含什么】主画布上绘制的**全部输入**：
         * - 绘图资源：各条曲线数据、参考线、检测曲线、副参数视口、morph 叠加、
         *   剪贴板预览、选区块、live 编辑覆盖、中央提示文字 `overlayText`；
         * - 视口：`viewSize`、`pxPerSec`、`scrollLeft`、`dpr`（滚动/缩放会改变投影）；
         * - 主题与字体（颜色解析与文字宽度都会影响像素结果）。
         *
         * 【音阶高亮为什么不在签名里】它已迁到 **GL 网格层**（主画布不再绘制它），
         * 其输入（音级集合 / 强调线颜色）由 `PianoRollGridSpec` 的几何签名负责。
         * 此前这里编入的 `effectiveProjectScale` / `s.tempoMap` / `s.pitchSnapUnit` /
         * `s.scaleHighlightMode` / `s.toolMode` / `snapToggleHeld` 都随该分支的删除
         * 一并移除——它们对应的绘制入参已不存在，"编进来"只会让缓存无谓失效。
         *
         * 【必须**不**包含什么】播放头位置——它已由 GL 叠加层绘制。把它编进签名会让
         * 播放帧的签名每帧变化、缓存失效，那就退回"每帧重绘曲线"。
         *
         * 【比较语义】这是一个**数组**，交给 `isSameMainCanvasSignature` 逐项
         * `Object.is` 比较：**对象 / 数组项按引用参与**（Redux 只在内容变化时换引用，
         * 比引用既快又准），原始值按数值。**绝不拼接字符串**——此前后缀是
         * `.join("|")`，而 `join` 会把每个对象 / 数组元素串成字面量
         * `"[object Object]"`，两个内容完全不同的选区因此得到同一个签名，缓存命中、
         * 主画布在清屏前就 return，旧选区框留在画布上不消失（缺陷 #4，详见
         * `pianoRoll/mainCanvasSignature.ts`）。引用比较之所以能逐帧失效，是因为
         * 选区在变化时被赋**新对象**；绘制中的 live 覆盖改成了原地更新，故它
         * 不参与引用比较，而是以**显式版本号**参与（见下方签名项）。
         *
         * 【必须与绘制入参一一对应】下面每一项都刻意对应 `drawPianoRoll` 的某个
         * 入参（或影响其投影的视口量），避免"签名里写了 A、实际喂给绘制的是 B"。
         * 漏项的代价是"该图层不再更新"，因此这里**宁可多编**：低频变化的项一并纳入，
         * 成本只是偶尔多一次重绘。最后再列一遍 `secondaryParamViews` 并非冗余——
         * 它与 `paramViewsRef.current` 同为曲线数据源，两者都必须进签名。
         */
        const mainContentSignature: MainCanvasSignature = [
            viewSize.w,
            viewSize.h,
            pxPerSecRef.current,
            scrollLeftRef.current,
            Math.round((window.devicePixelRatio || 1) * 100),
            editParam,
            themeMode,
            fontFamily,
            pitchEnabled ? 1 : 0,
            // 帧 → sec 的换算系数：选区框与剪贴板预览的 x 由它投影。它与 BPM
            // **无关**（工程级常量栅格），因此改 BPM 不再需要重绘主画布；签名里
            // 保留它是为了"帧周期一旦变化，选区几何必须跟着失效"这条契约。
            paramView?.framePeriodMs ?? 5,
            // 数据与几何（按引用比较）。刻意与传给 drawPianoRoll 的字段一一对应，
            // 避免"签名里写了 A、实际喂给绘制的是 B"这种漂移。
            detectedPitchCurves,
            referencePitchOverlays,
            secondaryParamViews,
            visibleSecondaryParamIds,
            paramMorphOverlays,
            s.showClipboardPreview ? clipboardRef.current : null,
            selectionRef.current,
            // 绘制中的 live 覆盖：用**显式版本号**而非对象引用参与签名。
            // 覆盖层改为原地更新后引用不再变化，比引用即可正确失效（且"无覆盖"
            // → 0 与"新覆盖"→ 非 0 天然可分）。见 mainCanvasSignature 的约束说明。
            liveEditOverrideRef.current?.version ?? 0,
            // 中央提示文字（"音高被硬禁用"的原因）：本面板**新增**的字符串签名项，
            // 切换参数 / 轨道组时会变（禁用原因出现或消失）。
            // 注意 `drawPianoRoll` 另有多项字符串入参（editParam / fontFamily 等），
            // 它们在上方各自的位置参与签名——本项并非"唯一"字符串项。
            overlayText,
            // 视口中心/跨度（用 ref 值，避免依赖 React 渲染时机）
            pitchViewRef.current.center,
            pitchViewRef.current.span,
            paramViewsRef.current,
            secondaryParamViews,
        ];

        drawPianoRoll({
            axisCanvas: axisCanvasRef.current,
            canvas: canvasRef.current,
            viewSize: viewSizeRef.current,
            editParam,
            pitchView: pitchViewRef.current,
            paramViews: paramViewsRef.current,
            valueToY,
            paramView: pitchEnabled ? paramView : null,
            secondaryParamViews: pitchEnabled ? secondaryParamViews : {},
            secondaryParamIds: pitchEnabled ? visibleSecondaryParamIds : [],
            showSecondaryParam: pitchEnabled && visibleSecondaryParamIds.length > 0,
            // 与上面签名里的 `overlayText` 是**同一个变量**（不重复写表达式）。
            overlayText,
            liveEditOverride: liveEditOverrideRef.current,
            selection: selectionRef.current,
            axis: drawAxis,
            framePeriodMs: paramView?.framePeriodMs ?? 5,
            // 画布每帧重绘（onFrame invalidate），播放头必须用插值的视觉值：
            // 用 Redux 提交值会让 60fps 的重绘画着同一个旧播放头（且与标尺
            // 的 DOM 插值播放头节奏不一致、短暂错位）。
            playheadSec: visualPlayheadSecRef.current,
            referencePitchOverlays,
            detectedPitchCurves,
            isDark: themeMode === "dark",
            // ── 下列五个 `skip*` 恒为 true：这些图层**只有 GL 一个绘制者** ──────
            //
            // 内核是唯一渲染路径，键盘几何 / 轴文字 / 轴画布 / 播放头 / 曲线
            // 全归 GL。恒 true 意味着 Canvas2D 侧**永久跳过**这些图层。
            //
            // 【已知并接受的限制】WebGL2 在**运行期**失败时，这些图层无人绘制（实测：
            // 贯穿宽度的横向网格线 24 → 0 条，音高网格整片消失）。参数编辑器在无
            // WebGL2 的环境下本就无法工作，补一套 Canvas2D 回退等于把已迁走的渲染层
            // 再实现一遍，收益与成本完全不成比例——用户已明确决定不修。详见设计文档
            // `docs/superpowers/specs/2026-09-13-timeline-single-path-design.md` §2.4。
            //
            // 特殊说明 1：跳过的都是 **GL 已完全接管**的图层；选区块、morph 手柄等仍在
            // 下面由主画布绘制（见 render.ts 对 skipPlayhead / skipCurves 的说明）。
            //
            // 特殊说明 2：**没有 `skipGrid`**——网格的 Canvas2D 分支已被整体删除
            // （它原本就因本处恒传 `true` 而不可达）。音阶高亮随网格一起迁到 GL。
            skipKeyboardGeometry: true,
            skipAxisText: true,
            // 轴画布全部内容归 GL -> 整张跳过（含清屏）。
            skipAxisCanvas: true,
            skipPlayhead: true,
            // 曲线归 GL（阶段 3）。morph 手柄不在曲线层内，仍由主画布绘制。
            skipCurves: true,
            // 主画布内容缓存（Task 6）：签名只含**主画布自己绘制的内容**与视口，
            // 不含播放头（它已归 GL 叠加层）——这正是播放帧能跳过曲线重绘的原因。
            mainContentSignature,
            fontFamily,
            clipboardPreview: s.showClipboardPreview ? clipboardRef.current : null,
            // 形变控制线：每段选区一条（选区已统一为多区间 `ParamSelection`）。
            // 【合并说明】feature/tools 分支此处另行传过 `pitchSnapUnit` /
            // `projectScale` / `scaleHighlightMode` / `scaleSegments` / `toolMode` /
            // `snapToggleHeld`——这些入参在本分支已随「Canvas2D 音高网格分支整体
            // 删除」一并移除（网格与音阶高亮恒由 GL 绘制，见 `render.ts` 的说明），
            // 继续传只会被 `drawPianoRoll` 静默忽略。
            paramMorphOverlays,
        });
    };

    const handleEditActionRef = useRef<(op: string) => void>(() => {});
    // Stable callback that delegates to the latest handleEditOp via ref
    const stableEditAction = useCallback((op: string) => {
        handleEditActionRef.current(op);
    }, []);

    const interactions = usePianoRollInteractions({
        dispatch,
        rootTrackId,
        editParam,
        pitchEnabled,
        toolMode: s.toolMode,
        framePeriodMs: paramView?.framePeriodMs ?? 5,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerSecRef,
        // 交互侧坐标换算的视口真值：与渲染侧（`resolvePanelRenderViewport`）同源，
        // 避免框选 / 命中测试读到量化提交滞后的 `scrollLeftRef`。见该字段的说明。
        getViewportTruth: useCallback(() => hostRef.current?.getViewport() ?? null, []),
        horizontalZoomChainRef,
        onHorizontalZoom: handleHorizontalZoom,
        syncTimelineEnabled: s.paramEditorSyncTimeline,
        timelineOffsetRef,
        setPitchView,
        setParamViewport,
        pitchViewRef,
        paramViewsRef,
        scrollerRef,
        canvasRef,
        viewSizeRef,
        selectionRef,
        selectionUi,
        setSelectionUi,
        setCanvasCursor,
        strokeRef,
        panRef,
        paramView,
        paramViewRef,
        bumpRefreshToken,
        syncScrollLeft,
        invalidate,
        yToViewportT,
        yToValue,
        valueToY,
        clampViewport,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        onParamCommitSucceeded,
        requestWaveformRepaint,
        commitStroke,
        setParamView,
        liveEditOverrideRef,
        liveEditActiveRef,
        prVerticalZoomKb,
        horizontalZoomKb,
        scrollHorizontalKb,
        scrollVerticalKb,
        scrollbarZoomKb,
        paramMorphKb,
        paramMultiSelectKb,
        paramStretchKb: stretchKb,
        vibratoAmplitudeAdjustKb,
        vibratoFrequencyAdjustKb,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
        cycleDragDirectionKb,
        paramFineAdjustKb,
        onContextMenu: useCallback((x: number, y: number) => {
            setCtxMenu({ x, y });
        }, []),
        getPlayheadSec: getVisualPlayheadSec,
        playheadZoomEnabled: s.playheadZoomEnabled,
        paramEditorSeekPlayheadEnabled: s.paramEditorSeekPlayheadEnabled,
        pitchSnapEnabled: s.pitchSnapEnabled,
        pitchSnapUnit: s.pitchSnapUnit,
        projectScale: effectiveProjectScale,
        /** Tempo Map 感知：按帧时刻解析生效音阶。 */
        scaleAtSec: projectScaleAtSec,
        pitchSnapToleranceCents: s.pitchSnapToleranceCents,
        keybindingMap: mergedKeybindings,
        onEditAction: stableEditAction,
        dragDirection: activeDragDirection,
        onCycleDragDirection: useCallback(
            (tool: "select" | "draw" | "vibrato") => {
                dispatch(cycleDragDirection(tool));
                void dispatch(persistUiSettings());
            },
            [dispatch],
        ),
        edgeSmoothnessPercent: s.edgeSmoothnessPercent,
        onMorphOverlayChange: setParamMorphOverlays,
        currentParamRange,
        onPitchSnapGestureActiveChange: useCallback((active: boolean) => {
            setSnapGestureActive(active);
        }, []),
        paramValuePopupEnabled: s.showParamValuePopup,
        onParamValuePreviewChange: useCallback(
            (
                next: {
                    clientX: number;
                    clientY: number;
                    value: number;
                    displayText?: string;
                } | null,
            ) => {
                setParamValuePreview(next);
            },
            [],
        ),
    });

    // 参数数据更新后重算当前悬停浮窗：键盘平移参数线（"=" / "-" / "]" / "["
    // 及其 Shift/Ctrl 变体）、撤销/重做、远端写入都只更新 paramView 数据，
    // 不会触发 pointermove —— 悬停值是 pointermove 时的快照，若不在此重算，
    // 浮窗会一直显示旧值直到用户再次移动鼠标。声明顺序在 paramViewRef 同步
    // effect（上方）之后，重算读到的是本次渲染的最新数据。
    // refreshParamValuePreview 先解构出稳定引用：直接依赖 interactions 对象
    // 会让 effect 每次渲染都触发（setState 新对象 → 无限重渲染）。
    const { refreshParamValuePreview } = interactions;
    useEffect(() => {
        refreshParamValuePreview();
    }, [paramView, refreshParamValuePreview]);

    const onScrollerWheelNative = interactions.onScrollerWheelNative;
    /**
     * 原生滚动容器的 `scroll` 事件（面板侧的唯一入口）。
     *
     * 【绝大部分是镜像回声，但不是全部】
     * 内核是唯一渲染路径，原生 scroller 只是**镜像**：`onFrame` 每帧把内核真值写回
     * 它，该写入会触发原生 `scroll` 事件。旧代码在这里**无条件**读回位置再写回内核，
     * 而那一刻读到的是上一帧镜像的旧值（实测内核已到 548.054、事件里读到 540.5），
     * 使内核被回退 7.554px，逐帧往复即用户报告的「上下拖经常拖不动 / 阶梯感」。
     *
     * 但"无条件忽略"同样不对：原生容器仍是 `overflow: scroll`，**触摸拖拽 /
     * 触控板惯性 / 焦点滚入视口**这三类输入只会产生原生 `scroll`，没有任何显式
     * 入口——忽略它等于这些输入完全失效（旧实现在这里读回位置正是为了它们）。
     *
     * 【判据：与「上次镜像写入值」比，**不是**与「内核当前值」比】
     * 本处理函数一度拿事件值与 `host.getViewport()`（内核**当前**值）比较。那在连续
     * 滚动 / 缩放时必然失效：事件报的是**上一帧**写下的值，内核此刻已前进一整帧，
     * 两者必然不等 → 回声被稳定地误收成"容器自己动了" → 采纳并（同步开启时）把它
     * 推回共享视口。推回的还带着浏览器的设备像素量化误差（≤0.5px），于是两个面板
     * 进入亚像素往复——即用户报告的「启用同步后滚轮缩放仍然抽动」（同步关闭时不会
     * 推回共享视口，所以看不到）。
     *
     * 基准必须是**我上次写进容器的值**（`lastMirroredScroll*Ref`）：回声报的正是它
     * （误差仅来自浏览器量化，有界），而真实输入会把容器带到别的值上。这正是
     * `timeline/scrollEcho` 为轨道头写下的同一条结论（判来源，不判与真值的距离），
     * 现在两个容器共用同一个纯函数。
     *
     * @param event 原生 `scroll` 事件。
     * @returns 无返回值。
     */
    const onScrollerScroll = useCallback(
        (event: React.UIEvent<HTMLDivElement>) => {
            // 应用共享视口写入期间不采纳：那一刻原生镜像还停在上一帧的旧值
            //（与 `timelineSyncApplyingRef` 的既有用法同一理由）。
            if (timelineSyncApplyingRef.current) return;
            const scroller = event.currentTarget;
            const host = hostRef.current;
            if (host === null) return;
            if (
                !isMirrorEcho({
                    mirroredPx: lastMirroredScrollLeftRef.current,
                    nativePx: scroller.scrollLeft,
                })
            ) {
                syncScrollLeft(scroller);
            }
            if (
                !isMirrorEcho({
                    mirroredPx: lastMirroredScrollTopRef.current,
                    nativePx: scroller.scrollTop,
                })
            ) {
                host.setScrollTop(scroller.scrollTop);
                // 原生滚动事件驱动的竖向滚动（触摸 / 触控板 / 焦点滚入）：
                // 与横向同因——内核写完后必须**同任务**提交各图层，否则竖向内容
                // （音高键 / 数值轴 / 曲线 / 播放头）比原生滚动慢一帧（见 `paintNow`）。
                host.paintNow();
            }
        },
        [syncScrollLeft],
    );
    const scrollerWheelHandlerRef = useRef(onScrollerWheelNative);

    useLayoutEffect(() => {
        scrollerWheelHandlerRef.current = onScrollerWheelNative;
    });

    useEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;

        const handler: EventListener = (evt) => {
            scrollerWheelHandlerRef.current(evt as globalThis.WheelEvent);
        };

        el.addEventListener("wheel", handler, {
            passive: false,
        } as globalThis.AddEventListenerOptions);
        return () => {
            el.removeEventListener("wheel", handler);
        };
    }, []); // 空依赖

    // 参数切换或参数描述符变化后，刷新竖向滚动条位置，保证滚动条与当前视口保持一致。
    useLayoutEffect(() => {
        syncVerticalScrollbarForViewport(editParam, getCurrentViewportForScrollbar(editParam));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [editParam, processorParams]);

    // Auto-scroll: keep playhead visible in parameter editor during playback
    useEffect(() => {
        if (s.paramEditorSyncTimeline) return;
        if (!s.autoScrollEnabled || !s.runtime.isPlaying) return;
        const scroller = scrollerRef.current;
        if (!scroller) return;
        const next = computeAutoFollowScrollLeft({
            playheadSec: visualPlayheadSecRef.current,
            pxPerSec,
            viewportWidth: scroller.clientWidth,
            contentWidth,
        });
        if (Math.abs(scroller.scrollLeft - next) > 0.5) {
            scroller.scrollLeft = next;
            syncScrollLeft(scroller);
        }
        // syncScrollLeft reads the latest scroll state through refs; see onFrame above.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        s.paramEditorSyncTimeline,
        s.autoScrollEnabled,
        s.runtime.isPlaying,
        s.playheadSec,
        pxPerSec,
        contentWidth,
    ]);

    // Piano keys (axis) area: keep touchpad wheel behavior aligned with the main editor.
    useEffect(() => {
        const el = axisWrapRef.current;
        if (!el) return;

        const handler = (e: WheelEvent) => {
            // During vibrato/line drag, wheel always adjusts vibrato parameters.
            // Defer to the scroller's wheel handler which has full vibrato drag logic.
            if (document.body.hasAttribute("data-piano-roll-vibrato-drag-active")) {
                e.preventDefault();
                return;
            }

            const noModifierPressed = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
            const isWheelBindingRequested = (kb: Keybinding) => {
                if (isNoneBinding(kb)) return noModifierPressed;
                return isModifierActive(kb, e);
            };
            const horizontalScrollRequested = isWheelBindingRequested(scrollHorizontalKb);
            const pianoVerticalScrollRequested = isWheelBindingRequested(pianoKeysVerticalScrollKb);
            const pianoVerticalZoomRequested = isWheelBindingRequested(pianoKeysVerticalZoomKb);
            const horizontalZoomRequested = isWheelBindingRequested(horizontalZoomKb);

            const bounds = el.getBoundingClientRect();
            // 竖直分母必须用**绘图区**高度（`viewSize.h`），不是纵轴列的高度：
            // 列比滚动视口高出一条自绘水平滚动条的行，且轴位图/琴键是按
            // `viewSize.h` 投影的（见内核的 AXIS_TICK_LABEL_DESCENT_PX 说明）。
            // 用列高会让滚轮/悬停算出的值比渲染出来的位置偏低。
            const h = Math.max(1, viewSizeRef.current.h);
            const pointerY = clamp(e.clientY - bounds.top, 0, h);
            // t: 0=top, 1=bottom — same semantics as usePianoRollInteractions
            const t = pointerY / h;

            const wheelAction = getParamEditorWheelAction({
                deltaX: e.deltaX,
                deltaY: e.deltaY,
                horizontalScrollRequested,
                verticalPanRequested: pianoVerticalScrollRequested,
                verticalZoomRequested: pianoVerticalZoomRequested,
                horizontalZoomRequested,
            });

            const applyVerticalPanDelta = (deltaY: number) => {
                const delta = (-deltaY / h) * 0.5;
                if (editParam === "pitch") {
                    const cur = pitchViewRef.current;
                    const next = clampViewport("pitch", {
                        span: cur.span,
                        center: cur.center + delta * cur.span,
                    });
                    setPitchView(next);
                } else {
                    const cur = paramViewsRef.current[editParam] ?? {
                        center: 0.5,
                        span: 1,
                    };
                    const next = clampViewport(editParam, {
                        span: cur.span,
                        center: cur.center + delta * cur.span,
                    });
                    setParamViewport(editParam, next);
                }
                invalidate();
            };

            const horizontalDelta = Math.abs(e.deltaX) > 0.5 ? e.deltaX : e.deltaY;

            if (wheelAction === "free-scroll") {
                e.preventDefault();
                const scroller = scrollerRef.current;
                if (scroller) {
                    scroller.scrollLeft += e.deltaX;
                    syncScrollLeftRef.current(scroller);
                }
                applyVerticalPanDelta(e.deltaY);
                return;
            }

            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                const scroller = scrollerRef.current;
                if (!scroller) return;
                scroller.scrollLeft += horizontalDelta;
                syncScrollLeftRef.current(scroller);
                return;
            }

            if (wheelAction === "vertical-pan") {
                e.preventDefault();
                applyVerticalPanDelta(e.deltaY);
                return;
            }

            if (wheelAction === "horizontal-zoom") {
                // 主画布/轨道视图负责水平缩放；钢琴键区只拦截，避免原生滚动抢走事件。
                e.preventDefault();
                return;
            }

            if (wheelAction !== "vertical-zoom") {
                return;
            }

            e.preventDefault();

            const valueAtPointer =
                editParam === "pitch"
                    ? (() => {
                          const view = pitchViewRef.current;
                          const absMin = PITCH_MIN_MIDI;
                          const absMax = PITCH_MAX_MIDI;
                          const span = clamp(view.span, 1e-6, absMax - absMin);
                          const min = clamp(view.center - span / 2, absMin, absMax - span);
                          return clamp(min + (1 - t) * span, absMin, absMax);
                      })()
                    : (() => {
                          const desc = processorParamsRef.current?.find(
                              (d: ProcessorParamDescriptor) => d.id === editParam,
                          );
                          const absMin =
                              desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
                          const absMax =
                              desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
                          const view = paramViewsRef.current[editParam] ?? {
                              center: (absMin + absMax) / 2,
                              span: absMax - absMin || 1,
                          };
                          const span = clamp(view.span, 1e-6, absMax - absMin || 1);
                          const min = clamp(view.center - span / 2, absMin, absMax - span);
                          return clamp(min + (1 - t) * span, absMin, absMax);
                      })();

            const factor = e.deltaY < 0 ? 0.9 : 1.1;

            if (editParam === "pitch") {
                const cur = pitchViewRef.current;
                const nextSpan = cur.span * factor;
                const next = clampViewport("pitch", {
                    span: nextSpan,
                    center: valueAtPointer - (0.5 - t) * nextSpan,
                });
                setPitchView(next);
            } else {
                const cur = paramViewsRef.current[editParam] ?? {
                    center: 0.5,
                    span: 1,
                };
                const nextSpan = cur.span * factor;
                const next = clampViewport(editParam, {
                    span: nextSpan,
                    center: valueAtPointer - (0.5 - t) * nextSpan,
                });
                setParamViewport(editParam, next);
            }
            invalidate();
        };

        el.addEventListener("wheel", handler, {
            passive: false,
        } as globalThis.AddEventListenerOptions);
        return () => {
            el.removeEventListener("wheel", handler);
        };
    }, [
        editParam,
        setPitchView,
        setParamViewport,
        invalidate,
        scrollHorizontalKb,
        pianoKeysVerticalScrollKb,
        pianoKeysVerticalZoomKb,
        horizontalZoomKb,
    ]);

    // Piano keys (axis) hover: play sine wave sound when pointer moves over keys
    useEffect(() => {
        // 【只有音高参数的左轴才是钢琴卷帘】其余参数的左轴是数值刻度（音量 / 动态 /
        // 音分…），没有"对应的音高"可发声——对着刻度按下就响是纯粹的噪声，且音量 /
        // 动态的刻度点击另有语义（切换展示单位，见下一个 effect）。非音高时直接在
        // 注册监听之前返回，连手势状态都不建立。
        if (editParam !== "pitch") return;

        const el = axisWrapRef.current;
        if (!el) return;

        let isPointerDown = false;
        let activeMidiNote: number | null = null;

        const getMidiNoteFromY = (clientY: number): number => {
            const bounds = el.getBoundingClientRect();
            const y = clientY - bounds.top;
            // 同滚轮路径：分母取绘图区高度，与琴键实例的投影同源
            //（见内核的 AXIS_TICK_LABEL_DESCENT_PX 说明）。
            const h = Math.max(1, viewSizeRef.current.h);
            const t = 1 - clamp(y / h, 0, 1);
            const absMin = PITCH_MIN_MIDI;
            const absMax = PITCH_MAX_MIDI;
            const view = pitchViewRef.current;
            const span = clamp(view.span, 1e-6, absMax - absMin);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            // 使用 floor 与渲染逻辑一致
            return Math.floor(clamp(min + t * span, absMin, absMax));
        };

        const playNoteIfChanged = (midiNote: number) => {
            if (midiNote !== activeMidiNote) {
                if (activeMidiNote !== null) {
                    pianoKeySound.stop(activeMidiNote);
                }
                activeMidiNote = midiNote;
                pianoKeySound.play(midiNote, 0.25);
            }
        };

        const stopNote = () => {
            if (activeMidiNote !== null) {
                pianoKeySound.stop(activeMidiNote);
                activeMidiNote = null;
            }
        };

        const onPointerDown = (e: PointerEvent) => {
            if (e.button !== 0) return;
            isPointerDown = true;
            const midiNote = getMidiNoteFromY(e.clientY);
            playNoteIfChanged(midiNote);
        };

        const onPointerMove = (e: PointerEvent) => {
            if (!isPointerDown) return;
            const midiNote = getMidiNoteFromY(e.clientY);
            playNoteIfChanged(midiNote);
        };

        const onPointerUp = () => {
            isPointerDown = false;
            stopNote();
        };

        const onPointerLeave = () => {
            if (isPointerDown) {
                stopNote();
            }
        };

        el.addEventListener("pointerdown", onPointerDown);
        el.addEventListener("pointermove", onPointerMove);
        window.addEventListener("pointerup", onPointerUp);
        el.addEventListener("pointerleave", onPointerLeave);

        return () => {
            el.removeEventListener("pointerdown", onPointerDown);
            el.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            el.removeEventListener("pointerleave", onPointerLeave);
            stopNote();
        };
    }, [editParam, pitchViewRef]);

    // ── 纵轴标尺：左键点击切换展示单位（音量 / 动态）──────────────────────────
    //
    // 「倍率 ↔ dB」只对**线性幅值倍率**参数有意义（1× = 0 dB，见 paramAxisUnits）。
    // 其余参数的左轴单位由参数语义唯一确定（音高是音名、音分就是音分），因此支持
    // 判定不通过时**完全不注册监听**——点击左轴保持"无操作"，而不是静默写一个
    // 无意义的设置项。
    //
    // 用 `click` 而不是 pointerdown/up 自配对：`click` 只在按下与抬起都落在本元素
    // （或本元素内的子元素）上时触发，天然排除"从轴列拖到画布"这类手势；轴列上
    // 目前没有其它拖拽手势，因此不需要更复杂的裁决。
    useEffect(() => {
        const el = axisWrapRef.current;
        if (!el) return;
        if (!supportsParamAxisUnit(editParam)) return;
        const onClick = (e: MouseEvent) => {
            // 只认左键：中键留给内核的平移手势，右键留给上下文菜单。
            if (e.button !== 0) return;
            dispatch(toggleParamAxisUnit(editParam));
            void dispatch(persistUiSettings());
        };
        el.addEventListener("click", onClick);
        return () => el.removeEventListener("click", onClick);
    }, [dispatch, editParam]);

    // ── 纵轴标尺的悬浮读数（`弹出展示参数`）──────────────────────────────────
    //
    // 与曲线上的浮窗共用同一个开关（`showParamValuePopup`）。读数口径按左轴的**内容**
    // 分两种：
    // - 钢琴卷帘（音高）：**只给音名**（E4 / D4），不给音分 —— 左轴本来就是按琴键
    //   分行画出来的，"E4+12" 并不指向某个键，反而会被误读成另一个音；
    // - 数值刻度（音量 / 动态 / 音分 / 张力…）：给该 y 处的参数值，其中音量 / 动态
    //   按纵轴展示单位读数（切 dB 时就是 dB）。
    useEffect(() => {
        const el = axisWrapRef.current;
        if (!el) return;
        if (!s.showParamValuePopup) {
            setAxisValuePreview(null);
            return;
        }

        const describe = (clientY: number): string => {
            const bounds = el.getBoundingClientRect();
            // 分母必须是**绘图区**高度（`viewSize.h`），不是轴列高度：轴列比滚动
            // 视口高出标尺行与底部滚动条行，用列高会让读数整体偏低（与轴上滚轮
            // 处理同一条约束，见上方 wheel effect 的说明）。
            const h = Math.max(1, viewSizeRef.current.h);
            const y = clamp(clientY - bounds.top, 0, h);
            if (editParam === "pitch") {
                // 与琴键实例同一投影：向下取整到"这一行属于哪个键"。
                return midiToLabel(Math.floor(yToValue("pitch", y, h)));
            }
            return formatParamValuePreview(yToValue(editParam, y, h));
        };

        const onPointerMove = (e: PointerEvent) => {
            const text = describe(e.clientY);
            if (text.length === 0) {
                setAxisValuePreview(null);
                return;
            }
            setAxisValuePreview({ clientX: e.clientX, clientY: e.clientY, text });
        };
        const onPointerLeave = () => setAxisValuePreview(null);

        el.addEventListener("pointermove", onPointerMove);
        el.addEventListener("pointerleave", onPointerLeave);
        return () => {
            el.removeEventListener("pointermove", onPointerMove);
            el.removeEventListener("pointerleave", onPointerLeave);
            setAxisValuePreview(null);
        };
    }, [s.showParamValuePopup, editParam, yToValue, formatParamValuePreview]);

    // 选区在所有工具模式下保留并持续渲染（选区带 / 剪贴板预览在 render.ts
    // 不按工具门控）：绘制 / 直线颤音工具下不再自动清空，用户可以换轨后
    // 直接对保留的选区做复制/剪切（路由见 focusRouting.resolveCopyCutRoute，
    // 依据 session.paramSelectionActive / selectionContext）。与选区的交互
    // （拖拽 / 右键菜单 / 变形）仍只在选择工具下生效（各交互路径已有
    // toolMode 门控）。

    useEffect(() => {
        setCanvasCursor(s.toolMode === "select" ? "default" : "crosshair");
    }, [s.toolMode]);

    useEffect(() => {
        setCtxMenu(null);
    }, [s.toolMode]);

    // 同步数据加载状态到全局 Context
    useEffect(() => {
        updatePianoRollStatus({
            dataLoading: isLoading,
        });
    }, [isLoading, updatePianoRollStatus]);

    /**
     * 参数编辑器「全选」：把整条参数曲线（0 → 工程时长）设为选区。
     *
     * 独立成函数有两个用途：`selectAll` 菜单命令本身，以及 `handleEditOp` 开头的
     * **隐式全选**（无选区时先把作用域铺满整条曲线，再执行原本需要选区的操作）。
     *
     * @returns 是否真的设置了选区；工具模式不是「选择」时为 `false`（与菜单命令
     *   同一守卫：绘制 / 直线 / 颤音工具下不产生选区），调用方据此保持原行为。
     */
    const selectAllParamRange = useCallback((): boolean => {
        if (s.toolMode !== "select") return false;
        // 整条曲线 = 帧 [0, 工程末端)。**数据路径**直接按整数帧构造：全选的边界就是
        // 首帧与末帧，不该被"最近中点"再挪半帧（见 paramSelection 的两个构造器）。
        const totalFrames = Math.max(
            0,
            timeToFrame(dynamicProjectSec, paramView?.framePeriodMs ?? 5),
        );
        selectionRef.current = selectionFromFrames(0, totalFrames);
        setSelectionUi(selectionRef.current);
        invalidate();
        return true;
    }, [s.toolMode, dynamicProjectSec, paramView?.framePeriodMs, invalidate]);

    // ── Edit operation handler (shared by context menu + MenuBar events) ──
    const handleEditOp = useCallback(
        async (op: string, data?: Record<string, unknown>) => {
            if (!rootTrackId) return;
            const fp = paramView?.framePeriodMs ?? 5;

            // ── 无选区时的隐式全选 ────────────────────────────────────────
            // 本编辑器里的操作几乎都以**参数选区**为作用域（见
            // SELECTION_SCOPE_EXEMPT_OPS 的说明）。没有选区时它们原本静默什么都
            // 不做 —— 用户得先"全选"再点一次，而右键菜单里点下去毫无反应更容易被
            // 当成功能坏了。这里统一改为：**先全选，再执行**，作用域 = 整条参数曲线
            // （与用户手动全选完全等价：同样的 `selectionFromFrames(0, 工程末端)`）。
            //
            // 非「选择」工具下全选不生效（与菜单命令同一守卫），此时行为与改动前
            // 逐字一致：这些操作在绘制类工具里本就没有意义。
            const hadSelectionAtEntry = selectionRef.current;
            if (
                !SELECTION_SCOPE_EXEMPT_OPS.has(op) &&
                (!hadSelectionAtEntry || hadSelectionAtEntry.length === 0)
            ) {
                selectAllParamRange();
            }

            if (op === "selectAll") {
                selectAllParamRange();
                return;
            }
            if (op === "deselect") {
                if (s.toolMode !== "select") return;
                selectionRef.current = null;
                setSelectionUi(null);
                invalidate();
                return;
            }

            // 双击 Clip（无拖拽，ClipItem 派发）：按 Clip 起止范围在参数编辑器
            // 内创建选区，并把交互焦点切到参数编辑器侧 ——
            // 复制/剪切路由（resolveCopyCutRoute 依据 selectionContext，经由下方
            // selectionUi 同步派发 setParamSelectionActive 标记）与活动表面
            // （focusSurface，外来源粘贴兜底等）随之指向参数编辑器。
            //
            // mode（来自时间轴的双击手势）：
            //   - "replace"（缺省）：替换为该块范围，与旧行为逐字一致；
            //   - "add"：把该块范围并入（重叠/相接自动合并）；
            //   - "toggle"：该块范围已被完整覆盖则挖掉，否则并入 —— 同一个块
            //     连按两次回到原状（`modifier.clipRangeToParamSelection` 手势）。
            if (op === "selectClipParamRange") {
                const clipId = typeof data?.clipId === "string" ? data.clipId : "";
                const clip = store.getState().session.clips.find((entry) => entry.id === clipId);
                if (!clip) return;
                const { startBound, endBound } = clipTimeRangeToFrameBounds(clip, fp);
                const rawMode = typeof data?.mode === "string" ? data.mode : "replace";
                const mode: "replace" | "add" | "toggle" =
                    rawMode === "add" || rawMode === "toggle" ? rawMode : "replace";
                selectionRef.current =
                    mode === "add"
                        ? addFrameRange(selectionRef.current, startBound, endBound)
                        : mode === "toggle"
                          ? toggleFrameRange(selectionRef.current, startBound, endBound)
                          : selectionFromFrames(startBound, endBound - startBound);
                setSelectionUi(selectionRef.current);
                setActiveSurfaceExplicit("pianoRoll");
                invalidate();
                return;
            }

            // 音频块范围 → 参数编辑器选区（批量入口；单个音频块的双击手势见
            // selectClipParamRange 的 add/toggle 模式）。
            //
            // 只取**当前参数编辑器所属根轨道组**内的音频块：参数编辑器一次只
            // 展示一条根轨道的参数，跨轨道的块范围对它没有意义（静默忽略，避免
            // 用户以为"加进去了"）。多段求并/相减交给 paramSelection 归一化
            // （相邻自动合并；相减可能把一段切成两段 —— 断层即数据）。
            if (op === "addClipsToParamSelection" || op === "removeClipsFromParamSelection") {
                const session = store.getState().session;
                const requestedIds = Array.isArray(data?.clipIds)
                    ? (data.clipIds as unknown[]).filter(
                          (id): id is string => typeof id === "string",
                      )
                    : session.multiSelectedClipIds.length > 0
                      ? session.multiSelectedClipIds
                      : session.selectedClipId
                        ? [session.selectedClipId]
                        : [];
                const ranges: FrameRange[] = [];
                for (const id of requestedIds) {
                    const clip = session.clips.find((entry) => entry.id === id);
                    if (!clip) continue;
                    if (resolveRootTrackId(session.tracks, clip.trackId) !== rootTrackId) continue;
                    const { startBound, endBound } = clipTimeRangeToFrameBounds(clip, fp);
                    ranges.push({ startFrame: startBound, frameCount: endBound - startBound });
                }
                if (ranges.length === 0) return;

                let next: ParamSelection | null = selectionRef.current;
                if (op === "addClipsToParamSelection") {
                    next = addFrameRanges(next, ranges);
                } else {
                    for (const range of ranges) {
                        next = subtractFrameRange(next, range.startFrame, frameRangeEnd(range));
                    }
                }
                selectionRef.current = next;
                setSelectionUi(next);
                setActiveSurfaceExplicit("pianoRoll");
                invalidate();
                return;
            }

            // VocalShifter clipboard paste stays a dedicated menu action
            // (file-based clipboard). 多选区：把全部选区段交给后端，落盘时只写
            // 落在这些段内的帧（断层不会被填充），偏移基准为首段起点。
            if (op === "pasteVocalShifter") {
                const sel2 = selectionRef.current;
                const selRanges = sel2 ? selectionToFrameRanges(sel2) : [];
                void dispatch(
                    pasteVocalShifterClipboard({
                        selectionRanges: selRanges.length > 0 ? selRanges : undefined,
                        activeParam: editParam,
                    }),
                );
                bumpRefreshToken();
                return;
            }

            // REAPERMedia fallback used by the normal paste operation when no
            // HiFiShifter param clipboard data is available. 多选区下这个回退只
            // 关心"往哪里贴"，用选区包围区间（首段起点 → 末段终点）。
            const pasteReaperClipboardFallback = () => {
                const sel2 = selectionRef.current;
                let selArgs:
                    | {
                          selectionStartFrame?: number;
                          selectionMaxFrames?: number;
                      }
                    | undefined;
                const bounding = selectionBoundingSpan(sel2);
                if (bounding) {
                    selArgs = {
                        selectionStartFrame: Math.max(0, bounding.startFrame),
                        selectionMaxFrames: Math.max(1, bounding.frameCount),
                    };
                }
                void (async () => {
                    try {
                        // Standard MIDI File on the clipboard opens the unified
                        // MIDI import dialog.
                        const midiCheck = await paramsApi.readMidiClipboardToMemory();
                        if (midiCheck.ok && midiCheck.guid) {
                            midiDialogSourceRef.current = "reaperClipboard";
                            setClipboardGuid(midiCheck.guid);
                            setMidiPath(null);
                            setMidiDialogSelection(
                                sel2 ? sel2.map((range) => ({ ...range })) : null,
                            );
                            midiDialogOpenParamsRef.current = {
                                editParam: s.editParam,
                                toolMode: s.toolMode,
                            };
                            setMidiDialogOpen(true);
                            return;
                        }
                    } catch {
                        // Check failed; fall back to ordinary REAPERMedia paste.
                    }
                    try {
                        // Avoid surfacing a paste error when the system
                        // clipboard does not contain REAPERMedia data at all.
                        const reaperCheck = await webApi.hasReaperClipboard();
                        if (!reaperCheck?.ok || !reaperCheck?.available) return;
                    } catch {
                        return;
                    }
                    void dispatch(pasteReaperClipboard(selArgs));
                })();
                bumpRefreshToken();
            };

            /**
             * 解析参数线剪贴板：内部缓存 → 系统剪贴板（后者优先，保持"最后复制的
             * 胜出"：时间轴复制过 Clip 后系统槽位已换，内部缓存随之失效）。
             */
            const readParamClipboardForPaste = async (): Promise<ParamClipboardData | null> => {
                let clip = clipboardRef.current;
                try {
                    const fromSystem = await readSystemClipboardObject("param");
                    if (fromSystem) {
                        clip = fromSystem;
                        clipboardRef.current = clip;
                    }
                } catch {
                    // 系统剪贴板不可用 → 退回内部缓存。
                }
                return clip ?? null;
            };

            // ── 粘贴：剪贴板优先；无选区时按剪贴板 + 播放光标推导选区 ──────────
            // 粘贴与其它操作不同：它的"作用对象"完全由**剪贴板**决定（从哪里开始、
            // 铺多宽、中间有几个断层）。因此无选区时**不做全选**，而是以当前播放
            // 光标为复制起点，按剪贴板自己的段布局重建选区（段数 / 段长 / 断层照搬），
            // 再把数据贴进去 —— 选区就是"粘贴会落到哪里"。
            //
            // 【推导出的选区**先不发布**】它要与粘贴后的曲线**同一次提交**落地：
            // 先亮出一个空选区、隔几毫秒再填上数据，会让用户看到"先划选区、再粘贴"
            // 两步；而用户的心智是"选区出现时粘贴就已经完成了"。因此这里只把它交给
            // 下面的目标帧换算，真正的 ref/UI 更新发生在 paste 分支里（与本地曲线
            // 更新同一次 setState 批处理）。
            //
            // 剪贴板里没有参数线数据时保持既有语义：落到 REAPERMedia / MIDI 剪贴板
            // 回退（那个回退的目标不是参数选区）；音高编辑不可用时同理。
            let pasteClipboard: ParamClipboardData | null = null;
            let derivedPasteSelection: ParamSelection | null = null;
            if (op === "paste") {
                if (!pitchEnabled) {
                    pasteReaperClipboardFallback();
                    return;
                }
                pasteClipboard = await readParamClipboardForPaste();
                if (!pasteClipboard) {
                    pasteReaperClipboardFallback();
                    return;
                }
                const selAtEntry = selectionRef.current;
                if (!selAtEntry || selAtEntry.length === 0) {
                    derivedPasteSelection = pasteTargetSelectionFromClipboard({
                        clipboard: pasteClipboard,
                        // 锚点 = 播放光标所在帧（与粘贴的帧制口径一致）。
                        anchorFrame: timeToFrame(s.playheadSec, fp),
                    });
                    if (!derivedPasteSelection) {
                        pasteReaperClipboardFallback();
                        return;
                    }
                }
            }

            // 目标选区：现有选区，或（粘贴且无选区时）刚推导出的那一份。
            const sel =
                (selectionRef.current?.length ? selectionRef.current : null) ??
                derivedPasteSelection;
            if (!sel || sel.length === 0) return;
            if (!pitchEnabled) return;

            // 选区的帧区间集合（逐段独立）。选区本来就是帧制，这里只做起点夹取、
            // 帧数钳制与越界截断（见 selectionToFrameRanges）。
            const selFrameRanges = selectionToFrameRanges(sel);
            if (selFrameRanges.length === 0) return;
            const firstRange = selFrameRanges[0];
            const startFrame = firstRange.startFrame;
            const frameCount = firstRange.frameCount;

            /**
             * 逐段执行「取数 → 变换 → 回写」。
             *
             * 撤销点纪律：**整个批次只在第一个真正写入的段上打一次撤销点**
             * （回调返回 false 表示该段未写入，撤销点顺延到下一段）。
             *
             * @param writeRange (段, 段号, 本次是否应打撤销点) => 是否实际写入
             */
            const runPerRange = async (
                writeRange: (
                    range: FrameRange,
                    rangeIndex: number,
                    isFirstWrite: boolean,
                ) => Promise<boolean>,
            ): Promise<boolean> => {
                let wrote = false;
                for (let i = 0; i < selFrameRanges.length; i += 1) {
                    const didWrite = await writeRange(selFrameRanges[i], i, !wrote);
                    if (didWrite) wrote = true;
                }
                return wrote;
            };

            /**
             * 粘贴的「本地先落」：把写入值先写进本地 paramView，并把**推导出的选区**
             * 在同一次 React 提交里发布。
             *
             * 【为什么需要】后端回写 + 重新取数要走若干个 IPC 往返。若只发布选区、
             * 等取数回来才显示曲线，用户看到的是"先出现一个空选区、隔一会儿才填上
             * 数据"两步；而用户的心智是"选区一出现，粘贴就已经完成"。这里与其它提交
             * 路径同一范式（先本地、后后端；失败时由 `bumpRefreshToken` 的取数纠正），
             * 于是选区与曲线在**同一次 setState 批处理**里落地 —— 感知上是一步。
             *
             * 已有选区时（`derivedPasteSelection` 为 null）只更新曲线，行为不变。
             *
             * @param writes 绝对帧号 + 逐帧值（全分辨率）。
             */
            const applyPasteLocally = (
                writes: readonly { startFrame: number; values: number[] }[],
            ): void => {
                if (derivedPasteSelection) {
                    selectionRef.current = derivedPasteSelection;
                    setSelectionUi(derivedPasteSelection);
                }
                const pv = paramViewRef.current;
                if (pv && writes.length > 0) {
                    const step = Math.max(1, Math.floor(pv.stride));
                    const nextEdit = pv.edit.slice();
                    for (const write of writes) {
                        for (let i = 0; i < write.values.length; i += 1) {
                            const idx = Math.round((write.startFrame + i - pv.startFrame) / step);
                            if (idx >= 0 && idx < nextEdit.length) {
                                nextEdit[idx] = write.values[i];
                            }
                        }
                    }
                    setParamView({ ...pv, edit: nextEdit });
                }
                invalidate();
            };

            // 音量 ↔ 动态 曲线互转（后端单事务：基线补偿换算 + 源参数归位）。
            //
            // 【为什么换算在后端】两者最终增益的语义不同：volume 是乘性增益，
            // dyn 是「目标电平 / 原声基线」。等效互转需要逐帧基线补偿
            // （dyn_target = volume × orig、volume = target/orig），而权威基线与
            // "曲线哪些帧有数据"只有后端知道 —— 前端只传选区（旧的前端纯搬迁
            // 已被证伪：gain = v/orig ≠ v，且会把未画帧物化成显式基线值）。
            //
            // 撤销点在后端单次打点：Ctrl+Z 一次回退整个互转（含源归位）。
            if (op === "convertVolumeToDyn" || op === "convertDynToVolume") {
                // 菜单项只在选中 volume / dyn 时出现，但键盘/程序化触发仍要复核，
                // 避免用一个不匹配的方向覆盖掉用户真正在编辑的参数。
                const fromNarrowed: "volume" | "dyn" =
                    op === "convertVolumeToDyn" ? "volume" : "dyn";
                if (editParam !== fromNarrowed) return;
                const plan = planParamConversion(fromNarrowed);
                if (!plan) return;

                const res = await paramsApi.convertMixParam(
                    rootTrackId,
                    fromNarrowed,
                    selFrameRanges.map((range) => ({
                        startFrame: range.startFrame,
                        frameCount: range.frameCount,
                    })),
                );
                if (!res?.ok) {
                    // 基线分析未就绪是最常见的失败原因：保持静默（与其它操作对
                    // not-ok 的处理一致），dyn_orig_updated 事件后用户重试即可。
                    return;
                }
                bumpRefreshToken();
                // 目标参数可能刚获得第一个非默认值（尤其动态）→ 切换显示，
                // 让用户立刻看到互转结果而不是停在空白的源参数上。
                dispatch(setEditParam(plan.targetParam));
                return;
            }

            // 选区编辑统一入口：取数/编辑/边缘淡化/回写全部在
            // selectionEditApply 模块内完成（delta 空间交叉淡化 + 毫秒定标）。
            // 多选区逐段独立执行，整批只打一个撤销点。
            // 平滑度解析顺序保持旧语义：对话框显式传入 → store 全局设置。
            const runSelectionEdit = async (
                editSelection: (currentSelectionVals: number[]) => number[],
                extension?: SelectionEditExtension,
                options?: { preserveDynSentinels?: boolean },
            ) => {
                const ok = await applySelectionEditOverRanges({
                    ranges: selFrameRanges,
                    trackId: rootTrackId,
                    param: editParam,
                    framePeriodMs: fp,
                    smoothnessPercent: clamp(
                        Number(
                            (data?.edgeSmoothnessPercent as number | undefined) ??
                                s.edgeSmoothnessPercent,
                        ) || 0,
                        0,
                        100,
                    ),
                    editSelection,
                    extension,
                    isEditable: editParam === "pitch" ? editablePitchValue : undefined,
                    ...options,
                });
                if (ok) bumpRefreshToken();
            };

            switch (op) {
                case "copy": {
                    // 逐段取全分辨率数据；段偏移 = 该段起点 − 首段起点，断层以
                    // 偏移空洞的形式进入剪贴板（这是"不合并断层"的根本保障）。
                    const segments: ParamClipboardSegment[] = [];
                    let framePeriodMsFromBackend = fp;
                    for (const range of selFrameRanges) {
                        const res = await paramsApi.getParamFrames(
                            rootTrackId,
                            editParam,
                            range.startFrame,
                            range.frameCount,
                            1,
                            true,
                            isDynParam(editParam),
                        );
                        if (!res?.ok) continue;
                        const payload = res as ParamFramesPayload;
                        if (segments.length === 0) {
                            framePeriodMsFromBackend = Number(payload.frame_period_ms ?? fp) || fp;
                        }
                        // dyn：未画帧在复制时就编码回哨兵（负值）。后端
                        // set_param_frames 入口原样接受负值 = 沿用原声，因此
                        // 粘贴路径无需任何特判 —— "未画"语义跨复制/粘贴存活。
                        const sentinels = payload.edit_sentinel;
                        const values = (payload.edit ?? []).map((v, i) =>
                            sentinels?.[i] === true ? DYN_FOLLOW_ORIG : Number(v) || 0,
                        );
                        if (values.length === 0) continue;
                        segments.push({
                            startFrame: range.startFrame - startFrame,
                            values,
                        });
                    }
                    if (segments.length === 0) return;
                    const clipboardData: ParamClipboardData = {
                        param: editParam,
                        framePeriodMs: framePeriodMsFromBackend,
                        segments,
                    };
                    clipboardRef.current = clipboardData;
                    try {
                        await writeSystemClipboardObject(toParamClipboardPayload(clipboardData));
                    } catch {
                        // ignore clipboard write failures
                    }
                    // 刷新剪贴板预览
                    invalidate();
                    break;
                }
                case "cut": {
                    // 复制部分与 copy 完全同构（多段 + 偏移），随后把各段恢复为
                    // 原始值；恢复与复制在同一批次内完成，撤销一次整体回退。
                    const segments: ParamClipboardSegment[] = [];
                    let framePeriodMsFromBackend = fp;
                    let restoredAny = false;
                    for (const range of selFrameRanges) {
                        const res = await paramsApi.getParamFrames(
                            rootTrackId,
                            editParam,
                            range.startFrame,
                            range.frameCount,
                            1,
                            true,
                            isDynParam(editParam),
                        );
                        if (!res?.ok) continue;
                        const payload = res as ParamFramesPayload;
                        if (segments.length === 0) {
                            framePeriodMsFromBackend = Number(payload.frame_period_ms ?? fp) || fp;
                        }
                        // dyn：未画帧编码回哨兵（与 copy 同口径，见该处说明）。
                        const sentinels = payload.edit_sentinel;
                        const values = (payload.edit ?? []).map((v, i) =>
                            sentinels?.[i] === true ? DYN_FOLLOW_ORIG : Number(v) || 0,
                        );
                        if (values.length > 0) {
                            segments.push({
                                startFrame: range.startFrame - startFrame,
                                values,
                            });
                        }
                        // 第一个真正写入的段打撤销点
                        await paramsApi.restoreParamFrames(
                            rootTrackId,
                            editParam,
                            range.startFrame,
                            range.frameCount,
                            !restoredAny,
                        );
                        restoredAny = true;
                    }
                    if (segments.length > 0) {
                        const clipboardData: ParamClipboardData = {
                            param: editParam,
                            framePeriodMs: framePeriodMsFromBackend,
                            segments,
                        };
                        clipboardRef.current = clipboardData;
                        try {
                            await writeSystemClipboardObject(
                                toParamClipboardPayload(clipboardData),
                            );
                        } catch {
                            // ignore clipboard write failures
                        }
                    }
                    invalidate();
                    bumpRefreshToken();
                    break;
                }
                case "paste": {
                    // 剪贴板已在上方（进入通用守卫之前）解析过 —— 无选区时的目标
                    // 选区就是据它推导的，这里直接复用同一份，避免再读一次系统剪贴板。
                    const clip = pasteClipboard;
                    if (!clip) return;

                    // 剪贴板 → 目标选区的映射（预览与粘贴同源）：交集之外不写，
                    // 断层两侧都保持原值。
                    const writes = mapClipboardToTargetRanges({
                        targetRanges: selFrameRanges,
                        clipboard: clip,
                    });
                    if (writes.length === 0) return;

                    if (clip.param !== editParam) {
                        // 跨参数粘贴：仅支持 pitch → 子轨音高偏移参数（逐段转换）。
                        const canConvert =
                            clip.param === "pitch" &&
                            (isChildPitchOffsetCentsParam(editParam) ||
                                isChildPitchOffsetDegreesParam(editParam));
                        if (!canConvert) return;
                        const targetParam = parseChildPitchOffsetParam(editParam);
                        if (!targetParam) return;
                        const resolvedRootTrackId = resolveRootTrackId(
                            s.tracks,
                            targetParam.trackId,
                        );
                        if (!resolvedRootTrackId || resolvedRootTrackId !== rootTrackId) {
                            return;
                        }

                        const convertedWrites: Array<{ startFrame: number; values: number[] }> = [];
                        for (const write of writes) {
                            const converted = await buildChildOffsetPasteValuesHelper({
                                tracks: s.tracks,
                                rootTrackId,
                                targetTrackId: targetParam.trackId,
                                startFrame: write.startFrame,
                                frameCount: write.values.length,
                                clipboardPitch: write.values,
                                mode: targetParam.mode as "cents" | "degrees",
                                paramsApi,
                                pitchDeltaToDegreeSteps: pitchDeltaToDegreeSteps,
                                projectScale: effectiveProjectScale,
                                // Tempo Map 感知：按帧时刻解析生效音阶。
                                scaleAtFrame: (frame: number) =>
                                    projectScaleAtSec((frame * fp) / 1000) ?? effectiveProjectScale,
                            });
                            if (!converted) continue;
                            convertedWrites.push({
                                startFrame: write.startFrame,
                                values: converted.slice(0, write.values.length),
                            });
                        }
                        if (convertedWrites.length === 0) return;
                        // 选区（若为新推导）+ 本地曲线：同一次提交落地，再走后端回写。
                        applyPasteLocally(convertedWrites);
                        try {
                            await uploadFullResCurveSegments({
                                trackId: rootTrackId,
                                param: editParam,
                                segments: convertedWrites,
                            });
                        } catch (err) {
                            console.error("[pianoRoll] paste (converted) failed", err);
                        } finally {
                            // 无论成败都重新取数：失败时把上面乐观写入的曲线纠正回后端真值
                            // （与拖拽提交路径同一纪律）。
                            bumpRefreshToken();
                        }
                        break;
                    }

                    const pasteWrites = writes.map((write) => ({
                        startFrame: write.startFrame,
                        values: write.values,
                    }));
                    // 选区（若为新推导）+ 本地曲线：同一次提交落地，再走后端回写。
                    applyPasteLocally(pasteWrites);
                    try {
                        await uploadFullResCurveSegments({
                            trackId: rootTrackId,
                            param: editParam,
                            segments: pasteWrites,
                        });
                    } catch (err) {
                        console.error("[pianoRoll] paste failed", err);
                    } finally {
                        // 同拖拽提交路径：失败时用重新取数把乐观写入的曲线纠正回来。
                        bumpRefreshToken();
                    }
                    break;
                }
                case "initialize": {
                    await runPerRange(async (range, _index, isFirstWrite) => {
                        const restored = await paramsApi.restoreParamFrames(
                            rootTrackId,
                            editParam,
                            range.startFrame,
                            range.frameCount,
                            isFirstWrite,
                        );
                        return Boolean(restored?.ok);
                    });
                    bumpRefreshToken();
                    break;
                }
                case "average": {
                    const strengthPercent = clamp(Number(data?.strength ?? 100) || 0, 0, 100);
                    if (strengthPercent <= 0) return;
                    const res = await paramsApi.getParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        frameCount,
                        1,
                        true,
                        isDynParam(editParam),
                    );
                    if (!res?.ok) return;
                    const payload = res as ParamFramesPayload;
                    const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
                    if (vals.length === 0) return;
                    const result = averageSelectionValues(vals, editParam, strengthPercent);
                    // dyn：未画帧写回哨兵（防止"沿用原声"被物化成显式目标电平）。
                    if (isDynParam(editParam)) {
                        restoreDynSentinels(result, payload.edit_sentinel);
                    }
                    await paramsApi.setParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        result,
                        true,
                    );
                    bumpRefreshToken();
                    break;
                }
                case "transposeCents": {
                    const cents = Number(data?.cents ?? 0);
                    if (cents === 0) return;
                    const delta = cents / 100;
                    await runSelectionEdit(
                        (vals) =>
                            editParam === "pitch"
                                ? vals.map((v) => (v === 0 ? 0 : v + delta))
                                : vals.map((v) => v + delta),
                        { kind: "deltaAt", deltaAt: () => delta },
                    );
                    break;
                }
                case "transposeDegrees": {
                    const degrees = Number(data?.degrees ?? 0);
                    const scaleToken = String(data?.scale ?? "__project__");
                    // “工程音阶”受 Tempo Map 影响：按每个帧的时刻取生效音阶。
                    const fixedScale: ScaleLike | null =
                        scaleToken === "__project__" ? null : resolveScaleFromToken(scaleToken);
                    const degreeSteps = degreeInputToScaleSteps(degrees);
                    if (degreeSteps === 0) return;
                    const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
                    await runSelectionEdit(
                        (vals) => {
                            return editParam === "pitch"
                                ? vals.map((midi, i) => {
                                      if (midi === 0) return 0;
                                      const scale =
                                          fixedScale ??
                                          projectScaleAtSec(((startFrame + i) * fpMs) / 1000) ??
                                          "C";
                                      return transposePitchByScaleSteps(midi, degreeSteps, scale);
                                  })
                                : vals.map((midi, i) => {
                                      const scale =
                                          fixedScale ??
                                          projectScaleAtSec(((startFrame + i) * fpMs) / 1000) ??
                                          "C";
                                      return transposePitchByScaleSteps(midi, degreeSteps, scale);
                                  });
                        },
                        // 选区外的延拓 delta：按延拓帧自己的生效音阶逐帧计算
                        {
                            kind: "deltaAt",
                            deltaAt: (frame, baseValue) => {
                                const scale =
                                    fixedScale ?? projectScaleAtSec((frame * fpMs) / 1000) ?? "C";
                                return (
                                    transposePitchByScaleSteps(baseValue, degreeSteps, scale) -
                                    baseValue
                                );
                            },
                        },
                    );
                    break;
                }
                case "setPitch": {
                    const parsed = Number(data?.value ?? data?.midiNote);
                    const midiNote = Number.isFinite(parsed) ? parsed : 60;
                    await runSelectionEdit(
                        (vals) =>
                            editParam === "pitch"
                                ? vals.map((v) => (v === 0 ? 0 : midiNote))
                                : vals.map(() => midiNote),
                        // 选区外延拓 = 目标值本身：向目标的自然滑移
                        { kind: "editedAt", editedAt: () => midiNote },
                        // 显式写常量：用户意图是覆盖整个选区 → 不保留未画哨兵。
                        { preserveDynSentinels: false },
                    );
                    break;
                }
                case "shiftParamUpSelection":
                case "shiftParamDownSelection": {
                    // 长按重复的节拍守卫：上一拍（后端读写仍在途）未完成时
                    // 跳过本拍 —— 在途标记经 selectionEditInFlight 与 App
                    // 端 fire 共享，两端双重检查避免重复事件堆积。
                    if (!beginSelectionParamEdit()) return;
                    try {
                        const descriptor = processorParamsRef.current.find(
                            (param) => param.id === editParam,
                        );
                        const magnitude = parseParamShiftMagnitude(data?.magnitude);
                        // dyn 是倍率域（0 = 静音）：**档位命令**上下移动用乘性 ——
                        // 一次 ±shift = ×2^±magnitude（默认 ×2 / ×0.5 = ±6 dB），
                        // 与 DAW 的"增益 ±6 dB"同一语义；0 帧保持 0。选区外延拓同样
                        // 按乘性表达：`deltaAt = base × (factor − 1)`，边缘淡化在
                        // delta 空间里得到的就是"同一系数作用下的差值"。
                        //
                        // ⚠ 拖拽**不走**这条法则：拖拽一律是值域内线性偏移，因为
                        // 只有线性偏移能让被抓住的那一点始终停在光标下（见
                        // `paramRanges.shiftValueForDrag`）。命令是离散的增益档位，
                        // 不涉及"跟手"，两者语义不同、不应互相"对齐"。
                        if (isDynParam(editParam)) {
                            // 档位（对齐 pitch 的 fine/normal/coarse 节奏）：
                            // fine ≈ +0.6 dB、normal = ×2（+6 dB）、coarse = ×4（+12 dB）。
                            const dynStepDelta =
                                magnitude === "fine" ? 0.05 : magnitude === "coarse" ? 1.0 : 0.5;
                            const factor = dynMultiplicativeFactor(
                                op === "shiftParamUpSelection" ? dynStepDelta : -dynStepDelta,
                            );
                            await runSelectionEdit((vals) => vals.map((v) => v * factor), {
                                kind: "deltaAt",
                                deltaAt: (_f, base) => base * (factor - 1),
                            });
                            break;
                        }
                        const step = getParamShiftStep(editParam, descriptor, magnitude);
                        const delta = op === "shiftParamUpSelection" ? step : -step;
                        // 不透传 data?.edgeSmoothnessPercent：键盘路径的事件
                        // detail 不携带该值，直接沿用 store 的边缘平滑设置
                        // （runSelectionEdit 内部的解析顺序已保证该行为）。
                        await runSelectionEdit((vals) => vals.map((v) => v + delta), {
                            kind: "deltaAt",
                            deltaAt: () => delta,
                        });
                    } finally {
                        endSelectionParamEdit();
                    }
                    break;
                }
                case "smooth": {
                    const strength = clamp((Number(data?.strength ?? 50) || 0) / 100, 0, 1);
                    if (strength <= 0) return;
                    const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
                    // 多取两侧各 3σ 帧上下文：高斯平滑用真实延拓做边界，
                    // 平滑结果与选区外曲线无缝（旧实现只取选区内、边界处
                    // 会产生新台阶）。多选区逐段独立取上下文与 σ。
                    const pad = smoothContextPadFrames(strength, fpMs);
                    await runPerRange(async (range, _index, isFirstWrite) => {
                        const ctxStart = Math.max(0, range.startFrame - pad);
                        const leftLen = range.startFrame - ctxStart;
                        const res = await paramsApi.getParamFrames(
                            rootTrackId,
                            editParam,
                            ctxStart,
                            leftLen + range.frameCount + pad,
                            1,
                            true,
                            isDynParam(editParam),
                        );
                        if (!res?.ok) return false;
                        const payload = res as ParamFramesPayload;
                        const all = (payload.edit ?? []).map((v) => Number(v));
                        const vals = all.slice(leftLen, leftLen + range.frameCount);
                        if (vals.length === 0) return false;
                        const result = smoothSelectionValues(vals, editParam, strength, {
                            framePeriodMs: fpMs,
                            leftContext: all.slice(0, leftLen),
                            rightContext: all.slice(leftLen + range.frameCount),
                        });
                        // dyn：未画帧写回哨兵（防止"沿用原声"被物化成显式目标电平）。
                        if (isDynParam(editParam)) {
                            restoreDynSentinels(
                                result,
                                payload.edit_sentinel?.slice(leftLen, leftLen + range.frameCount),
                            );
                        }
                        const written = await paramsApi.setParamFrames(
                            rootTrackId,
                            editParam,
                            range.startFrame,
                            result,
                            isFirstWrite,
                        );
                        return Boolean(written?.ok);
                    });
                    bumpRefreshToken();
                    break;
                }
                case "addVibrato": {
                    const amplitude = Number(data?.amplitude ?? 30);
                    const rateHz = Number(data?.rate ?? 5.5);
                    const period = rateHz > 0 ? 1000 / rateHz : 200;
                    const attack = Number(data?.attack ?? 50);
                    const release = Number(data?.release ?? 50);
                    const phase = Number(data?.phase ?? 0);
                    // 多选区：咬合/释放包络按**每段自身时长**定标，各段独立。
                    await runPerRange(async (range, _index, isFirstWrite) => {
                        const res = await paramsApi.getParamFrames(
                            rootTrackId,
                            editParam,
                            range.startFrame,
                            range.frameCount,
                            1,
                            true,
                            isDynParam(editParam),
                        );
                        if (!res?.ok) return false;
                        const payload = res as ParamFramesPayload;
                        const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
                        const fpMs = Number(payload.frame_period_ms ?? fp) || fp;
                        const totalMs = vals.length * fpMs;
                        const attackMs = Math.min(attack, totalMs / 2);
                        const releaseMs = Math.min(release, totalMs / 2);
                        // For pitch: amplitude in cents → divide by 100 to get semitones
                        // For dyn: amplitude is a **depth percentage** (±N% ratio
                        // modulation) — multiplicative so drawn silence stays silent.
                        // For other params: amplitude is a raw value used directly as max deviation
                        const isPitchVib = editParam === "pitch";
                        const isDynVib = isDynParam(editParam);
                        const ampFactor = isPitchVib
                            ? amplitude / 100
                            : isDynVib
                              ? amplitude / 100
                              : amplitude;
                        const result = vals.map((v, i) => {
                            const tMs = i * fpMs;
                            let env = 1;
                            if (tMs < attackMs) env = tMs / Math.max(1, attackMs);
                            else if (tMs > totalMs - releaseMs)
                                env = (totalMs - tMs) / Math.max(1, releaseMs);
                            const phaseRad = (phase * Math.PI) / 180;
                            const vib = Math.sin(
                                (2 * Math.PI * tMs) / Math.max(1, period) + phaseRad,
                            );
                            // dyn：乘性调制（v × (1 + 深度·包络·正弦)）—— 静音帧
                            // （v = 0）保持 0；深度 > 100% 时负半周钳到 0 = 静音。
                            const next = isDynVib
                                ? v * (1 + ampFactor * env * vib)
                                : v + ampFactor * env * vib;
                            return isDynVib ? Math.max(0, next) : next;
                        });
                        // dyn：未画帧写回哨兵（"沿用原声"不被颤音物化）。
                        if (isDynParam(editParam)) {
                            restoreDynSentinels(result, payload.edit_sentinel);
                        }
                        const written = await paramsApi.setParamFrames(
                            rootTrackId,
                            editParam,
                            range.startFrame,
                            result,
                            isFirstWrite,
                        );
                        return Boolean(written?.ok);
                    });
                    bumpRefreshToken();
                    break;
                }
                case "quantize": {
                    if (editParam !== "pitch") {
                        const fallbackUnit = currentParamQuantizeUnit;
                        const quantizeUnit = Math.abs(
                            Number(data?.quantizeUnit ?? fallbackUnit) || fallbackUnit,
                        );
                        if (!Number.isFinite(quantizeUnit) || quantizeUnit <= 0) return;
                        const tolerance = Math.abs(
                            Number(data?.tolerance ?? data?.toleranceCents ?? 0) || 0,
                        );
                        const defaultValue = currentParamDefaultValue;
                        // 走统一选区编辑编排（含边缘淡化，平滑度取对话框/全局设置；
                        // 强度为 0 时与旧的纯选区写入逐字节一致）。量化 delta 在选区
                        // 外的延拓用缺省规则：边界帧实际 delta 常数延拓。
                        await runSelectionEdit((vals) =>
                            vals.map((v) => {
                                const stepCount = Math.round((v - defaultValue) / quantizeUnit);
                                const snapped = defaultValue + stepCount * quantizeUnit;
                                if (Math.abs(v - snapped) <= tolerance) return v;
                                return snapped + (v > snapped ? 1 : -1) * tolerance;
                            }),
                        );
                        break;
                    }

                    const unit = (data?.unit as string) ?? "semitone";
                    const scaleToken = String(data?.scale ?? "__project__");
                    // “工程音阶”受 Tempo Map 影响：按每个帧的时刻取生效音阶。
                    const fixedScale: ScaleLike | null =
                        scaleToken === "__project__" ? null : resolveScaleFromToken(scaleToken);
                    const toleranceCents = Math.abs(
                        Math.round(Number(data?.toleranceCents ?? 0) || 0),
                    );
                    const toleranceSemitone = toleranceCents / 100;
                    // project base scale is controlled from toolbar; do not change it here
                    const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
                    const scaleAt = (i: number): ScaleLike =>
                        fixedScale ?? projectScaleAtSec(((startFrame + i) * fpMs) / 1000) ?? "C";
                    await runSelectionEdit((vals) =>
                        unit === "semitone"
                            ? vals.map((v) =>
                                  editParam === "pitch" && v === 0
                                      ? 0
                                      : (() => {
                                            const snapped = snapToSemitone(v);
                                            return Math.abs(v - snapped) <= toleranceSemitone
                                                ? v
                                                : snapped +
                                                      (v - snapped > 0 ? 1 : -1) *
                                                          toleranceSemitone;
                                        })(),
                              )
                            : vals.map((v, i) =>
                                  editParam === "pitch" && v === 0
                                      ? 0
                                      : (() => {
                                            const snapped = snapToScale(v, scaleAt(i));
                                            return Math.abs(v - snapped) <= toleranceSemitone
                                                ? v
                                                : snapped +
                                                      (v - snapped > 0 ? 1 : -1) *
                                                          toleranceSemitone;
                                        })(),
                              ),
                    );
                    break;
                }
                case "meanQuantize": {
                    if (editParam !== "pitch") {
                        const fallbackUnit = currentParamQuantizeUnit;
                        const quantizeUnit = Math.abs(
                            Number(data?.quantizeUnit ?? fallbackUnit) || fallbackUnit,
                        );
                        if (!Number.isFinite(quantizeUnit) || quantizeUnit <= 0) return;
                        const tolerance = Math.abs(
                            Number(data?.tolerance ?? data?.toleranceCents ?? 0) || 0,
                        );
                        const defaultValue = currentParamDefaultValue;
                        // 均值量化是整体平移：delta 在 editSelection 内计算并捕获，
                        // 选区外按同一 delta 延拓（保持均值量化语义）。
                        let valueMeanDelta = 0;
                        await runSelectionEdit(
                            (vals) => {
                                if (vals.length === 0) return vals;
                                const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                                const stepCount = Math.round((avg - defaultValue) / quantizeUnit);
                                const quantizedAvg = defaultValue + stepCount * quantizeUnit;
                                valueMeanDelta = quantizedAvg - avg;
                                return vals.map((v) => {
                                    const moved = v + valueMeanDelta;
                                    if (Math.abs(moved - v) <= tolerance) return v;
                                    return moved + (v > moved ? 1 : -1) * tolerance;
                                });
                            },
                            { kind: "deltaAt", deltaAt: () => valueMeanDelta },
                        );
                        break;
                    }

                    const unit = (data?.unit as string) ?? "semitone";
                    const scaleToken = String(data?.scale ?? "__project__");
                    // “工程音阶”受 Tempo Map 影响：均值吸附使用选区中点时刻的生效音阶，
                    // 整体平移量保持统一（均值量化语义）。
                    const fixedScale: ScaleLike | null =
                        scaleToken === "__project__" ? null : resolveScaleFromToken(scaleToken);
                    const toleranceCents = Math.abs(
                        Math.round(Number(data?.toleranceCents ?? 0) || 0),
                    );
                    const toleranceSemitone = toleranceCents / 100;
                    const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
                    let meanDelta = 0;
                    await runSelectionEdit(
                        (vals) => {
                            // pitch=0 视为未编辑，不参与均值；全部未浊时 delta=0，
                            // 结果与输入逐帧相同（相比旧的直接 return 会多一个
                            // 无变化的撤销点，无副作用，可接受）。
                            const nonZero = vals.filter((v) => v !== 0);
                            if (nonZero.length === 0) return vals.slice();
                            const avg = nonZero.reduce((a, b) => a + b, 0) / nonZero.length;
                            const midScale =
                                fixedScale ??
                                projectScaleAtSec(
                                    ((startFrame + Math.floor(vals.length / 2)) * fpMs) / 1000,
                                ) ??
                                "C";
                            const quantizedAvg =
                                unit === "semitone"
                                    ? snapToSemitone(avg)
                                    : snapToScale(avg, midScale);
                            meanDelta = quantizedAvg - avg;
                            return vals.map((v) => {
                                if (v === 0) return 0;
                                const moved = v + meanDelta;
                                return Math.abs(moved - v) <= toleranceSemitone
                                    ? v
                                    : moved + (v - moved > 0 ? 1 : -1) * toleranceSemitone;
                            });
                        },
                        { kind: "deltaAt", deltaAt: () => meanDelta },
                    );
                    break;
                }
            }
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps -- s.editParam/s.toolMode 在调用时经 s 快照读取；加入依赖会改变对话框快照捕获/监听重挂时序（既有模式）
        [
            rootTrackId,
            editParam,
            s.tracks,
            paramView?.framePeriodMs,
            dynamicProjectSec,
            s.edgeSmoothnessPercent,
            effectiveProjectScale,
            projectScaleAtSec,
            resolveScaleFromToken,
            currentParamRange,
            currentParamDefaultValue,
            currentParamQuantizeUnit,
            pitchEnabled,
            pitchDeltaToDegreeSteps,
            bumpRefreshToken,
            invalidate,
            dispatch,
            selectAllParamRange,
        ],
    );

    // Keep the ref in sync so usePianoRollInteractions can dispatch edit ops
    handleEditActionRef.current = (op: string) => void handleEditOp(op);

    // Listen for edit operations dispatched from MenuBar / 全局路由
    // hifi:editOp 是参数编辑器专属通道（事件名即契约）：全局路由
    // （focusRouting.resolveEditOpRoute）按活动编辑表面把编辑操作定向派发
    // 到这里，消费者信任事件、不再自行判断焦点 —— 旧版在此重猜
    // activeElement / body 属性，与时间轴侧判断互相矛盾，正是复制/剪切/
    // 粘贴冲突的根因。selectAll/deselect 的工具模式守卫在 handleEditOp 内。
    useEffect(() => {
        const handler = (e: Event) => {
            const detail = (e as CustomEvent).detail;
            if (!detail?.op) return;
            const { op, ...data } = detail;
            void handleEditOp(op, data);
        };
        window.addEventListener("hifi:editOp", handler);
        return () => window.removeEventListener("hifi:editOp", handler);
    }, [handleEditOp]);

    // 单剪贴板纪律：时间轴复制/剪切替换整个应用剪贴板后（copyClips 成功时
    // 派发 hifi:clipboardReplaced），参数线内部剪贴板缓存随之失效 —— 否则
    // "复制 Clip 后在参数编辑器粘贴"会把更早复制、已被剪贴板替换掉的参数线
    // 数据从内部缓存复活，违反"剪贴板只保留最后复制的一份"的语义。
    useEffect(() => {
        const handler = () => {
            clipboardRef.current = null;
            invalidate();
        };
        window.addEventListener("hifi:clipboardReplaced", handler);
        return () => window.removeEventListener("hifi:clipboardReplaced", handler);
    }, [invalidate]);

    // Dispatch helper: context menu dialog ops → open MenuBar dialogs
    const openEditDialog = useCallback(
        (dialog: string) => {
            // 为颤音对话框附带当前参数范围信息
            let paramRange: { min: number; max: number } | undefined;
            if (dialog === "addVibrato") {
                const desc = processorParamsRef.current.find((d) => d.id === editParam);
                if (desc?.kind.type === "automation_curve") {
                    paramRange = {
                        min: desc.kind.min_value,
                        max: desc.kind.max_value,
                    };
                }
            }
            window.dispatchEvent(
                new CustomEvent("hifi:openEditDialog", {
                    detail: { dialog, paramRange },
                }),
            );
        },
        [editParam],
    );

    /**
     * 「另存为音高参考」：每个选区段生成一个独立的 Pitch Ref clip
     * （不合并断层 —— 合并会把缺口处也填上参考音高）。
     *
     * 无选区时先做一次隐式全选（作用域 = 整条参数曲线），与右键菜单里其它
     * 以选区为作用域的操作一致 —— 否则这个菜单项点下去会毫无反应。
     */
    const handleSaveAsPitchRef = useCallback(async () => {
        if (!rootTrackId) return;
        if (!selectionRef.current || selectionRef.current.length === 0) {
            selectAllParamRange();
        }
        const sel = selectionRef.current;
        if (!sel || sel.length === 0) return;

        const fp = paramView?.framePeriodMs ?? 5;
        const selFrameRanges = selectionToFrameRanges(sel);
        if (selFrameRanges.length === 0) return;

        // 逐段取 pitch → MIDI 音符事件（与旧单选区同一转换，保留浮点音高）
        const clipTemplates: Array<{
            startSec: number;
            lengthSec: number;
            midiNoteData: Array<{
                startSec: number;
                endSec: number;
                note: number;
                velocity: number;
                channel: number;
            }>;
        }> = [];
        for (const range of selFrameRanges) {
            const res = await paramsApi.getParamFrames(
                rootTrackId,
                "pitch",
                range.startFrame,
                range.frameCount,
                1,
            );
            if (!res?.ok || !res.edit) continue;
            const pitchValues: number[] = (res.edit as number[]).map((v) => Number(v) || 0);
            if (pitchValues.length === 0) continue;

            const startSec = (range.startFrame * fp) / 1000;
            const lengthSec = Math.max(0.01, (pitchValues.length * fp) / 1000);

            // Convert pitch values (semitones) to MIDI note events
            // 保留原始浮点音高值，不进行半音量化
            const fpSec = fp / 1000;
            const midiNoteData: Array<{
                startSec: number;
                endSec: number;
                note: number;
                velocity: number;
                channel: number;
            }> = [];
            let segStartFrame = 0;
            let currentNote = pitchValues[0];
            for (let i = 1; i < pitchValues.length; i++) {
                const note = pitchValues[i];
                if (Math.abs(note - currentNote) > 0.001) {
                    midiNoteData.push({
                        startSec: segStartFrame * fpSec,
                        endSec: i * fpSec,
                        note: currentNote,
                        velocity: 100,
                        channel: 0,
                    });
                    segStartFrame = i;
                    currentNote = note;
                }
            }
            midiNoteData.push({
                startSec: segStartFrame * fpSec,
                endSec: pitchValues.length * fpSec,
                note: currentNote,
                velocity: 100,
                channel: 0,
            });
            clipTemplates.push({ startSec, lengthSec, midiNoteData });
        }
        if (clipTemplates.length === 0) return;

        // Determine target track: try the track above the currently selected track.
        // If no track above exists, or the above track has overlapping clips
        // in the import time range, create a new track above the current track.
        const orderedTrackIds = s.tracks.map((t) => t.id);
        const trackIndexById: Record<string, number> = {};
        orderedTrackIds.forEach((id, idx) => {
            trackIndexById[id] = idx;
        });

        const currentIdx = s.selectedTrackId ? (trackIndexById[s.selectedTrackId] ?? -1) : -1;
        let targetTrackId: string | null = null;

        if (currentIdx > 0) {
            const aboveTrackId = orderedTrackIds[currentIdx - 1];

            const hasOverlap = s.clips.some(
                (c) =>
                    c.trackId === aboveTrackId &&
                    clipTemplates.some(
                        (template) =>
                            c.startSec < template.startSec + template.lengthSec &&
                            c.startSec + c.lengthSec > template.startSec,
                    ),
            );
            if (!hasOverlap) {
                const currentTrack = s.tracks.find((t) => t.id === s.selectedTrackId);
                const aboveTrack = s.tracks.find((t) => t.id === aboveTrackId);
                if (
                    currentTrack &&
                    aboveTrack &&
                    currentTrack.depth != null &&
                    aboveTrack.depth != null &&
                    currentTrack.depth >= aboveTrack.depth
                ) {
                    targetTrackId = aboveTrackId;
                }
            }
        }

        if (!targetTrackId) {
            // Create a new track above the current track
            const currentTrack = s.tracks.find((t) => t.id === s.selectedTrackId);
            const newTrackPayload: Record<string, unknown> = {
                name: undefined,
                parentTrackId: currentTrack?.parentId ?? null,
            };
            if (currentIdx >= 0) {
                newTrackPayload.index = currentIdx;
            }
            const result = await dispatch(
                addTrackRemote(newTrackPayload as { name?: string; parentTrackId?: string | null }),
            ).unwrap();
            const added = result as {
                selected_track_id?: string;
                tracks?: Array<{ id: string }>;
            };
            targetTrackId =
                added.selected_track_id ?? added.tracks?.[added.tracks.length - 1]?.id ?? null;
        }

        if (!targetTrackId) return;

        await dispatch(
            createClipsRemote({
                templates: clipTemplates.map((template) => ({
                    trackId: targetTrackId as string,
                    name: "Pitch Ref",
                    startSec: template.startSec,
                    lengthSec: template.lengthSec,
                    midiNoteData: template.midiNoteData,
                    midiFillGaps: true,
                })),
            }),
        );
    }, [
        selectionRef,
        rootTrackId,
        paramView,
        s.tracks,
        s.selectedTrackId,
        s.clips,
        dispatch,
        selectAllParamRange,
    ]);

    /**
     * 「导出 MIDI」：每个选区段作为一条独立的导出条目
     * （后端按 track 条目逐条导出，断层因此不会被填上音符）。
     */
    const handleExportMidiFromEditor = useCallback(async () => {
        if (!rootTrackId) return;
        // 无选区时先隐式全选（作用域 = 整条参数曲线），与菜单里其它以选区为
        // 作用域的操作一致。
        if (!selectionRef.current || selectionRef.current.length === 0) {
            selectAllParamRange();
        }
        const sel = selectionRef.current;
        if (!sel || sel.length === 0) return;

        const saveResult = await coreApi.pickMidiOutputPath();
        if (!saveResult.ok || saveResult.canceled || !saveResult.path) return;

        const selectedTrack = s.tracks.find((t) => t.id === s.selectedTrackId);
        const trackName = selectedTrack?.name ?? "Track";
        const scaleNotes =
            SCALE_NOTES[(s.project?.baseScale as keyof typeof SCALE_NOTES) ?? "C"] ?? SCALE_NOTES.C;

        await paramsApi.exportPitchToMidi({
            outputPath: saveResult.path,
            tracks: sel.map((range) => {
                // 半开帧区间 → 秒（右端是最后一帧的右缘）。
                const fp = paramView?.framePeriodMs ?? 5;
                const startSec = framesToTime(range.startFrame, fp);
                const endSec = Math.max(startSec + 0.01, framesToTime(frameRangeEnd(range), fp));
                return {
                    trackId: s.selectedTrackId ?? rootTrackId,
                    rootTrackId,
                    name: trackName,
                    startSec,
                    endSec,
                };
            }),
            bpm: s.bpm,
            beatsPerBar: s.project?.beatsPerBar ?? 4,
            baseScale: s.project?.baseScale ?? "C",
            projectScaleNotes: scaleNotes,
        });
    }, [rootTrackId, selectionRef, paramView?.framePeriodMs, s, selectAllParamRange]);

    // Pitch Snap 设置弹窗状态
    const [pitchSnapOpen, setPitchSnapOpen] = useState(false);

    const vibratoToolIcon = (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M1.5 7.5C3 7.5 3 3.5 4.5 3.5C6 3.5 6 11.5 7.5 11.5C9 11.5 9 3.5 10.5 3.5C12 3.5 12 7.5 13.5 7.5"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );

    const pitchSnapSemitoneIcon = (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M2.5 12.5H6.5V9.5H10.5V5.5H12.5"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );

    const pitchSnapScaleIcon = (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <rect x="2.5" y="10.5" width="4" height="2" rx="0.5" fill="currentColor" />
            <rect x="6.5" y="7.5" width="4" height="2" rx="0.5" fill="currentColor" />
            <rect x="10.5" y="4.5" width="2.5" height="2" rx="0.5" fill="currentColor" />
        </svg>
    );

    const currentDrawToolIcon = currentDrawTool === "vibrato" ? vibratoToolIcon : <Pencil1Icon />;

    // 统一刻度源：标尺刻度与背景网格线共用，与时间线侧同一实现，
    // 保证两个面板的网格/标尺位置严格同源于 axis 投影。
    const timelineTicks = useMemo(
        () =>
            buildTimelineTicks({
                axis: prAxis,
                bpm: s.bpm,
                beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
                grid: s.grid,
                primaryUnit: s.primaryTimeUnit,
                secondaryUnit: s.secondaryTimeUnit,
                minLabelSpacingPx: s.rulerLabelSpacingPx,
                minGridSpacingPx: s.timelineSnap.gridMinSpacingPx,
                swingPercent: s.timelineSnap.swingEnabled ? s.timelineSnap.swingPercent : 0,
                tempoMap: s.tempoMap,
            }),
        [
            prAxis,
            s.bpm,
            s.beats,
            s.grid,
            s.primaryTimeUnit,
            s.secondaryTimeUnit,
            s.rulerLabelSpacingPx,
            s.timelineSnap,
            s.tempoMap,
        ],
    );
    const timeContext = useMemo<TimeFormatContext>(
        () => ({
            bpm: s.bpm,
            beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
            grid: s.grid,
            tempoMap: s.tempoMap,
        }),
        [s.bpm, s.beats, s.grid, s.tempoMap],
    );

    const handlePrimaryUnitChange = useCallback(
        (unit: TimeUnit) => {
            dispatch(setPrimaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const handleSecondaryUnitChange = useCallback(
        (unit: TimeUnitChoice) => {
            dispatch(setSecondaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );

    const handleTempoMapChange = useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
        },
        [dispatch],
    );
    const handleTempoMapCommit = useCallback(
        (next: TempoMap | null) => {
            dispatch(setTempoMap(next));
            void dispatch(setTempoMapRemote(next));
        },
        [dispatch],
    );
    const handleCopyPlayheadTime = useCallback(async () => {
        const text = formatCursorTime(
            s.primaryTimeUnit,
            s.secondaryTimeUnit,
            Number(s.playheadSec ?? 0),
            timeContext,
        ).combined;
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            // 忽略复制失败
        }
    }, [s.primaryTimeUnit, s.secondaryTimeUnit, s.playheadSec, timeContext]);

    return (
        <Flex
            ref={paramEditorRef}
            direction="column"
            className="relative h-full w-full bg-qt-graph-bg border-t border-qt-border"
            // 编辑表面声明：文档级 pointerdown/focusin 捕获据此把整个参数
            // 编辑器（工具栏/标尺/卷帘）解析为「pianoRoll」表面，作为复制/
            // 剪切/粘贴等编辑快捷键的归属依据（见 focusSurface.ts）。
            data-hs-surface="pianoRoll"
        >
            {/* Header / Parameter Switch */}
            <Flex
                align="center"
                justify="between"
                className="h-8 bg-qt-base border-b border-qt-border px-2 shrink-0"
            >
                <Flex align="center" gap="2" style={{ flex: "1 1 auto", minWidth: 0 }}>
                    <IconButton
                        size="1"
                        variant={s.paramEditorSyncTimeline ? "solid" : "ghost"}
                        data-tooltip={tAny("sync_timeline_view_tooltip")}
                        aria-label={tAny("sync_timeline_view")}
                        tabIndex={-1}
                        onClick={() => {
                            dispatch(setParamEditorSyncTimeline(!s.paramEditorSyncTimeline));
                            void dispatch(persistUiSettings());
                        }}
                    >
                        {s.paramEditorSyncTimeline ? <Link2Icon /> : <LinkBreak2Icon />}
                    </IconButton>
                    <Text size="1" weight="bold" color="gray">
                        {tAny("param_editor_short")}
                    </Text>
                    {/* 音高吸附按钮，紧邻 param_editor 右侧，留 8px 空白 */}
                    <Flex gap="1" align="center" style={{ marginLeft: 8 }}>
                        <IconButton
                            size="1"
                            variant={s.toolModeGroup === "select" ? "solid" : "ghost"}
                            data-tooltip={t("select")}
                            tabIndex={-1}
                            onClick={() => dispatch(setToolMode("select"))}
                        >
                            <CursorArrowIcon />
                        </IconButton>
                        <Box style={{ position: "relative" }} data-hs-context-menu>
                            <IconButton
                                size="1"
                                variant={s.toolModeGroup === "draw" ? "solid" : "ghost"}
                                data-tooltip={drawToolButtonTitle}
                                tabIndex={-1}
                                onClick={() => dispatch(setToolMode(currentDrawTool))}
                                onContextMenu={(e) => {
                                    e.preventDefault();
                                    setDrawToolMenuOpen(true);
                                }}
                            >
                                <Box
                                    style={{
                                        position: "relative",
                                        width: 15,
                                        height: 15,
                                    }}
                                >
                                    <Box
                                        style={{
                                            position: "absolute",
                                            inset: 0,
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                        }}
                                    >
                                        {currentDrawToolIcon}
                                    </Box>
                                    <Box
                                        style={{
                                            position: "absolute",
                                            right: -1,
                                            bottom: -1,
                                            width: 6,
                                            height: 6,
                                            opacity: 0.7,
                                        }}
                                    >
                                        <svg
                                            width="6"
                                            height="6"
                                            viewBox="0 0 6 6"
                                            fill="none"
                                            xmlns="http://www.w3.org/2000/svg"
                                        >
                                            <path d="M0 6L6 0V6Z" fill="currentColor" />
                                        </svg>
                                    </Box>
                                </Box>
                            </IconButton>

                            {drawToolMenuOpen && (
                                <Box
                                    ref={drawToolMenuRef}
                                    data-hs-context-menu
                                    className="absolute left-0 top-[calc(100%+4px)] z-30 min-w-[190px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
                                >
                                    {[
                                        {
                                            mode: "draw" as const,
                                            label: tAny("draw_tool"),
                                            icon: <Pencil1Icon />,
                                        },
                                        {
                                            mode: "vibrato" as const,
                                            label: tAny("vibrato_draw_tool"),
                                            icon: vibratoToolIcon,
                                        },
                                    ].map((item) => {
                                        const active = currentDrawTool === item.mode;
                                        return (
                                            <button
                                                key={item.mode}
                                                type="button"
                                                className={`w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover`}
                                                onClick={() => {
                                                    dispatch(setToolMode(item.mode));
                                                    setDrawToolMenuOpen(false);
                                                }}
                                                onPointerDown={(e) => e.stopPropagation()}
                                            >
                                                <Flex align="center" gap="2">
                                                    <Box
                                                        style={{
                                                            display: "flex",
                                                            width: 15,
                                                            height: 15,
                                                            alignItems: "center",
                                                            justifyContent: "center",
                                                        }}
                                                    >
                                                        {item.icon}
                                                    </Box>
                                                    <Text size="1">{item.label}</Text>
                                                </Flex>
                                                {active ? <CheckIcon /> : null}
                                            </button>
                                        );
                                    })}
                                </Box>
                            )}
                        </Box>

                        <Box
                            style={{
                                width: 1,
                                height: 18,
                                background: "var(--gray-8)",
                                marginInline: 4,
                                opacity: 0.9,
                            }}
                        />
                        {/* 拖动方向按钮 */}
                        <IconButton
                            size="1"
                            color="gray"
                            variant={activeDragDirection === "free" ? "ghost" : "solid"}
                            data-tooltip={`${tAny("drag_direction")}: ${tAny(activeDragDirection === "free" ? "drag_direction_free" : activeDragDirection === "x-only" ? "drag_direction_x_only" : "drag_direction_y_only")}${
                                isNoneBinding(cycleDragDirectionKb)
                                    ? ""
                                    : ` (${formatKeybinding(cycleDragDirectionKb, "")})`
                            }`}
                            tabIndex={-1}
                            onClick={() => {
                                dispatch(cycleDragDirection(activeDragDirectionTool));
                                void dispatch(persistUiSettings());
                            }}
                        >
                            {activeDragDirection === "free" ? (
                                <svg
                                    width="15"
                                    height="15"
                                    viewBox="0 0 15 15"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path
                                        d="M3.5 11.5L11.5 3.5M11.5 3.5L8 3.5M11.5 3.5L11.5 7M3.5 11.5L7 11.5M3.5 11.5L3.5 8"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                </svg>
                            ) : activeDragDirection === "x-only" ? (
                                <svg
                                    width="15"
                                    height="15"
                                    viewBox="0 0 15 15"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path
                                        d="M2 7.5H13M2 7.5L4.5 5M2 7.5L4.5 10M13 7.5L10.5 5M13 7.5L10.5 10"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                </svg>
                            ) : (
                                <svg
                                    width="15"
                                    height="15"
                                    viewBox="0 0 15 15"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path
                                        d="M7.5 2V13M7.5 2L5 4.5M7.5 2L10 4.5M7.5 13L5 10.5M7.5 13L10 10.5"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                </svg>
                            )}
                        </IconButton>
                        <Box style={{ position: "relative" }} data-hs-context-menu>
                            <IconButton
                                size="1"
                                variant={effectivePitchSnapVisual ? "solid" : "ghost"}
                                data-tooltip={`${t("pitch_snap")}: ${
                                    effectivePitchSnapVisual
                                        ? s.pitchSnapUnit === "semitone"
                                            ? tAny("quantize_semitone")
                                            : tAny("quantize_scale")
                                        : tAny("pitch_snap_off")
                                }`}
                                tabIndex={-1}
                                onClick={() => {
                                    dispatch(togglePitchSnap());
                                    void dispatch(persistUiSettings());
                                }}
                                onContextMenu={(e) => {
                                    e.preventDefault();
                                    setPitchSnapMenuOpen(true);
                                }}
                            >
                                <Box
                                    style={{
                                        position: "relative",
                                        width: 15,
                                        height: 15,
                                    }}
                                >
                                    <Box
                                        style={{
                                            position: "absolute",
                                            inset: 0,
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                        }}
                                    >
                                        {!effectivePitchSnapVisual ? (
                                            <Box
                                                style={{
                                                    position: "relative",
                                                    width: 15,
                                                    height: 15,
                                                    opacity: 0.45,
                                                }}
                                            >
                                                {pitchSnapSemitoneIcon}
                                                <svg
                                                    className="absolute inset-0"
                                                    width="15"
                                                    height="15"
                                                    viewBox="0 0 15 15"
                                                    fill="none"
                                                    xmlns="http://www.w3.org/2000/svg"
                                                >
                                                    <path
                                                        d="M3 3L12 12"
                                                        stroke="currentColor"
                                                        strokeWidth="1.2"
                                                        strokeLinecap="round"
                                                    />
                                                </svg>
                                            </Box>
                                        ) : s.pitchSnapUnit === "semitone" ? (
                                            pitchSnapSemitoneIcon
                                        ) : (
                                            pitchSnapScaleIcon
                                        )}
                                    </Box>
                                    <Box
                                        style={{
                                            position: "absolute",
                                            right: -1,
                                            bottom: -1,
                                            width: 6,
                                            height: 6,
                                            opacity: 0.7,
                                        }}
                                    >
                                        <svg
                                            width="6"
                                            height="6"
                                            viewBox="0 0 6 6"
                                            fill="none"
                                            xmlns="http://www.w3.org/2000/svg"
                                        >
                                            <path d="M0 6L6 0V6Z" fill="currentColor" />
                                        </svg>
                                    </Box>
                                </Box>
                            </IconButton>

                            {pitchSnapMenuOpen && (
                                <Box
                                    ref={pitchSnapMenuRef}
                                    data-hs-context-menu
                                    className="absolute left-0 top-[calc(100%+4px)] z-30 min-w-[190px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
                                >
                                    <button
                                        type="button"
                                        className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                        onClick={() => {
                                            dispatch(setPitchSnapUnit("semitone"));
                                            if (!s.pitchSnapEnabled) {
                                                dispatch(togglePitchSnap());
                                            }
                                            void dispatch(persistUiSettings());
                                            setPitchSnapMenuOpen(false);
                                        }}
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <Flex align="center" gap="2">
                                            <Box
                                                style={{
                                                    display: "flex",
                                                    width: 15,
                                                    height: 15,
                                                    alignItems: "center",
                                                    justifyContent: "center",
                                                }}
                                            >
                                                {pitchSnapSemitoneIcon}
                                            </Box>
                                            <span>{tAny("pitch_snap_menu_semitone")}</span>
                                        </Flex>
                                        {s.pitchSnapUnit === "semitone" ? <CheckIcon /> : null}
                                    </button>
                                    <button
                                        type="button"
                                        className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                        onClick={() => {
                                            dispatch(setPitchSnapUnit("scale"));
                                            if (!s.pitchSnapEnabled) {
                                                dispatch(togglePitchSnap());
                                            }
                                            void dispatch(persistUiSettings());
                                            setPitchSnapMenuOpen(false);
                                        }}
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <Flex align="center" gap="2">
                                            <Box
                                                style={{
                                                    display: "flex",
                                                    width: 15,
                                                    height: 15,
                                                    alignItems: "center",
                                                    justifyContent: "center",
                                                }}
                                            >
                                                {pitchSnapScaleIcon}
                                            </Box>
                                            <span>{tAny("pitch_snap_menu_scale")}</span>
                                        </Flex>
                                        {s.pitchSnapUnit === "scale" ? <CheckIcon /> : null}
                                    </button>
                                    <div className="my-1 border-t border-qt-border" />
                                    <button
                                        type="button"
                                        className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                        onClick={() => {
                                            setPitchSnapMenuOpen(false);
                                            setPitchSnapOpen(true);
                                        }}
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <span>{tAny("pitch_snap_settings_action")}</span>
                                    </button>
                                </Box>
                            )}
                        </Box>
                        <IconButton
                            size="1"
                            variant={s.scaleHighlightMode === "always" ? "solid" : "ghost"}
                            data-tooltip={tAny("scale_highlight")}
                            tabIndex={-1}
                            onClick={() => {
                                dispatch(
                                    setScaleHighlightMode(
                                        s.scaleHighlightMode === "always" ? "off" : "always",
                                    ),
                                );
                                void dispatch(persistUiSettings());
                            }}
                        >
                            {s.scaleHighlightMode === "always" ? (
                                <svg
                                    width="14"
                                    height="14"
                                    viewBox="0 0 14 14"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <circle cx="5" cy="9" r="2.2" fill="currentColor" />
                                    <path
                                        d="M7 4V8.5"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                    />
                                    <path
                                        d="M7 4L11 3.2"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        strokeLinecap="round"
                                    />
                                </svg>
                            ) : (
                                <svg
                                    width="14"
                                    height="14"
                                    viewBox="0 0 14 14"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <circle
                                        cx="5"
                                        cy="9"
                                        r="2.2"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        fill="none"
                                    />
                                    <path
                                        d="M7 4V8.5"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                    />
                                    <path
                                        d="M7 4L11 3.2"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        strokeLinecap="round"
                                    />
                                </svg>
                            )}
                        </IconButton>
                        <IconButton
                            size="1"
                            variant={s.lockParamLinesEnabled ? "solid" : "ghost"}
                            data-tooltip={t("lock_param_lines")}
                            tabIndex={-1}
                            onClick={() => {
                                dispatch(toggleLockParamLines());
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <svg
                                width="15"
                                height="15"
                                viewBox="0 0 15 15"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <rect
                                    x="3"
                                    y="6"
                                    width="9"
                                    height="7"
                                    rx="1"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    fill="none"
                                />
                                <path
                                    d="M5 6V4.5C5 3.12 6.12 2 7.5 2C8.88 2 10 3.12 10 4.5V6"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    fill="none"
                                />
                            </svg>
                        </IconButton>
                        <Flex align="center" gap="1" ml="2" style={{ minWidth: 0, flexShrink: 1 }}>
                            <Text size="1" data-tooltip={tAny("edge_smoothness")}>
                                {tAny("edge_smoothness_short")}:
                            </Text>
                            <input
                                ref={edgeSmoothnessWheelRef}
                                className="qt-range"
                                type="range"
                                min={0}
                                max={100}
                                step={1}
                                value={Math.round(s.edgeSmoothnessPercent)}
                                onChange={(e) => {
                                    const next = Number(e.currentTarget.value);
                                    dispatch(setEdgeSmoothnessPercent(next));
                                }}
                                onPointerUp={() => {
                                    void dispatch(persistUiSettings());
                                }}
                                onKeyUp={() => {
                                    void dispatch(persistUiSettings());
                                }}
                                style={{
                                    // 根据工具栏拥挤程度自动伸缩：宽裕时最多 120px，拥挤时缩到 48px
                                    flex: "1 1 auto",
                                    width: 120,
                                    minWidth: 48,
                                    maxWidth: 120,
                                }}
                            />
                            <Text size="1" style={{ minWidth: 36, textAlign: "right" }}>
                                {Math.round(s.edgeSmoothnessPercent)}%
                            </Text>
                        </Flex>
                    </Flex>
                </Flex>

                {/* Pitch Snap 设置弹窗 */}
                <PitchSnapSettingsDialog open={pitchSnapOpen} onOpenChange={setPitchSnapOpen} />
                <TimelineDisplaySettingsDialog
                    open={timeDisplaySettingsOpen}
                    onOpenChange={setTimeDisplaySettingsOpen}
                />

                <Flex gap="2" align="center">
                    <Flex gap="1" align="center">
                        {/* 参考轨道组 / 导入 MIDI：仅当切换到“音高”参数时显示（位置固定在“音高”左侧）。
                            按钮样式与其他工具按钮一致（Radix soft），简写 + ToolTip 保留全称。 */}
                        {rootTrack && editParam === "pitch" ? (
                            <React.Fragment>
                                <DropdownMenu.Root>
                                    <DropdownMenu.Trigger data-tooltip={t("reference_root_tracks")}>
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="gray"
                                            style={{ cursor: "pointer" }}
                                        >
                                            {buildReferenceRootTrackTriggerElement(
                                                `${tAny("reference_root_tracks_short")}${
                                                    visibleReferenceRootTrackIds.length > 0
                                                        ? ` (${visibleReferenceRootTrackIds.length})`
                                                        : ""
                                                }`,
                                            )}
                                            <ChevronDownIcon width="12" height="12" />
                                        </Button>
                                    </DropdownMenu.Trigger>
                                    <DropdownMenu.Content variant="soft" color="gray">
                                        <DropdownMenu.Item
                                            onSelect={() =>
                                                updateVisibleReferenceRootTrackIds(
                                                    referenceRootTrackOptions.map(
                                                        (track) => track.id,
                                                    ),
                                                )
                                            }
                                        >
                                            {t("reference_root_tracks_all")}
                                        </DropdownMenu.Item>
                                        <DropdownMenu.Item
                                            onSelect={() => updateVisibleReferenceRootTrackIds([])}
                                        >
                                            {t("reference_root_tracks_clear")}
                                        </DropdownMenu.Item>
                                        <DropdownMenu.Separator />
                                        {referenceRootTrackOptions.length === 0 ? (
                                            <DropdownMenu.Item disabled>
                                                {t("reference_root_tracks_empty")}
                                            </DropdownMenu.Item>
                                        ) : (
                                            referenceRootTrackOptions.map((track) => (
                                                <DropdownMenu.CheckboxItem
                                                    key={track.id}
                                                    checked={visibleReferenceRootTrackIds.includes(
                                                        track.id,
                                                    )}
                                                    onCheckedChange={() => {
                                                        dispatch(
                                                            toggleVisibleReferenceRootTrackId(
                                                                track.id,
                                                            ),
                                                        );
                                                        void dispatch(persistUiSettings());
                                                    }}
                                                    onPointerEnter={() =>
                                                        setHoveredReferenceRootTrackId(track.id)
                                                    }
                                                    onPointerLeave={() =>
                                                        setHoveredReferenceRootTrackId(null)
                                                    }
                                                >
                                                    <Flex align="center" gap="2">
                                                        <span
                                                            className="inline-block h-2.5 w-2.5 rounded-full"
                                                            style={{
                                                                background:
                                                                    buildReferencePitchStrokeColor(
                                                                        track.color,
                                                                        true,
                                                                    ),
                                                            }}
                                                        />
                                                        <span>{track.name}</span>
                                                    </Flex>
                                                </DropdownMenu.CheckboxItem>
                                            ))
                                        )}
                                    </DropdownMenu.Content>
                                </DropdownMenu.Root>
                                <span
                                    className="inline-flex"
                                    data-tooltip={pitchHardDisableReason ?? tAny("midi_import")}
                                >
                                    <Button
                                        size="1"
                                        /* 跟随全局强调色（原独立 blue 与播放键的 iris 是两种蓝） */
                                        variant="soft"
                                        onClick={handleOpenMidiDialog}
                                        disabled={!pitchEnabled}
                                        style={{ cursor: "pointer" }}
                                    >
                                        {tAny("midi_import")}
                                    </Button>
                                </span>
                            </React.Fragment>
                        ) : null}
                        {selectedIsChildTrack &&
                        (childPitchOffsetCentsParam || childPitchOffsetDegreesParam) ? (
                            <DropdownMenu.Root>
                                <ParamToolbarPill
                                    label={pitchGroupLabel}
                                    labelTooltip={pitchGroupTooltip}
                                    active={pitchGroupActive}
                                    onSelect={() => dispatch(setEditParam("pitch"))}
                                    eyeMode={
                                        pitchGroupActive
                                            ? "main"
                                            : secondaryParamVisible["pitch"]
                                              ? "on"
                                              : "off"
                                    }
                                    onToggleEye={() => toggleSecondaryParam("pitch")}
                                    eyeTooltip={
                                        secondaryParamVisible["pitch"]
                                            ? t("secondary_overlay_tooltip_visible")
                                            : t("secondary_overlay_tooltip_hidden")
                                    }
                                    eyeLabel={
                                        secondaryParamVisible["pitch"]
                                            ? t("hide_secondary_param")
                                            : t("show_secondary_param")
                                    }
                                    dropdown={
                                        <DropdownMenu.Trigger
                                            className="param-pill__seg param-pill__seg--chev"
                                            data-tooltip={pitchGroupTooltip}
                                            tabIndex={-1}
                                        >
                                            <ChevronDownIcon width="12" height="12" />
                                        </DropdownMenu.Trigger>
                                    }
                                />
                                <DropdownMenu.Content variant="soft" color="gray">
                                    <DropdownMenu.RadioGroup
                                        value={editParam}
                                        onValueChange={(value) => dispatch(setEditParam(value))}
                                    >
                                        <DropdownMenu.RadioItem value="pitch">
                                            {t("child_pitch_root_option")}
                                        </DropdownMenu.RadioItem>
                                        {childPitchOffsetCentsParam ? (
                                            <DropdownMenu.RadioItem
                                                value={childPitchOffsetCentsParam}
                                            >
                                                {t("child_pitch_cents_option")}
                                            </DropdownMenu.RadioItem>
                                        ) : null}
                                        {childPitchOffsetDegreesParam ? (
                                            <DropdownMenu.RadioItem
                                                value={childPitchOffsetDegreesParam}
                                            >
                                                {t("child_pitch_degrees_option")}
                                            </DropdownMenu.RadioItem>
                                        ) : null}
                                    </DropdownMenu.RadioGroup>
                                </DropdownMenu.Content>
                            </DropdownMenu.Root>
                        ) : (
                            <ParamToolbarPill
                                label={t("param_btn_pitch")}
                                labelTooltip={t("pitch")}
                                active={editParam === "pitch"}
                                onSelect={() => dispatch(setEditParam("pitch"))}
                                eyeMode={
                                    editParam === "pitch"
                                        ? "main"
                                        : secondaryParamVisible["pitch"]
                                          ? "on"
                                          : "off"
                                }
                                onToggleEye={() => toggleSecondaryParam("pitch")}
                                eyeTooltip={
                                    secondaryParamVisible["pitch"]
                                        ? t("secondary_overlay_tooltip_visible")
                                        : t("secondary_overlay_tooltip_hidden")
                                }
                                eyeLabel={
                                    secondaryParamVisible["pitch"]
                                        ? t("hide_secondary_param")
                                        : t("show_secondary_param")
                                }
                            />
                        )}
                        {/* 由后端 processorParams 驱动的动态参数按钮（按算法排列后的顺序） */}
                        {orderedProcessorParams.map((p) => {
                            if (p.id === "formant_shift_cents") {
                                return (
                                    <ParamGroupButton
                                        key={p.id}
                                        rootParamId={p.id}
                                        rootLabel={getProcessorParamShortLabel(p)}
                                        rootMenuLabel={t("child_formant_root_option")}
                                        rootTooltip={getProcessorParamLabel(p)}
                                        childParamId={
                                            selectedIsChildTrack ? childFormantOffsetParam : null
                                        }
                                        childLabel={t("child_formant_mode")}
                                        childMenuLabel={t("child_formant_offset_option")}
                                        rootActive={editParam === p.id}
                                        childActive={editParam === childFormantOffsetParam}
                                        secondaryVisible={secondaryParamVisible[p.id] ?? false}
                                        hideSecondaryLabel={t("hide_secondary_param")}
                                        showSecondaryLabel={t("show_secondary_param")}
                                        hideSecondaryTooltip={t("secondary_overlay_tooltip_hidden")}
                                        showSecondaryTooltip={t(
                                            "secondary_overlay_tooltip_visible",
                                        )}
                                        onSelectRoot={() => dispatch(setEditParam(p.id))}
                                        onSelectChild={() => {
                                            if (childFormantOffsetParam) {
                                                dispatch(setEditParam(childFormantOffsetParam));
                                            }
                                        }}
                                        onToggleSecondary={() => toggleSecondaryParam(p.id)}
                                    />
                                );
                            }

                            // 音量 / 动态：同量纲的两个混音级参数，用一个药丸 + 下拉切换。
                            // 只在遇到 "volume" 时渲染这一次（"dyn" 会被跳过），
                            // 否则会得到两个内容相同的按钮。
                            if (p.id === "volume") {
                                const hasDyn = orderedProcessorParams.some((q) => q.id === "dyn");
                                const volDesc = processorParamsRef.current.find(
                                    (d) => d.id === "volume",
                                );
                                const dynDesc = processorParamsRef.current.find(
                                    (d) => d.id === "dyn",
                                );
                                const groupSecondaryVisible =
                                    (secondaryParamVisible["volume"] ?? false) ||
                                    (secondaryParamVisible["dyn"] ?? false);
                                return (
                                    <ParamGroupButton
                                        key="volume-group"
                                        rootParamId="volume"
                                        rootLabel={
                                            volDesc
                                                ? getProcessorParamShortLabel(volDesc)
                                                : t("param_btn_volume")
                                        }
                                        rootMenuLabel={t("volume_label")}
                                        rootTooltip={t("volume_label")}
                                        childParamId={hasDyn ? "dyn" : null}
                                        childLabel={
                                            dynDesc
                                                ? getProcessorParamShortLabel(dynDesc)
                                                : t("param_btn_dyn")
                                        }
                                        childMenuLabel={t("dyn_label")}
                                        rootActive={editParam === "volume"}
                                        childActive={editParam === "dyn"}
                                        secondaryVisible={groupSecondaryVisible}
                                        hideSecondaryLabel={t("hide_secondary_param")}
                                        showSecondaryLabel={t("show_secondary_param")}
                                        hideSecondaryTooltip={t("secondary_overlay_tooltip_hidden")}
                                        showSecondaryTooltip={t(
                                            "secondary_overlay_tooltip_visible",
                                        )}
                                        alwaysShowDropdown
                                        onSelectRoot={() => dispatch(setEditParam("volume"))}
                                        onSelectChild={() => {
                                            if (hasDyn) dispatch(setEditParam("dyn"));
                                        }}
                                        onToggleSecondary={() => {
                                            // 眼睛对整组生效：音量与动态的副参数叠加一起切换，
                                            // 否则会出现"眼睛亮着但只有一条叠加线"的分裂观感。
                                            if (groupSecondaryVisible) {
                                                if (secondaryParamVisible["volume"]) {
                                                    toggleSecondaryParam("volume");
                                                }
                                                if (secondaryParamVisible["dyn"]) {
                                                    toggleSecondaryParam("dyn");
                                                }
                                            } else {
                                                if (!secondaryParamVisible["volume"]) {
                                                    toggleSecondaryParam("volume");
                                                }
                                                if (hasDyn && !secondaryParamVisible["dyn"]) {
                                                    toggleSecondaryParam("dyn");
                                                }
                                            }
                                        }}
                                    />
                                );
                            }

                            // dyn 已并入上面的音量组药丸，不再单独渲染。
                            if (p.id === "dyn") {
                                return null;
                            }

                            const paramActive = editParam === p.id;
                            const paramEyeVisible = secondaryParamVisible[p.id] ?? false;

                            // nsf-hifigan：把“气声开关”以图标片段融合进“气声音量”药丸。
                            let breathTrailing: React.ReactNode = null;
                            if (p.id === "breath_gain") {
                                const breathDesc = processorStaticParams.find(
                                    (sp) => sp.id === "breath_enabled",
                                );
                                const breathDefault =
                                    breathDesc && breathDesc.kind.type === "static_enum"
                                        ? breathDesc.kind.default_value
                                        : 0;
                                const breathOn =
                                    (processorStaticValues["breath_enabled"] ?? breathDefault) ===
                                    1;
                                breathTrailing = (
                                    <button
                                        type="button"
                                        tabIndex={-1}
                                        className="param-pill__seg param-pill__seg--breath"
                                        data-tooltip={
                                            breathOn
                                                ? t("breath_tooltip_on")
                                                : t("breath_tooltip_off")
                                        }
                                        aria-label={`${t("breath_mode_label")}: ${
                                            breathOn ? t("switch_on") : t("switch_off")
                                        }`}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            void handleStaticParamChange(
                                                "breath_enabled",
                                                breathOn ? 0 : 1,
                                            );
                                        }}
                                    >
                                        <BreathAirIcon off={!breathOn} />
                                    </button>
                                );
                            }

                            return (
                                <ParamToolbarPill
                                    key={p.id}
                                    label={getProcessorParamShortLabel(p)}
                                    labelTooltip={getProcessorParamLabel(p)}
                                    active={paramActive}
                                    onSelect={() => dispatch(setEditParam(p.id))}
                                    eyeMode={paramActive ? "main" : paramEyeVisible ? "on" : "off"}
                                    onToggleEye={() => toggleSecondaryParam(p.id)}
                                    eyeTooltip={
                                        paramEyeVisible
                                            ? t("secondary_overlay_tooltip_visible")
                                            : t("secondary_overlay_tooltip_hidden")
                                    }
                                    eyeLabel={
                                        paramEyeVisible
                                            ? t("hide_secondary_param")
                                            : t("show_secondary_param")
                                    }
                                    trailing={breathTrailing}
                                />
                            );
                        })}
                    </Flex>

                    {rootTrack ? (
                        <Flex align="center" gap="2">
                            {processorStaticParams.map((param) => {
                                if (param.kind.type !== "static_enum") return null;
                                const currentValue =
                                    processorStaticValues[param.id] ?? param.kind.default_value;

                                // 气声开关已融合进“气声音量”参数的药丸中（见下方
                                // breath_gain 的 trailing 片段），此处不再单独渲染。
                                if (param.id === "breath_enabled") {
                                    return null;
                                }

                                // vslib 的合成模式：改为支持滚轮切换的下拉栏。
                                if (param.id === "synth_mode") {
                                    const stringOptions = param.kind.options.map(([, value]) =>
                                        String(value),
                                    );
                                    const currentString = String(currentValue);
                                    const selectOptions = param.kind.options.map(
                                        ([label, value]) => ({
                                            value,
                                            label: getStaticOptionLabel(param.id, label, value),
                                        }),
                                    );
                                    const currentOptionLabel =
                                        selectOptions.find(
                                            (opt) => String(opt.value) === currentString,
                                        )?.label ?? currentString;
                                    return (
                                        <Select.Root
                                            key={param.id}
                                            value={currentString}
                                            onValueChange={(v) =>
                                                void handleStaticParamChange(param.id, Number(v))
                                            }
                                        >
                                            <Select.Trigger
                                                // 与“算法”下拉栏一致使用固定宽度，选项切换时宽度不变
                                                className="w-[140px]"
                                                data-tooltip={`${t("vslib_synth_mode_label")}: ${currentOptionLabel}`}
                                                onWheel={(event) => {
                                                    applySelectWheelChange({
                                                        event,
                                                        currentValue: currentString,
                                                        options: stringOptions,
                                                        onChange: (next) =>
                                                            void handleStaticParamChange(
                                                                param.id,
                                                                Number(next),
                                                            ),
                                                    });
                                                }}
                                            />
                                            <Select.Content>
                                                {selectOptions.map((opt) => (
                                                    <Select.Item
                                                        key={`${param.id}-${opt.value}`}
                                                        value={String(opt.value)}
                                                    >
                                                        {opt.label}
                                                    </Select.Item>
                                                ))}
                                            </Select.Content>
                                        </Select.Root>
                                    );
                                }

                                return (
                                    <Flex key={param.id} align="center" gap="1">
                                        <Text
                                            size="1"
                                            color="gray"
                                            data-tooltip={getProcessorParamLabel(param)}
                                        >
                                            {getProcessorParamLabel(param)}
                                        </Text>
                                        {param.kind.options.map(([label, value]) => (
                                            <Button
                                                key={`${param.id}-${value}`}
                                                size="1"
                                                variant={currentValue === value ? "solid" : "soft"}
                                                color={currentValue === value ? "blue" : "gray"}
                                                onClick={() => {
                                                    void handleStaticParamChange(param.id, value);
                                                }}
                                                style={{
                                                    cursor: "pointer",
                                                }}
                                            >
                                                {getStaticOptionLabel(param.id, label, value)}
                                            </Button>
                                        ))}
                                    </Flex>
                                );
                            })}
                            <Text size="1" color="gray" data-tooltip={tAny("algo_label")}>
                                {tAny("algo_label_short")}
                            </Text>
                            <Select.Root
                                value={
                                    ["world_dll", "nsf_hifigan_onnx", "vslib", "none"].includes(
                                        rootTrack.pitchAnalysisAlgo,
                                    )
                                        ? rootTrack.pitchAnalysisAlgo
                                        : "nsf_hifigan_onnx"
                                }
                                onValueChange={(v) => {
                                    if (!rootTrackId) return;
                                    dispatch(
                                        setTrackStateRemote({
                                            trackId: rootTrackId,
                                            pitchAnalysisAlgo: v,
                                        }),
                                    );
                                }}
                            >
                                <Select.Trigger
                                    className="min-w-[140px]"
                                    onWheel={(event) => {
                                        const currentValue = [
                                            "world_dll",
                                            "nsf_hifigan_onnx",
                                            "vslib",
                                            "none",
                                        ].includes(rootTrack.pitchAnalysisAlgo)
                                            ? rootTrack.pitchAnalysisAlgo
                                            : "nsf_hifigan_onnx";
                                        applySelectWheelChange({
                                            event,
                                            currentValue,
                                            options: [
                                                "world_dll",
                                                "nsf_hifigan_onnx",
                                                "vslib",
                                                "none",
                                            ],
                                            onChange: (next) => {
                                                if (!rootTrackId) return;
                                                dispatch(
                                                    setTrackStateRemote({
                                                        trackId: rootTrackId,
                                                        pitchAnalysisAlgo: next,
                                                    }),
                                                );
                                            },
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value="world_dll">world</Select.Item>
                                    <Select.Item value="nsf_hifigan_onnx">nsf-hifigan</Select.Item>
                                    <Select.Item value="vslib">vslib</Select.Item>
                                    <Select.Item value="none">{t("none")}</Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </Flex>
                    ) : null}
                </Flex>
            </Flex>

            {/* Task 6.5: 参数面板顶部添加进度条区 ?*/}
            {/* Note/Curve Editor Area */}
            <Flex className="flex-1 overflow-hidden relative">
                {/* Left axis + corner */}
                <Flex direction="column" className="shrink-0">
                    <Box
                        className="bg-qt-window border-b border-qt-border relative"
                        style={{
                            width: AXIS_W,
                            height: timeRulerHeightPx(
                                Boolean(
                                    s.tempoMap && s.tempoMap.points.length > 0 && s.tempoMapVisible,
                                ),
                            ),
                        }}
                    >
                        {/* 速度映射小按钮（右下角）：显示/创建 或 清空/隐藏。 */}
                        <TempoMapCornerButton />
                    </Box>
                    <div
                        ref={axisWrapRef}
                        className="bg-qt-window border-r border-qt-border relative"
                        // 支持切换展示单位的参数（音量 / 动态）整列可点：光标给
                        // pointer 作为"这里可点"的提示，否则该交互完全不可发现。
                        style={{
                            width: AXIS_W,
                            flex: 1,
                            cursor: axisUnitToggleAvailable ? "pointer" : undefined,
                        }}
                    >
                        {/* 键盘轴 GL 层（阶段 2/3）：铺在 Canvas2D 轴画布**下面**
                            （DOM 顺序在前、无 z-index），画键盘几何与音名标签。
                            下方那块 Canvas2D 轴画布仍保留在 DOM 中，但绘制已被
                            `skipAxisCanvas: true` **整张跳过（含清屏）**——轴上一切都
                            归 GL。宿主自己按 `axisWidthPx` 给 GL 轴画布定尺寸，不依赖
                            这块画布，故不留它也不影响 GL；保留它只是为了不牵动
                            `drawPianoRoll` 的入参契约（见下方 `skip*` 说明）。 */}
                        <canvas
                            ref={glAxisCanvasRef}
                            className="absolute inset-0 pointer-events-none"
                            aria-hidden
                        />
                        <canvas ref={axisCanvasRef} className="absolute inset-0" />

                        {/* 纵轴展示单位角标（仅音量 / 动态）：显示当前读数单位，同时
                            是"点击可切换"的可见提示（整列光标为 pointer）。
                            刻意**不**挂 onClick —— 点击会冒泡到上面的轴列处理器，
                            再挂一个会切换两次（净效果为"点了没反应"）。 */}
                        {axisUnitToggleAvailable ? (
                            <div
                                className="absolute top-0 right-0 z-10 px-1 text-[9px] leading-[14px] text-qt-text-muted"
                                aria-hidden
                            >
                                {editParamAxisUnit === "db"
                                    ? tAny("param_axis_unit_db")
                                    : tAny("param_axis_unit_ratio")}
                            </div>
                        ) : null}

                        {/* 纵轴浮动读数（`弹出展示参数` 的轴列形态）：挂在轴列坐标系里，
                            底边对齐光标向上展开——与曲线浮窗同一观感。 */}
                        {s.showParamValuePopup && axisValuePreview
                            ? (() => {
                                  const rect = axisWrapRef.current?.getBoundingClientRect();
                                  if (!rect) return null;
                                  return (
                                      <div
                                          className="absolute z-20 pointer-events-none bg-qt-panel border border-qt-border rounded px-2 py-1 text-[11px] leading-none text-qt-text"
                                          style={{
                                              left: axisValuePreview.clientX - rect.left,
                                              top: axisValuePreview.clientY - rect.top,
                                              transform: "translate(0, -100%)",
                                              whiteSpace: "nowrap",
                                          }}
                                      >
                                          {axisValuePreview.text}
                                      </div>
                                  );
                              })()
                            : null}
                    </div>
                </Flex>

                {/* Right: ruler + scrollable canvas */}
                <Flex direction="column" className="flex-1 min-w-0 select-none">
                    <TimeRuler
                        scrollLeft={scrollLeft}
                        ticks={timelineTicks}
                        pxPerSec={pxPerSec}
                        viewportWidth={viewSize.w}
                        playheadSec={s.playheadSec}
                        positionPlayheadFromProps={false}
                        playheadLineRef={attachRulerPlayheadLine}
                        playheadHeadRef={attachRulerPlayheadHead}
                        contentRef={rulerContentRef}
                        timeContext={timeContext}
                        primaryUnit={s.primaryTimeUnit}
                        secondaryUnit={s.secondaryTimeUnit}
                        onPrimaryUnitChange={handlePrimaryUnitChange}
                        onSecondaryUnitChange={handleSecondaryUnitChange}
                        onOpenSettings={() => setTimeDisplaySettingsOpen(true)}
                        onCopyPlayheadTime={() => void handleCopyPlayheadTime()}
                        t={t as (key: string) => string}
                        tempoMap={s.tempoMap}
                        tempoMapVisible={s.tempoMapVisible}
                        projectSec={dynamicProjectSec}
                        grid={s.grid}
                        snapEnabled={s.snapEnabled}
                        timelineSnap={s.timelineSnap}
                        projectScale={effectiveProjectScale}
                        projectScaleName={
                            s.project.useCustomScale
                                ? (s.project.customScale?.name ?? undefined)
                                : undefined
                        }
                        fallbackDenominator={s.project.timeSignatureDenominator}
                        customScalePresets={s.customScalePresets}
                        onTempoMapChange={handleTempoMapChange}
                        onTempoMapCommit={handleTempoMapCommit}
                        onMouseDown={(e) => {
                            interactions.onRulerMouseDown(e);
                        }}
                    />

                    {/* 自绘滚动条的定位容器：滚动条必须是**滚动容器之外**
                        的兄弟节点。放在滚动容器内部会被内容一起滚走（绝对定位在滚动
                        容器里仍随内容平移），这是自绘滚动条最经典的错位根因。 */}
                    <div className="flex-1 min-w-0 relative">
                        <div
                            ref={scrollerRef}
                            // 隐藏原生滚动条（自绘条取代），但**保留 `overflow: scroll`**
                            // ——原生 scroller 是被动镜像，必须保留滚动范围才能接受宿主
                            // 每帧的程序化回写，也让尚未迁移的输入代码（中键平移等）
                            // 继续可读可写。`.custom-scrollbar` 不再需要：原生条恒不可见。
                            //
                            // 底部预留 8px（bottom-2）：给自绘水平滚动条独占一行。滚动
                            // 条若叠加在内容上，会挡住贴底的参数线；让 scroller 在水平
                            // 条上方收边，二者互不重叠（竖直条同步缩短，见下方轨道）。
                            className="absolute left-0 right-0 top-0 bg-qt-graph-bg overflow-x-scroll overflow-y-scroll hide-scrollbar outline-none focus:outline-none focus-visible:outline-none"
                            // 底部预留一行给自绘水平滚动条（见 PARAM_EDITOR_BOTTOM_BAR_PX）
                            style={{ bottom: PARAM_EDITOR_BOTTOM_BAR_PX }}
                            data-piano-roll-scroller
                            tabIndex={0}
                            onAuxClick={interactions.onScrollerAuxClick}
                            onScroll={onScrollerScroll}
                            onContextMenu={interactions.onScrollerContextMenu}
                            onKeyDown={interactions.onScrollerKeyDown}
                        >
                            {/* Sticky viewport overlay: grid + canvas do not physically scroll */}
                            <div
                                className="sticky left-0 top-0 h-full"
                                style={{ width: viewSize.w, overflow: "hidden", zIndex: 1 }}
                            >
                                <div className="relative h-full" style={{ width: viewSize.w }}>
                                    <BackgroundGrid
                                        contentWidth={contentWidth}
                                        contentHeight={viewSize.h}
                                        viewportWidth={viewSize.w}
                                        scrollLeft={scrollLeft}
                                        pxPerBeat={pxPerBeat}
                                        grid={s.grid}
                                        beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                                        visible={s.timelineSnap.gridVisible}
                                        minSpacingPx={s.timelineSnap.gridMinSpacingPx}
                                        swingPercent={
                                            s.timelineSnap.swingEnabled
                                                ? s.timelineSnap.swingPercent
                                                : 0
                                        }
                                        layerRef={gridLayerRef}
                                        ticks={timelineTicks}
                                        // 【必须提供视口总线：否则网格会画在滞后的偏移上】
                                        // 不传总线时网格的绘制偏移取自 `scrollLeft` prop，而它是
                                        // **量化提交**的 React state（256px 死区，见
                                        // `gridDrawViewport.ts`）：参数编辑器自己的滚动（滚轮 /
                                        // 拖 thumb / 触摸）只改内核与镜像，**不**逐帧提交 state
                                        // ——最后一次不足 256px 的位移永远不会提交。此后任何一次
                                        // React 重绘（例如时间轴缩放带来的 `pxPerBeat` 变化）都会
                                        // 用这个滞后值画网格，网格就停在错误偏移上，直到下一次滚动。
                                        // 实测（Chrome，先小幅滚动参数编辑器再滚轮缩放时间轴）：
                                        // 网格与自身标尺相差 **59px**（基线只有 8px 的标签内缩）。
                                        // 传总线后偏移一律取总线快照（内核真值），与标尺 / 画布 /
                                        // 波形 / 曲线取同一份视口。
                                        //
                                        // 不传 `layerOrder`：参数编辑器没有统一帧提交器，注册会被
                                        // 跳过；网格仍由 `gridRedrawBridge` 的命令式路径重绘（宿主
                                        // 每帧调用），这里只需要「偏移取总线」这一条契约。
                                        viewportBus={pianoRollViewportBus}
                                        sticky
                                    />

                                    <PianoRollWaveformSurface
                                        clips={clipPeaks}
                                        widthPx={viewSize.w}
                                        heightPx={viewSize.h}
                                        scrollLeftPx={scrollLeft}
                                        pxPerSec={pxPerSec}
                                        colors={waveformColors}
                                        amplitudeMap={pianoRollAmplitudeMap}
                                    />

                                    {/* GL 静态层（阶段 2）：网格等静态图层。
                                        层级说明：它是**最底层**——DOM 顺序在 Canvas2D
                                        主画布之前，且不设 z-index（Canvas2D 主画布用
                                        `absolute inset-0` 覆盖其上）。因此 GL 层只画
                                        静态底图，曲线 / 选区 / 播放头由 Canvas2D 与 GL
                                        叠加层画在上层。
                                        恒挂载：GL 是网格的**唯一**绘制者（Canvas2D
                                        侧的整段音高网格分支已被删除，不是"跳过"而是
                                        不存在了），不挂就等于网格消失。 */}
                                    <canvas
                                        ref={glCanvasRef}
                                        data-piano-roll-gl-scene
                                        className="absolute inset-0 pointer-events-none"
                                        aria-hidden
                                    />

                                    {/* `data-piano-roll-canvas`：主曲线画布的稳定选择器。
                                        浏览器自动化验证（scripts/dev-shot.mjs）需要按元素
                                        截图逐像素比对曲线，而 canvas 本身没有可锚定的属性；
                                        页面里有多个同尺寸 canvas，按尺寸猜会命中错误的那一个。 */}
                                    <canvas
                                        ref={canvasRef}
                                        data-piano-roll-canvas
                                        className="absolute inset-0"
                                        style={{
                                            cursor: canvasCursor,
                                            // 阻止 WebView 把笔/触摸手势截走做原生
                                            // 滚动（会产生 pointercancel 打断笔画）。
                                            // scroller 自身不加：触摸屏单指滚动依赖原生。
                                            touchAction: "none",
                                        }}
                                        onPointerMove={interactions.onCanvasPointerMove}
                                        onPointerLeave={interactions.onCanvasPointerLeave}
                                        onPointerDown={interactions.onCanvasPointerDown}
                                    />

                                    {/* 动态叠加层（阶段 2/3）：播放头与选区。
                                        层序：DOM 顺序在曲线画布**之后** => 覆盖其上。
                                        指针事件全部穿透（interactions 挂在曲线画布上），
                                        否则会挡住参数编辑的命中测试。
                                        恒挂载：播放头归 GL 叠加层（Canvas2D 侧
                                        `skipPlayhead: true`）。 */}
                                    <canvas
                                        ref={glOverlayCanvasRef}
                                        className="absolute inset-0 pointer-events-none"
                                        aria-hidden
                                    />
                                    {s.showParamValuePopup &&
                                        paramValuePreview &&
                                        (() => {
                                            const rect = canvasRef.current?.getBoundingClientRect();
                                            if (!rect) return null;
                                            return (
                                                <div
                                                    className="absolute z-20 pointer-events-none bg-qt-panel border border-qt-border rounded px-2 py-1 text-[11px] leading-none text-qt-text"
                                                    style={{
                                                        left: paramValuePreview.clientX - rect.left,
                                                        top: paramValuePreview.clientY - rect.top,
                                                        transform: "translate(0, -100%)",
                                                        whiteSpace: "nowrap",
                                                    }}
                                                >
                                                    {paramValuePreview.displayText ??
                                                        formatParamValuePreview(
                                                            paramValuePreview.value,
                                                        )}
                                                </div>
                                            );
                                        })()}
                                </div>
                            </div>

                            {/* Spacer：提供横向内容宽度与竖向滚动范围，实际绘制仍固定在 sticky 视口层。 */}
                            <div
                                className="relative"
                                style={{
                                    width: paddedContentWidth,
                                    height: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
                                    pointerEvents: "none",
                                }}
                                aria-hidden
                            />
                        </div>

                        {/* 自绘滚动条：几何由宿主每帧写入。
                            样式对齐原生滚动条（`.custom-scrollbar`，本面板已不再使用）：
                            - thumb 取 `--qt-scrollbar-thumb`（浅色主题下才看得见），
                              而不是固定半透明黑；
                            - 轨道**透明**，加底色会多出一条灰带；
                            - 8px 厚 + 胶囊圆角，对应 macOS 的 overlay thin 滚动条。
                            - 水平条**独占一行**：scroller 已在 bottom-2 收边（见上），
                              轨道落在预留行内，不再叠加内容；竖直条同步在 bottom-2
                              收边，右下角让位给水平条（与原生滚动条的角落行为一致）。
                            - 外层即**轨道**：承接「点空白翻页」。宿主的 thumb 处理器
                              会 `stopPropagation`，因此到达轨道的按下必然不在 thumb 上。
                            恒挂载：原生滚动条已被 `.hide-scrollbar` 隐藏，自绘条是
                            用户可见的**唯一**滚动条。 */}
                        <div
                            ref={vScrollbarTrackRef}
                            className="absolute right-0 top-0 w-2 z-20"
                            style={{ bottom: PARAM_EDITOR_BOTTOM_BAR_PX }}
                        >
                            <div
                                ref={vScrollbarThumbRef}
                                className="absolute left-0 w-full rounded-full bg-[var(--qt-scrollbar-thumb)]"
                            />
                        </div>
                        <div
                            ref={hScrollbarTrackRef}
                            className="absolute bottom-0 left-0 right-0 z-20"
                            style={{ height: PARAM_EDITOR_BOTTOM_BAR_PX }}
                        >
                            <div
                                ref={hScrollbarThumbRef}
                                className="absolute top-0 h-full rounded-full bg-[var(--qt-scrollbar-thumb)]"
                            />
                        </div>
                    </div>
                </Flex>
            </Flex>
            {paramEditorMidiDragOver ? (
                <div className="pointer-events-none absolute left-1/2 top-10 z-40 -translate-x-1/2 rounded border border-qt-snap-source/70 bg-qt-panel/95 px-3 py-1.5 text-[12px] text-qt-text shadow-lg">
                    {tAny("param_editor_drop_midi_hint")}
                </div>
            ) : null}
            <MidiTrackSelectDialog
                open={midiDialogOpen}
                onOpenChange={setMidiDialogOpen}
                midiPath={midiPath}
                importTarget={
                    midiDialogSourceRef.current === "reaperClipboard"
                        ? importTargetReaperClipboard
                        : importTargetParamEditor
                }
                onImportTargetChange={handleImportTargetChange}
                rootTrackComposeEnabled={rootTrack?.composeEnabled ?? true}
                onRequestEnableCompose={handleRequestEnableCompose}
                clipboardGuid={clipboardGuid}
                selectionRanges={midiSelRanges}
                onImported={handleMidiImported}
                onImportAsClip={handleImportAsClip}
                importPosition={importPosition}
                onImportPositionChange={handleImportPositionChange}
                selectionAvailable={midiSelectionAvailable}
                fillGaps={fillGaps}
                onFillGapsChange={handleFillGapsChange}
                projectBpm={s.bpm}
                importBpmAsProject={importBpmAsProject}
                onImportBpmAsProjectChange={handleImportBpmAsProjectChange}
                noteBpmMode={noteBpmMode}
                onNoteBpmModeChange={handleNoteBpmModeChange}
                specifiedBpm={specifiedBpm}
                onSpecifiedBpmChange={handleSpecifiedBpmChange}
                multiTrackMerge={multiTrackMerge}
                onMultiTrackMergeChange={handleMultiTrackMergeChange}
                closeLeadingGap={closeLeadingGap}
                onCloseLeadingGapChange={handleCloseLeadingGapChange}
                importTempoMapEnabled={importTempoMapEnabled}
                onImportTempoMapEnabledChange={handleImportTempoMapEnabledChange}
                importTempoMapTempo={importTempoMapTempo}
                onImportTempoMapTempoChange={handleImportTempoMapTempoChange}
                importTempoMapTimeSignature={importTempoMapTimeSignature}
                onImportTempoMapTimeSignatureChange={handleImportTempoMapTimeSignatureChange}
                importTempoMapKeySignature={importTempoMapKeySignature}
                onImportTempoMapKeySignatureChange={handleImportTempoMapKeySignatureChange}
            />
            {ctxMenu && s.toolMode === "select" && (
                <EditContextMenu
                    x={ctxMenu.x}
                    y={ctxMenu.y}
                    isPitchParam={editParam === "pitch"}
                    onClose={() => setCtxMenu(null)}
                    onCopy={() => void handleEditOp("copy")}
                    onCut={() => void handleEditOp("cut")}
                    onPaste={() => void handleEditOp("paste")}
                    onSelectAll={() => void handleEditOp("selectAll")}
                    onDeselect={() => void handleEditOp("deselect")}
                    onInitialize={() => void handleEditOp("initialize")}
                    onTransposeCents={() => openEditDialog("transposeCents")}
                    onTransposeDegrees={() => openEditDialog("transposeDegrees")}
                    onSetPitch={() => openEditDialog("setPitch")}
                    onAverage={() => openEditDialog("average")}
                    onSmooth={() => openEditDialog("smooth")}
                    onAddVibrato={() => openEditDialog("addVibrato")}
                    onQuantize={() => openEditDialog("quantize")}
                    onMeanQuantize={() => openEditDialog("meanQuantize")}
                    onSaveAsPitchRef={() => void handleSaveAsPitchRef()}
                    onExportMidi={() => void handleExportMidiFromEditor()}
                    // 音量 ↔ 动态 互转：参数本身就是这两个之一时始终可用。
                    //
                    // 【为什么不按"有无选区"门控】这里曾经只在有选区时显示（理由是
                    // "无选区时整条互换应由初始化表达"）。现在菜单里所有以选区为作用域
                    // 的操作都遵循同一条规则：**无选区时先隐式全选再执行**（见
                    // `handleEditOp` 开头），因此"整条互换"就是一次明确的全选 + 互换，
                    // 不再需要靠隐藏菜单项来回避。
                    isVolumeParam={
                        editParam === "volume" && planParamConversion(editParam) !== null
                    }
                    isDynParam={editParam === "dyn" && planParamConversion(editParam) !== null}
                    onConvertVolumeToDyn={() => void handleEditOp("convertVolumeToDyn")}
                    onConvertDynToVolume={() => void handleEditOp("convertDynToVolume")}
                />
            )}
        </Flex>
    );
};
