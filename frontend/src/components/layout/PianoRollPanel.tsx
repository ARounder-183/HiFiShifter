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
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeline";
import type { TempoMap } from "../../utils/tempoMap";
import { buildScaleSegments, effectiveScaleAtSec } from "../../utils/tempoMap";
import { setTempoMapRemote } from "../../features/session/thunks/tempoMapThunks";
import { publishPianoRollSelection } from "../../utils/pianoRollSelectionBus";
import { resolveHorizontalWheelZoom } from "./timeline/runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "./timeline/runtime/timelineZoomBounds";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";

import { AXIS_W, PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "./pianoRoll/constants";
import { drawPianoRoll } from "./pianoRoll/render";
import type { DetectedPitchCurve, ReferencePitchOverlay } from "./pianoRoll/render";
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
    applySelectionEditWithEdgeSmoothing,
    type SelectionEditExtension,
} from "./pianoRoll/selectionEditApply";
import { editablePitchValue } from "./pianoRoll/paramSmoothing";
import { usePianoRollData } from "./pianoRoll/usePianoRollData";
import { useClipsPeaksForPianoRoll } from "./pianoRoll/useClipsPeaksForPianoRoll";
import { PianoRollWaveformSurface } from "./pianoRoll/PianoRollWaveformSurface";
import { pianoRollViewportBus } from "./pianoRoll/pianoRollViewportBus";
import { buildTimelineTicks } from "./timeline/runtime/buildTimelineTicks.js";
import {
    createTimelineAxis,
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
import { readDevicePixelRatio, snapToDevicePx } from "../../utils/devicePixelLine";
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
    selectKeybinding,
    selectMergedKeybindings,
} from "../../features/keybindings/keybindingsSlice";

import { usePianoRollStatusUpdate } from "../../contexts/PianoRollStatusContext";
import { MidiTrackSelectDialog } from "./MidiTrackSelectDialog";
import { settingsApi } from "../../services/api/settings";
import { EditContextMenu } from "../editDialogs/EditContextMenu";
import { getDynamicProjectSec } from "../../features/session/projectBoundary";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { parseCustomScaleToken } from "../../utils/scaleSelection";
import {
    centerFromVerticalScrollTop,
    verticalScrollTopFromCenter,
} from "./pianoRoll/verticalScrollMapping";
import {
    isPianoRollCurveGlEnabled,
    isPianoRollGlSceneEnabled,
    isPianoRollKernelEnabled,
} from "./timeline/kernel/featureFlag";
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
} from "./pianoRoll/kernel/host/pianoRollKernelData";
import { PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX } from "./pianoRoll/kernel/scroll/verticalValueScroll";
import { normalizeCssColor, resolvePianoRollColors } from "./pianoRoll/colors";
import { parseRgbaColor } from "./timeline/runtime/timelineClipGlRenderer";

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX = PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX;

/**
 * 是否启用参数编辑器渲染内核（阶段 1：滚动 / 视口所有权）。
 *
 * 在**模块加载时**读取一次（与时间轴内核同一约定）：切换开关后刷新页面生效。
 * 未显式设置时关闭（见 `isPianoRollKernelEnabled`），因此默认行为与迁移前完全一致。
 */
const PARAM_EDITOR_KERNEL_ENABLED = isPianoRollKernelEnabled();

/**
 * 是否启用参数编辑器 GL 场景层（阶段 2）。
 *
 * 特殊说明：必须与内核开关**同时**成立才有效——GL 层依赖内核提供的视口真值
 * （`u_viewOrigin` 由内核的滚动位置驱动）。两者都关 / 只开内核时，绘制完全走
 * Phase 1 已验证的 Canvas2D 路径。
 */
const PARAM_EDITOR_GL_SCENE_ENABLED = PARAM_EDITOR_KERNEL_ENABLED && isPianoRollGlSceneEnabled();

/**
 * 是否启用参数编辑器曲线 GL 层（阶段 3）。
 *
 * 特殊说明：它**嵌套在** GL 场景层之内——曲线层复用主 GL 画布的上下文，没有 GL
 * 场景层就没有曲线层。因此两个开关必须同时成立。
 *
 * 实测依据（Phase 3 计划 R7）：最小缩放下 3 分钟曲线约 36000 个可见点，Canvas2D
 * 描边需 71.7ms/帧（≈14fps），GL 全路径 4.6ms（15.6×）。
 */
const PARAM_EDITOR_CURVE_GL_ENABLED =
    PARAM_EDITOR_GL_SCENE_ENABLED && isPianoRollCurveGlEnabled();

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

/**
 * 参数编辑器工具栏的参数显示顺序排名（数值越小越靠左）。
 * - 「音高」为核心参数，固定在最左侧（在 JSX 中单独渲染，不在此排序）；
 * - 「音量/声像」是所有算法的共通参数，固定在最右侧；
 * - 中间参数随算法不同而变化。
 */
function getParamToolbarRank(paramId: string, algo: string | undefined | null): number {
    switch (algo) {
        case "nsf_hifigan_onnx":
            // 音高、共振峰、气声音量、张力、音量、声像
            switch (paramId) {
                case "formant_shift_cents":
                    return 10;
                case "breath_gain":
                    return 20;
                case "hifigan_tension":
                    return 30;
                case "volume":
                    return 90;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
        case "vslib":
            // 音高、共振峰、气声强度、音量、声像
            switch (paramId) {
                case "formant_shift_cents":
                    return 10;
                case "breathiness":
                    return 20;
                case "volume":
                    return 90;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
        default:
            // world / 其它：仅保证音量/声像在右侧，其余保持后端顺序
            switch (paramId) {
                case "volume":
                    return 90;
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

type FormantParamButtonProps = {
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
};

const FormantParamButton: React.FC<FormantParamButtonProps> = ({
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
}) => {
    const eyeMode: "main" | "on" | "off" =
        rootActive || childActive ? "main" : secondaryVisible ? "on" : "off";

    if (!childParamId) {
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
                    <DropdownMenu.RadioItem value={childParamId}>
                        {childMenuLabel ?? childLabel}
                    </DropdownMenu.RadioItem>
                </DropdownMenu.RadioGroup>
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
};

export const PianoRollPanel: React.FC = () => {
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
     * 【为什么声明在这里】`invalidate`（紧随其后）在内核模式下要把标脏转交宿主，
     * 因此必须在它之前声明。放在下方原来的位置会让 `invalidate` 的闭包引用一个
     * 尚未初始化的 `const`（TDZ）——首次调用即抛 ReferenceError。
     */
    const hostRef = useRef<PianoRollKernelHost | null>(null);
    const drawRef = useRef<() => void>(() => {});
    /**
     * 请求重绘（所有会改变画面的数据/状态变更都经此入口）。
     *
     * 【内核模式下为什么要转交宿主】曲线层搬到 GL 后，画面由**两个**渲染循环驱动：
     * 面板自己的 rAF（Canvas2D）与内核宿主的 rAF（GL 层）。只调度前者会让 GL 层
     * 永远停留在旧内容上——典型症状是"曲线数据到了但屏幕上不出现，直到滚动一下
     * 才显示"。因此内核模式下标脏一律交给宿主：宿主的帧提交里会回调 `onFrame`
     * → `applyScrollLayers` → `drawRef.current()`，Canvas2D 与 GL 同帧一起刷新。
     *
     * 特殊说明：`drawRef.current()` 自身不调用 `invalidate()`，所以这条链不会自激。
     * 宿主尚未创建（挂载前 / 卸载后）时退回面板自己的 rAF，避免丢失首次绘制。
     */
    const invalidate = useCallback(() => {
        const host = hostRef.current;
        if (PARAM_EDITOR_KERNEL_ENABLED && host != null) {
            host.invalidate();
            return;
        }
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            drawRef.current();
        });
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
    const stretchKb = useAppSelector((state) => selectKeybinding(state, "modifier.clipStretch"));
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
    // 记录打开弹窗时的选区（拍数），用于后续计算帧偏移
    const [midiDialogSelection, setMidiDialogSelection] = useState<{
        aBeat: number;
        bBeat: number;
    } | null>(null);

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
            if (Math.abs(value) >= 100) return value.toFixed(1);
            if (Math.abs(value) >= 10) return value.toFixed(2);
            return value.toFixed(3);
        },
        [editParam],
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
            setMidiDialogSelection(sel ? { ...sel } : null);
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

    const dynamicProjectSec = useMemo(() => getDynamicProjectSec(s.clips), [s.clips]);
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
    // 同步开关的 ref 镜像：rAF 原子提交与每帧对账循环读取最新值，避免陈旧闭包。
    const paramEditorSyncTimelineRef = useRef(s.paramEditorSyncTimeline);
    // 渲染期立即同步 ref，确保同步视口在 layout effect 落地时，
    // Canvas 读取到的是与标尺/网格同一帧的新缩放与滚动值。
    // 仅在值变化时回写 ref：渲染期的 state→ref 同步必须与被提交的状态同帧存在，
    // 无条件覆盖会抹掉 rAF 原子提交中已落地的值（refs 只能由 render/提交写入）。
    if (scrollLeftRef.current !== scrollLeft) scrollLeftRef.current = scrollLeft;
    if (pxPerBeatRef.current !== pxPerBeat) pxPerBeatRef.current = pxPerBeat;
    if (pxPerSecRef.current !== pxPerSec) pxPerSecRef.current = pxPerSec;
    paramEditorSyncTimelineRef.current = s.paramEditorSyncTimeline;
    const timelineSyncApplyingRef = useRef(false);
    const timelineOffsetRef = useRef(0);
    const [timelineOffsetPx, setTimelineOffsetPx] = useState(0);
    const pendingParamSyncViewportRef = useRef<{
        nativeScrollLeft: number;
        pxPerSec: number;
    } | null>(null);
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
    useLayoutEffect(() => {
        const update = () => {
            const next = measureTimelineViewportOffsetPx();
            timelineOffsetRef.current = next;
            setTimelineOffsetPx((prev) => (Math.abs(prev - next) < 0.5 ? prev : next));
        };
        update();
        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(update);
            const scroller = scrollerRef.current;
            if (scroller) observer.observe(scroller);
            const track = document.querySelector<HTMLElement>("[data-timeline-scroller]");
            if (track) observer.observe(track);
            return () => observer.disconnect();
        }
        window.addEventListener("resize", update);
        return () => window.removeEventListener("resize", update);
    }, []);

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
        const scroller = scrollerRef.current;
        if (scroller) {
            // 内核模式：先按绘制坐标把位置交给内核（它负责换算原生坐标并镜像回写），
            // 再走采纳路径同步 state。旧实现则直接写原生 scroller 后由 syncScrollLeft 采纳。
            if (PARAM_EDITOR_KERNEL_ENABLED) {
                applyHorizontalScrollPosition(newScrollLeft);
                scrollLeftRef.current = newScrollLeft;
                lastScrollLeftRef.current = newScrollLeft;
                setScrollLeft(newScrollLeft);
                return;
            }
            scroller.scrollLeft = newScrollLeft;
            syncScrollLeft(scroller);
            return;
        }
        scrollLeftRef.current = newScrollLeft;
        setScrollLeft(newScrollLeft);
        // eslint-disable-next-line react-hooks/exhaustive-deps
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
            const store = timelineViewportSync.get();
            const offset = timelineOffsetRef.current;
            const drawingScrollLeft = timelineViewportNativeToState(store.scrollLeft, offset);
            // 纯滚动（pxPerSec 未变）：在同一个事件帧内同步落地——原生
            // scroller、标尺/网格层（applyScrollLayers）与轨道视图同帧提交，
            // 两个面板严丝合缝。state 仅作事后对齐（React 在绘制前提交）。
            const scroller = scrollerRef.current;
            if (scroller && Math.abs(store.pxPerSec - pxPerSecRef.current) <= 1e-9) {
                timelineSyncApplyingRef.current = true;
                pxPerSecRef.current = store.pxPerSec;
                scrollLeftRef.current = drawingScrollLeft;
                lastScrollLeftRef.current = drawingScrollLeft;
                // 同步落地走统一载体（内核模式下即宿主；它会换算原生坐标并镜像回写）。
                applyHorizontalScrollPosition(drawingScrollLeft);
                applyScrollLayers(drawingScrollLeft);
                setScrollLeft(drawingScrollLeft);
                timelineSyncApplyingRef.current = false;
                return;
            }
            // 缩放（pxPerSec 变化）：内容宽度必须先按新 pxPerSec 重排，维持
            // “先提交 state，再由 layout effect 落地”的既有路径。
            timelineSyncApplyingRef.current = true;
            pendingParamSyncViewportRef.current = {
                nativeScrollLeft: store.scrollLeft,
                pxPerSec: store.pxPerSec,
            };
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
                applyHorizontalScrollPosition(next);
                scrollLeftRef.current = next;
                lastScrollLeftRef.current = next;
                setScrollLeft(next);
                applyScrollLayers(next);
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
        pendingParamSyncViewportRef.current = {
            nativeScrollLeft: store.scrollLeft,
            pxPerSec: store.pxPerSec,
        };
        setScrollLeft(drawingScrollLeft);
        timelineSyncApplyingRef.current = false;
    }, [timelineOffsetPx, s.paramEditorSyncTimeline]);

    // 同步视口必须等内容宽度按新 pxPerSec 更新后再落到 DOM。
    // 否则设置 scroller.scrollLeft 时会被浏览器钳回旧的最大滚动位置，
    // 形成“缩放已变、滚动没变”的水平漂移。
    useLayoutEffect(() => {
        const pending = pendingParamSyncViewportRef.current;
        if (!pending || !s.paramEditorSyncTimeline) return;
        if (Math.abs(pxPerSec - pending.pxPerSec) > 1e-9) return;
        if (Math.abs(timelineOffsetPx - timelineOffsetRef.current) > 0.5) return;

        const offset = timelineOffsetRef.current;
        const drawingScrollLeft = timelineViewportNativeToState(pending.nativeScrollLeft, offset);
        if (Math.abs(scrollLeft - drawingScrollLeft) > 0.5) return;

        pendingParamSyncViewportRef.current = null;
        const scroller = scrollerRef.current;
        if (!scroller) return;

        timelineSyncApplyingRef.current = true;
        pxPerSecRef.current = pending.pxPerSec;
        pxPerBeatRef.current = pending.pxPerSec * (60 / Math.max(1e-6, s.bpm));
        scrollLeftRef.current = drawingScrollLeft;
        // `pending.nativeScrollLeft` 是共享视口的原生值；换算成绘制坐标后走统一载体
        // （内核模式会再加回偏移，旧实现直接写原生，两者落到同一位置）。
        applyHorizontalScrollPosition(drawingScrollLeft);
        // 【内核模式下不能再调 syncScrollLeft】它是「读原生 scroller → 采纳为真值」的
        // 路径，而此时原生镜像还停留在**上一帧的旧值**（本函数的写入要等内核下一帧才
        // 回写），于是会把刚提交的目标位置又覆盖回旧值——表现为「同步从 1200 拨回 0
        // 时参数编辑器不动」。旧实现里原生就是事实源，读回来即是刚写的值，故无此问题。
        if (!PARAM_EDITOR_KERNEL_ENABLED) {
            syncScrollLeft(scroller);
        }
        applyScrollLayers(drawingScrollLeft);
        timelineSyncApplyingRef.current = false;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pxPerSec, scrollLeft, s.paramEditorSyncTimeline, timelineOffsetPx]);

    useLayoutEffect(() => {
        const pending = horizontalZoomPendingRef.current;
        if (!pending) return;
        if (Math.abs(pending.nextScale - pxPerSec) > 1e-9) return;
        horizontalZoomPendingRef.current = null;
        horizontalZoomChainRef.current = null;
        const scroller = scrollerRef.current;
        if (!scroller) return;

        const syncEnabled = s.paramEditorSyncTimeline;
        const offset = syncEnabled ? timelineOffsetRef.current : 0;
        const native = pending.nextScrollLeft;
        const next = timelineViewportNativeToState(native, offset);
        applyHorizontalScrollPosition(next);
        if (lastScrollLeftRef.current !== next) {
            lastScrollLeftRef.current = next;
            scrollLeftRef.current = next;
        }
        // 同步模式下手动缩放后必须把新的 pxPerSec 写回共享视口；即使滚动位置
        // 没有变化（例如光标位于左侧同步空白区时锚定在工程起点，next 仍为 -offset），
        // 也要广播缩放，否则轨道视图不会跟着缩放。
        if (syncEnabled && !timelineSyncApplyingRef.current) {
            timelineViewportSync.setViewport({
                scrollLeft: native,
                pxPerSec,
            });
        }
        applyScrollLayers(next);
        // 防止浏览器对原生滚动位置的钳制造成漂移：立即校正到理论值。
        // 内核模式下由宿主的镜像回写负责校正（它每帧都会把原生位置对齐真值）。
        if (PARAM_EDITOR_KERNEL_ENABLED) {
            setScrollLeft(next);
            return;
        }
        const expectedNative = timelineViewportStateToNative(next, offset);
        if (Math.abs(scroller.scrollLeft - expectedNative) > 0.5) {
            scroller.scrollLeft = expectedNative;
        }
        // 同步更新状态：让标尺/网格与画布在同一帧对齐，消除缩放闪屏。
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

    const queueHorizontalZoom = useCallback(
        (nextPxPerSec: number, nextNativeScrollLeft: number) => {
            // 事件内只记录缩放意图，**绝不**在提交前改写 refs/state：任何
            // “ref 先行”都会让夹缝中的 rAF 绘制读到“新缩放 + 旧滚动”的
            // 混合投影——参数线/原始音高线/参考线等所有线条整体抽搐一帧。
            horizontalZoomPendingRef.current = {
                nextScale: nextPxPerSec,
                nextScrollLeft: nextNativeScrollLeft,
            };
            // 与 TimelineScrollArea 相同的“一帧一次原子提交”：rAF 合并同帧
            // 内的连续缩放事件；flushSync 把两个 state 在同一批提交中落地，
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
            queueHorizontalZoom(nextPxPerSec, nativeNextScrollLeft);
        },
        [s.paramEditorSyncTimeline, queueHorizontalZoom],
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
            invalidate(); // 绕过 React 渲染，直接命令 Canvas 重绘
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [invalidate],
    );

    const paramViewsRef = useRef<Record<string, ValueViewport>>({});
    const setParamViewport = useCallback(
        (param: string, next: ValueViewport) => {
            paramViewsRef.current = { ...paramViewsRef.current, [param]: next };
            syncVerticalScrollbarForViewport(param as ParamName, next);
            invalidate(); // 绕过 React 渲染，直接命令 Canvas 重绘
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
        if (editParam === "volume" || editParam === "dyn_edit") {
            return 1;
        }
        return 0;
    }, [editParam]);

    const currentParamQuantizeUnit = useMemo(() => {
        if (isChildPitchOffsetCentsParam(editParam)) return 100;
        if (isChildPitchOffsetDegreesParam(editParam)) return 0.5;
        if (isChildFormantOffsetCentsParam(editParam)) return 50;
        if (editParam === "volume" || editParam === "dyn_edit") return 0.05;
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

    const secPerBeat = 60 / Math.max(1e-6, s.bpm);
    const contentWidth = Math.max(1, Math.ceil(dynamicProjectSec * pxPerSec));

    const scrollerRef = useRef<HTMLDivElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisWrapRef = useRef<HTMLDivElement | null>(null);
    const lastScrollLeftRef = useRef<number | null>(null);
    const scrollStateRafRef = useRef<number | null>(null);

    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const gridLayerRef = useRef<HTMLDivElement | null>(null);

    // ── 渲染内核（阶段 1：滚动 / 视口所有权）────────────────────────────
    //
    // 【开启时所有权如何反转】旧实现把原生 scroller 的 `scrollLeft/scrollTop`
    // 当唯一事实源，各图层靠 `syncScrollLeft` → `applyScrollLayers` 跟随。内核
    // 模式下反过来：宿主持有真值，**原生 scroller 退化为被动镜像**——宿主每帧把
    // 真值写回它，使尚未迁移的输入代码（中键拖拽、框选自动滚动等直接读写
    // `scroller.scrollLeft` 的地方）读到的仍是同一份位置，行为不变。
    //
    /** 内核模式下的数据镜像（每次 render 更新字段，宿主每帧现读）。 */
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
    });
    /** 自绘滚动条的 thumb（仅内核模式挂载）。 */
    const hScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarThumbRef = useRef<HTMLDivElement | null>(null);
    /** 自绘滚动条的**轨道**（承接「点空白翻页」，仅内核模式挂载）。 */
    const hScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    const vScrollbarTrackRef = useRef<HTMLDivElement | null>(null);
    /** GL 静态层画布（阶段 2，仅内核 + GL 开关都开启时挂载）。 */
    const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 键盘轴 GL 画布（阶段 2，Task 4）。独立画布：轴列不随横向滚动移动。 */
    const glAxisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    /** 动态叠加层 GL 画布（阶段 2，Task 6）：播放头与选区，位于曲线层之上。 */
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
    // 想让原生最大滚动位置为“工程宽 + 同步偏移”，spacer 需再加一个视口宽。
    const paddedContentWidth = useMemo(
        () => contentWidth + viewSize.w + (s.paramEditorSyncTimeline ? timelineOffsetPx : 0),
        [contentWidth, viewSize.w, s.paramEditorSyncTimeline, timelineOffsetPx],
    );

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
                const playheadLeftPx = visualPlayheadSec * pxPerSecRef.current;
                // 播放头 DOM 写入统一设备像素吸附（与时间线侧同一函数）：
                // 分数 DPR 下不吸附的落点相位随播放连续变化，1/2 物理像素
                // 交替 —— 即"播放时粗细不一"。
                const dpr = readDevicePixelRatio();
                const snappedPlayheadLeftPx = snapToDevicePx(playheadLeftPx, dpr);
                if (rulerPlayheadLineRef.current) {
                    rulerPlayheadLineRef.current.style.left = `${snappedPlayheadLeftPx}px`;
                }
                if (rulerPlayheadHeadRef.current) {
                    rulerPlayheadHeadRef.current.style.left = `${snappedPlayheadLeftPx}px`;
                }
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
        pianoRollViewportBus.emit(next, pxPerSecRef.current, viewSizeRef.current.w);
        // 播放头 DOM 线并入同帧提交：缩放（pxPerSec 变化）时立即对齐新投影，
        // 避免与画布的播放头错位一帧（与 useVisualPlayhead 的 onFrame 同源）。
        // 设备像素吸附与其余播放头写入点一致（见 onFrame 注释）。
        const playheadLeftPx = snapToDevicePx(
            visualPlayheadSecRef.current * pxPerSecRef.current,
            readDevicePixelRatio(),
        );
        if (rulerPlayheadLineRef.current) {
            rulerPlayheadLineRef.current.style.left = `${playheadLeftPx}px`;
        }
        if (rulerPlayheadHeadRef.current) {
            rulerPlayheadHeadRef.current.style.left = `${playheadLeftPx}px`;
        }
    }

    /**
     * 把「绘制坐标」的水平位置提交到当前滚动载体。
     *
     * 【为什么要单独一个函数】面板里有若干**权威写入**点（时间轴同步、键盘缩放、
     * 缩放事务落地、关闭同步时还原位置）：它们算出目标位置后直接落到滚动载体。
     * 内核模式下载体是宿主（内部换算原生坐标并镜像回写），旧实现下载体是原生
     * scroller（需自行加回偏移）。把分支收在一处，避免每调用点各写一份换算。
     *
     * 特殊说明：滚轮 / 自动滚屏等**增量**路径不走这里——它们仍写原生 scroller，
     * 再由 `syncScrollLeft` 采纳进内核（见该函数注释）。
     *
     * @param drawingScrollLeft 目标水平位置（绘制坐标）。
     */
    function applyHorizontalScrollPosition(drawingScrollLeft: number): void {
        const host = hostRef.current;
        if (PARAM_EDITOR_KERNEL_ENABLED && host) {
            host.setScrollLeft(drawingScrollLeft);
            return;
        }
        const scroller = scrollerRef.current;
        if (!scroller) return;
        const offset = paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0;
        scroller.scrollLeft = timelineViewportStateToNative(drawingScrollLeft, offset);
    }

    /**
     * 采纳一次水平滚动位置变化（原生 scroller → 真值）。
     *
     * 流程：原生坐标 → 绘制坐标 → 同步开关时推送共享视口 → 通知各图层 → 量化提交 state。
     *
     * 【内核模式下的语义转变（重要）】旧实现里本函数是「原生是事实源 → 各层跟随」的
     * 唯一扩散点。内核模式下它变成**采纳点**：尚未迁移的输入代码（滚轮、中键平移、
     * BPM 换算、自动滚屏）仍然直接写 `scroller.scrollLeft`，这里把该值收进内核
     * （`host.setScrollLeft`），由内核完成钳制、镜像回写与帧提交。这样：
     * - 所有既有手势**逐条保持可用**，无需在本任务里重写输入路径（Task 7 才迁移）；
     * - 钳制与渲染真值收敛到内核一处，不会出现两套边界；
     * - 收敛性：内核回写的值与内核当前值相同 → 不产生状态变化 → 不再触发帧，
     *   因此「写原生 → 采纳 → 镜像回写 → 再触发 onScroll」不会形成循环。
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
            // 原生滚动位置 == 共享视口值（轨道坐标），直接推送。
            timelineViewportSync.setViewport({
                scrollLeft: scroller.scrollLeft,
                pxPerSec: pxPerSecRef.current,
            });
        }
        // 内核模式：交给内核（它会按新边界钳制、镜像回写并在下一帧提交各图层）。
        const host = hostRef.current;
        if (PARAM_EDITOR_KERNEL_ENABLED && host) {
            host.setScrollLeft(next);
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
    // - **原生坐标**：`[0, 内容宽 + offset]`——旧实现 `paddedContentWidth` 撑出的域；
    // - **绘制坐标**：`[−offset, 内容宽]`——含负值，各图层（标尺/网格/画布/波形）用。
    //
    // 内核的位置字段恒被钳到 `[0, max]`（无法表示负值），所以内核持有**原生坐标**，
    // 本面板消费的 `axis.scrollLeftPx` 是**绘制坐标**。换算只在宿主边界发生
    // （见宿主 `horizontalOffsetPx`），面板这一侧不再自行换算，避免出现第三份口径。
    useEffect(() => {
        if (!PARAM_EDITOR_KERNEL_ENABLED) return;
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
            glSceneEnabled: PARAM_EDITOR_GL_SCENE_ENABLED,
            // 偏移经 ref 读取（同步开关与布局偏移都在运行时变化，闭包捕获会读到挂载时的旧值）。
            horizontalOffsetPx: () =>
                paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0,
            sync: {
                rulerContent: rulerContentRef.current,
                gridLayer: gridLayerRef.current,
            },
            // 帧提交：宿主已完成滚动条几何与标尺 / 网格的 DOM 写入，这里只做
            // 「画布 + 波形 + 播放头」三项与旧实现同帧的提交。复用同一个
            // `applyScrollLayers`，保证两种模式的绘制路径**逐字一致**（迁移不得
            // 改变视觉，唯一区别是谁来触发它：旧实现是 scroll 事件，内核是 rAF）。
            onFrame: (axis) => {
                const drawing = axis.scrollLeftPx;
                scrollLeftRef.current = drawing;
                applyScrollLayers(drawing);
                // 【为什么必须同步 lastScrollLeftRef】镜像回写会让原生 scroller 触发
                // `scroll` 事件，进而走到 `syncScrollLeft`。把"上一次已知位置"同步成
                // 内核真值后，该事件的 `next` 与之相等 → 立即早退，**不会**把值推回
                // 共享视口（否则镜像回写会被误当成用户滚动，把时间轴也推着走）。
                lastScrollLeftRef.current = drawing;
                // 镜像回写：让尚未迁移的输入代码（中键拖拽 / 框选自动滚动）读到的
                // 原生位置始终等于内核真值。仅在真正不一致时写，避免每帧触发样式重算。
                const native = timelineViewportStateToNative(
                    drawing,
                    paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0,
                );
                if (shouldWriteNumber(container.scrollLeft, native, 0.5)) {
                    container.scrollLeft = native;
                }
                if (shouldWriteNumber(container.scrollTop, axis.scrollTopPx, 0.5)) {
                    container.scrollTop = axis.scrollTopPx;
                }
            },
            onScrollLeftCommit: () => {
                // 量化提交：标尺的刻度范围由 React 按视口计算，不同步就会出现
                // 「滚动后刻度消失」（与旧实现 `syncScrollLeft` 的收尾一致）。
                // 注意 `px` 是**绘制坐标**（宿主对外统一口径）。
                if (scrollStateRafRef.current == null) {
                    scrollStateRafRef.current = requestAnimationFrame(() => {
                        scrollStateRafRef.current = null;
                        setScrollLeft(scrollLeftRef.current);
                    });
                }
            },
            onUserScrollLeft: (drawingScrollLeft) => {
                // 用户手势（拖 thumb / 点轨道翻页）：与旧实现拖原生滚动条同语义——
                // 需要把新位置推给共享视口，否则同步模式下滚动时间轴不会跟随。
                //
                // 【为什么不走 syncScrollLeft】那条路径依赖 `scroll` 事件，而内核的
                // 镜像是同帧程序化写入的，事件到达时无法再区分"用户滚动"与"自身回写"。
                // 由宿主动上报手势来源是唯一可靠判据（见 `onUserScrollLeft` 说明）。
                scrollLeftRef.current = drawingScrollLeft;
                lastScrollLeftRef.current = drawingScrollLeft;
                const offset = paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0;
                if (paramEditorSyncTimelineRef.current && !timelineSyncApplyingRef.current) {
                    timelineViewportSync.setViewport({
                        scrollLeft: timelineViewportStateToNative(drawingScrollLeft, offset),
                        pxPerSec: pxPerSecRef.current,
                    });
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
    }, []);

    useLayoutEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;
        // 内核模式下滚动位置由宿主持有，挂载时不得用原生值反向覆盖内核真值
        // （那会在首帧把位置清零）。内核自己会标脏首帧。
        if (PARAM_EDITOR_KERNEL_ENABLED) return;
        syncScrollLeft(el);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [contentWidth, s.grid, s.beats]);

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
     * 网格线颜色为数值 RGBA → 绑定 `valueToY`。
     *
     * 特殊说明 1：`valueToY` 传的是**面板自己的**那个函数（只绑定 `editParam`），
     * 与 Canvas2D 路径共用同一份投影——这正是两种渲染模式网格不会错位的原因。
     *
     * 特殊说明 2：颜色经 `parseRgbaColor` 解析为 0..1 浮点，因为 GL 上传统一用浮点；
     * 在渲染热路径上反复解析 CSS 字符串是纯浪费，故在镜像更新（低频）时做。
     * 解析失败（拿到 NaN）时退回**不透明黑**——宁可颜色不对也不要上传 NaN，
     * NaN 会让整个实例属性失效、整层消失。
     *
     * @returns 网格输入；GL 关闭或无法解析时返回 null（宿主按"没有网格"处理）。
     */
    function buildGridSpec(): PianoRollGridSpec | null {
        // 显式标注为字面量联合：不加标注时 TS 会把嵌套三元推断成 `string`，
        // 导致返回值无法赋给 `PianoRollGridSpec["kind"]`。
        const kind: PianoRollGridSpec["kind"] | null =
            editParam === "pitch"
                ? "pitch"
                : isChildPitchOffsetCentsParam(editParam)
                  ? "cents"
                  : isChildPitchOffsetDegreesParam(editParam)
                    ? "degrees"
                    : isChildFormantOffsetCentsParam(editParam)
                      ? "formantCents"
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
            // 与传给 drawPianoRoll 的 fontFamily 同一个值（第 3268 行），
            // 保证两种渲染模式的字形完全一致。
            fontFamily,
            tensionLabelRgba: toRgba(colors.tensionLabel),
            tensionLineRgba: toRgba(colors.tensionLine),
        };

        // 键盘轴颜色只在音高参数下提供：非 pitch 时 GL 层据此判定"没有键盘"
        // 并清空几何（否则切到别的参数后键盘会残留在画布上）。
        if (kind !== "pitch") return base;
        return {
            ...base,
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

    useLayoutEffect(() => {
        if (!PARAM_EDITOR_KERNEL_ENABLED) return;
        const bounds = getParamValueBoundsForScrollbar(editParam);
        const view = clampViewport(editParam, getCurrentViewportForScrollbar(editParam));
        kernelDataRef.current.projectSec = dynamicProjectSec;
        kernelDataRef.current.valueDomain = {
            min: bounds.min,
            max: bounds.max,
            span: view.span,
        };
        // GL 场景层的网格输入（阶段 2）。仅在 GL 开关开启时构建：颜色解析会创建
        // DOM 探针（`normalizeCssColor`），未启用时不该付这份成本。
        kernelDataRef.current.grid = PARAM_EDITOR_GL_SCENE_ENABLED ? buildGridSpec() : null;
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
    ]);

    // 渲染期刷新 syncScrollLeft 引用（其函数体随每次渲染重建）。
    const syncScrollLeftRef = useRef(syncScrollLeft);
    syncScrollLeftRef.current = syncScrollLeft;

    // 每帧对账自愈（镜像时间轴侧 reconcile）：原生 scroller 是滚动/缩放的
    // 唯一事实源，sticky 画布层经 refs/总线跟随。任何路径漏发/迟发了这些值
    // （提交被浏览器钳制/量化、异常中断的缩放事务……）都会让画布与 DOM
    // 内容层错位，表现为线条抽搐一帧；这里每帧以原生值对账，发现失步立即
    // 经 syncScrollLeft 重发（refs → bus → 标尺/网格/画布同帧），把残余
    // 错位变成被治愈的一帧。绝大多数帧只是两次数值比较，空闲开销可忽略。
    //
    // 【内核模式下停用】真值源已经反转（内核持有、原生是被动镜像），再由本循环
    // 以原生值反向对账就变成**两个方向的自愈互相打架**：内核刚写镜像，本循环又
    // 把镜像读回来当作"权威"重发一次。内核自己每帧提交、且镜像由内核回写，
    // 因此这里的职责整体消失。
    useEffect(() => {
        if (PARAM_EDITOR_KERNEL_ENABLED) return;
        let raf = 0;
        const reconcile = () => {
            raf = requestAnimationFrame(reconcile);
            const scroller = scrollerRef.current;
            if (!scroller) return;
            const offset = paramEditorSyncTimelineRef.current ? timelineOffsetRef.current : 0;
            const drawingScrollLeft = timelineViewportNativeToState(scroller.scrollLeft, offset);
            if (
                lastScrollLeftRef.current != null &&
                Math.abs(lastScrollLeftRef.current - drawingScrollLeft) <= 0.25
            ) {
                return;
            }
            syncScrollLeftRef.current(scroller);
        };
        raf = requestAnimationFrame(reconcile);
        return () => cancelAnimationFrame(raf);
    }, []);

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
        return { min: 0, max: 1 };
    }

    function getCurrentViewportForScrollbar(param: ParamName): ValueViewport {
        if (param === "pitch") {
            return pitchViewRef.current;
        }

        const bounds = getParamValueBoundsForScrollbar(param);
        return (
            paramViewsRef.current[param] ?? {
                center: (bounds.min + bounds.max) / 2,
                span: Math.max(1e-6, bounds.max - bounds.min),
            }
        );
    }

    /**
     * 把值域视口同步到竖向滚动位置（值域 → 像素）。
     *
     * 流程：钳制视口 → 更新内核数据镜像的值域 → 把 `center` 交给滚动载体。
     *
     * 特殊说明 1（内核模式）：滚动载体从原生 scroller 换成内核宿主。两者语义一致
     * （都是 0..1600 的像素域），因此这里只换载体、不换映射——手感由
     * `verticalScrollTopFromCenter` 单一来源保证，两个模式不会分叉。
     * 宿主内部自带去重，故无需在这里再判一次差值。
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
        if (PARAM_EDITOR_KERNEL_ENABLED && host) {
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
        }
    }

    /**
     * 从竖向像素滚动位置反推值域视口（像素 → 值域）。
     *
     * 流程：取当前视口与值域边界 → 由像素位置反算中心值 → 钳制后写回视口。
     *
     * 特殊说明：内核对齐阶段该函数只由原生滚动条路径调用（内核模式下竖向滚动
     * 完全由内核拥有，见 Task 7 的输入迁移）。保留它是为了不改变旧路径行为。
     *
     * @param scrollTop 竖向像素滚动位置（0..1600）。
     */
    function applyViewportFromVerticalScrollbar(scrollTop: number): void {
        const param = editParam;
        const currentView = clampViewport(param, getCurrentViewportForScrollbar(param));
        const bounds = getParamValueBoundsForScrollbar(param);
        const nextCenter = centerFromVerticalScrollTop({
            min: bounds.min,
            max: bounds.max,
            span: currentView.span,
            scrollTop,
            scrollRangePx: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
        });
        const nextView = clampViewport(param, {
            span: currentView.span,
            center: nextCenter,
        });

        if (Math.abs(nextView.center - currentView.center) <= 1e-6) {
            return;
        }

        if (param === "pitch") {
            setPitchView(nextView);
        } else {
            setParamViewport(param, nextView);
        }
    }

    const selectionRef = useRef<{ aBeat: number; bBeat: number } | null>(null);
    // 记录打开 MIDI 弹窗时的 editParam / toolMode 快照，避免异步加载轨道期间 Redux 状态变化导致 selectionAvailable 跳变
    const midiDialogOpenParamsRef = useRef<{
        editParam: string;
        toolMode: string;
    }>({ editParam: "pitch", toolMode: "select" });
    const [selectionUi, setSelectionUi] = useState<{
        aBeat: number;
        bBeat: number;
    } | null>(null);
    // 参数线选区存在性入仓（session.paramSelectionActive）：复制/剪切按
    // "当前选中了什么"路由时以此判定参数侧（selectionRef 的每次变更都
    // 成对经过 setSelectionUi，见 usePianoRollInteractions）。拖拽期间的
    // 重复派发值不变，immer 判定无修改直接跳过。
    useEffect(() => {
        dispatch(setParamSelectionActive(selectionUi !== null));
    }, [dispatch, selectionUi]);
    const [paramMorphOverlay, setParamMorphOverlay] = useState<ParamMorphOverlay | null>(null);
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

    const clipboardRef = useRef<{
        param: ParamName;
        framePeriodMs: number;
        values: number[];
    } | null>(null);

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
        secPerBeat,
        scrollLeft,
        pxPerBeat,
        viewWidth: viewSize.w,
        viewSizeRef,
        scrollLeftRef,
        pxPerBeatRef,
        invalidate,
        liveEditActiveRef,
    });

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

    // 计算 MIDI 导入的选区帧约束（与 pasteReaper 逻辑一致）
    const midiSelArgs = useMemo(() => {
        if (!midiDialogSelection) return {};
        const fp = paramView?.framePeriodMs ?? 5;
        const a = Math.min(midiDialogSelection.aBeat, midiDialogSelection.bBeat);
        const b = Math.max(midiDialogSelection.aBeat, midiDialogSelection.bBeat);
        const sf = Math.max(0, Math.floor((a * secPerBeat * 1000) / fp));
        const fc = Math.max(1, Math.ceil(((b - a) * secPerBeat * 1000) / fp));
        return { selectionStartFrame: sf, selectionMaxFrames: fc };
    }, [midiDialogSelection, paramView?.framePeriodMs, secPerBeat]);

    // selection 导入模式是否可用（基于弹窗打开时的快照，避免异步加载轨道时状态变化）
    const midiSelectionAvailable = useMemo(() => {
        if (!midiDialogSelection) return false;
        const p = midiDialogOpenParamsRef.current;
        return p.editParam === "pitch" && p.toolMode === "select";
    }, [midiDialogSelection]);

    // 将当前选区（帧范围）发布到总线，供 MenuBar 等判断“工程音阶”是否受 Tempo Map 影响。
    useEffect(() => {
        const sel = selectionUi;
        if (!sel) {
            publishPianoRollSelection(null);
            return;
        }
        const fp = paramView?.framePeriodMs ?? 5;
        const a = Math.min(sel.aBeat, sel.bBeat);
        const b = Math.max(sel.aBeat, sel.bBeat);
        publishPianoRollSelection({
            startFrame: Math.max(0, Math.floor((a * secPerBeat * 1000) / fp)),
            frameCount: Math.max(1, Math.ceil(((b - a) * secPerBeat * 1000) / fp)),
            framePeriodMs: fp,
        });
        return () => {
            publishPianoRollSelection(null);
        };
    }, [selectionUi, paramView?.framePeriodMs, secPerBeat]);

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
            for (let i = 0; i < override.edit.length; i += 1) {
                const frame = paramView.startFrame + i * paramView.stride;
                if (frame < drag.startFrame || frame >= windowEndFrame) continue;
                // 以 paramView 的原始帧为基准重复推导（预览事件幂等，不叠加）。
                const base = paramView.edit[i] ?? 0;
                override.edit[i] = shiftPitchValue(base, deltaSemitones);
            }
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
                nextEdit[i] = shiftPitchValue(paramView.edit[i] ?? 0, deltaSemitones);
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
    ]);

    // 包装 commitStroke：在 pointer-up 提交笔画后，清除 liveEditActive 状态，
    // 并触发可能被延迟 ?pitch_orig_updated 曲线刷新 ?
    const commitStroke: typeof commitStrokeBase = useCallback(
        async (points, mode) => {
            await commitStrokeBase(points, mode);
            liveEditActiveRef.current = false;
            notifyLiveEditEnded();
        },
        [commitStrokeBase, notifyLiveEditEnded],
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

    // scaleSegments 帧间缓存：播放头 invalidate 会让画布每帧重绘，但
    // tempoMap / 工程音阶 / 可见区间在帧间通常不变。按引用 + 0.02s 量化
    // 区间做 key，未变时复用上帧结果，避免每帧遍历 tempo 段并分配新数组。
    // 可见区间两侧本就各留 5s 余量，量化引入的 0.02s 漂移不会影响覆盖。
    const scaleSegmentsCacheRef = useRef<{
        tempoMap: unknown;
        scale: unknown;
        qStart: number;
        qEnd: number;
        result: ReturnType<typeof buildScaleSegments>;
    }>({ tempoMap: null, scale: null, qStart: NaN, qEnd: NaN, result: [] });

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

            // 选区高亮：裁剪到选区矩形（与 Canvas2D 的 ctx.clip 对应）
            const selection = selectionRef.current;
            if (selection && editValues.length >= 2) {
                const clip = selectionClipRect(axis, selection);
                if (clip && clip.w > 0) {
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
        }

        // ── ⑦ 剪贴板预览（不同的投影语义：从选区起点按原始帧距排布）──
        const preview = clipboardRef.current;
        const selection = selectionRef.current;
        if (
            preview &&
            selection &&
            preview.param === editParam &&
            preview.values.length > 0
        ) {
            const clip = selectionClipRect(axis, selection);
            const beatToSec = Math.max(1e-9, secPerBeat);
            if (clip && clip.w > 0) {
                layers.push({
                    values: preview.values,
                    param: editParam,
                    startFrame: 0,
                    stride: 1,
                    framePeriodMs: preview.framePeriodMs,
                    lineWidthPx: 2,
                    rgba: parseRgbaColor(normalizeCssColor(resolveClipboardPreviewColor(isDark))),
                    dash: toDashTuple(getFixedDashPattern(4, 4)),
                    projection: "clipboard",
                    clipStartSec: Math.min(selection.aBeat, selection.bBeat) * beatToSec,
                    clipEndSec: Math.max(selection.aBeat, selection.bBeat) * beatToSec,
                    clipRect: clip,
                    valueToY: (v) => project(editParam, v),
                });
            }
        }

        return layers;
    }

    /**
     * 把选区（beat）换算为视口坐标的裁剪矩形。
     *
     * @param axis 当前投影。
     * @param selection 选区（beat）。
     * @returns 裁剪矩形；选区为空或宽度为 0 时返回 null。
     */
    function selectionClipRect(
        axis: TimelineAxis,
        selection: { aBeat: number; bBeat: number },
    ): { x: number; y: number; w: number; h: number } | null {
        const beatToSec = Math.max(1e-9, secPerBeat);
        const selMin = Math.min(selection.aBeat, selection.bBeat) * beatToSec;
        const selMax = Math.max(selection.aBeat, selection.bBeat) * beatToSec;
        const x0 = secToViewportPx(axis, selMin);
        const x1 = secToViewportPx(axis, selMax);
        const w = x1 - x0;
        if (!(w > 0)) return null;
        return { x: x0, y: 0, w, h: viewSizeRef.current.h };
    }

    // Keep draw function always up-to-date (invalidate() is stable and calls drawRef.current()).
    drawRef.current = () => {
        // 滚动热路径的投影：用 ref 构造，因为滚动时 ref 同步更新而 React
        // state 滞后一帧（渲染期的 prAxis 不能用于此处）。
        const drawAxis = createTimelineAxis({
            pxPerSec: pxPerSecRef.current,
            scrollLeftPx: scrollLeftRef.current,
            viewportWidthPx: viewSizeRef.current.w,
            dpr: window.devicePixelRatio || 1,
        });
        const scaleSegStartQ =
            Math.round(Math.max(0, viewportStartSec(drawAxis) - 5) / 0.02) * 0.02;
        const scaleSegEndQ = Math.round((viewportEndSec(drawAxis) + 5) / 0.02) * 0.02;
        const segCache = scaleSegmentsCacheRef.current;
        if (
            segCache.tempoMap !== s.tempoMap ||
            segCache.scale !== effectiveProjectScale ||
            segCache.qStart !== scaleSegStartQ ||
            segCache.qEnd !== scaleSegEndQ
        ) {
            segCache.tempoMap = s.tempoMap;
            segCache.scale = effectiveProjectScale;
            segCache.qStart = scaleSegStartQ;
            segCache.qEnd = scaleSegEndQ;
            segCache.result = buildScaleSegments(
                s.tempoMap,
                effectiveProjectScale,
                scaleSegStartQ,
                scaleSegEndQ,
            );
        }
        // 曲线图层（阶段 3）：每帧重建描述符列表。
        //
        // 【为什么要每帧构建】滚动/缩放会改变可见段与 `axis`，而 GL 侧要在绘制时
        // 才投影；描述符里的 `values` 引用与视口无关（数据没变），因此这里的成本
        // 只是一次浅层数组构建，不复制采样值。
        if (PARAM_EDITOR_KERNEL_ENABLED && PARAM_EDITOR_CURVE_GL_ENABLED) {
            kernelDataRef.current.curves = buildCurveLayers(drawAxis);
        } else {
            kernelDataRef.current.curves = null;
        }

        // 叠加层镜像（选区块 + 播放头）必须**每帧**更新：播放头用的是插值的
        // 视觉值 `visualPlayheadSecRef`，它不由 React 渲染驱动（见下方注释），
        // 因此不能在 render 期写入镜像——那样播放头会停在旧的提交值上。
        if (PARAM_EDITOR_KERNEL_ENABLED) {
            // 只喂播放头：选区块仍由主画布绘制（见 PianoRollOverlaySpec 说明）。
            kernelDataRef.current.overlay = {
                playheadSec: visualPlayheadSecRef.current,
            };
        }
        /**
         * 主画布的内容签名（阶段 2 Task 6）。
         *
         * 【必须包含什么】主画布上绘制的**全部输入**：
         * - 绘图资源：各条曲线数据、参考线、检测曲线、副参数视口、morph 叠加、
         *   剪贴板预览、选区块、live 编辑覆盖、音阶高亮（含 tempoMap 段）；
         * - 视口：`viewSize`、`pxPerSec`、`scrollLeft`、`dpr`（滚动/缩放会改变投影）；
         * - 主题与字体（颜色解析与文字宽度都会影响像素结果）；
         * - `pitchAnalysisPending`（它会提前 return，改变绘制内容）。
         *
         * 【必须**不**包含什么】播放头位置——它已由 GL 叠加层绘制。把它编进签名会让
         * 播放帧的签名每帧变化、缓存失效，那就退回"每帧重绘曲线"。
         *
         * 【为什么用引用数组 + join】大部分输入是数组/对象引用（Redux 只在内容变化时
         * 换引用），直接比引用既快又准；数值项显式列举。漏项的代价是"该图层不再更新"，
         * 因此这里**宁可多编**：低频变化的项一并纳入，成本只是偶尔多一次重绘。
         */
        const mainContentSignature = [
            viewSize.w,
            viewSize.h,
            pxPerSecRef.current,
            scrollLeftRef.current,
            Math.round((window.devicePixelRatio || 1) * 100),
            editParam,
            themeMode,
            fontFamily,
            pitchEnabled ? 1 : 0,
            // 数据与几何（引用比较）。刻意与传给 drawPianoRoll 的字段一一对应，
            // 避免"签名里写了 A、实际喂给绘制的是 B"这种漂移。
            detectedPitchCurves,
            referencePitchOverlays,
            secondaryParamViews,
            visibleSecondaryParamIds,
            paramMorphOverlay,
            s.showClipboardPreview ? clipboardRef.current : null,
            selectionRef.current,
            liveEditOverrideRef.current,
            effectiveProjectScale,
            s.tempoMap,
            segCache.result,
            s.pitchSnapUnit,
            s.scaleHighlightMode,
            s.toolMode,
            snapToggleHeld,
            // 视口中心/跨度（用 ref 值，避免依赖 React 渲染时机）
            pitchViewRef.current.center,
            pitchViewRef.current.span,
            paramViewsRef.current,
            secondaryParamViews,
        ].join("|");

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
            overlayText: !pitchEnabled
                ? editParam === "pitch"
                    ? pitchHardDisableReason
                    : childPitchHardDisableReason
                : null,
            liveEditOverride: liveEditOverrideRef.current,
            selection: selectionRef.current,
            axis: drawAxis,
            secPerBeat,
            // 画布每帧重绘（onFrame invalidate），播放头必须用插值的视觉值：
            // 用 Redux 提交值会让 60fps 的重绘画着同一个旧播放头（且与标尺
            // 的 DOM 插值播放头节奏不一致、短暂错位）。
            playheadSec: visualPlayheadSecRef.current,
            referencePitchOverlays,
            detectedPitchCurves,
            isDark: themeMode === "dark",
            // 阶段 2：GL 层接管网格时，Canvas2D 必须跳过它（两张画布叠放，
            // 都画会半透明叠加 + 亚像素重影）。GL 未启用时行为与迁移前一致。
            skipGrid: PARAM_EDITOR_GL_SCENE_ENABLED,
            // 键盘几何与轴文字都归 GL（Task 4 / Task 5）。
            skipKeyboardGeometry: PARAM_EDITOR_GL_SCENE_ENABLED,
            skipAxisText: PARAM_EDITOR_GL_SCENE_ENABLED,
            // 轴画布全部内容归 GL（Task 4/5）-> 整张跳过（含清屏）。
            skipAxisCanvas: PARAM_EDITOR_GL_SCENE_ENABLED,
            // 播放头归 GL 叠加层（Task 6）。**选区块不在此列**：它属于曲线之下的
            // 图层，仍由主画布绘制（见 render.ts 的 skipPlayhead 说明）。
            skipPlayhead: PARAM_EDITOR_GL_SCENE_ENABLED,
            // 曲线归 GL（阶段 3）。morph 手柄不在曲线层内，仍由主画布绘制。
            skipCurves: PARAM_EDITOR_CURVE_GL_ENABLED,
            // 主画布内容缓存（Task 6）：签名只含**主画布自己绘制的内容**与视口，
            // 不含播放头（它已归 GL 叠加层）——这正是播放帧能跳过曲线重绘的原因。
            mainContentSignature: PARAM_EDITOR_GL_SCENE_ENABLED ? mainContentSignature : undefined,
            fontFamily,
            clipboardPreview: s.showClipboardPreview ? clipboardRef.current : null,
            // pitch snap visual helpers
            pitchSnapUnit: s.pitchSnapUnit,
            projectScale: effectiveProjectScale,
            scaleHighlightMode: s.scaleHighlightMode,
            // 可见秒区间由 axis 提供：此前这里写作 scrollLeft / pxPerSec
            // （先除后乘），与其余图层的换算不等价。
            scaleSegments: segCache.result,
            toolMode: s.toolMode,
            snapToggleHeld: snapToggleHeld,
            paramMorphOverlay,
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
        secPerBeat,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerBeatRef,
        pxPerSecRef,
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
        paramStretchKb: stretchKb,
        vibratoAmplitudeAdjustKb,
        vibratoFrequencyAdjustKb,
        vibratoDragAmplitudeIncreaseKb,
        vibratoDragAmplitudeDecreaseKb,
        vibratoDragFrequencyIncreaseKb,
        vibratoDragFrequencyDecreaseKb,
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
        onMorphOverlayChange: setParamMorphOverlay,
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
    const onScrollerScroll = useCallback(
        (e: React.UIEvent<HTMLDivElement>) => {
            interactions.onScrollerScroll(e);

            const scroller = e.currentTarget;
            const currentView = clampViewport(editParam, getCurrentViewportForScrollbar(editParam));
            const bounds = getParamValueBoundsForScrollbar(editParam);
            const expectedTop = verticalScrollTopFromCenter({
                min: bounds.min,
                max: bounds.max,
                span: currentView.span,
                center: currentView.center,
                scrollRangePx: PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX,
            });

            // 与当前视口计算出的滚动位置几乎一致时，说明是横向滚动或程序同步，不需要反向回写。
            if (Math.abs(scroller.scrollTop - expectedTop) <= 0.75) {
                return;
            }

            applyViewportFromVerticalScrollbar(scroller.scrollTop);
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [editParam, interactions],
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
            const h = Math.max(1, bounds.height);
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
        const el = axisWrapRef.current;
        if (!el) return;

        let isPointerDown = false;
        let activeMidiNote: number | null = null;

        const getMidiNoteFromY = (clientY: number): number => {
            const bounds = el.getBoundingClientRect();
            const y = clientY - bounds.top;
            const h = Math.max(1, bounds.height);
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
    }, [pitchViewRef]);

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

    // ── Edit operation handler (shared by context menu + MenuBar events) ──
    const handleEditOp = useCallback(
        async (op: string, data?: Record<string, unknown>) => {
            if (!rootTrackId) return;
            const fp = paramView?.framePeriodMs ?? 5;

            if (op === "selectAll") {
                if (s.toolMode !== "select") return;
                const totalBeats = dynamicProjectSec / secPerBeat;
                selectionRef.current = { aBeat: 0, bBeat: totalBeats };
                setSelectionUi({ aBeat: 0, bBeat: totalBeats });
                invalidate();
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
            // 内创建选区，并把交互焦点切到参数编辑器侧 —— 复制/剪切路由
            // （resolveCopyCutRoute 依据 selectionContext，经由下方 selectionUi
            // 同步派发 setParamSelectionActive 标记）与活动表面
            // （focusSurface，外来源粘贴兜底等）随之指向参数编辑器。
            if (op === "selectClipParamRange") {
                const clipId = typeof data?.clipId === "string" ? data.clipId : "";
                const clip = store.getState().session.clips.find((entry) => entry.id === clipId);
                if (!clip) return;
                const aBeat = Math.max(0, clip.startSec / secPerBeat);
                const bBeat = Math.max(0, (clip.startSec + clip.lengthSec) / secPerBeat);
                selectionRef.current = { aBeat, bBeat };
                setSelectionUi({ aBeat, bBeat });
                setActiveSurfaceExplicit("pianoRoll");
                invalidate();
                return;
            }

            // VocalShifter clipboard paste stays a dedicated menu action
            // (file-based clipboard), and works with or without selection.
            if (op === "pasteVocalShifter") {
                const sel2 = selectionRef.current;
                let selArgs:
                    | {
                          selectionStartFrame?: number;
                          selectionMaxFrames?: number;
                      }
                    | undefined;
                if (sel2) {
                    const a = Math.min(sel2.aBeat, sel2.bBeat);
                    const b = Math.max(sel2.aBeat, sel2.bBeat);
                    const sf = Math.max(0, Math.floor((a * secPerBeat * 1000) / fp));
                    const fc = Math.max(1, Math.ceil(((b - a) * secPerBeat * 1000) / fp));
                    selArgs = {
                        selectionStartFrame: sf,
                        selectionMaxFrames: fc,
                    };
                }
                void dispatch(
                    pasteVocalShifterClipboard({
                        ...selArgs,
                        activeParam: editParam,
                    }),
                );
                bumpRefreshToken();
                return;
            }

            // REAPERMedia fallback used by the normal paste operation when no
            // HiFiShifter param clipboard data is available.
            const pasteReaperClipboardFallback = () => {
                const sel2 = selectionRef.current;
                let selArgs:
                    | {
                          selectionStartFrame?: number;
                          selectionMaxFrames?: number;
                      }
                    | undefined;
                if (sel2) {
                    const a = Math.min(sel2.aBeat, sel2.bBeat);
                    const b = Math.max(sel2.aBeat, sel2.bBeat);
                    const sf = Math.max(0, Math.floor((a * secPerBeat * 1000) / fp));
                    const fc = Math.max(1, Math.ceil(((b - a) * secPerBeat * 1000) / fp));
                    selArgs = {
                        selectionStartFrame: sf,
                        selectionMaxFrames: fc,
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
                            setMidiDialogSelection(sel2 ? { ...sel2 } : null);
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

            const selAtEntry = selectionRef.current;
            // Normal paste prefers HiFiShifter param data. When there is no
            // pitch selection (or pitch editing is unavailable), the normal
            // paste still tries REAPERMedia data, matching the removed
            // dedicated "Paste Reaper Clipboard Data" action.
            if (op === "paste" && (!selAtEntry || !pitchEnabled)) {
                pasteReaperClipboardFallback();
                return;
            }

            const sel = selectionRef.current;
            if (!sel) return;
            if (!pitchEnabled) return;

            const aBeat = Math.min(sel.aBeat, sel.bBeat);
            const bBeat = Math.max(sel.aBeat, sel.bBeat);
            const startSec = aBeat * secPerBeat;
            const durSec = Math.max(0, (bBeat - aBeat) * secPerBeat);
            const startFrame = Math.max(0, Math.floor((startSec * 1000) / fp));
            const frameCount = clamp(Math.ceil((durSec * 1000) / fp), 1, 200_000);

            // 选区编辑统一入口：取数/编辑/边缘淡化/回写全部在
            // selectionEditApply 模块内完成（delta 空间交叉淡化 + 毫秒定标）。
            // 平滑度解析顺序保持旧语义：对话框显式传入 → store 全局设置。
            const runSelectionEdit = async (
                editSelection: (currentSelectionVals: number[]) => number[],
                extension?: SelectionEditExtension,
            ) => {
                const ok = await applySelectionEditWithEdgeSmoothing({
                    trackId: rootTrackId,
                    param: editParam,
                    startFrame,
                    frameCount,
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
                });
                if (ok) bumpRefreshToken();
            };

            switch (op) {
                case "copy": {
                    const res = await paramsApi.getParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        frameCount,
                        1,
                    );
                    if (!res?.ok) return;
                    const payload = res as ParamFramesPayload;
                    clipboardRef.current = {
                        param: editParam,
                        framePeriodMs: Number(payload.frame_period_ms ?? fp) || fp,
                        values: (payload.edit ?? []).map((v) => Number(v) || 0),
                    };
                    try {
                        await writeSystemClipboardObject({
                            version: 1,
                            kind: "param",
                            param: editParam,
                            framePeriodMs: Number(payload.frame_period_ms ?? fp) || fp,
                            values: (payload.edit ?? []).map((v) => Number(v) || 0),
                        });
                    } catch {
                        // ignore clipboard write failures
                    }
                    // 刷新剪贴板预览
                    invalidate();
                    break;
                }
                case "cut": {
                    const res = await paramsApi.getParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        frameCount,
                        1,
                    );
                    if (!res?.ok) return;
                    const payload = res as ParamFramesPayload;
                    clipboardRef.current = {
                        param: editParam,
                        framePeriodMs: Number(payload.frame_period_ms ?? fp) || fp,
                        values: (payload.edit ?? []).map((v) => Number(v) || 0),
                    };
                    try {
                        await writeSystemClipboardObject({
                            version: 1,
                            kind: "param",
                            param: editParam,
                            framePeriodMs: Number(payload.frame_period_ms ?? fp) || fp,
                            values: (payload.edit ?? []).map((v) => Number(v) || 0),
                        });
                    } catch {
                        // ignore clipboard write failures
                    }
                    invalidate();
                    // 初始化（恢复原始值）
                    await paramsApi.restoreParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        frameCount,
                        true,
                    );
                    bumpRefreshToken();
                    break;
                }
                case "paste": {
                    let clip = clipboardRef.current;
                    try {
                        const fromSystem = await readSystemClipboardObject("param");
                        if (fromSystem?.kind === "param") {
                            clip = {
                                param: fromSystem.param,
                                framePeriodMs: Number(fromSystem.framePeriodMs) || fp,
                                values: Array.isArray(fromSystem.values)
                                    ? fromSystem.values.map((v) => Number(v) || 0)
                                    : [],
                            };
                            clipboardRef.current = clip;
                        }
                    } catch {
                        // ignore and fallback to internal clipboard
                    }
                    if (!clip) {
                        // No HiFiShifter param clipboard data: try REAPERMedia.
                        pasteReaperClipboardFallback();
                        return;
                    }

                    let pasteValues: number[];
                    if (clip.param === editParam) {
                        pasteValues =
                            clip.values.length > frameCount
                                ? clip.values.slice(0, frameCount)
                                : clip.values;
                    } else if (
                        clip.param === "pitch" &&
                        (isChildPitchOffsetCentsParam(editParam) ||
                            isChildPitchOffsetDegreesParam(editParam))
                    ) {
                        const targetParam = parseChildPitchOffsetParam(editParam);
                        if (!targetParam) return;
                        const resolvedRootTrackId = resolveRootTrackId(
                            s.tracks,
                            targetParam.trackId,
                        );
                        if (!resolvedRootTrackId || resolvedRootTrackId !== rootTrackId) {
                            return;
                        }

                        const converted = await buildChildOffsetPasteValuesHelper({
                            tracks: s.tracks,
                            rootTrackId,
                            targetTrackId: targetParam.trackId,
                            startFrame,
                            frameCount,
                            clipboardPitch: clip.values,
                            mode: targetParam.mode as "cents" | "degrees",
                            paramsApi,
                            pitchDeltaToDegreeSteps: pitchDeltaToDegreeSteps,
                            projectScale: effectiveProjectScale,
                            // Tempo Map 感知：按帧时刻解析生效音阶。
                            scaleAtFrame: (frame: number) =>
                                projectScaleAtSec((frame * fp) / 1000) ?? effectiveProjectScale,
                        });
                        if (!converted) return;

                        pasteValues = converted.slice(0, frameCount);
                    } else {
                        return;
                    }

                    await paramsApi.setParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        pasteValues,
                        true,
                    );
                    bumpRefreshToken();
                    break;
                }
                case "initialize": {
                    await paramsApi.restoreParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        frameCount,
                        true,
                    );
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
                    );
                    if (!res?.ok) return;
                    const payload = res as ParamFramesPayload;
                    const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
                    if (vals.length === 0) return;
                    const result = averageSelectionValues(vals, editParam, strengthPercent);
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
                    // 多取两侧各 3σ 帧上下文：高斯平滑用真实延拓做边界，
                    // 平滑结果与选区外曲线无缝（旧实现只取选区内、边界处
                    // 会产生新台阶）。
                    const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
                    const pad = smoothContextPadFrames(strength, fpMs);
                    const ctxStart = Math.max(0, startFrame - pad);
                    const leftLen = startFrame - ctxStart;
                    const res = await paramsApi.getParamFrames(
                        rootTrackId,
                        editParam,
                        ctxStart,
                        leftLen + frameCount + pad,
                        1,
                    );
                    if (!res?.ok) return;
                    const payload = res as ParamFramesPayload;
                    const all = (payload.edit ?? []).map((v) => Number(v));
                    const vals = all.slice(leftLen, leftLen + frameCount);
                    if (vals.length === 0) return;
                    const result = smoothSelectionValues(vals, editParam, strength, {
                        framePeriodMs: fpMs,
                        leftContext: all.slice(0, leftLen),
                        rightContext: all.slice(leftLen + frameCount),
                    });
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
                case "addVibrato": {
                    const amplitude = Number(data?.amplitude ?? 30);
                    const rateHz = Number(data?.rate ?? 5.5);
                    const period = rateHz > 0 ? 1000 / rateHz : 200;
                    const attack = Number(data?.attack ?? 50);
                    const release = Number(data?.release ?? 50);
                    const phase = Number(data?.phase ?? 0);
                    const res = await paramsApi.getParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        frameCount,
                        1,
                    );
                    if (!res?.ok) return;
                    const payload = res as ParamFramesPayload;
                    const vals = (payload.edit ?? []).map((v) => Number(v) || 0);
                    const fpMs = Number(payload.frame_period_ms ?? fp) || fp;
                    const totalMs = vals.length * fpMs;
                    const attackMs = Math.min(attack, totalMs / 2);
                    const releaseMs = Math.min(release, totalMs / 2);
                    // For pitch: amplitude in cents → divide by 100 to get semitones
                    // For other params: amplitude is a raw value used directly as max deviation
                    const isPitchVib = editParam === "pitch";
                    const ampFactor = isPitchVib ? amplitude / 100 : amplitude;
                    const result = vals.map((v, i) => {
                        const tMs = i * fpMs;
                        let env = 1;
                        if (tMs < attackMs) env = tMs / Math.max(1, attackMs);
                        else if (tMs > totalMs - releaseMs)
                            env = (totalMs - tMs) / Math.max(1, releaseMs);
                        const phaseRad = (phase * Math.PI) / 180;
                        const vib = Math.sin((2 * Math.PI * tMs) / Math.max(1, period) + phaseRad);
                        return v + ampFactor * env * vib;
                    });
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
            secPerBeat,
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

    const handleSaveAsPitchRef = useCallback(async () => {
        const sel = selectionRef.current;
        if (!sel || !rootTrackId) return;

        const aBeat = Math.min(sel.aBeat, sel.bBeat);
        const bBeat = Math.max(sel.aBeat, sel.bBeat);
        const startSec = aBeat * secPerBeat;
        const lengthSec = Math.max(0.01, (bBeat - aBeat) * secPerBeat);

        const fp = paramView?.framePeriodMs ?? 5;
        const startFrame = Math.max(0, Math.floor((startSec * 1000) / fp));
        const frameCount = Math.max(1, Math.ceil((lengthSec * 1000) / fp));

        const res = await paramsApi.getParamFrames(rootTrackId, "pitch", startFrame, frameCount, 1);
        if (!res?.ok || !res.edit) return;

        const pitchValues: number[] = (res.edit as number[]).map((v) => Number(v) || 0);

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
        if (pitchValues.length > 0) {
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
        }

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
                    c.startSec < startSec + lengthSec &&
                    c.startSec + c.lengthSec > startSec,
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
                templates: [
                    {
                        trackId: targetTrackId,
                        name: "Pitch Ref",
                        startSec,
                        lengthSec,
                        midiNoteData,
                        midiFillGaps: true,
                    },
                ],
            }),
        );
    }, [
        selectionRef,
        rootTrackId,
        secPerBeat,
        paramView,
        s.tracks,
        s.selectedTrackId,
        s.clips,
        dispatch,
    ]);

    const handleExportMidiFromEditor = useCallback(async () => {
        if (!rootTrackId) return;
        const sel = selectionRef.current;
        if (!sel) return;

        const saveResult = await coreApi.pickMidiOutputPath();
        if (!saveResult.ok || saveResult.canceled || !saveResult.path) return;

        const aBeat = Math.min(sel.aBeat, sel.bBeat);
        const bBeat = Math.max(sel.aBeat, sel.bBeat);
        const startSec = aBeat * secPerBeat;
        const endSec = Math.max(startSec + 0.01, bBeat * secPerBeat);

        const selectedTrack = s.tracks.find((t) => t.id === s.selectedTrackId);
        const trackName = selectedTrack?.name ?? "Track";
        const scaleNotes =
            SCALE_NOTES[(s.project?.baseScale as keyof typeof SCALE_NOTES) ?? "C"] ?? SCALE_NOTES.C;

        await paramsApi.exportPitchToMidi({
            outputPath: saveResult.path,
            tracks: [
                {
                    trackId: s.selectedTrackId ?? rootTrackId,
                    rootTrackId,
                    name: trackName,
                    startSec,
                    endSec,
                },
            ],
            bpm: s.bpm,
            beatsPerBar: s.project?.beatsPerBar ?? 4,
            baseScale: s.project?.baseScale ?? "C",
            projectScaleNotes: scaleNotes,
        });
    }, [rootTrackId, selectionRef, secPerBeat, s]);

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
                            data-tooltip={`${tAny("drag_direction")}: ${tAny(activeDragDirection === "free" ? "drag_direction_free" : activeDragDirection === "x-only" ? "drag_direction_x_only" : "drag_direction_y_only")}`}
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
                                    <FormantParamButton
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
                        style={{ width: AXIS_W, flex: 1 }}
                    >
                        {/* 键盘轴 GL 层（阶段 2，Task 4）：铺在 Canvas2D 轴画布**下面**
                            （DOM 顺序在前、无 z-index），只画几何；音名标签仍由
                            Canvas2D 画在上层（Task 5 才迁标签）。 */}
                        {PARAM_EDITOR_GL_SCENE_ENABLED ? (
                            <canvas
                                ref={glAxisCanvasRef}
                                className="absolute inset-0 pointer-events-none"
                                aria-hidden
                            />
                        ) : null}
                        <canvas ref={axisCanvasRef} className="absolute inset-0" />
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
                        playheadLineRef={rulerPlayheadLineRef}
                        playheadHeadRef={rulerPlayheadHeadRef}
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

                    {/* 内核模式下自绘滚动条的定位容器：滚动条必须是**滚动容器之外**
                        的兄弟节点。放在滚动容器内部会被内容一起滚走（绝对定位在滚动
                        容器里仍随内容平移），这是自绘滚动条最经典的错位根因。 */}
                    <div className="flex-1 min-w-0 relative">
                        <div
                            ref={scrollerRef}
                            className={
                                // 内核模式：隐藏原生滚动条（自绘条取代），但**保留
                                // `overflow: scroll`**——原生 scroller 此时是被动镜像，
                                // 必须保留滚动范围才能接受宿主每帧的程序化回写，也让
                                // 尚未迁移的输入代码（中键平移等）继续可读可写。
                                PARAM_EDITOR_KERNEL_ENABLED
                                    ? "absolute inset-0 bg-qt-graph-bg overflow-x-scroll overflow-y-scroll hide-scrollbar outline-none focus:outline-none focus-visible:outline-none"
                                    : "absolute inset-0 bg-qt-graph-bg overflow-x-scroll overflow-y-scroll custom-scrollbar outline-none focus:outline-none focus-visible:outline-none"
                            }
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
                                        sticky
                                    />

                                    <PianoRollWaveformSurface
                                        clips={clipPeaks}
                                        widthPx={viewSize.w}
                                        heightPx={viewSize.h}
                                        scrollLeftPx={scrollLeft}
                                        pxPerSec={pxPerSec}
                                        colors={waveformColors}
                                    />

                                    {/* GL 静态层（阶段 2）：网格等静态图层。
                                        层级说明：它是**最底层**——DOM 顺序在 Canvas2D
                                        主画布之前，且不设 z-index（Canvas2D 主画布用
                                        `absolute inset-0` 覆盖其上）。因此 GL 层只画
                                        静态底图，曲线 / 选区 / 播放头仍由 Canvas2D
                                        画在上层。
                                        仅在开关开启时挂载：未开启时不创建画布，
                                        避免多申请一个 WebGL 上下文。 */}
                                    {PARAM_EDITOR_GL_SCENE_ENABLED ? (
                                        <canvas
                                            ref={glCanvasRef}
                                            data-piano-roll-gl-scene
                                            className="absolute inset-0 pointer-events-none"
                                            aria-hidden
                                        />
                                    ) : null}

                                    {/* `data-piano-roll-canvas`：主曲线画布的稳定选择器。
                                        浏览器自动化验证（scripts/dev-shot.mjs）需要按元素
                                        截图逐像素比对曲线，而 canvas 本身没有可锚定的属性；
                                        内核模式下页面里有多个同尺寸 canvas，按尺寸猜会命中
                                        错误的那一个。 */}
                                    <canvas
                                        ref={canvasRef}
                                        data-piano-roll-canvas
                                        className="absolute inset-0"
                                        style={{ cursor: canvasCursor }}
                                        onPointerMove={interactions.onCanvasPointerMove}
                                        onPointerLeave={interactions.onCanvasPointerLeave}
                                        onPointerDown={interactions.onCanvasPointerDown}
                                    />

                                    {/* 动态叠加层（阶段 2，Task 6）：播放头与选区。
                                        层序：DOM 顺序在曲线画布**之后** => 覆盖其上。
                                        指针事件全部穿透（interactions 挂在曲线画布上），
                                        否则会挡住参数编辑的命中测试。 */}
                                    {PARAM_EDITOR_GL_SCENE_ENABLED ? (
                                        <canvas
                                            ref={glOverlayCanvasRef}
                                            className="absolute inset-0 pointer-events-none"
                                            aria-hidden
                                        />
                                    ) : null}
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

                        {/* 自绘滚动条（仅内核模式）：几何由宿主每帧写入。
                            样式对齐旧实现的原生滚动条（`.custom-scrollbar`）：
                            - thumb 取 `--qt-scrollbar-thumb`（浅色主题下才看得见），
                              而不是固定半透明黑；
                            - 轨道**透明**（旧实现是 `scrollbar-color: … transparent`），
                              加底色会多出一条灰带；
                            - 8px 厚 + 胶囊圆角，对应 macOS 的 overlay thin 滚动条
                              （旧实现的原生滚动条不占布局，这里绝对定位叠加，行为等价）。
                            - 外层即**轨道**：承接「点空白翻页」。宿主的 thumb 处理器
                              会 `stopPropagation`，因此到达轨道的按下必然不在 thumb 上。 */}
                        {PARAM_EDITOR_KERNEL_ENABLED ? (
                            <>
                                <div
                                    ref={vScrollbarTrackRef}
                                    className="absolute right-0 top-0 bottom-0 w-2 z-20"
                                >
                                    <div
                                        ref={vScrollbarThumbRef}
                                        className="absolute left-0 w-full rounded-full bg-[var(--qt-scrollbar-thumb)]"
                                    />
                                </div>
                                <div
                                    ref={hScrollbarTrackRef}
                                    className="absolute bottom-0 left-0 right-0 h-2 z-20"
                                >
                                    <div
                                        ref={hScrollbarThumbRef}
                                        className="absolute top-0 h-full rounded-full bg-[var(--qt-scrollbar-thumb)]"
                                    />
                                </div>
                            </>
                        ) : null}
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
                selectionStartFrame={midiSelArgs.selectionStartFrame}
                selectionMaxFrames={midiSelArgs.selectionMaxFrames}
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
                />
            )}
        </Flex>
    );
};
