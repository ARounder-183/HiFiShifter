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
import { useAppDispatch, useAppSelector } from "../../app/hooks";
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
import { findFirstExternalPathAction } from "./timeline/dnd";
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
    buildRulerTicks,
    clamp,
    formatCursorTime,
    gridStepBeats,
} from "./timeline";
import { timeRulerHeightPx } from "./timeline/rulerHeight";
import { TempoMapCornerButton } from "./timeline/TempoMapCornerButton";
import { invokeGridRedrawHandler } from "./timeline/gridRedrawBridge";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeline";
import type { TempoMap } from "../../utils/tempoMap";
import {
    buildScaleSegments,
    buildTempoGridLineXsForViewport,
    effectiveScaleAtSec,
} from "../../utils/tempoMap";
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
import { averageSelectionValues, smoothSelectionValues } from "./pianoRoll/selectionTransforms";
import { usePianoRollData } from "./pianoRoll/usePianoRollData";
import { useClipsPeaksForPianoRoll } from "./pianoRoll/useClipsPeaksForPianoRoll";
import { PianoRollWaveformSurface } from "./pianoRoll/PianoRollWaveformSurface";
import { usePianoRollInteractions } from "./pianoRoll/usePianoRollInteractions";
import { useLiveParamEditing } from "./pianoRoll/useLiveParamEditing";
import { getParamShiftStep } from "./pianoRoll/paramShiftStep";
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

import { useAsyncPitchRefresh } from "../../hooks/useAsyncPitchRefresh";
import { ProgressBar } from "../ProgressBar";

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

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX = 1600;

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
    /** 是否为主参数（激活）：整个药丸整体高亮 */
    active: boolean;
    /** 激活时的强调色（音高=grass，共振峰/其它=amber） */
    accent: "grass" | "amber";
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
 * 各片段共享一块连续背景（由 data-accent-color 决定激活色），
 * 片段间用细分隔线区分；悬停时只高亮当前片段，提示其独立可点击。
 */
const ParamToolbarPill: React.FC<ParamToolbarPillProps> = ({
    label,
    labelTooltip,
    active,
    accent,
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
            data-accent-color={accent}
            data-active={active ? "true" : undefined}
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
                accent="amber"
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
                accent="amber"
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
    const rafRef = useRef<number | null>(null);
    const visualPlayheadSecRef = useRef(0);
    const rulerPlayheadLineRef = useRef<HTMLDivElement | null>(null);
    const rulerPlayheadHeadRef = useRef<HTMLDivElement | null>(null);
    const drawRef = useRef<() => void>(() => {});
    const invalidate = useCallback(() => {
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
    const pianoRollCopyKb = useAppSelector((state) => selectKeybinding(state, "pianoRoll.copy"));
    const pianoRollPasteKb = useAppSelector((state) => selectKeybinding(state, "pianoRoll.paste"));
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

    // Task 6.3: 集成 useAsyncPitchRefresh Hook
    const asyncRefresh = useAsyncPitchRefresh();
    const [showSuccessMessage] = useState(false);

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
    // 渲染期立即同步 ref，确保同步视口在 layout effect 落地时，
    // Canvas 读取到的是与标尺/网格同一帧的新缩放与滚动值。
    scrollLeftRef.current = scrollLeft;
    pxPerBeatRef.current = pxPerBeat;
    pxPerSecRef.current = pxPerSec;
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
                scroller.scrollLeft = store.scrollLeft;
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
            // 禁用时移除偏移补偿：把原生滚动位置还原为绘制坐标。
            if (scroller) {
                const next = Math.max(0, scrollLeftRef.current);
                scroller.scrollLeft = next;
                scrollLeftRef.current = next;
                lastScrollLeftRef.current = next;
                setScrollLeft(next);
                applyScrollLayers(next);
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
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
        scroller.scrollLeft = pending.nativeScrollLeft;
        syncScrollLeft(scroller);
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
        scroller.scrollLeft = native;
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
        const expectedNative = timelineViewportStateToNative(next, offset);
        if (Math.abs(scroller.scrollLeft - expectedNative) > 0.5) {
            scroller.scrollLeft = expectedNative;
        }
        // 同步更新状态：让标尺/网格与画布在同一帧对齐，消除缩放闪屏。
        setScrollLeft(next);
        // eslint-disable-next-line react-hooks/exhaustive-deps
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
            pxPerBeatRef.current = nextPxPerSec * (60 / Math.max(1e-6, s.bpm));
            pxPerSecRef.current = nextPxPerSec;
            horizontalZoomPendingRef.current = {
                nextScale: nextPxPerSec,
                nextScrollLeft: nextNativeScrollLeft,
            };
            setPxPerSec(nextPxPerSec);
        },
        [s.bpm, setPxPerSec],
    );

    useEffect(() => {
        function onZoomFocused(e: Event) {
            const { playheadSec, projectSec } = zoomTimelineStateRef.current;
            const active = document.activeElement as HTMLElement | null;
            const inPianoRoll =
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]") ||
                document.body.getAttribute("data-hs-focus-window") === "pianoRoll";
            if (!inPianoRoll) return;

            const factor = Number((e as CustomEvent<{ factor?: number }>).detail?.factor ?? 1);
            if (!Number.isFinite(factor) || factor <= 0) return;

            const scroller = scrollerRef.current;
            if (!scroller) return;

            const syncEnabled = s.paramEditorSyncTimeline;
            const zoom = resolveHorizontalWheelZoom({
                factor,
                basePxPerSec: pxPerSecRef.current,
                baseScrollLeft: syncEnabled
                    ? timelineViewportSync.get().scrollLeft
                    : scrollLeftRef.current,
                totalSec: projectSec,
                viewportWidth: scroller.clientWidth,
                playheadZoomEnabled: true,
                playheadSec: Number(playheadSec ?? 0) || 0,
                anchorScreenX: 0,
                minPxPerSec: resolveTimelineMinPxPerSec({
                    baseMinPxPerSec: MIN_PX_PER_SEC,
                    projectSec,
                    viewportWidthPx: scroller.clientWidth,
                }),
                maxPxPerSec: MAX_PX_PER_SEC,
            });
            if (!zoom) return;

            queueHorizontalZoom(zoom.nextPxPerSec, zoom.nextScrollLeft);
        }

        window.addEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
        return () =>
            window.removeEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
    }, [s.paramEditorSyncTimeline, queueHorizontalZoom]);

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
    }, [editParam, processorParams]);

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
    }, [editParam, processorParams]);

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
            const result = await paramsApi.setStaticParam(rootTrackId, paramId, value, true);
            if (result.ok) {
                setProcessorStaticValues((prev) => ({
                    ...prev,
                    [paramId]: value,
                }));
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
        return getVisibleSecondaryParamIds({
            editParam,
            processorParamIds: processorParamsRef.current.map((p) => p.id as ParamName),
            secondaryParamVisible,
        });
    }, [editParam, processorParams, secondaryParamVisible]);

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
    const gridBoundaryRef = useRef<HTMLDivElement | null>(null);

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
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // The ruler is React-rendered, but the main graph is canvas-rendered.
    // Ensure playhead changes (seek / playback) trigger a redraw.
    useEffect(() => {
        invalidate();
    }, [s.playheadSec, invalidate]);

    const isTransportAdvancing = s.runtime.isPlaying && s.runtime.playbackPositionSec > 1e-4;

    useVisualPlayhead({
        syncedPlayheadSec: s.playheadSec,
        isTransportAdvancing,
        onFrame: useCallback(
            (visualPlayheadSec: number) => {
                visualPlayheadSecRef.current = visualPlayheadSec;
                const playheadLeftPx = visualPlayheadSec * pxPerSecRef.current;
                if (rulerPlayheadLineRef.current) {
                    rulerPlayheadLineRef.current.style.left = `${playheadLeftPx}px`;
                }
                if (rulerPlayheadHeadRef.current) {
                    rulerPlayheadHeadRef.current.style.left = `${playheadLeftPx}px`;
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
        };
    }, []);

    function applyScrollLayers(next: number) {
        if (rulerContentRef.current) {
            rulerContentRef.current.style.transform = `translateX(${-next}px)`;
        }

        if (gridLayerRef.current) {
            invokeGridRedrawHandler(gridLayerRef.current, next);
        }

        if (gridBoundaryRef.current) {
            const left = contentWidth - 1 - next;
            gridBoundaryRef.current.style.left = `${left}px`;
            gridBoundaryRef.current.style.opacity =
                left >= -2 && left <= viewSizeRef.current.w + 2 ? "0.9" : "0";
        }

        // 同步绘制：滚动事件在绘制前触发，画布必须与标尺/网格（上方已同步
        // 落地）在同一帧内提交，否则滚动中会出现画布与网格的分层漂移。
        drawRef.current();
    }

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
        applyScrollLayers(next);
        if (scrollStateRafRef.current == null) {
            scrollStateRafRef.current = requestAnimationFrame(() => {
                scrollStateRafRef.current = null;
                setScrollLeft(scrollLeftRef.current);
            });
        }
    }

    useLayoutEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;
        syncScrollLeft(el);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [contentWidth, s.grid, s.beats]);

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

    function syncVerticalScrollbarForViewport(param: ParamName, view: ValueViewport): void {
        const scroller = scrollerRef.current;
        if (!scroller) return;

        const clampedView = clampViewport(param, view);
        const bounds = getParamValueBoundsForScrollbar(param);
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

    const handleMidiImported = useCallback(
        (_result: { notes_imported: number; frames_touched: number }) => {
            refreshNow();
        },
        [refreshNow],
    );

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

    // 可见区域的 sec 范围（统一用 sec 坐标系）
    const visibleStartSec = scrollLeft / Math.max(1e-9, pxPerSec);
    const visibleEndSec = visibleStartSec + viewSize.w / Math.max(1e-9, pxPerSec);

    // Per-clip 波形 peaks（替代原来的 mix 波形）
    const clipPeaks = useClipsPeaksForPianoRoll({
        clips: trackClips,
        visibleStartSec,
        visibleEndSec,
        pxPerSec,
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

    // Keep draw function always up-to-date (invalidate() is stable and calls drawRef.current()).
    drawRef.current = () => {
        drawPianoRoll({
            axisCanvas: axisCanvasRef.current,
            canvas: canvasRef.current,
            viewSize: viewSizeRef.current,
            editParam,
            pitchView: pitchViewRef.current,
            paramViews: paramViewsRef.current,
            valueToY,
            clipPeaks,
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
            pxPerSec: pxPerSecRef.current,
            scrollLeft: scrollLeftRef.current,
            secPerBeat,
            // 画布每帧重绘（onFrame invalidate），播放头必须用插值的视觉值：
            // 用 Redux 提交值会让 60fps 的重绘画着同一个旧播放头（且与标尺
            // 的 DOM 插值播放头节奏不一致、短暂错位）。
            playheadSec: visualPlayheadSecRef.current,
            waveformColors,
            referencePitchOverlays,
            detectedPitchCurves,
            isDark: themeMode === "dark",
            fontFamily,
            clipboardPreview: s.showClipboardPreview ? clipboardRef.current : null,
            // pitch snap visual helpers
            pitchSnapUnit: s.pitchSnapUnit,
            projectScale: effectiveProjectScale,
            scaleHighlightMode: s.scaleHighlightMode,
            scaleSegments: buildScaleSegments(
                s.tempoMap,
                effectiveProjectScale,
                Math.max(0, scrollLeftRef.current / Math.max(1e-9, pxPerSecRef.current) - 5),
                (scrollLeftRef.current + viewSizeRef.current.w) /
                    Math.max(1e-9, pxPerSecRef.current) +
                    5,
            ),
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
        selectedTrackId: effectiveSelectedTrackId,
        tracks: s.tracks,
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
        clipboardRef,
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
        pianoRollCopyKb,
        pianoRollPasteKb,
        prVerticalZoomKb,
        horizontalZoomKb,
        scrollHorizontalKb,
        scrollVerticalKb,
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
        playheadSec: s.playheadSec,
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
                    syncScrollLeft(scroller);
                }
                applyVerticalPanDelta(e.deltaY);
                return;
            }

            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                const scroller = scrollerRef.current;
                if (!scroller) return;
                scroller.scrollLeft += horizontalDelta;
                syncScrollLeft(scroller);
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

    // Silence unused state warnings; selectionUi is future UI.
    void selectionUi;

    useEffect(() => {
        setCanvasCursor(s.toolMode === "select" ? "default" : "crosshair");
    }, [s.toolMode]);

    useEffect(() => {
        setCtxMenu(null);
    }, [s.toolMode]);

    // 切换工具时清除选区
    useEffect(() => {
        selectionRef.current = null;
        setSelectionUi(null);
        invalidate();
    }, [s.toolMode]);

    // 同步 isLoading 和 asyncRefresh 状态到全局 Context
    useEffect(() => {
        updatePianoRollStatus({
            dataLoading: isLoading,
            asyncRefreshActive: asyncRefresh.isLoading,
            asyncRefreshProgress: asyncRefresh.progress,
            asyncRefreshStatus: asyncRefresh.status,
            asyncRefreshError: asyncRefresh.error,
        });
    }, [
        isLoading,
        asyncRefresh.isLoading,
        asyncRefresh.progress,
        asyncRefresh.status,
        asyncRefresh.error,
        updatePianoRollStatus,
    ]);

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

            const applySelectionEditWithEdgeSmoothing = async (
                editSelection: (currentSelectionVals: number[]) => number[],
                smoothnessInput?: number,
            ) => {
                const smoothness = clamp(
                    Number(
                        smoothnessInput ??
                            (data?.edgeSmoothnessPercent as number | undefined) ??
                            s.edgeSmoothnessPercent,
                    ) || 0,
                    0,
                    100,
                );

                const maxTransitionFrames = Math.floor(frameCount / 2);
                const transitionFrames =
                    smoothness > 0 && maxTransitionFrames > 0
                        ? Math.round((smoothness / 100) * maxTransitionFrames)
                        : 0;
                const halfSpan = transitionFrames > 0 ? transitionFrames / 2 : 0;
                const extend = Math.max(0, Math.ceil(halfSpan));

                const extStart = Math.max(0, startFrame - extend);
                const extCount = frameCount + Math.max(0, startFrame - extStart) + extend;
                const selOffset = startFrame - extStart;

                const res = await paramsApi.getParamFrames(
                    rootTrackId,
                    editParam,
                    extStart,
                    extCount,
                    1,
                );
                if (!res?.ok) return;

                const payload = res as ParamFramesPayload;
                const beforeDense = (payload.edit ?? []).map((v) => Number(v) || 0);
                if (beforeDense.length <= 0) return;

                const selEnd = Math.min(beforeDense.length - 1, selOffset + frameCount - 1);
                if (selOffset < 0 || selOffset >= beforeDense.length || selEnd < selOffset) {
                    return;
                }
                const actualSelLen = selEnd - selOffset + 1;
                const currentSel = beforeDense.slice(selOffset, selOffset + actualSelLen);
                const nextSel = editSelection(currentSel);

                const editedDense = beforeDense.slice();
                for (let i = 0; i < actualSelLen; i += 1) {
                    editedDense[selOffset + i] = Number(nextSel[i] ?? currentSel[i] ?? 0) || 0;
                }

                if (smoothness > 0 && transitionFrames > 0) {
                    const calcMean = (arr: number[]) => {
                        let sum = 0;
                        let count = 0;
                        for (let i = 0; i < actualSelLen; i += 1) {
                            const v = Number(arr[selOffset + i] ?? 0);
                            if (editParam === "pitch" && v === 0) continue;
                            sum += v;
                            count += 1;
                        }
                        return { sum, count };
                    };

                    const beforeMean = calcMean(beforeDense);
                    const afterMean = calcMean(editedDense);
                    const meanDelta =
                        beforeMean.count > 0 && afterMean.count > 0
                            ? Math.abs(
                                  afterMean.sum / afterMean.count -
                                      beforeMean.sum / beforeMean.count,
                              )
                            : 0;

                    let boundaryDelta = 0;
                    let boundaryCount = 0;
                    if (selOffset > 0) {
                        boundaryDelta += Math.abs(
                            Number(beforeDense[selOffset] ?? 0) -
                                Number(beforeDense[selOffset - 1] ?? 0),
                        );
                        boundaryCount += 1;
                    }
                    if (selEnd < beforeDense.length - 1) {
                        boundaryDelta += Math.abs(
                            Number(beforeDense[selEnd] ?? 0) - Number(beforeDense[selEnd + 1] ?? 0),
                        );
                        boundaryCount += 1;
                    }
                    const boundaryMean = boundaryCount > 0 ? boundaryDelta / boundaryCount : 0;
                    const changeFactor = clamp(meanDelta / (meanDelta + boundaryMean + 1e-6), 0, 1);

                    if (changeFactor > 0) {
                        const snapshot = editedDense.slice();
                        const span = Math.max(1e-9, 2 * halfSpan);

                        if (selOffset > 0) {
                            const left = Math.max(0, Math.floor(selOffset - halfSpan));
                            const right = Math.min(
                                editedDense.length - 1,
                                Math.ceil(selOffset + halfSpan),
                            );
                            for (let idx = left; idx <= right; idx += 1) {
                                const t = clamp((idx - (selOffset - halfSpan)) / span, 0, 1);
                                const outsideIdx = Math.min(selOffset - 1, idx);
                                const insideIdx = Math.max(selOffset, idx);
                                const outsideVal = snapshot[outsideIdx] ?? editedDense[idx];
                                const insideVal = snapshot[insideIdx] ?? editedDense[idx];
                                const smoothed = outsideVal + (insideVal - outsideVal) * t;
                                editedDense[idx] =
                                    snapshot[idx] + (smoothed - snapshot[idx]) * changeFactor;
                            }
                        }

                        if (selEnd < editedDense.length - 1) {
                            const left = Math.max(0, Math.floor(selEnd - halfSpan));
                            const right = Math.min(
                                editedDense.length - 1,
                                Math.ceil(selEnd + halfSpan),
                            );
                            for (let idx = left; idx <= right; idx += 1) {
                                const t = clamp((idx - (selEnd - halfSpan)) / span, 0, 1);
                                const insideIdx = Math.min(selEnd, idx);
                                const outsideIdx = Math.max(selEnd + 1, idx);
                                const insideVal = snapshot[insideIdx] ?? editedDense[idx];
                                const outsideVal = snapshot[outsideIdx] ?? editedDense[idx];
                                const smoothed = insideVal + (outsideVal - insideVal) * t;
                                editedDense[idx] =
                                    snapshot[idx] + (smoothed - snapshot[idx]) * changeFactor;
                            }
                        }
                    }
                }

                await paramsApi.setParamFrames(rootTrackId, editParam, extStart, editedDense, true);
                bumpRefreshToken();
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
                    await applySelectionEditWithEdgeSmoothing(
                        (vals) =>
                            editParam === "pitch"
                                ? vals.map((v) => (v === 0 ? 0 : v + delta))
                                : vals.map((v) => v + delta),
                        Number(data?.edgeSmoothnessPercent),
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
                    await applySelectionEditWithEdgeSmoothing((vals) => {
                        const fpMs = Number(paramView?.framePeriodMs ?? fp) || fp;
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
                    }, Number(data?.edgeSmoothnessPercent));
                    break;
                }
                case "setPitch": {
                    const parsed = Number(data?.value ?? data?.midiNote);
                    const midiNote = Number.isFinite(parsed) ? parsed : 60;
                    await applySelectionEditWithEdgeSmoothing(
                        (vals) =>
                            editParam === "pitch"
                                ? vals.map((v) => (v === 0 ? 0 : midiNote))
                                : vals.map(() => midiNote),
                        Number(data?.edgeSmoothnessPercent),
                    );
                    break;
                }
                case "shiftParamUpSelection":
                case "shiftParamDownSelection": {
                    const descriptor = processorParamsRef.current.find(
                        (param) => param.id === editParam,
                    );
                    const step = getParamShiftStep(editParam, descriptor);
                    const delta = op === "shiftParamUpSelection" ? step : -step;
                    await applySelectionEditWithEdgeSmoothing(
                        (vals) => vals.map((v) => v + delta),
                        Number(data?.edgeSmoothnessPercent),
                    );
                    break;
                }
                case "smooth": {
                    const strength = clamp((Number(data?.strength ?? 50) || 0) / 100, 0, 1);
                    if (strength <= 0) return;
                    const res = await paramsApi.getParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        frameCount,
                        1,
                    );
                    if (!res?.ok) return;
                    const payload = res as ParamFramesPayload;
                    const vals = (payload.edit ?? []).map((v) => Number(v));
                    const result = smoothSelectionValues(vals, editParam, strength);
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
                        const quantized = vals.map((v) => {
                            const stepCount = Math.round((v - defaultValue) / quantizeUnit);
                            const snapped = defaultValue + stepCount * quantizeUnit;
                            if (Math.abs(v - snapped) <= tolerance) return v;
                            return snapped + (v > snapped ? 1 : -1) * tolerance;
                        });
                        await paramsApi.setParamFrames(
                            rootTrackId,
                            editParam,
                            startFrame,
                            quantized,
                            true,
                        );
                        bumpRefreshToken();
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
                    const scaleAt = (i: number): ScaleLike =>
                        fixedScale ?? projectScaleAtSec(((startFrame + i) * fpMs) / 1000) ?? "C";
                    const quantized =
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
                              );
                    await paramsApi.setParamFrames(
                        rootTrackId,
                        editParam,
                        startFrame,
                        quantized,
                        true,
                    );
                    bumpRefreshToken();
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
                        const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                        const stepCount = Math.round((avg - defaultValue) / quantizeUnit);
                        const quantizedAvg = defaultValue + stepCount * quantizeUnit;
                        const delta = quantizedAvg - avg;
                        const result = vals.map((v) => {
                            const moved = v + delta;
                            if (Math.abs(moved - v) <= tolerance) return v;
                            return moved + (v > moved ? 1 : -1) * tolerance;
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
                    // pitch=0 视为未编辑，不参与均值
                    const nonZero = editParam === "pitch" ? vals.filter((v) => v !== 0) : vals;
                    if (nonZero.length === 0) return;
                    const avg = nonZero.reduce((a, b) => a + b, 0) / nonZero.length;
                    const midScale =
                        fixedScale ??
                        projectScaleAtSec(
                            ((startFrame + Math.floor(vals.length / 2)) *
                                (Number(payload.frame_period_ms ?? fp) || fp)) /
                                1000,
                        ) ??
                        "C";
                    const quantizedAvg =
                        unit === "semitone" ? snapToSemitone(avg) : snapToScale(avg, midScale);
                    const delta = quantizedAvg - avg;
                    const result =
                        editParam === "pitch"
                            ? vals.map((v) => {
                                  if (v === 0) return 0;
                                  const moved = v + delta;
                                  return Math.abs(moved - v) <= toleranceSemitone
                                      ? v
                                      : moved + (v - moved > 0 ? 1 : -1) * toleranceSemitone;
                              })
                            : vals.map((v) => {
                                  const moved = v + delta;
                                  return Math.abs(moved - v) <= toleranceSemitone
                                      ? v
                                      : moved + (v - moved > 0 ? 1 : -1) * toleranceSemitone;
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
            }
        },
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
        ],
    );

    // Keep the ref in sync so usePianoRollInteractions can dispatch edit ops
    handleEditActionRef.current = (op: string) => void handleEditOp(op);

    // Listen for edit operations dispatched from MenuBar
    useEffect(() => {
        const handler = (e: Event) => {
            const detail = (e as CustomEvent).detail;
            if (!detail?.op) return;
            const { op, ...data } = detail;
            const active = document.activeElement as HTMLElement | null;
            const inPianoRoll =
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]") ||
                document.body.getAttribute("data-hs-focus-window") === "pianoRoll";
            const inTrackHeader =
                Boolean(active?.closest?.("[data-track-list-panel]")) ||
                document.body.getAttribute("data-hs-focus-window") === "trackHeader";

            if (op === "paste" && !inPianoRoll && !inTrackHeader) {
                return;
            }
            if (op === "selectAll" || op === "deselect") {
                if (!inPianoRoll || s.toolMode !== "select") {
                    return;
                }
            }
            void handleEditOp(op, data);
        };
        window.addEventListener("hifi:editOp", handler);
        return () => window.removeEventListener("hifi:editOp", handler);
    }, [handleEditOp, s.toolMode]);

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

    const timeRulerTicks = useMemo(
        () =>
            buildRulerTicks({
                pxPerSec,
                scrollLeft,
                viewportWidth: viewSize.w,
                projectSec: dynamicProjectSec,
                bpm: s.bpm,
                beatsPerBar: Math.max(1, Math.round(s.beats || 4)),
                grid: s.grid,
                primaryUnit: s.primaryTimeUnit,
                secondaryUnit: s.secondaryTimeUnit,
                minLabelSpacingPx: s.rulerLabelSpacingPx,
                tempoMap: s.tempoMap,
            }),
        [
            pxPerSec,
            scrollLeft,
            viewSize.w,
            dynamicProjectSec,
            s.bpm,
            s.beats,
            s.grid,
            s.primaryTimeUnit,
            s.secondaryTimeUnit,
            s.rulerLabelSpacingPx,
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

    // ── Tempo Map 显式网格线（参数编辑器背景网格）─────────────
    const tempoGridLineXs = useMemo(
        () =>
            buildTempoGridLineXsForViewport({
                tempoMap: s.tempoMap,
                scrollLeft,
                viewportWidth: viewSize.w,
                pxPerSec,
                projectSec: dynamicProjectSec,
                stepBeats: gridStepBeats(s.grid),
                fallbackBpm: s.bpm,
                fallbackBeatsPerBar: Math.max(1, Math.round(s.beats || 4)),
                swingPercent: s.timelineSnap.swingEnabled ? s.timelineSnap.swingPercent : 0,
                minSpacingPx: s.timelineSnap.gridMinSpacingPx,
            }),
        [
            s.tempoMap,
            s.bpm,
            s.beats,
            s.grid,
            s.timelineSnap,
            scrollLeft,
            viewSize.w,
            pxPerSec,
            dynamicProjectSec,
        ],
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
                        color="gray"
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
                            color="gray"
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
                                color="gray"
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
                                color="gray"
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
                            color="gray"
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
                            color="gray"
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
                                type="range"
                                min={0}
                                max={100}
                                step={1}
                                value={Math.round(s.edgeSmoothnessPercent)}
                                onWheel={(e) => {
                                    e.preventDefault();
                                    const fine = isModifierActive(paramFineAdjustKb, e.nativeEvent);
                                    const step = fine ? 1 : 5;
                                    const dir = e.deltaY < 0 ? 1 : -1;
                                    const next = clamp(
                                        Math.round(s.edgeSmoothnessPercent) + dir * step,
                                        0,
                                        100,
                                    );
                                    dispatch(setEdgeSmoothnessPercent(next));
                                    void dispatch(persistUiSettings());
                                }}
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
                                        variant="soft"
                                        color="blue"
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
                                    accent="grass"
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
                                accent="grass"
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
                                    accent="amber"
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
            {asyncRefresh.isLoading && (
                <Flex className="px-3 py-2 bg-qt-base border-b border-qt-border">
                    <ProgressBar
                        percentage={asyncRefresh.progress}
                        label={tAny("refreshing_pitch_data") || "Refreshing pitch data"}
                        showCancel={true}
                        onCancel={async () => {
                            // Task 6.6: 取消按钮点击时调 ?cancelRefresh()
                            await asyncRefresh.cancelRefresh();
                        }}
                        estimatedRemaining={asyncRefresh.estimatedRemaining}
                    />
                </Flex>
            )}

            {/* Task 6.7: 任务完成后显示成功提 ?*/}
            {showSuccessMessage && (
                <Flex
                    align="center"
                    gap="2"
                    className="px-3 py-2 bg-green-900/20 border-b border-green-700 text-green-300 text-sm"
                >
                    <span>&#x2713;</span>
                    <span></span>
                </Flex>
            )}

            {/* Task 6.8: 任务失败时显示错误消息和重试按钮 */}
            {asyncRefresh.status === "failed" && asyncRefresh.error && (
                <Flex
                    align="center"
                    justify="between"
                    className="px-3 py-2 bg-red-900/20 border-b border-red-700 text-red-300 text-sm"
                >
                    <span></span>
                    <Button
                        size="1"
                        variant="soft"
                        color="red"
                        onClick={() => rootTrackId && void asyncRefresh.startRefresh(rootTrackId)}
                    >
                        {tAny("retry") || "Retry"}
                    </Button>
                </Flex>
            )}

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
                        <canvas ref={axisCanvasRef} className="absolute inset-0" />
                    </div>
                </Flex>

                {/* Right: ruler + scrollable canvas */}
                <Flex direction="column" className="flex-1 min-w-0 select-none">
                    <TimeRuler
                        contentWidth={contentWidth}
                        scrollLeft={scrollLeft}
                        ticks={timeRulerTicks}
                        pxPerBeat={pxPerBeat}
                        pxPerSec={pxPerSec}
                        secPerBeat={secPerBeat}
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
                            document.body.setAttribute("data-hs-focus-window", "pianoRoll");
                            interactions.onRulerMouseDown(e);
                        }}
                    />

                    <div
                        ref={scrollerRef}
                        className="flex-1 bg-qt-graph-bg overflow-x-scroll overflow-y-scroll relative custom-scrollbar outline-none focus:outline-none focus-visible:outline-none"
                        data-piano-roll-scroller
                        tabIndex={0}
                        onFocus={() => {
                            document.body.setAttribute("data-hs-focus-window", "pianoRoll");
                        }}
                        onMouseDownCapture={(e) => {
                            document.body.setAttribute("data-hs-focus-window", "pianoRoll");
                            interactions.onScrollerMouseDownCapture(e);
                        }}
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
                                    boundaryRef={gridBoundaryRef}
                                    weakLineXs={tempoGridLineXs?.weak ?? null}
                                    strongLineXs={tempoGridLineXs?.strong ?? null}
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

                                <canvas
                                    ref={canvasRef}
                                    className="absolute inset-0"
                                    style={{ cursor: canvasCursor }}
                                    onPointerMove={interactions.onCanvasPointerMove}
                                    onPointerLeave={interactions.onCanvasPointerLeave}
                                    onPointerDown={interactions.onCanvasPointerDown}
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
