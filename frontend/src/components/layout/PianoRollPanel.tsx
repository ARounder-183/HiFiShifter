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
import { Flex, Text, Button, Select, Box, IconButton } from "@radix-ui/themes";
import {
    CursorArrowIcon,
    EyeOpenIcon,
    EyeClosedIcon,
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
    setScaleHighlightMode,
    toggleClipboardPreview,
    toggleParamValuePopup,
    toggleLockParamLines,
    cycleDragDirection,
    setToolMode,
    persistUiSettings,
} from "../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../features/session/trackUtils";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { getWaveformColors } from "../../theme/waveformColors";
import type { ProcessorParamDescriptor } from "../../types/api";
import { paramsApi } from "../../services/api/params";
import type { ParamFramesPayload } from "../../types/api";
import {
    degreeInputToScaleSteps,
    isScaleKey,
    snapToScale,
    snapToSemitone,
    transposePitchByScaleSteps,
} from "../../utils/musicalScales";
import { computeAnchoredHorizontalZoom } from "../../utils/horizontalZoom";
import { isModifierActive, isNoneBinding } from "../../features/keybindings/keybindingsSlice";
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
    gridStepBeats,
} from "./timeline";

import { AXIS_W, PITCH_MAX_MIDI, PITCH_MIN_MIDI } from "./pianoRoll/constants";
import { drawPianoRoll } from "./pianoRoll/render";
import type { DetectedPitchCurve } from "./pianoRoll/render";
import { averageSelectionValues, smoothSelectionValues } from "./pianoRoll/selectionTransforms";
import { usePianoRollData } from "./pianoRoll/usePianoRollData";
import { useClipsPeaksForPianoRoll } from "./pianoRoll/useClipsPeaksForPianoRoll";
import { usePianoRollInteractions } from "./pianoRoll/usePianoRollInteractions";
import { useLiveParamEditing } from "./pianoRoll/useLiveParamEditing";
import { getParamShiftStep } from "./pianoRoll/paramShiftStep";
import {
    buildChildPitchOffsetCentsParam,
    buildChildPitchOffsetDegreesParam,
    childPitchOffsetValueToDisplay,
    CHILD_PITCH_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_DEGREES_RANGE,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
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
import { coreApi } from "../../services/api/core";
import { EditContextMenu } from "../editDialogs/EditContextMenu";
import { getDynamicProjectSec } from "../../features/session/projectBoundary";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { parseCustomScaleToken } from "../../utils/scaleSelection";

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

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
    const resolveScaleFromToken = useCallback(
        (scaleToken: string): ScaleLike => {
            if (scaleToken === "__project__") {
                return effectiveProjectScale;
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
        [effectiveProjectScale, s.customScalePresets],
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
    const mergedKeybindings = useAppSelector(selectMergedKeybindings);
    // 是否按住切换吸附的修饰键（临时切换吸附时用于高亮显示）
    const [snapToggleHeld, setSnapToggleHeld] = useState(false);
    // 仅在参数编辑实际操作期间（选择拖拽/绘制）参与临时吸附视觉切换
    const [snapGestureActive, setSnapGestureActive] = useState(false);

    useEffect(() => {
        const kb = mergedKeybindings["modifier.clipNoSnap"];
        if (!kb) return;
        const onKey = (e: KeyboardEvent) => {
            const active = isModifierActive(kb, e as any);
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
    const { mode: themeMode } = useAppTheme();
    const waveformColors = useMemo(() => getWaveformColors(themeMode, "piano-roll"), [themeMode]);

    const effectivePitchSnapVisual =
        snapGestureActive && snapToggleHeld ? !s.pitchSnapEnabled : s.pitchSnapEnabled;

    // Task 6.3: 集成 useAsyncPitchRefresh Hook
    const asyncRefresh = useAsyncPitchRefresh();
    const [showSuccessMessage] = useState(false);

    // MIDI 导入弹窗状态
    const [midiDialogOpen, setMidiDialogOpen] = useState(false);
    const [midiPath, setMidiPath] = useState<string | null>(null);
    // 记录打开弹窗时的选区（拍数），用于后续计算帧偏移
    const [midiDialogSelection, setMidiDialogSelection] = useState<{
        aBeat: number;
        bBeat: number;
    } | null>(null);

    // 右键编辑菜单状态
    const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);
    const [drawToolMenuOpen, setDrawToolMenuOpen] = useState(false);
    const drawToolMenuRef = useRef<HTMLDivElement | null>(null);
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
        if (!drawToolMenuOpen) return;
        const onPointerDown = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (drawToolMenuRef.current?.contains(target)) return;
            setDrawToolMenuOpen(false);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setDrawToolMenuOpen(false);
            }
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [drawToolMenuOpen]);

    const handleOpenMidiDialog = useCallback(async () => {
        try {
            const res = await coreApi.openMidiDialog();
            if (res.ok && !res.canceled && res.path) {
                // 快照当前选区（拍为单位）
                const sel = selectionRef.current;
                setMidiDialogSelection(sel ? { ...sel } : null);
                setMidiPath(res.path);
                setMidiDialogOpen(true);
            }
        } catch {
            // 静默忽略
        }
    }, []);

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
    const keyboardZoomPendingRef = useRef<{
        nextScale: number;
        nextScrollLeft: number;
    } | null>(null);

    // BPM 变化时，按比例调 ?scrollLeft，保持视口中心点的秒数不 ?
    // scrollLeft_new = scrollLeft_old × (bpm_old / bpm_new)
    const prevBpmRef = useRef(s.bpm);
    useEffect(() => {
        const prevBpm = prevBpmRef.current;
        prevBpmRef.current = s.bpm;
        if (Math.abs(prevBpm - s.bpm) < 1e-9) return;
        const ratio = prevBpm / Math.max(1e-6, s.bpm);
        const newScrollLeft = scrollLeftRef.current * ratio;
        scrollLeftRef.current = newScrollLeft;
        setScrollLeft(newScrollLeft);
    }, [s.bpm]);

    useEffect(() => {
        scrollLeftRef.current = scrollLeft;
    }, [scrollLeft]);

    useEffect(() => {
        pxPerBeatRef.current = pxPerBeat;
        pxPerSecRef.current = pxPerSec;
        const timer = setTimeout(() => {
            localStorage.setItem("hifishifter.paramPxPerSec", String(pxPerSec));
        }, 500);
        return () => clearTimeout(timer);
    }, [pxPerBeat, pxPerSec]);

    useLayoutEffect(() => {
        const pending = keyboardZoomPendingRef.current;
        if (!pending) return;
        if (Math.abs(pending.nextScale - pxPerSec) > 1e-9) return;
        const scroller = scrollerRef.current;
        if (!scroller) return;

        keyboardZoomPendingRef.current = null;
        scroller.scrollLeft = pending.nextScrollLeft;
        syncScrollLeft(scroller);
    }, [pxPerSec]);

    const zoomTimelineStateRef = useRef({
        playheadSec: s.playheadSec,
        projectSec: s.projectSec,
    });
    useLayoutEffect(() => {
        zoomTimelineStateRef.current = {
            playheadSec: s.playheadSec,
            projectSec: s.projectSec,
        };
    });

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

            const zoom = computeAnchoredHorizontalZoom({
                currentScale: pxPerSecRef.current,
                factor,
                minScale: MIN_PX_PER_SEC,
                maxScale: MAX_PX_PER_SEC,
                scrollLeft: scroller.scrollLeft,
                viewportWidth: scroller.clientWidth,
                anchorSec: Number(playheadSec ?? 0) || 0,
                contentSec: projectSec,
            });
            if (!zoom) return;

            keyboardZoomPendingRef.current = {
                nextScale: zoom.nextScale,
                nextScrollLeft: zoom.nextScrollLeft,
            };
            setPxPerSec(zoom.nextScale);
        }

        window.addEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
        return () =>
            window.removeEventListener("hifi:zoomTimelineFocus", onZoomFocused as EventListener);
    }, []); // 空依赖

    const setPxPerBeatImmediate = useCallback(
        (next: number) => {
            // next 是新的 pxPerBeat，需要反推回 pxPerSec
            const nextPxPerSec = next / (60 / Math.max(1e-6, s.bpm));
            pxPerBeatRef.current = next;
            pxPerSecRef.current = nextPxPerSec;
            setPxPerSec(nextPxPerSec);
        },
        [s.bpm, setPxPerSec],
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
            invalidate(); // 绕过 React 渲染，直接命令 Canvas 重绘
        },
        [invalidate],
    );

    const paramViewsRef = useRef<Record<string, ValueViewport>>({});
    const setParamViewport = useCallback(
        (param: string, next: ValueViewport) => {
            paramViewsRef.current = { ...paramViewsRef.current, [param]: next };
            invalidate(); // 绕过 React 渲染，直接命令 Canvas 重绘
        },
        [invalidate],
    );

    const rootTrackId = useMemo(() => {
        return resolveRootTrackId(s.tracks, effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, s.tracks]);

    const rootTrack = useMemo(() => {
        if (!rootTrackId) return null;
        return s.tracks.find((tr) => tr.id === rootTrackId) ?? null;
    }, [s.tracks, rootTrackId]);

    // 声码器参数描述符（由 algo 动态定制面板）
    const [processorParams, setProcessorParams] = useState<ProcessorParamDescriptor[]>([]);
    const processorParamsRef = useRef<ProcessorParamDescriptor[]>([]);
    const [processorStaticParams, setProcessorStaticParams] = useState<ProcessorParamDescriptor[]>(
        [],
    );
    const [processorStaticValues, setProcessorStaticValues] = useState<Record<string, number>>({});
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
        if (isChildPitchOffsetCentsParam(editParam) || isChildPitchOffsetDegreesParam(editParam)) {
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
            : CHILD_PITCH_OFFSET_DEGREES_RANGE;
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
                    return t("hifigan_volume_label");
                case "volume":
                    return t("vslib_volume_label");
                case "synth_mode":
                    return t("vslib_synth_mode_label");
                case "pan":
                    return t("vslib_pan_label");
                case "breathiness":
                    return t("vslib_breathiness_label");
                default:
                    return param.display_name;
            }
        },
        [t],
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

    const dynamicProjectSec = useMemo(() => getDynamicProjectSec(s.clips), [s.clips]);

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

    function positiveMod(value: number, mod: number): number {
        if (!Number.isFinite(value) || !Number.isFinite(mod) || mod <= 0) return 0;
        const r = value % mod;
        return (r + mod) % mod;
    }

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
                if (s.autoScrollEnabled && s.runtime.isPlaying) {
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
            [contentWidth, invalidate, s.autoScrollEnabled, s.runtime.isPlaying],
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

    function syncScrollLeft(scroller: HTMLDivElement) {
        const next = scroller.scrollLeft;
        if (lastScrollLeftRef.current != null && lastScrollLeftRef.current === next) {
            return;
        }
        lastScrollLeftRef.current = next;
        scrollLeftRef.current = next;

        if (rulerContentRef.current) {
            rulerContentRef.current.style.transform = `translateX(${-next}px)`;
        }

        if (gridLayerRef.current) {
            const weakStepPx = Math.max(1e-6, pxPerBeatRef.current * gridStepBeats(s.grid));
            const barStepPx = Math.max(
                1e-6,
                pxPerBeatRef.current * Math.max(1, Math.round(s.beats || 4)),
            );
            const weakOffsetPx = -positiveMod(next, weakStepPx);
            const barOffsetPx = -positiveMod(next, barStepPx);
            gridLayerRef.current.style.backgroundPosition = `${weakOffsetPx}px 0px, ${barOffsetPx}px 0px`;
        }

        if (gridBoundaryRef.current) {
            const left = contentWidth - 1 - next;
            gridBoundaryRef.current.style.left = `${left}px`;
            gridBoundaryRef.current.style.opacity =
                left >= -2 && left <= viewSizeRef.current.w + 2 ? "0.9" : "0";
        }

        if (scrollStateRafRef.current == null) {
            scrollStateRafRef.current = requestAnimationFrame(() => {
                scrollStateRafRef.current = null;
                setScrollLeft(scrollLeftRef.current);
            });
        }

        invalidate();
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

    const selectionRef = useRef<{ aBeat: number; bBeat: number } | null>(null);
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
        bumpRefreshToken,
        refreshNow,
        refreshSecondaryNow,
        notifyLiveEditEnded,
        isLoading,
    } = usePianoRollData({
        editParam,
        secondaryParamIds: visibleSecondaryParamIds,
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
        if (visibleSecondaryParamIds.length > 0) {
            void refreshSecondaryNowRef.current();
            return;
        }
        invalidate();
    }, [visibleSecondaryParamIds, invalidate, rootTrackId]);

    const handleMidiImported = useCallback(
        (_result: { notes_imported: number; frames_touched: number }) => {
            // 导入完成后刷新参数面板
            refreshNow();
        },
        [refreshNow],
    );

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

    // 检测音高曲线更新时触发重绘
    useEffect(() => {
        invalidate();
    }, [detectedPitchCurves, invalidate]);

    // Ensure pitch-snap related changes immediately redraw
    useEffect(() => {
        invalidate();
    }, [
        s.pitchSnapEnabled,
        s.pitchSnapUnit,
        effectiveProjectScale,
        s.scaleHighlightMode,
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
            playheadSec: s.playheadSec,
            waveformColors,
            detectedPitchCurves,
            isDark: themeMode === "dark",
            clipboardPreview: s.showClipboardPreview ? clipboardRef.current : null,
            // pitch snap visual helpers
            pitchSnapUnit: s.pitchSnapUnit,
            projectScale: effectiveProjectScale,
            scaleHighlightMode: s.scaleHighlightMode,
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
        bpm: s.bpm,
        dynamicProjectSec,
        scrollLeftRef,
        pxPerBeatRef,
        setPxPerBeat: setPxPerBeatImmediate,
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

    // Auto-scroll: keep playhead visible in parameter editor during playback
    useEffect(() => {
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
    }, [s.autoScrollEnabled, s.runtime.isPlaying, s.playheadSec, pxPerSec, contentWidth]);

    // Piano keys (axis) area: keep touchpad wheel behavior aligned with the main editor.
    useEffect(() => {
        const el = axisWrapRef.current;
        if (!el) return;

        const handler = (e: WheelEvent) => {
            const noModifierPressed = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
            const isWheelBindingRequested = (kb: Keybinding) => {
                if (isNoneBinding(kb)) return noModifierPressed;
                return isModifierActive(kb, e as any);
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

            if (wheelAction === "horizontal-scroll") {
                e.preventDefault();
                const scroller = scrollerRef.current;
                if (!scroller) return;
                scroller.scrollLeft += horizontalScrollRequested ? e.deltaY : e.deltaX;
                syncScrollLeft(scroller);
                return;
            }

            if (wheelAction === "vertical-pan") {
                e.preventDefault();
                const delta = (-e.deltaY / h) * 0.5;
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

            // External clipboard paste ops – work with or without selection
            if (op === "pasteReaper" || op === "pasteVocalShifter") {
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
                if (op === "pasteReaper") {
                    void dispatch(pasteReaperClipboard(selArgs));
                } else {
                    void dispatch(
                        pasteVocalShifterClipboard({
                            ...selArgs,
                            activeParam: editParam,
                        }),
                    );
                }
                bumpRefreshToken();
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
                    if (!clip) return;

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
                            mode: targetParam.mode,
                            paramsApi,
                            pitchDeltaToDegreeSteps: pitchDeltaToDegreeSteps,
                            projectScale: effectiveProjectScale,
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
                    const scale: ScaleLike = resolveScaleFromToken(scaleToken);
                    const degreeSteps = degreeInputToScaleSteps(degrees);
                    if (degreeSteps === 0) return;
                    await applySelectionEditWithEdgeSmoothing(
                        (vals) =>
                            editParam === "pitch"
                                ? vals.map((midi) =>
                                      midi === 0
                                          ? 0
                                          : transposePitchByScaleSteps(midi, degreeSteps, scale),
                                  )
                                : vals.map((midi) =>
                                      transposePitchByScaleSteps(midi, degreeSteps, scale),
                                  ),
                        Number(data?.edgeSmoothnessPercent),
                    );
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
                    const scale: ScaleLike = resolveScaleFromToken(scaleToken);
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
                            : vals.map((v) =>
                                  editParam === "pitch" && v === 0
                                      ? 0
                                      : (() => {
                                            const snapped = snapToScale(v, scale);
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
                    const scale: ScaleLike = resolveScaleFromToken(scaleToken);
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
                    const quantizedAvg =
                        unit === "semitone" ? snapToSemitone(avg) : snapToScale(avg, scale);
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

    const currentDrawToolIcon = currentDrawTool === "vibrato" ? vibratoToolIcon : <Pencil1Icon />;

    const timeRulerBars = useMemo(() => {
        const beatsPerBar = Math.max(1, Math.round(s.beats || 4));
        const totalBeats = Math.max(1, Math.ceil(s.projectSec / secPerBeat));
        const result: Array<{ beat: number; label: string }> = [];
        let barIndex = 1;
        for (let beat = 0; beat <= totalBeats; beat += beatsPerBar) {
            result.push({ beat, label: `${barIndex}.1` });
            barIndex += 1;
        }
        return result;
    }, [s.beats, s.projectSec, secPerBeat]);

    return (
        <Flex direction="column" className="h-full w-full bg-qt-graph-bg border-t border-qt-border">
            {/* Header / Parameter Switch */}
            <Flex
                align="center"
                justify="between"
                className="h-8 bg-qt-base border-b border-qt-border px-2 shrink-0"
            >
                <Flex align="center" gap="2">
                    <Text size="1" weight="bold" color="gray">
                        {t("param_editor")}
                    </Text>
                    {/* 音高吸附和剪贴板预览按钮，紧邻 param_editor 右侧，留 8px 空白 */}
                    <Flex gap="1" align="center" style={{ marginLeft: 8 }}>
                        <IconButton
                            size="1"
                            variant={s.toolModeGroup === "select" ? "solid" : "ghost"}
                            color="gray"
                            title={t("select")}
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
                                title={drawToolButtonTitle}
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
                                            width: 0,
                                            height: 0,
                                            borderLeft: "4px solid transparent",
                                            borderTop: "4px solid currentColor",
                                            opacity: 0.7,
                                        }}
                                    />
                                </Box>
                            </IconButton>

                            {drawToolMenuOpen && (
                                <Box
                                    ref={drawToolMenuRef}
                                    data-hs-context-menu
                                    style={{
                                        position: "absolute",
                                        left: 0,
                                        top: "calc(100% + 4px)",
                                        minWidth: 190,
                                        padding: 4,
                                        borderRadius: 6,
                                        border: "1px solid var(--gray-6)",
                                        background: "var(--gray-2)",
                                        boxShadow: "0 8px 24px rgba(0,0,0,0.22)",
                                        zIndex: 30,
                                    }}
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
                                            <Flex
                                                key={item.mode}
                                                align="center"
                                                justify="between"
                                                px="2"
                                                py="1"
                                                style={{
                                                    cursor: "pointer",
                                                    borderRadius: 4,
                                                    background: active
                                                        ? "var(--accent-4)"
                                                        : "transparent",
                                                }}
                                                onClick={() => {
                                                    dispatch(setToolMode(item.mode));
                                                    setDrawToolMenuOpen(false);
                                                }}
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
                                            </Flex>
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
                            title={`${tAny("drag_direction")}: ${tAny(activeDragDirection === "free" ? "drag_direction_free" : activeDragDirection === "x-only" ? "drag_direction_x_only" : "drag_direction_y_only")}`}
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
                        <IconButton
                            size="1"
                            variant={effectivePitchSnapVisual ? "solid" : "ghost"}
                            color="gray"
                            title={`${t("pitch_snap")}: ${
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
                                setPitchSnapOpen(true);
                            }}
                        >
                            {!effectivePitchSnapVisual ? (
                                <svg
                                    width="15"
                                    height="15"
                                    viewBox="0 0 15 15"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path
                                        d="M3 12L12 3"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                    />
                                    <path
                                        d="M10 2V10.5C10 11.88 8.88 13 7.5 13C6.12 13 5 11.88 5 10.5C5 9.12 6.12 8 7.5 8"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        opacity="0.6"
                                    />
                                </svg>
                            ) : s.pitchSnapUnit === "semitone" ? (
                                <svg
                                    width="15"
                                    height="15"
                                    viewBox="0 0 15 15"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path
                                        d="M3 5.5H12M3 9.5H12"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                    />
                                    <circle
                                        cx="7.5"
                                        cy="7.5"
                                        r="4.2"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        opacity="0.7"
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
                                        d="M2.5 10.5L5.5 4.5L8.5 10.5L11.5 6"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                    <circle cx="2.5" cy="10.5" r="1" fill="currentColor" />
                                    <circle cx="5.5" cy="4.5" r="1" fill="currentColor" />
                                    <circle cx="8.5" cy="10.5" r="1" fill="currentColor" />
                                    <circle cx="11.5" cy="6" r="1" fill="currentColor" />
                                </svg>
                            )}
                        </IconButton>
                        <IconButton
                            size="1"
                            variant={s.scaleHighlightMode === "always" ? "solid" : "ghost"}
                            color="gray"
                            title={tAny("scale_highlight")}
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
                            variant={s.showClipboardPreview ? "solid" : "ghost"}
                            color="gray"
                            title={t("clipboard_preview")}
                            tabIndex={-1}
                            onClick={() => {
                                dispatch(toggleClipboardPreview());
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
                                    y="1"
                                    width="9"
                                    height="13"
                                    rx="1"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    fill="none"
                                />
                                <path
                                    d="M5.5 1V2.5H9.5V1"
                                    stroke="currentColor"
                                    strokeWidth="0.8"
                                />
                                <path
                                    d="M5 6L7 8L10 5"
                                    stroke="currentColor"
                                    strokeWidth="1.2"
                                    opacity="0.7"
                                />
                            </svg>
                        </IconButton>
                        <IconButton
                            size="1"
                            variant={s.showParamValuePopup ? "solid" : "ghost"}
                            color="gray"
                            title={t("param_value_popup")}
                            tabIndex={-1}
                            onClick={() => {
                                dispatch(toggleParamValuePopup());
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
                                <path
                                    d="M2.5 3.5H12.5V10.5H6.2L3.2 13.5V10.5H2.5V3.5Z"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    fill="none"
                                />
                                <path
                                    d="M5 6H10"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    strokeLinecap="round"
                                />
                                <path
                                    d="M5 8H8.8"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    strokeLinecap="round"
                                />
                            </svg>
                        </IconButton>
                        <IconButton
                            size="1"
                            variant={s.lockParamLinesEnabled ? "solid" : "ghost"}
                            color="gray"
                            title={t("lock_param_lines")}
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
                        <Flex align="center" gap="1" ml="2">
                            <Text size="1">{tAny("edge_smoothness")}:</Text>
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
                                style={{ width: 120 }}
                            />
                            <Text size="1" style={{ minWidth: 36, textAlign: "right" }}>
                                {Math.round(s.edgeSmoothnessPercent)}%
                            </Text>
                        </Flex>
                    </Flex>
                </Flex>

                {/* Pitch Snap 设置弹窗 */}
                <PitchSnapSettingsDialog open={pitchSnapOpen} onOpenChange={setPitchSnapOpen} />

                <Flex gap="2" align="center">
                    <Flex gap="1" align="center">
                        {selectedIsChildTrack && childPitchOffsetCentsParam ? (
                            <Button
                                size="1"
                                variant={
                                    editParam === childPitchOffsetCentsParam ? "solid" : "soft"
                                }
                                color={editParam === childPitchOffsetCentsParam ? "cyan" : "gray"}
                                onClick={() => dispatch(setEditParam(childPitchOffsetCentsParam))}
                                style={{ cursor: "pointer" }}
                            >
                                {t("child_pitch_mode_cents")}
                            </Button>
                        ) : null}
                        {selectedIsChildTrack && childPitchOffsetDegreesParam ? (
                            <Button
                                size="1"
                                variant={
                                    editParam === childPitchOffsetDegreesParam ? "solid" : "soft"
                                }
                                color={editParam === childPitchOffsetDegreesParam ? "cyan" : "gray"}
                                onClick={() => dispatch(setEditParam(childPitchOffsetDegreesParam))}
                                style={{ cursor: "pointer" }}
                            >
                                {t("child_pitch_mode_degrees")}
                            </Button>
                        ) : null}
                        <Button
                            size="1"
                            variant={editParam === "pitch" ? "solid" : "soft"}
                            color={editParam === "pitch" ? "grass" : "gray"}
                            onClick={() => dispatch(setEditParam("pitch"))}
                            style={{ cursor: "pointer" }}
                        >
                            {t("pitch")}
                        </Button>
                        {/*  ?editParam 不是 pitch 时，显示 pitch 副参数开 ?*/}
                        {editParam !== "pitch" && pitchEnabled ? (
                            <IconButton
                                size="1"
                                variant={secondaryParamVisible["pitch"] ? "soft" : "ghost"}
                                color={secondaryParamVisible["pitch"] ? "blue" : "gray"}
                                onClick={() => toggleSecondaryParam("pitch")}
                                style={{ cursor: "pointer" }}
                                title={
                                    secondaryParamVisible["pitch"]
                                        ? t("hide_secondary_param")
                                        : t("show_secondary_param")
                                }
                            >
                                {secondaryParamVisible["pitch"] ? (
                                    <EyeOpenIcon />
                                ) : (
                                    <EyeClosedIcon />
                                )}
                            </IconButton>
                        ) : null}
                        {/* 由后端 processorParams 驱动的动态参数按钮 */}
                        {processorParams.map((p) => (
                            <React.Fragment key={p.id}>
                                <Button
                                    size="1"
                                    variant={editParam === p.id ? "solid" : "soft"}
                                    color={editParam === p.id ? "amber" : "gray"}
                                    onClick={() => dispatch(setEditParam(p.id))}
                                    style={{ cursor: "pointer" }}
                                >
                                    {getProcessorParamLabel(p)}
                                </Button>
                                {editParam !== p.id ? (
                                    <IconButton
                                        size="1"
                                        variant={secondaryParamVisible[p.id] ? "soft" : "ghost"}
                                        color={secondaryParamVisible[p.id] ? "orange" : "gray"}
                                        onClick={() => toggleSecondaryParam(p.id)}
                                        style={{ cursor: "pointer" }}
                                        title={
                                            secondaryParamVisible[p.id]
                                                ? t("hide_secondary_param")
                                                : t("show_secondary_param")
                                        }
                                    >
                                        {secondaryParamVisible[p.id] ? (
                                            <EyeOpenIcon />
                                        ) : (
                                            <EyeClosedIcon />
                                        )}
                                    </IconButton>
                                ) : null}
                            </React.Fragment>
                        ))}
                    </Flex>

                    {rootTrack ? (
                        <Flex align="center" gap="2">
                            <Text size="1" color="gray">
                                {t("algo_label")}
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
                            {processorStaticParams.map((param) => {
                                if (param.kind.type !== "static_enum") return null;
                                const currentValue =
                                    processorStaticValues[param.id] ?? param.kind.default_value;
                                return (
                                    <Flex key={param.id} align="center" gap="1">
                                        <Text size="1" color="gray">
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
                            {editParam === "pitch" ? (
                                <Button
                                    size="1"
                                    variant="soft"
                                    color="blue"
                                    onClick={handleOpenMidiDialog}
                                    disabled={!pitchEnabled}
                                    style={{ cursor: "pointer" }}
                                    title={pitchHardDisableReason ?? undefined}
                                >
                                    {(t as (key: string) => string)("midi_import")}
                                </Button>
                            ) : null}
                        </Flex>
                    ) : null}
                </Flex>
            </Flex>

            {/* Task 6.5: 参数面板顶部添加进度条区 ?*/}
            {asyncRefresh.isLoading && (
                <Flex className="px-3 py-2 bg-qt-base border-b border-qt-border">
                    <ProgressBar
                        percentage={asyncRefresh.progress}
                        label={(t as any)("refreshing_pitch_data") || "Refreshing pitch data"}
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
                        {(t as any)("retry") || "Retry"}
                    </Button>
                </Flex>
            )}

            {/* Note/Curve Editor Area */}
            <Flex className="flex-1 overflow-hidden relative">
                {/* Left axis + corner */}
                <Flex direction="column" className="shrink-0">
                    <Box
                        className="h-6 bg-qt-window border-b border-qt-border"
                        style={{ width: AXIS_W }}
                    />
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
                        bars={timeRulerBars}
                        pxPerBeat={pxPerBeat}
                        pxPerSec={pxPerSec}
                        secPerBeat={secPerBeat}
                        playheadSec={s.playheadSec}
                        playheadLineRef={rulerPlayheadLineRef}
                        playheadHeadRef={rulerPlayheadHeadRef}
                        contentRef={rulerContentRef}
                        onMouseDown={(e) => {
                            document.body.setAttribute("data-hs-focus-window", "pianoRoll");
                            interactions.onRulerMouseDown(e);
                        }}
                    />

                    <div
                        ref={scrollerRef}
                        className="flex-1 bg-qt-graph-bg overflow-x-scroll overflow-y-hidden relative custom-scrollbar outline-none focus:outline-none focus-visible:outline-none"
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
                        onScroll={interactions.onScrollerScroll}
                        onContextMenu={interactions.onScrollerContextMenu}
                        onKeyDown={interactions.onScrollerKeyDown}
                    >
                        {/* Spacer to provide scrollable width (must not consume full height) */}
                        <div
                            className="relative"
                            style={{ width: contentWidth, height: 1 }}
                            aria-hidden
                        />

                        {/* Sticky viewport overlay: grid + canvas do not physically scroll */}
                        <div
                            className="sticky left-0 top-0 h-full"
                            style={{ width: viewSize.w, overflow: "hidden" }}
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
                                    layerRef={gridLayerRef}
                                    boundaryRef={gridBoundaryRef}
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
                    </div>
                </Flex>
            </Flex>
            <MidiTrackSelectDialog
                open={midiDialogOpen}
                onOpenChange={setMidiDialogOpen}
                midiPath={midiPath}
                selectionStartFrame={midiSelArgs.selectionStartFrame}
                selectionMaxFrames={midiSelArgs.selectionMaxFrames}
                onImported={handleMidiImported}
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
                />
            )}
        </Flex>
    );
};
