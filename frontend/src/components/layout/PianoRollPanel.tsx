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
import { EyeOpenIcon, EyeClosedIcon } from "@radix-ui/react-icons";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    setEditParam,
    setTrackStateRemote,
} from "../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../features/session/trackUtils";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { getWaveformColors } from "../../theme/waveformColors";
import type { ProcessorParamDescriptor } from "../../types/api";
import { paramsApi } from "../../services/api/params";
import {
    listResamplers,
    browseResamplerExe,
    addResampler,
    type ResamplerEntry,
} from "../../services/api/resampler";

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
import { usePianoRollData } from "./pianoRoll/usePianoRollData";
import { useClipsPeaksForPianoRoll } from "./pianoRoll/useClipsPeaksForPianoRoll";
import { usePianoRollInteractions } from "./pianoRoll/usePianoRollInteractions";
import { useLiveParamEditing } from "./pianoRoll/useLiveParamEditing";
import {
    getActiveSecondaryParamId,
    toggleSecondaryParamVisibility,
} from "./pianoRoll/secondaryOverlaySelection";
import type {
    ParamName,
    StrokeMode,
    StrokePoint,
    ValueViewport,
} from "./pianoRoll/types";
import { selectKeybinding } from "../../features/keybindings/keybindingsSlice";

import { useAsyncPitchRefresh } from "../../hooks/useAsyncPitchRefresh";
import { ProgressBar } from "../ProgressBar";

import { usePianoRollStatusUpdate } from "../../contexts/PianoRollStatusContext";
import { MidiTrackSelectDialog } from "./MidiTrackSelectDialog";
import { coreApi } from "../../services/api/core";

export const PianoRollPanel: React.FC = () => {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const s = useAppSelector((state: RootState) => state.session);
    const editParam = s.editParam as ParamName;
    const pianoRollCopyKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.copy"),
    );
    const pianoRollPasteKb = useAppSelector((state) =>
        selectKeybinding(state, "pianoRoll.paste"),
    );
    const prVerticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoRollVerticalZoom"),
    );
    const { mode: themeMode } = useAppTheme();
    const waveformColors = useMemo(
        () => getWaveformColors(themeMode),
        [themeMode],
    );

    // Task 6.3: 集成 useAsyncPitchRefresh Hook
    const asyncRefresh = useAsyncPitchRefresh();
    const [showSuccessMessage] = useState(false);

    // MIDI 导入弹窗状态
    const [midiDialogOpen, setMidiDialogOpen] = useState(false);
    const [midiPath, setMidiPath] = useState<string | null>(null);

    // ─── 外部 Resampler 列表 ────────────────────────────────────
    const [resamplerList, setResamplerList] = useState<ResamplerEntry[]>([]);
    useEffect(() => {
        void listResamplers().then(setResamplerList).catch(() => {});
    }, []);

    const handleOpenMidiDialog = useCallback(async () => {
        try {
            const res = await coreApi.openMidiDialog();
            if (res.ok && !res.canceled && res.path) {
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

    const [scrollLeft, setScrollLeft] = useState(0);
    const [pxPerSec, setPxPerSec] = useState(() => {
        const stored = Number(
            localStorage.getItem("hifishifter.paramPxPerSec"),
        );
        return Number.isFinite(stored) && stored > 0
            ? Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, stored))
            : DEFAULT_PX_PER_SEC;
    });
    // 渲染时根�?BPM 换算 pxPerBeat：pxPerBeat = pxPerSec × (60 / bpm)
    const pxPerBeat = pxPerSec * (60 / Math.max(1e-6, s.bpm));
    const scrollLeftRef = useRef(scrollLeft);
    const pxPerBeatRef = useRef(pxPerBeat);
    const pxPerSecRef = useRef(pxPerSec);

    // BPM 变化时，按比例调�?scrollLeft，保持视口中心点的秒数不�?
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
        localStorage.setItem("hifishifter.paramPxPerSec", String(pxPerSec));
    }, [pxPerBeat, pxPerSec]);

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
        setSecondaryParamVisible((prev) =>
            toggleSecondaryParamVisibility(prev, param),
        );
    }, []);

    const [pitchView, setPitchView] = useState<ValueViewport>(() => ({
        center: (PITCH_MIN_MIDI + PITCH_MAX_MIDI) / 2,
        span: PITCH_MAX_MIDI - PITCH_MIN_MIDI,
    }));
    // 按参数 id 屘化的视口状态（音高以外的所有参数）
    const [paramViews, setParamViews] = useState<Record<string, ValueViewport>>(
        {},
    );
    const pitchViewRef = useRef(pitchView);
    const paramViewsRef = useRef(paramViews);

    useEffect(() => {
        pitchViewRef.current = pitchView;
    }, [pitchView]);
    useEffect(() => {
        paramViewsRef.current = paramViews;
    }, [paramViews]);

    /** 更新单个非音高参数的视口 */
    const setParamViewport = useCallback(
        (param: string, next: ValueViewport) => {
            setParamViews((prev) => ({ ...prev, [param]: next }));
            // 同时就地更新 ref，保证 pointer 回调内能立即得到新属性
            paramViewsRef.current = { ...paramViewsRef.current, [param]: next };
        },
        [],
    );

    const rootTrackId = useMemo(() => {
        return resolveRootTrackId(s.tracks, effectiveSelectedTrackId);
    }, [effectiveSelectedTrackId, s.tracks]);

    const rootTrack = useMemo(() => {
        if (!rootTrackId) return null;
        return s.tracks.find((tr) => tr.id === rootTrackId) ?? null;
    }, [s.tracks, rootTrackId]);

    // 声码器参数描述符（由 algo 动态定制面板）
    const [processorParams, setProcessorParams] = useState<
        ProcessorParamDescriptor[]
    >([]);
    const processorParamsRef = useRef<ProcessorParamDescriptor[]>([]);
    const [processorStaticParams, setProcessorStaticParams] = useState<
        ProcessorParamDescriptor[]
    >([]);
    const [processorStaticValues, setProcessorStaticValues] = useState<
        Record<string, number>
    >({});

    // 当 algo 变化时，重新抓取参数描述符
    useEffect(() => {
        const algo = rootTrack?.pitchAnalysisAlgo ?? "world_dll";
        let cancelled = false;
        paramsApi
            .getProcessorParams(algo)
            .then((params) => {
                if (cancelled) return;
                // 只保留 AutomationCurve 类型（可以绘制曲线的）
                const curvable = params.filter(
                    (p) => p.kind.type === "automation_curve",
                );
                const staticParams = params.filter(
                    (p) => p.kind.type === "static_enum",
                );
                processorParamsRef.current = curvable;
                setProcessorParams(curvable);
                setProcessorStaticParams(staticParams);
                // 初始化还没有视口的参数
                setParamViews((prev) => {
                    const next = { ...prev };
                    for (const p of curvable) {
                        if (!next[p.id] && p.kind.type === "automation_curve") {
                            const { min_value, max_value, default_value } =
                                p.kind;
                            const span = max_value - min_value;
                            next[p.id] = {
                                center: default_value,
                                span: span > 0 ? span : 1,
                            };
                        }
                    }
                    return next;
                });

                if (!rootTrackId || staticParams.length === 0) {
                    setProcessorStaticValues({});
                    return;
                }

                Promise.all(
                    staticParams.map((param) =>
                        paramsApi.getStaticParam(rootTrackId, param.id),
                    ),
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
            const result = await paramsApi.setStaticParam(
                rootTrackId,
                paramId,
                value,
                true,
            );
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
            return label;
        },
        [t],
    );

    // 当 processorParams 变化时，若 editParam 不在可用集合内，自动回退到 pitch
    useEffect(() => {
        const available = new Set([
            "pitch",
            ...processorParams.map((p) => p.id),
        ]);
        if (!available.has(editParam)) {
            dispatch(setEditParam("pitch"));
        }
    }, [processorParams, editParam, dispatch]);

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
        if (rootTrack.pitchAnalysisAlgo === "none")
            return t("pitch_requires_algo");
        return null;
    }, [editParam, rootTrack, t]);

    const pitchEnabled =
        editParam !== "pitch" || pitchHardDisableReason == null;

    const activeSecondaryParamId = useMemo(() => {
        const next = getActiveSecondaryParamId({
            editParam,
            processorParamIds: processorParams.map((p) => p.id),
            secondaryParamVisible,
        });
        if (next === "pitch" && !pitchEnabled) {
            return null;
        }
        return next;
    }, [editParam, processorParams, secondaryParamVisible, pitchEnabled]);

    const secPerBeat = 60 / Math.max(1e-6, s.bpm);
    const contentWidth = Math.max(8, Math.ceil(s.projectSec * pxPerSec));

    const scrollerRef = useRef<HTMLDivElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const axisCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const rafRef = useRef<number | null>(null);
    const drawRef = useRef<() => void>(() => {});
    const lastScrollLeftRef = useRef<number | null>(null);
    const scrollStateRafRef = useRef<number | null>(null);

    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const gridLayerRef = useRef<HTMLDivElement | null>(null);
    const gridBoundaryRef = useRef<HTMLDivElement | null>(null);

    function positiveMod(value: number, mod: number): number {
        if (!Number.isFinite(value) || !Number.isFinite(mod) || mod <= 0)
            return 0;
        const r = value % mod;
        return (r + mod) % mod;
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

    const invalidate = useCallback(() => {
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            drawRef.current();
        });
    }, []);

    // The ruler is React-rendered, but the main graph is canvas-rendered.
    // Ensure playhead changes (seek / playback) trigger a redraw.
    useEffect(() => {
        invalidate();
    }, [s.playheadSec, invalidate]);

    function syncScrollLeft(scroller: HTMLDivElement) {
        const next = scroller.scrollLeft;
        if (
            lastScrollLeftRef.current != null &&
            lastScrollLeftRef.current === next
        ) {
            return;
        }
        lastScrollLeftRef.current = next;
        scrollLeftRef.current = next;

        if (rulerContentRef.current) {
            rulerContentRef.current.style.transform = `translateX(${-next}px)`;
        }

        if (gridLayerRef.current) {
            const weakStepPx = Math.max(
                1e-6,
                pxPerBeatRef.current * gridStepBeats(s.grid),
            );
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

    const valueToY = useCallback(
        (param: ParamName, v: number, h: number): number => {
            const H = Math.max(1, h);
            if (param === "pitch") {
                const absMin = PITCH_MIN_MIDI;
                const absMax = PITCH_MAX_MIDI;
                const view = pitchViewRef.current;
                const span = clamp(view.span, 1e-6, absMax - absMin);
                const min = clamp(
                    view.center - span / 2,
                    absMin,
                    absMax - span,
                );
                const t =
                    (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
                return (1 - t) * H;
            }

            const desc = processorParamsRef.current.find((d) => d.id === param);
            const absMin =
                desc?.kind.type === "automation_curve"
                    ? desc.kind.min_value
                    : 0;
            const absMax =
                desc?.kind.type === "automation_curve"
                    ? desc.kind.max_value
                    : 1;
            const view = paramViewsRef.current[param] ?? {
                center: (absMin + absMax) / 2,
                span: absMax - absMin || 1,
            };
            const span = clamp(view.span, 1e-6, absMax - absMin || 1);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            const t = (clamp(v, absMin, absMax) - min) / Math.max(1e-9, span);
            return (1 - t) * H;
        },
        [],
    );

    const yToViewportT = useCallback((y: number, h: number): number => {
        const H = Math.max(1, h);
        return clamp(y / H, 0, 1);
    }, []);

    const yToValue = useCallback(
        (param: ParamName, y: number, h: number): number => {
            const H = Math.max(1, h);
            const t = 1 - clamp(y / H, 0, 1);
            if (param === "pitch") {
                const absMin = PITCH_MIN_MIDI;
                const absMax = PITCH_MAX_MIDI;
                const view = pitchViewRef.current;
                const span = clamp(view.span, 1e-6, absMax - absMin);
                const min = clamp(
                    view.center - span / 2,
                    absMin,
                    absMax - span,
                );
                return clamp(min + t * span, absMin, absMax);
            }
            const desc = processorParamsRef.current.find((d) => d.id === param);
            const absMin =
                desc?.kind.type === "automation_curve"
                    ? desc.kind.min_value
                    : 0;
            const absMax =
                desc?.kind.type === "automation_curve"
                    ? desc.kind.max_value
                    : 1;
            const view = paramViewsRef.current[param] ?? {
                center: (absMin + absMax) / 2,
                span: absMax - absMin || 1,
            };
            const span = clamp(view.span, 1e-6, absMax - absMin || 1);
            const min = clamp(view.center - span / 2, absMin, absMax - span);
            return clamp(min + t * span, absMin, absMax);
        },
        [],
    );

    function clampViewport(param: ParamName, v: ValueViewport): ValueViewport {
        if (param === "pitch") {
            const absMin = PITCH_MIN_MIDI;
            const absMax = PITCH_MAX_MIDI;
            const span = clamp(v.span, 6, absMax - absMin);
            const center = clamp(
                v.center,
                absMin + span / 2,
                absMax - span / 2,
            );
            return { center, span };
        }
        const desc = processorParamsRef.current.find((d) => d.id === param);
        const absMin =
            desc?.kind.type === "automation_curve" ? desc.kind.min_value : 0;
        const absMax =
            desc?.kind.type === "automation_curve" ? desc.kind.max_value : 1;
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

    // 用于通知 usePianoRollData 当前是否处于 live 编辑状态（pointer down 期间�?true）�?
    // pitch_orig_updated 事件到达时若�?true，则延迟曲线刷新�?pointer-up 后执行�?
    const liveEditActiveRef = useRef(false);

    const {
        paramView,
        setParamView,
        secondaryParamView,
        bumpRefreshToken,
        refreshNow,
        refreshSecondaryNow,
        notifyLiveEditEnded,
        isLoading,
    } = usePianoRollData({
        editParam,
        secondaryParamId: activeSecondaryParamId,
        pitchEnabled,
        paramsEpoch:
            (s as unknown as { paramsEpoch?: number }).paramsEpoch ?? 0,
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

    const visibleSecondaryParamView = useMemo(() => {
        if (!activeSecondaryParamId || !secondaryParamView) {
            return null;
        }
        return secondaryParamView.key.includes(`|${activeSecondaryParamId}|`)
            ? secondaryParamView
            : null;
    }, [activeSecondaryParamId, secondaryParamView]);

    const refreshSecondaryNowRef = useRef(refreshSecondaryNow);
    useEffect(() => {
        refreshSecondaryNowRef.current = refreshSecondaryNow;
    }, [refreshSecondaryNow]);

    useEffect(() => {
        if (!rootTrackId) {
            invalidate();
            return;
        }
        if (activeSecondaryParamId) {
            void refreshSecondaryNowRef.current();
            return;
        }
        invalidate();
    }, [activeSecondaryParamId, invalidate, rootTrackId]);

    const handleMidiImported = useCallback(
        (_result: { notes_imported: number; frames_touched: number }) => {
            // 导入完成后刷新参数面板
            refreshNow();
        },
        [refreshNow],
    );

    // 获取当前 track 下的所�?clips，用�?per-clip 波形叠加绘制
    // 获取轨道组内所有 clips（包含 root 轨道及所有子轨道的 clip）
    const trackClips = useMemo(
        () => s.clips.filter((c) => groupTrackIds.has(c.trackId)),
        [s.clips, groupTrackIds],
    );

    // 可见区域的 sec 范围（统一用 sec 坐标系）
    const visibleStartSec = scrollLeft / Math.max(1e-9, pxPerSec);
    const visibleEndSec =
        visibleStartSec + viewSize.w / Math.max(1e-9, pxPerSec);

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
    }, [
        clipPeaks,
        paramView,
        visibleSecondaryParamView,
        pxPerBeat,
        viewSize.w,
        viewSize.h,
        invalidate,
    ]);

    useEffect(() => {
        invalidate();
    }, [pitchView, paramViews, editParam, activeSecondaryParamId, invalidate]);

    // 检测音高曲线更新时触发重绘（必须在 detectedPitchCurves 声明之后�?
    // useEffect 已移�?detectedPitchCurves useMemo 定义之后，见下方�?

    const paramViewRef = useRef<
        import("./pianoRoll/types").ParamViewSegment | null
    >(null);
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
    // 并触发可能被延迟�?pitch_orig_updated 曲线刷新�?
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
                return clip && groupTrackIds.has(clip.trackId);
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
            paramView,
            secondaryParamView: visibleSecondaryParamView,
            secondaryParamId: activeSecondaryParamId,
            showSecondaryParam: activeSecondaryParamId != null,
            overlayText:
                editParam === "pitch" && !pitchEnabled
                    ? pitchHardDisableReason
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
        });
    };

    const interactions = usePianoRollInteractions({
        dispatch,
        rootTrackId,
        editParam,
        pitchEnabled,
        toolMode: s.toolMode,
        secPerBeat,
        bpm: s.bpm,
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
    });

    const onScrollerWheelNative = interactions.onScrollerWheelNative;

    useEffect(() => {
        const el = scrollerRef.current;
        if (!el) return;

        const handler: EventListener = (evt) => {
            onScrollerWheelNative(evt as globalThis.WheelEvent);
        };

        el.addEventListener("wheel", handler, {
            passive: false,
        } as globalThis.AddEventListenerOptions);
        return () => {
            el.removeEventListener("wheel", handler);
        };
    }, [onScrollerWheelNative]);

    // Silence unused state warnings; selectionUi is future UI.
    void selectionUi;

    useEffect(() => {
        setCanvasCursor(s.toolMode === "select" ? "default" : "crosshair");
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

    return (
        <Flex
            direction="column"
            className="h-full w-full bg-qt-graph-bg border-t border-qt-border"
        >
            {/* Header / Parameter Switch */}
            <Flex
                align="center"
                justify="between"
                className="h-8 bg-qt-base border-b border-qt-border px-2 shrink-0"
            >
                <Text size="1" weight="bold" color="gray">
                    {t("param_editor")}
                </Text>

                <Flex gap="2" align="center">
                    <Flex gap="1" align="center">
                        <Button
                            size="1"
                            variant={editParam === "pitch" ? "solid" : "soft"}
                            color={editParam === "pitch" ? "grass" : "gray"}
                            onClick={() => dispatch(setEditParam("pitch"))}
                            style={{ cursor: "pointer" }}
                        >
                            {t("pitch")}
                        </Button>
                        {/* �?editParam 不是 pitch 时，显示 pitch 副参数开�?*/}
                        {editParam !== "pitch" && pitchEnabled ? (
                            <IconButton
                                size="1"
                                variant={
                                    secondaryParamVisible["pitch"]
                                        ? "soft"
                                        : "ghost"
                                }
                                color={
                                    secondaryParamVisible["pitch"]
                                        ? "blue"
                                        : "gray"
                                }
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
                                    variant={
                                        editParam === p.id ? "solid" : "soft"
                                    }
                                    color={
                                        editParam === p.id ? "amber" : "gray"
                                    }
                                    onClick={() => dispatch(setEditParam(p.id))}
                                    style={{ cursor: "pointer" }}
                                >
                                    {getProcessorParamLabel(p)}
                                </Button>
                                {editParam !== p.id ? (
                                    <IconButton
                                        size="1"
                                        variant={
                                            secondaryParamVisible[p.id]
                                                ? "soft"
                                                : "ghost"
                                        }
                                        color={
                                            secondaryParamVisible[p.id]
                                                ? "orange"
                                                : "gray"
                                        }
                                        onClick={() =>
                                            toggleSecondaryParam(p.id)
                                        }
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

                    {editParam === "pitch" && rootTrack ? (
                        <Flex align="center" gap="2">
                            <Text size="1" color="gray">
                                {t("algo_label")}
                            </Text>
                            <Select.Root
                                value={
                                    [
                                        "world_dll",
                                        "nsf_hifigan_onnx",
                                        "vslib",
                                        "none",
                                    ].includes(rootTrack.pitchAnalysisAlgo) ||
                                    rootTrack.pitchAnalysisAlgo.startsWith(
                                        "external_resampler:",
                                    )
                                        ? rootTrack.pitchAnalysisAlgo
                                        : "world_dll"
                                }
                                onValueChange={(v) => {
                                    if (!rootTrackId) return;
                                    if (v === "__add_resampler") {
                                        void (async () => {
                                            const path =
                                                await browseResamplerExe();
                                            if (!path) return;
                                            const name =
                                                path
                                                    .split(/[\\/]/)
                                                    .pop()
                                                    ?.replace(
                                                        /\.exe$/i,
                                                        "",
                                                    ) ?? "Resampler";
                                            const entry =
                                                await addResampler(
                                                    name,
                                                    path,
                                                );
                                            if (entry) {
                                                setResamplerList((prev) => [
                                                    ...prev,
                                                    entry,
                                                ]);
                                                dispatch(
                                                    setTrackStateRemote({
                                                        trackId:
                                                            rootTrackId,
                                                        pitchAnalysisAlgo: `external_resampler:${entry.id}`,
                                                    }),
                                                );
                                            }
                                        })();
                                        return;
                                    }
                                    dispatch(
                                        setTrackStateRemote({
                                            trackId: rootTrackId,
                                            pitchAnalysisAlgo: v,
                                        }),
                                    );
                                }}
                            >
                                <Select.Trigger className="min-w-[140px]" />
                                <Select.Content>
                                    <Select.Item value="world_dll">
                                        world
                                    </Select.Item>
                                    <Select.Item value="nsf_hifigan_onnx">
                                        nsf-hifigan
                                    </Select.Item>
                                    <Select.Item value="vslib">
                                        vslib
                                    </Select.Item>
                                    <Select.Item value="none">
                                        {t("none")}
                                    </Select.Item>
                                    {resamplerList.length > 0 && (
                                        <Select.Separator />
                                    )}
                                    {resamplerList.map((rs) => (
                                        <Select.Item
                                            key={rs.id}
                                            value={`external_resampler:${rs.id}`}
                                            disabled={!rs.available}
                                        >
                                            {rs.displayName}
                                            {!rs.available && " ⚠️"}
                                        </Select.Item>
                                    ))}
                                    <Select.Separator />
                                    <Select.Item value="__add_resampler">
                                        + {t("add_resampler") ?? "添加 Resampler..."}
                                    </Select.Item>
                                </Select.Content>
                            </Select.Root>

                            {processorStaticParams.map((param) => {
                                if (param.kind.type !== "static_enum")
                                    return null;
                                const currentValue =
                                    processorStaticValues[param.id] ??
                                    param.kind.default_value;
                                return (
                                    <Flex key={param.id} align="center" gap="1">
                                        <Text size="1" color="gray">
                                            {getProcessorParamLabel(param)}
                                        </Text>
                                        {param.kind.options.map(
                                            ([label, value]) => (
                                                <Button
                                                    key={`${param.id}-${value}`}
                                                    size="1"
                                                    variant={
                                                        currentValue === value
                                                            ? "solid"
                                                            : "soft"
                                                    }
                                                    color={
                                                        currentValue === value
                                                            ? "blue"
                                                            : "gray"
                                                    }
                                                    onClick={() => {
                                                        void handleStaticParamChange(
                                                            param.id,
                                                            value,
                                                        );
                                                    }}
                                                    style={{
                                                        cursor: "pointer",
                                                    }}
                                                >
                                                    {getStaticOptionLabel(
                                                        param.id,
                                                        label,
                                                        value,
                                                    )}
                                                </Button>
                                            ),
                                        )}
                                    </Flex>
                                );
                            })}
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
                        </Flex>
                    ) : null}
                </Flex>
            </Flex>

            {/* Task 6.5: 参数面板顶部添加进度条区�?*/}
            {asyncRefresh.isLoading && (
                <Flex className="px-3 py-2 bg-qt-base border-b border-qt-border">
                    <ProgressBar
                        percentage={asyncRefresh.progress}
                        label={
                            (t as any)("refreshing_pitch_data") ||
                            "Refreshing pitch data"
                        }
                        showCancel={true}
                        onCancel={async () => {
                            // Task 6.6: 取消按钮点击时调�?cancelRefresh()
                            await asyncRefresh.cancelRefresh();
                        }}
                        estimatedRemaining={asyncRefresh.estimatedRemaining}
                    />
                </Flex>
            )}

            {/* Task 6.7: 任务完成后显示成功提�?*/}
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
                        onClick={() =>
                            rootTrackId &&
                            void asyncRefresh.startRefresh(rootTrackId)
                        }
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
                        className="bg-qt-window border-r border-qt-border relative"
                        style={{ width: AXIS_W, flex: 1 }}
                    >
                        <canvas
                            ref={axisCanvasRef}
                            className="absolute inset-0"
                        />
                    </div>
                </Flex>

                {/* Right: ruler + scrollable canvas */}
                <Flex direction="column" className="flex-1 min-w-0">
                    <TimeRuler
                        contentWidth={contentWidth}
                        scrollLeft={scrollLeft}
                        bars={(() => {
                            const beatsPerBar = Math.max(
                                1,
                                Math.round(s.beats || 4),
                            );
                            const totalBeats = Math.max(
                                1,
                                Math.ceil(s.projectSec / secPerBeat),
                            );
                            const result: Array<{
                                beat: number;
                                label: string;
                            }> = [];
                            let barIndex = 1;
                            for (
                                let beat = 0;
                                beat <= totalBeats;
                                beat += beatsPerBar
                            ) {
                                result.push({ beat, label: `${barIndex}.1` });
                                barIndex += 1;
                            }
                            return result;
                        })()}
                        pxPerBeat={pxPerBeat}
                        pxPerSec={pxPerSec}
                        secPerBeat={secPerBeat}
                        playheadSec={s.playheadSec}
                        contentRef={rulerContentRef}
                        onMouseDown={interactions.onRulerMouseDown}
                    />

                    <div
                        ref={scrollerRef}
                        className="flex-1 bg-qt-graph-bg overflow-x-auto overflow-y-hidden relative custom-scrollbar"
                        data-piano-roll-scroller
                        tabIndex={0}
                        onMouseDownCapture={
                            interactions.onScrollerMouseDownCapture
                        }
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
                            <div
                                className="relative h-full"
                                style={{ width: viewSize.w }}
                            >
                                <BackgroundGrid
                                    contentWidth={contentWidth}
                                    contentHeight={viewSize.h}
                                    viewportWidth={viewSize.w}
                                    scrollLeft={scrollLeft}
                                    pxPerBeat={pxPerBeat}
                                    grid={s.grid}
                                    beatsPerBar={Math.max(
                                        1,
                                        Math.round(s.beats || 4),
                                    )}
                                    layerRef={gridLayerRef}
                                    boundaryRef={gridBoundaryRef}
                                />

                                <canvas
                                    ref={canvasRef}
                                    className="absolute inset-0"
                                    style={{ cursor: canvasCursor }}
                                    onPointerMove={
                                        interactions.onCanvasPointerMove
                                    }
                                    onPointerLeave={
                                        interactions.onCanvasPointerLeave
                                    }
                                    onPointerDown={
                                        interactions.onCanvasPointerDown
                                    }
                                />
                            </div>
                        </div>
                    </div>
                </Flex>
            </Flex>
            <MidiTrackSelectDialog
                open={midiDialogOpen}
                onOpenChange={setMidiDialogOpen}
                midiPath={midiPath}
                offsetSec={s.playheadSec}
                onImported={handleMidiImported}
            />
        </Flex>
    );
};
