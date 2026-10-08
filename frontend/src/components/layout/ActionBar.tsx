// hs-interaction-exempt: 主工具栏是紧凑 chrome（size 1、内联底色、BPM 有手势累加器），能力层原语是表单尺寸；本文件的滚轮与精细调整接线已完备（BPM/节拍器音量/三个下拉均有），故刻意保留。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    isPluginMode,
    canControlHostTransport,
    dawControlledReason,
} from "../../services/hostCapabilities";
import { createPortal } from "react-dom";
import { Flex, Select, TextField, Button, IconButton, Box } from "@radix-ui/themes";
import {
    CheckIcon,
    DoubleArrowRightIcon,
    PauseIcon,
    Pencil1Icon,
    PlayIcon,
    StopIcon,
} from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import { PitchSnapSettingsDialog } from "./PitchSnapSettingsDialog";
import { SnapGridSettingsDialog } from "./SnapGridSettingsDialog";
import { SplitTransitionSettingsDialog } from "./SplitTransitionSettingsDialog";
import { CustomScaleDialog } from "./CustomScaleDialog";
import { AppContextMenu } from "../../ui/Menu";
import { AppToolbarSeparator } from "../../ui/Toolbar";
import { AppIconButton, AppSlider, AppSliderReadout } from "../../ui";

import {
    playOriginal,
    stopAudioPlayback,
    setBpm,
    updateTransportBpm,
    setProjectTimelineSettingsRemote,
    toggleAutoCrossfade,
    toggleSplitTransition,
    toggleSnap,
    togglePlayheadZoom,
    toggleAutoScroll,
    toggleIgnoreGrouping,
    cycleRippleMode,
    setRippleMode,
    toggleParamEditorSeekPlayhead,
    toggleParamEditorTimelineClickSelectTrack,
    persistUiSettings,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setTempoMap,
    undoRemote,
    redoRemote,
} from "../../features/session/sessionSlice";
import { setTempoMapRemote } from "../../features/session/thunks/tempoMapThunks";
import { updateMetronome } from "../../features/session/thunks/transportThunks";
import {
    computeEffectiveSnap,
    isSnapGestureActive,
    subscribeSnapGesture,
} from "../../utils/timelineSnapping";
import type { TempoMapScaleData, TempoTimeSignature } from "../../utils/tempoMap";
import {
    clampBpm,
    effectiveScaleAtSec,
    pointIndexAtSec,
    scaleLikeToScaleData,
    TEMPO_DENOMINATORS,
    tempoAtSec,
    updateTempoPoint,
} from "../../utils/tempoMap";
import { SCALE_KEYS, SCALE_LABELS, type ScaleLike } from "../../utils/musicalScales";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { useNonPassiveWheel } from "../../utils/useNonPassiveWheel";
import { createFrameCommitter, type FrameCommitter } from "../../utils/commitOncePerFrame";
import {
    formatKeybindingList,
    isModifierActive,
    selectKeybinding,
    selectKeybindings,
} from "../../features/keybindings/keybindingsSlice";
import { openPanelById, selectPanelVisible, togglePanelVisible } from "../../features/dock/dockApi";
import {
    PANEL_FILE_BROWSER,
    PANEL_NOTEBOOK,
    PANEL_UNDO_HISTORY,
} from "../dock/registerBuiltinPanels";
import { store } from "../../app/store";
import {
    cancelRecordingCountdown,
    loadRecordingApps,
    loadRecordingDevices,
    loadRecordingSettings,
    saveRecordingSettings,
    startRecordingFlow,
    stopRecordingFlow,
} from "../../features/recording/recordingSlice";
import type { RecordingSettings } from "../../services/api/recording";
import { RecordingSettingsDialog } from "./RecordingSettingsDialog";

/** 节拍器图标（机身 + 摆锤；与 TempoMapCornerButton 的节拍器造型一致）。 */
function MetronomeIcon() {
    return (
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            {/* 节拍器机身（梯形轮廓） */}
            <path
                d="M5.6 2.2 H10.4 L13 13.4 H3 Z"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinejoin="round"
            />
            {/* 摆锤 */}
            <path
                d="M8 11.2 L11 4.4"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
            />
            {/* 配重圆点 */}
            <circle cx="11" cy="4.4" r="1.1" fill="currentColor" />
        </svg>
    );
}

/** 撤销图标：钩形弧线箭头（左向），DAW 惯用造型（Lucide undo-2）。 */
function UndoIcon() {
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M9 14 4 9l5-5" />
            <path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11" />
        </svg>
    );
}

/** 重做图标：撤销的镜像（右向钩形，Lucide redo-2）。 */
function RedoIcon() {
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="m15 14 5-5-5-5" />
            <path d="M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13" />
        </svg>
    );
}

/** ActionBar 实际消费的 session 字段子集（配合 shallowEqual 阻断播放轮询的重渲染）。
 *  新增消费字段时必须同步补充到这里。 */
const selectActionBarSession = (state: RootState) => {
    const session = state.session;
    return {
        // 撤销/重做按钮的可用性（后端 history_state 事件驱动的镜像）
        historyRedoDepth: session.historyRedoDepth,
        historyUndoDepth: session.historyUndoDepth,
        autoCrossfadeEnabled: session.autoCrossfadeEnabled,
        autoScrollEnabled: session.autoScrollEnabled,
        beats: session.beats,
        bpm: session.bpm,
        grid: session.grid,
        ignoreGrouping: session.ignoreGrouping,
        metronomeAccent: session.metronomeAccent,
        metronomeEnabled: session.metronomeEnabled,
        metronomeGain: session.metronomeGain,
        metronomeMode: session.metronomeMode,
        metronomeSound: session.metronomeSound,
        paramEditorSeekPlayheadEnabled: session.paramEditorSeekPlayheadEnabled,
        paramEditorTimelineClickSelectTrackEnabled:
            session.paramEditorTimelineClickSelectTrackEnabled,
        playheadSec: session.playheadSec,
        playheadZoomEnabled: session.playheadZoomEnabled,
        project: session.project,
        rippleMode: session.rippleMode,
        snapEnabled: session.snapEnabled,
        splitTransitionEnabled: session.splitTransitionEnabled,
        tempoMap: session.tempoMap,
    };
};

export function ActionBar() {
    const dispatch = useAppDispatch();
    // 只选取本组件实际消费的字段子集并以 shallowEqual 比较：播放期间
    // runtime.playbackPositionSec 每 ~33ms 变一次，整片 session 的对象引用
    // 随之失效，若直接订阅 state.session，本组件（含全部子菜单定义）会以
    // ≥30Hz 重渲染。
    const s = useAppSelector(selectActionBarSession, shallowEqual);
    // runtime 的两个标量单独订阅：runtime 对象随播放轮询每 tick 新建，
    // 但 isPlaying 本身只在播放/暂停时变化。
    const isPlaying = useAppSelector((state: RootState) => state.session.runtime.isPlaying);
    const fileBrowserVisible = useAppSelector(selectPanelVisible(PANEL_FILE_BROWSER));
    const notebookVisible = useAppSelector(selectPanelVisible(PANEL_NOTEBOOK));
    const recording = useAppSelector((state: RootState) => state.recording);
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const { t, tf } = useI18n();

    const [pitchSnapOpen, setPitchSnapOpen] = useState(false);
    const [snapSettingsOpen, setSnapSettingsOpen] = useState(false);
    const [splitTransitionOpen, setSplitTransitionOpen] = useState(false);
    const [customScaleOpen, setCustomScaleOpen] = useState(false);
    const [recordingSettingsOpen, setRecordingSettingsOpen] = useState(false);
    const [recordingMenuPos, setRecordingMenuPos] = useState<{ x: number; y: number } | null>(null);
    const [metronomeMenuPos, setMetronomeMenuPos] = useState<{ x: number; y: number } | null>(null);
    const metronomeMenuRef = useRef<HTMLDivElement | null>(null);
    // 「操作记录」面板：右键撤销/重做按钮打开。它是**可停靠面板**，显隐由停靠
    // 布局决定（不再是本组件的局部 state），因此它能被拖到任意位置、与其他面板
    // 合并成标签页，也能被"布局"菜单统一管理。
    const undoButtonRef = useRef<HTMLButtonElement | null>(null);
    const openHistoryPanel = useCallback(() => {
        // 【落点提示：按钮矩形】从撤销/重做按钮打开时，面板浮在按钮**正下方**
        // （下方放不下则翻到上方）—— 用户刚点的按钮就是他的注意力所在，把它丢到
        // 屏幕角落会让人以为没打开。见 `resolveFloatNearRect`。
        const rect = undoButtonRef.current?.getBoundingClientRect() ?? null;
        openPanelById(
            dispatch,
            store.getState,
            PANEL_UNDO_HISTORY,
            rect ? { x: rect.x, y: rect.y, w: rect.width, h: rect.height } : null,
        );
    }, [dispatch]);
    // 按钮 tooltip 里的快捷键提示（跟随用户在快捷键设置中的自定义绑定）。
    // 重做默认绑了两个键，tooltip 要把两个都写出来（`;` 连接）。
    const undoShortcutKb = useAppSelector((state: RootState) =>
        selectKeybindings(state, "edit.undo"),
    );
    const redoShortcutKb = useAppSelector((state: RootState) =>
        selectKeybindings(state, "edit.redo"),
    );

    // ── "拖动时切换吸附"（modifier.clipNoSnap）────────────────────────
    // 时间轴拖拽手势进行中且按住该修饰键时，工具栏吸附按钮临时显示为
    // 取反后的状态（与参数编辑器音高吸附按钮的做法一致）。
    const noSnapKb = useAppSelector((state: RootState) =>
        selectKeybinding(state, "modifier.clipNoSnap"),
    );
    const [snapToggleHeld, setSnapToggleHeld] = useState(false);
    const [snapGestureActive, setSnapGestureActive] = useState(isSnapGestureActive());

    useEffect(() => {
        const kb = noSnapKb;
        const sync = (e: KeyboardEvent | null) => {
            setSnapToggleHeld(e ? isModifierActive(kb, e) : false);
        };
        const onKey = (e: KeyboardEvent) => sync(e);
        const onBlur = () => sync(null);
        window.addEventListener("keydown", onKey as EventListener);
        window.addEventListener("keyup", onKey as EventListener);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKey as EventListener);
            window.removeEventListener("keyup", onKey as EventListener);
            window.removeEventListener("blur", onBlur);
        };
    }, [noSnapKb]);

    useEffect(() => subscribeSnapGesture(() => setSnapGestureActive(isSnapGestureActive())), []);

    // 拖拽手势期间按住修饰键 → 吸附按钮临时显示取反后的状态。
    const effectiveSnapVisual = computeEffectiveSnap(
        s.snapEnabled,
        snapGestureActive && snapToggleHeld,
    );

    useEffect(() => {
        if (!metronomeMenuPos) return;
        const onPointerDown = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (metronomeMenuRef.current?.contains(target)) return;
            setMetronomeMenuPos(null);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") setMetronomeMenuPos(null);
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [metronomeMenuPos]);

    const recordingSourceLabel = (() => {
        switch (recording.settings.captureMode) {
            case "loopback":
                return tf("recording_mode_loopback");
            case "application":
                return tf("recording_mode_application");
            default:
                return tf("recording_mode_device");
        }
    })();

    const recordingDeviceLabel = (() => {
        const { captureMode } = recording.settings;
        if (captureMode === "device") {
            // "default" 是后端枚举出的合成项（未本地化），始终显示本地化文案。
            if (recording.settings.sourceDevice === "default") {
                return tf("recording_device_default");
            }
            const device = recording.devices.find(
                (item) => !item.isLoopback && item.id === recording.settings.sourceDevice,
            );
            return device?.name ?? tf("recording_device_default");
        }
        if (captureMode === "loopback") {
            if (
                recording.settings.loopbackDevice === "default" ||
                recording.settings.loopbackDevice === "loopback:default"
            ) {
                return tf("recording_loopback_default");
            }
            const device = recording.devices.find(
                (item) => item.isLoopback && item.id === recording.settings.loopbackDevice,
            );
            return device?.name ?? tf("recording_loopback_default");
        }
        const app = recording.apps.find((item) => item.id === recording.settings.captureAppId);
        return app?.name || recording.settings.captureAppName || tf("recording_application");
    })();

    const recordingTooltip = [
        recording.active
            ? tf("recording_tooltip_stop")
            : recording.countdownRemaining > 0
              ? tf("recording_tooltip_cancel_countdown")
              : tf("recording_tooltip_start"),
        `${tf("recording_source_mode")}: ${recordingSourceLabel}`,
        `${tf("recording_device")}: ${recordingDeviceLabel}`,
    ].join("\n");

    async function applyRecordingSettings(patch: Partial<RecordingSettings>) {
        try {
            await dispatch(saveRecordingSettings({ ...recording.settings, ...patch })).unwrap();
        } catch {
            // 快速设置失败时保持菜单关闭；详细错误仍可在录音设置对话框中查看。
        } finally {
            setRecordingMenuPos(null);
        }
    }

    function formatBpmValue(value: number): string {
        const normalized = Number(value);
        return Number.isFinite(normalized) ? String(normalized) : "120";
    }

    const [bpmText, setBpmText] = useState(() => formatBpmValue(s.bpm || 120));
    /** 用户正在输入时置位，阻止显示值变化覆写输入草稿。 */
    const [bpmDirty, setBpmDirty] = useState(false);

    // Tempo Map 存在时，BPM 显示为播放头位置的生效速度。
    const displayBpm = useMemo(() => {
        if (s.tempoMap && s.tempoMap.points.length > 0) {
            const at = tempoAtSec(s.tempoMap, s.playheadSec, {
                bpm: s.bpm,
                beatsPerBar: s.beats || 4,
            });
            return at.bpm;
        }
        return s.bpm || 120;
    }, [s.tempoMap, s.playheadSec, s.bpm, s.beats]);

    const displayBeats = useMemo(() => {
        if (s.tempoMap && s.tempoMap.points.length > 0) {
            const at = tempoAtSec(s.tempoMap, s.playheadSec, {
                bpm: s.bpm,
                beatsPerBar: s.beats || 4,
            });
            return at.numerator;
        }
        return Math.round(s.beats || 4);
    }, [s.tempoMap, s.playheadSec, s.bpm, s.beats]);

    // Tempo Map 存在时，拍号分母显示播放头位置的实际值（如 3/8、6/8）；否则为工程基准值。
    const displayDenominator = useMemo(() => {
        if (s.tempoMap && s.tempoMap.points.length > 0) {
            const at = tempoAtSec(s.tempoMap, s.playheadSec, {
                bpm: s.bpm,
                beatsPerBar: s.beats || 4,
            });
            return at.denominator;
        }
        return s.project.timeSignatureDenominator || 4;
    }, [s.tempoMap, s.playheadSec, s.bpm, s.beats, s.project.timeSignatureDenominator]);

    // 工程音阶（无 Tempo Map 时的显示与回退值）。
    const projectScaleLike = useMemo<ScaleLike | null>(
        () =>
            s.project?.useCustomScale && s.project?.customScale
                ? s.project.customScale.notes
                : (s.project?.baseScale ?? "C"),
        [s.project],
    );

    // Tempo Map 存在时，基准音阶显示播放头位置及以前最近变化点的生效音阶。
    const displayScale = useMemo<ScaleLike | null>(() => {
        if (s.tempoMap && s.tempoMap.points.length > 0) {
            return (
                effectiveScaleAtSec(s.tempoMap, s.playheadSec, projectScaleLike ?? undefined) ??
                null
            );
        }
        return projectScaleLike;
    }, [s.tempoMap, s.playheadSec, projectScaleLike]);

    /** 播放头位置最近变化点的自定义音阶名称（用于显示）。 */
    const tempoCustomScaleName = useMemo(() => {
        if (!s.tempoMap || s.tempoMap.points.length === 0) return null;
        for (let i = pointIndexAtSec(s.tempoMap, s.playheadSec); i >= 0; i -= 1) {
            const scale = s.tempoMap.points[i].scale;
            if (scale?.notes && scale.notes.length > 0) {
                return scale.name || scale.notes.join(", ");
            }
            if (scale?.key) return null;
        }
        return null;
    }, [s.tempoMap, s.playheadSec]);

    /** 当前显示音阶是否即工程自定义音阶（显示/选项归并用）。 */
    const displayScaleMatchesProjectCustom = useMemo(() => {
        if (!Array.isArray(displayScale)) return false;
        if (!s.project?.useCustomScale || !s.project?.customScale) return false;
        const a = displayScale;
        const b = s.project.customScale.notes;
        return a.length === b.length && a.every((v, i) => v === b[i]);
    }, [displayScale, s.project]);

    const showTempoCustomScaleItem =
        s.tempoMap != null &&
        s.tempoMap.points.length > 0 &&
        Array.isArray(displayScale) &&
        !displayScaleMatchesProjectCustom;

    const displayScaleSelectValue = Array.isArray(displayScale)
        ? displayScaleMatchesProjectCustom
            ? "__custom__"
            : "__tempo_custom__"
        : typeof displayScale === "string" &&
            (SCALE_KEYS as readonly string[]).includes(displayScale)
          ? displayScale
          : "__custom__";

    const baseScaleWheelOptions = [
        ...SCALE_KEYS,
        ...(s.project?.customScale ? (["__custom__"] as const) : []),
        ...(showTempoCustomScaleItem ? (["__tempo_custom__"] as const) : []),
        "__custom_dialog__",
    ];

    // 显示值变化时同步输入框（渲染期调整，避免 effect 级联渲染）。
    const displayBpmText = formatBpmValue(displayBpm);
    if (!bpmDirty && displayBpmText !== bpmText) {
        setBpmText(displayBpmText);
    }

    /**
     * Tempo Map 存在时：更新从播放头位置开始、往前寻找的最近一个变化点
     * （初始点即工程基准记录，同样参与更新；不再自动新建变化点）。
     */
    const updateTempoPointAtPlayhead = useCallback(
        (patch: {
            bpm?: number;
            timeSignature?: TempoTimeSignature | null;
            scale?: TempoMapScaleData | null;
        }) => {
            if (!s.tempoMap || s.tempoMap.points.length === 0) return null;
            const map = s.tempoMap;
            const idx = pointIndexAtSec(map, s.playheadSec);
            const nextMap = updateTempoPoint(map, map.points[idx].id, patch);
            dispatch(setTempoMap(nextMap));
            void dispatch(setTempoMapRemote(nextMap));
            return nextMap;
        },
        [s.tempoMap, s.playheadSec, dispatch],
    );

    /** 基准音阶变更：有 Tempo Map 时写最近变化点，否则写工程音阶。 */
    const applyBaseScale = useCallback(
        (next: ScaleLike | null, customName?: string) => {
            if (s.tempoMap && s.tempoMap.points.length > 0) {
                updateTempoPointAtPlayhead({ scale: scaleLikeToScaleData(next, customName) });
                return;
            }
            if (next == null) return;
            if (Array.isArray(next)) {
                if (s.project?.customScale) {
                    dispatch(setProjectCustomScaleRemote(s.project.customScale));
                }
                return;
            }
            if ((SCALE_KEYS as readonly string[]).includes(next as string)) {
                dispatch(setProjectBaseScaleRemote(next as (typeof SCALE_KEYS)[number]));
            }
        },
        [s.tempoMap, s.project, dispatch, updateTempoPointAtPlayhead],
    );

    /**
     * BPM 的**落地提交**（每帧最多一次）。
     *
     * 【为什么要合并】一次滚轮手势会产生几十个 wheel 事件，每个事件各提交一次
     * 意味着订阅方（标尺刻度、网格、波形）全量重算几十次 —— 用户看到的就是标尺
     * "抽搐"，且滚得越快越明显。合并到帧粒度后，手势期间画面仍然逐帧跟随
     * （手感不变），但每帧只重算一次。
     */
    const applyBpmCommit = useCallback(
        (value: number) => {
            if (s.tempoMap && s.tempoMap.points.length > 0) {
                updateTempoPointAtPlayhead({ bpm: value });
                return;
            }
            dispatch(setBpm(value));
            void dispatch(updateTransportBpm(value));
        },
        [s.tempoMap, updateTempoPointAtPlayhead, dispatch],
    );
    /** 提交体经 ref 现读：宿主回调只创建一次，闭包捕获会用到旧的 tempoMap。 */
    const applyBpmCommitRef = useRef(applyBpmCommit);
    applyBpmCommitRef.current = applyBpmCommit;
    const bpmCommitRef = useRef<FrameCommitter<number> | null>(null);
    if (bpmCommitRef.current === null) {
        bpmCommitRef.current = createFrameCommitter<number>((value) =>
            applyBpmCommitRef.current(value),
        );
    }
    useEffect(() => {
        const committer = bpmCommitRef.current;
        return () => {
            // 卸载前落地最后一次滚轮值：否则"最后一格"永远到不了 store。
            committer?.flush();
        };
    }, []);

    /**
     * 滚轮累积的起点。
     *
     * 【为什么不能直接读 `bpmText`】`setBpmText` 是异步的，同一帧内的多个 wheel
     * 事件会读到同一个旧值，各算出同一个"下一格"，连续滚动因此只前进一格。
     * 这里用 ref 持有手势内的当前值；手势结束（提交后静默 200ms）或外部改动时失效。
     */
    const wheelBpmBaseRef = useRef<number | null>(null);
    const wheelBpmAtRef = useRef(0);

    /**
     * BPM 输入的滚轮调值。
     *
     * 【为什么用非被动原生监听】React 的 `onWheel` 是 passive 的，里面的
     * `preventDefault()` 是空操作（本仓库其它数值滚轮控件都用 `useNonPassiveWheel`
     * 或 `useRangeWheelGuard`）：滚轮调值会同时滚动祖先容器，产生第二路视觉位移，
     * 并触发浏览器干预告警。
     */
    const attachBpmWheel = useNonPassiveWheel<HTMLInputElement>((e) => {
        // 【必须调用】换成非被动原生监听的意义正是让这两句生效：滚轮调值不得同时
        // 滚动祖先容器（ActionBar 自己是 overflow-x-auto），也不得被其它滚轮逻辑
        // 顺带处理。曾经在改写监听方式时把它们一起删掉，结果滚轮既改 BPM 又滚动
        // 容器，产生第二路视觉位移 —— 用户报告的"仍然抽搐"。
        e.preventDefault();
        e.stopPropagation();
        const now = performance.now();
        // 手势起点：距上一次滚轮超过 200ms，或尚未开始过手势。
        if (wheelBpmBaseRef.current === null || now - wheelBpmAtRef.current > 200) {
            const current = Number(bpmText);
            wheelBpmBaseRef.current = Number.isFinite(current) ? current : displayBpm;
        }
        wheelBpmAtRef.current = now;
        const direction = e.deltaY < 0 ? 1 : -1;
        const step = isModifierActive(paramFineAdjustKb, e) ? 0.1 : 1;
        const next = Math.round((wheelBpmBaseRef.current + direction * step) * 1000) / 1000;
        // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
        const clamped = clampBpm(next);
        if (Math.abs(clamped - wheelBpmBaseRef.current) < 1e-9) return; // 已到边界
        wheelBpmBaseRef.current = clamped;
        // 文本逐事件更新（手感不变，且它不触发标尺重算）；Redux 提交合并到每帧一次。
        setBpmText(formatBpmValue(clamped));
        setBpmDirty(false);
        bpmCommitRef.current?.schedule(clamped);
    });

    function commitBpm(nextText?: string) {
        const raw = (nextText ?? bpmText).trim();
        const next = Number(raw);
        setBpmDirty(false);
        // 键盘/失焦提交结束当前滚轮手势：下一次滚轮从新值重新累积。
        wheelBpmBaseRef.current = null;
        if (!Number.isFinite(next)) {
            setBpmText(formatBpmValue(displayBpm));
            return;
        }
        // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
        const clamped = clampBpm(next);
        if (s.tempoMap && s.tempoMap.points.length > 0) {
            updateTempoPointAtPlayhead({ bpm: clamped });
            setBpmText(formatBpmValue(clamped));
            return;
        }
        dispatch(setBpm(clamped));
        void dispatch(updateTransportBpm(clamped));
        setBpmText(formatBpmValue(clamped));
    }

    function formatRecordingTime(seconds: number): string {
        const total = Math.max(0, Math.floor(seconds));
        const minutes = Math.floor(total / 60);
        const secs = total % 60;
        return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }

    function recordingErrorMessage(code: string): string {
        // Backend errors may carry a `:detail` suffix (e.g.
        // "recording_error_wasapi_init:0x80004005"); localize the base key.
        const baseKey = code.split(":")[0] ?? code;
        const text = tf(baseKey);
        if (text && text !== baseKey) return text;
        return tf(
            code.startsWith("recording_error_stop")
                ? "recording_error_stop_failed"
                : "recording_error_start_failed",
        );
    }

    // Custom styles for Radix components to match Qt look
    // Note: Radix Themes handles a lot, but we might need overrides for exact pixel matching if needed.
    // For now, we use standard Radix "gray" theme which fits well.

    return (
        <Flex
            align="center"
            gap="3"
            className="h-qt-bar-main bg-qt-window border-b border-qt-border px-1 text-qt-text flex-nowrap overflow-x-auto overflow-y-hidden min-w-0 custom-scrollbar"
        >
            {/* BPM & Time */}
            <Flex align="center" gap="2" className="shrink-0">
                {/* Metronome */}
                <Box style={{ position: "relative" }} data-hs-context-menu>
                    <AppIconButton
                        active={s.metronomeEnabled}
                        // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                        emphasis="accent"
                        tooltip={t("action_metronome")}
                        icon={<MetronomeIcon />}
                        onClick={() => {
                            void dispatch(
                                updateMetronome({ metronomeEnabled: !s.metronomeEnabled }),
                            );
                        }}
                        onContextMenu={(event) => {
                            event.preventDefault();
                            setMetronomeMenuPos({ x: event.clientX, y: event.clientY });
                        }}
                    />
                    {metronomeMenuPos &&
                        createPortal(
                            <div
                                ref={metronomeMenuRef}
                                data-hs-context-menu="1"
                                className="hs-menu hs-menu--no-scroll"
                                style={{ left: metronomeMenuPos.x, top: metronomeMenuPos.y }}
                            >
                                <div className="hs-menu__label">{t("metronome_volume")}</div>
                                <div className="hs-menu__body flex items-center gap-2">
                                    {/*
                                     * 用 `AppSlider` 而不是裸 `<input type="range">`：滚轮步进
                                     * （粗 5% / 精细修饰键 1%）与"滚轮不带动祖先滚动"都由原语
                                     * 内建，`percent` 单位语义给出的正是这两个步长。
                                     *
                                     * 这里**不需要** `onPointerDown` 阻止冒泡：菜单的"点外面
                                     * 关闭"监听在 window 捕获阶段，且已经先判 `contains(target)`
                                     * 直接放行菜单内部的指针事件（见上方 effect）。
                                     *
                                     * 拖动期间逐帧 dispatch：节拍器音量必须**边拖边听得见**，
                                     * 而增益只经 `webApi.setMetronome` 到达引擎。调用链本身是
                                     * 串行的（`metronomeInvokeChain`），因此不会并发压垮 IPC。
                                     */}
                                    <AppSlider
                                        value={Math.round(s.metronomeGain * 100)}
                                        unit="percent"
                                        min={0}
                                        max={100}
                                        ariaLabel={t("metronome_volume")}
                                        onChange={(next) => {
                                            void dispatch(
                                                updateMetronome({ metronomeGain: next / 100 }),
                                            );
                                        }}
                                    />
                                    <AppSliderReadout>
                                        {Math.round(s.metronomeGain * 100)}%
                                    </AppSliderReadout>
                                </div>
                                <div className="hs-menu__separator" role="separator" />
                                <div className="hs-menu__label">{t("metronome_mode")}</div>
                                {(
                                    [
                                        ["grid", "metronome_mode_grid"],
                                        ["beat", "metronome_mode_beat"],
                                        ["bar", "metronome_mode_bar"],
                                    ] as const
                                ).map(([mode, key]) => (
                                    <button
                                        key={mode}
                                        type="button"
                                        className="hs-menu__item"
                                        onClick={() => {
                                            void dispatch(updateMetronome({ metronomeMode: mode }));
                                            setMetronomeMenuPos(null);
                                        }}
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <span className="hs-menu__label-text">{t(key)}</span>
                                        <span className="hs-menu__trail">
                                            {s.metronomeMode === mode ? (
                                                <span className="hs-menu__check">
                                                    <CheckIcon />
                                                </span>
                                            ) : null}
                                        </span>
                                    </button>
                                ))}
                                <div className="hs-menu__separator" role="separator" />
                                <div className="hs-menu__label">{t("metronome_sound")}</div>
                                {(
                                    [
                                        ["click", "metronome_sound_click"],
                                        ["woodblock", "metronome_sound_woodblock"],
                                        ["beep", "metronome_sound_beep"],
                                    ] as const
                                ).map(([sound, key]) => (
                                    <button
                                        key={sound}
                                        type="button"
                                        className="hs-menu__item"
                                        onClick={() => {
                                            void dispatch(
                                                updateMetronome({ metronomeSound: sound }),
                                            );
                                            setMetronomeMenuPos(null);
                                        }}
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <span className="hs-menu__label-text">{t(key)}</span>
                                        <span className="hs-menu__trail">
                                            {s.metronomeSound === sound ? (
                                                <span className="hs-menu__check">
                                                    <CheckIcon />
                                                </span>
                                            ) : null}
                                        </span>
                                    </button>
                                ))}
                                <div className="hs-menu__separator" role="separator" />
                                <button
                                    type="button"
                                    className="hs-menu__item"
                                    onClick={() => {
                                        void dispatch(
                                            updateMetronome({
                                                metronomeAccent: !s.metronomeAccent,
                                            }),
                                        );
                                    }}
                                    onPointerDown={(e) => e.stopPropagation()}
                                >
                                    <span className="hs-menu__label-text">
                                        {t("metronome_accent")}
                                    </span>
                                    <span className="hs-menu__trail">
                                        {s.metronomeAccent ? (
                                            <span className="hs-menu__check">
                                                <CheckIcon />
                                            </span>
                                        ) : null}
                                    </span>
                                </button>
                            </div>,
                            document.body,
                        )}
                </Box>
                <span className="hs-type-muted">{t("common_bpm")}:</span>
                <TextField.Root
                    ref={attachBpmWheel}
                    disabled={isPluginMode()}
                    size="1"
                    value={bpmText}
                    data-tooltip={
                        isPluginMode()
                            ? dawControlledReason()
                            : s.tempoMap && s.tempoMap.points.length > 0
                              ? tf("tempo_map_actionbar_tip")
                              : undefined
                    }
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        setBpmDirty(true);
                        setBpmText(e.target.value);
                    }}
                    onBlur={() => commitBpm()}
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                        if (e.key === "Enter") {
                            e.preventDefault();
                            commitBpm();
                            (e.currentTarget as HTMLInputElement).blur();
                        } else if (e.key === "Escape") {
                            e.preventDefault();
                            setBpmDirty(false);
                            setBpmText(formatBpmValue(displayBpm));
                            (e.currentTarget as HTMLInputElement).blur();
                        }
                    }}
                    style={{
                        width: 60,
                        textAlign: "center",
                        backgroundColor: "var(--qt-base)",
                    }}
                />
                <span className="hs-type-muted">{t("time_signature")}:</span>
                <Flex align="center" gap="1">
                    <TextField.Root
                        size="1"
                        type="number"
                        value={String(displayBeats)}
                        disabled={isPluginMode()}
                        data-tooltip={
                            isPluginMode()
                                ? dawControlledReason()
                                : s.tempoMap && s.tempoMap.points.length > 0
                                  ? tf("tempo_map_actionbar_tip")
                                  : undefined
                        }
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            const raw = e.target.value.trim();
                            const parsed = Number(raw);
                            if (!Number.isFinite(parsed)) return;
                            // Clamp locally to avoid sending huge values to backend
                            const clamped = Math.min(32, Math.max(1, Math.round(parsed)));
                            // 与显示值比较（Tempo Map 下为播放头位置生效值）。
                            if (clamped === Math.round(displayBeats)) return;
                            if (s.tempoMap && s.tempoMap.points.length > 0) {
                                updateTempoPointAtPlayhead({
                                    timeSignature: {
                                        numerator: clamped,
                                        denominator: displayDenominator,
                                    },
                                });
                                return;
                            }
                            void dispatch(
                                setProjectTimelineSettingsRemote({
                                    beatsPerBar: clamped,
                                    timeSignatureDenominator: displayDenominator,
                                    gridSize: s.grid,
                                }),
                            );
                        }}
                        onWheel={(e: React.WheelEvent<HTMLInputElement>) => {
                            e.preventDefault();
                            e.stopPropagation();
                            const direction = e.deltaY < 0 ? 1 : -1;
                            // 基础值取播放头位置的生效值（Tempo Map 下为最近变化点），
                            // 与 BPM / 基准音阶一致。
                            const current = Math.max(1, Math.min(32, Math.round(displayBeats)));
                            const next = Math.max(1, Math.min(32, current + direction));
                            if (next === current) return;
                            if (s.tempoMap && s.tempoMap.points.length > 0) {
                                updateTempoPointAtPlayhead({
                                    timeSignature: {
                                        numerator: next,
                                        denominator: displayDenominator,
                                    },
                                });
                                return;
                            }
                            void dispatch(
                                setProjectTimelineSettingsRemote({
                                    beatsPerBar: next,
                                    timeSignatureDenominator: displayDenominator,
                                    gridSize: s.grid,
                                }),
                            );
                        }}
                        style={{
                            width: 42,
                            textAlign: "center",
                            backgroundColor: "var(--qt-base)",
                        }}
                    />
                    <span className="hs-type-muted">/</span>
                    <Select.Root
                        size="1"
                        value={String(displayDenominator)}
                        disabled={isPluginMode()}
                        onValueChange={(v) => {
                            const next = Number(v) || 4;
                            if (next === displayDenominator) return;
                            if (s.tempoMap && s.tempoMap.points.length > 0) {
                                updateTempoPointAtPlayhead({
                                    timeSignature: {
                                        numerator: displayBeats,
                                        denominator: next,
                                    },
                                });
                                return;
                            }
                            void dispatch(
                                setProjectTimelineSettingsRemote({
                                    beatsPerBar: s.beats,
                                    timeSignatureDenominator: next,
                                    gridSize: s.grid,
                                }),
                            );
                        }}
                    >
                        <Select.Trigger
                            data-tooltip={isPluginMode() ? dawControlledReason() : undefined}
                            style={{
                                width: 48,
                                backgroundColor: "var(--qt-base)",
                                justifyContent: "center",
                            }}
                            onWheel={(event) => {
                                applySelectWheelChange({
                                    event,
                                    currentValue: String(displayDenominator),
                                    options: TEMPO_DENOMINATORS.map((d) => String(d)),
                                    onChange: (v) => {
                                        const next = Number(v) || 4;
                                        if (next === displayDenominator) return;
                                        if (s.tempoMap && s.tempoMap.points.length > 0) {
                                            updateTempoPointAtPlayhead({
                                                timeSignature: {
                                                    numerator: displayBeats,
                                                    denominator: next,
                                                },
                                            });
                                            return;
                                        }
                                        void dispatch(
                                            setProjectTimelineSettingsRemote({
                                                beatsPerBar: s.beats,
                                                timeSignatureDenominator: next,
                                                gridSize: s.grid,
                                            }),
                                        );
                                    },
                                });
                            }}
                        />
                        <Select.Content>
                            {TEMPO_DENOMINATORS.map((d) => (
                                <Select.Item key={d} value={String(d)}>
                                    {d}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                </Flex>

                <span className="hs-type-muted">{t("common_grid")}:</span>
                <Select.Root
                    value={s.grid}
                    size="1"
                    onValueChange={(v) => {
                        void dispatch(
                            setProjectTimelineSettingsRemote({
                                beatsPerBar: s.beats,
                                timeSignatureDenominator: displayDenominator,
                                gridSize: v,
                            }),
                        );
                    }}
                >
                    <Select.Trigger
                        style={{ backgroundColor: "var(--qt-base)" }}
                        onWheel={(event) => {
                            applySelectWheelChange({
                                event,
                                currentValue: s.grid,
                                options: [
                                    "1/1",
                                    "1/2",
                                    "1/4",
                                    "1/8",
                                    "1/16",
                                    "1/32",
                                    "1/64",
                                    "1/2d",
                                    "1/4d",
                                    "1/8d",
                                    "1/16d",
                                    "1/32d",
                                    "1/64d",
                                    "1/2t",
                                    "1/4t",
                                    "1/8t",
                                    "1/16t",
                                    "1/32t",
                                    "1/64t",
                                ],
                                onChange: (next) => {
                                    void dispatch(
                                        setProjectTimelineSettingsRemote({
                                            beatsPerBar: s.beats,
                                            timeSignatureDenominator: displayDenominator,
                                            gridSize: next,
                                        }),
                                    );
                                },
                            });
                        }}
                    />
                    <Select.Content style={{ maxHeight: "none", overflow: "visible" }}>
                        <Select.Group>
                            <Select.Label>{tf("grid_note_normal")}</Select.Label>
                            <Select.Item value="1/1">1/1</Select.Item>
                            <Select.Item value="1/2">1/2</Select.Item>
                            <Select.Item value="1/4">1/4</Select.Item>
                            <Select.Item value="1/8">1/8</Select.Item>
                            <Select.Item value="1/16">1/16</Select.Item>
                            <Select.Item value="1/32">1/32</Select.Item>
                            <Select.Item value="1/64">1/64</Select.Item>
                        </Select.Group>
                        <Select.Separator />
                        <Select.Group>
                            <Select.Label>{tf("grid_note_dotted")}</Select.Label>
                            <Select.Item value="1/2d">1/2.</Select.Item>
                            <Select.Item value="1/4d">1/4.</Select.Item>
                            <Select.Item value="1/8d">1/8.</Select.Item>
                            <Select.Item value="1/16d">1/16.</Select.Item>
                            <Select.Item value="1/32d">1/32.</Select.Item>
                            <Select.Item value="1/64d">1/64.</Select.Item>
                        </Select.Group>
                        <Select.Separator />
                        <Select.Group>
                            <Select.Label>{tf("grid_note_triplet")}</Select.Label>
                            <Select.Item value="1/2t">1/2t</Select.Item>
                            <Select.Item value="1/4t">1/4t</Select.Item>
                            <Select.Item value="1/8t">1/8t</Select.Item>
                            <Select.Item value="1/16t">1/16t</Select.Item>
                            <Select.Item value="1/32t">1/32t</Select.Item>
                            <Select.Item value="1/64t">1/64t</Select.Item>
                        </Select.Group>
                    </Select.Content>
                </Select.Root>
                <span className="hs-type-muted">{t("base_scale")}:</span>
                {/* 【为什么插件里也能改】音阶是 HiFiShifter 自有的设置 —— REAPER
                    没有工程调号概念，宿主不提供它。插件把它持久化在用户设置里
                    （`set_project_base_scale`），因此这个选择器是真的能用的。
                    此前按模式一刀切禁用它，等于把一整套锚定音阶的功能（音高吸附、
                    级数渲染、渲染缓存键）一起关掉了。 */}
                <Select.Root
                    value={displayScaleSelectValue}
                    size="1"
                    onValueChange={(v) => {
                        if (v === "__custom_dialog__") {
                            setCustomScaleOpen(true);
                            return;
                        }
                        if (v === "__custom__") {
                            if (s.project?.customScale) {
                                applyBaseScale(
                                    s.project.customScale.notes,
                                    s.project.customScale.name,
                                );
                            }
                            return;
                        }
                        if (v === "__tempo_custom__") {
                            if (Array.isArray(displayScale)) {
                                applyBaseScale(displayScale, tempoCustomScaleName ?? undefined);
                            }
                            return;
                        }
                        if ((SCALE_KEYS as readonly string[]).includes(v)) {
                            applyBaseScale(v as (typeof SCALE_KEYS)[number]);
                        }
                    }}
                >
                    <Select.Trigger
                        style={{ backgroundColor: "var(--qt-base)" }}
                        onWheel={(event) => {
                            applySelectWheelChange({
                                event,
                                currentValue: displayScaleSelectValue,
                                options: baseScaleWheelOptions,
                                onChange: (next) => {
                                    if (next === "__custom_dialog__") {
                                        setCustomScaleOpen(true);
                                        return;
                                    }
                                    if (next === "__custom__") {
                                        if (s.project?.customScale) {
                                            applyBaseScale(
                                                s.project.customScale.notes,
                                                s.project.customScale.name,
                                            );
                                        }
                                        return;
                                    }
                                    if (next === "__tempo_custom__") {
                                        if (Array.isArray(displayScale)) {
                                            applyBaseScale(
                                                displayScale,
                                                tempoCustomScaleName ?? undefined,
                                            );
                                        }
                                        return;
                                    }
                                    if ((SCALE_KEYS as readonly string[]).includes(next)) {
                                        applyBaseScale(next as (typeof SCALE_KEYS)[number]);
                                    }
                                },
                            });
                        }}
                    />
                    <Select.Content style={{ maxHeight: "none", overflow: "visible" }}>
                        <Select.Group>
                            {SCALE_KEYS.map((k) => (
                                <Select.Item key={k} value={k}>
                                    {SCALE_LABELS[k]}
                                </Select.Item>
                            ))}
                        </Select.Group>
                        {showTempoCustomScaleItem ? (
                            <>
                                <Select.Separator />
                                <Select.Group>
                                    <Select.Item value="__tempo_custom__">
                                        {tempoCustomScaleName ?? tf("custom_scale_short")}
                                    </Select.Item>
                                </Select.Group>
                            </>
                        ) : null}
                        {s.project?.customScale ? (
                            <>
                                <Select.Separator />
                                <Select.Group>
                                    <Select.Item value="__custom__">
                                        {`${tf("custom_scale_label")}: ${s.project.customScale.name}`}
                                    </Select.Item>
                                </Select.Group>
                            </>
                        ) : null}
                        <Select.Separator />
                        <Select.Group>
                            <Select.Item value="__custom_dialog__">
                                {tf("custom_scale_action")}
                            </Select.Item>
                        </Select.Group>
                    </Select.Content>
                </Select.Root>
            </Flex>

            <AppToolbarSeparator />

            {/* Transport */}
            <Flex gap="1" className="shrink-0">
                <Button
                    variant="soft"
                    color="gray"
                    size="1"
                    onClick={() => {
                        dispatch(stopAudioPlayback({ restoreAnchor: true }));
                    }}
                    data-tooltip={isPluginMode() ? t("plugin_transport_stop") : t("action_stop")}
                    disabled={isPluginMode() && !canControlHostTransport()}
                >
                    <StopIcon />
                </Button>
                <IconButton
                    variant="solid"
                    size="1"
                    onClick={() => {
                        if (isPlaying) {
                            dispatch(stopAudioPlayback());
                            return;
                        }
                        dispatch(playOriginal());
                    }}
                    data-tooltip={
                        isPluginMode()
                            ? t("plugin_transport_play")
                            : isPlaying
                              ? tf("action_pause")
                              : t("action_play_out")
                    }
                    disabled={isPluginMode() && !canControlHostTransport()}
                >
                    {isPlaying ? <PauseIcon /> : <PlayIcon />}
                </IconButton>
                <Box style={{ position: "relative" }} data-hs-context-menu>
                    <IconButton
                        size="1"
                        /* 录音语义色：待机态就用 soft 红点（旧版待机是灰色 ghost 点，
                           录音键的"红"只在录制中才出现，语义色缺失） */
                        variant={recording.active ? "solid" : "soft"}
                        color="red"
                        data-tooltip={recordingTooltip}
                        disabled={
                            isPluginMode() || (recording.busy && recording.countdownRemaining === 0)
                        }
                        onClick={() => {
                            if (recording.active) {
                                void dispatch(stopRecordingFlow());
                            } else if (recording.countdownRemaining > 0) {
                                void dispatch(cancelRecordingCountdown());
                            } else {
                                void dispatch(startRecordingFlow());
                            }
                        }}
                        onContextMenu={(event) => {
                            event.preventDefault();
                            if (isPluginMode()) return;
                            setRecordingMenuPos({ x: event.clientX, y: event.clientY });
                            void dispatch(loadRecordingSettings());
                            // 每次打开菜单都强制重新枚举设备/应用，
                            // 避免展示上次加载的过时列表（设备热插拔、应用退出等）。
                            void dispatch(loadRecordingDevices({ force: true }));
                            void dispatch(loadRecordingApps({ force: true }));
                        }}
                    >
                        {recording.active ? (
                            <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
                                <rect x="4" y="4" width="7" height="7" rx="1.2" />
                            </svg>
                        ) : (
                            <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
                                <circle cx="7.5" cy="7.5" r="4.2" />
                            </svg>
                        )}
                    </IconButton>
                    {recordingMenuPos && (
                        <AppContextMenu
                            x={recordingMenuPos.x}
                            y={recordingMenuPos.y}
                            ariaLabel={tf("recording_source_mode")}
                            onClose={() => setRecordingMenuPos(null)}
                            items={[
                                {
                                    key: "mode-heading",
                                    heading: true,
                                    label: tf("recording_source_mode"),
                                },
                                {
                                    key: "mode-device",
                                    label: tf("recording_mode_device"),
                                    checked: recording.settings.captureMode === "device",
                                    onSelect: () =>
                                        void applyRecordingSettings({ captureMode: "device" }),
                                },
                                {
                                    key: "mode-loopback",
                                    label: tf("recording_mode_loopback"),
                                    checked: recording.settings.captureMode === "loopback",
                                    onSelect: () =>
                                        void applyRecordingSettings({ captureMode: "loopback" }),
                                },
                                {
                                    key: "mode-application",
                                    label: tf("recording_mode_application"),
                                    checked: recording.settings.captureMode === "application",
                                    onSelect: () =>
                                        void applyRecordingSettings({ captureMode: "application" }),
                                },
                                {
                                    key: "source-heading",
                                    heading: true,
                                    separatorBefore: true,
                                    label: tf(
                                        recording.settings.captureMode === "application"
                                            ? "recording_application"
                                            : "recording_device",
                                    ),
                                },
                                ...(recording.settings.captureMode === "device"
                                    ? [
                                          {
                                              key: "device-default",
                                              label: tf("recording_device_default"),
                                              checked:
                                                  recording.settings.sourceDevice === "default",
                                              onSelect: () =>
                                                  void applyRecordingSettings({
                                                      sourceDevice: "default",
                                                  }),
                                          },
                                          ...recording.devices
                                              .filter(
                                                  (device) =>
                                                      !device.isLoopback && device.id !== "default",
                                              )
                                              .map((device) => ({
                                                  key: device.id,
                                                  label: device.name,
                                                  checked:
                                                      recording.settings.sourceDevice === device.id,
                                                  onSelect: () =>
                                                      void applyRecordingSettings({
                                                          sourceDevice: device.id,
                                                      }),
                                              })),
                                      ]
                                    : recording.settings.captureMode === "loopback"
                                      ? [
                                            {
                                                key: "loopback-default",
                                                label: tf("recording_loopback_default"),
                                                checked:
                                                    recording.settings.loopbackDevice === "default",
                                                onSelect: () =>
                                                    void applyRecordingSettings({
                                                        loopbackDevice: "default",
                                                    }),
                                            },
                                            ...recording.devices
                                                .filter(
                                                    (device) =>
                                                        device.isLoopback &&
                                                        device.id !== "loopback:default",
                                                )
                                                .map((device) => ({
                                                    key: device.id,
                                                    label: device.name,
                                                    checked:
                                                        recording.settings.loopbackDevice ===
                                                        device.id,
                                                    onSelect: () =>
                                                        void applyRecordingSettings({
                                                            loopbackDevice: device.id,
                                                        }),
                                                })),
                                        ]
                                      : [
                                            ...(recording.settings.captureAppId &&
                                            !recording.apps.some(
                                                (app) => app.id === recording.settings.captureAppId,
                                            )
                                                ? [
                                                      {
                                                          key: recording.settings.captureAppId,
                                                          label:
                                                              recording.settings.captureAppName ||
                                                              recording.settings.captureAppId,
                                                          checked: true,
                                                          onSelect: () =>
                                                              void applyRecordingSettings({
                                                                  captureAppId:
                                                                      recording.settings
                                                                          .captureAppId,
                                                                  captureAppName:
                                                                      recording.settings
                                                                          .captureAppName,
                                                                  captureAppProcess:
                                                                      recording.settings
                                                                          .captureAppProcess,
                                                              }),
                                                      },
                                                  ]
                                                : []),
                                            ...recording.apps.map((app) => ({
                                                key: app.id,
                                                label: app.name,
                                                checked: recording.settings.captureAppId === app.id,
                                                onSelect: () =>
                                                    void applyRecordingSettings({
                                                        captureAppId: app.id,
                                                        captureAppName: app.name,
                                                        captureAppProcess: app.processName,
                                                    }),
                                            })),
                                        ]),
                                {
                                    key: "settings",
                                    label: tf("recording_context_settings"),
                                    separatorBefore: true,
                                    onSelect: () => {
                                        setRecordingMenuPos(null);
                                        setRecordingSettingsOpen(true);
                                    },
                                },
                            ]}
                        />
                    )}
                </Box>
                {recording.active || recording.countdownRemaining > 0 ? (
                    <Flex align="center" gap="1" className="shrink-0">
                        <span
                            className="hs-type-label tabular-nums"
                            style={
                                recording.active ? { color: "var(--qt-danger-text)" } : undefined
                            }
                        >
                            {recording.countdownRemaining > 0
                                ? `-${recording.countdownRemaining}`
                                : formatRecordingTime(recording.elapsedSec)}
                        </span>
                        <div
                            style={{
                                width: 48,
                                height: 6,
                                borderRadius: "var(--qt-radius-pill)",
                                background: "var(--qt-border)",
                                overflow: "hidden",
                                flexShrink: 0,
                            }}
                        >
                            <div
                                style={{
                                    width: `${Math.min(100, Math.round((recording.level || 0) * 100))}%`,
                                    height: "100%",
                                    background:
                                        recording.level > 0.98
                                            ? "var(--qt-danger-text)"
                                            : "var(--qt-danger-border)",
                                    transition: "width 80ms linear",
                                }}
                            />
                        </div>
                    </Flex>
                ) : null}
                {recording.error ? (
                    <span
                        className="hs-type-label truncate"
                        data-tooltip={recording.error}
                        style={{ maxWidth: 220, color: "var(--qt-danger-text)" }}
                    >
                        {recordingErrorMessage(recording.error)}
                    </span>
                ) : null}
            </Flex>

            <AppToolbarSeparator />

            {/* ── 撤销 / 重做 ──────────────────────────────────────────
                独立成组、两侧以分隔线与其他按钮隔开；右键打开「操作记录」
                （非模态浮动窗口，打开期间照常编辑轨道，条目实时刷新）。 */}
            <Flex gap="1" className="shrink-0">
                <IconButton
                    ref={undoButtonRef}
                    size="1"
                    variant="ghost"
                    disabled={s.historyUndoDepth <= 0}
                    tabIndex={-1}
                    data-tooltip={`${t("menu_undo")} (${formatKeybindingList(undoShortcutKb, "")})`}
                    onClick={() => {
                        void dispatch(undoRemote());
                    }}
                    onContextMenu={(event) => {
                        event.preventDefault();
                        openHistoryPanel();
                    }}
                >
                    <UndoIcon />
                </IconButton>
                <IconButton
                    size="1"
                    variant="ghost"
                    disabled={s.historyRedoDepth <= 0}
                    tabIndex={-1}
                    data-tooltip={`${t("menu_redo")} (${formatKeybindingList(redoShortcutKb, "")})`}
                    onClick={() => {
                        void dispatch(redoRemote());
                    }}
                    onContextMenu={(event) => {
                        event.preventDefault();
                        openHistoryPanel();
                    }}
                >
                    <RedoIcon />
                </IconButton>
            </Flex>

            <AppToolbarSeparator />

            {/* File Browser Toggle */}
            <Flex gap="1" className="shrink-0">
                <AppIconButton
                    active={fileBrowserVisible}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("fb_title")}
                    onClick={() => togglePanelVisible(dispatch, store.getState, PANEL_FILE_BROWSER)}
                    icon={
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 15 15"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="M2 3.5C2 3.22386 2.22386 3 2.5 3H5.29289L6.64645 4.35355C6.74021 4.44732 6.86739 4.5 7 4.5H12.5C12.7761 4.5 13 4.72386 13 5V11.5C13 11.7761 12.7761 12 12.5 12H2.5C2.22386 12 2 11.7761 2 11.5V3.5Z"
                                fill="currentColor"
                            />
                        </svg>
                    }
                />
                <AppIconButton
                    active={notebookVisible}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={t("common_notebook")}
                    onClick={() => togglePanelVisible(dispatch, store.getState, PANEL_NOTEBOOK)}
                    icon={<Pencil1Icon />}
                />
            </Flex>

            <AppToolbarSeparator />

            {/* Toolbar Toggles */}
            <Flex align="center" gap="1" className="shrink-0">
                {/* Auto Crossfade */}
                <AppIconButton
                    active={s.autoCrossfadeEnabled}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("auto_crossfade")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleAutoCrossfade());
                        void dispatch(persistUiSettings());
                    }}
                    icon={
                        /* X icon for crossfade */
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 15 15"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="M2 12L7.5 3L13 12"
                                stroke="currentColor"
                                strokeWidth="1.2"
                                fill="none"
                            />
                            <path
                                d="M2 3L7.5 12L13 3"
                                stroke="currentColor"
                                strokeWidth="1.2"
                                fill="none"
                                opacity="0.5"
                            />
                        </svg>
                    }
                />

                {/* Split Transition */}
                <AppIconButton
                    active={s.splitTransitionEnabled}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("split_transition_tooltip")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleSplitTransition());
                        void dispatch(persistUiSettings());
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        setSplitTransitionOpen(true);
                    }}
                    icon={
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 15 15"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path d="M7.5 1.5V13.5" stroke="currentColor" strokeWidth="1.2" />
                            <path
                                d="M3.5 3.5L7.5 5.5L3.5 7.5Z"
                                fill="currentColor"
                                opacity="0.85"
                            />
                            <path
                                d="M11.5 7.5L7.5 9.5L11.5 11.5Z"
                                fill="currentColor"
                                opacity="0.45"
                            />
                        </svg>
                    }
                />

                {/* Snap */}
                <AppIconButton
                    active={effectiveSnapVisual}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={`${tf("common_snap")}${
                        snapGestureActive && snapToggleHeld
                            ? ` · ${tf("common_snap")}: ${tf("snap_toggle_inverted")}`
                            : ""
                    }`}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleSnap());
                        void dispatch(persistUiSettings());
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        setSnapSettingsOpen(true);
                    }}
                    icon={
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="m6 15-4-4 6.75-6.77a7.79 7.79 0 0 1 11 11L13 22l-4-4 6.39-6.36a2.14 2.14 0 0 0-3-3L6 15Z"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            />
                            <path
                                d="m5 8 4 4"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            />
                            <path
                                d="m12 15 4 4"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            />
                        </svg>
                    }
                />

                {/* Ripple Edit (Auto Follow) */}
                <RippleModeButton
                    mode={s.rippleMode}
                    onCycle={() => {
                        dispatch(cycleRippleMode());
                        void dispatch(persistUiSettings());
                    }}
                    onSelect={(next) => {
                        dispatch(setRippleMode(next));
                        void dispatch(persistUiSettings());
                    }}
                />

                <AppToolbarSeparator />

                {/* Playhead Zoom */}
                <AppIconButton
                    active={s.playheadZoomEnabled}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("playhead_zoom")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(togglePlayheadZoom());
                        void dispatch(persistUiSettings());
                    }}
                    icon={
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 15 15"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path d="M7.5 2V13" stroke="currentColor" strokeWidth="1.2" />
                            <path d="M6 3.5L7.5 2L9 3.5" stroke="currentColor" strokeWidth="1" />
                            <path
                                d="M5.5 5.5L4 7.5L5.5 9.5"
                                stroke="currentColor"
                                strokeWidth="1.2"
                            />
                            <path
                                d="M9.5 5.5L11 7.5L9.5 9.5"
                                stroke="currentColor"
                                strokeWidth="1.2"
                            />
                            <path
                                d="M3 12H12"
                                stroke="currentColor"
                                strokeWidth="0.8"
                                opacity="0.5"
                            />
                        </svg>
                    }
                />

                {/* Auto Scroll (horizontal arrows) */}
                <AppIconButton
                    active={s.autoScrollEnabled}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("auto_scroll")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleAutoScroll());
                        void dispatch(persistUiSettings());
                    }}
                    icon={<DoubleArrowRightIcon width="15" height="15" />}
                />

                <AppToolbarSeparator />

                <AppIconButton
                    active={s.paramEditorSeekPlayheadEnabled}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("param_editor_seek_playhead")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleParamEditorSeekPlayhead());
                        void dispatch(persistUiSettings());
                    }}
                    icon={
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 15 15"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="M2 2.5H13"
                                stroke="currentColor"
                                strokeWidth="0.8"
                                opacity="0.5"
                            />
                            <path
                                d="M2 12.5H13"
                                stroke="currentColor"
                                strokeWidth="0.8"
                                opacity="0.5"
                            />
                            <path d="M7.5 3.5V11.5" stroke="currentColor" strokeWidth="1.2" />
                            <path d="M6 4.5L7.5 3L9 4.5" stroke="currentColor" strokeWidth="1" />
                            <path
                                d="M7.8 8.2C8.9 8.2 9.8 9.1 9.8 10.2C9.8 11.3 8.9 12.2 7.8 12.2C6.9 12.2 6.2 11.6 6 10.8H7.8V8.2Z"
                                fill="currentColor"
                            />
                        </svg>
                    }
                />

                {/* Allow timeline clicks to switch the parameter editor track */}
                <AppIconButton
                    active={s.paramEditorTimelineClickSelectTrackEnabled}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("param_editor_timeline_click_select_track")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleParamEditorTimelineClickSelectTrack());
                        void dispatch(persistUiSettings());
                    }}
                    icon={
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 15 15"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <defs>
                                <marker
                                    id="hs-track-switch-arrow"
                                    viewBox="0 0 6 6"
                                    refX="3"
                                    refY="3"
                                    markerWidth="5"
                                    markerHeight="5"
                                    orient="auto-start-reverse"
                                >
                                    <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" />
                                </marker>
                            </defs>
                            <rect
                                x="1.5"
                                y="2"
                                width="8"
                                height="3"
                                rx="1"
                                stroke="currentColor"
                                strokeWidth="1"
                            />
                            <rect
                                x="5.5"
                                y="10"
                                width="8"
                                height="3"
                                rx="1"
                                stroke="currentColor"
                                strokeWidth="1"
                            />
                            <line
                                x1="9.5"
                                y1="4.5"
                                x2="5.5"
                                y2="10.5"
                                stroke="currentColor"
                                strokeWidth="1"
                                markerStart="url(#hs-track-switch-arrow)"
                                markerEnd="url(#hs-track-switch-arrow)"
                            />
                        </svg>
                    }
                />

                <AppToolbarSeparator />

                {/* Ignore Grouping (broken chain) */}
                <AppIconButton
                    active={s.ignoreGrouping}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("ignore_grouping")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleIgnoreGrouping());
                        void dispatch(persistUiSettings());
                    }}
                    icon={
                        <svg
                            width="15"
                            height="15"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        >
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                            <line
                                x1="2"
                                y1="2"
                                x2="22"
                                y2="22"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                opacity="0.7"
                            />
                        </svg>
                    }
                />
            </Flex>

            {/* Pitch Snap Settings Dialog */}
            {pitchSnapOpen && (
                <PitchSnapSettingsDialog open={pitchSnapOpen} onOpenChange={setPitchSnapOpen} />
            )}

            {splitTransitionOpen && (
                <SplitTransitionSettingsDialog
                    open={splitTransitionOpen}
                    onOpenChange={setSplitTransitionOpen}
                />
            )}

            {customScaleOpen && (
                <CustomScaleDialog open={customScaleOpen} onOpenChange={setCustomScaleOpen} />
            )}

            <RecordingSettingsDialog
                open={recordingSettingsOpen}
                onOpenChange={setRecordingSettingsOpen}
            />

            {snapSettingsOpen && (
                <SnapGridSettingsDialog
                    open={snapSettingsOpen}
                    onOpenChange={setSnapSettingsOpen}
                />
            )}

            {/* Snap Context Menu removed: right-click opens the settings dialog above. */}
        </Flex>
    );
}

// ── 波纹编辑（自动跟进）按钮 ──────────────────────────────────────

/** 波纹编辑图标：一组“被向前推的剪辑块”+ 右侧箭头，表达后续剪辑自动跟进。
 *  `multiTrack` 为 true（全部轨道模式）时显示两行剪辑块，否则显示单行（按轨道模式）。
 */
function RippleIcon({ multiTrack }: { multiTrack: boolean }) {
    const rows = multiTrack ? [2.9, 6.9] : [4.9];
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            {rows.map((y) => (
                <g key={y}>
                    <rect
                        x="1.6"
                        y={y}
                        width="2.9"
                        height="2.4"
                        rx="0.6"
                        fill="currentColor"
                        opacity="0.45"
                    />
                    <rect
                        x="5.1"
                        y={y}
                        width="2.9"
                        height="2.4"
                        rx="0.6"
                        fill="currentColor"
                        opacity="0.75"
                    />
                    <rect x="8.6" y={y} width="2.9" height="2.4" rx="0.6" fill="currentColor" />
                </g>
            ))}
            <path
                d="M1.6 12.4H11.6"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
            />
            <path
                d="M8.9 10.5L11.6 12.4L8.9 14.3"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

/** 波纹模式选择菜单（右键打开），与 Snap / 分割过渡的右键菜单行为一致。 */
function RippleModeMenu({
    x,
    y,
    mode,
    onChange,
    onClose,
}: {
    x: number;
    y: number;
    mode: "off" | "track" | "all";
    onChange: (mode: "off" | "track" | "all") => void;
    onClose: () => void;
}) {
    const { tf } = useI18n();
    const options: Array<{ value: "off" | "track" | "all"; label: string }> = [
        { value: "off", label: tf("ripple_mode_off") as string },
        { value: "track", label: tf("ripple_mode_track") as string },
        { value: "all", label: tf("ripple_mode_all") as string },
    ];

    return (
        <AppContextMenu
            x={x}
            y={y}
            onClose={onClose}
            items={options.map((opt) => ({
                key: opt.value,
                label: opt.label,
                checked: mode === opt.value,
                onSelect: () => onChange(opt.value),
            }))}
        />
    );
}

/** 波纹编辑工具栏按钮：左键三态循环切换，右键打开模式菜单。 */
function RippleModeButton({
    mode,
    onCycle,
    onSelect,
}: {
    mode: "off" | "track" | "all";
    onCycle: () => void;
    onSelect: (mode: "off" | "track" | "all") => void;
}) {
    const { tf } = useI18n();
    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);

    return (
        <>
            <AppIconButton
                active={mode !== "off"}
                // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                emphasis="accent"
                tooltip={(tf(`ripple_tooltip_${mode}`) as string) ?? tf("common_ripple")}
                tabIndex={-1}
                onClick={onCycle}
                onContextMenu={(e) => {
                    e.preventDefault();
                    setMenu({ x: e.clientX, y: e.clientY });
                }}
                icon={<RippleIcon multiTrack={mode === "all"} />}
            />
            {menu && (
                <RippleModeMenu
                    x={menu.x}
                    y={menu.y}
                    mode={mode}
                    onChange={(next) => {
                        onSelect(next);
                        setMenu(null);
                    }}
                    onClose={() => setMenu(null)}
                />
            )}
        </>
    );
}
