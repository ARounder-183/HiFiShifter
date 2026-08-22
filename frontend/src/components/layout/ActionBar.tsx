import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Flex, Select, TextField, Button, IconButton, Separator, Text, Box } from "@radix-ui/themes";
import {
    CheckIcon,
    DoubleArrowRightIcon,
    PauseIcon,
    Pencil1Icon,
    PlayIcon,
    StopIcon,
} from "@radix-ui/react-icons";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import { PitchSnapSettingsDialog } from "./PitchSnapSettingsDialog";
import { SnapGridSettingsDialog } from "./SnapGridSettingsDialog";
import { SplitTransitionSettingsDialog } from "./SplitTransitionSettingsDialog";
import { CustomScaleDialog } from "./CustomScaleDialog";

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
} from "../../features/session/sessionSlice";
import { setTempoMapRemote } from "../../features/session/thunks/tempoMapThunks";
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
import { isModifierActive, selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import { toggleVisible } from "../../features/fileBrowser/fileBrowserSlice";
import { toggleNotebookVisible } from "../../features/notebook/notebookSlice";
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

export function ActionBar() {
    const dispatch = useAppDispatch();
    const s = useAppSelector((state: RootState) => state.session);
    const fileBrowserVisible = useAppSelector((state: RootState) => state.fileBrowser.visible);
    const notebookVisible = useAppSelector((state: RootState) => state.notebook.visible);
    const recording = useAppSelector((state: RootState) => state.recording);
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const [pitchSnapOpen, setPitchSnapOpen] = useState(false);
    const [snapSettingsOpen, setSnapSettingsOpen] = useState(false);
    const [splitTransitionOpen, setSplitTransitionOpen] = useState(false);
    const [customScaleOpen, setCustomScaleOpen] = useState(false);
    const [recordingSettingsOpen, setRecordingSettingsOpen] = useState(false);
    const [recordingMenuPos, setRecordingMenuPos] = useState<{ x: number; y: number } | null>(null);
    const recordingMenuRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (!recordingMenuPos) return;
        const onPointerDown = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (recordingMenuRef.current?.contains(target)) return;
            setRecordingMenuPos(null);
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") setRecordingMenuPos(null);
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown, true);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown, true);
        };
    }, [recordingMenuPos]);

    const recordingSourceLabel = (() => {
        switch (recording.settings.captureMode) {
            case "loopback":
                return tAny("recording_mode_loopback");
            case "application":
                return tAny("recording_mode_application");
            default:
                return tAny("recording_mode_device");
        }
    })();

    const recordingDeviceLabel = (() => {
        const { captureMode } = recording.settings;
        if (captureMode === "device") {
            const device = recording.devices.find(
                (item) => !item.isLoopback && item.id === recording.settings.sourceDevice,
            );
            return device?.name ?? tAny("recording_device_default");
        }
        if (captureMode === "loopback") {
            if (recording.settings.loopbackDevice === "default") {
                return tAny("recording_loopback_default");
            }
            const device = recording.devices.find(
                (item) => item.isLoopback && item.id === recording.settings.loopbackDevice,
            );
            return device?.name ?? tAny("recording_loopback_default");
        }
        const app = recording.apps.find((item) => item.id === recording.settings.captureAppId);
        return app?.name || recording.settings.captureAppName || tAny("recording_application");
    })();

    const recordingTooltip = [
        recording.active
            ? tAny("recording_tooltip_stop")
            : recording.countdownRemaining > 0
              ? tAny("recording_tooltip_cancel_countdown")
              : tAny("recording_tooltip_start"),
        `${tAny("recording_source_mode")}: ${recordingSourceLabel}`,
        `${tAny("recording_device")}: ${recordingDeviceLabel}`,
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
                : s.project?.baseScale ?? "C",
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

    function commitBpm(nextText?: string) {
        const raw = (nextText ?? bpmText).trim();
        const next = Number(raw);
        setBpmDirty(false);
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
        const text = tAny(baseKey);
        if (text && text !== baseKey) return text;
        return tAny(
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
            className="h-8 bg-qt-window border-b border-qt-border px-1 text-qt-text flex-nowrap overflow-x-auto overflow-y-hidden min-w-0 custom-scrollbar"
        >
            {/* BPM & Time */}
            <Flex align="center" gap="2" className="shrink-0">
                <Text size="1" className="text-qt-text-muted">
                    {t("bpm")}:
                </Text>
                <TextField.Root
                    size="1"
                    value={bpmText}
                    title={
                        s.tempoMap && s.tempoMap.points.length > 0
                            ? tAny("tempo_map_actionbar_tip")
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
                    onWheel={(e: React.WheelEvent<HTMLInputElement>) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const direction = e.deltaY < 0 ? 1 : -1;
                        const step = isModifierActive(paramFineAdjustKb, e) ? 0.1 : 1;
                        const current = Number(bpmText);
                        const base = Number.isFinite(current) ? current : Number(displayBpm);
                        const nextRaw = base + direction * step;
                        const next = Math.round(nextRaw * 1000) / 1000;
                        // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
                        const clamped = clampBpm(next);
                        if (s.tempoMap && s.tempoMap.points.length > 0) {
                            updateTempoPointAtPlayhead({ bpm: clamped });
                        } else {
                            dispatch(setBpm(clamped));
                            void dispatch(updateTransportBpm(clamped));
                        }
                        setBpmText(formatBpmValue(clamped));
                        setBpmDirty(false);
                    }}
                    style={{
                        width: 60,
                        textAlign: "center",
                        backgroundColor: "var(--qt-base)",
                    }}
                />
                <Text size="1" className="text-qt-text-muted">
                    {t("time_signature")}:
                </Text>
                <Flex align="center" gap="1">
                    <TextField.Root
                        size="1"
                        type="number"
                        value={String(displayBeats)}
                        title={
                            s.tempoMap && s.tempoMap.points.length > 0
                                ? tAny("tempo_map_actionbar_tip")
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
                    <Text size="1" className="text-qt-text-muted">
                        /
                    </Text>
                    <Select.Root
                        size="1"
                        value={String(displayDenominator)}
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

                <Text size="1" className="text-qt-text-muted">
                    {t("grid")}:
                </Text>
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
                            <Select.Label>{tAny("grid_note_normal")}</Select.Label>
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
                            <Select.Label>{tAny("grid_note_dotted")}</Select.Label>
                            <Select.Item value="1/2d">1/2.</Select.Item>
                            <Select.Item value="1/4d">1/4.</Select.Item>
                            <Select.Item value="1/8d">1/8.</Select.Item>
                            <Select.Item value="1/16d">1/16.</Select.Item>
                            <Select.Item value="1/32d">1/32.</Select.Item>
                            <Select.Item value="1/64d">1/64.</Select.Item>
                        </Select.Group>
                        <Select.Separator />
                        <Select.Group>
                            <Select.Label>{tAny("grid_note_triplet")}</Select.Label>
                            <Select.Item value="1/2t">1/2t</Select.Item>
                            <Select.Item value="1/4t">1/4t</Select.Item>
                            <Select.Item value="1/8t">1/8t</Select.Item>
                            <Select.Item value="1/16t">1/16t</Select.Item>
                            <Select.Item value="1/32t">1/32t</Select.Item>
                            <Select.Item value="1/64t">1/64t</Select.Item>
                        </Select.Group>
                    </Select.Content>
                </Select.Root>
                <Text size="1" className="text-qt-text-muted">
                    {t("base_scale")}:
                </Text>
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
                                applyBaseScale(s.project.customScale.notes, s.project.customScale.name);
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
                                        {tempoCustomScaleName ?? tAny("custom_scale_short")}
                                    </Select.Item>
                                </Select.Group>
                            </>
                        ) : null}
                        {s.project?.customScale ? (
                            <>
                                <Select.Separator />
                                <Select.Group>
                                    <Select.Item value="__custom__">
                                        {`${tAny("custom_scale_label")}: ${s.project.customScale.name}`}
                                    </Select.Item>
                                </Select.Group>
                            </>
                        ) : null}
                        <Select.Separator />
                        <Select.Group>
                            <Select.Item value="__custom_dialog__">
                                {tAny("custom_scale_action")}
                            </Select.Item>
                        </Select.Group>
                    </Select.Content>
                </Select.Root>
            </Flex>

            <Separator orientation="vertical" size="2" />

            {/* Transport */}
            <Flex gap="1" className="shrink-0">
                <Button
                    variant="soft"
                    color="gray"
                    size="1"
                    onClick={() => {
                        dispatch(stopAudioPlayback({ restoreAnchor: true }));
                    }}
                    data-tooltip={t("action_stop")}
                >
                    <StopIcon />
                </Button>
                <IconButton
                    variant="solid"
                    size="1"
                    onClick={() => {
                        if (s.runtime.isPlaying) {
                            dispatch(stopAudioPlayback());
                            return;
                        }
                        dispatch(playOriginal());
                    }}
                    data-tooltip={s.runtime.isPlaying ? tAny("action_pause") : t("action_play_out")}
                >
                    {s.runtime.isPlaying ? <PauseIcon /> : <PlayIcon />}
                </IconButton>
                <Box style={{ position: "relative" }} data-hs-context-menu>
                    <IconButton
                        size="1"
                        variant={recording.active ? "solid" : "ghost"}
                        color={recording.active ? "red" : "gray"}
                        data-tooltip={recordingTooltip}
                        disabled={recording.busy && recording.countdownRemaining === 0}
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
                            setRecordingMenuPos({ x: event.clientX, y: event.clientY });
                            void dispatch(loadRecordingSettings());
                            void dispatch(loadRecordingDevices());
                            void dispatch(loadRecordingApps());
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
                        <div
                            ref={recordingMenuRef}
                            data-hs-context-menu
                            className="fixed z-50 min-w-[220px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
                            style={{ left: recordingMenuPos.x, top: recordingMenuPos.y }}
                        >
                            <div className="px-3 py-1 text-[11px] uppercase tracking-wide text-qt-text-muted">
                                {tAny("recording_source_mode")}
                            </div>
                            <button
                                type="button"
                                className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                onClick={() => void applyRecordingSettings({ captureMode: "device" })}
                                onPointerDown={(e) => e.stopPropagation()}
                            >
                                <span>{tAny("recording_mode_device")}</span>
                                {recording.settings.captureMode === "device" ? <CheckIcon /> : null}
                            </button>
                            <button
                                type="button"
                                className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                onClick={() =>
                                    void applyRecordingSettings({ captureMode: "loopback" })
                                }
                                onPointerDown={(e) => e.stopPropagation()}
                            >
                                <span>{tAny("recording_mode_loopback")}</span>
                                {recording.settings.captureMode === "loopback" ? <CheckIcon /> : null}
                            </button>
                            <button
                                type="button"
                                className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                onClick={() =>
                                    void applyRecordingSettings({ captureMode: "application" })
                                }
                                onPointerDown={(e) => e.stopPropagation()}
                            >
                                <span>{tAny("recording_mode_application")}</span>
                                {recording.settings.captureMode === "application" ? (
                                    <CheckIcon />
                                ) : null}
                            </button>
                            <div className="my-1 border-t border-qt-border" />
                            <div className="px-3 py-1 text-[11px] uppercase tracking-wide text-qt-text-muted">
                                {tAny(
                                    recording.settings.captureMode === "application"
                                        ? "recording_application"
                                        : "recording_device",
                                )}
                            </div>
                            {recording.settings.captureMode === "device" ? (
                                <>
                                    <button
                                        type="button"
                                        className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                        onClick={() =>
                                            void applyRecordingSettings({ sourceDevice: "default" })
                                        }
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <span>{tAny("recording_device_default")}</span>
                                        {recording.settings.sourceDevice === "default" ? (
                                            <CheckIcon />
                                        ) : null}
                                    </button>
                                    {recording.devices
                                        .filter((device) => !device.isLoopback)
                                        .map((device) => (
                                            <button
                                                key={device.id}
                                                type="button"
                                                className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                                onClick={() =>
                                                    void applyRecordingSettings({
                                                        sourceDevice: device.id,
                                                    })
                                                }
                                                onPointerDown={(e) => e.stopPropagation()}
                                            >
                                                <span className="truncate">{device.name}</span>
                                                {recording.settings.sourceDevice === device.id ? (
                                                    <CheckIcon />
                                                ) : null}
                                            </button>
                                        ))}
                                </>
                            ) : recording.settings.captureMode === "loopback" ? (
                                <>
                                    <button
                                        type="button"
                                        className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                        onClick={() =>
                                            void applyRecordingSettings({ loopbackDevice: "default" })
                                        }
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <span>{tAny("recording_loopback_default")}</span>
                                        {recording.settings.loopbackDevice === "default" ? (
                                            <CheckIcon />
                                        ) : null}
                                    </button>
                                    {recording.devices
                                        .filter((device) => device.isLoopback)
                                        .map((device) => (
                                            <button
                                                key={device.id}
                                                type="button"
                                                className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                                onClick={() =>
                                                    void applyRecordingSettings({
                                                        loopbackDevice: device.id,
                                                    })
                                                }
                                                onPointerDown={(e) => e.stopPropagation()}
                                            >
                                                <span className="truncate">{device.name}</span>
                                                {recording.settings.loopbackDevice === device.id ? (
                                                    <CheckIcon />
                                                ) : null}
                                            </button>
                                        ))}
                                </>
                            ) : (
                                <>
                                    {recording.settings.captureAppId &&
                                    !recording.apps.some(
                                        (app) => app.id === recording.settings.captureAppId,
                                    ) ? (
                                        <button
                                            type="button"
                                            className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                            onClick={() =>
                                                void applyRecordingSettings({
                                                    captureAppId: recording.settings.captureAppId,
                                                    captureAppName:
                                                        recording.settings.captureAppName,
                                                    captureAppProcess:
                                                        recording.settings.captureAppProcess,
                                                })
                                            }
                                            onPointerDown={(e) => e.stopPropagation()}
                                        >
                                            <span className="truncate">
                                                {recording.settings.captureAppName ||
                                                    recording.settings.captureAppId}
                                            </span>
                                            <CheckIcon />
                                        </button>
                                    ) : null}
                                    {recording.apps.map((app) => (
                                        <button
                                            key={app.id}
                                            type="button"
                                            className="w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                            onClick={() =>
                                                void applyRecordingSettings({
                                                    captureAppId: app.id,
                                                    captureAppName: app.name,
                                                    captureAppProcess: app.processName,
                                                })
                                            }
                                            onPointerDown={(e) => e.stopPropagation()}
                                        >
                                            <span className="truncate">{app.name}</span>
                                            {recording.settings.captureAppId === app.id ? (
                                                <CheckIcon />
                                            ) : null}
                                        </button>
                                    ))}
                                </>
                            )}
                            <div className="my-1 border-t border-qt-border" />
                            <button
                                type="button"
                                className="w-full flex items-center gap-3 px-3 py-1.5 text-left text-[12px] transition-colors hover:bg-qt-button-hover"
                                onClick={() => {
                                    setRecordingMenuPos(null);
                                    setRecordingSettingsOpen(true);
                                }}
                                onPointerDown={(e) => e.stopPropagation()}
                            >
                                <span>{tAny("recording_context_settings")}</span>
                            </button>
                        </div>
                    )}
                </Box>
                {recording.active || recording.countdownRemaining > 0 ? (
                    <Flex align="center" gap="1" className="shrink-0">
                        <Text
                            size="1"
                            color={recording.active ? "red" : "gray"}
                            className="tabular-nums"
                        >
                            {recording.countdownRemaining > 0
                                ? `-${recording.countdownRemaining}`
                                : formatRecordingTime(recording.elapsedSec)}
                        </Text>
                        <div
                            style={{
                                width: 48,
                                height: 6,
                                borderRadius: 3,
                                background: "var(--qt-border)",
                                overflow: "hidden",
                                flexShrink: 0,
                            }}
                        >
                            <div
                                style={{
                                    width: `${Math.min(100, Math.round((recording.level || 0) * 100))}%`,
                                    height: "100%",
                                    background: recording.level > 0.98 ? "red" : "#e5484d",
                                    transition: "width 80ms linear",
                                }}
                            />
                        </div>
                    </Flex>
                ) : null}
                {recording.error ? (
                    <Text
                        size="1"
                        color="red"
                        title={recording.error}
                        className="truncate"
                        style={{ maxWidth: 220 }}
                    >
                        {recordingErrorMessage(recording.error)}
                    </Text>
                ) : null}
            </Flex>

            <Separator orientation="vertical" size="2" />

            {/* File Browser Toggle */}
            <Flex gap="1" className="shrink-0">
                <IconButton
                    size="1"
                    variant={fileBrowserVisible ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("fb_title")}
                    onClick={() => dispatch(toggleVisible())}
                >
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
                </IconButton>
                <IconButton
                    size="1"
                    variant={notebookVisible ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={t("notebook")}
                    onClick={() => dispatch(toggleNotebookVisible())}
                >
                    <Pencil1Icon />
                </IconButton>
            </Flex>

            <Separator orientation="vertical" size="2" />

            {/* Toolbar Toggles */}
            <Flex align="center" gap="1" className="shrink-0">
                {/* Auto Crossfade */}
                <IconButton
                    size="1"
                    variant={s.autoCrossfadeEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("auto_crossfade")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleAutoCrossfade());
                        void dispatch(persistUiSettings());
                    }}
                >
                    {/* X icon for crossfade */}
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
                </IconButton>

                {/* Split Transition */}
                <IconButton
                    size="1"
                    variant={s.splitTransitionEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("split_transition_tooltip")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleSplitTransition());
                        void dispatch(persistUiSettings());
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        setSplitTransitionOpen(true);
                    }}
                >
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
                </IconButton>

                {/* Snap */}
                <IconButton
                    size="1"
                    variant={s.snapEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("snap")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleSnap());
                        void dispatch(persistUiSettings());
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        setSnapSettingsOpen(true);
                    }}
                >
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
                </IconButton>

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

                <Separator orientation="vertical" size="2" />

                {/* Playhead Zoom */}
                <IconButton
                    size="1"
                    variant={s.playheadZoomEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("playhead_zoom")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(togglePlayheadZoom());
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
                        <path d="M7.5 2V13" stroke="currentColor" strokeWidth="1.2" />
                        <path d="M6 3.5L7.5 2L9 3.5" stroke="currentColor" strokeWidth="1" />
                        <path d="M5.5 5.5L4 7.5L5.5 9.5" stroke="currentColor" strokeWidth="1.2" />
                        <path d="M9.5 5.5L11 7.5L9.5 9.5" stroke="currentColor" strokeWidth="1.2" />
                        <path d="M3 12H12" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
                    </svg>
                </IconButton>

                {/* Auto Scroll (horizontal arrows) */}
                <IconButton
                    size="1"
                    variant={s.autoScrollEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("auto_scroll")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleAutoScroll());
                        void dispatch(persistUiSettings());
                    }}
                >
                    <DoubleArrowRightIcon width="15" height="15" />
                </IconButton>

                <Separator orientation="vertical" size="2" />

                <IconButton
                    size="1"
                    variant={s.paramEditorSeekPlayheadEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("param_editor_seek_playhead")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleParamEditorSeekPlayhead());
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
                        <path d="M2 2.5H13" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
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
                </IconButton>

                {/* Allow timeline clicks to switch the parameter editor track */}
                <IconButton
                    size="1"
                    variant={s.paramEditorTimelineClickSelectTrackEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("param_editor_timeline_click_select_track")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleParamEditorTimelineClickSelectTrack());
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
                </IconButton>

                <Separator orientation="vertical" size="2" />

                {/* Ignore Grouping (broken chain) */}
                <IconButton
                    size="1"
                    variant={s.ignoreGrouping ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("ignore_grouping")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleIgnoreGrouping());
                        void dispatch(persistUiSettings());
                    }}
                >
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
                </IconButton>
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
                <SnapGridSettingsDialog open={snapSettingsOpen} onOpenChange={setSnapSettingsOpen} />
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
                    <rect
                        x="8.6"
                        y={y}
                        width="2.9"
                        height="2.4"
                        rx="0.6"
                        fill="currentColor"
                    />
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const menuRef = useRef<HTMLDivElement>(null);
    const onCloseRef = useRef(onClose);

    useLayoutEffect(() => {
        onCloseRef.current = onClose;
    }, [onClose]);

    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw) {
            el.style.left = `${Math.max(0, vw - rect.width)}px`;
        }
        if (rect.bottom > vh) {
            el.style.top = `${Math.max(0, vh - rect.height)}px`;
        }

        const onPointerDown = (e: PointerEvent) => {
            if (el && !el.contains(e.target as Node)) {
                onCloseRef.current();
            }
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                e.preventDefault();
                onCloseRef.current();
            }
        };
        window.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("keydown", onKeyDown);
        return () => {
            window.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("keydown", onKeyDown);
        };
    }, [x, y]);

    const options: Array<{ value: "off" | "track" | "all"; label: string }> = [
        { value: "off", label: tAny("ripple_mode_off") as string },
        { value: "track", label: tAny("ripple_mode_track") as string },
        { value: "all", label: tAny("ripple_mode_all") as string },
    ];

    return (
        <div
            ref={menuRef}
            data-hs-context-menu="1"
            className="fixed z-50 min-w-[150px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
        >
            {options.map((opt) => (
                <button
                    key={opt.value}
                    className="px-3 py-1.5 text-left w-full text-[12px] transition-colors flex items-center justify-between gap-3 hover:bg-qt-highlight hover:text-white"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => {
                        e.stopPropagation();
                        onChange(opt.value);
                        onClose();
                    }}
                >
                    <span>{opt.label}</span>
                    {mode === opt.value && <span className="text-qt-accent">✓</span>}
                </button>
            ))}
        </div>
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);

    return (
        <>
            <IconButton
                size="1"
                variant={mode !== "off" ? "solid" : "ghost"}
                color="gray"
                data-tooltip={(tAny(`ripple_tooltip_${mode}`) as string) ?? tAny("ripple")}
                tabIndex={-1}
                onClick={onCycle}
                onContextMenu={(e) => {
                    e.preventDefault();
                    setMenu({ x: e.clientX, y: e.clientY });
                }}
            >
                <RippleIcon multiTrack={mode === "all"} />
            </IconButton>
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
