import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Flex, Select, TextField, Button, IconButton, Separator, Text } from "@radix-ui/themes";
import {
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
    toggleGridSnap,
    togglePlayheadZoom,
    toggleAutoScroll,
    toggleIgnoreGrouping,
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

export function ActionBar() {
    const dispatch = useAppDispatch();
    const s = useAppSelector((state: RootState) => state.session);
    const fileBrowserVisible = useAppSelector((state: RootState) => state.fileBrowser.visible);
    const notebookVisible = useAppSelector((state: RootState) => state.notebook.visible);
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const [pitchSnapOpen, setPitchSnapOpen] = useState(false);
    const [splitTransitionOpen, setSplitTransitionOpen] = useState(false);
    const [customScaleOpen, setCustomScaleOpen] = useState(false);
    const [gridSnapMenuPos, setGridSnapMenuPos] = useState<{ x: number; y: number } | null>(null);

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

                {/* Grid Snap */}
                <IconButton
                    size="1"
                    variant={s.gridSnapEnabled ? "solid" : "ghost"}
                    color="gray"
                    data-tooltip={tAny("grid_snap")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(toggleGridSnap());
                        void dispatch(persistUiSettings());
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        setGridSnapMenuPos({ x: e.clientX, y: e.clientY });
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
                            d="M2 2V13M5.5 2V13M9 2V13M12.5 2V13"
                            stroke="currentColor"
                            strokeWidth="0.8"
                            opacity="0.5"
                        />
                        <path d="M7.5 4L7.5 11" stroke="currentColor" strokeWidth="1.5" />
                        <path d="M5.5 6L7.5 4L9.5 6" stroke="currentColor" strokeWidth="1" />
                        <path d="M5.5 9L7.5 11L9.5 9" stroke="currentColor" strokeWidth="1" />
                    </svg>
                </IconButton>

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

            {/* Grid Snap Context Menu */}
            {gridSnapMenuPos && (
                <GridSnapContextMenu
                    x={gridSnapMenuPos.x}
                    y={gridSnapMenuPos.y}
                    currentGrid={s.grid}
                    onSelect={(grid) => {
                        void dispatch(
                            setProjectTimelineSettingsRemote({
                                beatsPerBar: s.beats,
                                timeSignatureDenominator: displayDenominator,
                                gridSize: grid,
                            }),
                        );
                        setGridSnapMenuPos(null);
                    }}
                    onClose={() => setGridSnapMenuPos(null)}
                    t={tAny}
                />
            )}
        </Flex>
    );
}

/** Grid snap note type definitions for the context menu */
const GRID_SNAP_ITEMS: Array<{ value: string; labelKey: string } | "separator"> = [
    { value: "1/1", labelKey: "grid_snap_whole" },
    { value: "1/2", labelKey: "grid_snap_half" },
    { value: "1/4", labelKey: "grid_snap_quarter" },
    { value: "1/8", labelKey: "grid_snap_8th" },
    { value: "1/16", labelKey: "grid_snap_16th" },
    { value: "1/32", labelKey: "grid_snap_32nd" },
    { value: "1/64", labelKey: "grid_snap_64th" },
    "separator",
    { value: "1/2d", labelKey: "grid_snap_dotted_half" },
    { value: "1/4d", labelKey: "grid_snap_dotted_quarter" },
    { value: "1/8d", labelKey: "grid_snap_dotted_8th" },
    { value: "1/16d", labelKey: "grid_snap_dotted_16th" },
    { value: "1/32d", labelKey: "grid_snap_dotted_32nd" },
    { value: "1/64d", labelKey: "grid_snap_dotted_64th" },
    "separator",
    { value: "1/2t", labelKey: "grid_snap_triplet_half" },
    { value: "1/4t", labelKey: "grid_snap_triplet_quarter" },
    { value: "1/8t", labelKey: "grid_snap_triplet_8th" },
    { value: "1/16t", labelKey: "grid_snap_triplet_16th" },
    { value: "1/32t", labelKey: "grid_snap_triplet_32nd" },
    { value: "1/64t", labelKey: "grid_snap_triplet_64th" },
];

function GridSnapContextMenu({
    x,
    y,
    currentGrid,
    onSelect,
    onClose,
    t,
}: {
    x: number;
    y: number;
    currentGrid: string;
    onSelect: (grid: string) => void;
    onClose: () => void;
    t: (key: string) => string;
}) {
    const menuRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClick = (e: globalThis.MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
                onClose();
            }
        };
        const handleKey = (e: globalThis.KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("mousedown", handleClick, true);
        window.addEventListener("keydown", handleKey, true);
        return () => {
            window.removeEventListener("mousedown", handleClick, true);
            window.removeEventListener("keydown", handleKey, true);
        };
    }, [onClose]);

    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw) el.style.left = `${vw - rect.width}px`;
        if (rect.bottom > vh) el.style.top = `${vh - rect.height}px`;
    }, [x, y]);

    const style: React.CSSProperties = {
        position: "fixed",
        left: x,
        top: y,
        zIndex: 10000,
        minWidth: 180,
        background: "var(--qt-panel)",
        border: "1px solid var(--qt-border)",
        borderRadius: 10,
        padding: "4px 0",
        boxShadow: "0 20px 44px rgba(0,0,0,0.28)",
        display: "block",
        height: "auto",
        overflow: "visible",
    };

    return createPortal(
        <div ref={menuRef} style={style}>
            {GRID_SNAP_ITEMS.map((item, i) => {
                if (item === "separator") {
                    return (
                        <div
                            key={`sep-${i}`}
                            style={{ height: 1, background: "var(--qt-divider)", margin: "4px 0" }}
                        />
                    );
                }
                const isActive = item.value === currentGrid;
                return (
                    <div
                        key={item.value}
                        onClick={() => onSelect(item.value)}
                        style={{
                            padding: "5px 12px",
                            cursor: "pointer",
                            fontSize: 13,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            background: isActive
                                ? "color-mix(in oklab, var(--qt-highlight) 22%, transparent)"
                                : "transparent",
                            color: isActive ? "var(--qt-text)" : "inherit",
                        }}
                        onMouseEnter={(e) => {
                            if (!isActive)
                                (e.currentTarget as HTMLDivElement).style.background =
                                    "var(--qt-hover)";
                        }}
                        onMouseLeave={(e) => {
                            (e.currentTarget as HTMLDivElement).style.background = isActive
                                ? "color-mix(in oklab, var(--qt-highlight) 22%, transparent)"
                                : "transparent";
                        }}
                    >
                        <span>{t(item.labelKey)}</span>
                        {isActive && <span style={{ marginLeft: 8 }}>✓</span>}
                    </div>
                );
            })}
        </div>,
        document.body,
    );
}
