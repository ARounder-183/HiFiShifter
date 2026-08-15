import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Checkbox, Dialog, Flex, Select, Text, TextField } from "@radix-ui/themes";
import type { GridSize, TimelineSnapSettings } from "../../../features/session/sessionTypes";
import type { ScaleLike } from "../../../utils/musicalScales";
import { SCALE_KEYS, SCALE_LABELS } from "../../../utils/musicalScales";
import type { CustomScalePreset } from "../../../utils/customScales";
import {
    clampBpm,
    clampDenominator,
    clampNumerator,
    createTempoPointAt,
    effectiveScaleAtSec,
    effectiveTimeSignatureAt,
    formatTempoBpm,
    formatTimeSignature,
    normalizeScaleData,
    parseTempoPointText,
    previousScaleAtSec,
    removeTempoPoint,
    TEMPO_DENOMINATORS,
    tempoMapSegments,
    tempoFlagLabelWidthPx,
    tempoPointFlagLabel,
    tempoPointScaleShortLabel,
    updateTempoPoint,
} from "../../../utils/tempoMap";
import type { TempoMap, TempoPoint, TempoMapScaleData } from "../../../utils/tempoMap";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeFormat";
import { formatCursorTime } from "./timeFormat";
import { snapTimelinePosition } from "../../../utils/timelineSnapping";
import { useAppSelector } from "../../../app/hooks";
import { isModifierActive, selectKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { applySelectWheelChange } from "../../../utils/selectWheel";

/** Tempo Map 行高度（不含分隔线）。 */
export const TEMPO_ROW_HEIGHT_PX = 17;

/**
 * “输入状态下拖动标签”的意图判定容差（px）：
 * 标签输入框内按住左键拖动时，只要指针保持在输入框边界（含该容差）
 * 以内，就保持 TextBox 的“选中内容”行为 —— 无论拖了多远（长标签
 * 的全选等操作不会被误判）；一旦指针明显越出输入框边界，才判定用户
 * 意图为“拖动标签”。比单纯的距离阈值更智能：输入框内部的任意距离
 * 拖动都属于选中文本，越出边界则几乎不可能还是“选中内容”的意图。
 */
const INLINE_DRAG_TO_MOVE_BOUNDARY_MARGIN_PX = 10;

export interface TempoPointEditRequest {
    /** 编辑已有变化点（为 null 表示新建）。 */
    pointId: string | null;
    /** 新建点的位置（秒）；编辑已有变化点时忽略。 */
    positionSec: number | null;
    /** 对话框初始焦点。 */
    focus: "tempo" | "timeSignature" | "scale" | null;
    /**
     * 编辑已有变化点的方式：`dialog` = 弹出“速度映射变化点”编辑窗口
     * （默认）；`inline` = 直接进入标签的输入编辑状态（悬浮标签双击）。
     */
    mode?: "dialog" | "inline";
}

interface TempoMapRulerRowProps {
    tempoMap: TempoMap | null;
    visible: boolean;
    pxPerSec: number;
    scrollLeft: number;
    viewportWidth: number;
    projectSec: number;
    grid: GridSize;
    snapEnabled: boolean;
    /** 完整吸附/网格设置（与工具栏一致）。 */
    snapSettings?: TimelineSnapSettings;
    /** 工程基准 BPM / 每小节拍数 / 拍号分母（无 Tempo Map 时新建首点用）。 */
    fallbackBpm: number;
    fallbackBeatsPerBar: number;
    fallbackDenominator?: number;
    /** 工程音阶（跟随工程音阶时显示用）。 */
    projectScale: ScaleLike | null;
    projectScaleName?: string;
    customScalePresets: readonly CustomScalePreset[];
    /** 时间单位与上下文（变化点 ToolTip 的“位置”行按用户主/副时间单位显示）。 */
    primaryUnit: TimeUnit;
    secondaryUnit: TimeUnitChoice;
    timeContext: TimeFormatContext;
    t: (key: string) => string;
    /** 本地即时更新（拖动过程中频繁调用，仅更新 Redux，不同步后端）。 */
    onChange: (next: TempoMap | null) => void;
    /** 离散提交（对话框确认、右键菜单、拖拽结束），同步后端。 */
    onCommit: (next: TempoMap | null) => void;
    /** 由时间标尺右键菜单 / 悬浮标签发出的编辑/新建请求。 */
    editRequest: TempoPointEditRequest | null;
    onEditRequestHandled: () => void;
    /** 编辑对话框打开/关闭通知（用于抑制标尺悬浮时间提示）。 */
    onDialogOpenChange?: (open: boolean) => void;
    /**
     * 内联输入框显示在视口左侧（覆盖悬浮标签位置）的状态通知：
     * 仅当双击悬浮标签（其管辖旗帜在画面外）进入编辑时才为 true；
     * 双击画面内的固定标签进入编辑时为 false —— 悬浮标签保持显示。
     */
    onFloatingInlineEditChange?: (overlaying: boolean) => void;
}

/** ScaleLike → 显示文本（键音阶显示 "C / Am"，自定义音阶显示名称）。 */
function scaleLikeLabel(scale: ScaleLike | null | undefined, name?: string): string | null {
    if (!scale) return null;
    if (Array.isArray(scale)) {
        return name || null;
    }
    return SCALE_LABELS[scale as keyof typeof SCALE_LABELS] ?? scale;
}

// ────────────────────────────────────────────────────────────────────────────
// 变化点编辑对话框
// ────────────────────────────────────────────────────────────────────────────

interface TempoPointDialogProps {
    open: boolean;
    point: TempoPoint | null;
    isFirst: boolean;
    /** “跟随之前的拍号”选项展示的上一变化点拍号标签。 */
    previousTimeSignatureLabel: string;
    /** “跟随之前的拍号”时字段展示/解除跟随用的上一变化点实际拍号。 */
    previousTimeSignature: { numerator: number; denominator: number };
    /** “跟随之前的音阶”选项展示的上一变化点音阶标签。 */
    previousScaleLabel: string;
    customScalePresets: readonly CustomScalePreset[];
    focus: "tempo" | "timeSignature" | "scale" | null;
    t: (key: string) => string;
    onCancel: () => void;
    onConfirm: (patch: {
        bpm: number;
        timeSignature: { numerator: number; denominator: number } | null;
        scale: TempoMapScaleData | null;
    }) => void;
    onDelete: () => void;
}

function TempoPointDialog({
    open,
    point,
    isFirst,
    previousTimeSignatureLabel,
    previousTimeSignature,
    previousScaleLabel,
    customScalePresets,
    focus,
    t,
    onCancel,
    onConfirm,
    onDelete,
}: TempoPointDialogProps) {
    // 对话框通过 key 重挂载来复位表单状态（打开新点时由父组件更换 key）。
    const [bpmText, setBpmText] = useState(() =>
        point ? formatTempoBpm(point.bpm) : "120",
    );
    // 拍号：跟随之前的拍号时，输入框展示“上一变化点实际生效”的拍号（禁用态）。
    // 初始点即工程基准记录，必须显式携带拍号，不能跟随。
    const [sigFollow, setSigFollow] = useState(() =>
        point ? !isFirst && point.timeSignature == null : false,
    );
    const [numText, setNumText] = useState(() =>
        point
            ? String(point.timeSignature?.numerator ?? previousTimeSignature.numerator)
            : "4",
    );
    const [denominator, setDenominator] = useState(() =>
        point?.timeSignature?.denominator ?? previousTimeSignature.denominator,
    );
    const [scaleValue, setScaleValue] = useState(() => {
        if (!point?.scale) return "inherit";
        if (point.scale.key) return `key:${point.scale.key}`;
        // 尝试按音级匹配自定义预设；不匹配时保留原始数据。
        const preset = customScalePresets.find(
            (p) =>
                Array.isArray(point.scale?.notes) &&
                p.notes.length === point.scale!.notes!.length &&
                p.notes.every((n, i) => n === point.scale!.notes![i]),
        );
        return preset ? `custom:${preset.id}` : "custom:temp";
    });
    const [customNotes, setCustomNotes] = useState<number[] | null>(() =>
        point?.scale?.notes ?? null,
    );
    const [customName, setCustomName] = useState(() => point?.scale?.name ?? "");
    const bpmRef = useRef<HTMLInputElement | null>(null);
    const numRef = useRef<HTMLInputElement | null>(null);

    // 修饰键“参数微调”：滚轮调节 BPM 时步长 0.1，否则 1（与左上角 BPM 控件一致）。
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    useEffect(() => {
        if (!open) return;
        requestAnimationFrame(() => {
            if (focus === "tempo") bpmRef.current?.select();
            else if (focus === "timeSignature") numRef.current?.select();
        });
    }, [open, focus]);

    const commit = useCallback(() => {
        if (!point) return;
        const bpm = clampBpm(Number(bpmText) || 120);
        let timeSignature: { numerator: number; denominator: number } | null = null;
        if (!sigFollow) {
            timeSignature = {
                numerator: clampNumerator(Number(numText) || 4),
                denominator: clampDenominator(denominator),
            };
        }
        let scale: TempoMapScaleData | null = null;
        if (scaleValue.startsWith("key:")) {
            const key = scaleValue.slice(4);
            if ((SCALE_KEYS as readonly string[]).includes(key)) scale = { key };
        } else if (scaleValue === "custom:temp") {
            scale = normalizeScaleData({ name: customName, notes: customNotes ?? undefined });
        } else if (scaleValue.startsWith("custom:")) {
            const presetId = scaleValue.slice(7);
            const preset = customScalePresets.find((p) => String(p.id) === presetId);
            if (preset) scale = { name: preset.name, notes: [...preset.notes] };
        }
        onConfirm({ bpm, timeSignature, scale });
    }, [
        point,
        bpmText,
        sigFollow,
        numText,
        denominator,
        scaleValue,
        customNotes,
        customName,
        customScalePresets,
        onConfirm,
    ]);

    if (!point) return null;

    // 滚轮选项：音阶 = 继承 + 内置键 + 自定义预设；拍号分母 = TEMPO_DENOMINATORS。
    const scaleWheelOptions = [
        "inherit",
        ...SCALE_KEYS.map((k) => `key:${k}`),
        ...customScalePresets.map((p) => `custom:${p.id}`),
    ];

    const applyBpmWheel = (e: React.WheelEvent<HTMLInputElement>) => {
        e.preventDefault();
        e.stopPropagation();
        const direction = e.deltaY < 0 ? 1 : -1;
        const step = isModifierActive(paramFineAdjustKb, e) ? 0.1 : 1;
        const current = Number(bpmText);
        const base = Number.isFinite(current) ? current : Number(point?.bpm ?? 120);
        const nextRaw = base + direction * step;
        const next = Math.round(nextRaw * 1000) / 1000;
        setBpmText(formatTempoBpm(clampBpm(next)));
    };

    // 拍号分子滚轮：跟随状态下先解除跟随（以实际生效值为基础继续调节）。
    const applyNumeratorWheel = (e: React.WheelEvent<HTMLInputElement>) => {
        e.preventDefault();
        e.stopPropagation();
        const direction = e.deltaY < 0 ? 1 : -1;
        const current = Number(numText);
        const base = Number.isFinite(current) && current >= 1 ? Math.round(current) : 4;
        setSigFollow(false);
        setNumText(String(Math.min(32, Math.max(1, base + direction))));
    };

    return (
        <Dialog.Root open={open} onOpenChange={(next) => !next && onCancel()}>
            <Dialog.Content maxWidth="420px">
                <Dialog.Title>
                    {isFirst ? t("tempo_map_dialog_title_initial") : t("tempo_map_dialog_title")}
                </Dialog.Title>
                <Flex direction="column" gap="3" mt="3">
                    <Flex gap="2" align="center">
                        <Text size="1" className="text-qt-text-muted shrink-0" style={{ width: 96 }}>
                            {t("bpm")}
                        </Text>
                        <TextField.Root
                            size="1"
                            ref={bpmRef}
                            value={bpmText}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                setBpmText(e.target.value)
                            }
                            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                                if (e.key === "Enter") commit();
                            }}
                            onWheel={applyBpmWheel}
                            style={{ width: 90 }}
                        />
                        <Text size="1" className="text-qt-text-muted">
                            {t("tempo_map_bpm_range")}
                        </Text>
                    </Flex>
                    <Flex gap="2" align="center">
                        <Text size="1" className="text-qt-text-muted shrink-0" style={{ width: 96 }}>
                            {t("tempo_map_time_signature")}
                        </Text>
                        <TextField.Root
                            size="1"
                            ref={numRef}
                            value={numText}
                            disabled={sigFollow}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                                if (sigFollow) return;
                                setNumText(e.target.value);
                            }}
                            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                                if (e.key === "Enter") commit();
                            }}
                            onWheel={applyNumeratorWheel}
                            style={{ width: 48 }}
                        />
                        <Text size="1">/</Text>
                        <Select.Root
                            size="1"
                            value={String(denominator)}
                            disabled={sigFollow}
                            onValueChange={(v) => {
                                if (sigFollow) return;
                                setDenominator(Number(v) || 4);
                            }}
                        >
                            <Select.Trigger
                                style={{ width: 56 }}
                                onWheel={(event) => {
                                    applySelectWheelChange({
                                        event,
                                        currentValue: String(denominator),
                                        options: TEMPO_DENOMINATORS.map((d) => String(d)),
                                        onChange: (v) => {
                                            setSigFollow(false);
                                            setDenominator(Number(v) || 4);
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
                    {!isFirst ? (
                        <Flex gap="2" align="center" style={{ marginTop: -8 }}>
                            <Checkbox
                                size="1"
                                checked={sigFollow}
                                disabled={isFirst}
                                onCheckedChange={(checked) => setSigFollow(checked === true)}
                            />
                            <Text size="1" className="text-qt-text-muted">
                                {t("tempo_map_ts_inherit")} ({previousTimeSignatureLabel})
                            </Text>
                        </Flex>
                    ) : null}
                    <Flex gap="2" align="center">
                        <Text size="1" className="text-qt-text-muted shrink-0" style={{ width: 96 }}>
                            {t("tempo_map_scale")}
                        </Text>
                        <Select.Root
                            size="1"
                            value={scaleValue}
                            onValueChange={(v) => {
                                if (v.startsWith("custom:")) {
                                    const presetId = v.slice(7);
                                    const preset = customScalePresets.find(
                                        (p) => String(p.id) === presetId,
                                    );
                                    setCustomNotes(preset ? [...preset.notes] : null);
                                    setCustomName(preset?.name ?? "");
                                }
                                setScaleValue(v);
                            }}
                        >
                            <Select.Trigger
                                style={{ minWidth: 190 }}
                                onWheel={(event) => {
                                    applySelectWheelChange({
                                        event,
                                        currentValue: scaleValue,
                                        options: scaleWheelOptions,
                                        onChange: (v) => {
                                            if (v.startsWith("custom:")) {
                                                const presetId = v.slice(7);
                                                const preset = customScalePresets.find(
                                                    (p) => String(p.id) === presetId,
                                                );
                                                setCustomNotes(preset ? [...preset.notes] : null);
                                                setCustomName(preset?.name ?? "");
                                            }
                                            setScaleValue(v);
                                        },
                                    });
                                }}
                            />
                            <Select.Content>
                                <Select.Item value="inherit">
                                    {t("tempo_map_scale_inherit")} ({previousScaleLabel})
                                </Select.Item>
                                <Select.Group>
                                    <Select.Label>{t("scale_builtin_group")}</Select.Label>
                                    {SCALE_KEYS.map((key) => (
                                        <Select.Item key={key} value={`key:${key}`}>
                                            {SCALE_LABELS[key]}
                                        </Select.Item>
                                    ))}
                                </Select.Group>
                                {customScalePresets.length > 0 ? (
                                    <Select.Group>
                                        <Select.Label>{t("scale_custom_group")}</Select.Label>
                                        {customScalePresets.map((preset) => (
                                            <Select.Item key={preset.id} value={`custom:${preset.id}`}>
                                                {preset.name || preset.id}
                                            </Select.Item>
                                        ))}
                                    </Select.Group>
                                ) : null}
                            </Select.Content>
                        </Select.Root>
                    </Flex>
                </Flex>
                <Flex justify="between" align="center" mt="4" gap="2">
                    <Button variant="soft" color="red" size="1" disabled={isFirst} onClick={onDelete}>
                        {t("tempo_map_delete_point")}
                    </Button>
                    <Flex gap="2">
                        <Button variant="soft" color="gray" size="1" onClick={onCancel}>
                            {t("cancel")}
                        </Button>
                        <Button size="1" onClick={commit}>
                            {t("ok")}
                        </Button>
                    </Flex>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Tempo Map 标尺行
// ────────────────────────────────────────────────────────────────────────────

export const TempoMapRulerRow: React.FC<TempoMapRulerRowProps> = ({
    tempoMap,
    visible,
    pxPerSec,
    scrollLeft,
    viewportWidth,
    projectSec,
    grid,
    snapEnabled,
    snapSettings,
    fallbackBpm,
    fallbackBeatsPerBar,
    fallbackDenominator,
    projectScale,
    projectScaleName,
    customScalePresets,
    primaryUnit,
    secondaryUnit,
    timeContext,
    t,
    onChange,
    onCommit,
    editRequest,
    onEditRequestHandled,
    onDialogOpenChange,
    onFloatingInlineEditChange,
}) => {
    const [selectedId, setSelectedId] = useState<string | null>(null);
    /** 旗帜内联编辑（双击蓝色标签 → 原地 TextBox）。 */
    const [editingPointId, setEditingPointId] = useState<string | null>(null);
    const [editingText, setEditingText] = useState("");
    const [dialogState, setDialogState] = useState<{
        seq: number;
        pointId: string | null;
        point: TempoPoint | null;
        isFirst: boolean;
        focus: "tempo" | "timeSignature" | "scale" | null;
        /** 新建点暂存：对话框确认时一次性提交；取消不产生任何数据。 */
        pendingCreate: boolean;
        pendingPositionSec: number | null;
        pendingBaseMap: TempoMap | null;
    } | null>(null);
    const dragRef = useRef<{
        pointId: string;
        startClientX: number;
        startSec: number;
    } | null>(null);
    const [draggingId, setDraggingId] = useState<string | null>(null);
    const dialogStateRef = useRef(dialogState);
    useEffect(() => {
        dialogStateRef.current = dialogState;
    }, [dialogState]);
    const dialogSeqRef = useRef(0);

    // Tempo Map 标签拖动使用与 Clip 相同的吸附候选/阈值逻辑，因此直接读取
    // 当前时间轴中的 Clip / 轨道 / 选区 / 播放头作为候选目标。
    const timelineClips = useAppSelector((state) => state.session.clips);
    const timelineTracks = useAppSelector((state) => state.session.tracks);
    const selectedClipIds = useAppSelector((state) =>
        state.session.multiSelectedClipIds.length > 0
            ? state.session.multiSelectedClipIds
            : state.session.selectedClipId
              ? [state.session.selectedClipId]
              : [],
    );
    const playheadSec = useAppSelector((state) => state.session.playheadSec);
    const noSnapKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipNoSnap"),
    );

    // 编辑对话框打开/关闭时通知父级（用于抑制标尺悬浮时间提示）。
    useEffect(() => {
        onDialogOpenChange?.(dialogState != null);
    }, [dialogState, onDialogOpenChange]);
    const dragDraftRef = useRef<TempoMap | null>(null);
    /** 已消费的编辑请求：防止同一个请求在 effect 重跑时被重复处理（重复建点）。 */
    const consumedEditRequestRef = useRef<TempoPointEditRequest | null>(null);

    const openDialogState = useCallback(
        (
            pointId: string | null,
            point: TempoPoint | null,
            isFirst: boolean,
            focus: "tempo" | "timeSignature" | "scale" | null,
            pending?: { positionSec: number; baseMap: TempoMap | null },
        ) => {
            dialogSeqRef.current += 1;
            setDialogState({
                seq: dialogSeqRef.current,
                pointId,
                point,
                isFirst,
                focus,
                pendingCreate: pending != null,
                pendingPositionSec: pending?.positionSec ?? null,
                pendingBaseMap: pending?.baseMap ?? null,
            });
        },
        [],
    );

    const visibleRow = visible && tempoMap != null && tempoMap.points.length > 0;
    const fallback = useMemo(
        () => ({
            bpm: fallbackBpm,
            beatsPerBar: fallbackBeatsPerBar,
            denominator: fallbackDenominator ?? 4,
        }),
        [fallbackBpm, fallbackBeatsPerBar, fallbackDenominator],
    );

    // 仅当内联输入框实际显示在视口左侧（覆盖悬浮标签位置，即双击悬浮标签进入
    // 编辑）时通知父级隐藏悬浮标签；双击画面内的固定标签进入编辑时不隐藏。
    const editingOverlaysFloatingLabel = useMemo(() => {
        if (!editingPointId || !tempoMap || !visibleRow) return false;
        const point = tempoMap.points.find((p) => p.id === editingPointId);
        if (!point) return false;
        const left = point.positionSec * pxPerSec;
        return left + tempoFlagLabelWidthPx(tempoPointFlagLabel(point)) < scrollLeft;
    }, [editingPointId, tempoMap, visibleRow, pxPerSec, scrollLeft]);

    useEffect(() => {
        onFloatingInlineEditChange?.(editingOverlaysFloatingLabel);
    }, [editingOverlaysFloatingLabel, onFloatingInlineEditChange]);

    /** 新建变化点用的工程基准值：有 Tempo Map 时取 0 位置点，否则取工程记录。 */
    const baseFallbackOf = useCallback(
        (base: TempoMap | null) => {
            if (base && base.points.length > 0) {
                const sig =
                    base.points[0].timeSignature ?? { numerator: 4, denominator: 4 };
                return {
                    bpm: base.points[0].bpm,
                    beatsPerBar: sig.numerator,
                    denominator: sig.denominator,
                };
            }
            return fallback;
        },
        [fallback],
    );

    /** 使用与 Clip 完全相同的吸附引擎对 Tempo Map 位置进行吸附。 */
    const snapTempoPosition = useCallback(
        (sec: number, base: TempoMap | null, baseBpm: number, originSec?: number) => {
            if (!snapEnabled || !snapSettings?.enabled) return sec;
            const beatsPerBar =
                base && base.points.length > 0
                    ? (base.points[0].timeSignature?.numerator ?? fallbackBeatsPerBar)
                    : fallbackBeatsPerBar;
            return snapTimelinePosition(
                {
                    settings: snapSettings,
                    grid,
                    bpm: baseBpm,
                    beatsPerBar,
                    tempoMap: base,
                    pxPerSec,
                    clips: timelineClips,
                    tracks: timelineTracks,
                    selectedClipIds,
                    playheadSec,
                    object: "mediaItem",
                    originSec,
                    anchorTrackId: null,
                },
                sec,
            ).sec;
        },
        [
            snapEnabled,
            snapSettings,
            grid,
            fallbackBeatsPerBar,
            pxPerSec,
            timelineClips,
            timelineTracks,
            selectedClipIds,
            playheadSec,
        ],
    );

    /** 防重复提交：Enter 提交后输入框卸载可能再次触发 blur；Esc 取消后同理。 */
    const inlineEditLockRef = useRef(false);

    const startInlineEdit = useCallback((point: TempoPoint) => {
        inlineEditLockRef.current = false;
        setSelectedId(point.id);
        setEditingPointId(point.id);
        setEditingText(tempoPointFlagLabel(point));
    }, []);

    // ── 处理时间标尺右键菜单 / 悬浮标签发来的编辑/新建请求 ──
    // 新建变化点采用“暂存”模式：先在本地构建临时点用于对话框初值，
    // 用户点“确定”时才一次性提交到 Redux/后端；点“取消”不产生任何数据。
    useEffect(() => {
        if (!editRequest) return;
        if (consumedEditRequestRef.current === editRequest) return;
        consumedEditRequestRef.current = editRequest;
        const { pointId, positionSec, focus, mode } = editRequest;
        onEditRequestHandled();

        if (pointId) {
            // 编辑已有变化点。
            if (!tempoMap) return;
            const point = tempoMap.points.find((p) => p.id === pointId);
            if (!point) return;
            if (mode === "inline") {
                // 悬浮标签双击：直接进入该变化点的输入编辑状态。
                queueMicrotask(() => {
                    startInlineEdit(point);
                });
                return;
            }
            queueMicrotask(() => {
                setSelectedId(point.id);
                openDialogState(point.id, point, tempoMap.points[0].id === point.id, focus);
            });
            return;
        }
        if (positionSec == null) return;

        // 新建变化点（暂存，不提交）。
        const base = tempoMap ?? null;
        const mapFallback = baseFallbackOf(base);
        // 与双击新增一致：吸附开启时使用 Clip 吸附规则。
        let sec = positionSec;
        if (snapEnabled && base) {
            sec = snapTempoPosition(positionSec, base, mapFallback.bpm);
        }
        const { map, point } = createTempoPointAt(base, sec, mapFallback, {
            projectScale: projectScale ?? undefined,
            projectScaleName,
        });
        queueMicrotask(() => {
            setSelectedId(point.id);
            openDialogState(
                point.id,
                point,
                map.points[0].id === point.id,
                focus,
                { positionSec: sec, baseMap: base },
            );
        });
    }, [
        editRequest,
        tempoMap,
        snapEnabled,
        snapTempoPosition,
        baseFallbackOf,
        projectScale,
        projectScaleName,
        onEditRequestHandled,
        openDialogState,
        startInlineEdit,
    ]);

    // ── 旗帜内联编辑（双击蓝色标签 → 原地 TextBox）──
    /**
     * “双击空白处新增”的待提交点：双击只做本地创建（onChange），
     * 与随后的内联编辑合并为一次后端提交（onCommit）——保证用户
     * 只需一次撤销即可完全撤销这次“添加变化点”的编辑；
     * Esc 取消内联编辑时按对话框“取消”语义一并放弃本次新增（纯本地回退，
     * 后端从未见过该点，不产生任何撤销步）。
     */
    const pendingInlineAddRef = useRef<{
        pointId: string;
        baseMap: TempoMap | null;
    } | null>(null);

    /** 离散提交（同步后端）并清空待提交标记（任何提交都会带上待提交点）。 */
    const commitMap = useCallback(
        (next: TempoMap | null) => {
            pendingInlineAddRef.current = null;
            onCommit(next);
        },
        [onCommit],
    );

    /**
     * 解析并应用内联编辑文本。
     * - `commitNow = true`：应用并提交后端（Enter / 失焦路径）；
     * - `commitNow = false`：仅本地应用（onChange），返回应用后的 map
     *   （供“输入中直接拖动标签”路径在指针抬起时随拖拽一并提交）。
     * - 解析失败：待提交的新增点保持现状（返回当前 map 以便提交），
     *   其它点静默放弃（返回 null）。
     * - 返回 null 表示没有任何需要提交的编辑结果。
     */
    const applyInlineEdit = useCallback(
        (commitNow: boolean): TempoMap | null => {
            const id = editingPointId;
            if (!id) return null;
            const pending = pendingInlineAddRef.current;
            if (!tempoMap) {
                // 行隐藏等异常状态：回退尚未提交的新增点。
                if (pending && pending.baseMap) onChange(pending.baseMap);
                pendingInlineAddRef.current = null;
                return null;
            }
            const pointIndex = tempoMap.points.findIndex((p) => p.id === id);
            const point = pointIndex >= 0 ? tempoMap.points[pointIndex] : null;
            if (!point) {
                // 点已不存在（如已被撤销）：清空待提交状态即可。
                pendingInlineAddRef.current = null;
                return null;
            }
            const parsed = parseTempoPointText(editingText, customScalePresets);
            if (!parsed) {
                // 解析失败：若该点是本次双击新增（尚未提交后端），保持现状
                // 并返回当前 map（由调用方决定何时提交），保证新增不丢失；
                // 否则静默放弃本次编辑。
                if (pending && pending.pointId === id) {
                    if (commitNow) {
                        commitMap(tempoMap);
                        return null;
                    }
                    return tempoMap;
                }
                pendingInlineAddRef.current = null;
                return null;
            }
            // 初始点（工程基准记录）不能“跟随之前的拍号”：输入未含拍号时保持原拍号。
            if (pointIndex === 0 && !parsed.timeSignature) {
                parsed.timeSignature =
                    point.timeSignature ?? { numerator: 4, denominator: 4 };
            }
            const next = updateTempoPoint(tempoMap, id, parsed);
            if (commitNow) {
                commitMap(next);
                return null;
            }
            onChange(next);
            return next;
        },
        [editingPointId, editingText, tempoMap, customScalePresets, onChange, commitMap],
    );

    /** 确认：解析成功则应用到变化点；解析失败静默放弃（不应用）。 */
    const commitInlineEdit = useCallback(() => {
        const id = editingPointId;
        if (!id || inlineEditLockRef.current) return;
        inlineEditLockRef.current = true;
        setEditingPointId(null);
        applyInlineEdit(true);
    }, [editingPointId, applyInlineEdit]);

    const cancelInlineEdit = useCallback(() => {
        inlineEditLockRef.current = true;
        setEditingPointId(null);
        const pending = pendingInlineAddRef.current;
        if (pending) {
            // Esc：放弃本次双击新增（后端从未收到该点，纯本地回退，无撤销步）。
            pendingInlineAddRef.current = null;
            onChange(pending.baseMap);
        }
    }, [onChange]);

    /**
     * “输入状态下直接拖动标签”的意图探测。
     *
     * 判定规则：指针保持在输入框边界（含容差）以内 → 选中文字；
     * 明显越出边界 → 判定为“拖动标签”，确认编辑并从当前指针位置
     * 开始拖动（指针抬起时把“编辑 + 移动”一次性提交后端）。
     *
     * ★ 已选中字符时的处理：WebView2（Chromium）在输入框内已有选区时，
     * 按下左键拖动会启动原生“拖拽选中文本”，pointermove 会被原生拖拽
     * 吞掉、永远到不了本监听器。因此在按下时若有选区就 preventDefault
     * 阻止原生拖拽（抬起时手动恢复“点击放置光标”），并对部分内核路径
     * 仍会触发的 `dragstart` 做兜底：取消原生拖拽并直接判定为拖动标签。
     * 初始点不可拖动，始终保留普通文本选择行为。
     */
    const inlineDragArmedRef = useRef(false);

    /** 判定为“拖动标签”后的统一接管：确认编辑 + 从 (clientX, clientY) 开始拖动。 */
    const armInlineDrag = useCallback(
        (point: TempoPoint, clientX: number) => {
            if (inlineDragArmedRef.current) return;
            inlineDragArmedRef.current = true;
            try {
                document.getSelection()?.removeAllRanges();
            } catch {
                // 忽略选择清除失败
            }
            // 确认编辑（仅本地应用；指针抬起时随拖拽一并提交后端，
            // 保证“编辑 + 移动”只产生一个撤销步）。
            inlineEditLockRef.current = true;
            setEditingPointId(null);
            const appliedMap = applyInlineEdit(false);
            // 从当前指针位置开始拖动（保持指针与标签的相对偏移不变）。
            dragRef.current = {
                pointId: point.id,
                startClientX: clientX,
                startSec: point.positionSec,
            };
            // 即便指针抬起时没有任何进一步移动，也要把“确认的编辑/新增”
            // 提交出去（否则仅停留在本地 Redux）。
            dragDraftRef.current =
                appliedMap ?? (pendingInlineAddRef.current ? tempoMap : null);
            setDraggingId(point.id);
            setSelectedId(point.id);
        },
        [applyInlineEdit, tempoMap],
    );

    const startInlineDragProbe = useCallback(
        (point: TempoPoint, isFirst: boolean, e: React.PointerEvent<HTMLInputElement>) => {
            if (isFirst || e.button !== 0) return;
            const inputEl = e.currentTarget;
            const rect = inputEl.getBoundingClientRect();
            const margin = INLINE_DRAG_TO_MOVE_BOUNDARY_MARGIN_PX;
            const startClientX = e.clientX;
            const startClientY = e.clientY;
            const hadSelection =
                inputEl.selectionStart !== inputEl.selectionEnd;
            inlineDragArmedRef.current = false;

            const cleanup = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
            };

            const onMove = (ev: PointerEvent) => {
                if (inlineDragArmedRef.current) return;
                // 智能判定：指针仍在输入框边界（含容差）内 → 选中文字；
                // 越出边界 → 拖动标签。
                const inside =
                    ev.clientX >= rect.left - margin &&
                    ev.clientX <= rect.right + margin &&
                    ev.clientY >= rect.top - margin &&
                    ev.clientY <= rect.bottom + margin;
                if (inside) return;
                armInlineDrag(point, ev.clientX);
            };

            const onUp = (ev: PointerEvent) => {
                cleanup();
                if (inlineDragArmedRef.current) return;
                if (!hadSelection) return;
                // 按下时因存在选区而阻止了默认行为（防止原生文本拖拽吞掉
                // pointermove），这里手动恢复“点击放置光标”：
                // 位移很小（点击）时把插入光标放到点击位置。
                const dx = ev.clientX - startClientX;
                const dy = ev.clientY - startClientY;
                if (dx * dx + dy * dy > 36) return;
                const charWidth = 6; // 与标签宽度估算一致（9px 字号）
                const idx = Math.max(
                    0,
                    Math.min(
                        inputEl.value.length,
                        Math.round((ev.clientX - rect.left - 4 + charWidth / 2) / charWidth),
                    ),
                );
                try {
                    inputEl.setSelectionRange(idx, idx);
                } catch {
                    // 忽略光标放置失败
                }
            };

            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [armInlineDrag],
    );

    /**
     * 右键输入框 → 打开“速度映射变化点”编辑窗口：
     * 对话框初值继承自输入框中的参数（解析失败时回退为变化点当前值）。
     * 输入框关闭后由对话框的“确定”一次性提交（与输入编辑合并为
     * 一个撤销步）。
     */
    const openDialogFromInlineEdit = useCallback(() => {
        const id = editingPointId;
        if (!id || !tempoMap) return;
        const point = tempoMap.points.find((p) => p.id === id);
        if (!point) return;
        const parsed = parseTempoPointText(editingText, customScalePresets);
        const dialogPoint: TempoPoint = parsed
            ? {
                  ...point,
                  bpm: parsed.bpm,
                  timeSignature: parsed.timeSignature,
                  scale: parsed.scale,
              }
            : point;
        // 关闭输入框（不触发 blur 提交）；待提交状态保留，确认时一并提交。
        inlineEditLockRef.current = true;
        setEditingPointId(null);
        openDialogState(id, dialogPoint, tempoMap.points[0].id === id, null);
    }, [editingPointId, editingText, tempoMap, customScalePresets, openDialogState]);

    // ── 双击空白处：本地创建“继承前一个标签”的变化点并进入内联编辑 ──
    // 新变化点继承该位置生效的 BPM，拍号与音阶均为“跟随”（即 createTempoPointAt
    // 的默认语义：timeSignature: null、scale: null）；创建先只写入本地 Redux，
    // 待内联编辑确认（Enter / 失焦）时一次性提交后端 —— 添加与编辑合并为
    // 一个撤销步；Esc 取消时放弃本次新增。
    const handleRowDoubleClick = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            // 行本身位于被 translateX(-scrollLeft) 平移的内容层内：
            // getBoundingClientRect().left 已包含滚动位移，直接相减即为
            // 内容坐标 x；若再额外加 scrollLeft 会重复计滚动，
            // 使新建点位置随滚动偏移（视图越靠右偏离越大）。
            const bounds = e.currentTarget.getBoundingClientRect();
            let sec = Math.max(0, (e.clientX - bounds.left) / Math.max(1e-9, pxPerSec));
            const base = tempoMap ?? null;
            const mapFallback = baseFallbackOf(base);
            if (snapEnabled) {
                sec = snapTempoPosition(sec, base, mapFallback.bpm);
            }
            const { map, point } = createTempoPointAt(base, sec, mapFallback, {
                projectScale: projectScale ?? undefined,
                projectScaleName,
            });
            // 与已有点过近（< 1e-6s）时该点会被规范化丢弃：放弃本次双击，避免
            // 进入一个“即将消失”的标签的内联编辑状态。
            const tooClose = map.points.some(
                (p) => p.id !== point.id && Math.abs(p.positionSec - point.positionSec) < 1e-6,
            );
            if (tooClose) return;
            // 上一次双击的新增点若尚未提交，先合并提交（保持其存在），
            // 否则其“待提交”状态会被本次覆盖而无法单独取消。
            if (pendingInlineAddRef.current && tempoMap) {
                commitMap(tempoMap);
            }
            // 仅本地创建；随后进入该新标签的输入编辑状态。
            onChange(map);
            pendingInlineAddRef.current = { pointId: point.id, baseMap: base };
            startInlineEdit(point);
        },
        [
            tempoMap,
            pxPerSec,
            snapEnabled,
            snapTempoPosition,
            baseFallbackOf,
            projectScale,
            projectScaleName,
            onChange,
            commitMap,
            startInlineEdit,
        ],
    );

    // ── 拖拽移动变化点 ──
    const startFlagDrag = useCallback(
        (point: TempoPoint, isFirst: boolean, e: React.PointerEvent) => {
            if (isFirst || e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            try {
                (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
            } catch {
                // ignore
            }
            dragRef.current = {
                pointId: point.id,
                startClientX: e.clientX,
                startSec: point.positionSec,
            };
            setDraggingId(point.id);
            setSelectedId(point.id);
        },
        [],
    );

    useEffect(() => {
        if (!draggingId || !tempoMap) return;
        const handleMove = (e: PointerEvent) => {
            const drag = dragRef.current;
            if (!drag || !tempoMap) return;
            const dx = e.clientX - drag.startClientX;
            const rawSec = Math.max(0, drag.startSec + dx / Math.max(1e-9, pxPerSec));
            const mapFallback = baseFallbackOf(tempoMap);
            let sec = rawSec;
            if (snapEnabled && !isModifierActive(noSnapKb, e)) {
                sec = snapTempoPosition(rawSec, tempoMap, mapFallback.bpm, drag.startSec);
            }
            // 不与其它点重叠、不越过相邻点、不越过工程末尾：
            // 用相邻点钳制实现（最小间距按工程 BPM 折算 1/16 拍）。
            // 快速拖拽若跨越相邻点，点数组会乱序（updateTempoPoint 虽会
            // 防御性重排，但 UI 上点会互相穿插），钳制是正确行为。
            const minGapSec = 60 / Math.max(1, mapFallback.bpm) / 16;
            const idx = tempoMap.points.findIndex((p) => p.id === drag.pointId);
            if (idx >= 0) {
                const prevSec =
                    idx > 0 ? tempoMap.points[idx - 1].positionSec + minGapSec : 0;
                const nextSec =
                    idx + 1 < tempoMap.points.length
                        ? tempoMap.points[idx + 1].positionSec - minGapSec
                        : projectSec;
                // 相邻点间距不足 2×minGapSec 时退化为钳到 prevSec。
                sec = Math.min(Math.max(prevSec, sec), Math.max(nextSec, prevSec));
            }
            const draft = updateTempoPoint(tempoMap, drag.pointId, { positionSec: sec });
            dragDraftRef.current = draft;
            onChange(draft);
        };
        const handleUp = () => {
            const draft = dragDraftRef.current;
            dragRef.current = null;
            dragDraftRef.current = null;
            setDraggingId(null);
            if (draft) commitMap(draft);
        };
        window.addEventListener("pointermove", handleMove);
        window.addEventListener("pointerup", handleUp);
        window.addEventListener("pointercancel", handleUp);
        return () => {
            window.removeEventListener("pointermove", handleMove);
            window.removeEventListener("pointerup", handleUp);
            window.removeEventListener("pointercancel", handleUp);
        };
    }, [
        draggingId,
        tempoMap,
        pxPerSec,
        snapEnabled,
        noSnapKb,
        snapTempoPosition,
        baseFallbackOf,
        onChange,
        commitMap,
        projectSec,
    ]);

    // ── 可见性计算 ──
    const visibleState = useMemo(() => {
        if (!visibleRow || !tempoMap) return null;
        const bufferPx = Math.max(120, viewportWidth * 0.3);
        const leftPx = scrollLeft - bufferPx;
        const rightPx = scrollLeft + viewportWidth + bufferPx;
        const flags: Array<{ point: TempoPoint; left: number; isFirst: boolean; index: number }> = [];
        for (let i = 0; i < tempoMap.points.length; i += 1) {
            const left = tempoMap.points[i].positionSec * pxPerSec;
            if (left >= leftPx - 220 && left <= rightPx + 220) {
                flags.push({ point: tempoMap.points[i], left, isFirst: i === 0, index: i });
            }
        }
        const segments = tempoMapSegments(tempoMap, projectSec);
        // 段内参数不再以重复文字展示（由变化点旗帜 + 视口左侧悬浮标签承担）。
        return { flags, segments };
    }, [visibleRow, tempoMap, pxPerSec, scrollLeft, viewportWidth, projectSec]);

    const segmentsForBoundaries = visibleState?.segments ?? [];
    const segmentBoundaries = segmentsForBoundaries.map((seg) => seg.startSec * pxPerSec);

    const closeDialog = () => {
        // 从内联输入打开的对话框被取消时：待提交的新增点（双击新增）
        // 若仍未提交，按“点击别处确认新增”的语义提交，保持前后端一致。
        if (pendingInlineAddRef.current && tempoMap) {
            commitMap(tempoMap);
        }
        setDialogState(null);
    };

    const confirmDialog = (patch: {
        bpm: number;
        timeSignature: { numerator: number; denominator: number } | null;
        scale: TempoMapScaleData | null;
    }) => {
        const st = dialogStateRef.current;
        if (!st?.pointId) {
            setDialogState(null);
            return;
        }
        const pointId = st.pointId;
        // 初始点即工程基准记录，不能“跟随之前的拍号”：防御性保持显式拍号。
        if (st.isFirst && !patch.timeSignature) {
            patch.timeSignature = {
                numerator: Math.max(1, Math.min(32, Math.round(fallback.beatsPerBar || 4))),
                denominator: clampDenominator(fallback.denominator ?? 4),
            };
        }
        // 暂存的新建点：此刻一次性创建并应用表单值（取消则不产生任何数据）。
        if (st.pendingCreate && st.pendingPositionSec != null) {
            const base = st.pendingBaseMap ?? null;
            const mapFallback = baseFallbackOf(base);
            const { map } = createTempoPointAt(base, st.pendingPositionSec, mapFallback, {
                projectScale: projectScale ?? undefined,
                projectScaleName,
            });
            commitMap(updateTempoPoint(map, pointId, patch));
            setDialogState(null);
            return;
        }
        if (!tempoMap) {
            setDialogState(null);
            return;
        }
        commitMap(updateTempoPoint(tempoMap, pointId, patch));
        setDialogState(null);
    };

    const deleteDialogPoint = () => {
        const st = dialogStateRef.current;
        if (!st?.pointId) return;
        // 暂存的新建点尚未提交：直接关闭即可。
        if (st.pendingCreate) {
            setDialogState(null);
            setSelectedId(null);
            return;
        }
        if (!tempoMap) return;
        commitMap(removeTempoPoint(tempoMap, st.pointId));
        setDialogState(null);
        setSelectedId(null);
    };

    /** “跟随之前的音阶”选项展示的上一变化点音阶标签（初始点之前的音阶即工程音阶）。 */
    const previousScaleLabelFor = useCallback(
        (positionSec: number): string => {
            const prev = previousScaleAtSec(tempoMap, positionSec, projectScale ?? undefined);
            return scaleLikeLabel(prev, projectScaleName) ?? "—";
        },
        [tempoMap, projectScale, projectScaleName],
    );

    /** 某位置“之前”生效的拍号（用于“跟随之前的拍号”展示/解析；初始点之前即工程基准记录）。 */
    const previousTimeSignatureFor = useCallback(
        (positionSec: number): { numerator: number; denominator: number } => {
            if (positionSec <= 1e-9) {
                return {
                    numerator: Math.max(1, Math.min(32, Math.round(fallbackBeatsPerBar || 4))),
                    denominator: clampDenominator(fallbackDenominator ?? 4),
                };
            }
            if (!tempoMap || tempoMap.points.length === 0) {
                return {
                    numerator: Math.max(1, Math.min(32, Math.round(fallbackBeatsPerBar || 4))),
                    denominator: clampDenominator(fallbackDenominator ?? 4),
                };
            }
            const idx = tempoMap.points.findIndex((p) => p.positionSec >= positionSec - 1e-6);
            // 前一个点（若本身即第一个点则为其自身）。
            const prevIndex = Math.max(0, idx - 1);
            return effectiveTimeSignatureAt(tempoMap, prevIndex);
        },
        [tempoMap, fallbackBeatsPerBar, fallbackDenominator],
    );

    const previousTimeSignatureLabelFor = useCallback(
        (positionSec: number): string =>
            formatTimeSignature(previousTimeSignatureFor(positionSec)),
        [previousTimeSignatureFor],
    );

    /**
     * 变化点旗帜的自定义悬浮提示（项目统一 ToolTip 样式，data-tooltip + AppTooltip）：
     * 位置按用户主/副时间单位显示；拍号/音阶为“跟随”时展示实际生效值。
     */
    const buildFlagTooltip = useCallback(
        (pointIndex: number): string => {
            if (!tempoMap) return "";
            const point = tempoMap.points[pointIndex];
            const cursor = formatCursorTime(
                primaryUnit,
                secondaryUnit,
                point.positionSec,
                timeContext,
            );
            const positionLine = cursor.secondaryLabel
                ? `${cursor.primaryLabel} / ${cursor.secondaryLabel}`
                : cursor.primaryLabel;
            const sig = effectiveTimeSignatureAt(tempoMap, pointIndex);
            const effScale = effectiveScaleAtSec(tempoMap, point.positionSec, projectScale ?? undefined);
            const effScaleLabel = scaleLikeLabel(effScale, projectScaleName) ?? "—";
            return [
                `${t("tempo_map_tooltip_position")}${positionLine}`,
                `${t("tempo_map_tooltip_bpm")}${formatTempoBpm(point.bpm)}`,
                `${t("tempo_map_tooltip_time_signature")}${formatTimeSignature(sig)}`,
                `${t("tempo_map_tooltip_scale")}${effScaleLabel}`,
            ].join("\n");
        },
        [tempoMap, primaryUnit, secondaryUnit, timeContext, projectScale, projectScaleName, t],
    );

    /** 标签的内联输入框（输入编辑状态）。 */
    const renderInlineInput = (point: TempoPoint, isFirst: boolean) => (
        <input
            autoFocus
            value={editingText}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setEditingText(e.target.value)
            }
            onFocus={(e) => e.target.select()}
            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                e.stopPropagation();
                if (e.key === "Enter") {
                    e.preventDefault();
                    commitInlineEdit();
                } else if (e.key === "Escape") {
                    e.preventDefault();
                    cancelInlineEdit();
                }
            }}
            onBlur={commitInlineEdit}
            onPointerDown={(e) => {
                e.stopPropagation();
                if (isFirst || e.button !== 0) return;
                // 已有选中字符时，阻止 WebView2 的原生
                // “拖拽选中文本”（否则 pointermove 会被
                // 原生拖拽吞掉，拖拽判定失效）。
                const inputEl = e.currentTarget;
                if (inputEl.selectionStart !== inputEl.selectionEnd) {
                    e.preventDefault();
                }
                // 指针越出输入框边界时判定为“拖动标签”：
                // 确认编辑并从当前指针位置开始拖拽该标签。
                startInlineDragProbe(point, isFirst, e);
            }}
            onDragStart={(e) => {
                // 兜底：部分内核路径仍会触发原生文本拖拽的
                // dragstart —— 取消原生拖拽并直接判定为
                // “拖动标签”（确认编辑 + 开始拖拽）。
                if (isFirst) return;
                e.preventDefault();
                armInlineDrag(point, e.clientX);
            }}
            onMouseDown={(e) => e.stopPropagation()}
            onContextMenu={(e) => {
                // 右键输入框 → 打开“速度映射变化点”编辑窗口
                // （参数继承自输入框内容，确认后一步撤销）。
                e.preventDefault();
                e.stopPropagation();
                openDialogFromInlineEdit();
            }}
            onDoubleClick={(e) => e.stopPropagation()}
            className="text-[9px] leading-[11px] font-medium rounded-[2px] outline-none border-none"
            style={{
                backgroundColor: "var(--qt-panel)",
                color: "var(--qt-text)",
                width: Math.max(40, editingText.length * 6 + 18),
                padding: "0 4px",
                height: 15,
                boxShadow:
                    "inset 0 0 0 1px color-mix(in srgb, var(--qt-border) 70%, transparent)",
            }}
        />
    );

    // 行隐藏（无 Tempo Map 数据）时对话框仍必须能打开：
    // “新建第一个变化点”是暂存模式，此时 map 尚未提交，行仍处于隐藏状态。
    if (!visibleRow) {
        return dialogState ? (
            <TempoPointDialog
                key={dialogState.seq}
                open
                point={dialogState.point}
                isFirst={dialogState.isFirst}
                previousTimeSignatureLabel={
                    dialogState.point
                        ? previousTimeSignatureLabelFor(dialogState.point.positionSec)
                        : "—"
                }
                previousTimeSignature={
                    dialogState.point
                        ? previousTimeSignatureFor(dialogState.point.positionSec)
                        : { numerator: 4, denominator: 4 }
                }
                previousScaleLabel={
                    dialogState.point
                        ? previousScaleLabelFor(dialogState.point.positionSec)
                        : "—"
                }
                customScalePresets={customScalePresets}
                focus={dialogState.focus}
                t={t}
                onCancel={closeDialog}
                onConfirm={confirmDialog}
                onDelete={deleteDialogPoint}
            />
        ) : null;
    }

    return (
        <div
            className="absolute left-0 select-none"
            style={{
                top: 48,
                height: TEMPO_ROW_HEIGHT_PX + 4,
                paddingTop: 4,
                // 行位于被 translateX(-scrollLeft) 平移的内容层内：`left-0 right-0`
                // 的宽度只等于视口宽度（内容坐标 0..viewportWidth），水平滚动后
                // 视口右侧区域不在行的命中范围内，双击“新增变化点”会失效
                // （事件落到标尺上并误移动播放头）—— 显式撑到视口右缘 + 标签余量。
                // 注意不要用 projectSec * pxPerSec（高倍缩放时可达数亿像素，
                // 浏览器布局开销巨大）；右侧只需覆盖可见区域即可。
                width: scrollLeft + viewportWidth + 400,
            }}
            onDoubleClick={handleRowDoubleClick}
        >
            {/* 与时间标尺的分隔横线由 TimeRuler 在标尺盒内渲染（视口固定宽度，不随缩放伸缩）。 */}
            {visibleState ? (
                <div className="absolute inset-0" style={{ top: 4 }}>
                    {/* 段分隔标记 */}
                    {segmentBoundaries.map((left, index) => (
                        <div
                            key={`b_${index}_${left}`}
                            className="absolute top-0 bottom-0 w-px"
                            style={{ left, backgroundColor: "var(--qt-border)", opacity: 0.9 }}
                        />
                    ))}
                    {/* 变化点旗帜 */}
                    {visibleState.flags.map(({ point, left, isFirst, index }) => {
                        const inlineEditing = editingPointId === point.id;
                        const sigText = point.timeSignature
                            ? formatTimeSignature(point.timeSignature)
                            : null;
                        const scaleText = tempoPointScaleShortLabel(point.scale);
                        const tooltipText = buildFlagTooltip(index);
                        // 旗帜完全滚出画面左侧时（悬浮标签双击进入编辑），
                        // 输入框改在视口左侧显示（即悬浮标签的位置）。
                        const flagFullyOffscreen =
                            left + tempoFlagLabelWidthPx(tempoPointFlagLabel(point)) < scrollLeft;

                        return (
                            <div
                                key={point.id}
                                className="absolute top-0 bottom-0 flex items-center group select-none"
                                style={{
                                    left,
                                    cursor:
                                        isFirst
                                            ? "pointer"
                                            : draggingId === point.id
                                              ? "grabbing"
                                              : "grab",
                                }}
                                onPointerDown={(e) => startFlagDrag(point, isFirst, e)}
                                onDoubleClick={(e) => {
                                    e.stopPropagation();
                                    startInlineEdit(point);
                                }}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setSelectedId(point.id);
                                }}
                                onContextMenu={(e) => {
                                    // 右键标签 → 直接弹出“速度映射变化点”编辑窗口
                                    // （不再显示时间标尺的右键上下文菜单）。
                                    e.preventDefault();
                                    e.stopPropagation();
                                    openDialogState(point.id, point, isFirst, null);
                                }}
                            >
                                <div
                                    className="w-px h-full shrink-0"
                                    style={{ backgroundColor: "var(--qt-highlight)" }}
                                />
                                {inlineEditing && !flagFullyOffscreen
                                    ? renderInlineInput(point, isFirst)
                                    : null}
                                {!inlineEditing ? (
                                    <div
                                        className="px-1 rounded-[2px] text-[9px] leading-[11px] whitespace-nowrap font-medium"
                                        style={{
                                            backgroundColor: "var(--qt-panel)",
                                            color: "var(--qt-text)",
                                            boxShadow:
                                                "inset 0 0 0 1px color-mix(in srgb, var(--qt-border) 70%, transparent)",
                                            opacity: draggingId === point.id ? 1 : 0.92,
                                            outline:
                                                selectedId === point.id
                                                    ? "1px solid var(--qt-highlight)"
                                                    : "none",
                                            outlineOffset: 1,
                                        }}
                                        data-tooltip={tooltipText}
                                    >
                                        {formatTempoBpm(point.bpm)}
                                        {sigText ? (
                                            <span className="opacity-85"> {sigText}</span>
                                        ) : null}
                                        {scaleText ? (
                                            <span className="opacity-85"> - {scaleText}</span>
                                        ) : null}
                                    </div>
                                ) : null}
                            </div>
                        );
                    })}
                    {/* 悬浮标签双击进入编辑：旗帜在画面外时，输入框显示在视口左侧
                        （与悬浮标签同一位置，覆盖其上方）。 */}
                    {(() => {
                        const offscreenFlag = visibleState.flags.find((f) => {
                            if (editingPointId !== f.point.id) return false;
                            return (
                                f.left +
                                    tempoFlagLabelWidthPx(tempoPointFlagLabel(f.point)) <
                                scrollLeft
                            );
                        });
                        if (!offscreenFlag) return null;
                        return (
                            <div
                                className="absolute top-0 bottom-0 flex items-center"
                                style={{ left: scrollLeft + 2, zIndex: 26 }}
                            >
                                {renderInlineInput(offscreenFlag.point, offscreenFlag.isFirst)}
                            </div>
                        );
                    })()}
                </div>
            ) : null}

            {dialogState ? (
                <TempoPointDialog
                    key={dialogState.seq}
                    open
                    point={dialogState.point}
                    isFirst={dialogState.isFirst}
                    previousTimeSignatureLabel={
                        dialogState.point
                            ? previousTimeSignatureLabelFor(dialogState.point.positionSec)
                            : "—"
                    }
                    previousTimeSignature={
                        dialogState.point
                            ? previousTimeSignatureFor(dialogState.point.positionSec)
                            : { numerator: 4, denominator: 4 }
                    }
                    previousScaleLabel={
                        dialogState.point
                            ? previousScaleLabelFor(dialogState.point.positionSec)
                            : "—"
                    }
                    customScalePresets={customScalePresets}
                    focus={dialogState.focus}
                    t={t}
                    onCancel={closeDialog}
                    onConfirm={confirmDialog}
                    onDelete={deleteDialogPoint}
                />
            ) : null}
        </div>
    );
};
