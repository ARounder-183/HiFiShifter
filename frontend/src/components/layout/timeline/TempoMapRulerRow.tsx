import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Checkbox, Dialog, Flex, Select, Text, TextField } from "@radix-ui/themes";
import type { GridSize } from "../../../features/session/sessionTypes";
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
    snapSecToTempoGrid,
    TEMPO_DENOMINATORS,
    tempoMapSegments,
    tempoPointFlagLabel,
    tempoPointScaleShortLabel,
    updateTempoPoint,
} from "../../../utils/tempoMap";
import type { TempoMap, TempoPoint, TempoMapScaleData } from "../../../utils/tempoMap";
import type { TimeFormatContext, TimeUnit, TimeUnitChoice } from "./timeFormat";
import { formatCursorTime } from "./timeFormat";
import { gridStepBeats } from "./grid";
import { useAppSelector } from "../../../app/hooks";
import { isModifierActive, selectKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { applySelectWheelChange } from "../../../utils/selectWheel";

/** Tempo Map 行高度（不含分隔线）。 */
export const TEMPO_ROW_HEIGHT_PX = 17;

export interface TempoPointEditRequest {
    /** 编辑已有变化点（为 null 表示新建）。 */
    pointId: string | null;
    /** 新建点的位置（秒）；编辑已有变化点时忽略。 */
    positionSec: number | null;
    /** 对话框初始焦点。 */
    focus: "tempo" | "timeSignature" | "scale" | null;
}

interface TempoMapRulerRowProps {
    tempoMap: TempoMap | null;
    visible: boolean;
    pxPerSec: number;
    scrollLeft: number;
    viewportWidth: number;
    projectSec: number;
    grid: GridSize;
    gridSnapEnabled: boolean;
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
    /** 由时间标尺右键菜单发出的编辑/新建请求。 */
    editRequest: TempoPointEditRequest | null;
    onEditRequestHandled: () => void;
    /** 编辑对话框打开/关闭通知（用于抑制标尺悬浮时间提示）。 */
    onDialogOpenChange?: (open: boolean) => void;
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
    gridSnapEnabled,
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

    // ── 处理时间标尺右键菜单发来的编辑/新建请求 ──
    // 新建变化点采用“暂存”模式：先在本地构建临时点用于对话框初值，
    // 用户点“确定”时才一次性提交到 Redux/后端；点“取消”不产生任何数据。
    useEffect(() => {
        if (!editRequest) return;
        if (consumedEditRequestRef.current === editRequest) return;
        consumedEditRequestRef.current = editRequest;
        const { pointId, positionSec, focus } = editRequest;
        onEditRequestHandled();

        if (pointId) {
            // 编辑已有变化点。
            if (!tempoMap) return;
            const point = tempoMap.points.find((p) => p.id === pointId);
            if (!point) return;
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
        const { map, point } = createTempoPointAt(base, positionSec, mapFallback, {
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
                { positionSec, baseMap: base },
            );
        });
    }, [
        editRequest,
        tempoMap,
        baseFallbackOf,
        projectScale,
        projectScaleName,
        onEditRequestHandled,
        openDialogState,
    ]);

    // ── 双击空白处：新建变化点并打开对话框（暂存，确认后提交） ──
    const handleRowDoubleClick = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            const bounds = e.currentTarget.getBoundingClientRect();
            let sec = Math.max(0, (e.clientX - bounds.left + scrollLeft) / Math.max(1e-9, pxPerSec));
            const base = tempoMap ?? null;
            const mapFallback = baseFallbackOf(base);
            if (gridSnapEnabled) {
                sec = snapSecToTempoGrid(sec, base, gridStepBeats(grid), mapFallback.bpm);
            }
            const { map, point } = createTempoPointAt(base, sec, mapFallback, {
                projectScale: projectScale ?? undefined,
                projectScaleName,
            });
            setSelectedId(point.id);
            openDialogState(point.id, point, map.points[0].id === point.id, null, {
                positionSec: sec,
                baseMap: base,
            });
        },
        [
            tempoMap,
            scrollLeft,
            pxPerSec,
            gridSnapEnabled,
            grid,
            baseFallbackOf,
            projectScale,
            projectScaleName,
            openDialogState,
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
            let sec = Math.max(0, drag.startSec + dx / Math.max(1e-9, pxPerSec));
            const mapFallback = baseFallbackOf(tempoMap);
            if (gridSnapEnabled) {
                sec = snapSecToTempoGrid(sec, tempoMap, gridStepBeats(grid), mapFallback.bpm);
            }
            // 不与其它点重叠（最小间距按当前段 BPM 折算 1/16 拍）。
            const minGapSec = 60 / Math.max(1, mapFallback.bpm) / 16;
            for (const p of tempoMap.points) {
                if (p.id === drag.pointId) continue;
                if (Math.abs(p.positionSec - sec) < minGapSec) return;
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
            if (draft) onCommit(draft);
        };
        window.addEventListener("pointermove", handleMove);
        window.addEventListener("pointerup", handleUp);
        window.addEventListener("pointercancel", handleUp);
        return () => {
            window.removeEventListener("pointermove", handleMove);
            window.removeEventListener("pointerup", handleUp);
            window.removeEventListener("pointercancel", handleUp);
        };
    }, [draggingId, tempoMap, pxPerSec, gridSnapEnabled, grid, baseFallbackOf, onChange, onCommit]);

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

    // ── 旗帜内联编辑（双击蓝色标签 → 原地 TextBox）──
    /** 防重复提交：Enter 提交后输入框卸载可能再次触发 blur；Esc 取消后同理。 */
    const inlineEditLockRef = useRef(false);
    const startInlineEdit = useCallback((point: TempoPoint) => {
        inlineEditLockRef.current = false;
        setSelectedId(point.id);
        setEditingPointId(point.id);
        setEditingText(tempoPointFlagLabel(point));
    }, []);

    /** 确认：解析成功则应用到变化点；解析失败静默放弃（不应用）。 */
    const commitInlineEdit = useCallback(() => {
        const id = editingPointId;
        if (!id || inlineEditLockRef.current) return;
        inlineEditLockRef.current = true;
        setEditingPointId(null);
        if (!tempoMap) return;
        const pointIndex = tempoMap.points.findIndex((p) => p.id === id);
        const point = pointIndex >= 0 ? tempoMap.points[pointIndex] : null;
        if (!point) return;
        const parsed = parseTempoPointText(editingText, customScalePresets);
        if (!parsed) return;
        // 初始点（工程基准记录）不能“跟随之前的拍号”：输入未含拍号时保持原拍号。
        if (pointIndex === 0 && !parsed.timeSignature) {
            parsed.timeSignature =
                point.timeSignature ?? { numerator: 4, denominator: 4 };
        }
        onCommit(updateTempoPoint(tempoMap, id, parsed));
    }, [editingPointId, editingText, tempoMap, customScalePresets, onCommit]);

    const cancelInlineEdit = useCallback(() => {
        inlineEditLockRef.current = true;
        setEditingPointId(null);
    }, []);

    const closeDialog = () => setDialogState(null);

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
            onCommit(updateTempoPoint(map, pointId, patch));
            setDialogState(null);
            return;
        }
        if (!tempoMap) {
            setDialogState(null);
            return;
        }
        onCommit(updateTempoPoint(tempoMap, pointId, patch));
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
        onCommit(removeTempoPoint(tempoMap, st.pointId));
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
            className="absolute left-0 right-0 select-none"
            style={{ top: 48, height: TEMPO_ROW_HEIGHT_PX + 4, paddingTop: 4 }}
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
                            >
                                <div
                                    className="w-px h-full shrink-0"
                                    style={{ backgroundColor: "var(--qt-highlight)" }}
                                />
                                {inlineEditing ? (
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
                                        onPointerDown={(e) => e.stopPropagation()}
                                        onMouseDown={(e) => e.stopPropagation()}
                                        onDoubleClick={(e) => e.stopPropagation()}
                                        className="text-[9px] leading-[11px] font-medium rounded-[2px] outline-none border-none"
                                        style={{
                                            backgroundColor: "var(--qt-highlight)",
                                            color: "var(--qt-window)",
                                            width: Math.max(40, editingText.length * 6 + 18),
                                            padding: "0 4px",
                                            height: 15,
                                            boxShadow: "inset 0 0 0 1px color-mix(in srgb, var(--qt-window) 55%, transparent)",
                                        }}
                                    />
                                ) : (
                                    <div
                                        className="px-1 rounded-[2px] text-[9px] leading-[11px] whitespace-nowrap font-medium"
                                        style={{
                                            backgroundColor: "var(--qt-highlight)",
                                            color: "var(--qt-window)",
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
                                )}
                            </div>
                        );
                    })}
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
