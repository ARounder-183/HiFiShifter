/**
 * 颤音预设管理器。
 *
 * 【结构】左侧预设列表、右侧编辑器（波形预览 + 参数），照 `CustomScaleDialog`
 * 的分工：列表负责"选哪一个"，编辑器负责"捏成什么样"。
 *
 * 【只读系统预设】系统预设禁用一切字段，只有「复制为自定义」可用。这样出厂
 * 预设永远可复原，而"改坏了"不会变成不可逆 —— 比"允许改但提供重置"更简单，
 * 也不会让用户对着一个与出厂说明不符的"自然"预设困惑。
 *
 * 【草稿-保存】编辑改的是局部草稿，点「保存」才 dispatch + 落盘。与自动备份 /
 * 录音 / 渲染缓存几个对话框一致；预设是"调完再定"的东西，逐键落盘只会让撤销
 * 与失败恢复都变复杂。
 */

import { useEffect, useMemo, useState } from "react";
import { Box, Flex, ScrollArea, TextField } from "@radix-ui/themes";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    persistUiSettings,
    removeVibratoPreset,
    reorderVibratoPreset,
    setActiveVibratoPreset,
    upsertVibratoPreset,
} from "../../features/session/sessionSlice";
import {
    MAX_VIBRATO_PRESETS,
    createVibratoPresetId,
    duplicateVibratoPreset,
    sanitizeVibratoPreset,
} from "../../features/vibrato/vibratoPresets";
import { resolveVibratoPresets } from "../../features/vibrato/vibratoPresetList";
import { shapeUsesSkew } from "../../features/vibrato/vibratoCycle";
import { depthStepUnitFor } from "../../features/vibrato/vibratoDepth";
import { estimateCycles } from "../../features/vibrato/vibratoCurve";
import type {
    BaselineMode,
    EnvelopeCurve,
    VibratoPreset,
    VibratoRateMode,
    WaveShape,
} from "../../features/vibrato/vibratoTypes";
import {
    AppButton,
    AppConfirmDialog,
    AppDialog,
    AppField,
    AppForm,
    AppFormSection,
    AppListRow,
    AppNumberField,
    AppSegmentedControl,
    AppSelect,
    AppSlider,
    AppSliderReadout,
    AppSwitchRow,
} from "../../ui";
import { VibratoPreviewCanvas } from "./vibrato/VibratoPreviewCanvas";
import {
    BASELINE_MODE_KEYS,
    BASELINE_MODE_ORDER,
    ENVELOPE_CURVE_KEYS,
    ENVELOPE_CURVE_ORDER,
    RATE_MODE_KEYS,
    WAVE_SHAPE_KEYS,
    WAVE_SHAPE_ORDER,
    buildVibratoPreview,
    depthToCents,
    depthForParam,
    formatNumber,
    vibratoPresetDescription,
    vibratoPresetLabel,
    vibratoPresetSummary,
} from "./vibrato/vibratoDialogLogic";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 预设编辑时的"当前参数"：深度按它换算成原生单位显示与编辑。 */
    editParam?: string;
    paramRange?: { min: number; max: number };
}

/** 列表列的宽度（CSS 像素）。 */
const LIST_WIDTH = 208;
/**
 * 同排两栏共用的高度上限。
 *
 * 【为什么两栏共用一个值】两栏是同一个 `Flex` 行的兄弟，共用上限才能等高 ——
 * 否则列表栏会比参数栏短一截，右下方空出一块。
 *
 * 【为什么必须封顶】对话框正文区自己是可滚动的；只要内容总高不超过它，正文区
 * 就不出现滚动条，于是页面上只有两个**并排**的面板滚动区（不是嵌套的两层）。
 * `min(46vh, 400px)` 让"预览 + 两栏 + 标题 + 页脚"落在对话框 86vh 的上限内。
 */
const PANE_MAX_HEIGHT = "min(46vh, 400px)";

export function VibratoPresetDialog({
    open,
    onOpenChange,
    editParam = "pitch",
    paramRange,
}: Props) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const session = useAppSelector((state: RootState) => state.session);

    const resolved = useMemo(
        () => resolveVibratoPresets(session.vibratoPresets),
        [session.vibratoPresets],
    );
    /**
     * 编辑中的草稿。`null` 表示尚未选过 —— 打开时按活动预设播种。
     *
     * 【为什么不用 `useDialogDraft`】那个钩子播种的是"打开那一刻的值"，而这里
     * 草稿会随列表点选反复替换；用普通 state + 打开时的 effect 更直白。
     */
    const [draft, setDraft] = useState<VibratoPreset | null>(null);
    const [deleteTarget, setDeleteTarget] = useState<VibratoPreset | null>(null);

    // 打开时播种：优先用活动预设，找不到就回落到列表首项。
    useEffect(() => {
        if (!open) return;
        const active =
            resolved.all.find((preset) => preset.id === session.activeVibratoPresetId) ??
            resolved.all[0];
        if (active) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时按当前活动预设播种局部草稿（既有模式）
            setDraft(active);
        }
    }, [open, resolved.all, session.activeVibratoPresetId]);

    const isBuiltin = Boolean(draft?.builtin);
    const previewSamples = useMemo(() => (draft ? buildVibratoPreview(draft) : null), [draft]);

    /** 草稿的局部更新（不落盘）。 */
    function patch(changes: Partial<VibratoPreset>) {
        setDraft((prev) => (prev ? { ...prev, ...changes } : prev));
    }

    function persistPreset(preset: VibratoPreset) {
        dispatch(upsertVibratoPreset(preset));
        void dispatch(persistUiSettings());
    }

    function selectPreset(preset: VibratoPreset) {
        setDraft(preset);
    }

    /** 设为当前使用（拖拽 / 菜单都用它）。 */
    function activatePreset(preset: VibratoPreset) {
        dispatch(setActiveVibratoPreset(preset.id));
        void dispatch(persistUiSettings());
    }

    function handleSave() {
        if (!draft || isBuiltin) return;
        const normalized = sanitizeVibratoPreset(draft);
        persistPreset(normalized);
        setDraft(normalized);
    }

    function handleDuplicate() {
        if (!draft) return;
        const copy = duplicateVibratoPreset(draft);
        persistPreset(copy);
        selectPreset(copy);
        activatePreset(copy);
    }

    function handleCreate() {
        const created = sanitizeVibratoPreset({
            id: createVibratoPresetId(),
            name: t("vibrato_manager_new"),
        });
        persistPreset(created);
        selectPreset(created);
    }

    function handleDelete() {
        if (!deleteTarget) return;
        dispatch(removeVibratoPreset(deleteTarget.id));
        void dispatch(persistUiSettings());
        const removedId = deleteTarget.id;
        setDeleteTarget(null);
        setDraft((prev) => (prev?.id === removedId ? null : prev));
    }

    function movePreset(preset: VibratoPreset, delta: 1 | -1) {
        const index = resolved.user.findIndex((item) => item.id === preset.id);
        if (index < 0) return;
        const next = index + delta;
        if (next < 0 || next >= resolved.user.length) return;
        dispatch(reorderVibratoPreset({ id: preset.id, toIndex: next }));
        void dispatch(persistUiSettings());
    }

    const depthUnit = depthStepUnitFor(editParam);
    const depthValue = draft ? depthForParam(draft.depthCents, editParam, paramRange) : 0;
    const cycleEstimate = draft ? estimateCycles(draft, 320, 5) : 0;

    const customCount = resolved.user.length;
    const atCap = customCount >= MAX_VIBRATO_PRESETS;

    return (
        <>
            <AppDialog
                open={open}
                onOpenChange={onOpenChange}
                title={t("vibrato_manager_title")}
                size="xl"
                actions={[
                    {
                        id: "delete",
                        label: t("vibrato_manager_delete"),
                        intent: "danger",
                        align: "start",
                        disabled: !draft || isBuiltin,
                        // 异步包装：删除走二次确认，不关闭主对话框。
                        onClick: async () => {
                            setDeleteTarget(draft);
                        },
                    },
                    {
                        id: "new",
                        label: t("vibrato_manager_new"),
                        disabled: atCap,
                        onClick: handleCreate,
                    },
                    {
                        id: "duplicate",
                        label: t("vibrato_manager_duplicate"),
                        disabled: !draft,
                        onClick: handleDuplicate,
                    },
                    {
                        id: "save",
                        label: t("ok"),
                        intent: "primary",
                        disabled: !draft || isBuiltin,
                        onClick: handleSave,
                    },
                ]}
            >
                <Flex direction="column" gap="3">
                    {/* ---- 波形预览（整行置顶，不参与任何滚动） ----
                        放在两栏之上而不是塞进参数流的头部：整行宽度读波形更清楚，
                        且它不属于任何滚动区，调参数时**永远**不会滚出视野。 */}
                    {draft && previewSamples ? (
                        <>
                            <Box className="rounded border border-qt-border bg-qt-panel p-2">
                                <VibratoPreviewCanvas
                                    samples={previewSamples}
                                    ariaLabel={t("vibrato_preview")}
                                />
                                <Flex justify="between" mt="1">
                                    <span className="hs-type-caption">
                                        {`±${formatNumber(previewSamples.peakCents)} ${t("vibrato_unit_cents")}`}
                                    </span>
                                    <span className="hs-type-caption">
                                        {t("vibrato_cycles_estimate").replace(
                                            "{count}",
                                            formatNumber(cycleEstimate),
                                        )}
                                    </span>
                                </Flex>
                            </Box>
                            {isBuiltin ? (
                                <span className="hs-type-caption">
                                    {t("vibrato_manager_readonly")}
                                </span>
                            ) : null}
                        </>
                    ) : null}

                    <Flex gap="4" align="start">
                        {/* ---- 预设列表 ---- */}
                        <Flex
                            direction="column"
                            gap="2"
                            style={{ width: LIST_WIDTH, flexShrink: 0 }}
                        >
                            <ScrollArea
                                style={{ maxHeight: PANE_MAX_HEIGHT }}
                                scrollbars="vertical"
                                type="auto"
                            >
                                <Flex direction="column" gap="1" pr="2">
                                    <span className="hs-type-muted">
                                        {t("vibrato_manager_group_system")}
                                    </span>
                                    {resolved.system.map((preset) => (
                                        <PresetRow
                                            key={preset.id}
                                            preset={preset}
                                            selected={draft?.id === preset.id}
                                            active={session.activeVibratoPresetId === preset.id}
                                            onSelect={() => selectPreset(preset)}
                                            onActivate={() => activatePreset(preset)}
                                        />
                                    ))}

                                    <Box pt="2">
                                        <span className="hs-type-muted">
                                            {t("vibrato_manager_group_user")}
                                        </span>
                                    </Box>
                                    {resolved.user.length === 0 ? (
                                        <span className="hs-type-caption">
                                            {t("vibrato_manager_empty")}
                                        </span>
                                    ) : (
                                        resolved.user.map((preset, index) => (
                                            <Flex key={preset.id} align="center" gap="1">
                                                <Box style={{ minWidth: 0, flex: 1 }}>
                                                    <PresetRow
                                                        preset={preset}
                                                        selected={draft?.id === preset.id}
                                                        active={
                                                            session.activeVibratoPresetId ===
                                                            preset.id
                                                        }
                                                        onSelect={() => selectPreset(preset)}
                                                        onActivate={() => activatePreset(preset)}
                                                    />
                                                </Box>
                                                <AppButton
                                                    size="sm"
                                                    emphasis="soft"
                                                    disabled={index === 0}
                                                    aria-label={t("vibrato_manager_move_up")}
                                                    onClick={() => movePreset(preset, -1)}
                                                >
                                                    ▲
                                                </AppButton>
                                                <AppButton
                                                    size="sm"
                                                    emphasis="soft"
                                                    disabled={index === resolved.user.length - 1}
                                                    aria-label={t("vibrato_manager_move_down")}
                                                    onClick={() => movePreset(preset, 1)}
                                                >
                                                    ▼
                                                </AppButton>
                                            </Flex>
                                        ))
                                    )}
                                </Flex>
                            </ScrollArea>
                        </Flex>

                        {/* ---- 编辑器 ---- */}
                        <Box style={{ minWidth: 0, flex: 1 }}>
                            {draft && previewSamples ? (
                                <Flex direction="column" gap="3">
                                    <ScrollArea
                                        style={{ maxHeight: PANE_MAX_HEIGHT }}
                                        scrollbars="vertical"
                                        type="auto"
                                    >
                                        <Box pr="2">
                                            <AppForm>
                                                {!isBuiltin ? (
                                                    <AppField label={t("vibrato_manager_name")}>
                                                        <TextField.Root
                                                            size="2"
                                                            value={draft.name}
                                                            aria-label={t("vibrato_manager_name")}
                                                            onChange={(event) =>
                                                                patch({ name: event.target.value })
                                                            }
                                                        />
                                                    </AppField>
                                                ) : null}
                                                <AppFormSection title={t("vibrato_section_wave")}>
                                                    <AppField label={t("vibrato_shape_label")}>
                                                        <AppSelect
                                                            value={
                                                                draft.cycle.kind === "shape"
                                                                    ? draft.cycle.shape
                                                                    : "sine"
                                                            }
                                                            disabled={isBuiltin}
                                                            onValueChange={(value) =>
                                                                patch({
                                                                    cycle: {
                                                                        kind: "shape",
                                                                        shape: value as WaveShape,
                                                                        skew:
                                                                            draft.cycle.kind ===
                                                                            "shape"
                                                                                ? draft.cycle.skew
                                                                                : 0.5,
                                                                    },
                                                                })
                                                            }
                                                            options={WAVE_SHAPE_ORDER.map(
                                                                (shape) => ({
                                                                    value: shape,
                                                                    label: t(
                                                                        WAVE_SHAPE_KEYS[shape],
                                                                    ),
                                                                }),
                                                            )}
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_skew")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={2}
                                                                max={98}
                                                                disabled={
                                                                    isBuiltin ||
                                                                    draft.cycle.kind !== "shape" ||
                                                                    !shapeUsesSkew(
                                                                        draft.cycle.kind === "shape"
                                                                            ? draft.cycle.shape
                                                                            : "sine",
                                                                    )
                                                                }
                                                                value={
                                                                    draft.cycle.kind === "shape"
                                                                        ? Math.round(
                                                                              draft.cycle.skew *
                                                                                  100,
                                                                          )
                                                                        : 50
                                                                }
                                                                ariaLabel={t("vibrato_skew")}
                                                                onChange={(next) =>
                                                                    patch({
                                                                        cycle: {
                                                                            kind: "shape",
                                                                            shape:
                                                                                draft.cycle.kind ===
                                                                                "shape"
                                                                                    ? draft.cycle
                                                                                          .shape
                                                                                    : "sine",
                                                                            skew: next / 100,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(
                                                                    (draft.cycle.kind === "shape"
                                                                        ? draft.cycle.skew
                                                                        : 0.5) * 100,
                                                                )}%`}
                                                            </AppSliderReadout>
                                                        </Flex>
                                                    </AppField>
                                                </AppFormSection>

                                                <AppFormSection title={t("vibrato_section_depth")}>
                                                    <AppField label={t("vibrato_depth_label")}>
                                                        <AppNumberField
                                                            value={depthValue}
                                                            unit={depthUnit}
                                                            disabled={isBuiltin}
                                                            min={0}
                                                            ariaLabel={t("vibrato_depth_label")}
                                                            onChange={(next) =>
                                                                patch({
                                                                    depthCents: depthToCents(
                                                                        next,
                                                                        editParam,
                                                                        paramRange,
                                                                    ),
                                                                })
                                                            }
                                                            onCommit={() => undefined}
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_depth_ramp")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.depthRamp.start}
                                                                unit="percentFine"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                max={2}
                                                                suffix={t(
                                                                    "vibrato_depth_ramp_start",
                                                                )}
                                                                ariaLabel={`${t("vibrato_depth_ramp")} ${t("vibrato_depth_ramp_start")}`}
                                                                onCommit={(next) =>
                                                                    patch({
                                                                        depthRamp: {
                                                                            ...draft.depthRamp,
                                                                            start: next,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                            <AppNumberField
                                                                value={draft.depthRamp.end}
                                                                unit="percentFine"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                max={2}
                                                                suffix={t("vibrato_depth_ramp_end")}
                                                                ariaLabel={`${t("vibrato_depth_ramp")} ${t("vibrato_depth_ramp_end")}`}
                                                                onCommit={(next) =>
                                                                    patch({
                                                                        depthRamp: {
                                                                            ...draft.depthRamp,
                                                                            end: next,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_bias")}>
                                                        <AppNumberField
                                                            value={draft.biasCents}
                                                            unit="cents"
                                                            disabled={isBuiltin}
                                                            ariaLabel={t("vibrato_bias")}
                                                            onCommit={(next) =>
                                                                patch({ biasCents: next })
                                                            }
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_irregularity")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={0}
                                                                max={100}
                                                                disabled={isBuiltin}
                                                                value={Math.round(
                                                                    draft.irregularity,
                                                                )}
                                                                ariaLabel={t(
                                                                    "vibrato_irregularity",
                                                                )}
                                                                onChange={(next) =>
                                                                    patch({ irregularity: next })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(draft.irregularity)}%`}
                                                            </AppSliderReadout>
                                                        </Flex>
                                                    </AppField>
                                                </AppFormSection>

                                                <AppFormSection title={t("vibrato_section_rate")}>
                                                    <AppField label={t("vibrato_rate_mode")}>
                                                        {isBuiltin ? (
                                                            <span className="hs-type-label">
                                                                {t(RATE_MODE_KEYS[draft.rateMode])}
                                                            </span>
                                                        ) : (
                                                            <AppSegmentedControl<VibratoRateMode>
                                                                value={draft.rateMode}
                                                                ariaLabel={t("vibrato_rate_mode")}
                                                                onChange={(next) =>
                                                                    patch({ rateMode: next })
                                                                }
                                                                options={(
                                                                    ["hz", "cycles"] as const
                                                                ).map((mode) => ({
                                                                    value: mode,
                                                                    label: t(RATE_MODE_KEYS[mode]),
                                                                }))}
                                                            />
                                                        )}
                                                    </AppField>
                                                    {draft.rateMode === "hz" ? (
                                                        <AppField label={t("vibrato_rate_label")}>
                                                            <AppNumberField
                                                                value={draft.rateHz}
                                                                unit="vibratoHz"
                                                                disabled={isBuiltin}
                                                                min={0.1}
                                                                max={20}
                                                                ariaLabel={t("vibrato_rate_label")}
                                                                onCommit={(next) =>
                                                                    patch({ rateHz: next })
                                                                }
                                                            />
                                                        </AppField>
                                                    ) : (
                                                        <AppField label={t("vibrato_cycles")}>
                                                            <AppNumberField
                                                                value={draft.cycles}
                                                                unit="integer"
                                                                disabled={isBuiltin}
                                                                min={0.5}
                                                                max={128}
                                                                ariaLabel={t("vibrato_cycles")}
                                                                onCommit={(next) =>
                                                                    patch({ cycles: next })
                                                                }
                                                            />
                                                        </AppField>
                                                    )}
                                                    <AppField label={t("vibrato_rate_ramp")}>
                                                        <AppNumberField
                                                            value={draft.rateRampEnd}
                                                            unit="percentFine"
                                                            disabled={isBuiltin}
                                                            min={0.25}
                                                            max={4}
                                                            ariaLabel={t("vibrato_rate_ramp")}
                                                            onCommit={(next) =>
                                                                patch({ rateRampEnd: next })
                                                            }
                                                        />
                                                    </AppField>
                                                    <AppSwitchRow
                                                        label={t("vibrato_align_cycles")}
                                                        checked={draft.alignCycles}
                                                        disabled={isBuiltin}
                                                        onCheckedChange={(checked) =>
                                                            patch({ alignCycles: checked })
                                                        }
                                                    />
                                                </AppFormSection>

                                                <AppFormSection
                                                    title={t("vibrato_section_envelope")}
                                                >
                                                    <AppField label={t("vibrato_attack")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.attackMs}
                                                                unit="milliseconds"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                ariaLabel={t("vibrato_attack")}
                                                                onCommit={(next) =>
                                                                    patch({ attackMs: next })
                                                                }
                                                            />
                                                            <AppSelect
                                                                value={draft.attackCurve}
                                                                disabled={isBuiltin}
                                                                ariaLabel={t("vibrato_curve")}
                                                                onValueChange={(value) =>
                                                                    patch({
                                                                        attackCurve:
                                                                            value as EnvelopeCurve,
                                                                    })
                                                                }
                                                                options={ENVELOPE_CURVE_ORDER.map(
                                                                    (curve) => ({
                                                                        value: curve,
                                                                        label: t(
                                                                            ENVELOPE_CURVE_KEYS[
                                                                                curve
                                                                            ],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_release")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.releaseMs}
                                                                unit="milliseconds"
                                                                disabled={isBuiltin}
                                                                min={0}
                                                                ariaLabel={t("vibrato_release")}
                                                                onCommit={(next) =>
                                                                    patch({ releaseMs: next })
                                                                }
                                                            />
                                                            <AppSelect
                                                                value={draft.releaseCurve}
                                                                disabled={isBuiltin}
                                                                ariaLabel={t("vibrato_curve")}
                                                                onValueChange={(value) =>
                                                                    patch({
                                                                        releaseCurve:
                                                                            value as EnvelopeCurve,
                                                                    })
                                                                }
                                                                options={ENVELOPE_CURVE_ORDER.map(
                                                                    (curve) => ({
                                                                        value: curve,
                                                                        label: t(
                                                                            ENVELOPE_CURVE_KEYS[
                                                                                curve
                                                                            ],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_phase")}>
                                                        <AppNumberField
                                                            value={draft.startPhaseDeg}
                                                            unit="integer"
                                                            disabled={isBuiltin}
                                                            min={0}
                                                            max={360}
                                                            ariaLabel={t("vibrato_phase")}
                                                            onCommit={(next) =>
                                                                patch({ startPhaseDeg: next })
                                                            }
                                                        />
                                                    </AppField>
                                                </AppFormSection>

                                                <AppFormSection
                                                    title={t("vibrato_section_baseline")}
                                                >
                                                    <AppField label={t("vibrato_baseline")}>
                                                        <AppSelect
                                                            value={draft.baseline}
                                                            disabled={isBuiltin}
                                                            onValueChange={(value) =>
                                                                patch({
                                                                    baseline: value as BaselineMode,
                                                                })
                                                            }
                                                            options={BASELINE_MODE_ORDER.map(
                                                                (mode) => ({
                                                                    value: mode,
                                                                    label: t(
                                                                        BASELINE_MODE_KEYS[mode],
                                                                    ),
                                                                }),
                                                            )}
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_blend")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={0}
                                                                max={100}
                                                                disabled={
                                                                    isBuiltin ||
                                                                    draft.baseline !== "existing"
                                                                }
                                                                value={Math.round(draft.blend)}
                                                                ariaLabel={t("vibrato_blend")}
                                                                onChange={(next) =>
                                                                    patch({ blend: next })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(draft.blend)}%`}
                                                            </AppSliderReadout>
                                                        </Flex>
                                                    </AppField>
                                                </AppFormSection>
                                            </AppForm>
                                        </Box>
                                    </ScrollArea>
                                </Flex>
                            ) : (
                                <span className="hs-type-caption">
                                    {t("vibrato_manager_empty")}
                                </span>
                            )}
                        </Box>
                    </Flex>
                </Flex>
            </AppDialog>

            <AppConfirmDialog
                open={deleteTarget !== null}
                onOpenChange={(next) => {
                    if (!next) setDeleteTarget(null);
                }}
                title={t("vibrato_manager_delete")}
                message={t("vibrato_manager_delete_confirm").replace(
                    "{name}",
                    deleteTarget ? vibratoPresetLabel(deleteTarget, t) : "",
                )}
                confirmLabel={t("vibrato_manager_delete")}
                cancelLabel={t("cancel")}
                intent="danger"
                onConfirm={handleDelete}
            />
        </>
    );
}

interface PresetRowProps {
    preset: VibratoPreset;
    selected: boolean;
    active: boolean;
    onSelect: () => void;
    onActivate: () => void;
}

/**
 * 列表行：单击选中（编辑它），双击设为当前使用。
 *
 * 【为什么分开】"编辑某个预设"与"现在就用某个预设"是两件事：用户可能正在调
 * 一个还没调好的预设，却仍然希望拖拽用着上一个。合起来会让编辑动作顺带改掉
 * 当前音色。
 */
function PresetRow({ preset, selected, active, onSelect, onActivate }: PresetRowProps) {
    const { t } = useI18n();
    const description = vibratoPresetDescription(preset, t);
    return (
        <AppListRow
            selected={selected}
            density="compact"
            role="option"
            onClick={onSelect}
            onDoubleClick={onActivate}
            title={description ?? vibratoPresetSummary(preset, t)}
        >
            <Flex align="center" gap="1" style={{ minWidth: 0 }}>
                {active ? <span aria-hidden="true">●</span> : null}
                <span
                    className="hs-type-label"
                    style={{
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                    }}
                >
                    {vibratoPresetLabel(preset, t)}
                </span>
            </Flex>
        </AppListRow>
    );
}
