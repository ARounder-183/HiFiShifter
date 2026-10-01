/**
 * 「添加颤音」应用弹窗。
 *
 * 【为什么要有它】「添加颤音」曾经是右键菜单里的一个直接动作：点下去立刻按活动
 * 预设改写选区，用户没有"先看看会变成什么样"的机会。这里恢复"添加要有一次确认"
 * 的交互形态，内容换成预设驱动：左列选预设，右侧把**选区真实曲线**套上该预设
 * 画出来（原曲线弱化 + 结果强调），再配深度 / 速率两个快捷旋钮。
 *
 * 【与管理器的分工】管理器回答"这个预设长什么样"（改库），本弹窗回答"套到这段
 * 上长什么样"（用库）。两者不互相内嵌：把"保存到预设"和"应用"塞进同一个动作会
 * 让语义搅在一起 —— 因此"保存到预设"是一个默认关闭的显式勾选。
 *
 * 【草稿语义】旋钮改的是**本地副本**，选中另一个预设即重置；"应用"把完整预设
 * 交给编辑管线（而不是 presetId），本地微调才传得过去。
 */

import { useEffect, useMemo, useState } from "react";
import { Box, Flex, ScrollArea } from "@radix-ui/themes";

import { useAppDispatch } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import { persistUiSettings, upsertVibratoPreset } from "../../features/session/sessionSlice";
import {
    VIBRATO_LIMITS,
    isBuiltinVibratoPresetId,
    sanitizeVibratoPreset,
} from "../../features/vibrato/vibratoPresets";
import { depthStepUnitFor } from "../../features/vibrato/vibratoDepth";
import { planVibratoTarget } from "../../features/vibrato/vibratoPitch";
import type { VibratoPreset } from "../../features/vibrato/vibratoTypes";
import { AppDialog, AppField, AppListRow, AppNumberField, AppSwitchRow } from "../../ui";
import { VibratoPresetGlyph } from "./vibrato/VibratoPresetGlyph";
import { VibratoPreviewCanvas } from "./vibrato/VibratoPreviewCanvas";
import {
    buildAppliedPreview,
    depthForParam,
    depthToCents,
    formatNumber,
    vibratoPresetLabel,
    vibratoPresetSummary,
} from "./vibrato/vibratoDialogLogic";

/** 内容区定高：与预设管理器一致（R5 的定高 wrapper + flex 双栏，外层永不滚动）。 */
const CONTENT_HEIGHT = "min(60vh, 560px)";
/** 左列预设列表宽度（CSS 像素）。 */
const LIST_WIDTH = 220;

export interface VibratoApplyDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 可用预设（系统 + 用户），打开时预选 `activePresetId`。 */
    presets: readonly VibratoPreset[];
    activePresetId: string;
    editParam: string;
    paramRange?: { min: number; max: number };
    /**
     * 取选区（无选区时取曲线头部）的原始帧值。
     *
     * 由宿主提供：只有面板手里有 `rootTrackId` / `editParam` / 选区。返回 `null`
     * 表示取不到数据（例如刚打开还没有参数曲线）。
     */
    loadOriginal: () => Promise<{ values: number[]; framePeriodMs: number } | null>;
    /** 应用：把（可能已微调的）完整预设交给编辑管线。 */
    onApply: (preset: VibratoPreset) => void;
    /** 从选区提取预设（页脚 start 位，作用于当前选区，与本弹窗的预设选择无关）。 */
    onExtract?: () => void;
}

export function VibratoApplyDialog({
    open,
    onOpenChange,
    presets,
    activePresetId,
    editParam,
    paramRange,
    loadOriginal,
    onApply,
    onExtract,
}: VibratoApplyDialogProps) {
    const { t } = useI18n();
    const dispatch = useAppDispatch();

    // 草稿用惰性初始化：宿主每次打开都给本组件换一个 `key`（见面板），于是
    // "打开即预选当前活动预设、勾选复位"由**重新挂载**完成 —— 不需要在 effect
    // 里同步 setState（那会触发级联渲染，也被 lint 禁止）。
    const [draft, setDraft] = useState<VibratoPreset | null>(
        () => presets.find((preset) => preset.id === activePresetId) ?? presets[0] ?? null,
    );
    const [original, setOriginal] = useState<{ values: number[]; framePeriodMs: number } | null>(
        null,
    );
    const [saveToPreset, setSaveToPreset] = useState(false);
    const [loading, setLoading] = useState(true);

    // 只在打开时抓取预览数据；异步回调里的 setState 不受"effect 内同步 setState"
    // 约束。`loadOriginal` 变化（切换参数 / 轨道）时重取一次。
    useEffect(() => {
        if (!open) return;
        let cancelled = false;
        void loadOriginal().then((result) => {
            if (cancelled) return;
            setOriginal(result);
            setLoading(false);
        });
        return () => {
            cancelled = true;
        };
    }, [open, loadOriginal]);

    const previewSamples = useMemo(() => {
        if (!draft || !original) return null;
        return buildAppliedPreview({
            preset: draft,
            original: original.values,
            param: editParam,
            framePeriodMs: original.framePeriodMs,
            range: paramRange,
        });
    }, [draft, original, editParam, paramRange]);

    const isBuiltin = draft ? isBuiltinVibratoPresetId(draft.id) : false;
    const depthUnit = depthStepUnitFor(editParam);
    /**
     * 选区里**没有可调制的音符段**。
     *
     * 音高参数下这意味着无从调制：值为 0 的未检测帧、以及浊清边界上低而非零的过渡帧
     * （跟踪器给的 20~40 Hz 低估）都不是音符；连续有声帧还要够长（见
     * `vibratoPitch.ts`）。预览为空，应用也不会改动任何帧 —— 提交侧对每个选区段
     * 各自判定。这里只负责把"为什么没有预览"说清楚：把"没数据"和"这段没有音高"
     * 混成同一句提示，用户会以为是自己没选对区域。
     *
     * 【为什么不在这里禁用「应用」】预览只取**第一个**选区段；多选区时其余段可能
     * 有音符，按第一段禁用会把本来能应用的选区挡掉。
     */
    const unvoicedPitch =
        editParam === "pitch" &&
        original !== null &&
        planVibratoTarget(editParam, original.values, original.framePeriodMs) === null;

    const patch = (partial: Partial<VibratoPreset>) =>
        setDraft((current) => (current ? { ...current, ...partial } : current));

    const handleApply = () => {
        if (!draft) return;
        const normalized = sanitizeVibratoPreset(draft);
        if (saveToPreset && !isBuiltin) {
            dispatch(upsertVibratoPreset(normalized));
            void dispatch(persistUiSettings());
        }
        onApply(normalized);
        onOpenChange(false);
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("menu_add_vibrato")}
            size="xl"
            defaultActionId="apply"
            actions={[
                ...(onExtract
                    ? [
                          {
                              id: "extract",
                              label: t("vibrato_extract_action"),
                              align: "start" as const,
                              onClick: () => {
                                  onOpenChange(false);
                                  onExtract();
                              },
                          },
                      ]
                    : []),
                {
                    id: "cancel",
                    label: t("cancel"),
                    onClick: () => onOpenChange(false),
                },
                {
                    id: "apply",
                    label: t("vibrato_apply_apply"),
                    intent: "primary" as const,
                    disabled: !draft,
                    onClick: handleApply,
                },
            ]}
        >
            <Flex direction="column" gap="3" className="min-h-0" style={{ height: CONTENT_HEIGHT }}>
                <Flex gap="4" align="stretch" className="min-h-0 flex-1">
                    {/* ---- 预设列表（单选） ---- */}
                    <Flex
                        direction="column"
                        gap="2"
                        className="min-h-0"
                        style={{ width: LIST_WIDTH, flexShrink: 0 }}
                    >
                        <span className="hs-type-muted">{t("vibrato_apply_presets")}</span>
                        <ScrollArea
                            className="hs-scroll-area"
                            style={{ height: "100%" }}
                            scrollbars="vertical"
                            type="auto"
                        >
                            <Flex direction="column" gap="1" pr="2" role="listbox">
                                {presets.map((preset) => (
                                    <AppListRow
                                        key={preset.id}
                                        selected={draft?.id === preset.id}
                                        role="option"
                                        title={vibratoPresetSummary(preset, t)}
                                        onClick={() => setDraft(preset)}
                                    >
                                        <Flex align="center" gap="2" style={{ minWidth: 0 }}>
                                            <VibratoPresetGlyph
                                                preset={preset}
                                                width={40}
                                                height={14}
                                            />
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
                                ))}
                            </Flex>
                        </ScrollArea>
                    </Flex>

                    {/* ---- 套用预览 + 快捷旋钮 ---- */}
                    <Box className="min-h-0 flex flex-col" style={{ minWidth: 0, flex: 1 }}>
                        <ScrollArea
                            className="hs-scroll-area min-h-0 flex-1"
                            scrollbars="vertical"
                            type="auto"
                        >
                            <Flex direction="column" gap="3" pr="2">
                                <span className="hs-type-muted">{t("vibrato_apply_preview")}</span>
                                {previewSamples ? (
                                    <>
                                        <Box className="rounded border border-qt-border bg-qt-panel p-2">
                                            <VibratoPreviewCanvas
                                                samples={previewSamples}
                                                ariaLabel={t("vibrato_apply_preview")}
                                            />
                                            <Flex justify="between" align="center" mt="1">
                                                <span className="hs-type-caption">
                                                    {`±${formatNumber(previewSamples.peakCents)} ${t("vibrato_unit_cents")}`}
                                                </span>
                                                {isBuiltin ? (
                                                    <span className="hs-type-caption">
                                                        {t("vibrato_manager_readonly")}
                                                    </span>
                                                ) : null}
                                            </Flex>
                                        </Box>
                                    </>
                                ) : (
                                    <Box className="rounded border border-qt-border bg-qt-panel p-2">
                                        <span className="hs-type-caption">
                                            {loading
                                                ? t("common_loading")
                                                : unvoicedPitch
                                                  ? t("vibrato_apply_preview_no_pitch")
                                                  : t("vibrato_apply_preview_empty")}
                                        </span>
                                    </Box>
                                )}

                                {draft ? (
                                    <Flex gap="3" wrap="wrap">
                                        <AppField label={t("vibrato_depth_label")}>
                                            <AppNumberField
                                                value={depthForParam(
                                                    draft.depthCents,
                                                    editParam,
                                                    paramRange,
                                                )}
                                                unit={depthUnit}
                                                min={VIBRATO_LIMITS.depthCents.min}
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
                                        <AppField label={t("vibrato_rate_label")}>
                                            <AppNumberField
                                                value={draft.rateHz}
                                                unit="vibratoHz"
                                                min={0.1}
                                                max={20}
                                                ariaLabel={t("vibrato_rate_label")}
                                                onChange={(next) => patch({ rateHz: next })}
                                                onCommit={() => undefined}
                                            />
                                        </AppField>
                                    </Flex>
                                ) : null}

                                <AppSwitchRow
                                    control="checkbox"
                                    label={t("vibrato_apply_save_preset")}
                                    checked={saveToPreset}
                                    disabled={!draft || isBuiltin}
                                    onCheckedChange={setSaveToPreset}
                                />
                            </Flex>
                        </ScrollArea>
                    </Box>
                </Flex>
            </Flex>
        </AppDialog>
    );
}
