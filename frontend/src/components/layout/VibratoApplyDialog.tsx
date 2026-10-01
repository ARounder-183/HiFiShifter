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
    duplicateVibratoPreset,
    isBuiltinVibratoPresetId,
    nextDuplicatePresetName,
    sanitizeVibratoPreset,
} from "../../features/vibrato/vibratoPresets";
import { buildContourAuditionPair, vibratoAudition } from "../../features/vibrato/vibratoAudition";
import { depthStepUnitFor } from "../../features/vibrato/vibratoDepth";
import { planVibratoTarget } from "../../features/vibrato/vibratoPitch";
import type { VibratoPreset } from "../../features/vibrato/vibratoTypes";
import { AppButton, AppDialog, AppField, AppListRow, AppNumberField, AppSwitchRow } from "../../ui";
import { VibratoPresetGlyph } from "./vibrato/VibratoPresetGlyph";
import { VibratoPreviewCanvas } from "./vibrato/VibratoPreviewCanvas";
import {
    buildAppliedPreview,
    fitPreviewRangeCents,
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
    /**
     * 打开时**预选**哪个预设；省略则用当前活动预设。
     *
     * 从预设管理器返回时带上它 —— 用户刚才在管理器里点选的那一条（**不是**当前使用
     * 的那条）就是他要接着应用的。弹窗每次打开都换 key 重挂载，所以这个值只需在
     * 挂载那一刻对得上。
     */
    initialPresetId?: string;
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
    /**
     * 「编辑预设」：跳到预设管理器去改库（捏预设、改名字、导入导出）。
     *
     * 参数是**当前选中**的预设 id：管理器要用它继续编辑，而不是回到"当前使用"的那条。
     * 改完怎么回来由宿主安排 —— 管理器那边会出现「返回添加颤音」。跳转前先走本弹窗的
     * 关闭路径（停试听）。
     */
    onEditPresets?: (presetId: string) => void;
}

export function VibratoApplyDialog({
    open,
    onOpenChange,
    presets,
    activePresetId,
    initialPresetId,
    editParam,
    paramRange,
    loadOriginal,
    onApply,
    onExtract,
    onEditPresets,
}: VibratoApplyDialogProps) {
    const { t } = useI18n();
    const dispatch = useAppDispatch();

    // 草稿用惰性初始化：宿主每次打开都给本组件换一个 `key`（见面板），于是
    // "打开即预选当前活动预设、勾选复位"由**重新挂载**完成 —— 不需要在 effect
    // 里同步 setState（那会触发级联渲染，也被 lint 禁止）。
    const [draft, setDraft] = useState<VibratoPreset | null>(
        () =>
            presets.find((preset) => preset.id === (initialPresetId ?? activePresetId)) ??
            presets[0] ??
            null,
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
     * 正在试听的是哪一条（`null` = 没在响）。
     *
     * 用"哪一条"而不是布尔量：两个按钮各自要显示自己的播放 / 停止态，而试听器同一
     * 时刻只允许一路声音（新的会抢占旧的）。
     */
    const [audition, setAudition] = useState<"source" | "result" | null>(null);

    /**
     * 「适应」的令牌：每点一次自增，让下面的 `useMemo` 重算纵轴。
     *
     * 用令牌而不是"拟合函数 + state"，是为了把**什么时候重算**直接写在依赖数组里
     * （见下），读的人不必去追 effect 的触发条件。
     */
    const [refitToken, setRefitToken] = useState(0);

    /**
     * 预览纵轴的半幅（cents）。
     *
     * 【为什么必须固定住】与预设管理器同一套逻辑：标尺若跟着当前深度自适应，波形
     * 永远填满画布 —— 调深度时看到的只是整幅在竖直方向"抖一下"，读不出幅度大小。
     * 标尺固定下来，波形高度才等于深度，配合画布上的刻度标签可以直读。
     *
     * 【什么时候重算】依赖数组就是答案：**换选区**（`original`）、**换预设**
     * （`draft?.id`）、**点「适应」**（令牌）。深度 / 速率的改动刻意不在其中 ——
     * `previewSamples` 的内容会变，但那是"编辑期间"，标尺不动。
     */
    const previewHalfCents = useMemo(
        () => fitPreviewRangeCents(previewSamples ? previewSamples.peakCents : 0),
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 内容变化（深度 / 速率）刻意不重算标尺，见上
        [original, draft?.id, refitToken],
    );

    /** A/B 试听的两条曲线：同一中心、同一基音，差异只来自颤音。 */
    const auditionCurves = useMemo(() => {
        if (!previewSamples || !original) return null;
        return buildContourAuditionPair(
            previewSamples.contour,
            previewSamples.wave,
            original.framePeriodMs,
        );
    }, [previewSamples, original]);

    // 卸载兜底：窗口消失了声音不能继续响。这里不含 setState —— 按钮状态复位在
    // `handleOpenChange`（事件处理器）里做，不违反 effect 的规则（与预设管理器同款）。
    useEffect(() => () => vibratoAudition.stop(), []);
    // 父级把 `open` 置 false（不是经本组件关闭）时也要停掉声音。
    useEffect(() => {
        if (!open) vibratoAudition.stop();
    }, [open]);

    /** 关闭对话框：先停试听、复位按钮，再向上传播。 */
    function handleOpenChange(next: boolean) {
        if (!next) {
            vibratoAudition.stop();
            setAudition(null);
        }
        onOpenChange(next);
    }

    /** 播放 / 停止某一条试听。自然结束后引擎回调把按钮切回「播放」。 */
    function toggleAudition(kind: "source" | "result") {
        if (audition === kind || !auditionCurves) {
            vibratoAudition.stop();
            setAudition(null);
            return;
        }
        const started = vibratoAudition.play(auditionCurves[kind], () => setAudition(null));
        setAudition(started ? kind : null);
    }
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
        /*
         * 「保存到预设」对**系统预设**同样成立：系统预设本身只读，于是自动另存为一份
         * 自定义副本（与管理器里「复制为自定义」同一套命名规则：按显示名预填编号，
         * 并避开已占用的名字），并把这份副本作为应用对象 —— 用户存下的和听到的是同一个
         * 东西，而不是"存了一份、应用了另一份"。
         */
        const saved = saveToPreset
            ? isBuiltin
                ? duplicateVibratoPreset(
                      normalized,
                      nextDuplicatePresetName(
                          vibratoPresetLabel(draft, t) || t("vibrato_manager_new"),
                          presets.map((preset) => vibratoPresetLabel(preset, t)),
                      ),
                  )
                : normalized
            : null;
        if (saved) {
            dispatch(upsertVibratoPreset(saved));
            void dispatch(persistUiSettings());
        }
        onApply(saved ?? normalized);
        handleOpenChange(false);
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={handleOpenChange}
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
                                  handleOpenChange(false);
                                  onExtract();
                              },
                          },
                      ]
                    : []),
                ...(onEditPresets
                    ? [
                          {
                              id: "editPresets",
                              label: t("vibrato_apply_edit_presets"),
                              align: "start" as const,
                              onClick: () => {
                                  if (!draft) return;
                                  handleOpenChange(false);
                                  onEditPresets(draft.id);
                              },
                          },
                      ]
                    : []),
                {
                    id: "cancel",
                    label: t("cancel"),
                    onClick: () => handleOpenChange(false),
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
                                            {/*
                                             * 一张图里叠两条：背景虚线是原参数线轮廓
                                             * （按自身范围铺满，形状可见），前景是颤音偏移
                                             * 与其包络（按 cents 标尺）。两者差两个数量级，
                                             * 因此各用各的标尺 —— 见画布的说明。
                                             */}
                                            <VibratoPreviewCanvas
                                                samples={previewSamples}
                                                halfCents={previewHalfCents}
                                                ariaLabel={t("vibrato_apply_preview")}
                                            />
                                            <Flex
                                                justify="between"
                                                align="center"
                                                mt="1"
                                                gap="2"
                                                wrap="wrap"
                                            >
                                                {/* 纵轴读数：它属于前景那条颤音偏移曲线，
                                                    因此贴着左侧 —— 与刻度标签同侧。 */}
                                                <span className="hs-type-caption">
                                                    {`±${formatNumber(previewSamples.vibratoPeakCents)} ${t("vibrato_unit_cents")}`}
                                                </span>
                                                <Flex
                                                    align="center"
                                                    gap="2"
                                                    wrap="wrap"
                                                    style={{ minWidth: 0 }}
                                                >
                                                    {/* 重新拟合纵轴：与预设管理器同款按钮。 */}
                                                    <AppButton
                                                        size="sm"
                                                        emphasis="soft"
                                                        onClick={() => setRefitToken((t) => t + 1)}
                                                    >
                                                        {t("vibrato_preview_fit")}
                                                    </AppButton>
                                                    {/*
                                                     * 试听 A/B：原参数线与新参数线各一个按钮。
                                                     * 两条曲线共用同一个中心与基音，因此听感差异
                                                     * 只来自颤音本身 —— 这正是"对比"要的。
                                                     */}
                                                    {(
                                                        [
                                                            [
                                                                "source",
                                                                "vibrato_apply_audition_source",
                                                            ],
                                                            [
                                                                "result",
                                                                "vibrato_apply_audition_result",
                                                            ],
                                                        ] as const
                                                    ).map(([kind, labelKey]) => (
                                                        <AppButton
                                                            key={kind}
                                                            size="sm"
                                                            emphasis={
                                                                audition === kind ? "solid" : "soft"
                                                            }
                                                            aria-pressed={audition === kind}
                                                            /*
                                                             * 文案**恒定**，不随播放状态换成
                                                             * "停止试听"：那样按钮宽度会变，整行
                                                             * 跟着跳（用户报过）。播放态改由强调色
                                                             * + `aria-pressed` 表达，点击的后果写在
                                                             * `title` 里。
                                                             */
                                                            title={
                                                                audition === kind
                                                                    ? t("vibrato_audition_stop")
                                                                    : t(labelKey)
                                                            }
                                                            disabled={!auditionCurves}
                                                            onClick={() => toggleAudition(kind)}
                                                        >
                                                            {t(labelKey)}
                                                        </AppButton>
                                                    ))}
                                                </Flex>
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
                                    disabled={!draft}
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
