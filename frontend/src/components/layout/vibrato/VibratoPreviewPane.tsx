/**
 * 颤音窗口的预览区：页签 + 画布 + 读数 + 试听。
 *
 * 【为什么单独一个组件】它同时服务两件事（"这个预设长什么样"与"套到选区上长
 * 什么样"），两块的采样、标尺、试听按钮都不一样，但**画布是同一块** —— 页签切换
 * 时元素位置不变，React 按同型同位置复用，画布不重挂载、不跳高。把这段留在窗口
 * 里会让那个 2000 行的组件再长 200 行，且页签逻辑与预设库逻辑完全无关。
 *
 * 【为什么不用库、不用 store】它只吃"算好的采样 + 回调"：采样由窗口用
 * `buildVibratoPreview` / `buildAppliedPreview` 算（两条路径的口径差别留在那边，
 * 与"应用"落盘同源），这里只负责画与转发手势。于是它可以单独渲染、单独测。
 *
 * 【两把标尺】两个页签各带一个 `halfCents`：预设波形只有几十分，套用预览要容下
 * 素材轮廓，量纲差两个数量级。各自固定（编辑期间不重算）才能让"高度 = 幅度"成立 ——
 * 具体规则见窗口里那两处拟合点。
 */

import { Box, Flex } from "@radix-ui/themes";
import { PlayIcon, StopIcon } from "@radix-ui/react-icons";

import { useI18n } from "../../../i18n/I18nProvider";
import { AppButton, AppIconButton, AppSegmentedControl, AppSelect } from "../../../ui";
import type { BaselineMode } from "../../../features/vibrato/vibratoTypes";
import { VibratoPreviewCanvas } from "./VibratoPreviewCanvas";
import type { PreviewHandleLayout, PreviewZone } from "./vibratoPreviewGestures";
import type { VibratoPreviewGestureInfo, VibratoPreviewInputState } from "./VibratoPreviewCanvas";
import type { VibratoAppliedPreview, VibratoPreviewSamples } from "./vibratoDialogLogic";
import { BASELINE_MODE_KEYS, BASELINE_MODE_ORDER, formatNumber } from "./vibratoDialogLogic";

/** 预览区的两个页签。 */
export type VibratoPreviewTab = "preset" | "applied";

/** 正在试听的是哪一条（`null` = 没在响）。 */
export type VibratoAuditionKind = "preset" | "source" | "result";

/** 套用页签画不出东西时的原因（决定占位文案）。 */
export type VibratoAppliedStatus = "loading" | "unvoiced" | "empty";

export interface VibratoPreviewPaneProps {
    tab: VibratoPreviewTab;
    onTabChange: (tab: VibratoPreviewTab) => void;
    /**
     * 有没有「套用到选区」这一页（= 宿主给了选区数据入口）。
     *
     * 为 `false` 时**整条段控都不渲染**：一个选项的段控是噪音，而且菜单栏那条路径
     * 背后没有选区，画一个点了没反应的页签只会让人以为坏了。
     */
    hasSelection: boolean;

    /** 预设波形页签：采样、标尺、手柄（可拖）。 */
    presetSamples: VibratoPreviewSamples;
    presetHalfCents: number;
    presetHandles?: PreviewHandleLayout;
    /** 周期估算读数（只对预设页签有意义：它的窗口时长是固定的）。 */
    cyclesEstimate: number;

    /** 套用页签：采样（`null` = 画不出来，看 `appliedStatus`）、标尺、手柄、占位原因。 */
    appliedSamples: VibratoAppliedPreview | null;
    appliedHalfCents: number;
    /**
     * 套用页签的手柄 —— 与 `presetHandles` **不是同一对**：手柄位置是"占该页签整段
     * 时长的比例"，而套用页签的时间轴是选区真实帧数（见窗口里的 `appliedWindowMs`）。
     */
    appliedHandles?: PreviewHandleLayout;
    appliedStatus: VibratoAppliedStatus;

    /**
     * 「摆放方式」（颤音围绕哪条曲线摆）。
     *
     * 【为什么在预览里也放一个】它决定的是"颤音挂在素材的哪条线上"，对「添加颤音」
     * 这个动作来说是要边看边定的参数 —— 放在波形右上角就地可改，用户不必去右下角的
     * 表单里翻（那里的同一个字段仍然保留：两处改的是同一个草稿）。
     *
     * 省略 `onBaselineChange` 时不渲染这个下拉。
     */
    baseline?: BaselineMode;
    onBaselineChange?: (mode: BaselineMode) => void;

    /*
     * 手势：两个页签共用同一套（命中与换算规则见 `vibratoPreviewGestures`）。
     * 拖的是同一个草稿 —— 在套用预览里拖渐入，与在预设波形里拖是同一件事，
     * 只是画布上的时间轴不同。
     */
    onGestureStart?: (zone: PreviewZone, info: VibratoPreviewGestureInfo) => void;
    onGestureMove?: (deltaX: number, deltaY: number, modifiers: VibratoPreviewInputState) => void;
    onGestureEnd?: () => void;
    /** 触控板捏合调深度（`deltaCents` 为正 = 加深）；见 `VibratoPreviewCanvas`。 */
    onPinchDepth?: (deltaCents: number) => void;

    /** 纵轴重新拟合（两个页签各自的那把）。 */
    onFit: () => void;

    /** 正在试听的那一条；`onAudition` 收到同一条表示"再点一次 = 停"。 */
    audition: VibratoAuditionKind | null;
    onAudition: (kind: VibratoAuditionKind) => void;
    /** 套用页签的两个试听按钮是否可用（取不到原参数线时为否）。 */
    appliedAuditionDisabled: boolean;
}

export function VibratoPreviewPane({
    tab,
    onTabChange,
    hasSelection,
    presetSamples,
    presetHalfCents,
    presetHandles,
    cyclesEstimate,
    appliedSamples,
    appliedHalfCents,
    appliedHandles,
    appliedStatus,
    baseline,
    onBaselineChange,
    onGestureStart,
    onGestureMove,
    onGestureEnd,
    onPinchDepth,
    onFit,
    audition,
    onAudition,
    appliedAuditionDisabled,
}: VibratoPreviewPaneProps) {
    const { t } = useI18n();
    /** 当前是不是在画"套用"那一页（且确实有这一页）。 */
    const onAppliedTab = tab === "applied" && hasSelection;
    /** 画不出东西时画占位文案 —— 两种情况要分开说（见窗口里的 `unvoicedPitch`）。 */
    const appliedPlaceholder =
        appliedStatus === "loading"
            ? t("common_loading")
            : appliedStatus === "unvoiced"
              ? t("vibrato_apply_preview_no_pitch")
              : t("vibrato_apply_preview_empty");

    return (
        <Box className="rounded border border-qt-border bg-qt-panel p-2">
            {/*
             * 头部一行：左边是页签段控（"看哪个问题"），右上角是**摆放方式**。
             *
             * 两个页签 = 这扇窗能回答的两个问题。段控而不是页签条：只有两个选项、
             * 且切换是"看的角度"而非"换一屏内容"，段控更轻。
             *
             * 摆放方式只出现在套用页：它决定"颤音挂在素材的哪条线上"，只有看着真实
             * 素材才谈得上选它；预设波形页没有素材，那个字段仍在右下角表单里。
             */}
            {hasSelection ? (
                <Flex mb="2" justify="between" align="center" gap="2" wrap="wrap">
                    <AppSegmentedControl<VibratoPreviewTab>
                        value={tab}
                        size="sm"
                        ariaLabel={t("vibrato_preview_tabs")}
                        onChange={onTabChange}
                        options={[
                            { value: "preset", label: t("vibrato_preview_tab_preset") },
                            { value: "applied", label: t("vibrato_apply_preview") },
                        ]}
                    />
                    {onAppliedTab && baseline !== undefined && onBaselineChange ? (
                        // `flexShrink: 0` + `fullWidth={false}`：这一组必须**按内容宽**
                        // 待着。`AppSelect` 默认铺满容器（表单里是对的），在这里会把
                        // 同一行的标题挤成一字一行 —— 而且它自己会撑到整行宽，看着像是
                        // 整个头部都是这个下拉。
                        <Flex align="center" gap="2" style={{ flexShrink: 0 }}>
                            <span className="hs-type-caption">{t("vibrato_baseline")}</span>
                            <AppSelect
                                value={baseline}
                                fullWidth={false}
                                // 定宽：选项文案长短差得远（"起点 → 终点" vs
                                // "保持现有曲线"），不定宽时切一下整行就跳。
                                minWidth={150}
                                ariaLabel={t("vibrato_baseline")}
                                onValueChange={(value) => onBaselineChange(value as BaselineMode)}
                                options={BASELINE_MODE_ORDER.map((mode) => ({
                                    value: mode,
                                    label: t(BASELINE_MODE_KEYS[mode]),
                                }))}
                            />
                        </Flex>
                    ) : null}
                </Flex>
            ) : null}

            {/*
             * 同一位置渲染同一个组件：切页签只换 `samples` / `halfCents` / `handles`，
             * 画布本身不重挂载（尺寸不跳、动画不闪）。
             *
             * 画不出来时（没取到数据 / 这段没有音高）连**读数行一起省掉**：那里报的是
             * "这条曲线摆多少"，此刻根本没有曲线 —— 留着它只会把预设波形的峰值冒充成
             * 选区的幅度。
             */}
            {onAppliedTab && !appliedSamples ? (
                <span className="hs-type-caption">{appliedPlaceholder}</span>
            ) : (
                <>
                    <VibratoPreviewCanvas
                        samples={onAppliedTab ? appliedSamples! : presetSamples}
                        ariaLabel={onAppliedTab ? t("vibrato_apply_preview") : t("vibrato_preview")}
                        halfCents={onAppliedTab ? appliedHalfCents : presetHalfCents}
                        handles={onAppliedTab ? appliedHandles : presetHandles}
                        onGestureStart={onGestureStart}
                        onGestureMove={onGestureMove}
                        onGestureEnd={onGestureEnd}
                        onPinchDepth={onPinchDepth}
                    />

                    <Flex justify="between" align="center" mt="1" gap="2" wrap="wrap">
                        <Flex gap="2" align="center" style={{ minWidth: 0 }} wrap="wrap">
                            {/*
                             * 读数：预设页签报"画出来的全部"的峰值；套用页签报**颤音自身**
                             * 的幅度（`vibratoPeakCents`）—— 后者画出来的峰值被素材轮廓
                             * 撑大，拿它当读数会把 40 分的颤音报成几千分。
                             */}
                            <span className="hs-type-caption">
                                {`±${formatNumber(
                                    onAppliedTab && appliedSamples
                                        ? appliedSamples.vibratoPeakCents
                                        : presetSamples.peakCents,
                                )} ${t("vibrato_unit_cents")}`}
                            </span>
                            {onAppliedTab ? null : (
                                <span className="hs-type-caption">
                                    {t("vibrato_cycles_estimate").replace(
                                        "{count}",
                                        formatNumber(cyclesEstimate),
                                    )}
                                </span>
                            )}
                        </Flex>
                        <Flex gap="1" align="center" wrap="wrap">
                            {/* 适应：把纵轴重新拟合到**当前页签**的内容。标尺在编辑期间刻意
                                保持不动（这样高度才等于深度），拖到超出量程或想重新看清形状时点它。 */}
                            <AppButton size="sm" emphasis="soft" onClick={onFit}>
                                {t("vibrato_preview_fit")}
                            </AppButton>
                            {onAppliedTab ? (
                                /*
                                 * A/B 试听：原参数线 / 新参数线各一个按钮。两条曲线共用同一个
                                 * 中心与基音，因此听感差异只来自颤音本身 —— 这正是"对比"要的。
                                 *
                                 * 文案**恒定**，不随播放状态换成"停止试听"：那样按钮宽度会变，
                                 * 整行跟着跳（用户报过）。播放态由强调色 + `aria-pressed` 表达，
                                 * 点击的后果写在 `title` 里。
                                 */
                                (["source", "result"] as const).map((kind) => (
                                    <AppButton
                                        key={kind}
                                        size="sm"
                                        emphasis={audition === kind ? "solid" : "soft"}
                                        aria-pressed={audition === kind}
                                        title={
                                            audition === kind
                                                ? t("vibrato_audition_stop")
                                                : t(
                                                      kind === "source"
                                                          ? "vibrato_apply_audition_source"
                                                          : "vibrato_apply_audition_result",
                                                  )
                                        }
                                        disabled={appliedAuditionDisabled}
                                        onClick={() => onAudition(kind)}
                                    >
                                        {t(
                                            kind === "source"
                                                ? "vibrato_apply_audition_source"
                                                : "vibrato_apply_audition_result",
                                        )}
                                    </AppButton>
                                ))
                            ) : (
                                /* 预设页签只有一个"这条预设听起来什么样"，一个图标按钮足够。 */
                                <AppIconButton
                                    tooltip={
                                        audition === "preset"
                                            ? t("vibrato_audition_stop")
                                            : t("vibrato_audition_play")
                                    }
                                    size="sm"
                                    aria-pressed={audition === "preset"}
                                    onClick={() => onAudition("preset")}
                                    icon={
                                        audition === "preset" ? (
                                            <StopIcon width="15" height="15" />
                                        ) : (
                                            <PlayIcon width="15" height="15" />
                                        )
                                    }
                                />
                            )}
                        </Flex>
                    </Flex>
                </>
            )}
        </Box>
    );
}
