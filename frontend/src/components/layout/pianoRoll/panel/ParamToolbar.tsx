// hs-interaction-exempt: 边缘平滑度裸 range（`attachEdgeSmoothnessWheel` 的原生滚轮监听、松开落盘）与工具栏的紧凑行高耦合，按迁移范围刻意保留；本文件其余取值控件已走能力层原语。
/**
 * 参数编辑器工具栏。
 *
 * 【为什么单独成文件】它曾经是面板里最长的一段视图（近 1400 行）：参数药丸、算法选择、
 * 参考轨道、导入入口、让位层级全在这里。它最初埋在近万行面板文件的中段，
 * "工具栏上多一个按钮要改哪儿"每次都要靠搜索。
 *
 * 【它拿到的字段是显式的】`Pick<PianoRollPanelModel, …>` 由 AST 扫描得出 ——
 * 工具栏能碰到的东西一目了然，越界即类型错误。
 */
import { PitchSnapSettingsDialog } from "../../PitchSnapSettingsDialog";
import { Flex, Box } from "@radix-ui/themes";
import { CursorArrowIcon, Link2Icon, LinkBreak2Icon } from "@radix-ui/react-icons";
import {
    setScaleHighlightMode,
    toggleLockParamLines,
    setToolModePersistent,
    persistUiSettings,
    setParamEditorSyncTimeline,
} from "../../../../features/session/sessionSlice";
import { TimelineDisplaySettingsDialog } from "../../TimelineDisplaySettingsDialog";
import { wholeDevicePxLength } from "../../../../utils/devicePixelLine";
import { AppIconButton } from "../../../../ui";
import { ToolbarSlot } from "./toolbarParts";
import { ParamButtons } from "./ParamButtons";
import { DrawToolMenu } from "./DrawToolMenu";
import { PitchSnapMenu } from "./PitchSnapMenu";
import { DragDirectionButton } from "./DragDirectionButton";
import { ReferenceTrackGroup } from "./ReferenceTrackGroup";
import { ChildPitchGroup } from "./ChildPitchGroup";
import { SmoothingSlider } from "./SmoothingSlider";
import { ParamStaticControls } from "./ParamStaticControls";
import { ToolbarOverflowMenu } from "./ToolbarOverflowMenu";
import { TIER_ALL_PILLS, TIER_STATIC_PARAMS } from "../toolbarOverflow";

import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

/**
 * 本文件用到的模型字段（AST 扫描得出）；`Pick<>` 保证它不会悄悄变宽。
 *
 * 其中 9 个只**透传**给 `<ParamButtons>`（不在这里解构）：它们是参数按钮那条链的
 * 输入，工具栏本身不读。
 */
export type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "activeDragDirection"
        | "activeDragDirectionTool"
        | "activeVibratoPreset"
        | "activeVibratoPresetId"
        | "attachEdgeSmoothnessWheel"
        | "childFormantOffsetParam"
        | "childPitchOffsetCentsParam"
        | "childPitchOffsetDegreesParam"
        | "composeEnabled"
        | "currentDrawTool"
        | "currentDrawToolIcon"
        | "cycleDragDirectionKb"
        | "devicePixelRatio"
        | "dispatch"
        | "drawToolButtonTitle"
        | "drawToolKb"
        | "drawToolMenuAnchorRef"
        | "drawToolMenuOpen"
        | "drawToolMenuRef"
        | "editParam"
        | "effectivePitchSnapVisual"
        | "enabledVibratoPresetList"
        | "gatedPillHideOrder"
        | "gatedPillsHidden"
        | "getProcessorParamLabel"
        | "getProcessorParamShortLabel"
        | "getStaticOptionLabel"
        | "handleOpenMidiDialog"
        | "handleStaticParamChange"
        | "lineToolIcon"
        | "lineToolKb"
        | "openVibratoDialog"
        | "orderedProcessorParams"
        | "pitchAlgoOptions"
        | "pitchEnabled"
        | "pitchGroupActive"
        | "pitchGroupLabel"
        | "pitchGroupTooltip"
        | "pitchHardDisableReason"
        | "pitchSnapMenuAnchorRef"
        | "pitchSnapMenuOpen"
        | "pitchSnapMenuRef"
        | "pitchSnapOpen"
        | "pitchSnapScaleIcon"
        | "pitchSnapSemitoneIcon"
        | "presetFlyoutMaxHeight"
        | "processorParamsRef"
        | "processorStaticParams"
        | "processorStaticValues"
        | "referenceRootTrackOptions"
        | "rootTrack"
        | "rootTrackId"
        | "s"
        | "secondaryParamVisible"
        | "selectedIsChildTrack"
        | "separationEnabled"
        | "separationSwitchAnchorParamId"
        | "setDrawToolMenuOpen"
        | "setHoveredReferenceRootTrackId"
        | "setPitchSnapMenuOpen"
        | "setPitchSnapOpen"
        | "setTimeDisplaySettingsOpen"
        | "t"
        | "tf"
        | "timeDisplaySettingsOpen"
        | "toggleSecondaryParam"
        | "toolbarRowRef"
        | "toolbarTier"
        | "updateVisibleReferenceRootTrackIds"
        | "vibratoToolIcon"
        | "vibratoToolKb"
        | "visibleReferenceRootTrackIds"
    >;
};

/** 工具栏能拿到的模型字段集合（子组件与「⋯」菜单共用这一份）。 */
export type ParamToolbarModel = Props["m"];

export function ParamToolbar({ m }: Props) {
    const {
        devicePixelRatio,
        dispatch,
        pitchSnapOpen,
        s,
        setPitchSnapOpen,
        setTimeDisplaySettingsOpen,
        t,
        tf,
        timeDisplaySettingsOpen,
        toolbarRowRef,
        toolbarTier,
    } = m;

    return (
        <Flex
            align="center"
            justify="between"
            className="h-qt-bar-main bg-qt-base border-b border-qt-border px-2 shrink-0 overflow-hidden whitespace-nowrap"
            ref={toolbarRowRef}
            /* 当前隐藏层级（0 = 全部显示）：观测与调试出口，见 toolbarOverflow.ts。 */
            data-toolbar-tier={toolbarTier}
        >
            <Flex
                align="center"
                gap="2"
                style={{ flex: "1 1 auto", minWidth: 0, overflow: "hidden" }}
            >
                <AppIconButton
                    active={s.paramEditorSyncTimeline}
                    // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                    emphasis="accent"
                    tooltip={tf("sync_timeline_view_tooltip")}
                    aria-label={tf("sync_timeline_view")}
                    tabIndex={-1}
                    onClick={() => {
                        dispatch(setParamEditorSyncTimeline(!s.paramEditorSyncTimeline));
                        void dispatch(persistUiSettings());
                    }}
                    icon={s.paramEditorSyncTimeline ? <Link2Icon /> : <LinkBreak2Icon />}
                />
                {/* 第 1 级隐藏：面板身份已由 dock 标签 / 浮窗标题表达，
                    这里是重复提示，横向不足时最先让位。 */}
                <ToolbarSlot hideAt={1} tier={toolbarTier}>
                    <span className="hs-type-label font-semibold">
                        {tf("param_editor_short")}
                    </span>{" "}
                </ToolbarSlot>
                {/* 工具按钮组（音高吸附等）+ 平滑度滑块。`marginLeft: 8` 是紧邻
                    `参数编辑器` 标题留出的空白。
                    【minWidth 必须显式置 0】flex 项默认 `min-width: auto`（= min-content），
                    这一组的 min-content 里含**滑块的 max-content（120px）**与标签全文，
                    实测 384px —— 面板比它窄时整组拒绝收缩、直接溢出，于是组内的滑块永远
                    停在 120px（用户报告"平滑度滑块无法缩小"）。置 0 后收缩按 base 分摊：
                    按钮（min-content = 24px）保持不动，滑块与标签先让位。 */}
                <Flex gap="1" align="center" style={{ marginLeft: 8, minWidth: 0 }}>
                    <AppIconButton
                        active={s.toolModeGroup === "select"}
                        // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                        emphasis="accent"
                        tooltip={t("common_select")}
                        tabIndex={-1}
                        onClick={() => void dispatch(setToolModePersistent("select"))}
                        icon={<CursorArrowIcon />}
                    />
                    {/* 锚点外壳：菜单本身挂到 `document.body`（见 AppAnchoredMenu），
                        这里只负责提供"菜单该贴在哪"的矩形。 */}
                    <DrawToolMenu m={m} />
                    <Box
                        style={{
                            // 竖分隔线：宽度取整数个物理像素，任意缩放下粗细恒定
                            // （1 CSS px 在 dpr=1.25 下是 1.25 物理像素，边缘必发虚）。
                            width: wholeDevicePxLength(1, devicePixelRatio),
                            height: 18,
                            background: "var(--gray-8)",
                            marginInline: 4,
                            opacity: 0.9,
                        }}
                    />
                    <DragDirectionButton m={m} />

                    <PitchSnapMenu m={m} />
                    <AppIconButton
                        active={s.scaleHighlightMode === "always"}
                        // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                        emphasis="accent"
                        tooltip={tf("scale_highlight")}
                        tabIndex={-1}
                        onClick={() => {
                            dispatch(
                                setScaleHighlightMode(
                                    s.scaleHighlightMode === "always" ? "off" : "always",
                                ),
                            );
                            void dispatch(persistUiSettings());
                        }}
                        icon={
                            s.scaleHighlightMode === "always" ? (
                                <svg
                                    width="14"
                                    height="14"
                                    viewBox="0 0 14 14"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <circle cx="5" cy="9" r="2.2" fill="currentColor" />
                                    <path
                                        d="M7 4V8.5"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                    />
                                    <path
                                        d="M7 4L11 3.2"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        strokeLinecap="round"
                                    />
                                </svg>
                            ) : (
                                <svg
                                    width="14"
                                    height="14"
                                    viewBox="0 0 14 14"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <circle
                                        cx="5"
                                        cy="9"
                                        r="2.2"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        fill="none"
                                    />
                                    <path
                                        d="M7 4V8.5"
                                        stroke="currentColor"
                                        strokeWidth="1.2"
                                        strokeLinecap="round"
                                    />
                                    <path
                                        d="M7 4L11 3.2"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        strokeLinecap="round"
                                    />
                                </svg>
                            )
                        }
                    />
                    <AppIconButton
                        active={s.lockParamLinesEnabled}
                        // 激活时用主题强调色（旧写法不带 color，Radix 回落强调色）
                        emphasis="accent"
                        tooltip={t("lock_param_lines")}
                        tabIndex={-1}
                        onClick={() => {
                            dispatch(toggleLockParamLines());
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
                                <rect
                                    x="3"
                                    y="6"
                                    width="9"
                                    height="7"
                                    rx="1"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    fill="none"
                                />
                                <path
                                    d="M5 6V4.5C5 3.12 6.12 2 7.5 2C8.88 2 10 3.12 10 4.5V6"
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    fill="none"
                                />
                            </svg>
                        }
                    />
                    <SmoothingSlider m={m} />
                </Flex>
            </Flex>

            {/* Pitch Snap 设置弹窗 */}
            <PitchSnapSettingsDialog open={pitchSnapOpen} onOpenChange={setPitchSnapOpen} />
            <TimelineDisplaySettingsDialog
                open={timeDisplaySettingsOpen}
                onOpenChange={setTimeDisplaySettingsOpen}
            />

            <Flex gap="2" align="center">
                {/* 第 11 级：药丸整排搬进「⋯」。它们是唯一入口，所以是"搬走"不是"藏掉"。 */}
                <ToolbarSlot hideAt={TIER_ALL_PILLS} tier={toolbarTier}>
                    <Flex gap="1" align="center">
                        <ReferenceTrackGroup m={m} />
                        <ChildPitchGroup m={m} />
                        {/* 由后端 processorParams 驱动的动态参数按钮（按算法排列后的顺序） */}
                        <ParamButtons m={m} />
                    </Flex>
                </ToolbarSlot>

                {/* 第 10 级：静态枚举参数与算法下拉搬进「⋯」。 */}
                <ToolbarSlot hideAt={TIER_STATIC_PARAMS} tier={toolbarTier}>
                    <ParamStaticControls m={m} />
                </ToolbarSlot>

                {/* 只有真的收走了东西才出现（见组件里的层级判据）。 */}
                <ToolbarOverflowMenu m={m} />
            </Flex>
        </Flex>
    );
}
