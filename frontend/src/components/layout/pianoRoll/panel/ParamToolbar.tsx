// hs-interaction-exempt: 边缘平滑度裸 range（`attachEdgeSmoothnessWheel` 的原生滚轮监听、松开落盘）与工具栏的紧凑行高耦合，按迁移范围刻意保留；本文件其余取值控件已走能力层原语。
/**
 * 参数编辑器工具栏。
 *
 * 【为什么单独成文件】它是面板里最长的一段视图（约 1240 行）：参数药丸、算法选择、
 * 参考轨道、导入入口、让位层级全在这里。此前它被埋在 9000 行面板的中段，
 * "工具栏上多一个按钮要改哪儿"每次都要靠搜索。
 *
 * 【它拿到的字段是显式的】`Pick<PianoRollPanelModel, …>` 由 AST 扫描得出 ——
 * 工具栏能碰到的东西一目了然，越界即类型错误。
 */
import { PitchSnapSettingsDialog } from "../../PitchSnapSettingsDialog";
import { Flex, Button, Box } from "@radix-ui/themes";
import { CursorArrowIcon, Link2Icon, LinkBreak2Icon } from "@radix-ui/react-icons";
import {
    setEdgeSmoothnessPercent,
    setTrackStateRemote,
    setScaleHighlightMode,
    toggleLockParamLines,
    setToolModePersistent,
    persistUiSettings,
    setParamEditorSyncTimeline,
} from "../../../../features/session/sessionSlice";
import { resolvePitchAlgoSelectValue } from "../../../../features/tracks/pitchAlgoOptions";
import { TimelineDisplaySettingsDialog } from "../../TimelineDisplaySettingsDialog";
import { wholeDevicePxLength } from "../../../../utils/devicePixelLine";
import { AppIconButton, AppSelect } from "../../../../ui";
import { ToolbarSlot } from "./toolbarParts";
import { ParamButtons } from "./ParamButtons";
import { DrawToolMenu } from "./DrawToolMenu";
import { PitchSnapMenu } from "./PitchSnapMenu";
import { DragDirectionButton } from "./DragDirectionButton";
import { ReferenceTrackGroup } from "./ReferenceTrackGroup";
import { ChildPitchGroup } from "./ChildPitchGroup";

import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

/**
 * 本文件用到的模型字段（AST 扫描得出）；`Pick<>` 保证它不会悄悄变宽。
 *
 * 其中 9 个只**透传**给 `<ParamButtons>`（不在这里解构）：它们是参数按钮那条链的
 * 输入，工具栏本身不读。
 */
type Props = {
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

export function ParamToolbar({ m }: Props) {
    const {
        attachEdgeSmoothnessWheel,
        devicePixelRatio,
        dispatch,
        getProcessorParamLabel,
        getStaticOptionLabel,
        handleStaticParamChange,
        pitchAlgoOptions,
        pitchSnapOpen,
        processorStaticParams,
        processorStaticValues,
        rootTrack,
        rootTrackId,
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
            <Flex align="center" gap="2" style={{ flex: "1 1 auto", minWidth: 0 }}>
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
                    <Flex align="center" gap="1" ml="2" style={{ minWidth: 0, flexShrink: 1 }}>
                        {/* 标签允许被压缩裁切（完整名称在悬停提示里）：横向极窄时
                            应当由它先让位，而不是把整行撑到溢出。省略号让"让位"
                            看起来是有意的降级，而不是渲染出错的半截字。
                            第 5 级隐藏：滑块与百分比数值仍在，全称在 ToolTip 里。 */}
                        <ToolbarSlot hideAt={5} tier={toolbarTier}>
                            <span
                                className="hs-type-label"
                                data-tooltip={tf("edge_smoothness")}
                                style={{
                                    minWidth: 0,
                                    whiteSpace: "nowrap",
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                }}
                            >
                                {tf("edge_smoothness_short")}:
                            </span>{" "}
                        </ToolbarSlot>
                        {/* 第 7 级才隐藏滑块：它是**控件**，比只读的百分比数值（第 6 级）
                            更该留到最后。悬停时用 ToolTip 报出当前百分比 —— 数值被隐藏
                            （第 6 级）之后仍能读到。 */}
                        <ToolbarSlot hideAt={7} tier={toolbarTier}>
                            <input
                                ref={attachEdgeSmoothnessWheel}
                                className="qt-range"
                                type="range"
                                min={0}
                                max={100}
                                step={1}
                                value={Math.round(s.edgeSmoothnessPercent)}
                                data-tooltip={`${tf("edge_smoothness")}: ${Math.round(
                                    s.edgeSmoothnessPercent,
                                )}%`}
                                onChange={(e) => {
                                    const next = Number(e.currentTarget.value);
                                    dispatch(setEdgeSmoothnessPercent(next));
                                }}
                                onPointerUp={() => {
                                    void dispatch(persistUiSettings());
                                }}
                                onKeyUp={() => {
                                    void dispatch(persistUiSettings());
                                }}
                                style={{
                                    // 【flex-basis 必须取 0，而不是 `auto`（= width: 120）】
                                    // 基准为 120 时，横向不足的收缩量会**按基准比例分摊**给
                                    // 同排所有项 —— 实测窄容器下它只缩到 102px，仍占着大头。
                                    // 基准为 0 时，它先让出空间、只取"别人用剩下的"，宽裕时再由
                                    // `maxWidth` 封顶在 120px。它是低频调节项，理应最先让步。
                                    //
                                    // 【下限取 16】基准为 0 的项在收缩阶段分摊到的是 0（0 × shrink
                                    // 恒为 0），因此**它的下限就是它实际能到的最小宽度**：下限多大，
                                    // 拥挤时它就让出多少。16 = 12px 滑块 + 两侧各 2px，是仍能拖动的
                                    // 最小值；再窄就只剩滑块本身、看不到轨道了。极窄时它本就该让位
                                    // （数值仍可读，滚轮调值照常可用）。
                                    flex: "1 1 0",
                                    // 非 flex 上下文（理论上不会发生）时的兜底宽度。
                                    width: 120,
                                    minWidth: 16,
                                    maxWidth: 120,
                                }}
                            />{" "}
                        </ToolbarSlot>
                        {/* 数值需要完整可读（"100%"），因此给它一个较小的固定下限，
                            但不再是 36px 那种"宁可溢出也不缩"的宽度。
                            第 6 级隐藏：它只是**读数**，比滑块（第 7 级）先让位。 */}
                        <ToolbarSlot hideAt={6} tier={toolbarTier}>
                            {/* data-toolbar-volatile：这个读数在拖滑块时每帧都变，
                                布局签名对它只记长度，否则每一帧都会触发一次测量。 */}
                            <span
                                className="hs-type-label"
                                data-toolbar-volatile
                                style={{ minWidth: 28, textAlign: "right" }}
                            >
                                {Math.round(s.edgeSmoothnessPercent)}%
                            </span>
                        </ToolbarSlot>
                    </Flex>
                </Flex>
            </Flex>

            {/* Pitch Snap 设置弹窗 */}
            <PitchSnapSettingsDialog open={pitchSnapOpen} onOpenChange={setPitchSnapOpen} />
            <TimelineDisplaySettingsDialog
                open={timeDisplaySettingsOpen}
                onOpenChange={setTimeDisplaySettingsOpen}
            />

            <Flex gap="2" align="center">
                <Flex gap="1" align="center">
                    <ReferenceTrackGroup m={m} />
                    <ChildPitchGroup m={m} />
                    {/* 由后端 processorParams 驱动的动态参数按钮（按算法排列后的顺序） */}
                    <ParamButtons m={m} />
                </Flex>

                {rootTrack ? (
                    <Flex align="center" gap="2">
                        {processorStaticParams.map((param) => {
                            if (param.kind.type !== "static_enum") return null;
                            const currentValue =
                                processorStaticValues[param.id] ?? param.kind.default_value;

                            // 气声分离（breath_enabled）**不在本段渲染**：它已改在
                            // 第一段（动态参数区）以纯图标分段渲染在被门禁组
                            // （气声 + 张力）的组首 —— 见上方 `orderedProcessorParams`
                            // 循环里的说明。留在本段（算法下拉之前）会让切换算法时
                            // 把音量 / 声像 / 算法整段推移，那正是本次要修的病根。
                            // 这里必须显式 `return null`：否则会落到下方的通用分支，
                            // 又渲染出一个带文本的按钮。
                            if (param.id === "breath_enabled") return null;

                            // vslib 的合成模式：改为支持滚轮切换的下拉栏。
                            if (param.id === "synth_mode") {
                                const currentString = String(currentValue);
                                const selectOptions = param.kind.options.map(([label, value]) => ({
                                    value,
                                    label: getStaticOptionLabel(param.id, label, value),
                                }));
                                const currentOptionLabel =
                                    selectOptions.find((opt) => String(opt.value) === currentString)
                                        ?.label ?? currentString;
                                return (
                                    <span
                                        key={param.id}
                                        data-tooltip={`${t("vslib_synth_mode_label")}: ${currentOptionLabel}`}
                                        className="inline-flex"
                                    >
                                        <AppSelect
                                            // 参数编辑器头部是 32px 的紧凑 chrome，邻居是 24px 图标按钮
                                            density="compact"
                                            value={currentString}
                                            onValueChange={(v) =>
                                                void handleStaticParamChange(param.id, Number(v))
                                            }
                                            fullWidth={false}
                                            // 与“算法”下拉栏一致使用固定宽度，选项切换时宽度不变
                                            className="w-[140px]"
                                            ariaLabel={t("vslib_synth_mode_label")}
                                            options={selectOptions.map((opt) => ({
                                                value: String(opt.value),
                                                label: opt.label,
                                            }))}
                                        />
                                    </span>
                                );
                            }

                            return (
                                <Flex key={param.id} align="center" gap="1">
                                    <span
                                        className="hs-type-label"
                                        data-tooltip={getProcessorParamLabel(param)}
                                    >
                                        {getProcessorParamLabel(param)}
                                    </span>
                                    {param.kind.options.map(([label, value]) => (
                                        <Button
                                            key={`${param.id}-${value}`}
                                            size="1"
                                            variant={currentValue === value ? "solid" : "soft"}
                                            color={currentValue === value ? "blue" : "gray"}
                                            onClick={() => {
                                                void handleStaticParamChange(param.id, value);
                                            }}
                                            style={{
                                                cursor: "pointer",
                                            }}
                                        >
                                            {getStaticOptionLabel(param.id, label, value)}
                                        </Button>
                                    ))}
                                </Flex>
                            );
                        })}
                        {/* 第 2 级隐藏：紧邻的下拉本身就显示当前算法名，
                            "算法"二字只是重复提示（全称在 ToolTip 里）。 */}
                        <ToolbarSlot hideAt={2} tier={toolbarTier}>
                            <span className="hs-type-label" data-tooltip={tf("algo_label")}>
                                {tf("algo_label_short")}
                            </span>{" "}
                        </ToolbarSlot>
                        <AppSelect
                            // 同上：头部紧凑条内的控件
                            density="compact"
                            value={resolvePitchAlgoSelectValue(
                                rootTrack.pitchAnalysisAlgo,
                                pitchAlgoOptions,
                            )}
                            onValueChange={(v) => {
                                if (!rootTrackId) return;
                                dispatch(
                                    setTrackStateRemote({
                                        trackId: rootTrackId,
                                        pitchAnalysisAlgo: v,
                                    }),
                                );
                            }}
                            fullWidth={false}
                            className="min-w-[140px]"
                            ariaLabel={tf("algo_label")}
                            options={pitchAlgoOptions}
                        />
                    </Flex>
                ) : null}
            </Flex>
        </Flex>
    );
}
