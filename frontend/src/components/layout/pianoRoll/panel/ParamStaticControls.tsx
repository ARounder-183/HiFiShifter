/**
 * 工具栏尾部：**静态枚举参数**（含 vslib 合成模式下拉）与**算法下拉**。
 *
 * 这些参数没有曲线可画（后端标成 `static_enum`），它们的值只有"选哪一个"——
 * 因此渲染成按钮组或下拉，写回走 `handleStaticParamChange`（fire-and-forget）。
 *
 * 【气声分离为什么不在这里】`breath_enabled` 以纯图标分段渲染在**动态参数区**
 * 被门禁组（气声 + 张力）的组首，因为它是那组的开关。若留在本段（算法下拉之前），
 * 切换算法时会把音量 / 声像 / 算法整段推移——那正是之前的病根。这里因此显式
 * `return null`，避免落到下面的通用分支又画出一个带文本的按钮。
 *
 * 【算法下拉为什么跟静态参数放一起】两者都是"当前算法的配置"，且都要在轨道
 * 存在时才有意义（`rootTrack` 为空时整段不渲染）。
 *
 * 【为什么这里仍然直用 Radix `Button`】选中态是 `solid + color="blue"`（硬编码
 * 蓝），而 `AppButton` 的选中态走主题强调色（本项目默认 iris）——迁移会改变色相，
 * 属于视觉决策，不在"拆文件"的范围内。设计系统门禁因此按**文件数量**结算：这段
 * 控件原本就是 `ParamToolbar.tsx` 里唯一的 `Button` 使用者，整体搬走后原文件
 * 不再引用它，**计数不变**（见 `araConformanceGates` 的棘轮）。
 */
import { Button, Flex } from "@radix-ui/themes";
import { setTrackStateRemote } from "../../../../features/session/sessionSlice";
import { resolvePitchAlgoSelectValue } from "../../../../features/tracks/pitchAlgoOptions";
import { AppSelect } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { ToolbarSlot } from "./toolbarParts";

/** 本文件用到的模型字段（AST 扫描得出）。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "dispatch"
        | "getProcessorParamLabel"
        | "getStaticOptionLabel"
        | "handleStaticParamChange"
        | "pitchAlgoOptions"
        | "processorStaticParams"
        | "processorStaticValues"
        | "rootTrack"
        | "rootTrackId"
        | "t"
        | "tf"
        | "toolbarTier"
    >;
};

export function ParamStaticControls({ m }: Props) {
    const {
        dispatch,
        getProcessorParamLabel,
        getStaticOptionLabel,
        handleStaticParamChange,
        pitchAlgoOptions,
        processorStaticParams,
        processorStaticValues,
        rootTrack,
        rootTrackId,
        t,
        tf,
        toolbarTier,
    } = m;

    return (
        <>
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
        </>
    );
}
