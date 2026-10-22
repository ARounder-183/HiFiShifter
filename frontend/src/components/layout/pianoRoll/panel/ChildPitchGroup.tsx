/**
 * **子轨道音高组**：只在选中子轨道时出现，一个带下拉的药丸 + 一只眼睛。
 *
 * 子轨道把音高分成了两种口径：`childPitchOffsetCentsParam`（音分偏移）与
 * `childPitchOffsetDegreesParam`（音级偏移）。下拉在两者之间切换，药丸本身显示
 * 当前口径（`pitchGroupLabel` / `pitchGroupTooltip` 由工具栏算好传下来）；
 * 眼睛切换该口径的副参数叠加曲线。
 *
 * 【为什么单独成文件】它和 `ParamButtons` 里的"共振峰组"是同一个交互形状
 * （药丸 + 下拉 + 眼睛），但触发条件完全不同（这条只看"选中的是不是子轨道"，
 * 不看参数描述符）。放在一起时，"为什么有时多出一个药丸"要靠读两个不相邻的
 * 条件表达式才能回答。
 */
import { DropdownMenu } from "@radix-ui/themes";
import { ChevronDownIcon } from "@radix-ui/react-icons";
import { setEditParam } from "../../../../features/session/sessionSlice";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { ParamToolbarPill } from "./toolbarParts";

/** 本文件用到的模型字段（AST 扫描得出）。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "childPitchOffsetCentsParam"
        | "childPitchOffsetDegreesParam"
        | "dispatch"
        | "editParam"
        | "pitchGroupActive"
        | "pitchGroupLabel"
        | "pitchGroupTooltip"
        | "secondaryParamVisible"
        | "selectedIsChildTrack"
        | "t"
        | "toggleSecondaryParam"
    >;
};

export function ChildPitchGroup({ m }: Props) {
    const {
        childPitchOffsetCentsParam,
        childPitchOffsetDegreesParam,
        dispatch,
        editParam,
        pitchGroupActive,
        pitchGroupLabel,
        pitchGroupTooltip,
        secondaryParamVisible,
        selectedIsChildTrack,
        t,
        toggleSecondaryParam,
    } = m;

    return (
        <>
            {selectedIsChildTrack &&
            (childPitchOffsetCentsParam || childPitchOffsetDegreesParam) ? (
                <DropdownMenu.Root>
                    <ParamToolbarPill
                        label={pitchGroupLabel}
                        labelTooltip={pitchGroupTooltip}
                        active={pitchGroupActive}
                        onSelect={() => dispatch(setEditParam("pitch"))}
                        eyeMode={
                            pitchGroupActive
                                ? "main"
                                : secondaryParamVisible["pitch"]
                                  ? "on"
                                  : "off"
                        }
                        onToggleEye={() => toggleSecondaryParam("pitch")}
                        eyeTooltip={
                            secondaryParamVisible["pitch"]
                                ? t("secondary_overlay_tooltip_visible")
                                : t("secondary_overlay_tooltip_hidden")
                        }
                        eyeLabel={
                            secondaryParamVisible["pitch"]
                                ? t("hide_secondary_param")
                                : t("show_secondary_param")
                        }
                        dropdown={
                            <DropdownMenu.Trigger
                                className="param-pill__seg param-pill__seg--chev"
                                data-tooltip={pitchGroupTooltip}
                                tabIndex={-1}
                            >
                                <ChevronDownIcon width="12" height="12" />
                            </DropdownMenu.Trigger>
                        }
                    />
                    <DropdownMenu.Content variant="soft" color="gray">
                        <DropdownMenu.RadioGroup
                            value={editParam}
                            onValueChange={(value) => dispatch(setEditParam(value))}
                        >
                            <DropdownMenu.RadioItem value="pitch">
                                {t("child_pitch_root_option")}
                            </DropdownMenu.RadioItem>
                            {childPitchOffsetCentsParam ? (
                                <DropdownMenu.RadioItem value={childPitchOffsetCentsParam}>
                                    {t("child_pitch_cents_option")}
                                </DropdownMenu.RadioItem>
                            ) : null}
                            {childPitchOffsetDegreesParam ? (
                                <DropdownMenu.RadioItem value={childPitchOffsetDegreesParam}>
                                    {t("child_pitch_degrees_option")}
                                </DropdownMenu.RadioItem>
                            ) : null}
                        </DropdownMenu.RadioGroup>
                    </DropdownMenu.Content>
                </DropdownMenu.Root>
            ) : (
                <ParamToolbarPill
                    label={t("param_btn_pitch")}
                    labelTooltip={t("common_pitch")}
                    active={editParam === "pitch"}
                    onSelect={() => dispatch(setEditParam("pitch"))}
                    eyeMode={
                        editParam === "pitch"
                            ? "main"
                            : secondaryParamVisible["pitch"]
                              ? "on"
                              : "off"
                    }
                    onToggleEye={() => toggleSecondaryParam("pitch")}
                    eyeTooltip={
                        secondaryParamVisible["pitch"]
                            ? t("secondary_overlay_tooltip_visible")
                            : t("secondary_overlay_tooltip_hidden")
                    }
                    eyeLabel={
                        secondaryParamVisible["pitch"]
                            ? t("hide_secondary_param")
                            : t("show_secondary_param")
                    }
                />
            )}
        </>
    );
}
