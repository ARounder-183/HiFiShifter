/**
 * 参考轨道组下拉 + 「导入 MIDI」（只在当前参数是**音高**时出现）。
 *
 * 参考轨道组决定画布上叠加哪些轨道的参考音高曲线：下拉里是"全部 / 清空 / 逐轨
 * 复选"，复选项带该轨道的描边颜色圆点，指针悬停时高亮那条参考线
 * （`setHoveredReferenceRootTrackId`）。
 *
 * 【为什么单独成文件】这段只在 `editParam === "pitch"` 时渲染，位置固定在
 * "音高"按钮左侧；它夹在工具栏里时，"为什么切到张力这些按钮就消失"这件事
 * 只能靠读一个内联三元表达式才知道。
 *
 * 【两个按钮的样式来源】都改用设计系统原语 `AppButton`（门禁要求，见
 * `araConformanceGates` 的"表单控件走设计系统原语"）：
 * - 参考轨道组：`variant="soft" color="gray"` → `intent="default"`（原语的默认值
 *   就是 soft + gray）；
 * - 导入 MIDI：`variant="soft"` 且不指定 color（= 强调色）→
 *   `intent="primary" emphasis="soft"`（primary 的底色是强调色，emphasis 把实心压成
 *   浅色）。两者与迁移前的渲染结果逐属性一致，`size="1"` 对应原语的 `size="sm"`。
 */
import React from "react";
import { DropdownMenu, Flex } from "@radix-ui/themes";
import { ChevronDownIcon } from "@radix-ui/react-icons";
import {
    persistUiSettings,
    toggleVisibleReferenceRootTrackId,
} from "../../../../features/session/sessionSlice";
import { AppButton } from "../../../../ui";
import { buildReferencePitchStrokeColor } from "../referenceRootTracks";
import { buildReferenceRootTrackTriggerElement } from "../referenceRootTrackTrigger";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { ToolbarSlot } from "./toolbarParts";

/** 本文件用到的模型字段（AST 扫描得出）。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "dispatch"
        | "editParam"
        | "handleOpenMidiDialog"
        | "pitchEnabled"
        | "pitchHardDisableReason"
        | "referenceRootTrackOptions"
        | "rootTrack"
        | "setHoveredReferenceRootTrackId"
        | "t"
        | "tf"
        | "toolbarTier"
        | "updateVisibleReferenceRootTrackIds"
        | "visibleReferenceRootTrackIds"
    >;
};

export function ReferenceTrackGroup({ m }: Props) {
    const {
        dispatch,
        editParam,
        handleOpenMidiDialog,
        pitchEnabled,
        pitchHardDisableReason,
        referenceRootTrackOptions,
        rootTrack,
        setHoveredReferenceRootTrackId,
        t,
        tf,
        toolbarTier,
        updateVisibleReferenceRootTrackIds,
        visibleReferenceRootTrackIds,
    } = m;

    return (
        <>
            {/* 参考轨道组 / 导入 MIDI：仅当切换到“音高”参数时显示（位置固定在“音高”左侧）。
                按钮走设计系统原语 AppButton（soft），简写 + ToolTip 保留全称。 */}
            {rootTrack && editParam === "pitch" ? (
                <React.Fragment>
                    <DropdownMenu.Root>
                        <DropdownMenu.Trigger data-tooltip={t("reference_root_tracks")}>
                            <AppButton size="sm" style={{ cursor: "pointer" }}>
                                {/* 第 9 级只保留下拉箭头：参考轨道组没有菜单 /
                                            快捷键入口，是**唯一入口**，不能整块隐藏，
                                            只能瘦身；全称仍在按钮的 ToolTip 里。 */}
                                <ToolbarSlot hideAt={9} tier={toolbarTier}>
                                    {buildReferenceRootTrackTriggerElement(
                                        `${tf("reference_root_tracks_short")}${
                                            visibleReferenceRootTrackIds.length > 0
                                                ? ` (${visibleReferenceRootTrackIds.length})`
                                                : ""
                                        }`,
                                    )}
                                </ToolbarSlot>
                                <ChevronDownIcon width="12" height="12" />
                            </AppButton>
                        </DropdownMenu.Trigger>
                        <DropdownMenu.Content variant="soft" color="gray">
                            <DropdownMenu.Item
                                onSelect={() =>
                                    updateVisibleReferenceRootTrackIds(
                                        referenceRootTrackOptions.map((track) => track.id),
                                    )
                                }
                            >
                                {t("reference_root_tracks_all")}
                            </DropdownMenu.Item>
                            <DropdownMenu.Item
                                onSelect={() => updateVisibleReferenceRootTrackIds([])}
                            >
                                {t("reference_root_tracks_clear")}
                            </DropdownMenu.Item>
                            <DropdownMenu.Separator />
                            {referenceRootTrackOptions.length === 0 ? (
                                <DropdownMenu.Item disabled>
                                    {t("reference_root_tracks_empty")}
                                </DropdownMenu.Item>
                            ) : (
                                referenceRootTrackOptions.map((track) => (
                                    <DropdownMenu.CheckboxItem
                                        key={track.id}
                                        checked={visibleReferenceRootTrackIds.includes(track.id)}
                                        onCheckedChange={() => {
                                            dispatch(toggleVisibleReferenceRootTrackId(track.id));
                                            void dispatch(persistUiSettings());
                                        }}
                                        onPointerEnter={() =>
                                            setHoveredReferenceRootTrackId(track.id)
                                        }
                                        onPointerLeave={() => setHoveredReferenceRootTrackId(null)}
                                    >
                                        <Flex align="center" gap="2">
                                            <span
                                                className="inline-block h-2.5 w-2.5 rounded-full"
                                                style={{
                                                    background: buildReferencePitchStrokeColor(
                                                        track.color,
                                                        true,
                                                    ),
                                                }}
                                            />
                                            <span>{track.name}</span>
                                        </Flex>
                                    </DropdownMenu.CheckboxItem>
                                ))
                            )}
                        </DropdownMenu.Content>
                    </DropdownMenu.Root>
                    {/* 第 8 级隐藏「导入 MIDI」：时间轴上同样能导入 MIDI
                                （`TimelinePanel` 也有入口），这里不是唯一路径。 */}
                    <ToolbarSlot hideAt={8} tier={toolbarTier}>
                        <span
                            className="inline-flex"
                            data-tooltip={pitchHardDisableReason ?? tf("midi_import")}
                        >
                            <AppButton
                                intent="primary"
                                emphasis="soft"
                                size="sm"
                                onClick={handleOpenMidiDialog}
                                disabled={!pitchEnabled}
                                style={{ cursor: "pointer" }}
                            >
                                {tf("midi_import")}
                            </AppButton>
                        </span>{" "}
                    </ToolbarSlot>
                </React.Fragment>
            ) : null}
        </>
    );
}
