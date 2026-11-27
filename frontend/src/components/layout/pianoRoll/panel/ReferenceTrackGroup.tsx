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
 *
 * 【菜单内容为什么是数据而不是手写行】这里曾经手写 `DropdownMenu.CheckboxItem`
 * （颜色圆点 + 悬停联动）。迁移到共享表面后，圆点交给条目的 `icon` 列 —— 那一列
 * 本来就是"前导图形"的位置，于是 6 条轨道的圆点自动对齐成一列，而"全部 / 清空"
 * 两行也自动让位对齐。手写行的唯一理由（Radix 的条目表达不了圆点）消失了。
 */
import React, { useRef, useState } from "react";
import { ChevronDownIcon } from "@radix-ui/react-icons";
import {
    persistUiSettings,
    toggleVisibleReferenceRootTrackId,
} from "../../../../features/session/sessionSlice";
import { AppButton } from "../../../../ui";
import { AppAnchoredMenu, type AppMenuItemSpec } from "../../../../ui/Menu";
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

    const anchorRef = useRef<HTMLButtonElement | null>(null);
    const menuRef = useRef<HTMLDivElement | null>(null);
    const [open, setOpen] = useState(false);

    const items: AppMenuItemSpec[] = [
        {
            key: "all",
            label: t("reference_root_tracks_all"),
            onSelect: () =>
                updateVisibleReferenceRootTrackIds(
                    referenceRootTrackOptions.map((track) => track.id),
                ),
        },
        {
            key: "clear",
            label: t("reference_root_tracks_clear"),
            onSelect: () => updateVisibleReferenceRootTrackIds([]),
        },
        ...(referenceRootTrackOptions.length === 0
            ? [
                  {
                      key: "empty",
                      label: t("reference_root_tracks_empty"),
                      disabled: true,
                      separatorBefore: true,
                  },
              ]
            : referenceRootTrackOptions.map((track, index) => ({
                  key: track.id,
                  label: track.name,
                  // 颜色圆点进前导列：它说明"这一条在画布上是哪个颜色"，
                  // 与"前导图形"是同一件事，因此不需要一行手写的复合标签。
                  icon: (
                      <span
                          className="inline-block h-2.5 w-2.5 rounded-full"
                          style={{
                              background: buildReferencePitchStrokeColor(track.color, true),
                          }}
                      />
                  ),
                  checked: visibleReferenceRootTrackIds.includes(track.id),
                  // 分隔线只加在轨道列表**第一项**上（迁移前是一个独立 Separator）。
                  separatorBefore: index === 0,
                  /*
                   * 【复选不收起菜单】勾轨道是"开关"，不是"命令"：用户要连着勾几条、
                   * 并逐条看画布上的曲线变化。每勾一次就关掉菜单等于让他重开 N 次。
                   */
                  keepOpen: true,
                  onSelect: () => {
                      dispatch(toggleVisibleReferenceRootTrackId(track.id));
                      void dispatch(persistUiSettings());
                  },
                  onHover: () => setHoveredReferenceRootTrackId(track.id),
                  onHoverEnd: () => setHoveredReferenceRootTrackId(null),
              }))),
    ];

    return (
        <>
            {/* 参考轨道组 / 导入 MIDI：仅当切换到"音高"参数时显示（位置固定在"音高"左侧）。
                按钮走设计系统原语 AppButton（soft），简写 + ToolTip 保留全称。 */}
            {rootTrack && editParam === "pitch" ? (
                <React.Fragment>
                    <AppButton
                        // 【为什么锚点必须是这个按钮本身】菜单按锚点的视口矩形定位；
                        // 交给外层包装元素会让"右对齐"落在包装盒的右边而不是按钮的右边。
                        ref={anchorRef}
                        size="sm"
                        style={{ cursor: "pointer" }}
                        data-tooltip={t("reference_root_tracks")}
                        aria-haspopup="menu"
                        aria-expanded={open}
                        onClick={() => setOpen((value) => !value)}
                    >
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
                    <AppAnchoredMenu
                        open={open}
                        anchorRef={anchorRef}
                        menuRef={menuRef}
                        items={items}
                        onClose={() => setOpen(false)}
                        align="end"
                        ariaLabel={t("reference_root_tracks")}
                    />
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
