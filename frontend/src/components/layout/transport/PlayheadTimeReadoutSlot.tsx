/**
 * 播放光标时间的**唯一主实例**分发点。
 *
 * 【为什么需要这一层】读数在工具栏、状态栏、轨道头三处都是 chrome，而"它此刻应该
 * 出现在哪儿"是一个用户偏好（`playheadTimePlacement`），不是某个面板的属性。
 * 每个挂载点都自己去读偏好、自己判断"该不该渲染"，迟早会出现两份实例同时可见
 * （用户报的正是"时间显示了两遍 / 一遍都没有"）。这里把判断收敛成一处：
 * **只有解析结果等于本槽位时才渲染**。
 *
 * 【为什么右键菜单也在这里】三个槽位上的菜单必须逐字一致（同一份
 * `useTimeContextMenuItems`），挂在读数组件里就等于把这份一致性复制三遍。
 */
import { useCallback, useState } from "react";
import { createPortal } from "react-dom";

import { useAppSelector } from "../../../app/hooks";
import {
    resolvePlayheadTimePlacement,
    type PlayheadTimeSlot,
} from "../../../features/session/playheadTimePlacement";
import { useI18n } from "../../../i18n/I18nProvider";
import { AppContextMenu } from "../../../ui/Menu";
import { menuPortalContainer } from "../../../ui/portalContainer";
import { TimelineDisplaySettingsDialog } from "../TimelineDisplaySettingsDialog";
import { copyPlayheadTimeToClipboard } from "../timeline/playheadTimeClipboard";
import { useTimeContextMenuItems } from "../timeline/timeContextMenu";
import { PlayheadTimeReadout } from "./PlayheadTimeReadout";
import { usePlayheadTimeSlotAvailability, useRegisterPlayheadTimeSlot } from "./playheadTimeSlots";

export interface PlayheadTimeReadoutSlotProps {
    slot: PlayheadTimeSlot;
}

export function PlayheadTimeReadoutSlot({ slot }: PlayheadTimeReadoutSlotProps) {
    const { t, tf } = useI18n();
    const placement = useAppSelector((state) => state.session.playheadTimePlacement);
    const availability = usePlayheadTimeSlotAvailability();
    // 无条件登记：轮不到自己时本组件返回 null，但它仍然"在那儿"，仍然算这一级可用
    // （见 playheadTimeSlots 的同名说明）。
    useRegisterPlayheadTimeSlot(slot);

    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const onCopyPlayheadTime = useCallback(() => copyPlayheadTimeToClipboard(), []);
    const onOpenSettings = useCallback(() => setSettingsOpen(true), []);
    // 菜单文案走 `tf`（字符串键）：与标尺菜单传入的是同一个入口，两处的键
    // 才可能逐字一致（见 timeContextMenu 文件头）。
    const items = useTimeContextMenuItems({ t: tf, onCopyPlayheadTime, onOpenSettings });

    const resolved = resolvePlayheadTimePlacement(placement, availability);
    if (resolved !== slot) return null;
    return (
        <>
            <PlayheadTimeReadout
                variant={slot}
                // 轨道头是紧凑副本：它没有独立时钟，插值只会让它与画布上的光标错开。
                interpolate={slot !== "trackHeader"}
                onContextMenu={(event) => {
                    event.preventDefault();
                    setMenu({ x: event.clientX, y: event.clientY });
                }}
            />
            {/*
              菜单必须 portal：`AppContextMenu` 的壳是 `position: fixed`，但工具栏
              与时间轴面板都有 `overflow: hidden` 的祖先 —— 留在原地会被裁掉，
              而这张菜单现在带二级子菜单（更宽更高），被裁一次就整块不可用。
            */}
            {menu
                ? createPortal(
                      <AppContextMenu
                          x={menu.x}
                          y={menu.y}
                          items={items}
                          ariaLabel={t("playhead_time_readout")}
                          floating
                          onClose={() => setMenu(null)}
                      />,
                      menuPortalContainer(),
                  )
                : null}
            {/* 与标尺菜单同一份"时间轴显示设置"入口：读数挪到哪儿都能就地改单位。 */}
            <TimelineDisplaySettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
        </>
    );
}
