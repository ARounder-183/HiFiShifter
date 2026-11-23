/**
 * 播放光标时间的**唯一主实例**分发点。
 *
 * 【为什么需要这一层】读数在工具栏、状态栏、轨道头三处都是 chrome，而"它此刻应该
 * 出现在哪儿"是一个用户偏好（`playheadTimePlacement`），不是某个面板的属性。
 * 每个挂载点都自己去读偏好、自己判断"该不该渲染"，迟早会出现两份实例同时可见
 * （用户报的正是"时间显示了两遍 / 一遍都没有"）。这里把判断收敛成一处：
 * **只有解析结果等于本槽位时才渲染**。
 *
 * 【为什么右键菜单也在这里】位置是同一份偏好，三个槽位上的菜单必须逐字一致；
 * 挂在读数组件里就等于把这份一致性复制三遍。
 */
import { useState } from "react";

import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import {
    PLAYHEAD_TIME_PLACEMENTS,
    playheadTimePlacementLabelKey,
    resolvePlayheadTimePlacement,
    type PlayheadTimeSlot,
} from "../../../features/session/playheadTimePlacement";
import {
    persistUiSettings,
    setPlayheadTimePlacement,
} from "../../../features/session/sessionSlice";
import { useI18n } from "../../../i18n/I18nProvider";
import { AppContextMenu } from "../../../ui/Menu";
import { PlayheadTimeReadout } from "./PlayheadTimeReadout";

export interface PlayheadTimeReadoutSlotProps {
    slot: PlayheadTimeSlot;
    /** 只显示主单位（轨道头）。 */
    compact?: boolean;
}

export function PlayheadTimeReadoutSlot({ slot, compact }: PlayheadTimeReadoutSlotProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const placement = useAppSelector((state) => state.session.playheadTimePlacement);
    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
    if (resolvePlayheadTimePlacement(placement) !== slot) return null;
    return (
        <>
            <PlayheadTimeReadout
                variant={slot}
                compact={compact}
                // 轨道头是紧凑副本：它没有独立时钟，插值只会让它与画布上的光标错开。
                interpolate={slot !== "trackHeader"}
                onContextMenu={(event) => {
                    event.preventDefault();
                    setMenu({ x: event.clientX, y: event.clientY });
                }}
            />
            {menu ? (
                <AppContextMenu
                    x={menu.x}
                    y={menu.y}
                    ariaLabel={t("playhead_time_placement")}
                    items={PLAYHEAD_TIME_PLACEMENTS.map((option) => ({
                        key: option,
                        label: t(playheadTimePlacementLabelKey(option)),
                        checked: placement === option,
                        onSelect: () => {
                            dispatch(setPlayheadTimePlacement(option));
                            void dispatch(persistUiSettings());
                        },
                    }))}
                    onClose={() => setMenu(null)}
                />
            ) : null}
        </>
    );
}
