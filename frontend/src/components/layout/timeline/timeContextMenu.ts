/**
 * 时间显示菜单项：**主/副单位 + 读数位置 + 复制光标时间 + 显示设置**。
 *
 * 【为什么抽成一份】标尺右键菜单与播放光标时间读数的右键菜单要给的正是同一组
 * 设置。各写一遍的下场是"两处都能改单位，但只有一处会把结果持久化"，或者两处的
 * 勾选态对不上 —— 用户看到的两个入口描述的不是同一件事。这里只有一个构建函数：
 * 纯函数 [buildTimeContextMenuItems] 负责形状，hook [useTimeContextMenuItems] 负责
 * 接线（读 Redux、写回并持久化）。
 *
 * 【为什么标尺的"速度映射"那一组不在这里】那组项是**在标尺上按坐标**成立的
 * （点在哪、命中哪个变化点），读数没有坐标可用；把它塞进共享构建器只会让每个
 * 调用点都要伪造一个 `clickedSec`。共享的是"时间显示"这一组，不是整张菜单。
 */
import { useCallback, useMemo } from "react";
import { shallowEqual } from "react-redux";

import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import {
    PLAYHEAD_TIME_PLACEMENTS,
    playheadTimePlacementLabelKey,
    resolvePlayheadTimePlacement,
    type PlayheadTimePlacement,
    type PlayheadTimeSlot,
} from "../../../features/session/playheadTimePlacement";
import {
    persistUiSettings,
    setPlayheadTimePlacement,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
} from "../../../features/session/sessionSlice";
import type { AppMenuItemSpec } from "../../../ui/Menu";
import { usePlayheadTimeSlotAvailability } from "../transport/playheadTimeSlots";
import { TIME_UNITS, TIME_UNIT_CHOICES, timeUnitLabelKey } from "./timeFormat";
import type { TimeUnit, TimeUnitChoice } from "./timeFormat";

export interface TimeContextMenuInput {
    t: (key: string) => string;
    primaryUnit: TimeUnit;
    secondaryUnit: TimeUnitChoice;
    placement: PlayheadTimePlacement;
    /** `auto` 此刻解析到的槽位：作为该项的尾部标注，回答"自动到底放哪儿"。 */
    resolvedSlot: PlayheadTimeSlot | "hidden";
    /** 本组项之前是否要先来一条分隔线（标尺菜单前面还有速度映射那一组）。 */
    leadingSeparator?: boolean;
    onSelectPrimary: (unit: TimeUnit) => void;
    onSelectSecondary: (unit: TimeUnitChoice) => void;
    onSelectPlacement: (placement: PlayheadTimePlacement) => void;
    onCopyPlayheadTime?: () => void;
    onOpenSettings?: () => void;
}

/**
 * 构建"时间显示"这一组菜单项。
 *
 * 结构：主单位（标题 + 四项）→ 副单位（标题 + 五项）→ 读数位置（**二级菜单**）→
 * 复制 → 显示设置。
 *
 * 【读数位置为什么必须收进二级菜单】它自身就是五个互斥项。平铺之后，这张菜单
 * 一半的行都在说"读数放哪儿"，而用户九成时间是来改单位的；更要紧的是，位置在
 * 标尺菜单里只是顺带的入口，不该与单位抢同一层。
 */
export function buildTimeContextMenuItems(input: TimeContextMenuInput): AppMenuItemSpec[] {
    const { t, primaryUnit, secondaryUnit, placement, resolvedSlot } = input;
    const items: AppMenuItemSpec[] = [
        {
            key: "primaryHeader",
            label: t("time_unit_primary"),
            heading: true,
            separatorBefore: input.leadingSeparator,
        },
        ...TIME_UNITS.map((unit) => ({
            key: `primary-${unit}`,
            label: t(timeUnitLabelKey(unit)),
            checked: primaryUnit === unit,
            onSelect: () => input.onSelectPrimary(unit),
        })),
        {
            key: "secondaryHeader",
            label: t("time_unit_secondary"),
            heading: true,
            separatorBefore: true,
        },
        ...TIME_UNIT_CHOICES.map((unit) => ({
            key: `secondary-${unit}`,
            label: unit === "none" ? t("time_unit_none") : t(timeUnitLabelKey(unit as TimeUnit)),
            checked: secondaryUnit === unit,
            onSelect: () => input.onSelectSecondary(unit),
        })),
        {
            key: "playheadTimePlacement",
            label: t("playhead_time_placement"),
            separatorBefore: true,
            items: PLAYHEAD_TIME_PLACEMENTS.map((option) => ({
                key: `placement-${option}`,
                label: t(playheadTimePlacementLabelKey(option)),
                checked: placement === option,
                // `auto` 的尾部标注：让"自动"这一项自己说清楚它此刻落在哪一处，
                // 用户不必去猜（链可能因窗口不同而解析到别处）。
                hint:
                    option === "auto" ? t(playheadTimePlacementLabelKey(resolvedSlot)) : undefined,
                onSelect: () => input.onSelectPlacement(option),
            })),
        },
    ];
    if (input.onCopyPlayheadTime) {
        items.push({
            key: "copyPlayheadTime",
            label: t("copy_playhead_time"),
            separatorBefore: true,
            onSelect: input.onCopyPlayheadTime,
        });
    }
    if (input.onOpenSettings) {
        items.push({
            key: "openSettings",
            label: t("timeline_display_settings"),
            separatorBefore: !input.onCopyPlayheadTime,
            onSelect: input.onOpenSettings,
        });
    }
    return items;
}

export interface TimeContextMenuHandlers {
    t: (key: string) => string;
    leadingSeparator?: boolean;
    onCopyPlayheadTime?: () => void;
    onOpenSettings?: () => void;
}

/**
 * 接线版：从会话切片读单位与放置策略，写回时**连同持久化**一起做。
 *
 * 两个入口（标尺 / 读数）共用它，因此"改单位"这件事只有一条实现 —— 包括那条
 * 容易被漏掉的 `persistUiSettings`。
 */
export function useTimeContextMenuItems(handlers: TimeContextMenuHandlers): AppMenuItemSpec[] {
    const dispatch = useAppDispatch();
    const { primaryUnit, secondaryUnit, placement } = useAppSelector(
        (state) => ({
            primaryUnit: state.session.primaryTimeUnit,
            secondaryUnit: state.session.secondaryTimeUnit,
            placement: state.session.playheadTimePlacement,
        }),
        shallowEqual,
    );
    const availability = usePlayheadTimeSlotAvailability();
    const resolvedSlot = resolvePlayheadTimePlacement(placement, availability);

    const onSelectPrimary = useCallback(
        (unit: TimeUnit) => {
            dispatch(setPrimaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const onSelectSecondary = useCallback(
        (unit: TimeUnitChoice) => {
            dispatch(setSecondaryTimeUnit(unit));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );
    const onSelectPlacement = useCallback(
        (next: PlayheadTimePlacement) => {
            dispatch(setPlayheadTimePlacement(next));
            void dispatch(persistUiSettings());
        },
        [dispatch],
    );

    const { t, leadingSeparator, onCopyPlayheadTime, onOpenSettings } = handlers;
    return useMemo(
        () =>
            buildTimeContextMenuItems({
                t,
                primaryUnit,
                secondaryUnit,
                placement,
                resolvedSlot,
                leadingSeparator,
                onSelectPrimary,
                onSelectSecondary,
                onSelectPlacement,
                onCopyPlayheadTime,
                onOpenSettings,
            }),
        [
            t,
            primaryUnit,
            secondaryUnit,
            placement,
            resolvedSlot,
            leadingSeparator,
            onSelectPrimary,
            onSelectSecondary,
            onSelectPlacement,
            onCopyPlayheadTime,
            onOpenSettings,
        ],
    );
}
