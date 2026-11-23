/**
 * "时间显示"这一组菜单项的**形状**测试。
 *
 * 【为什么是纯函数测试】这一组同时服务标尺右键菜单与播放光标时间读数的右键菜单。
 * 两处渲染的是同一份 `AppMenuItemSpec[]`，所以"菜单里到底有什么"可以在不挂 DOM
 * 的情况下钉死：位置收在二级菜单、速度映射不出现、`auto` 带解析结果标注。
 */
import { describe, expect, test } from "vitest";

import type { AppMenuItemSpec } from "../../../ui/Menu";
import { buildTimeContextMenuItems, type TimeContextMenuInput } from "./timeContextMenu";

function build(overrides: Partial<TimeContextMenuInput> = {}): AppMenuItemSpec[] {
    return buildTimeContextMenuItems({
        t: (key) => key,
        primaryUnit: "barBeats",
        secondaryUnit: "clock",
        placement: "auto",
        resolvedSlot: "toolbar",
        onSelectPrimary: () => undefined,
        onSelectSecondary: () => undefined,
        onSelectPlacement: () => undefined,
        ...overrides,
    });
}

function keys(items: AppMenuItemSpec[]): string[] {
    return items.map((item) => item.key);
}

describe("buildTimeContextMenuItems", () => {
    test("主单位与副单位平铺，位置收进二级菜单", () => {
        expect(keys(build())).toEqual([
            "primaryHeader",
            "primary-barBeats",
            "primary-barDivisions",
            "primary-seconds",
            "primary-clock",
            "secondaryHeader",
            "secondary-none",
            "secondary-barBeats",
            "secondary-barDivisions",
            "secondary-seconds",
            "secondary-clock",
            "playheadTimePlacement",
        ]);
    });

    test("速度映射那一组不在共享构建器里（它只属于标尺）", () => {
        expect(keys(build()).some((key) => key.toLowerCase().includes("tempo"))).toBe(false);
    });

    test("位置是五个互斥项，auto 带“此刻解析到哪儿”的尾部标注", () => {
        const placement = build().find((item) => item.key === "playheadTimePlacement");
        expect(placement?.onSelect).toBeUndefined();
        expect(placement?.items?.map((item) => item.key)).toEqual([
            "placement-auto",
            "placement-toolbar",
            "placement-statusBar",
            "placement-trackHeader",
            "placement-hidden",
        ]);
        const [auto, toolbar, , , hidden] = placement?.items ?? [];
        expect(auto.checked).toBe(true);
        expect(auto.hint).toBe("playhead_time_placement_toolbar");
        expect(toolbar.hint).toBeUndefined();
        expect(hidden.hint).toBeUndefined();
    });

    test("解析结果随槽位变化（链顺延后标注跟着变）", () => {
        const placement = build({ resolvedSlot: "statusBar" }).find(
            (item) => item.key === "playheadTimePlacement",
        );
        expect(placement?.items?.[0].hint).toBe("playhead_time_placement_status_bar");
    });

    test("勾选态反映当前主/副单位与放置策略", () => {
        const items = build({
            primaryUnit: "seconds",
            secondaryUnit: "none",
            placement: "trackHeader",
        });
        expect(items.find((item) => item.key === "primary-seconds")?.checked).toBe(true);
        expect(items.find((item) => item.key === "primary-clock")?.checked).toBe(false);
        expect(items.find((item) => item.key === "secondary-none")?.checked).toBe(true);
        const placement = items.find((item) => item.key === "playheadTimePlacement");
        expect(
            placement?.items?.find((item) => item.key === "placement-trackHeader")?.checked,
        ).toBe(true);
    });

    test("leadingSeparator 把分隔线交给首个标题（标尺菜单前面还有速度映射）", () => {
        expect(build()[0].separatorBefore).toBeUndefined();
        expect(build({ leadingSeparator: true })[0].separatorBefore).toBe(true);
        expect(build({ leadingSeparator: false })[0].separatorBefore).toBe(false);
    });

    test("复制与设置是可选的；只给设置时分隔线由它自己承担", () => {
        const copy = () => undefined;
        const settings = () => undefined;
        expect(keys(build())).not.toContain("copyPlayheadTime");
        expect(keys(build())).not.toContain("openSettings");

        const withBoth = build({ onCopyPlayheadTime: copy, onOpenSettings: settings });
        expect(keys(withBoth).slice(-2)).toEqual(["copyPlayheadTime", "openSettings"]);
        expect(withBoth.at(-2)?.separatorBefore).toBe(true);
        expect(withBoth.at(-1)?.separatorBefore).toBe(false);

        const settingsOnly = build({ onOpenSettings: settings });
        expect(keys(settingsOnly).at(-1)).toBe("openSettings");
        expect(settingsOnly.at(-1)?.separatorBefore).toBe(true);
    });
});
