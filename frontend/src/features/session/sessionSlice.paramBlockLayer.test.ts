import { expect, test } from "vitest";

import reducer, {
    isParamBlockLayerVisible,
    setAlwaysShowParamBlocks,
    setSelectToolMode,
    setToolMode,
} from "./sessionSlice";
import type { SelectToolMode, ToolMode } from "./sessionTypes";

/**
 * 块层可见性：**纯派生**，没有状态位。
 *
 * 这张表就是验收标准（见计划 1.1）。它与 `isBlockSelectionActive` 一起构成唯一的
 * 判据，任何「某个写入者顺手把图层打开」的实现都会在下面某一格失败。
 */
const cases: Array<{
    readonly toolMode: ToolMode;
    readonly selectToolMode: SelectToolMode;
    readonly alwaysShow: boolean;
    readonly visible: boolean;
}> = [
    { toolMode: "select", selectToolMode: "block", alwaysShow: false, visible: true },
    { toolMode: "select", selectToolMode: "block", alwaysShow: true, visible: true },
    { toolMode: "select", selectToolMode: "range", alwaysShow: false, visible: false },
    { toolMode: "select", selectToolMode: "range", alwaysShow: true, visible: true },
    { toolMode: "draw", selectToolMode: "range", alwaysShow: false, visible: false },
    { toolMode: "draw", selectToolMode: "range", alwaysShow: true, visible: true },
    // 绘制族 + 记忆里的 block：不在块编辑里，所以只看设置。
    { toolMode: "line", selectToolMode: "block", alwaysShow: false, visible: false },
    { toolMode: "vibrato", selectToolMode: "block", alwaysShow: true, visible: true },
];

test("isParamBlockLayerVisible 覆盖工具族 × 子模式 × 常显设置", () => {
    for (const item of cases) {
        const actual = isParamBlockLayerVisible({
            alwaysShowParamBlocks: item.alwaysShow,
            toolMode: item.toolMode,
            selectToolMode: item.selectToolMode,
        });
        expect(
            actual,
            `${item.toolMode}/${item.selectToolMode}/alwaysShow=${String(item.alwaysShow)}`,
        ).toBe(item.visible);
    }
});

test("块选择模式下关掉常显设置，图层仍然可见", () => {
    expect(
        isParamBlockLayerVisible({
            alwaysShowParamBlocks: false,
            toolMode: "select",
            selectToolMode: "block",
        }),
    ).toBe(true);
});

test("设置只写设置：任何工具切换都不写常显字段", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    expect(base.alwaysShowParamBlocks).toBe(false);
    for (const mode of ["select", "draw", "line", "vibrato"] as const) {
        expect(reducer(base, setToolMode(mode)).alwaysShowParamBlocks).toBe(false);
    }
    expect(reducer(base, setSelectToolMode("block")).alwaysShowParamBlocks).toBe(false);
    expect(reducer(base, setSelectToolMode("range")).alwaysShowParamBlocks).toBe(false);
});

test("工具只写工具：常显开关不改 toolMode / selectToolMode", () => {
    const block = reducer(undefined, { type: "@@INIT" });
    const inBlock = reducer(block, setSelectToolMode("block"));
    const enabled = reducer(inBlock, setAlwaysShowParamBlocks(true));
    expect(enabled.alwaysShowParamBlocks).toBe(true);
    expect(enabled.toolMode).toBe("select");
    expect(enabled.selectToolMode).toBe("block");
    const disabled = reducer(enabled, setAlwaysShowParamBlocks(false));
    expect(disabled.alwaysShowParamBlocks).toBe(false);
    expect(disabled.selectToolMode).toBe("block");
});
