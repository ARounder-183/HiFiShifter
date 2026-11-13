/**
 * 自动滚屏的模式字段：初始值、写入白名单、与开关正交、水合回落。
 *
 * 【为什么单独一个用例】"开关"与"模式"是两根正交的轴，而它们住在同一个状态对象里，
 * 最容易在改动中互相污染（例如把 `setAutoScrollMode` 写成"顺手打开"）。这里把
 * 四条边界逐条钉住：初始值、非法值静默忽略、两根轴互不影响、水合按白名单回落。
 */
import { expect, test } from "vitest";

import reducer, { setAutoScrollMode, toggleAutoScroll } from "./sessionSlice.ts";
import { loadUiSettings } from "./thunks/runtimeThunks.ts";
import type { UiSettings } from "../../services/api/settings.ts";

test("auto scroll mode: 初始值、写入白名单、与开关正交", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    expect(base.autoScrollMode).toBe("scroll");

    expect(reducer(base, setAutoScrollMode("page")).autoScrollMode).toBe("page");
    // 非法值静默忽略，不改变现状
    expect(reducer(base, setAutoScrollMode("teleport" as never)).autoScrollMode).toBe("scroll");

    // 开关不碰模式，模式不碰开关
    const on = reducer(base, toggleAutoScroll());
    expect(on.autoScrollEnabled).toBe(true);
    expect(on.autoScrollMode).toBe("scroll");
    expect(reducer(on, setAutoScrollMode("page")).autoScrollEnabled).toBe(true);
});

test("auto scroll mode: 水合按白名单回落", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const hydrate = (autoScrollMode: unknown) =>
        reducer(base, loadUiSettings.fulfilled({ autoScrollMode } as UiSettings, "req", undefined));

    expect(hydrate("page").autoScrollMode).toBe("page");
    expect(hydrate("scroll").autoScrollMode).toBe("scroll");
    expect(hydrate("sideways").autoScrollMode).toBe("scroll");
    expect(hydrate(undefined).autoScrollMode).toBe("scroll");
});
