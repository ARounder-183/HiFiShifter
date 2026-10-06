import { test } from "vitest";

import reducer, { setToolMode } from "./sessionSlice.ts";
import { loadUiSettings } from "./thunks/runtimeThunks.ts";
import type { UiSettings } from "../../services/api/settings.ts";
import type { ToolMode } from "./sessionTypes.ts";

/**
 * 参数编辑器的工具选择：
 * - 三个字段（`toolMode` / `toolModeGroup` / `drawToolMode`）必须由同一次切换一起
 *   写对 —— `drawToolMode` 是"按 `Tab` 回跳到哪个绘制工具"的依据，漏写就回不去；
 * - 它属于"本机记忆"，要从设置里读回，且野值不得污染状态。
 */
test("features/session/sessionSlice.toolMode.test.ts tool mode state and hydration", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const base = reducer(undefined, { type: "@@INIT" });
    assertEqual(base.toolMode, "draw", "初始工具是绘制");
    assertEqual(base.toolModeGroup, "draw", "初始分组是绘制组");
    assertEqual(base.drawToolMode, "draw", "初始绘制工具是绘制");

    // ── 切换绘制类工具：三个字段一起走 ──
    for (const mode of ["draw", "line", "vibrato"] as const) {
        const next = reducer(base, setToolMode(mode));
        assertEqual(next.toolMode, mode, `${mode}：toolMode`);
        assertEqual(next.toolModeGroup, "draw", `${mode}：留在绘制组`);
        assertEqual(next.drawToolMode, mode, `${mode}：记住绘制工具`);
    }

    // ── 切到选择工具：只改分组，不改"上次用的绘制工具" ──
    const toVibrato = reducer(base, setToolMode("vibrato"));
    const toSelect = reducer(toVibrato, setToolMode("select"));
    assertEqual(toSelect.toolMode, "select", "选择工具：toolMode");
    assertEqual(toSelect.toolModeGroup, "select", "选择工具：分组");
    assertEqual(toSelect.drawToolMode, "vibrato", "选择工具：保留上次的绘制工具（供 Tab 回跳）");

    // ── 水合：从设置里读回 ──
    // （这里只关心工具三字段，其余字段与这条断言无关。）
    const hydrate = (paramEditorTool: unknown) =>
        reducer(
            base,
            loadUiSettings.fulfilled({ paramEditorTool } as UiSettings, "req", undefined),
        );

    for (const mode of ["select", "draw", "line", "vibrato"] as const) {
        const state = hydrate(mode);
        assertEqual(state.toolMode, mode, `水合 ${mode}：toolMode`);
        assertEqual(
            state.toolModeGroup,
            mode === "select" ? "select" : "draw",
            `水合 ${mode}：分组`,
        );
        if (mode !== "select") {
            assertEqual(state.drawToolMode, mode, `水合 ${mode}：绘制工具`);
        }
    }

    // 手改配置里的野值 / 缺项一律忽略，回落出厂默认。
    assertEqual(hydrate("sideways").toolMode, "draw", "未知取值回落默认");
    assertEqual(hydrate(undefined).toolMode, "draw", "缺项回落默认");
    assertEqual(hydrate(null).toolMode, "draw", "null 回落默认");
    assertEqual(hydrate(3).toolMode, "draw", "非字符串回落默认");

    // 水合非法值时不得把状态写成"半切换"的样子。
    const wild = hydrate("nonsense");
    assertEqual(wild.toolModeGroup, "draw", "野值：分组保持默认");
    assertEqual(wild.drawToolMode, "draw", "野值：绘制工具保持默认");
});

/**
 * 每个 `ToolMode` 都必须是合法的持久化取值 —— 白名单漏一个，
 * 那个工具的选择就无法在重启后恢复，且不会有任何报错。
 */
test("features/session/sessionSlice.toolMode.test.ts every tool survives a round trip", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const all: ToolMode[] = ["select", "draw", "line", "vibrato"];
    for (const mode of all) {
        const persisted = reducer(base, setToolMode(mode));
        const restored = reducer(
            base,
            loadUiSettings.fulfilled(
                { paramEditorTool: persisted.toolMode } as UiSettings,
                "req",
                undefined,
            ),
        );
        if (restored.toolMode !== mode) {
            throw new Error(`${mode} 未能从设置恢复，得到 ${restored.toolMode}`);
        }
    }
});
