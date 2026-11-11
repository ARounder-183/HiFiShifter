import { test } from "vitest";

import reducer, { setParamBlocksVisible, setSelectToolMode, setToolMode } from "./sessionSlice.ts";
import { loadUiSettings } from "./thunks/runtimeThunks.ts";
import type { UiSettings } from "../../services/api/settings.ts";
import type { ToolMode } from "./sessionTypes.ts";

/**
 * 参数编辑器的工具状态：
 * - `toolMode` 只表示两个工具族（select / draw / line / vibrato）；
 * - `selectToolMode` 表示选择族的当前 / 上次子模式（range / block）；
 * - `drawToolMode` 记住绘制族上次成员，供 Tab 回跳；
 * - block 是选择族子模式，不是 `ToolMode` 的第三个成员。
 */
test("features/session/sessionSlice.toolMode.test.ts tool family state and hydration", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const base = reducer(undefined, { type: "@@INIT" });
    assertEqual(base.toolMode, "draw", "初始工具族是绘制");
    assertEqual(base.selectToolMode, "range", "初始选择子模式是 range");
    assertEqual(base.drawToolMode, "draw", "初始绘制工具是绘制");

    // ── 切换绘制族成员：只改当前工具与绘制记忆 ──
    for (const mode of ["draw", "line", "vibrato"] as const) {
        const next = reducer(base, setToolMode(mode));
        assertEqual(next.toolMode, mode, `${mode}：toolMode`);
        assertEqual(next.selectToolMode, "range", `${mode}：不污染选择子模式`);
        assertEqual(next.drawToolMode, mode, `${mode}：记住绘制工具`);
    }

    // ── 切到选择族：只改族，不改"上次用的绘制工具" ──
    const toVibrato = reducer(base, setToolMode("vibrato"));
    const toSelect = reducer(toVibrato, setToolMode("select"));
    assertEqual(toSelect.toolMode, "select", "选择族：toolMode");
    assertEqual(toSelect.selectToolMode, "range", "选择族：保留 range");
    assertEqual(toSelect.drawToolMode, "vibrato", "选择族：保留上次绘制工具（供 Tab 回跳）");

    // ── 设置选择族子模式：进入 block 自动打开块图层 ──
    const toBlock = reducer(toSelect, setSelectToolMode("block"));
    assertEqual(toBlock.toolMode, "select", "块选择：toolMode 仍是 select");
    assertEqual(toBlock.selectToolMode, "block", "块选择：子模式");
    assertEqual(toBlock.paramBlocksVisible, true, "块选择：强制显示块图层");

    // ── 关闭块图层：选择子模式必须收回 range，不能留下不可达的 block ──
    const hidden = reducer(toBlock, setParamBlocksVisible(false));
    assertEqual(hidden.paramBlocksVisible, false, "关闭图层：可见性");
    assertEqual(hidden.selectToolMode, "range", "关闭图层：块子模式收回 range");

    // ── 从绘制族后台恢复 block 记忆：可见性被不变量拉起 ──
    const fromDraw = reducer(toVibrato, setSelectToolMode("block"));
    assertEqual(fromDraw.toolMode, "select", "从绘制族进入块选择：选择族");
    assertEqual(fromDraw.paramBlocksVisible, true, "从绘制族进入块选择：显示图层");

    // ── 水合：工具族与选择子模式分别读取、各自白名单校验 ──
    const hydrate = (paramEditorTool: unknown, paramEditorSelectionMode?: unknown) =>
        reducer(
            base,
            loadUiSettings.fulfilled(
                { paramEditorTool, paramEditorSelectionMode } as UiSettings,
                "req",
                undefined,
            ),
        );

    for (const mode of ["select", "draw", "line", "vibrato"] as const) {
        const state = hydrate(mode);
        assertEqual(state.toolMode, mode, `水合 ${mode}：toolMode`);
        if (mode !== "select") {
            assertEqual(state.drawToolMode, mode, `水合 ${mode}：绘制工具`);
        }
    }

    const hydratedBlock = hydrate("select", "block");
    assertEqual(hydratedBlock.toolMode, "select", "水合 block：工具族");
    assertEqual(hydratedBlock.selectToolMode, "block", "水合 block：子模式");
    assertEqual(hydratedBlock.paramBlocksVisible, true, "水合 block：块图层可见");

    const unknownSelection = hydrate("select", "sideways");
    assertEqual(unknownSelection.selectToolMode, "range", "未知选择子模式回落 range");
    assertEqual(unknownSelection.paramBlocksVisible, false, "未知选择子模式不打开图层");

    // 旧配置里的 `paramEditorTool: "block"` 是未知工具族，回落默认绘制工具。
    assertEqual(hydrate("block").toolMode, "draw", "旧 block 工具值回落默认绘制");
    assertEqual(hydrate("sideways").toolMode, "draw", "未知工具族回落默认");
    assertEqual(hydrate(undefined).toolMode, "draw", "缺项回落默认");
    assertEqual(hydrate(null).toolMode, "draw", "null 回落默认");
    assertEqual(hydrate(3).toolMode, "draw", "非字符串回落默认");

    // 非法工具族不得把状态写成"半切换"的样子。
    const wild = hydrate("nonsense");
    assertEqual(wild.toolMode, "draw", "野值：工具族保持默认");
    assertEqual(wild.drawToolMode, "draw", "野值：绘制工具保持默认");
    assertEqual(wild.selectToolMode, "range", "野值：选择子模式保持默认");
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
                {
                    paramEditorTool: persisted.toolMode,
                    paramEditorSelectionMode: persisted.selectToolMode,
                } as UiSettings,
                "req",
                undefined,
            ),
        );
        if (restored.toolMode !== mode) {
            throw new Error(`${mode} 未能从设置恢复，得到 ${restored.toolMode}`);
        }
    }

    const block = reducer(base, setSelectToolMode("block"));
    const restoredBlock = reducer(
        base,
        loadUiSettings.fulfilled(
            {
                paramEditorTool: block.toolMode,
                paramEditorSelectionMode: block.selectToolMode,
            } as UiSettings,
            "req",
            undefined,
        ),
    );
    if (restoredBlock.toolMode !== "select" || restoredBlock.selectToolMode !== "block") {
        throw new Error("选择族 block 子模式未能从设置恢复");
    }
});
