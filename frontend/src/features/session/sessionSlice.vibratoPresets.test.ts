import { test } from "vitest";

import reducer, {
    cycleActiveVibratoPreset,
    removeVibratoPreset,
    reorderVibratoPreset,
    setActiveVibratoPreset,
    upsertVibratoPreset,
} from "./sessionSlice.ts";
import {
    SYSTEM_VIBRATO_PRESETS,
    DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
} from "../vibrato/systemPresets.ts";
import { MAX_VIBRATO_PRESETS, sanitizeVibratoPreset } from "../vibrato/vibratoPresets.ts";
import type { VibratoPreset } from "../vibrato/vibratoTypes.ts";

/**
 * 颤音预设的切片行为：
 * - 系统预设（`builtin.*`）不许混进用户列表，否则"系统预设只读"失效；
 * - 删除预设后活动 id 必须迁移到仍然存在的预设上，不能悬空；
 * - 数量上限在手改配置的情况下也要守住。
 */
test("features/session/sessionSlice.vibratoPresets.test.ts vibrato preset reducers", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const userPreset = (id: string, extra: Partial<VibratoPreset> = {}) =>
        sanitizeVibratoPreset({ id, name: id, ...extra });

    const base = reducer(undefined, { type: "@@INIT" });
    assertEqual(base.vibratoPresets.length, 0, "初始没有用户预设");
    assertEqual(
        base.activeVibratoPresetId,
        DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        "初始活动预设是出厂默认",
    );

    // 新增
    const added = reducer(base, upsertVibratoPreset(userPreset("custom_a")));
    assertEqual(added.vibratoPresets.length, 1, "新增一项");
    assertEqual(added.vibratoPresets[0]?.id, "custom_a", "新增的 id");

    // 同 id 覆盖而不是追加
    const overwritten = reducer(
        added,
        upsertVibratoPreset(userPreset("custom_a", { depthCents: 77 })),
    );
    assertEqual(overwritten.vibratoPresets.length, 1, "同 id 覆盖不追加");
    assertEqual(overwritten.vibratoPresets[0]?.depthCents, 77, "覆盖后的深度");

    // 系统预设 id 被拒（用户列表里混进 builtin. 会让只读保证失效）
    const rejected = reducer(added, upsertVibratoPreset(userPreset("builtin.natural")));
    assertEqual(rejected.vibratoPresets.length, 1, "builtin 前缀被拒绝");

    // 活动预设切换与环绕
    const withTwo = reducer(added, upsertVibratoPreset(userPreset("custom_b")));
    const cycled = reducer(withTwo, cycleActiveVibratoPreset(1));
    assertEqual(cycled.activeVibratoPresetId, SYSTEM_VIBRATO_PRESETS[1]?.id, "下一个预设");
    const cycledBack = reducer(cycled, cycleActiveVibratoPreset(-1));
    assertEqual(
        cycledBack.activeVibratoPresetId,
        DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        "上一个预设回到首项",
    );

    // 从首项向前环绕到用户段末尾
    const wrapped = reducer(withTwo, cycleActiveVibratoPreset(-1));
    assertEqual(wrapped.activeVibratoPresetId, "custom_b", "首项向前环绕到末尾");

    const explicit = reducer(withTwo, setActiveVibratoPreset("custom_a"));
    assertEqual(explicit.activeVibratoPresetId, "custom_a", "显式设定活动预设");

    // 删除活动预设 → id 迁移到滑进同一位置的那一项，不悬空
    const afterRemoval = reducer(explicit, removeVibratoPreset("custom_a"));
    assertEqual(afterRemoval.vibratoPresets.length, 1, "删除后剩一项");
    assertEqual(afterRemoval.activeVibratoPresetId, "custom_b", "活动 id 迁移到剩余项");
    assertEqual(
        afterRemoval.vibratoPresets.some((preset) => preset.id === "custom_a"),
        false,
        "被删的预设不再存在",
    );

    // 删除非活动预设不影响活动 id
    const keepActive = reducer(withTwo, removeVibratoPreset("custom_b"));
    assertEqual(
        keepActive.activeVibratoPresetId,
        DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        "删非活动项时活动 id 不变",
    );

    // 排序
    const three = reducer(
        reducer(added, upsertVibratoPreset(userPreset("custom_b"))),
        upsertVibratoPreset(userPreset("custom_c")),
    );
    const reordered = reducer(three, reorderVibratoPreset({ id: "custom_c", toIndex: 0 }));
    assertEqual(
        reordered.vibratoPresets.map((preset) => preset.id).join(","),
        "custom_c,custom_a,custom_b",
        "排序结果",
    );

    // 上限：手改配置塞进超量预设时不再追加
    let capped = reducer(undefined, { type: "@@INIT" });
    for (let i = 0; i < MAX_VIBRATO_PRESETS + 5; i += 1) {
        capped = reducer(capped, upsertVibratoPreset(userPreset(`custom_${i}`)));
    }
    assertEqual(capped.vibratoPresets.length, MAX_VIBRATO_PRESETS, "用户预设数量被上限截住");

    // 冗余：往用户列表里塞 builtin 前缀的项不会被采纳
    const guarded = reducer(
        base,
        upsertVibratoPreset(sanitizeVibratoPreset({ id: "builtin.deep" })),
    );
    assertEqual(guarded.vibratoPresets.length, 0, "builtin 前缀不写入用户列表");
});
