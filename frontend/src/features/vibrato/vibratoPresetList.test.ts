import { describe, expect, test } from "vitest";

import {
    DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
    SYSTEM_VIBRATO_PRESETS,
    builtinVibratoPresetId,
    BUILTIN_VIBRATO_ORDER,
} from "./systemPresets";
import {
    activeIdAfterRemoval,
    cycleVibratoPresetId,
    effectiveBuiltinPresetOrder,
    enabledVibratoPresets,
    findVibratoPreset,
    moveItemToIndex,
    reorderBuiltinPresetIds,
    reorderUserVibratoPresets,
    resolveActiveVibratoPreset,
    resolveVibratoPresets,
    systemVibratoPreset,
} from "./vibratoPresetList";
import { sanitizeVibratoPreset } from "./vibratoPresets";

const userPreset = (id: string, name = id) => sanitizeVibratoPreset({ id, name });

describe("resolveVibratoPresets", () => {
    test("系统预设在所有用户预设之前", () => {
        const { all } = resolveVibratoPresets([userPreset("custom_a")]);
        expect(all.slice(0, SYSTEM_VIBRATO_PRESETS.length).map((preset) => preset.id)).toEqual(
            SYSTEM_VIBRATO_PRESETS.map((preset) => preset.id),
        );
        expect(all[all.length - 1].id).toBe("custom_a");
    });

    test("用户列表为空 / 缺失时只有系统预设", () => {
        expect(resolveVibratoPresets(null).user).toEqual([]);
        expect(resolveVibratoPresets(undefined).all).toHaveLength(SYSTEM_VIBRATO_PRESETS.length);
    });

    test("混进用户列表的 builtin. 前缀 id 被剔除", () => {
        // 手改配置可能把系统预设塞进用户段；那会让"系统预设只读"失效。
        const { user } = resolveVibratoPresets([
            userPreset("builtin.natural"),
            userPreset("custom_a"),
        ]);
        expect(user.map((preset) => preset.id)).toEqual(["custom_a"]);
    });

    test("用户预设被规整后才返回", () => {
        const { user } = resolveVibratoPresets([
            { id: "custom_a", name: "A", depthCents: 99_999 } as never,
        ]);
        expect(user[0].depthCents).toBeLessThanOrEqual(1200);
    });

    test("系统预设对象标识稳定（供 React 依赖比较）", () => {
        expect(resolveVibratoPresets([]).system).toBe(SYSTEM_VIBRATO_PRESETS);
    });

    test("系统预设有 12 个且 id 唯一", () => {
        const ids = SYSTEM_VIBRATO_PRESETS.map((preset) => preset.id);
        expect(ids).toHaveLength(BUILTIN_VIBRATO_ORDER.length);
        expect(new Set(ids).size).toBe(ids.length);
    });

    test("「直线」排在出厂顺序首位，也是默认活动预设", () => {
        expect(BUILTIN_VIBRATO_ORDER[0]).toBe("straight");
        expect(SYSTEM_VIBRATO_PRESETS[0]?.id).toBe(builtinVibratoPresetId("straight"));
        expect(DEFAULT_ACTIVE_VIBRATO_PRESET_ID).toBe(builtinVibratoPresetId("straight"));
    });
});

describe("系统预设表", () => {
    test("每个键都能构造出预设，且 id 前缀正确", () => {
        for (const key of BUILTIN_VIBRATO_ORDER) {
            const preset = systemVibratoPreset(key);
            expect(preset.id).toBe(builtinVibratoPresetId(key));
            expect(preset.builtin).toBe(true);
        }
    });

    test("默认活动预设存在", () => {
        expect(
            findVibratoPreset(SYSTEM_VIBRATO_PRESETS, DEFAULT_ACTIVE_VIBRATO_PRESET_ID),
        ).toBeDefined();
    });

    test("直线预设深度为 0（直线/颤音工具共用一条代码路径）", () => {
        expect(systemVibratoPreset("straight").depthCents).toBe(0);
    });

    /*
     * 出厂预设**不再携带**「摆放方式」。
     *
     * 它已抽离成添加颤音时的参数（存在设置里，见 `BaselineMode`）：同一个预设在
     * 不同摆放方式下都该能用，因此"某个预设天生就叠在原曲线上"这种绑定不存在了。
     */
    test("出厂预设不携带摆放方式（它是添加颤音的参数）", () => {
        for (const preset of SYSTEM_VIBRATO_PRESETS) {
            expect("baseline" in preset, `${preset.id} 不该带 baseline`).toBe(false);
        }
        expect(systemVibratoPreset("breath").depthCents).toBeGreaterThan(0);
    });

    test("系统预设被冻结：就地改写会抛错", () => {
        const preset = SYSTEM_VIBRATO_PRESETS[0];
        expect(() => {
            (preset as { depthCents: number }).depthCents = 999;
        }).toThrow();
    });
});

describe("resolveActiveVibratoPreset", () => {
    const { all } = resolveVibratoPresets([userPreset("custom_a")]);

    test("按 id 命中", () => {
        expect(resolveActiveVibratoPreset(all, "custom_a").id).toBe("custom_a");
    });

    test("id 找不到时回落到出厂默认，且不改写传入的 id", () => {
        expect(resolveActiveVibratoPreset(all, "custom_deleted").id).toBe(
            DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        );
        expect(resolveActiveVibratoPreset(all, null).id).toBe(DEFAULT_ACTIVE_VIBRATO_PRESET_ID);
        expect(resolveActiveVibratoPreset(all, undefined).id).toBe(
            DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        );
    });

    test("空列表也能返回一个可用预设（不抛异常）", () => {
        expect(resolveActiveVibratoPreset([], "whatever").id).toBe(
            DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        );
    });
});

describe("cycleVibratoPresetId", () => {
    const { all } = resolveVibratoPresets([userPreset("custom_a")]);

    test("前后移动一步", () => {
        const first = all[0].id;
        const second = all[1].id;
        expect(cycleVibratoPresetId(all, first, 1)).toBe(second);
        expect(cycleVibratoPresetId(all, second, -1)).toBe(first);
    });

    test("两端环绕（下一个越过末尾回到开头）", () => {
        const last = all[all.length - 1].id;
        expect(cycleVibratoPresetId(all, last, 1)).toBe(all[0].id);
        expect(cycleVibratoPresetId(all, all[0].id, -1)).toBe(last);
    });

    test("未知 id 从默认活动预设起步（不返回 null）", () => {
        // 未知 id 会回落到默认活动预设（自然），而不是列表头部 —— 后者只是
        // 出厂顺序恰好把自然排在最前时的巧合。
        const fallbackIndex = all.findIndex(
            (preset) => preset.id === DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        );
        expect(fallbackIndex).toBeGreaterThanOrEqual(0);
        expect(cycleVibratoPresetId(all, "custom_missing", 1)).toBe(
            all[(fallbackIndex + 1) % all.length].id,
        );
    });

    test("空列表返回 null", () => {
        expect(cycleVibratoPresetId([], "x", 1)).toBeNull();
    });

    test("单项列表原地环绕", () => {
        const single = [userPreset("custom_only")];
        expect(cycleVibratoPresetId(single, "custom_only", 1)).toBe("custom_only");
        expect(cycleVibratoPresetId(single, "custom_only", -1)).toBe("custom_only");
    });
});

describe("activeIdAfterRemoval", () => {
    const { all } = resolveVibratoPresets([userPreset("custom_a"), userPreset("custom_b")]);

    test("删的不是当前项时活动 id 不变", () => {
        expect(activeIdAfterRemoval(all, "custom_a", "custom_b")).toBe("custom_a");
    });

    test("删的是当前项时落到滑进同一位置的那一项上", () => {
        // custom_a 位于用户段首位，删除后 custom_b 滑进该位置。
        expect(activeIdAfterRemoval(all, "custom_a", "custom_a")).toBe("custom_b");
    });

    test("删掉中间项时落到原位置上的那一项", () => {
        const middle = all[2];
        expect(activeIdAfterRemoval(all, middle.id, middle.id)).toBe(all[3].id);
    });

    test("删到只剩系统预设时回落到出厂默认", () => {
        const onlySystem = [...SYSTEM_VIBRATO_PRESETS];
        expect(activeIdAfterRemoval(onlySystem, DEFAULT_ACTIVE_VIBRATO_PRESET_ID, "custom_x")).toBe(
            DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        );
    });

    test("列表被删空时回落到出厂默认", () => {
        expect(activeIdAfterRemoval([], "custom_a", "custom_a")).toBe(
            DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
        );
    });
});

describe("reorderUserVibratoPresets", () => {
    const list = [userPreset("a"), userPreset("b"), userPreset("c")];

    test("向后移动", () => {
        expect(reorderUserVibratoPresets(list, "a", 2).map((p) => p.id)).toEqual(["b", "c", "a"]);
    });

    test("向前移动", () => {
        expect(reorderUserVibratoPresets(list, "c", 0).map((p) => p.id)).toEqual(["c", "a", "b"]);
    });

    test("越界索引被钳制到两端", () => {
        expect(reorderUserVibratoPresets(list, "a", 99).map((p) => p.id)).toEqual(["b", "c", "a"]);
        expect(reorderUserVibratoPresets(list, "c", -99).map((p) => p.id)).toEqual(["c", "a", "b"]);
    });

    test("原地不动时返回等值新数组", () => {
        const next = reorderUserVibratoPresets(list, "b", 1);
        expect(next.map((p) => p.id)).toEqual(["a", "b", "c"]);
        expect(next).not.toBe(list);
    });

    test("id 不存在时原样返回（不抛异常）", () => {
        expect(reorderUserVibratoPresets(list, "zzz", 0).map((p) => p.id)).toEqual(["a", "b", "c"]);
    });

    test("不修改传入的数组", () => {
        const snapshot = list.map((p) => p.id);
        reorderUserVibratoPresets(list, "a", 2);
        expect(list.map((p) => p.id)).toEqual(snapshot);
    });
});

describe("findVibratoPreset", () => {
    test("空 id 直接返回 undefined（不误命中）", () => {
        const { all } = resolveVibratoPresets([]);
        expect(findVibratoPreset(all, null)).toBeUndefined();
        expect(findVibratoPreset(all, undefined)).toBeUndefined();
        expect(findVibratoPreset(all, "")).toBeUndefined();
    });
});

describe("enabledVibratoPresets（停用过滤）", () => {
    const { all } = resolveVibratoPresets([userPreset("custom_a"), userPreset("custom_b")]);

    test("没有停用名单时原样返回全部", () => {
        expect(enabledVibratoPresets(all, [])).toEqual(all);
        expect(enabledVibratoPresets(all, null)).toEqual(all);
        expect(enabledVibratoPresets(all, undefined)).toEqual(all);
    });

    test("剔除被停用的条目，顺序不变", () => {
        const kept = enabledVibratoPresets(all, ["custom_a"]);
        expect(kept.map((preset) => preset.id)).not.toContain("custom_a");
        expect(kept.length).toBe(all.length - 1);
        // 其余顺序保持。
        expect(kept.map((preset) => preset.id)).toEqual(
            all.filter((preset) => preset.id !== "custom_a").map((preset) => preset.id),
        );
    });

    test("系统预设同样可被停用", () => {
        const kept = enabledVibratoPresets(all, [builtinVibratoPresetId("straight")]);
        expect(kept.some((preset) => preset.id === builtinVibratoPresetId("straight"))).toBe(false);
    });

    test("名单里的未知 id 不影响结果", () => {
        expect(enabledVibratoPresets(all, ["does_not_exist"])).toEqual(all);
    });

    test("全部停用时返回空列表（调用方据此不做切换）", () => {
        expect(
            enabledVibratoPresets(
                all,
                all.map((preset) => preset.id),
            ),
        ).toEqual([]);
    });

    test("不修改传入的列表", () => {
        const before = [...all];
        enabledVibratoPresets(all, ["custom_a"]);
        expect(all).toEqual(before);
    });
});

describe("effectiveBuiltinPresetOrder（系统预设顺序）", () => {
    const defaultIds = SYSTEM_VIBRATO_PRESETS.map((preset) => preset.id);

    test("没有自定义顺序时就是出厂顺序", () => {
        expect(effectiveBuiltinPresetOrder([])).toEqual(defaultIds);
        expect(effectiveBuiltinPresetOrder(null)).toEqual(defaultIds);
        expect(effectiveBuiltinPresetOrder(undefined)).toEqual(defaultIds);
    });

    test("按持久化顺序取有效项，未提到的按出厂顺序补在后面", () => {
        const moved = [builtinVibratoPresetId("deep"), builtinVibratoPresetId("straight")];
        const result = effectiveBuiltinPresetOrder(moved);
        expect(result.slice(0, 2)).toEqual(moved);
        expect(result).toHaveLength(defaultIds.length);
        expect(new Set(result)).toEqual(new Set(defaultIds));
    });

    test("无效项与重复项被剔除（旧配置 / 手改配置都能收敛）", () => {
        const soft = builtinVibratoPresetId("soft");
        const result = effectiveBuiltinPresetOrder([soft, "builtin.gone", soft]);
        expect(result[0]).toBe(soft);
        expect(result).toHaveLength(defaultIds.length);
        expect(result.filter((id) => id === soft)).toHaveLength(1);
    });
});

describe("reorderBuiltinPresetIds / moveItemToIndex", () => {
    test("移动到指定位置", () => {
        expect(reorderBuiltinPresetIds(["a", "b", "c"], "a", 2)).toEqual(["b", "c", "a"]);
        expect(reorderBuiltinPresetIds(["a", "b", "c"], "c", 0)).toEqual(["c", "a", "b"]);
    });

    test("越界钳制；未知 id 原样返回", () => {
        expect(reorderBuiltinPresetIds(["a", "b", "c"], "a", 99)).toEqual(["b", "c", "a"]);
        expect(reorderBuiltinPresetIds(["a", "b", "c"], "zz", 0)).toEqual(["a", "b", "c"]);
    });

    test("moveItemToIndex 对空数组与无效下标安全（两份排序共用它）", () => {
        expect(moveItemToIndex([], 0, 0)).toEqual([]);
        expect(moveItemToIndex(["a"], 0, 0)).toEqual(["a"]);
        expect(moveItemToIndex(["a", "b"], -1, 0)).toEqual(["a", "b"]);
        expect(moveItemToIndex(["a", "b"], 5, 0)).toEqual(["a", "b"]);
    });
});

describe("resolveVibratoPresets 的自定义系统顺序", () => {
    test("默认顺序返回那份稳定数组（React 依赖比较靠它）", () => {
        expect(resolveVibratoPresets([]).system).toBe(SYSTEM_VIBRATO_PRESETS);
        expect(resolveVibratoPresets([], []).system).toBe(SYSTEM_VIBRATO_PRESETS);
        expect(
            resolveVibratoPresets(
                [],
                SYSTEM_VIBRATO_PRESETS.map((preset) => preset.id),
            ).system,
        ).toBe(SYSTEM_VIBRATO_PRESETS);
    });

    test("自定义顺序生效，all 里的系统段也跟着变", () => {
        const order = [SYSTEM_VIBRATO_PRESETS[3].id, SYSTEM_VIBRATO_PRESETS[0].id];
        const resolved = resolveVibratoPresets([], order);
        expect(resolved.system[0].id).toBe(order[0]);
        expect(resolved.system[1].id).toBe(order[1]);
        expect(resolved.all.slice(0, 2).map((preset) => preset.id)).toEqual(order);
        // 系统段整体仍是同一批预设，只是顺序不同。
        expect(resolved.system).toHaveLength(SYSTEM_VIBRATO_PRESETS.length);
    });
});
