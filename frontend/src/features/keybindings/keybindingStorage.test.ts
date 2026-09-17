/**
 * 「拉伸」修饰键拆分的一次性迁移单测（`./keybindingStorage`）。
 *
 * 【要锁住的行为】拆分前 `modifier.clipStretch` 同时服务时间轴 clip 边缘与
 * 参数编辑器选区边缘；拆分后后者改读 `modifier.paramStretch`。用户若改绑过旧
 * 键，升级后必须把旧值继承到新键（否则"设置丢了"，行为与升级前不一致）。
 *
 * 【为什么在 node 环境可测】`keybindingStorage` 只依赖 `localStorage`
 * （全局对象），这里用最小桩替换，并覆盖「标记位粘性」这条易错的不变式。
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
    loadKeybindingOverrides,
    migrateStretchSplit,
    saveKeybindingOverrides,
} from "./keybindingStorage";
import type { KeybindingOverrides } from "./types";

const STORAGE_KEY = "hifishifter.keybindings";
const FLAG = "__stretchSplitMigrated";

/** 最小 localStorage 桩：只实现被用到的三个方法。 */
function installLocalStorageStub(): Map<string, string> {
    const store = new Map<string, string>();
    const stub = {
        getItem: (key: string) => store.get(key) ?? null,
        setItem: (key: string, value: string) => {
            store.set(key, value);
        },
        removeItem: (key: string) => {
            store.delete(key);
        },
    };
    (globalThis as { localStorage?: unknown }).localStorage = stub;
    return store;
}

describe("migrateStretchSplit（纯函数）", () => {
    it("旧「拉伸」覆盖项同时落到两个新动作", () => {
        const raw = { "modifier.clipStretch": { key: "control", modifierOnly: true, ctrl: true } };
        const { overrides, migrated } = migrateStretchSplit(raw);
        expect(migrated).toBe(true);
        expect(overrides["modifier.clipStretch"]).toEqual(raw["modifier.clipStretch"]);
        // 旧值必须继承到参数编辑器动作 —— 这是"设置没丢"的关键。
        expect(overrides["modifier.paramStretch"]).toEqual(raw["modifier.clipStretch"]);
    });

    it("已有标记位时不再迁移（用户的后续重置不会被撤销）", () => {
        const raw = {
            [FLAG]: true,
            "modifier.clipStretch": { key: "shift", modifierOnly: true, shift: true },
        };
        const { overrides, migrated } = migrateStretchSplit(raw);
        expect(migrated).toBe(false);
        expect(overrides["modifier.paramStretch"]).toBeUndefined();
    });

    it("无旧覆盖项时只置标记位，不凭空造覆盖项", () => {
        const { overrides, migrated } = migrateStretchSplit({});
        expect(migrated).toBe(true);
        expect(Object.keys(overrides)).toHaveLength(0);
    });

    it("元数据键不进入覆盖项（否则会被当成 actionId 合并进快捷键表）", () => {
        const { overrides } = migrateStretchSplit({
            [FLAG]: true,
            "playback.toggle": { key: "enter" },
        });
        expect(Object.keys(overrides)).toEqual(["playback.toggle"]);
    });

    it("新动作已有覆盖项时不覆盖用户的新选择", () => {
        const raw = {
            "modifier.clipStretch": { key: "control", modifierOnly: true, ctrl: true },
            "modifier.paramStretch": { key: "shift", modifierOnly: true, shift: true },
        };
        const { overrides } = migrateStretchSplit(raw);
        expect(overrides["modifier.paramStretch"]).toEqual({
            key: "shift",
            modifierOnly: true,
            shift: true,
        });
    });
});

describe("loadKeybindingOverrides / saveKeybindingOverrides", () => {
    let store: Map<string, string>;

    beforeEach(() => {
        store = installLocalStorageStub();
    });

    afterEach(() => {
        delete (globalThis as { localStorage?: unknown }).localStorage;
    });

    it("加载时迁移并回写（带标记位，只发生一次）", () => {
        store.set(
            STORAGE_KEY,
            JSON.stringify({
                "modifier.clipStretch": { key: "control", modifierOnly: true, ctrl: true },
            }),
        );
        const loaded = loadKeybindingOverrides();
        expect(loaded["modifier.paramStretch"]).toBeDefined();

        // 回写后的存储必须已带标记位（否则每次启动都会重放迁移）。
        const persisted = JSON.parse(store.get(STORAGE_KEY) ?? "{}");
        expect(persisted[FLAG]).toBe(true);
        // 且不应该把标记位当成覆盖项暴露给调用方。
        expect(loaded[FLAG as keyof KeybindingOverrides]).toBeUndefined();
    });

    it("【标记位粘性】后续保存不得丢掉标记位（否则迁移会重放）", () => {
        store.set(STORAGE_KEY, JSON.stringify({ "modifier.clipStretch": { key: "control" } }));
        loadKeybindingOverrides(); // 触发迁移 + 回写
        // 模拟用户随后改绑另一个动作（中间件保存路径）。
        saveKeybindingOverrides({ "playback.toggle": { key: "enter" } });
        const persisted = JSON.parse(store.get(STORAGE_KEY) ?? "{}");
        expect(persisted[FLAG]).toBe(true);
        expect(persisted["playback.toggle"]).toEqual({ key: "enter" });
    });

    it("空覆盖项且未迁移过 → 移除存储键", () => {
        saveKeybindingOverrides({});
        expect(store.has(STORAGE_KEY)).toBe(false);
    });

    it("迁移后即使用户清空所有覆盖项也保留标记位（不重放迁移）", () => {
        store.set(STORAGE_KEY, JSON.stringify({ "modifier.clipStretch": { key: "control" } }));
        loadKeybindingOverrides();
        saveKeybindingOverrides({});
        const persisted = JSON.parse(store.get(STORAGE_KEY) ?? "{}");
        expect(persisted[FLAG]).toBe(true);
        // 再次加载：不应重新造出 paramStretch。
        const reloaded = loadKeybindingOverrides();
        expect(reloaded["modifier.paramStretch"]).toBeUndefined();
    });

    it("解析失败 / 非对象载荷 → 空覆盖项且不抛错", () => {
        store.set(STORAGE_KEY, "not json");
        expect(loadKeybindingOverrides()).toEqual({});
        store.set(STORAGE_KEY, JSON.stringify([1, 2, 3]));
        expect(loadKeybindingOverrides()).toEqual({});
    });
});
