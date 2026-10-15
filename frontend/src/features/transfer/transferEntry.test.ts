// @vitest-environment jsdom
/**
 * Transfer **入口可见性**的契约。
 *
 * 【为什么这条必须有测试】用户实测：在不支持 ARA 的宿主里看不到任何 Transfer 入口。
 * 原因是入口当时只有"窗口打开时自动浮出"一条路，而那个判定依赖形态 —— 形态要等
 * `setActive` 之后 5 秒宽限期才落定，bootstrap 又是窗口创建那一刻算的。两个来源都必须
 * 算数：bootstrap（窗口开得晚）与状态快照（后端**立刻**知道自己是哪份会话）。
 */
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

/** 状态快照里的形态；由每个用例改写。 */
const applyState = vi.hoisted(() => ({
    current: null as { mode?: "transfer" | "ara" } | null,
}));

vi.mock("../ara/pluginApplyStore", () => ({
    getPluginApplySnapshot: () => ({ state: applyState.current }),
    subscribePluginApply: () => () => undefined,
}));

import { isTransferInstance, showsTransferEntry } from "./transferEntry";

function bootstrap(hostKind?: string): void {
    (window as unknown as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__ =
        hostKind === undefined ? undefined : { version: 1, hostKind };
}

beforeEach(() => {
    applyState.current = null;
    bootstrap(undefined);
});

afterEach(() => {
    bootstrap(undefined);
});

describe("「视图」菜单里的 Transfer 入口", () => {
    test("插件形态下**恒可见** —— 不依赖形态是否已落定", () => {
        // 【为什么这条是那个 bug 的回归】入口此前按"本实例是不是 Transfer 实例"显示，
        // 而那个判定要等 setActive 之后 5 秒宽限期；窗口开得早的会话里入口**完全不存在**。
        bootstrap("ara_only");
        applyState.current = null;
        expect(showsTransferEntry()).toBe(true);
        applyState.current = { mode: "ara" };
        expect(showsTransferEntry(), "ARA 宿主里也保留入口，由面板说明'不需要 Transfer'").toBe(
            true,
        );
    });

    test("独立 App 没有 Transfer 通路", () => {
        expect(showsTransferEntry()).toBe(false);
    });
});

describe("Transfer 面板的自动浮出判据", () => {
    test("独立 App 不浮出", () => {
        expect(isTransferInstance()).toBe(false);
    });

    test("bootstrap 已经落定为 transfer ⇒ 可见", () => {
        bootstrap("transfer");
        expect(isTransferInstance()).toBe(true);
    });

    test("bootstrap 还是 ara_only（窗口开得早）但快照说 transfer ⇒ 仍然可见", () => {
        bootstrap("ara_only");
        applyState.current = { mode: "transfer" };
        expect(isTransferInstance()).toBe(true);
    });

    test("两者都说 ARA ⇒ 不显示（点开只会看到一个空面板）", () => {
        bootstrap("ara_only");
        applyState.current = { mode: "ara" };
        expect(isTransferInstance()).toBe(false);
    });

    test("原生扩展宿主（reaper）不显示", () => {
        bootstrap("reaper");
        expect(isTransferInstance()).toBe(false);
    });
});
