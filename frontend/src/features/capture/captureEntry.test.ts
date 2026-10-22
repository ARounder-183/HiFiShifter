// @vitest-environment jsdom
/**
 * Capture **入口可见性**的契约。
 *
 * 【为什么这条必须有测试】用户实测：在不支持 ARA 的宿主里看不到任何 Capture 入口。
 * 原因是入口当时按"本实例是不是 Capture"显示，而那个判定要等 `setActive` 之后 5 秒
 * 宽限期才落定，bootstrap 又是窗口创建那一刻算的 —— 于是窗口开得早的会话里入口
 * **完全不存在**。现在判据只有"在不在插件里"，与形态落没落定无关。
 */
import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { showsCaptureEntry } from "./captureEntry";

function bootstrap(hostKind?: string): void {
    (window as unknown as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__ =
        hostKind === undefined ? undefined : { version: 1, hostKind };
}

beforeEach(() => {
    bootstrap(undefined);
});

afterEach(() => {
    bootstrap(undefined);
});

describe("「视图」菜单里的 Capture 入口", () => {
    test("插件形态下**恒可见** —— 不依赖形态是否已落定", () => {
        // 【为什么这条是那个 bug 的回归】入口此前按"本实例是不是 Capture 实例"显示，
        // 而那个判定要等 setActive 之后 5 秒宽限期；窗口开得早的会话里入口**完全不存在**。
        bootstrap("ara_only");
        expect(showsCaptureEntry()).toBe(true);
        bootstrap("capture");
        expect(showsCaptureEntry()).toBe(true);
        bootstrap("reaper");
        expect(showsCaptureEntry(), "ARA 宿主里也保留入口，由控件说明'不需要 Capture'").toBe(true);
    });

    test("独立 App 没有 Capture 通路", () => {
        expect(showsCaptureEntry()).toBe(false);
    });
});
