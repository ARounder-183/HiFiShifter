/**
 * 「选项」菜单里的分割过渡设置入口。
 *
 * 【为什么是源码级断言】`MenuBar` 没有测试宿主（2000+ 行、二十多个来自 `App` 的
 * 回调，挂起来的成本远大于它守住的逻辑）。这条入口的全部内容就是三处接线：菜单项
 * 置位 → 对话框受同一个状态控制 → 对话框真的被渲染。少任何一处，症状都是
 * "菜单里有这一项，点了什么都不发生" —— 而这正是源码级断言能抓住的那一类。
 *
 * 与 `tests/timelineViewportGates.test.ts` 同一手法：门禁读源码，不装宿主。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const source = readFileSync(join("src", "components", "layout", "MenuBar.tsx"), "utf8");

test("选项菜单有「分割过渡设置」项，且打开的是同一个对话框", () => {
    // 菜单项：置位打开状态。
    expect(source).toContain("onSelect={() => setSplitTransitionSettingsOpen(true)}");
    // 菜单文案复用对话框标题（与吸附/网格、参数块、搜索设置同一口径）。
    expect(source).toContain('tf("split_transition_settings_title")');
    // 对话框渲染，且 open 受同一个状态控制。
    expect(source).toMatch(
        /<SplitTransitionSettingsDialog\s+open=\{splitTransitionSettingsOpen\}\s+onOpenChange=\{setSplitTransitionSettingsOpen\}/,
    );
});
