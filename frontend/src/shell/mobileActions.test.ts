// 抽屉完整性：**每一条非修饰键动作都必须能在抽屉里被找到**。
//
// 【为什么要一条枚举式断言】抽屉是移动端**唯一**的功能入口。而它的分段来自
// `SECTION_ORDER` 这张手写表：`buildMobileActionSections` 只输出表里列出的
// 分组，表里没有的分组会被**静默丢掉**。这不是假设 —— `paramEditor`（9 条）与
// `pianoRoll`（22 条）就曾整段消失，其中 12 条 `pianoRoll.shiftParam*` 更是
// 没有任何按钮，只有键盘能触发。症状是"手机上找不到平滑 / 移调"，界面上却
// 看不出任何异常，也没有任何日志。
//
// 断言写法与 `ACTION_META` 直接挂钩：以后新增一个动作而忘了给它一个分组，
// 这条会红。
import { describe, expect, it } from "vitest";
import { ACTION_META } from "../features/keybindings/defaultKeybindings";
import type { ActionId } from "../features/keybindings/types";
import { buildMobileActionSections } from "./mobileActions";

/** 形态上**不该**出现在抽屉里的动作（与 `MOBILE_UNAVAILABLE` 同一口径）。 */
const EXPECTED_ABSENT = new Set<string>([
    // 系统剪贴板：移动端没有原生通道。
    "edit.pasteVocalShifter",
    // 宿主相关：移动端没有 DAW。
    "host.handKeyboardToHost",
    // 快速搜索弹窗内部的导航键。
    // 抽屉自己就是搜索面板：再摆一个「打开快速搜索」等于让用户从一个搜索框
    // 跳到另一个搜索框。
    "quickSearch.open",
    "quickSearch.navigate.up",
    "quickSearch.navigate.down",
    "quickSearch.preview",
    "quickSearch.confirm",
    "quickSearch.close",
]);

const sections = buildMobileActionSections(
    (key) => String(key),
    () => null,
);
const rendered = new Set(sections.flatMap((section) => section.items.map((item) => item.id)));

describe("移动端抽屉完整性", () => {
    it("★ 覆盖全部非修饰键动作（除白名单）", () => {
        const missing: string[] = [];
        for (const [id, meta] of Object.entries(ACTION_META) as Array<
            [ActionId, (typeof ACTION_META)[ActionId]]
        >) {
            // 修饰键动作去修饰键行（ModifierRail），不是可执行命令。
            if (meta.modifierOperationType !== undefined) continue;
            if (EXPECTED_ABSENT.has(id)) continue;
            if (!rendered.has(id)) missing.push(id);
        }
        expect(missing, `抽屉里缺失的动作：${missing.join(", ")}`).toEqual([]);
    });

    it("修饰键动作不出现在抽屉里（它们是状态，不是命令）", () => {
        for (const [id, meta] of Object.entries(ACTION_META) as Array<
            [ActionId, (typeof ACTION_META)[ActionId]]
        >) {
            if (meta.modifierOperationType === undefined) continue;
            expect(rendered.has(id), `${id} 不该出现在抽屉里`).toBe(false);
        }
    });

    it("参数编辑器与钢琴卷帘两段真的出现了（这两段曾经整段消失）", () => {
        const ids = sections.map((section) => section.id);
        expect(ids).toContain("paramEditor");
        expect(ids).toContain("pianoRoll");
        const pianoRoll = sections.find((section) => section.id === "pianoRoll");
        expect(pianoRoll?.items.length ?? 0).toBeGreaterThan(0);
    });
});
