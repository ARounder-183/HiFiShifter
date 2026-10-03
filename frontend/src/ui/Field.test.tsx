// @vitest-environment jsdom
/*
 * 字段行的排布契约。
 *
 * 【为什么必须有】用户报告"参数名与 ComboBox/TextBox 顶对齐、不居中"：整行曾是
 * `items-start`，标签只补了 `pt-0.5`（2px），而标签 19px、控件 31px，实测每行中心
 * 差 6px。
 *
 * 修法是把整行拆成两层（内层只放 [标签][控件] 并 `items-center`，提示/错误移到
 * 外层列再缩进到控件列）。这套结构有**三条不能破的约束**，逐条钉在下面 —— 任何
 * 一条被改回去，都会重新出现"顶对齐"或"提示把标签拖偏"：
 *   1. 标签与控件在同一行，且该行**居中**（不是顶对齐）；
 *   2. 提示/错误**不在**那一行里（多行提示不会把标签带偏）；
 *   3. 标签上**不许**再出现手工偏移（`pt-*`）—— 那正是当初补不够的那个 hack。
 */
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, test } from "vitest";

import { AppField } from "./Field";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function mountField(node: React.ReactNode) {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(node);
    });
    return {
        host,
        unmount: async () => {
            await act(async () => root.unmount());
            host.remove();
        },
    };
}

test("标签与控件同处一个居中行，提示在该行之外", async () => {
    const mounted = await mountField(
        <AppField label="容差" hint="允许差异">
            <input data-testid="ctl" />
        </AppField>,
    );
    try {
        const label = mounted.host.querySelector("label");
        const control = mounted.host.querySelector(".app-field__control");
        const hint = mounted.host.querySelector("span.hs-type-caption");
        expect(label, "没有渲染标签").not.toBeNull();
        expect(control, "没有渲染控件槽").not.toBeNull();
        expect(hint, "没有渲染提示").not.toBeNull();

        const row = label!.parentElement!;
        // 1. 居中，且标签与控件是同一行的两个成员
        expect(row.className).toContain("items-center");
        expect(row).toBe(control!.parentElement);
        // 2. 提示不在这一行里（否则多行提示会把标签在垂直方向带偏）
        expect(row.contains(hint!)).toBe(false);
        // 3. 不再有手工偏移
        expect(label!.className).not.toMatch(/\bpt-/);
    } finally {
        await mounted.unmount();
    }
});

test("提示缩进到控件列起点（与控件左缘对齐）", async () => {
    const mounted = await mountField(
        <AppField label="容差" hint="允许差异">
            <input />
        </AppField>,
    );
    try {
        const hint = mounted.host.querySelector("span.hs-type-caption") as HTMLElement;
        // 缩进量 = 标签列宽（由 AppForm 下发）+ 行内间距，因此只能是 calc + 令牌。
        expect(hint.style.marginLeft).toContain("calc(");
        expect(hint.style.marginLeft).toContain("--qt-space-4");
    } finally {
        await mounted.unmount();
    }
});

test("错误信息优先于提示，且同样在居中行之外", async () => {
    const mounted = await mountField(
        <AppField label="容差" hint="允许差异" error="超出范围">
            <input />
        </AppField>,
    );
    try {
        const label = mounted.host.querySelector("label")!;
        const text = mounted.host.textContent ?? "";
        expect(text).toContain("超出范围");
        expect(text).not.toContain("允许差异");
        expect(
            label.parentElement!.contains(mounted.host.querySelector("span.hs-type-caption")),
        ).toBe(false);
    } finally {
        await mounted.unmount();
    }
});
