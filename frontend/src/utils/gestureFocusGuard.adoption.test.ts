/*
 * ★ 门禁：**在 window 上装 pointermove 的拖拽手势，必须处理窗口失焦**。
 *
 * ## 防的是什么
 *
 * 这类手势的收尾只挂在 `pointerup` / `pointercancel` 上。用户在拖拽途中 Alt+Tab
 * 切走（或最小化）并在窗口外松手时，WebView2 不会把这两个事件送回本窗口 ⇒ 收尾
 * 永不执行 ⇒ 拖拽态与它的视觉残留（悬停环 / 吸附竖线 / 拖拽浮层 / 拖拽指示线）
 * 全部冻在画面上，而且因为 `onPointerMove` 通常不看 `event.buttons`，切回来后
 * **单纯移动鼠标就会继续拖**。
 *
 * 本轮实测命中五处：时间线 clip 悬停环、钢琴卷帘颤音 HUD、停靠系统的拖拽会话 /
 * 浮动窗 / 分隔条 / 侧栏、笔记本图片缩放、颤音对话框的预设重排。它们全是同一个
 * 缺口的重复 —— 而项目早就有现成机制（`utils/gestureFocusGuard` 的
 * `registerDragAbort`，或自行监听 blur/visibilitychange），只是没有被要求接上。
 *
 * 因此这里把"有没有接"变成可判定的文本事实：**凡是在 window 上装 pointermove 的
 * 模块，都必须出现失焦收尾**（三选一：`registerDragAbort(` / `addEventListener("blur"`
 * / `visibilitychange`）。未接的必须登记在 `NON_GESTURE` 里并写明为什么它不是拖拽。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

function walk(dir: string, out: string[] = []): string[] {
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
            if (entry !== "node_modules") walk(full, out);
        } else if (/\.tsx?$/.test(entry)) {
            out.push(full);
        }
    }
    return out;
}

/** 判据：装了 window/document 级 pointermove（拖拽手势的构造特征）。 */
const POINTER_MOVE_MARKER = 'addEventListener("pointermove"';

/** 失焦收尾的三条合法写法。 */
const FOCUS_HANDLING = ["registerDragAbort(", 'addEventListener("blur"', "visibilitychange"];

/**
 * 装了 pointermove 但**不是拖拽手势**的模块：它们没有"手势期间才存在"的视觉状态，
 * 因此不需要失焦收尾。每条都必须写明理由 —— 这份清单是"已审计"的记录，不是静音开关。
 */
const NON_GESTURE: Record<string, string> = {
    [join("src", "components", "AppTooltip.tsx")]:
        "浮标跟随指针（只读定位）：没有拖拽态，且 pointerout / 滚动 / 手势结束都会收起气泡",
    [join("src", "components", "layout", "QuickSearchPopup.tsx")]:
        "只把最近一次指针坐标记进 ref（打开时用它定位），不产生任何视觉状态",
    [join("src", "utils", "penInput.ts")]:
        "被动记录器（capture + passive，只读不拦截），与拖拽无关",
};

function scan(): { offenders: string[]; scanned: number } {
    const offenders: string[] = [];
    let scanned = 0;
    for (const file of walk(join("src"))) {
        if (/\.test\.tsx?$/.test(file)) continue;
        const source = readFileSync(file, "utf8");
        if (!source.includes(POINTER_MOVE_MARKER)) continue;
        scanned += 1;
        if (file in NON_GESTURE) continue;
        if (FOCUS_HANDLING.some((marker) => source.includes(marker))) continue;
        offenders.push(`  ${file}`);
    }
    return { offenders, scanned };
}

describe("拖拽手势的失焦收尾门禁", () => {
    it("★ 自检：确实扫到了装了 pointermove 的模块（否则下面的断言是空转）", () => {
        expect(scan().scanned).toBeGreaterThanOrEqual(10);
    });

    it("★ 每个装 window pointermove 的模块都必须有失焦收尾", () => {
        const { offenders } = scan();
        expect(
            offenders.length === 0
                ? []
                : [
                      "以下模块在 window 上装了 pointermove，但拖拽收尾只挂 pointerup/pointercancel：",
                      "窗口失焦（Alt+Tab / 最小化）后拖拽态与视觉残留会冻在画面上。",
                      "接 `registerDragAbort`（utils/gestureFocusGuard），或自行监听 blur / visibilitychange；",
                      "确认不是拖拽手势的，登记进本文件的 NON_GESTURE 并写明理由。",
                      ...offenders,
                  ].join("\n"),
        ).toEqual([]);
    });

    it("豁免清单不得包含已接上失焦收尾的模块（避免清单悄悄失效）", () => {
        const stale: string[] = [];
        for (const file of Object.keys(NON_GESTURE)) {
            const source = readFileSync(file, "utf8");
            if (FOCUS_HANDLING.some((marker) => source.includes(marker))) stale.push(`  ${file}`);
        }
        expect(
            stale.length === 0
                ? []
                : ["以下豁免项其实已经接了失焦收尾，请从 NON_GESTURE 移除：", ...stale].join("\n"),
        ).toEqual([]);
    });
});
