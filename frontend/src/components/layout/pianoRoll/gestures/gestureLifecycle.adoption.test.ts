/*
 * ★ 门禁：**装了指针捕获的手势，必须自己收尾**；**每个工具都必须有归属分支**。
 *
 * ## 防的是什么
 *
 * `710ff764`（把指针按下拆成「选择分支 / 颤音分支」两个模块）把自由绘制那一支
 * **整块漏掉了**：没有 `pointermove`、没有 `pointerup`、从不 `commitStroke`。
 * 提交信息里的等价性验证是真的 —— 它比对的是"搬走的那两块"，而消失的那一块
 * 根本不在比对集合里。
 *
 * 上一轮的等价性纪律对"搬错了"有效，对"少搬了"天然无效。所以这里补两条**结构事实**：
 *
 * 1. 一个模块只要调用 `setPointerCapture`，就必须同时注册 `pointermove` /
 *    `pointerup` / `pointercancel`，并按同样数量注销 —— "捕获了却不收尾"在结构上
 *    不可能通过；
 * 2. 指针按下的分派器必须对**每一个**工具分支都有调用点，且映射是穷举的纯函数
 *    （`pointerDownBranchFor`，见 `pointerDownBranchFor.test.ts`）。
 *
 * 这两条都不关心"代码搬没搬对"，只关心"手势有没有收尾""工具有没有归属"。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { pointerDownBranchFor } from "./canvasPointerDown";

const GESTURES_DIR = join("src", "components", "layout", "pianoRoll", "gestures");
const DISPATCHER = join(GESTURES_DIR, "canvasPointerDown.ts");

function walk(dir: string, out: string[] = []): string[] {
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
            if (entry !== "node_modules") walk(full, out);
        } else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) {
            out.push(full);
        }
    }
    return out;
}

/**
 * 一个模块是否把指针手势交给了协议（`startPointerGesture`）。
 *
 * 【为什么要认它】协议本身就注册并注销 move / up / cancel，因此"调了协议"与
 * "手写了那六行"是同一条不变量的两种实现。不认它，门禁会把迁移过的模块当成
 * "没有手势"而**静默停止覆盖** —— 那比误报更糟。
 */
const USES_PROTOCOL = "startPointerGesture(";

/** 数某个字面量在源码里出现的次数。 */
function count(source: string, needle: string): number {
    let total = 0;
    let at = source.indexOf(needle);
    while (at >= 0) {
        total += 1;
        at = source.indexOf(needle, at + needle.length);
    }
    return total;
}

describe("指针手势的生命周期门禁", () => {
    it("★ 自检：确实扫到了若干接管指针的模块（否则下面的断言是空转）", () => {
        const files = walk(GESTURES_DIR);
        const capturing = files.filter((file) => {
            const source = readFileSync(file, "utf8");
            return source.includes("setPointerCapture") || source.includes(USES_PROTOCOL);
        });
        expect(files.length).toBeGreaterThanOrEqual(8);
        // 迁移到协议之后仍有 11 个模块接管指针；门槛留 8 是为了容忍文件增删。
        expect(capturing.length).toBeGreaterThanOrEqual(8);
    });

    it("★ 每个接管指针的模块都注册并注销了 move / up / cancel", () => {
        const offenders: string[] = [];
        for (const file of walk(GESTURES_DIR)) {
            const source = readFileSync(file, "utf8");
            const captures = source.includes("setPointerCapture");
            const delegates = source.includes(USES_PROTOCOL);
            if (!captures && !delegates) continue;
            // 交给协议的模块由协议保证成对；这里只要求它确实交了。
            if (delegates) continue;
            for (const event of ["pointermove", "pointerup", "pointercancel"]) {
                const added = count(source, `addEventListener("${event}"`);
                const removed = count(source, `removeEventListener("${event}"`);
                if (added === 0) {
                    offenders.push(`  ${file}: 捕获了指针却没有注册 ${event}`);
                } else if (removed < added) {
                    offenders.push(`  ${file}: 注册了 ${added} 次 ${event}，只注销 ${removed} 次`);
                }
            }
        }
        expect(
            offenders.length === 0
                ? []
                : [
                      "以下模块调用了 setPointerCapture，但手势没有完整收尾：",
                      "缺 pointerup / pointercancel 意味着引用永远清不掉、live 编辑永远不落库。",
                      ...offenders,
                  ].join("\n"),
        ).toEqual([]);
    });

    it("★ 指针按下的分派器对每个分支都有调用点", () => {
        const source = readFileSync(DISPATCHER, "utf8");
        const missing = [
            ["handleSelectPointerDown(", "选择工具"],
            ["handleVibratoPointerDown(", "直线 / 颤音工具"],
            ["handleFreehandPointerDown(", "自由绘制 / 擦除"],
        ]
            .filter(([marker]) => !source.includes(marker))
            .map(([, label]) => `  ${label}`);
        expect(
            missing.length === 0
                ? []
                : [
                      "以下工具分支在 canvasPointerDown.ts 里没有调用点：",
                      "（710ff764 就是这样丢掉自由绘制的：模块没建、调用也没留）",
                      ...missing,
                  ].join("\n"),
        ).toEqual([]);
    });

    it("★ 工具到分支的映射是穷举的", () => {
        expect(pointerDownBranchFor("select")).toBe("select");
        expect(pointerDownBranchFor("line")).toBe("vibrato");
        expect(pointerDownBranchFor("vibrato")).toBe("vibrato");
        expect(pointerDownBranchFor("draw")).toBe("freehand");
        // 未知工具名也必须落到一个真实分支，不能掉进"什么都不做"。
        expect(pointerDownBranchFor("some-future-tool")).toBe("freehand");
    });
});
