/*
 * ★ 门禁：**指针离开轨道区 = 所有悬停态归零**。
 *
 * ## 防的是什么
 *
 * 悬停态（提示环、浮标内容）只在"命中身份变化"时才更新 —— 指针移出容器后不再有
 * `pointermove`，因此**离开时不清就永远不清**：clip 悬停环会一直亮在画面上
 * （用户报告的那类"高亮没清掉"）。淡变通道早就这么做了（`onPointerLeave` 里
 * `onFadeHover(null)`），clip 通道是漏的那一个。
 *
 * 宿主里有三个模块级悬停变量，全部必须在 `onPointerLeave` 里复位：
 * - `hoveredClipId` —— 提示环（细节层据此画 1px 深色描边）；
 * - `lastClipHoverKey` —— clip 浮标的去重键；清掉它才能让**再次进入**同一 clip
 *   时重新发布一次（读数不会停在离开前的旧值上）；
 * - `lastFadeHoverKey` —— 淡变浮标的去重键（同上）。
 *
 * 用文本事实做门禁：这三个名字必须都出现在 `onPointerLeave` 的函数体里。
 * 新增悬停通道时，本文件与 `onPointerLeave` 一起改 —— 那正是希望被提醒的时刻。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE = readFileSync(
    join("src", "components", "layout", "timeline", "kernel", "host", "timelineKernelHost.ts"),
    "utf8",
);

/** 去掉注释：注释里引用的符号名不得计入文本事实。 */
function stripComments(text: string): string {
    return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

const CODE = stripComments(SOURCE);

/** 取出 `function onPointerLeave(): void { … }` 的函数体。 */
function pointerLeaveBody(): string {
    const start = CODE.indexOf("function onPointerLeave(");
    expect(start, "找不到 onPointerLeave —— 函数改名后请同步本门禁").toBeGreaterThanOrEqual(0);
    const open = CODE.indexOf("{", start);
    let depth = 0;
    for (let at = open; at < CODE.length; at += 1) {
        if (CODE[at] === "{") depth += 1;
        else if (CODE[at] === "}") {
            depth -= 1;
            if (depth === 0) return CODE.slice(open, at + 1);
        }
    }
    throw new Error("onPointerLeave 函数体未闭合");
}

describe("指针离开时的悬停态复位", () => {
    it("★ onPointerLeave 必须复位全部悬停通道（clip 环 / clip 浮标键 / 淡变浮标键）", () => {
        const body = pointerLeaveBody();
        const missing = ["hoveredClipId", "lastClipHoverKey", "lastFadeHoverKey"].filter(
            (name) => !body.includes(name),
        );
        expect(
            missing.length === 0
                ? []
                : [
                      "onPointerLeave 没有复位以下悬停态（指针移出后不再有 pointermove，",
                      "悬停环 / 浮标会永久留在画面上）：",
                      ...missing.map((name) => `  ${name}`),
                  ].join("\n"),
        ).toEqual([]);
    });

    it("★ 清掉去重键时必须同时收起浮标内容（否则键与内容分叉）", () => {
        const body = pointerLeaveBody();
        // 两个通道各要有一处 on*Hover(null) 与键的复位配对。
        expect(body).toContain('lastFadeHoverKey = ""');
        expect(body).toContain('lastClipHoverKey = ""');
        expect(body).toContain("onFadeHover?.(null");
        expect(body).toContain("onClipHover?.(null");
    });
});
