/*
 * ★ 门禁：**失焦 / 页面隐藏收尾不得产生"用户没发出的编辑写入"**。
 *
 * ## 防的是什么
 *
 * `finalizeActiveGesture` 的 `synthetic` 参数表示"没有真实指针事件的收尾"
 * （`onWindowBlur` / `visibilitychange`）：`event` 的坐标是合成的 0。seek 与
 * pending-select 分支早已跳过坐标相关的提交（否则 `secAt(0)` 会把播放头拖到
 * 视口起点），但 Alt 单块右键手势的 box-select 分支曾只判 `!cancelled` ——
 * 于是 `hitAt(0, 0)` 在视图已下滚 / 右滚时可能命中某个 clip，把视口左上角
 * 对应的 clip 范围并入 / 挖掉参数编辑器选区。这是用户没有发出的编辑。
 *
 * 用文本事实做门禁：派发 `onClipParamSelectionGesture` 的分支必须同时带
 * `!synthetic` 守卫，与同函数的 seek 分支口径一致。新增"收尾时读坐标"的
 * 分支时，本文件与 `finalizeActiveGesture` 一起改。
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

/** 取出 `function finalizeActiveGesture(` 的函数体。 */
function finalizeBody(): string {
    const start = CODE.indexOf("function finalizeActiveGesture(");
    expect(start, "找不到 finalizeActiveGesture —— 函数改名后请同步本门禁").toBeGreaterThanOrEqual(
        0,
    );
    // 签名里第一个参数是对象类型 `{ readonly clientX... }`，不能直接找第一个
    // `{`；用返回类型后的函数体起始花括号。
    const open = CODE.indexOf("): void {", start);
    expect(open, "找不到函数体起始").toBeGreaterThanOrEqual(0);
    const bodyOpen = CODE.indexOf("{", open);
    let depth = 0;
    for (let at = bodyOpen; at < CODE.length; at += 1) {
        if (CODE[at] === "{") depth += 1;
        else if (CODE[at] === "}") {
            depth -= 1;
            if (depth === 0) return CODE.slice(open, at + 1);
        }
    }
    throw new Error("finalizeActiveGesture 函数体未闭合");
}

describe("合成收尾（失焦 / 页面隐藏）不得派发坐标相关的编辑", () => {
    it("★ Alt 单块手势的 box-select 分支必须带 !synthetic 守卫", () => {
        const body = finalizeBody();
        // 定位派发 onClipParamSelectionGesture 的那段 else-if：从它往前找到分支头。
        const dispatch = body.indexOf("onClipParamSelectionGesture");
        expect(dispatch, "找不到单块参数选区手势的派发点").toBeGreaterThanOrEqual(0);
        const branchStart = body.lastIndexOf("else if", dispatch);
        expect(branchStart, "派发点必须位于一个 else if 分支里").toBeGreaterThanOrEqual(0);
        const branchHead = body.slice(branchStart, body.indexOf("{", branchStart));
        expect(branchHead).toContain("!synthetic");
        expect(branchHead).toContain("!cancelled");
    });

    it("★ seek 分支同样跳过 synthetic（口径对照）", () => {
        const body = finalizeBody();
        const seek = body.indexOf('gesture.kind === "seek"');
        expect(seek).toBeGreaterThanOrEqual(0);
        const seekBranch = body.slice(seek, body.indexOf("}", body.indexOf("if", seek)) + 1);
        expect(seekBranch).toContain("!synthetic");
    });
});
