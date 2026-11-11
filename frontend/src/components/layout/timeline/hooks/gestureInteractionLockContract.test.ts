/*
 * ★ 门禁：**交互锁的生命周期必须与 sessionSlice 的计数器同口径**。
 *
 * ## 防的是什么
 *
 * 手势 1 的落库请求在异步 `.finally` 里才释放交互锁；这段窗口内用户可能已经
 * 开始手势 2。旧实现用布尔 ref 去重：手势 2 的 begin 被早退（不重复 dispatch），
 * 而手势 1 的 `.finally` 一到就无条件 `dispatch(endInteraction())`，把
 * `_interactionLockCount` 归零 —— 手势 2 仍在拖拽，锁却已经没了。随后抵达的
 * `*Remote.fulfilled` 旧快照不再被丢弃，乐观预览被覆盖，画面弹回原位。
 *
 * 修法是计数：每个手势 begin/end 各入账一次，只有 0→1 与 1→0 才碰 dispatch
 * 与几何映射。用文本事实把这四个边界钉住，防止日后有人把布尔改写回来。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE = readFileSync(
    join("src", "components", "layout", "timeline", "hooks", "useKernelGestureInteractionLock.ts"),
    "utf8",
);

/** 去掉注释：注释里引用的符号名不得计入文本事实。 */
function stripComments(text: string): string {
    return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

const CODE = stripComments(SOURCE);

function bodyOf(name: string): string {
    const start = CODE.indexOf(`const ${name} = React.useCallback(`);
    expect(start, `找不到 ${name}`).toBeGreaterThanOrEqual(0);
    const open = CODE.indexOf("{", start);
    let depth = 0;
    for (let at = open; at < CODE.length; at += 1) {
        if (CODE[at] === "{") depth += 1;
        else if (CODE[at] === "}") {
            depth -= 1;
            if (depth === 0) return CODE.slice(open, at + 1);
        }
    }
    throw new Error(`${name} 函数体未闭合`);
}

describe("内核手势交互锁：重叠手势不得提前释放", () => {
    it("★ 不再使用布尔去重，改用深度计数", () => {
        expect(CODE).not.toContain("kernelGestureInteractionActiveRef");
        expect(CODE).toContain("kernelGestureInteractionDepthRef");
    });

    it("★ begin 每个手势都入账，只在 0→1 派发 beginInteraction", () => {
        const body = bodyOf("beginKernelGestureInteraction");
        expect(body).toContain("kernelGestureInteractionDepthRef.current += 1");
        expect(body).toContain("kernelGestureInteractionDepthRef.current > 1");
        expect(body).toContain("dispatch(beginInteraction())");
    });

    it("★ end 每个手势都出账，只在 1→0 派发 endInteraction", () => {
        const body = bodyOf("endKernelGestureInteraction");
        expect(body).toContain("kernelGestureInteractionDepthRef.current -= 1");
        expect(body).toContain("kernelGestureInteractionDepthRef.current > 0");
        expect(body).toContain("dispatch(endInteraction())");
        // 归零检查必须出现在 dispatch 之前（否则会提前释放）。
        expect(body.indexOf("> 0")).toBeLessThan(body.indexOf("dispatch(endInteraction())"));
    });
});
