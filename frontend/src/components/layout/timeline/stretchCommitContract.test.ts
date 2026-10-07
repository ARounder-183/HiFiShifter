/*
 * ★ 门禁：拉伸提交链必须**单点收口**，且落库前合上取数闸门。
 *
 * ## 防的是什么
 *
 * 单 clip 拉伸与组拉伸是同源的两条路径，曾经各写一遍收尾，于是分叉：组路径在
 * 改写参数线后补了 `bumpParamsEpoch()`，单路径漏了 —— 松手后波形停在
 * 「新几何 × 旧曲线」且只有再做一次别的操作才恢复（用户报告的现象）。
 *
 * 另一件事：几何落库的 fulfilled handler 必然递增 `paramsEpoch` ⇒ 触发一次
 * 「新几何 × 旧曲线」的中间态取数。它必须在**落库派发之前**被闸门压住
 *（见 `loudnessFetchGate`），否则会先落地、让收尾判据提前撤下映射。
 *
 * 这两件事都靠"调用方记得接"就一定会有路径漏掉，因此用可判定的文本事实做门禁：
 * 闸门合上恰好两次、补取数恰好一处、两条路径都走同一个收尾函数。
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE = readFileSync(join("src", "components", "layout", "TimelinePanel.tsx"), "utf8");

/** 去掉块注释与行注释：注释里引用的符号名不得计入文本事实。 */
function stripComments(text: string): string {
    return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

const CODE = stripComments(SOURCE);

function countOccurrences(symbol: string): number {
    return CODE.split(symbol).length - 1;
}

describe("拉伸提交链的门禁", () => {
    it("★ 取数闸门在两条拉伸路径各合上一次（且都以 lockParamLines 为条件）", () => {
        // 恰好两处：单 clip 拉伸 + 组拉伸。
        expect(countOccurrences("holdLoudnessFetch()")).toBe(2);
        // 都必须是"要搬曲线才合上"，否则会白白压掉一次取数。
        expect(countOccurrences("if (lockParamLines) holdLoudnessFetch();")).toBe(2);
        // 释放只有一处（收口在 finishStretchCommitChain 里）。
        expect(countOccurrences("releaseLoudnessFetch()")).toBe(1);
    });

    it("★ 闸门合上必须早于落库派发（否则中间态取数已经发出）", () => {
        const holdIndexes: number[] = [];
        for (
            let at = CODE.indexOf("holdLoudnessFetch()");
            at >= 0;
            at = CODE.indexOf("holdLoudnessFetch()", at + 1)
        ) {
            holdIndexes.push(at);
        }
        expect(holdIndexes).toHaveLength(2);
        for (const holdAt of holdIndexes) {
            const persistAt = CODE.indexOf("setClipsStateBulkRemote", holdAt);
            expect(persistAt).toBeGreaterThan(holdAt);
            // 同一段收尾块内（不允许隔了很远的另一个调用点来"凑数"）。
            expect(persistAt - holdAt).toBeLessThan(400);
        }
    });

    it("★ 两条路径共用同一个收尾函数与同一次补取数（杜绝再次分叉）", () => {
        // 单 clip 与组拉伸各调用一次 finishStretchWithParamLines。
        expect(countOccurrences("await finishStretchWithParamLines({")).toBe(2);
        // 收尾链（释放闸门 + 补取数 + 释放交互锁）在两条路径各调用一次。
        expect(countOccurrences("finishStretchCommitChain(lockParamLines)")).toBe(2);
        // 补取数只允许出现在收尾函数内部：整文件恰好一处。
        expect(countOccurrences("dispatch(bumpParamsEpoch())")).toBe(1);
        // 两条路径各自的曲线改写入口仍然各只有一处（没有被复制成第二份）。
        expect(countOccurrences("stretchLinkedParams(")).toBe(1);
        expect(countOccurrences("stretchTrackLinkedParams(")).toBe(1);
    });
});
