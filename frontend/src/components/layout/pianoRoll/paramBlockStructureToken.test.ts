/**
 * 结构指纹（./paramBlockStructureToken）自检。
 *
 * 【本测试守护什么】问题 4「松手先回弹旧参数、再生效新参数」的第一块基石：
 * 视觉预览的失效判据必须是**结构**，不是 revision。所以这里逐字段钉住
 * 「哪些变化算结构、哪些不算」—— 尤其是 **value 不算**（否则一次拖动会撤销
 * 自己的预览）。
 */
import { describe, expect, it } from "vitest";

import { paramBlockStructureToken } from "./paramBlockStructureToken";
import type { ParamBlockEditSet, TakeSpan, UserParamBlock } from "./paramBlockTypes";

const SPAN: TakeSpan = {
    clipId: "clip",
    takeId: "take",
    cycle: 0,
    sourceStartSec: 0,
    sourceEndSec: 1,
    label: "manual",
};

function takeBlock(overrides: Partial<UserParamBlock> = {}): UserParamBlock {
    return {
        id: "b1",
        geometry: { kind: "take", members: [SPAN] },
        createdSeq: 1,
        ...overrides,
    };
}

function editsOf(blocks: readonly UserParamBlock[], revision = 1): ParamBlockEditSet {
    return { schemaVersion: 2, revision, blocks };
}

describe("paramBlockStructureToken", () => {
    it("空集有稳定且可区分的令牌", () => {
        expect(paramBlockStructureToken(editsOf([]))).toBe("structure-empty");
        expect(paramBlockStructureToken(editsOf([]))).not.toBe(
            paramBlockStructureToken(editsOf([takeBlock()])),
        );
    });

    it("★ value（块中心）变化不改结构", () => {
        const before = paramBlockStructureToken(editsOf([takeBlock({ value: 60 })]));
        const after = paramBlockStructureToken(editsOf([takeBlock({ value: 67 })]));
        expect(after).toBe(before);
    });

    it("★ revision / createdSeq / 数组顺序都不改结构", () => {
        const a = takeBlock({ id: "a", value: 60 });
        const b = takeBlock({ id: "b", value: 61 });
        const base = paramBlockStructureToken(editsOf([a, b], 1));
        expect(paramBlockStructureToken(editsOf([a, b], 99))).toBe(base);
        expect(paramBlockStructureToken(editsOf([b, a], 1))).toBe(base);
        expect(paramBlockStructureToken(editsOf([{ ...a, createdSeq: 7 }, b], 1))).toBe(base);
    });

    it("成员源区间变化 ⇒ 结构变化", () => {
        const base = paramBlockStructureToken(editsOf([takeBlock()]));
        const moved = paramBlockStructureToken(
            editsOf([
                takeBlock({
                    geometry: {
                        kind: "take",
                        members: [{ ...SPAN, sourceStartSec: 0.5 }],
                    },
                }),
            ]),
        );
        expect(moved).not.toBe(base);
    });

    it("过渡宽度变化 ⇒ 结构变化", () => {
        const base = paramBlockStructureToken(editsOf([takeBlock()]));
        expect(paramBlockStructureToken(editsOf([takeBlock({ transitionOutSec: 0.1 })]))).not.toBe(
            base,
        );
    });

    it("删除标记变化 ⇒ 结构变化", () => {
        const base = paramBlockStructureToken(editsOf([takeBlock()]));
        expect(paramBlockStructureToken(editsOf([takeBlock({ deleted: true })]))).not.toBe(base);
    });

    it("新增 / 删除块 ⇒ 结构变化", () => {
        const base = paramBlockStructureToken(editsOf([takeBlock()]));
        expect(paramBlockStructureToken(editsOf([takeBlock(), takeBlock({ id: "b2" })]))).not.toBe(
            base,
        );
    });

    it("group 的子块几何变化会递归到结构令牌", () => {
        const group = (childStart: number): UserParamBlock => ({
            id: "g1",
            createdSeq: 2,
            geometry: {
                kind: "group",
                children: [
                    {
                        id: "c1",
                        createdSeq: 1,
                        geometry: {
                            kind: "take",
                            members: [{ ...SPAN, sourceStartSec: childStart }],
                        },
                    },
                ],
            },
        });
        expect(paramBlockStructureToken(editsOf([group(0)]))).not.toBe(
            paramBlockStructureToken(editsOf([group(0.25)])),
        );
    });

    it("baseCover 覆盖变化 ⇒ 结构变化（基块衬底会重新露出来）", () => {
        const base = paramBlockStructureToken(editsOf([takeBlock()]));
        expect(
            paramBlockStructureToken(
                editsOf([takeBlock({ baseCover: { spans: [SPAN], full: true } })]),
            ),
        ).not.toBe(base);
    });
});
