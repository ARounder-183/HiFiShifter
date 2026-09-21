import { describe, expect, it } from "vitest";

import { isIdentitySnapshot, snapshotFromPayloads } from "./useLoudnessCurves";

/** 造一份最小可用载荷（只填 Snapshot 会读的字段）。 */
function payload(args: { edit?: number[]; orig?: number[]; framePeriodMs?: number }): never {
    return {
        ok: true,
        edit: args.edit ?? [],
        orig: args.orig ?? [],
        frame_period_ms: args.framePeriodMs ?? 5,
    } as never;
}

/**
 * 恒等判定的回归。
 *
 * 【为什么单测它】`identity = true` 会让面板**完全不挂幅度映射**，波形退化为
 * 线性直投 —— 此时编辑 volume / dyn 的 live 覆盖根本传不到波形上，实时预览
 * 直接失效（且要等提交后快照重取才恢复）。历史上正是把"用户还没画任何一笔
 * 时的基线 == 目标"误判成恒等，导致这个现象；判定必须是"有没有可能产生
 * 非 1 增益"，而不是"当前是否恰巧全是 1"。
 */
describe("isIdentitySnapshot", () => {
    it("volume 恒 1 且无基线 → 恒等（动态永远不可能贡献增益）", () => {
        expect(isIdentitySnapshot({ volume: [1, 1, 1], dynBaseline: [] })).toBe(true);
    });

    it("★ 未画任何一笔（哨兵解析成基线 → edit 恒等于 orig）仍须挂映射", () => {
        // 真实场景的忠实复现：后端把哨兵解析成原声基线返回，于是 `edit`
        // 逐帧等于 `orig`。逐帧比较会得出"增益全是 1"，但用户随时会在 dyn
        // 面板落下第一笔 —— 一旦据此判恒等 → 不挂映射 → live 覆盖传不到
        // 波形 → 实时预览失效（本测试存在的理由，见文件头说明）。
        const baseline = [0.58, 0.42, 0.9];
        const snap = snapshotFromPayloads(
            payload({ edit: [1, 1, 1] }),
            payload({ edit: baseline.slice(), orig: baseline.slice() }),
            5,
        );
        expect(snap).not.toBeNull();
        expect(snap?.identity).toBe(false);
    });

    it("基线含静音帧（0）也不得据此判恒等 —— 0 是合法且常见的值", () => {
        expect(
            isIdentitySnapshot({
                volume: [1],
                dynBaseline: [0],
            }),
        ).toBe(false);
    });

    it("volume 已偏离 1 → 非恒等（波形必须按包络起伏）", () => {
        expect(isIdentitySnapshot({ volume: [1, 0.5], dynBaseline: [] })).toBe(false);
    });

    it("空 volume 且无基线 → 恒等（无数据可施加）", () => {
        expect(isIdentitySnapshot({ volume: [], dynBaseline: [] })).toBe(true);
    });
});
