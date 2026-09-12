/**
 * 时间轴内核 · 轨道头窗口化滚动提交量化单测。
 *
 * 【本测试要钉住的核心不变量】
 * 1. **步长必须等于 overscan 安全预算的一半**：滞后超过预算会让窗口覆盖不到真正
 *    可见的行（渲染出空白行）；取一半留 2× 余量。
 * 2. **判定基准是"上次提交值"而不是"上一帧值"**：否则每帧的小位移永远达不到阈值，
 *    窗口化彻底停更。
 * 3. **步长不得为 0**：会让阈值判定恒真、量化失效（且可能形成无限 rAF 循环）。
 * 4. 非法入参不得抛异常（渲染路径不能因上游量测异常而中断）。
 */
import { describe, expect, it } from "vitest";

import { resolveScrollCommitStepPx, shouldCommitScroll } from "./scrollCommit";

describe("resolveScrollCommitStepPx（提交步长）", () => {
    it("★ 步长 = overscan 安全预算的一半", () => {
        // 实测默认：rowHeight 96、overscanRows 2 → 预算 192 → 步长 96。
        expect(resolveScrollCommitStepPx({ rowHeight: 96, overscanRows: 2 })).toBe(96);
        expect(resolveScrollCommitStepPx({ rowHeight: 80, overscanRows: 4 })).toBe(160);
    });

    it("★ 步长恒 ≥ 1px（为 0 会让量化失效）", () => {
        expect(resolveScrollCommitStepPx({ rowHeight: 0, overscanRows: 2 })).toBeGreaterThanOrEqual(
            1,
        );
        expect(
            resolveScrollCommitStepPx({ rowHeight: 96, overscanRows: 0 }),
        ).toBeGreaterThanOrEqual(1);
        // 极小行高 × 极小 overscan：真实值远小于 1px，必须被下限兜住。
        expect(resolveScrollCommitStepPx({ rowHeight: 0.5, overscanRows: 1 })).toBe(1);
    });

    it("非法入参不抛异常且退回安全值", () => {
        expect(resolveScrollCommitStepPx({ rowHeight: Number.NaN, overscanRows: 2 })).toBe(1);
        expect(resolveScrollCommitStepPx({ rowHeight: -100, overscanRows: 2 })).toBe(1);
        expect(resolveScrollCommitStepPx({ rowHeight: 96, overscanRows: Number.NaN })).toBe(1);
        expect(resolveScrollCommitStepPx({ rowHeight: 96, overscanRows: -3 })).toBe(1);
    });
});

describe("shouldCommitScroll（是否提交给 React）", () => {
    const step = 96;

    it("★ 位移达到步长才提交（量化掉每帧的小位移）", () => {
        // 这是本模块存在的理由：内核每帧前进约 4.5px，若不量化则每帧一次
        // React 提交（实测 80 步拖拽 130 次提交）。
        expect(shouldCommitScroll({ committedPx: 0, nextPx: 4.5, stepPx: step })).toBe(false);
        expect(shouldCommitScroll({ committedPx: 0, nextPx: 95, stepPx: step })).toBe(false);
        expect(shouldCommitScroll({ committedPx: 0, nextPx: 96, stepPx: step })).toBe(true);
        expect(shouldCommitScroll({ committedPx: 0, nextPx: 200, stepPx: step })).toBe(true);
    });

    it("★ 判定基准是「上次提交值」（否则窗口化永不停更）", () => {
        // 模拟连续 30 帧、每帧 +4.5px（真实拖拽的步进）。
        // 若拿"上一帧值"当基准，每帧差值都是 4.5 < 96 → 永远不提交。
        let committed = 0;
        let commits = 0;
        for (let i = 1; i <= 30; i++) {
            const next = i * 4.5;
            if (shouldCommitScroll({ committedPx: committed, nextPx: next, stepPx: step })) {
                committed = next;
                commits++;
            }
        }
        // 30 帧共前进 135px，按 96px 步长应提交 1 次（135 ≥ 96；第 2 次需 192）。
        expect(commits).toBe(1);
        // 且最终滞后（135 − 96 = 39）必须小于 overscan 安全预算（2 × 96 = 192）。
        expect(135 - committed).toBeLessThan(192);
    });

    it("反向滚动同样量化（推导对称）", () => {
        expect(shouldCommitScroll({ committedPx: 200, nextPx: 105, stepPx: step })).toBe(false);
        expect(shouldCommitScroll({ committedPx: 200, nextPx: 104, stepPx: step })).toBe(true);
        expect(shouldCommitScroll({ committedPx: 200, nextPx: 0, stepPx: step })).toBe(true);
    });

    it("★ 累积滞后不超过 overscan 安全预算（否则会渲染出空白行）", () => {
        // 全程 0→361（实测满量程），逐步 +4.5px，断言任意时刻滞后 < overscan 预算。
        const overscanBudget = 2 * 96; // overscanRows × rowHeight
        let committed = 0;
        let maxLag = 0;
        for (let next = 0; next <= 361; next += 4.5) {
            if (shouldCommitScroll({ committedPx: committed, nextPx: next, stepPx: step })) {
                committed = next;
            } else {
                maxLag = Math.max(maxLag, next - committed);
            }
        }
        // 步长 = 预算一半 ⇒ 滞后上界 ≈ 步长，恒小于预算。
        expect(maxLag).toBeLessThan(overscanBudget);
        expect(maxLag).toBeLessThanOrEqual(step);
    });

    it("非法入参不提交（避免 NaN 传染进渲染）", () => {
        // 【为什么每条都配一个有限对照】NaN 参与的比较恒为 false，所以"返回 false"
        // 这个断言在**没有守卫**时也成立——单看它测不出守卫是否存在（变异验证：
        // 删掉 `Number.isFinite` 守卫后原用例 9/9 照旧通过）。配上"同样的值换成有限
        // 数就该返回 true"的对照，才能把"守卫在起作用"与"恒返回 false"区分开。
        expect(shouldCommitScroll({ committedPx: Number.NaN, nextPx: 100, stepPx: step })).toBe(
            false,
        );
        // 对照：committed 换成有限值时应当提交（证明上面的 false 不是恒 false）
        expect(shouldCommitScroll({ committedPx: 0, nextPx: 100, stepPx: step })).toBe(true);

        expect(shouldCommitScroll({ committedPx: 0, nextPx: Number.NaN, stepPx: step })).toBe(
            false,
        );
        // 对照：next 有限时应当提交
        expect(shouldCommitScroll({ committedPx: 0, nextPx: 96, stepPx: step })).toBe(true);

        expect(shouldCommitScroll({ committedPx: 0, nextPx: 100, stepPx: Number.NaN })).toBe(false);
        // 对照：step 有限时应当提交
        expect(shouldCommitScroll({ committedPx: 0, nextPx: 100, stepPx: 96 })).toBe(true);
    });

    it("位置未变时不提交", () => {
        expect(shouldCommitScroll({ committedPx: 120, nextPx: 120, stepPx: step })).toBe(false);
    });
});
