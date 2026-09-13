/**
 * 播放头取值与标脏判定（./playheadInvalidation）行为自检。
 *
 * 【主要内容】
 * 1. `resolvePlayheadSec`：实时 getter 优先、缺省 / 非有限时回退数据镜像；
 * 2. `shouldRepaintForPlayhead`：位置变化 → 重绘；未变 / 亚像素抖动 → 不重绘；
 *    首次绘制（NaN 哨兵）→ 重绘；当前值非有限 → 不重绘。
 *
 * 【作用】宿主本身需要真实 WebGL2 上下文才能构造、无法直接单测，而「播放头变化时
 * 取值滞后 + 没有请求重绘」正是缺陷 #2 的根因。把这两条判定抽成纯函数，就能在没有
 * GPU 的环境下锁定行为（本工程 vitest 是 node 环境，无 jsdom）。
 *
 * 【与其他模块的关系】覆盖 `playheadInvalidation.ts`；被 `timelineKernelHost` 的
 * 帧提交（syncDom / draw）与滚轮缩放锚点消费。不依赖 React / DOM / WebGL。
 */
import { describe, expect, it } from "vitest";

import { resolvePlayheadSec, shouldRepaintForPlayhead } from "./playheadInvalidation";

describe("resolvePlayheadSec", () => {
    it("实时 getter 优先于数据镜像（镜像滞后一次提交）", () => {
        // 实测场景：seek 后镜像仍停在旧值 12.5，而实时值已是 4.293。
        expect(resolvePlayheadSec(4.293, 12.5)).toBe(4.293);
    });

    it("getter 缺省（未接线 / 单测）→ 回退镜像", () => {
        expect(resolvePlayheadSec(undefined, 12.5)).toBe(12.5);
    });

    it("getter 返回非有限值 → 回退镜像（不让 NaN 流进样式写入）", () => {
        expect(resolvePlayheadSec(Number.NaN, 12.5)).toBe(12.5);
        expect(resolvePlayheadSec(Number.POSITIVE_INFINITY, 12.5)).toBe(12.5);
    });

    it("镜像也是 0 时返回 0（0 是合法位置，不能被当成缺失）", () => {
        expect(resolvePlayheadSec(undefined, 0)).toBe(0);
    });
});

describe("shouldRepaintForPlayhead", () => {
    it("位置变化 → 需要重绘", () => {
        expect(shouldRepaintForPlayhead(12.5, 4.82)).toBe(true);
    });

    it("位置未变 → 不需要重绘（空闲零成本）", () => {
        expect(shouldRepaintForPlayhead(12.5, 12.5)).toBe(false);
    });

    it("差异在亚像素量级以下 → 不重绘（避免浮点噪声刷帧）", () => {
        expect(shouldRepaintForPlayhead(12.5, 12.5 + 1e-9)).toBe(false);
    });

    it("首次绘制（上次为 NaN 哨兵）→ 需要重绘", () => {
        expect(shouldRepaintForPlayhead(12.5, Number.NaN)).toBe(true);
    });

    it("当前位置非有限 → 不重绘（防御 NaN 灌进样式）", () => {
        expect(shouldRepaintForPlayhead(Number.NaN, 12.5)).toBe(false);
    });
});
