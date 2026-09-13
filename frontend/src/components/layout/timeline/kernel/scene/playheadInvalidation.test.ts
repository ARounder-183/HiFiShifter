/**
 * 播放头标脏判定（./playheadInvalidation）行为自检。
 *
 * 【主要内容】
 * 1. 位置变化 → 需要重绘；位置未变 → 不需要（空闲零成本）；
 * 2. 亚像素量级以下的浮点抖动 → 不重绘，避免每帧刷帧；
 * 3. 首次绘制（上次为 NaN 哨兵）→ 必须重绘；
 * 4. 当前值非有限 → 不重绘（防御 NaN 灌进样式）。
 *
 * 【作用】宿主本身需要真实 WebGL2 上下文才能构造、无法直接单测，而「播放头变化时
 * 没有请求重绘」正是缺陷 #2 的根因。把该判定抽成纯函数，就能在没有 GPU 的环境下
 * 锁定行为（本工程 vitest 是 node 环境，无 jsdom）。
 *
 * 【与其他模块的关系】覆盖 `playheadInvalidation.ts`；被 `timelineKernelHost` 的
 * 帧提交消费。不依赖 React / DOM / WebGL。
 */
import { describe, expect, it } from "vitest";

import { shouldRepaintForPlayhead } from "./playheadInvalidation";

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
