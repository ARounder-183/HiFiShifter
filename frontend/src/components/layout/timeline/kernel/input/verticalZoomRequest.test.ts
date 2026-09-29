/**
 * 竖直缩放「请求」解析（./verticalZoomRequest）行为自检。
 *
 * 【主要内容】
 * 1. 锚点不变式：落地后指针处的**行位置**与请求时相同（缩放前后指针下的内容不移动）；
 * 2. 上下限处收敛：夹到 MIN / MAX 后与基准相同 ⇒ 不产生请求；
 * 3. 累积基准：连续滚轮时每步从**未落地的请求行高**继续（中间步进不丢）；
 * 4. 落地位置用**已提交**的行高反算（不是请求值）；
 * 5. 非法倍率不产生请求（不把 NaN 写进行高）。
 *
 * 【作用】锁定"竖直缩放抽动"修复的时序前提：请求只携带不变式，位置在行高落地时才
 * 反算 —— 于是内核永不领先 React 改行高，也就没有"几何按新行高、视口按旧位置"的错配帧。
 *
 * 【与其他模块的关系】覆盖 `verticalZoomRequest.ts`；不依赖 DOM / React / GL。
 */

import { describe, expect, it } from "vitest";

import {
    resolveVerticalZoomScrollTop,
    resolveVerticalZoomStep,
    type ResolveVerticalZoomStepArgs,
} from "./verticalZoomRequest";

/** 应用一次请求后的行位置（行单位）——锚点不变式的观测量。 */
function rowUnitAt(scrollTop: number, pointerY: number, rowHeight: number): number {
    return (scrollTop + pointerY) / rowHeight;
}

function base(overrides: Partial<ResolveVerticalZoomStepArgs> = {}): ResolveVerticalZoomStepArgs {
    return {
        factor: 1.1,
        kernelRowHeight: 96,
        scrollTop: 200,
        pointerY: 38,
        baseRowHeight: 96,
        minRowHeight: 80,
        maxRowHeight: 192,
        ...overrides,
    };
}

describe("resolveVerticalZoomStep", () => {
    it("放大：行高按倍率取整，锚点行号取自内核当前值", () => {
        const request = resolveVerticalZoomStep(base({ factor: 1.1 }));
        expect(request).not.toBeNull();
        expect(request!.rowHeight).toBe(Math.round(96 * 1.1));
        expect(request!.anchorRowUnit).toBeCloseTo((200 + 38) / 96, 12);
        expect(request!.anchorScreenY).toBe(38);
    });

    it("锚点不变式：按已提交行高落地后，指针处的行位置不变", () => {
        const request = resolveVerticalZoomStep(base({ factor: 1.1 }))!;
        const committed = request.rowHeight;
        const nextScrollTop = resolveVerticalZoomScrollTop(request, committed);
        expect(rowUnitAt(nextScrollTop, request.anchorScreenY, committed)).toBeCloseTo(
            request.anchorRowUnit,
            12,
        );
    });

    it("锚点不变式在连续多步后仍成立（每步都重新解析）", () => {
        // 指针固定在 y=38，从 96 连续放大 4 步。
        let kernelRowHeight = 96;
        let scrollTop = 200;
        const pointerY = 38;
        const anchor = rowUnitAt(scrollTop, pointerY, kernelRowHeight);
        for (let step = 0; step < 4; step += 1) {
            const request = resolveVerticalZoomStep(
                base({
                    kernelRowHeight,
                    scrollTop,
                    baseRowHeight: kernelRowHeight,
                    pointerY,
                }),
            )!;
            scrollTop = resolveVerticalZoomScrollTop(request, request.rowHeight);
            kernelRowHeight = request.rowHeight;
        }
        expect(kernelRowHeight).toBeGreaterThan(96);
        expect(rowUnitAt(scrollTop, pointerY, kernelRowHeight)).toBeCloseTo(anchor, 12);
    });

    it("到达上限后不再产生请求（收敛，不空转）", () => {
        expect(
            resolveVerticalZoomStep(
                base({ baseRowHeight: 192, kernelRowHeight: 192, factor: 1.1 }),
            ),
        ).toBeNull();
    });

    it("到达下限后不再产生请求（收敛，不空转）", () => {
        expect(
            resolveVerticalZoomStep(base({ baseRowHeight: 80, kernelRowHeight: 80, factor: 0.9 })),
        ).toBeNull();
    });

    it("越界倍率被夹到上下限，而不是溢出", () => {
        // 96 × 10 = 960 → 夹到 192。
        expect(resolveVerticalZoomStep(base({ factor: 10 }))!.rowHeight).toBe(192);
        // 96 × 0.1 = 9.6 → 夹到 80。
        expect(resolveVerticalZoomStep(base({ factor: 0.1 }))!.rowHeight).toBe(80);
    });

    it("累积基准：连续请求从**未落地**的行高继续（中间步进不丢）", () => {
        const first = resolveVerticalZoomStep(base({ factor: 1.1 }))!;
        // 第二次事件到来时 React 还没落地：内核仍是 96，但基准取第一次请求的 106。
        const second = resolveVerticalZoomStep(
            base({ factor: 1.1, baseRowHeight: first.rowHeight, kernelRowHeight: 96 }),
        )!;
        expect(second.rowHeight).toBe(Math.round(Math.round(96 * 1.1) * 1.1));
        expect(second.rowHeight).toBeGreaterThan(first.rowHeight);
        // 锚点仍按内核当前值算（内容此刻确实还按 96 布局）。
        expect(second.anchorRowUnit).toBeCloseTo((200 + 38) / 96, 12);
    });

    it("非法倍率不产生请求", () => {
        expect(resolveVerticalZoomStep(base({ factor: Number.NaN }))).toBeNull();
        expect(resolveVerticalZoomStep(base({ factor: 0 }))).toBeNull();
        expect(resolveVerticalZoomStep(base({ factor: Number.POSITIVE_INFINITY }))).toBeNull();
    });
});

describe("resolveVerticalZoomScrollTop", () => {
    it("用**已提交**的行高反算（而不是请求值）", () => {
        const request = resolveVerticalZoomStep(base({ factor: 1.1 }))!;
        const committed = request.rowHeight + 11;
        expect(resolveVerticalZoomScrollTop(request, committed)).toBeCloseTo(
            request.anchorRowUnit * committed - request.anchorScreenY,
            12,
        );
    });

    it("不自行钳制（钳制归 ScrollKernel 单一职责）", () => {
        const request = { rowHeight: 96, anchorRowUnit: 0, anchorScreenY: 38 };
        expect(resolveVerticalZoomScrollTop(request, 96)).toBe(-38);
    });
});
