/**
 * 帧提交器自检。
 *
 * 【主要内容】验证统一帧提交的三条核心保证：
 * 1. **顺序**：图层按 LAYER_ORDER 固定顺序绘制，同序按注册序；
 * 2. **去重**：投影完全相同的提交不产生第二轮重绘；
 * 3. **隔离**：单个图层抛错不会阻断其它图层（一个图层的数据异常不应让整条
 *    时间线消失）。
 *
 * 【作用】此前每个图层各有触发方式（总线订阅 / 命令式桥接 / 直接改 DOM），
 * 新增视口变更路径时极易漏掉某层，只能靠"每帧对账"在下一帧补救，表现为一帧
 * 错位。提交器把入口收敛为一个，本测试守护这个收敛不被破坏。
 *
 * 【与其他模块的关系】覆盖 `timelineFrameCommitter.ts`，依赖 `timelineAxis.ts`
 * 提供投影与相等性比较。
 */

import { test, vi } from "vitest";

import * as frontendLog from "../../../../services/frontendLog";

import { LAYER_ORDER, createTimelineFrameCommitter } from "./timelineFrameCommitter.js";
import { createTimelineAxis, type TimelineAxis } from "../../renderKernel/timelineAxis.js";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (actual !== expected) {
        throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
    }
}

test("components/layout/timeline/runtime/timelineFrameCommitter.test.ts scripted checks", async () => {
    const base = createTimelineAxis({ pxPerSec: 100, viewportWidthPx: 800 });

    // ── 1. 按 order 升序绘制 ────────────────────────────────────────
    {
        const order: string[] = [];
        const committer = createTimelineFrameCommitter(base);
        // 故意乱序注册，验证绘制顺序只由 order 决定
        committer.register(
            { name: "waveform", paint: () => order.push("waveform") },
            LAYER_ORDER.waveform,
        );
        committer.register(
            { name: "gridBack", paint: () => order.push("gridBack") },
            LAYER_ORDER.gridBack,
        );
        committer.register(
            { name: "clipBody", paint: () => order.push("clipBody") },
            LAYER_ORDER.clipBody,
        );
        committer.register(
            { name: "gridOverlay", paint: () => order.push("gridOverlay") },
            LAYER_ORDER.gridOverlay,
        );

        committer.commit(createTimelineAxis({ pxPerSec: 200, viewportWidthPx: 800 }));
        assertEqual(
            order.join(">"),
            "gridBack>clipBody>waveform>gridOverlay",
            "paint order follows LAYER_ORDER",
        );
    }

    // ── 2. 同 order 内按注册顺序 ────────────────────────────────────
    {
        const order: string[] = [];
        const committer = createTimelineFrameCommitter(base);
        committer.register({ name: "a", paint: () => order.push("a") }, 50);
        committer.register({ name: "b", paint: () => order.push("b") }, 50);
        committer.commit(createTimelineAxis({ pxPerSec: 200 }));
        assertEqual(order.join(">"), "a>b", "same order keeps registration sequence");
    }

    // ── 3. 去重：相同投影不产生第二轮重绘 ───────────────────────────
    {
        let calls = 0;
        const committer = createTimelineFrameCommitter(base);
        committer.register({ name: "counter", paint: () => (calls += 1) }, LAYER_ORDER.clipBody);

        const axis = createTimelineAxis({ pxPerSec: 300, viewportWidthPx: 800 });
        committer.commit(axis);
        assertEqual(calls, 1, "first commit paints");
        // 结构与数值完全相同的新对象也必须被识别为「未变化」
        committer.commit(createTimelineAxis({ pxPerSec: 300, viewportWidthPx: 800 }));
        assertEqual(calls, 1, "identical projection is deduplicated");
        // 只有 scrollLeft 变化才算变化
        committer.commit(
            createTimelineAxis({ pxPerSec: 300, scrollLeftPx: 10, viewportWidthPx: 800 }),
        );
        assertEqual(calls, 2, "scrollLeft change repaints");
        // force 忽略去重
        committer.commit(axis, { force: true });
        assertEqual(calls, 3, "force bypasses dedup");
    }

    // ── 4. 异常隔离：单个图层抛错不影响其它图层 ─────────────────────
    {
        const order: string[] = [];
        const committer = createTimelineFrameCommitter(base);
        const errors: unknown[] = [];
        const errorSpy = vi
            .spyOn(frontendLog, "frontendLog")
            .mockImplementation((...args: unknown[]) => errors.push(args));
        try {
            committer.register(
                {
                    name: "boom",
                    paint: () => {
                        throw new Error("layer failure");
                    },
                },
                LAYER_ORDER.clipBody,
            );
            committer.register(
                { name: "after", paint: () => order.push("after") },
                LAYER_ORDER.waveform,
            );
            committer.commit(createTimelineAxis({ pxPerSec: 400 }));
        } finally {
            errorSpy.mockRestore();
        }
        assertEqual(order.join(">"), "after", "later layers still paint");
        assertEqual(errors.length, 1, "failure is reported once");
    }

    // ── 4b. 失败层在投影不变的下一次提交里被补画 ──────────────────────
    // 旧实现把 lastCommitted 写在派发之前：paint 抛错的层被去重永久跳过，
    // 即使后续反复提交同一投影也再不会重绘（画面永久空白/陈旧）。
    {
        const committer = createTimelineFrameCommitter(base);
        let shouldFail = true;
        let failedCalls = 0;
        let healthyCalls = 0;
        const axis = createTimelineAxis({ pxPerSec: 401, viewportWidthPx: 800 });
        const errorSpy = vi.spyOn(frontendLog, "frontendLog").mockImplementation(() => undefined);
        try {
            committer.register(
                {
                    name: "flaky",
                    paint: () => {
                        failedCalls += 1;
                        if (shouldFail) throw new Error("transient failure");
                    },
                },
                LAYER_ORDER.clipBody,
            );
            committer.register(
                { name: "healthy", paint: () => (healthyCalls += 1) },
                LAYER_ORDER.waveform,
            );
            committer.commit(axis);
            assertEqual(failedCalls, 1, "first attempt ran");
            assertEqual(healthyCalls, 1, "healthy layer painted once");
            shouldFail = false;
            // 投影完全相同：正常层应被去重，失败层仍必须补画。
            committer.commit(createTimelineAxis({ pxPerSec: 401, viewportWidthPx: 800 }));
            assertEqual(failedCalls, 2, "failed layer retried on identical projection");
            assertEqual(healthyCalls, 1, "healthy layer stays deduplicated");
            // 补画成功后回到稳态，不再无谓重绘。
            committer.commit(createTimelineAxis({ pxPerSec: 401, viewportWidthPx: 800 }));
            assertEqual(failedCalls, 2, "no retry after success");
        } finally {
            errorSpy.mockRestore();
        }
    }

    // ── 5. 注销后不再参与绘制 ───────────────────────────────────────
    {
        let calls = 0;
        const committer = createTimelineFrameCommitter(base);
        const unregister = committer.register({ name: "temp", paint: () => (calls += 1) }, 10);
        committer.commit(createTimelineAxis({ pxPerSec: 501 }));
        assertEqual(calls, 1, "registered layer paints");
        unregister();
        committer.commit(createTimelineAxis({ pxPerSec: 502 }));
        assertEqual(calls, 1, "unregistered layer stops painting");
        assertEqual(committer.layerCount(), 0, "layerCount drops to zero");
    }

    // ── 6. getSnapshot 返回最近一次提交的投影 ───────────────────────
    {
        const committer = createTimelineFrameCommitter(base);
        assertEqual(committer.getSnapshot(), base, "initial snapshot");
        const next: TimelineAxis = createTimelineAxis({ pxPerSec: 777, scrollLeftPx: 42 });
        committer.commit(next);
        assertEqual(committer.getSnapshot(), next, "snapshot after commit");
    }
});
