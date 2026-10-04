/**
 * 时间轴几何手势总线的契约回归。
 *
 * 【为什么单测它】它是「拖拽中波形错位」修复的**信号源**：手势开始时发布一份
 * 按下时的几何快照，消费方据此差分出时域映射。两个失败模式都只在真实拖拽里才
 * 暴露，且都不会报错：
 *
 * - 快照被后续乐观写入污染（发布的是引用而不是拷贝）⇒ 差分成恒等 ⇒ 修复静默失效；
 * - `end` 的通知漏发或重复发 ⇒ 消费方要么一直挂着映射，要么在松手瞬间提前撤下
 *   （波形闪回旧基线）。
 */
import { describe, expect, it, vi } from "vitest";

import {
    beginClipGeometryPreview,
    endClipGeometryPreview,
    getClipGeometryPreviewOrigin,
    subscribeClipGeometryPreview,
} from "./clipGeometryPreviewBus";

/** 每个用例都从"无手势"开始：总线是模块级单例，必须在用例间显式复位。 */
function reset(): void {
    endClipGeometryPreview();
}

describe("clipGeometryPreviewBus", () => {
    it("初始无手势", () => {
        reset();
        expect(getClipGeometryPreviewOrigin()).toBeNull();
    });

    it("begin 发布按下时的几何快照（含源窗口 / 速率等消费字段）", () => {
        reset();
        beginClipGeometryPreview([
            {
                id: "a",
                startSec: 1,
                lengthSec: 2,
                gain: 0.5,
                sourceStartSec: 0.25,
                sourceEndSec: 2.25,
                playbackRate: 1.5,
                reversed: true,
                loopEnabled: true,
            },
            { id: "b", startSec: 3, lengthSec: 4 },
        ]);
        expect(getClipGeometryPreviewOrigin()).toEqual([
            {
                clipId: "a",
                startSec: 1,
                lengthSec: 2,
                gain: 0.5,
                sourceStartSec: 0.25,
                sourceEndSec: 2.25,
                playbackRate: 1.5,
                reversed: true,
                loopEnabled: true,
            },
            // 缺省值按"无变化"：增益 1、源窗口 [0,0]、速率 1、正放、非 Loop。
            // 源窗口缺省会退化成"零跨度"，但映射只依赖锚点位置，几何本身来自
            // Redux 的真实值（这里的缺省只服务于测试与不完整调用方）。
            {
                clipId: "b",
                startSec: 3,
                lengthSec: 4,
                gain: 1,
                sourceStartSec: 0,
                sourceEndSec: 0,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
            },
        ]);
    });

    it("★ 快照是拷贝：手势中的乐观写入不会污染它（否则差分成恒等）", () => {
        reset();
        const clips = [{ id: "a", startSec: 1, lengthSec: 2, gain: 1 }];
        beginClipGeometryPreview(clips);
        // 模拟手势逐帧改写 Redux 里的同一批对象。
        clips[0]!.startSec = 99;
        expect(getClipGeometryPreviewOrigin()?.[0]?.startSec).toBe(1);
    });

    it("begin / end 各通知订阅者一次，重复 end 不重复通知", () => {
        reset();
        const listener = vi.fn();
        const unsubscribe = subscribeClipGeometryPreview(listener);
        beginClipGeometryPreview([{ id: "a", startSec: 0, lengthSec: 1 }]);
        expect(listener).toHaveBeenCalledTimes(1);
        endClipGeometryPreview();
        expect(listener).toHaveBeenCalledTimes(2);
        // 已经结束：再 end 是 no-op（避免消费方把"收尾状态"重复置一次）。
        endClipGeometryPreview();
        expect(listener).toHaveBeenCalledTimes(2);
        unsubscribe();
        beginClipGeometryPreview([{ id: "a", startSec: 0, lengthSec: 1 }]);
        expect(listener).toHaveBeenCalledTimes(2);
        reset();
    });

    it("订阅者抛异常不影响其它订阅者与手势本身", () => {
        reset();
        const bad = vi.fn(() => {
            throw new Error("boom");
        });
        const good = vi.fn();
        const un1 = subscribeClipGeometryPreview(bad);
        const un2 = subscribeClipGeometryPreview(good);
        expect(() =>
            beginClipGeometryPreview([{ id: "a", startSec: 0, lengthSec: 1 }]),
        ).not.toThrow();
        expect(good).toHaveBeenCalledTimes(1);
        un1();
        un2();
        reset();
    });
});
