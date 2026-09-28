/**
 * live 覆盖**换窗口重锚**的回归。
 *
 * ## 为什么必须有这份测试
 *
 * 用户报告的现象：用绘制工具在参数编辑器画 `音量` / `动态` 时，**有概率**
 * 按下处画出了一点，但拖拽过程中的轨迹完全不画。根因是 `paramView` 在笔画期间
 * 被换掉后，覆盖层被**整份清零**（见
 * docs/plans/2026-09-26-volume-dyn-drag-trail-fix.md §1.1）。
 *
 * 修法（阶段 2）是重锚：把**用户画过的帧**按绝对帧号搬进新窗口，未动的帧跟随
 * 新基准。这条路径只在"窗口被换掉"时才走到 —— 人工复现要恰好撞上取数回包，
 * 因此必须由单测锁住两种输入形态（计划里点名的"key 不匹配"与"引用变化"）：
 *
 * - **key 不匹配** = 换了一个窗口（起帧 / 帧数 / 步长不同）；
 * - **引用变化** = 同一个窗口重取数（键相同、`pv.edit` 是新的数组）。
 *
 * 两者都必须"已画值被保留"，而不是被清零。
 */
import { describe, expect, it } from "vitest";

import {
    parseLiveEditWindowKey,
    reanchorLiveEditWindow,
    writeDenseIntoLiveWindow,
} from "./liveEditWindow";

/** 造一个窗口：`length` 个采样、首帧 `startFrame`、步长 `stride`。 */
function windowOf(args: {
    startFrame: number;
    stride: number;
    length: number;
    fill: number;
}): number[] {
    return new Array<number>(args.length).fill(args.fill);
}

describe("parseLiveEditWindowKey", () => {
    it("拆出作用域与窗口参数", () => {
        const parsed = parseLiveEditWindowKey("v2|track-1|volume|1200|6400|1");
        expect(parsed.scope).toBe("v2|track-1|volume");
        expect(parsed.startFrame).toBe(1200);
        expect(parsed.stride).toBe(1);
    });

    /** 作用域只认「轨道 + 参数」：换窗口不得改变它，否则会被误判成换参数。 */
    it("同一轨道 / 参数下不同窗口的 scope 相同", () => {
        const a = parseLiveEditWindowKey("v2|track-1|dyn|0|6400|1");
        const b = parseLiveEditWindowKey("v2|track-1|dyn|900|3200|2");
        expect(a.scope).toBe(b.scope);
    });

    it("换参数 / 换轨时 scope 不同", () => {
        const base = parseLiveEditWindowKey("v2|track-1|volume|0|6400|1").scope;
        expect(parseLiveEditWindowKey("v2|track-1|dyn|0|6400|1").scope).not.toBe(base);
        expect(parseLiveEditWindowKey("v2|track-2|volume|0|6400|1").scope).not.toBe(base);
    });

    it("字段缺失时退化为保守默认（起帧 0 / 步长 1），不抛错", () => {
        const parsed = parseLiveEditWindowKey("garbage");
        expect(parsed.startFrame).toBe(0);
        expect(parsed.stride).toBe(1);
        expect(parsed.scope).toBe("garbage||");
    });
});

describe("reanchorLiveEditWindow：引用变化（同窗口重取数）", () => {
    /**
     * 同键、新基准：这是"取数回包换掉了 `pv.edit` 引用"的形态。
     *
     * 旧实现会 `pv.edit.slice()` 整份清零 —— 用户画的笔画当场消失。重锚必须
     * 保留用户画过的帧，并让**未画**的帧跟随新基准（外部改动照常显现）。
     */
    it("★ 已画值被保留，未画帧跟随新基准（不被清零）", () => {
        const startFrame = 100;
        const base = windowOf({ startFrame, stride: 1, length: 10, fill: 1 });
        const edit = base.slice();
        // 用户画了 3 帧（帧 103..105）。
        edit[3] = 0.2;
        edit[4] = 0.4;
        edit[5] = 0.6;
        // 后端重取：未画帧的基准值全变了（例如另一处编辑 / 撤销后的新数据）。
        const nextEdit = windowOf({ startFrame, stride: 1, length: 10, fill: 7 });

        const { values, drawnRange } = reanchorLiveEditWindow({
            edit,
            base,
            startFrame,
            stride: 1,
            nextEdit,
            nextStartFrame: startFrame,
            nextStride: 1,
        });

        expect(values[3]).toBe(0.2);
        expect(values[4]).toBe(0.4);
        expect(values[5]).toBe(0.6);
        // 未画的帧取新基准，而不是陈旧的旧基准。
        expect(values[0]).toBe(7);
        expect(values[9]).toBe(7);
        expect(drawnRange).toEqual({ lo: 3, hi: 5 });
    });

    it("覆盖层与旧基准逐帧相同时（用户还没画）→ 结果就是新基准", () => {
        const base = windowOf({ startFrame: 0, stride: 1, length: 8, fill: 1 });
        const nextEdit = windowOf({ startFrame: 0, stride: 1, length: 8, fill: 0.5 });
        const { values, drawnRange } = reanchorLiveEditWindow({
            edit: base.slice(),
            base,
            startFrame: 0,
            stride: 1,
            nextEdit,
            nextStartFrame: 0,
            nextStride: 1,
        });
        expect(values).toEqual(nextEdit);
        expect(drawnRange).toBeNull();
    });
});

describe("reanchorLiveEditWindow：key 不匹配（换窗口）", () => {
    /** 窗口向右平移：重叠区间的已画值必须落在**新的**下标上（按绝对帧号）。 */
    it("★ 窗口右移：已画值按绝对帧号搬到新下标", () => {
        const startFrame = 100;
        const base = windowOf({ startFrame, stride: 1, length: 10, fill: 1 });
        const edit = base.slice();
        edit[3] = 0.2; // 帧 103
        edit[4] = 0.4; // 帧 104

        const nextStartFrame = 102;
        const nextEdit = windowOf({
            startFrame: nextStartFrame,
            stride: 1,
            length: 10,
            fill: 5,
        });

        const { values, drawnRange } = reanchorLiveEditWindow({
            edit,
            base,
            startFrame,
            stride: 1,
            nextEdit,
            nextStartFrame,
            nextStride: 1,
        });

        // 帧 103 → 新下标 1；帧 104 → 新下标 2。
        expect(values[1]).toBe(0.2);
        expect(values[2]).toBe(0.4);
        expect(drawnRange).toEqual({ lo: 1, hi: 2 });
        // 窗口右移后落在旧窗口之外的帧取新基准。
        expect(values[9]).toBe(5);
        // 平移丢掉的那一帧（帧 102 之前的 100 / 101）不再存在。
        expect(values).toHaveLength(10);
    });

    /** 新窗口与旧窗口完全不相交：没有可搬的帧，结果就是新基准。 */
    it("窗口完全错开 → 无可搬帧，取新基准", () => {
        const base = windowOf({ startFrame: 0, stride: 1, length: 8, fill: 1 });
        const edit = base.slice();
        edit[0] = 9;
        const nextEdit = windowOf({ startFrame: 500, stride: 1, length: 8, fill: 3 });
        const { values, drawnRange } = reanchorLiveEditWindow({
            edit,
            base,
            startFrame: 0,
            stride: 1,
            nextEdit,
            nextStartFrame: 500,
            nextStride: 1,
        });
        expect(values).toEqual(nextEdit);
        expect(drawnRange).toBeNull();
    });

    /** 步长变化（快照降采样）：只有两个网格**恰好重合**的帧才搬。 */
    it("步长变化时按帧号对齐，网格不重合的帧取新基准", () => {
        const startFrame = 0;
        const stride = 2;
        const base = windowOf({ startFrame, stride, length: 8, fill: 1 });
        const edit = base.slice();
        edit[2] = 0.25; // 帧 4
        edit[3] = 0.75; // 帧 6

        // 新窗口步长 4：帧 0,4,8,... —— 只有帧 4 与旧网格重合。
        const nextStartFrame = 0;
        const nextStride = 4;
        const nextEdit = windowOf({
            startFrame: nextStartFrame,
            stride: nextStride,
            length: 5,
            fill: 2,
        });

        const { values, drawnRange } = reanchorLiveEditWindow({
            edit,
            base,
            startFrame,
            stride,
            nextEdit,
            nextStartFrame,
            nextStride,
        });

        expect(values[1]).toBe(0.25); // 帧 4
        expect(values[2]).toBe(2); // 帧 8（旧窗口没有这个采样点）
        expect(drawnRange).toEqual({ lo: 1, hi: 1 });
    });

    /** 退化步长无法反解下标：保守地不搬（退化为"以新基准重建"）。 */
    it("旧窗口步长退化（0）→ 不搬任何帧", () => {
        const base = windowOf({ startFrame: 0, stride: 1, length: 4, fill: 1 });
        const edit = base.slice();
        edit[0] = 5;
        const nextEdit = windowOf({ startFrame: 0, stride: 1, length: 4, fill: 6 });
        const { values, drawnRange } = reanchorLiveEditWindow({
            edit,
            base,
            startFrame: 0,
            stride: 0,
            nextEdit,
            nextStartFrame: 0,
            nextStride: 1,
        });
        expect(values).toEqual(nextEdit);
        expect(drawnRange).toBeNull();
    });

    /** `base` 短于 `edit`（错位防御）：不得越界读取，也不得搬越界的帧。 */
    it("旧基准短于覆盖层时不越界读取", () => {
        const edit = [1, 1, 0.3, 0.4, 0.5];
        const base = [1, 1]; // 只有前两帧有基准
        const nextEdit = windowOf({ startFrame: 0, stride: 1, length: 5, fill: 2 });
        const { values, drawnRange } = reanchorLiveEditWindow({
            edit,
            base,
            startFrame: 0,
            stride: 1,
            nextEdit,
            nextStartFrame: 0,
            nextStride: 1,
        });
        expect(values).toEqual(nextEdit);
        expect(drawnRange).toBeNull();
    });
});

describe("笔画全程：窗口被换掉后轨迹仍在", () => {
    /**
     * 端到端（纯逻辑层）复现用户报告的场景：
     * pointerdown → 连续 pointermove 累积轨迹 → **窗口被换掉** → 继续 pointermove。
     *
     * 断言：换窗口之后，**已经画过的整条轨迹**仍在覆盖层里（旧实现在这一步整份
     * 清零，于是"按下那一点画出来了、之后轨迹一点都没有"）。
     */
    it("★ 换窗口后已画轨迹完整保留，并能继续往下画", () => {
        const startFrame = 200;
        const stride = 1;
        const length = 64;
        const committed = windowOf({ startFrame, stride, length, fill: 1 });

        // pointerdown：建立覆盖层（基准 = 已提交曲线）。
        let edit = committed.slice();

        /** 一次 pointermove：把 [minF, maxF] 的稠密值写进覆盖层。 */
        function stroke(minF: number, maxF: number, value: number) {
            const dense = new Array<number>(maxF - minF + 1).fill(value);
            writeDenseIntoLiveWindow({
                edit,
                orig: committed,
                startFrame,
                stride,
                dense,
                denseStartFrame: minF,
                minF,
                maxF,
                mode: "draw",
            });
        }

        stroke(210, 212, 0.5);
        stroke(213, 215, 0.6);
        // 三笔之后：帧 210..215 都是画过的值。
        const drawnBefore = edit.slice(210 - startFrame, 216 - startFrame);
        expect(drawnBefore).toEqual([0.5, 0.5, 0.5, 0.6, 0.6, 0.6]);

        // ★ 窗口被换掉（同作用域、同起帧、新基准 —— 即"同窗口重取数回包"）。
        const refetchedBase = windowOf({ startFrame, stride, length, fill: 1 });
        const reanchored = reanchorLiveEditWindow({
            edit,
            base: committed,
            startFrame,
            stride,
            nextEdit: refetchedBase,
            nextStartFrame: startFrame,
            nextStride: stride,
        });
        edit = reanchored.values;

        // 已画轨迹仍在（旧实现这里会变回全是 1）。
        expect(edit.slice(210 - startFrame, 216 - startFrame)).toEqual([
            0.5, 0.5, 0.5, 0.6, 0.6, 0.6,
        ]);

        // 继续画：新的一段照常写入。
        stroke(216, 218, 0.7);
        expect(edit[216 - startFrame]).toBe(0.7);
        expect(edit[218 - startFrame]).toBe(0.7);
    });

    /**
     * 换窗口后 `resetLiveEditPreview` 必须擦在**新窗口的下标**上：直线 / 颤音工具
     * 每帧"先擦上一帧、再画本帧"，区间若仍是旧下标，擦除会落在错误的帧上，
     * 旧预览残留在图上。
     */
    it("重锚后回滚区间落在新窗口的下标上", () => {
        const startFrame = 0;
        const base = windowOf({ startFrame, stride: 1, length: 16, fill: 1 });
        const edit = base.slice();
        // 上一帧的直线预览：帧 4..6。
        writeDenseIntoLiveWindow({
            edit,
            orig: base,
            startFrame,
            stride: 1,
            dense: [0.9, 0.9, 0.9],
            denseStartFrame: 4,
            minF: 4,
            maxF: 6,
            mode: "draw",
        });

        // 窗口左移 2 帧（起帧 -2），新基准同样是 1。
        const nextStartFrame = -2;
        const nextEdit = windowOf({
            startFrame: nextStartFrame,
            stride: 1,
            length: 16,
            fill: 1,
        });
        const { values, drawnRange } = reanchorLiveEditWindow({
            edit,
            base,
            startFrame,
            stride: 1,
            nextEdit,
            nextStartFrame,
            nextStride: 1,
        });

        // 帧 4..6 → 新下标 6..8。
        expect(drawnRange).toEqual({ lo: 6, hi: 8 });
        expect(values[6]).toBe(0.9);
        expect(values[8]).toBe(0.9);
    });
});
