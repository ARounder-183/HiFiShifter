/**
 * 原生滚动容器的镜像回声判定单测（时间轴轨道头 + 参数编辑器 scroller 共用）。
 *
 * 【本测试要钉住的核心不变量】
 * 1. **镜像回声必须被识别**：宿主 `syncDom` 写进容器的值，其随之而来的 `scroll`
 *    事件必须判为回声，否则会把内核拖回去（实测每次 −9px，用户感受为"吸附感"）；
 * 2. **真实输入必须不被误判**：浏览器原生 scroll-into-view（Tab / PageDown / End
 *    切换焦点）报来的值必须判为非回声，否则键盘/焦点导航会失效；
 * 3. **从未回写过（NaN）时一律非回声**，保证首帧前的输入不被吞。
 *
 * 用例里的数值全部取自浏览器实测（1920×1200、40 步拖竖向滚动条）。
 */
import { describe, expect, it } from "vitest";

import { isMirrorEcho } from "./scrollEcho";

describe("isMirrorEcho（原生滚动容器的镜像回声判定）", () => {
    it("★ 事件值等于上次镜像写入值时判为回声", () => {
        // 实测：syncDom 写 28、事件报 28（浏览器量化），内核此刻已到 37.3。
        // 旧实现拿 28 与**内核当前值** 37.3 比（差 9.3 > 0.5），于是收下 → 回退 9.3px。
        expect(isMirrorEcho({ mirroredPx: 28, nativePx: 28 })).toBe(true);
        expect(isMirrorEcho({ mirroredPx: 46.5, nativePx: 46.5 })).toBe(true);
    });

    it("★ 真实 scroll-into-view（值与镜像不同）不判回声", () => {
        // 实测：聚焦最后一个控件把容器从 0 带到 308；PageDown 到 132；End 到 361。
        // 这些必须回灌内核，否则焦点导航时轨道头与时间轴会脱节。
        expect(isMirrorEcho({ mirroredPx: 0, nativePx: 308 })).toBe(false);
        expect(isMirrorEcho({ mirroredPx: 0, nativePx: 132 })).toBe(false);
        expect(isMirrorEcho({ mirroredPx: 124.5, nativePx: 361 })).toBe(false);
    });

    it("★ 内核推进导致的差值不影响回声判定（判据是镜像值，不是内核值）", () => {
        // 这是本模块存在的核心理由：拖拽时内核每帧前进，事件值永远落后一帧。
        // 若拿内核当前值当基准就会漏判（旧实现的 bug）。镜像值一致即回声。
        const kernelNow = 37.3;
        const mirrored = 28;
        const native = 28;
        expect(Math.abs(native - kernelNow)).toBeGreaterThan(0.5); // 旧判据会误收
        expect(isMirrorEcho({ mirroredPx: mirrored, nativePx: native })).toBe(true);
    });

    it("★ 从未回写过（NaN）不判回声——否则吞掉首帧前的真实输入", () => {
        expect(isMirrorEcho({ mirroredPx: Number.NaN, nativePx: 120 })).toBe(false);
        expect(isMirrorEcho({ mirroredPx: 120, nativePx: Number.NaN })).toBe(false);
    });

    it("量化造成的微小差异（≤ 容差）仍算回声", () => {
        expect(isMirrorEcho({ mirroredPx: 124.5, nativePx: 124.5 })).toBe(true);
        expect(isMirrorEcho({ mirroredPx: 66, nativePx: 66 })).toBe(true);
        // 0 也是合法位置（顶部），不能因为"假值"被漏判。
        expect(isMirrorEcho({ mirroredPx: 0, nativePx: 0 })).toBe(true);
    });

    it("★ 容差内的**非零**差异仍算回声（实测量化误差，不是罕见情形）", () => {
        // 【为什么要单独列非零差异】上面那批用例的值全都**完全相等**（差 0），
        // 于是"容差"本身根本没被断言：把实现里的 `<= tolerance` 改成 `<= 0`
        // （即取消容差）后整套用例照样通过——而容差正是本模块存在的理由。
        //
        // 实测：原生 scrollTop 被浏览器按设备像素量化，写→读误差是**常规**且有界的，
        // 不是注释里说的"极端二次量化"：
        // - dpr 1 → 可达 0.5（如写 20.5 读回 21）
        // - dpr 2 → 0.25（如写 540.694 读回 540.5、写 124.4 读回 124.5）
        // - dpr 3 → ≈0.167
        expect(isMirrorEcho({ mirroredPx: 540.5, nativePx: 540.69 })).toBe(true);
        expect(isMirrorEcho({ mirroredPx: 124.5, nativePx: 124.4 })).toBe(true);
        // dpr 1 最坏：正好 0.5，落在**闭区间**边界上（`<=` 是刻意包含的）
        expect(isMirrorEcho({ mirroredPx: 20.5, nativePx: 21 })).toBe(true);
        // dpr 3
        expect(isMirrorEcho({ mirroredPx: 30.167, nativePx: 30 })).toBe(true);
    });

    it("★ 刚超出容差的差异不判回声（容差不能被悄悄放大）", () => {
        // 与上一条成对：判据必须有两侧。只测"内部为真"时，把容差从 0.5 放大到 100
        // 也不会被发现，而那样会把真实的焦点滚动输入吞掉。
        expect(isMirrorEcho({ mirroredPx: 28, nativePx: 28.6 })).toBe(false);
        expect(isMirrorEcho({ mirroredPx: 20.5, nativePx: 21.1 })).toBe(false);
    });

    it("超出容差的差异不判回声", () => {
        expect(isMirrorEcho({ mirroredPx: 28, nativePx: 29 })).toBe(false);
    });

    it("容差可显式覆盖（0 = 严格相等）", () => {
        expect(
            isMirrorEcho({
                mirroredPx: 28,
                nativePx: 28.4,
                tolerancePx: 0,
            }),
        ).toBe(false);
        expect(
            isMirrorEcho({
                mirroredPx: 28,
                nativePx: 28,
                tolerancePx: 0,
            }),
        ).toBe(true);
    });

    it("参数编辑器的横向场景：内核已前进整帧、事件报上次写入值 → 仍判回声", () => {
        // 连续滚动 / 缩放中：内核此刻已在 548.054，上一帧写进容器的是 540.5，
        // 事件按 dpr=2 量化读回 540.69。
        expect(isMirrorEcho({ mirroredPx: 540.5, nativePx: 540.69 })).toBe(true);
        // 对照：触摸拖拽把容器带到明显不同的位置 → 必须采纳（C4 要支持的那三类输入）。
        expect(isMirrorEcho({ mirroredPx: 540.5, nativePx: 812 })).toBe(false);
    });

    it("非法容差退回默认值（不产生 NaN 判定）", () => {
        expect(
            isMirrorEcho({
                mirroredPx: 100,
                nativePx: 100,
                tolerancePx: Number.NaN,
            }),
        ).toBe(true);
    });
});
