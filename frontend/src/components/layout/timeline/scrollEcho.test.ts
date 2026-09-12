/**
 * 时间轴内核 · 轨道头滚动回灌判定单测。
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

import { isTrackListMirrorEcho } from "./scrollEcho";

describe("isTrackListMirrorEcho（轨道头镜像回声判定）", () => {
    it("★ 事件值等于上次镜像写入值时判为回声", () => {
        // 实测：syncDom 写 28、事件报 28（浏览器量化），内核此刻已到 37.3。
        // 旧实现拿 28 与**内核当前值** 37.3 比（差 9.3 > 0.5），于是收下 → 回退 9.3px。
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 28, nativeScrollTop: 28 })).toBe(true);
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 46.5, nativeScrollTop: 46.5 })).toBe(
            true,
        );
    });

    it("★ 真实 scroll-into-view（值与镜像不同）不判回声", () => {
        // 实测：聚焦最后一个控件把容器从 0 带到 308；PageDown 到 132；End 到 361。
        // 这些必须回灌内核，否则焦点导航时轨道头与时间轴会脱节。
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 0, nativeScrollTop: 308 })).toBe(false);
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 0, nativeScrollTop: 132 })).toBe(false);
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 124.5, nativeScrollTop: 361 })).toBe(
            false,
        );
    });

    it("★ 内核推进导致的差值不影响回声判定（判据是镜像值，不是内核值）", () => {
        // 这是本模块存在的核心理由：拖拽时内核每帧前进，事件值永远落后一帧。
        // 若拿内核当前值当基准就会漏判（旧实现的 bug）。镜像值一致即回声。
        const kernelNow = 37.3;
        const mirrored = 28;
        const native = 28;
        expect(Math.abs(native - kernelNow)).toBeGreaterThan(0.5); // 旧判据会误收
        expect(
            isTrackListMirrorEcho({ mirroredScrollTop: mirrored, nativeScrollTop: native }),
        ).toBe(true);
    });

    it("★ 从未回写过（NaN）不判回声——否则吞掉首帧前的真实输入", () => {
        expect(isTrackListMirrorEcho({ mirroredScrollTop: Number.NaN, nativeScrollTop: 120 })).toBe(
            false,
        );
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 120, nativeScrollTop: Number.NaN })).toBe(
            false,
        );
    });

    it("量化造成的微小差异（≤ 容差）仍算回声", () => {
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 124.5, nativeScrollTop: 124.5 })).toBe(
            true,
        );
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 66, nativeScrollTop: 66 })).toBe(true);
        // 0 也是合法位置（顶部），不能因为"假值"被漏判。
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 0, nativeScrollTop: 0 })).toBe(true);
    });

    it("超出容差的差异不判回声", () => {
        expect(isTrackListMirrorEcho({ mirroredScrollTop: 28, nativeScrollTop: 29 })).toBe(false);
    });

    it("容差可显式覆盖（0 = 严格相等）", () => {
        expect(
            isTrackListMirrorEcho({
                mirroredScrollTop: 28,
                nativeScrollTop: 28.4,
                tolerancePx: 0,
            }),
        ).toBe(false);
        expect(
            isTrackListMirrorEcho({
                mirroredScrollTop: 28,
                nativeScrollTop: 28,
                tolerancePx: 0,
            }),
        ).toBe(true);
    });

    it("非法容差退回默认值（不产生 NaN 判定）", () => {
        expect(
            isTrackListMirrorEcho({
                mirroredScrollTop: 100,
                nativeScrollTop: 100,
                tolerancePx: Number.NaN,
            }),
        ).toBe(true);
    });
});
