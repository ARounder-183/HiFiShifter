import { describe, expect, it } from "vitest";

import { createLiveOverrideReader } from "./liveLoudnessOverride";
import { createLiveEditOverride, projectLiveEdit } from "./strokeBuffer";
import type { LiveEditOverride, LiveProjectionWindow } from "./strokeBuffer";

/**
 * live 覆盖读取器。
 *
 * 【为什么值得单测】它是波形几何重建热路径上的唯一解析点：几何层按列内增益
 * 切片询问幅度映射（一次重建可达数万次），"按 (覆盖, 窗口, 缓冲版本) 缓存"这一条若失效
 * （例如每次新建视图对象），拖动音量/动态时就会重新出现数十毫秒级的帧 —— 而功能上
 * **完全看不出来**（数值仍然正确）。因此这里把"解析次数"与"对象复用"也钉住。
 */

const SCOPE = "v2|track-1|volume";

/** 造一份覆盖 + 一个投影窗口（窗口基准值故意与画过的值不同）。 */
function makeLive(
    values: readonly number[],
    options?: { scope?: string; startFrame?: number; stride?: number; base?: number[] },
): { live: LiveEditOverride; window: LiveProjectionWindow } {
    const scope = options?.scope ?? SCOPE;
    const startFrame = options?.startFrame ?? 120;
    const stride = options?.stride ?? 1;
    const live = createLiveEditOverride(scope, 1);
    live.buffer.write(startFrame, values);
    const base = options?.base ?? values.map(() => 0.5);
    return {
        live,
        window: {
            key: `@@{scope}|@@{startFrame}|@@{values.length}|@@{stride}`,
            edit: base,
            startFrame,
            stride,
        },
    };
}

describe("createLiveOverrideReader", () => {
    it("returns the curve view for the edited parameter and null for others", () => {
        const reader = createLiveOverrideReader();
        const { live, window } = makeLive([0.5, 0.6, 0.7]);

        const view = reader.read("volume", live, window);
        expect(view).not.toBeNull();
        expect(view?.startFrame).toBe(120);
        expect(view?.stride).toBe(1);
        // 覆盖的每一帧都画过 → 投影 == 画过的值，基准不参与。
        expect(Array.from(view?.values ?? [])).toEqual([0.5, 0.6, 0.7]);
        // 同一份覆盖只属于它编辑的那个参数。
        expect(reader.read("dyn", live, window)).toBeNull();
    });

    it("treats the dyn legacy alias as dyn", () => {
        const reader = createLiveOverrideReader();
        const { live, window } = makeLive([1, 2], { scope: "v2|track-1|dyn_edit" });

        expect(reader.read("dyn", live, window)).not.toBeNull();
        expect(reader.read("volume", live, window)).toBeNull();
    });

    it("projects onto the window the caller passes, not onto a key buried in the override", () => {
        // 覆盖只画了 4 帧；窗口更宽 → 未覆盖的帧取窗口基准值。
        const reader = createLiveOverrideReader();
        const { live } = makeLive([9, 9, 9, 9], { startFrame: 0, base: [1, 1, 1, 1, 1, 1, 1, 1] });
        const window: LiveProjectionWindow = {
            key: "v2|track-1|volume|0|8|1",
            edit: [1, 1, 1, 1, 1, 1, 1, 1],
            startFrame: 0,
            stride: 1,
        };

        const view = reader.read("volume", live, window);
        expect(Array.from(view?.values ?? [])).toEqual([9, 9, 9, 9, 1, 1, 1, 1]);
    });

    it("returns null when there is no live override", () => {
        const reader = createLiveOverrideReader();
        const { window } = makeLive([1]);

        expect(reader.read("volume", null, window)).toBeNull();
        expect(reader.read("dyn", null, window)).toBeNull();
    });

    it("returns null when there is no window yet", () => {
        const reader = createLiveOverrideReader();
        const { live } = makeLive([1]);

        expect(reader.read("volume", live, null)).toBeNull();
    });

    it("treats an empty buffer as no override", () => {
        const reader = createLiveOverrideReader();
        const live = createLiveEditOverride(SCOPE, 1);
        const window: LiveProjectionWindow = {
            key: "v2|t|volume|0|4|1",
            edit: [1, 1, 1, 1],
            startFrame: 0,
            stride: 1,
        };

        expect(reader.read("volume", live, window)).toBeNull();
        expect(reader.affectsWaveform(live)).toBe(false);
    });

    it("★ reuses the same view object across calls (steady-state zero allocation)", () => {
        // 热路径契约：同一份覆盖、同一窗口 → 每次都拿到**同一个**视图对象。
        // 逐次新建会让一次几何重建产生数万个短命对象（GC 拖进渲染关键路径）。
        const reader = createLiveOverrideReader();
        const { live, window } = makeLive([1, 2, 3]);

        const first = reader.read("volume", live, window);
        for (let i = 0; i < 1000; i += 1) {
            expect(reader.read("volume", live, window)).toBe(first);
        }
    });

    it("★ follows in-place edits of the same buffer", () => {
        // 拖动中缓冲是**原地增长**的，因此视图必须跟着变 —— 若在读取时把投影结果
        // 缓存住不失效，波形就会停在拖动起点。
        const reader = createLiveOverrideReader();
        const { live, window } = makeLive([1, 1, 1, 1]);

        expect(Array.from(reader.read("volume", live, window)?.values ?? [])).toEqual([1, 1, 1, 1]);
        live.buffer.write(121, [0.25]);
        expect(Array.from(reader.read("volume", live, window)?.values ?? [])).toEqual([
            1, 0.25, 1, 1,
        ]);
    });

    it("repairs when the window changes", () => {
        const reader = createLiveOverrideReader();
        const { live } = makeLive([1, 1], { startFrame: 0, base: [0, 0, 0, 0, 0, 0] });
        const first = reader.read("volume", live, {
            key: "v2|t|volume|0|2|1",
            edit: [0, 0],
            startFrame: 0,
            stride: 1,
        });
        const second = reader.read("volume", live, {
            key: "v2|t|volume|2|4|2",
            edit: [0, 0, 0, 0],
            startFrame: 2,
            stride: 2,
        });

        expect(Array.from(first?.values ?? [])).toEqual([1, 1]);
        // 窗口 [2,4,6,8]，覆盖只到帧 1 → 全部取基准值。
        expect(Array.from(second?.values ?? [])).toEqual([0, 0, 0, 0]);
        expect(second).not.toBe(first);
    });

    it("drops the cache when the override disappears", () => {
        const reader = createLiveOverrideReader();
        const { live, window } = makeLive([1]);
        expect(reader.read("dyn", live, window)).toBeNull();
        expect(reader.read("volume", live, window)).not.toBeNull();

        expect(reader.read("volume", null, window)).toBeNull();
        // 覆盖回来后重新投影（不是复用旧缓存）。
        expect(Array.from(reader.read("volume", live, window)?.values ?? [])).toEqual([1]);
    });

    it("does not confuse two different overrides with the same scope", () => {
        const reader = createLiveOverrideReader();
        const a = makeLive([1, 1]);
        const b = makeLive([0, 0]);
        // 两份覆盖的窗口 key 相同（同一作用域同一窗口），只有对象不同。
        expect(b.window.key).toBe(a.window.key);

        expect(Array.from(reader.read("volume", a.live, a.window)?.values ?? [])).toEqual([1, 1]);
        expect(Array.from(reader.read("volume", b.live, b.window)?.values ?? [])).toEqual([0, 0]);
    });

    it("keeps the projection cache on the override (shared with the curve layer)", () => {
        // 曲线层与波形幅度映射必须看到**同一份**投影：两处各自投影会让"屏幕上画的"
        // 与"波形按它算的"来自两次不同的计算。
        const reader = createLiveOverrideReader();
        const { live, window } = makeLive([4, 5]);

        const view = reader.read("volume", live, window);
        expect(view?.values).toBe(projectLiveEdit(live, window));
    });
});

/**
 * `affectsWaveform`：面板据此决定是否在绘制中强制重建波形几何。
 *
 * 【为什么必须准确】波形画的是「可听结果」（volume × dyn），与音高 / 共振峰
 * 无关。若对音高编辑也返回 true，绘制时每帧都要重建两千余列的包络（用户
 * 报告的另一种卡顿）；反之若对音量编辑返回 false，波形就不再实时跟随。
 */
describe("LiveOverrideReader.affectsWaveform", () => {
    it("is true for volume and dyn (including the dyn legacy alias)", () => {
        const reader = createLiveOverrideReader();
        for (const scope of ["v2|t|volume", "v2|t|dyn", "v2|t|dyn_edit"]) {
            const { live } = makeLive([1], { scope });
            expect(reader.affectsWaveform(live)).toBe(true);
        }
    });

    it("is false for parameters the waveform does not depend on", () => {
        const reader = createLiveOverrideReader();
        // 波形不依赖这些参数：绘制它们时不该重建波形几何。
        for (const paramId of ["pitch", "formant", "tension", "breathiness"]) {
            const { live } = makeLive([1], { scope: `v2|t|${paramId}` });
            expect(reader.affectsWaveform(live)).toBe(false);
        }
    });

    it("is false when there is no override or the buffer is empty", () => {
        const reader = createLiveOverrideReader();
        expect(reader.affectsWaveform(null)).toBe(false);
        expect(reader.affectsWaveform(createLiveEditOverride("v2|t|volume", 1))).toBe(false);
    });

    it("shares the projection cache with read()", () => {
        // 两个入口共用同一份缓存：不会因为多调用一个入口而重复投影
        // （拖动时两者都在热路径上被逐帧调用）。
        const reader = createLiveOverrideReader();
        const { live, window } = makeLive([1, 2], { startFrame: 32, stride: 2 });

        expect(reader.affectsWaveform(live)).toBe(true);
        const view = reader.read("volume", live, window);
        expect(view?.startFrame).toBe(32);
        expect(view?.stride).toBe(2);
        // 缓存未因两次入口而失效：同一份覆盖仍返回同一个视图对象。
        expect(reader.read("volume", live, window)).toBe(view);
    });
});
