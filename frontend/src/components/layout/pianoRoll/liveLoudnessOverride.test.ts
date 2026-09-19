import { describe, expect, it } from "vitest";

import { createLiveOverrideReader } from "./liveLoudnessOverride";

/**
 * live 覆盖读取器。
 *
 * 【为什么值得单测】它是波形几何重建热路径上的唯一解析点：几何层按列内增益
 * 切片询问幅度映射（一次重建可达数万次），"按覆盖对象身份缓存"这一条若失效
 * （例如改成按 key 字符串比较、或每次新建视图对象），拖动音量/动态时就会
 * 重新出现数十毫秒级的帧 —— 而功能上**完全看不出来**（数值仍然正确）。
 * 因此这里把"解析次数"与"对象复用"也钉住。
 */
describe("createLiveOverrideReader", () => {
    it("returns the curve view for the edited parameter and null for others", () => {
        const reader = createLiveOverrideReader();
        const live = { key: "v2|track-1|volume|120|40|1", edit: [0.5, 0.6, 0.7] };

        expect(reader.read("volume", live)).toEqual({
            startFrame: 120,
            stride: 1,
            values: live.edit,
        });
        // 同一份覆盖只属于它编辑的那个参数。
        expect(reader.read("dyn", live)).toBeNull();
    });

    it("treats the dyn legacy alias as dyn", () => {
        const reader = createLiveOverrideReader();
        const live = { key: "v2|track-1|dyn_edit|0|16|2", edit: [1, 2] };

        expect(reader.read("dyn", live)).toEqual({ startFrame: 0, stride: 2, values: live.edit });
        expect(reader.read("volume", live)).toBeNull();
    });

    it("parses the window (startFrame / stride) out of the key", () => {
        const reader = createLiveOverrideReader();
        const live = { key: "v2|track-9|dyn|48000|512|4", edit: [1] };

        expect(reader.read("dyn", live)).toEqual({
            startFrame: 48000,
            stride: 4,
            values: live.edit,
        });
    });

    it("falls back to sane defaults for a malformed key", () => {
        const reader = createLiveOverrideReader();
        // 缺字段 / 非数字：起点按 0、步长按 1（与旧实现 `Number(x) || 0` 同口径）。
        const live = { key: "v2|track-1|volume", edit: [1, 1] };

        expect(reader.read("volume", live)).toEqual({
            startFrame: 0,
            stride: 1,
            values: live.edit,
        });
    });

    it("returns null when there is no live override", () => {
        const reader = createLiveOverrideReader();

        expect(reader.read("volume", null)).toBeNull();
        expect(reader.read("dyn", null)).toBeNull();
    });

    it("treats an empty edit array as no override", () => {
        const reader = createLiveOverrideReader();

        expect(reader.read("volume", { key: "v2|t|volume|0|0|1", edit: [] })).toBeNull();
    });

    it("★ reuses the same view object across calls (steady-state zero allocation)", () => {
        // 热路径契约：同一份覆盖、同一参数 → 每次都拿到**同一个**视图对象。
        // 逐次新建会让一次几何重建产生数万个短命对象（GC 拖进渲染关键路径）。
        const reader = createLiveOverrideReader();
        const live = { key: "v2|track-1|volume|0|64|1", edit: [1, 2, 3] };

        const first = reader.read("volume", live);
        for (let i = 0; i < 1000; i += 1) {
            expect(reader.read("volume", live)).toBe(first);
        }
    });

    it("★ repairs when the override object identity changes", () => {
        // 覆盖的每次更新都写新对象（edit 数组原地更新，故不能只比数组引用）。
        // 对象换了 = 该重解析：新窗口 / 新步长必须被采纳。
        const reader = createLiveOverrideReader();
        const first = reader.read("volume", { key: "v2|t|volume|0|8|1", edit: [1, 1] });
        expect(first).toEqual({ startFrame: 0, stride: 1, values: [1, 1] });

        const edit = [2, 2];
        const second = reader.read("volume", { key: "v2|t|volume|16|8|3", edit });
        expect(second).toEqual({ startFrame: 16, stride: 3, values: edit });
        expect(second).not.toBe(first);
    });

    it("★ follows in-place edits of the same override object", () => {
        // 拖动中 edit 数组是**原地更新**的（applyDenseToLiveEdit 每次都新建
        // 覆盖对象但复用数组内容），因此视图必须引用同一个数组、读者能看见
        // 最新值 —— 若在读取时把数组拷贝出来，波形就会停在拖动起点。
        const reader = createLiveOverrideReader();
        const live = { key: "v2|t|volume|0|4|1", edit: [1, 1, 1, 1] };
        const view = reader.read("volume", live);
        expect(view?.values[0]).toBe(1);

        live.edit[0] = 0.25;
        expect(reader.read("volume", live)?.values[0]).toBe(0.25);
    });

    it("drops the cache when the override disappears", () => {
        const reader = createLiveOverrideReader();
        const live = { key: "v2|t|dyn|0|4|1", edit: [1] };
        expect(reader.read("dyn", live)).not.toBeNull();

        expect(reader.read("dyn", null)).toBeNull();
        // 覆盖回来后重新解析（不是复用旧缓存）。
        expect(reader.read("dyn", live)).toEqual({ startFrame: 0, stride: 1, values: live.edit });
    });

    it("does not confuse two different overrides with the same key text", () => {
        // 身份判定的反面用例：key 文本相同但对象不同 → 仍必须重解析
        // （两条不同轨道的 volume 曲线可以有同样的窗口）。
        const reader = createLiveOverrideReader();
        const key = "v2|track-1|volume|0|4|1";
        const a = reader.read("volume", { key, edit: [1, 1] });
        const b = reader.read("volume", { key, edit: [0, 0] });

        expect(a?.values).toEqual([1, 1]);
        expect(b?.values).toEqual([0, 0]);
        expect(b).not.toBe(a);
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
        expect(reader.affectsWaveform({ key: "v2|t|volume|0|4|1", edit: [1] })).toBe(true);
        expect(reader.affectsWaveform({ key: "v2|t|dyn|0|4|1", edit: [1] })).toBe(true);
        expect(reader.affectsWaveform({ key: "v2|t|dyn_edit|0|4|1", edit: [1] })).toBe(true);
    });

    it("is false for parameters the waveform does not depend on", () => {
        const reader = createLiveOverrideReader();
        // 波形不依赖这些参数：绘制它们时不该重建波形几何。
        for (const paramId of ["pitch", "formant", "tension", "breathiness"]) {
            expect(reader.affectsWaveform({ key: `v2|t|${paramId}|0|4|1`, edit: [1] })).toBe(false);
        }
    });

    it("is false when there is no override or the edit array is empty", () => {
        const reader = createLiveOverrideReader();
        expect(reader.affectsWaveform(null)).toBe(false);
        expect(reader.affectsWaveform({ key: "v2|t|volume|0|0|1", edit: [] })).toBe(false);
    });

    it("shares the parse cache with read()", () => {
        // 两个入口共用同一份缓存：不会因为多调用一个入口而重复解析
        // （拖动时两者都在热路径上被逐帧调用）。
        const reader = createLiveOverrideReader();
        const live = { key: "v2|t|volume|32|8|2", edit: [1, 2] };

        expect(reader.affectsWaveform(live)).toBe(true);
        const view = reader.read("volume", live);
        expect(view).toEqual({ startFrame: 32, stride: 2, values: live.edit });
        // 缓存未因两次入口而失效：同一份覆盖仍返回同一个视图对象。
        expect(reader.read("volume", live)).toBe(view);
    });
});
