/**
 * `separationGate` 的单测。
 *
 * 覆盖「气声分离开关闭 ⇒ 气声音量与张力被门禁」的判据，以及 `editParam`
 * 回退的收敛性。这些是 UI 门禁与后端剥离共同依赖的口径 ——
 * 阈值若与后端 `extra_param_enabled` 分叉，会出现"UI 可用但后端已剥离"
 * （用户画了没效果）或"UI 置灰但实际生效"（用户以为没效果却听到了）。
 */

import { describe, expect, it } from "vitest";

import {
    SEPARATION_GATED_PARAMS,
    SEPARATION_PARAM_ID,
    findBlockedEditParam,
    isGatedBySeparation,
    isSeparationEnabled,
} from "./separationGate";

describe("separationGate", () => {
    it("把气声音量与张力列为被门禁参数，且不含共振峰", () => {
        expect(SEPARATION_GATED_PARAMS).toContain("breath_gain");
        expect(SEPARATION_GATED_PARAMS).toContain("hifigan_tension");
        // 共振峰走 mel 阶段的 keyShift，与 HNSEP 无关，必须不受门禁。
        expect(SEPARATION_GATED_PARAMS).not.toContain("formant_shift_cents");
    });

    it("开关阈值与后端 extra_param_enabled 同口径（>= 0.5 为开）", () => {
        expect(isSeparationEnabled(1, 0)).toBe(true);
        expect(isSeparationEnabled(0.5, 0)).toBe(true);
        expect(isSeparationEnabled(0.49, 0)).toBe(false);
        expect(isSeparationEnabled(0, 0)).toBe(false);
    });

    it("开关值缺失时按描述符默认值处理（默认关）", () => {
        expect(isSeparationEnabled(undefined, 0)).toBe(false);
        expect(isSeparationEnabled(undefined, 1)).toBe(true);
    });

    it("开关关闭时门禁两个参数，开启时都放行", () => {
        expect(isGatedBySeparation("breath_gain", false)).toBe(true);
        expect(isGatedBySeparation("hifigan_tension", false)).toBe(true);
        expect(isGatedBySeparation("breath_gain", true)).toBe(false);
        expect(isGatedBySeparation("hifigan_tension", true)).toBe(false);
    });

    it("开关关闭也不门禁共振峰与其它参数", () => {
        expect(isGatedBySeparation("formant_shift_cents", false)).toBe(false);
        expect(isGatedBySeparation("pitch", false)).toBe(false);
        expect(isGatedBySeparation("volume", false)).toBe(false);
    });

    it("editParam 停在被门禁参数上时回退到 pitch", () => {
        expect(findBlockedEditParam("breath_gain", false, "pitch")).toBe("pitch");
        expect(findBlockedEditParam("hifigan_tension", false, "pitch")).toBe("pitch");
    });

    it("开关开启或参数未被门禁时不回退", () => {
        expect(findBlockedEditParam("breath_gain", true, "pitch")).toBeNull();
        expect(findBlockedEditParam("formant_shift_cents", false, "pitch")).toBeNull();
        expect(findBlockedEditParam("pitch", false, "pitch")).toBeNull();
    });

    it("回退是收敛的：回退后的参数不再被判据命中", () => {
        // 若回退目标本身也被门禁，effect 会反复 dispatch 形成死循环。
        // 这里断言"回退一次即稳定"这一契约。
        const target = findBlockedEditParam("hifigan_tension", false, "pitch");
        expect(target).not.toBeNull();
        expect(findBlockedEditParam(target as string, false, "pitch")).toBeNull();
    });

    it("开关 id 与后端常量一致", () => {
        expect(SEPARATION_PARAM_ID).toBe("breath_enabled");
    });
});
