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
    paramNeedingVisibilityOnGate,
    isGatedByCompose,
    isEffectParamGated,
    COMPOSE_GATED_PARAMS,
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

    describe("paramNeedingVisibilityOnGate（关开关后曲线仍需可见）", () => {
        it("回退发生时，被门禁的那个参数需要打开可见性", () => {
            // 曲线只在"是 editParam"或"可见性为真"时才绘制；回退会把它从
            // editParam 移走，所以必须同时打开它的可见性，否则曲线消失。
            expect(paramNeedingVisibilityOnGate("hifigan_tension", false, "pitch")).toBe(
                "hifigan_tension",
            );
            expect(paramNeedingVisibilityOnGate("breath_gain", false, "pitch")).toBe(
                "breath_gain",
            );
        });

        it("未发生回退时不需要改动可见性", () => {
            expect(paramNeedingVisibilityOnGate("hifigan_tension", true, "pitch")).toBeNull();
            expect(paramNeedingVisibilityOnGate("pitch", false, "pitch")).toBeNull();
            expect(
                paramNeedingVisibilityOnGate("formant_shift_cents", false, "pitch"),
            ).toBeNull();
        });

        it("与回退判据同源：需要回退 ⇔ 需要恢复可见性", () => {
            const params = ["pitch", "hifigan_tension", "breath_gain", "formant_shift_cents"];
            for (const on of [true, false]) {
                for (const id of params) {
                    const needsFallback = findBlockedEditParam(id, on, "pitch") !== null;
                    const needsVisibility =
                        paramNeedingVisibilityOnGate(id, on, "pitch") !== null;
                    expect(needsVisibility).toBe(needsFallback);
                }
            }
        });
    });


    describe("Compose 门禁（只影响合成参数，不影响混音参数）", () => {
        it("Compose 关闭时，共振峰/气声/张力都不可用", () => {
            for (const id of ["formant_shift_cents", "breath_gain", "hifigan_tension"]) {
                expect(isGatedByCompose(id, false)).toBe(true);
            }
        });

        it("Compose 开启时全部可用", () => {
            for (const id of COMPOSE_GATED_PARAMS) {
                expect(isGatedByCompose(id, true)).toBe(false);
            }
        });

        it("音量/声相/动态等混音级参数**不**受 Compose 门禁", () => {
            // 需求口径：Compose 只影响音高、共振峰、气声、张力；
            // 音量与声相是混音级参数，未开 Compose 也要生效。
            for (const id of ["volume", "pan", "dyn", "pitch"]) {
                expect(isGatedByCompose(id, false)).toBe(false);
            }
        });

        it("统一门禁：Compose 与分离开关任一关闭都会命中", () => {
            // Compose 关、分离开 ⇒ 命中（Compose 是更强的门禁）
            expect(isEffectParamGated("hifigan_tension", true, false)).toBe(true);
            // Compose 开、分离关 ⇒ 命中
            expect(isEffectParamGated("hifigan_tension", false, true)).toBe(true);
            // 两者都开 ⇒ 不命中
            expect(isEffectParamGated("hifigan_tension", true, true)).toBe(false);
        });

        it("共振峰只受 Compose 影响、不受分离开关影响", () => {
            // 共振峰走 mel 阶段，不依赖 HNSEP 分离。
            expect(isEffectParamGated("formant_shift_cents", false, true)).toBe(false);
            expect(isEffectParamGated("formant_shift_cents", true, false)).toBe(true);
        });
    });

});
