import { describe, expect, it } from "vitest";
import { findConflicts } from "./keybindingsSlice";
import { DEFAULT_KEYBINDINGS } from "./defaultKeybindings";
import type { KeybindingOverrides } from "./types";
import { createModifierOnlyBinding } from "./keybindingsSlice";

/** 构造修饰键绑定 */
function mod(modifier: "control" | "shift" | "alt"): ReturnType<typeof createModifierOnlyBinding> {
    return createModifierOnlyBinding({
        ctrl: modifier === "control",
        shift: modifier === "shift",
        alt: modifier === "alt",
    });
}

describe("findConflicts — 修饰键按场景检测", () => {
    it("不同场景重复使用同一修饰键不算冲突（Alt：Slip vs 淡化曲率）", () => {
        // fadeCurvatureDrag 默认即 Alt；clipSlipEdit 绑定到 Alt 属于不同拖拽目标
        const conflicts = findConflicts({}, "modifier.clipSlipEdit", mod("alt"));
        expect(conflicts).not.toContain("modifier.fadeCurvatureDrag");
        expect(conflicts).not.toContain("modifier.paramMorph");
        expect(conflicts).not.toContain("modifier.clipStretch");
    });

    it("滚轮修饰键：颤音滚轮与画布滚动滚轮是不同场景，不互相冲突", () => {
        // scrollHorizontal 默认 Shift；vibratoFrequencyAdjust 绑到 Shift 不影响画布滚动
        const conflicts = findConflicts({}, "modifier.vibratoFrequencyAdjust", mod("shift"));
        expect(conflicts).not.toContain("modifier.scrollHorizontal");
    });

    it("同一场景内键值相同才算冲突（clip.move：Slip vs 复制拖动）", () => {
        // clipCopyDrag 默认 Ctrl；把 clipSlipEdit 也绑到 Ctrl → 音频块主体拖拽歧义
        const conflicts = findConflicts({}, "modifier.clipSlipEdit", mod("control"));
        expect(conflicts).toContain("modifier.clipCopyDrag");
    });

    it("同一场景内键值相同才算冲突（clip.crossfade：手柄 vs 曲率）", () => {
        // fadeCurvatureDrag 默认 Alt；交叉点手柄也绑 Alt → 同一拖拽目标歧义
        const conflicts = findConflicts({}, "modifier.clipCrossfadeGrip", mod("alt"));
        expect(conflicts).toContain("modifier.fadeCurvatureDrag");
    });

    it("共享场景的修饰键（clipNoSnap 跨多个场景）在任一共享场景冲突", () => {
        // clipNoSnap 默认 Shift 且作用于 clip.edge；把 clipStretch（同为边缘拖拽）绑到 Shift
        const conflicts = findConflicts({}, "modifier.clipStretch", mod("shift"));
        expect(conflicts).toContain("modifier.clipNoSnap");
    });

    it("手势类型不同不冲突（淡化包络：点击 vs 拖拽）", () => {
        // fadeShapeCycleClick 是点击手势；fadeCurvatureDrag 是拖拽手势
        const conflicts = findConflicts({}, "modifier.fadeShapeCycleClick", mod("alt"));
        expect(conflicts).not.toContain("modifier.fadeCurvatureDrag");
    });

    it("手势类型不同不冲突（滚轮 vs 拖拽）", () => {
        // scrollVertical（滚轮）与 clipStretch（拖拽）同用 Alt 互不影响
        const conflicts = findConflicts({}, "modifier.scrollVertical", mod("alt"));
        expect(conflicts).not.toContain("modifier.clipStretch");
    });

    it("琴键区滚轮与画布滚轮是不同场景（pianoKeysVerticalZoom vs scrollVertical）", () => {
        const conflicts = findConflicts({}, "modifier.pianoKeysVerticalZoom", mod("alt"));
        expect(conflicts).not.toContain("modifier.scrollVertical");
    });

    it("同场景滚轮键值相同仍算冲突（wheel.timeline：竖直滚动 vs 竖直缩放）", () => {
        // pianoRollVerticalZoom 默认 Ctrl 且同时作用于时间轴与钢琴卷帘滚轮
        const conflicts = findConflicts({}, "modifier.scrollVertical", mod("control"));
        expect(conflicts).toContain("modifier.pianoRollVerticalZoom");
    });

    it("颤音滚轮修饰键同时为“无”仍视为冲突", () => {
        const overrides: KeybindingOverrides = {
            "modifier.vibratoFrequencyAdjust": { key: "__none__", modifierOnly: true },
        };
        const conflicts = findConflicts(overrides, "modifier.vibratoAmplitudeAdjust", {
            key: "__none__",
            modifierOnly: true,
        });
        expect(conflicts).toContain("modifier.vibratoFrequencyAdjust");
    });

    it("同场景点击修饰键冲突（clip.select：多选切换 vs 范围选择）", () => {
        // 两个选择修饰键都被绑到同一组合时点击语义歧义
        const conflicts = findConflicts({}, "modifier.clipMultiSelectToggle", mod("shift"));
        expect(conflicts).toContain("modifier.clipRangeSelect");
    });

    it("选择修饰键（点击）与拖拽修饰键同键不冲突（clipNoSnap 默认 Shift）", () => {
        // clipRangeSelect 默认 Shift；clipNoSnap 是拖拽手势，点击/拖拽可区分
        const conflicts = findConflicts({}, "modifier.clipRangeSelect", mod("shift"));
        expect(conflicts).not.toContain("modifier.clipNoSnap");
    });
});

describe("findConflicts — 键盘快捷键按作用域检测", () => {
    it("quickSearch 作用域内的绑定不与全局绑定冲突", () => {
        // quickSearch.confirm 默认 enter；playback.stop 默认也是 enter，但作用域不同
        const conflicts = findConflicts({}, "quickSearch.confirm", DEFAULT_KEYBINDINGS["playback.stop"]);
        expect(conflicts).not.toContain("playback.stop");
    });

    it("全局作用域内键值相同即冲突", () => {
        const conflicts = findConflicts({}, "clip.split", { key: "g" });
        expect(conflicts).toContain("clip.group");
    });
});
