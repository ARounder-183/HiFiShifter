/**
 * MIDI 音高曲线生成单测。
 *
 * 【主要内容】覆盖 `midiPitchCurve.ts` 的两组核心数学：
 * `generateMidiCurveFromNotes`（音符 → 逐帧音高曲线）与
 * `resolveLoopCycleDescriptor`（Loop 回绕周期/锚点解析）。
 *
 * 【作用】把"搬迁过程中被写错但不会报错"的数学钉死：曲线铺放、trim、
 * playbackRate 拉伸、reversed 镜像、fillGaps 与 Loop 周期退化。
 *
 * 【与其他模块的关系】被测模块从孤儿组件 `components/waveform/MidiPitchTrackCanvas.tsx`
 * 逐字搬来，与后端 `emit_clip_pitch_data_for_clip` 的 MIDI 分支逐帧对齐。
 * 这里的断言按**搬迁前**的真实行为取值——它们是这次搬迁的回归守卫。
 *
 * 【为什么必须有】这些数学的任何一处偏差都不会抛错，只会让时间线折线与
 * 音频/后端推送曲线之间出现**恒定相位差**，极难归因。
 */
import { describe, expect, it } from "vitest";

import { generateMidiCurveFromNotes, resolveLoopCycleDescriptor } from "./midiPitchCurve";

const NOTES = [
    { startSec: 0, endSec: 1, note: 60 },
    { startSec: 1, endSec: 2, note: 64 },
];

describe("generateMidiCurveFromNotes", () => {
    it("基本铺放：音符区间内的帧取该音高", () => {
        const curve = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 1, false, false, null);
        expect(curve[0]).toBe(60);
        expect(curve[curve.length - 1]).toBe(64);
    });

    it("trim（源窗口）只保留窗口内音符", () => {
        const curve = generateMidiCurveFromNotes(NOTES, 1, 1, 2, 1, false, false, null);
        expect(curve.every((v) => v === 64 || v === 0)).toBe(true);
        expect(curve.includes(60)).toBe(false);
    });

    it("playbackRate 拉伸：速率 2 时内容占更少帧", () => {
        const normal = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 1, false, false, null);
        const fast = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 2, false, false, null);
        expect(fast.lastIndexOf(64)).toBeLessThan(normal.lastIndexOf(64));
    });

    it("reversed 倒放：音高顺序反转", () => {
        const curve = generateMidiCurveFromNotes(NOTES, 2, 0, 2, 1, true, false, null);
        expect(curve[0]).toBe(64);
    });

    it("fillGaps 填补空隙", () => {
        const gapped = [
            { startSec: 0, endSec: 0.5, note: 60 },
            { startSec: 1.5, endSec: 2, note: 67 },
        ];
        const filled = generateMidiCurveFromNotes(gapped, 2, 0, 2, 1, false, true, null);
        expect(filled.every((v) => v > 0)).toBe(true);
    });

    it("无音符 / 零时长不抛错", () => {
        expect(() => generateMidiCurveFromNotes([], 1, 0, 1, 1, false, false, null)).not.toThrow();
        expect(() =>
            generateMidiCurveFromNotes(NOTES, 0, 0, 0, 1, false, false, null),
        ).not.toThrow();
    });
});

describe("resolveLoopCycleDescriptor", () => {
    it("loopEnabled 为假 → null", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: false,
                contentDurationSec: 4,
                sourceStartSec: 0,
                sourceEndSec: 2,
            }),
        ).toBeNull();
    });

    it("有媒体时长：周期取媒体时长、cycleFromMedia 为真", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: 4,
                sourceStartSec: 1,
                sourceEndSec: 3,
            }),
        ).toEqual({ cycleSec: 4, fwdAnchorSec: 1, revAnchorEndSec: 3, cycleFromMedia: true });
    });

    it("纯 MIDI（无媒体时长）：退化为窗口跨度、不 clamp 倒放锚点", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: null,
                sourceStartSec: 2,
                sourceEndSec: 7,
            }),
        ).toEqual({ cycleSec: 5, fwdAnchorSec: 2, revAnchorEndSec: 7, cycleFromMedia: false });
    });

    it("倒放锚点只 clamp 到媒体时长上界、不做 max(0)", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: 2,
                sourceStartSec: -1,
                sourceEndSec: -0.5,
            })?.revAnchorEndSec,
        ).toBe(-0.5);
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: 2,
                sourceStartSec: 0,
                sourceEndSec: 9,
            })?.revAnchorEndSec,
        ).toBe(2);
    });

    it("周期退化为 0 → null", () => {
        expect(
            resolveLoopCycleDescriptor({
                loopEnabled: true,
                contentDurationSec: null,
                sourceStartSec: 3,
                sourceEndSec: 3,
            }),
        ).toBeNull();
    });
});
