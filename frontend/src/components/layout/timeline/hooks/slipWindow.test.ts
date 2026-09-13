/**
 * slipWindow.ts 的单测。
 *
 * 【主要内容】
 * 覆盖 `toBoundarySnapClip`（媒体边界吸附视图）的归一化规则：内容时长 D 的解析
 * 优先级（帧数/采样率 → durationSec → 音高参考块覆盖值）、播放速率与长度的
 * 缺省归一化、以及「是否参与吸附」的判定字段 `isContentBearing`；
 * 以及 `computeSlipWindow` 的**方向约定**与三条分支（loop 环绕 / 非 loop 正放
 * 派生窗口 / 其余保持跨度）。
 *
 * 【为什么值得测】
 * 这几条规则决定 loop 边界吸附的候选族落在哪——分叉只在 loop Clip 跨媒体边界时
 * 体现（听感/波形层面），一旦写错极难归因；纯函数单测是最廉价的护栏。
 * `computeSlipWindow` 的**符号**尤其值得锁：调用方分属两个坐标域（旧实现传窗口域、
 * 内核传屏幕域需取反），符号搞反会让 slip 整体反向——而"反向"在波形上看不出对错，
 * 只有对比素材内容才能发现，正是曾经漏到用户手里的缺陷。
 */
import { describe, expect, it } from "vitest";
import { computeSlipWindow, toBoundarySnapClip } from "./slipWindow";

/** 构造一个最小可用的 clip（缺省值贴近真实数据）。 */
function clip(overrides: Record<string, unknown> = {}) {
    return {
        id: "c1",
        trackId: "t1",
        startSec: 0,
        lengthSec: 4,
        sourceStartSec: 0,
        sourceEndSec: 4,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        sourcePath: "/mock/a.wav",
        durationFrames: 12 * 44100,
        sourceSampleRate: 44100,
        durationSec: 12,
        ...overrides,
    } as never;
}

describe("toBoundarySnapClip", () => {
    it("内容时长优先 durationFrames / sourceSampleRate", () => {
        const view = toBoundarySnapClip(
            clip({ durationFrames: 8 * 44100, sourceSampleRate: 44100, durationSec: 99 }),
        );
        expect(view.contentDurationSec).toBeCloseTo(8, 9);
    });

    it("帧数缺失时回退 durationSec", () => {
        const view = toBoundarySnapClip(
            clip({ durationFrames: null, sourceSampleRate: null, durationSec: 7.5 }),
        );
        expect(view.contentDurationSec).toBeCloseTo(7.5, 9);
    });

    it("两者都缺失时内容时长为 null（候选族退化，由调用方兜底）", () => {
        const view = toBoundarySnapClip(
            clip({ durationFrames: null, sourceSampleRate: null, durationSec: null }),
        );
        expect(view.contentDurationSec).toBeNull();
    });

    it("播放速率非法时归一为 1；长度为负时归一为 0", () => {
        const view = toBoundarySnapClip(clip({ playbackRate: 0, lengthSec: -3 }));
        expect(view.playbackRate).toBe(1);
        expect(view.lengthSec).toBe(0);
    });

    it("有源媒体 → isContentBearing 为 true（参与边界吸附）", () => {
        expect(toBoundarySnapClip(clip()).isContentBearing).toBe(true);
    });

    it("无源媒体且无 MIDI 音符 → isContentBearing 为 false", () => {
        const view = toBoundarySnapClip(
            clip({ sourcePath: "", midiNoteData: [], midiNoteCount: 0 }),
        );
        expect(view.isContentBearing).toBe(false);
    });

    it("无源媒体但有 MIDI 音符 → 仍参与吸附", () => {
        const view = toBoundarySnapClip(
            clip({ sourcePath: "", midiNoteData: [{ startSec: 0, lengthSec: 1, pitch: 60 }] }),
        );
        expect(view.isContentBearing).toBe(true);
    });
});

describe("computeSlipWindow（方向约定 + 三条分支）", () => {
    it("正值 = 源窗口向素材后段平移（起点与终点同时增大）", () => {
        const next = computeSlipWindow(clip(), 0.5);
        expect(next).not.toBeNull();
        expect(next?.sourceStartSec).toBeCloseTo(0.5, 9);
        // 非 loop 正放走"派生窗口"分支：终点 = 起点 + 长度 × 速率。
        expect(next?.sourceEndSec).toBeCloseTo(0.5 + 4, 9);
    });

    it("负值 = 向前段平移（起点与终点同时减小，可越过 0 表示前导静音）", () => {
        const next = computeSlipWindow(clip(), -0.5);
        expect(next?.sourceStartSec).toBeCloseTo(-0.5, 9);
        expect(next?.sourceEndSec).toBeCloseTo(-0.5 + 4, 9);
    });

    it("**屏幕域换算**：向右拖（屏幕正位移）必须传负的窗口平移量", () => {
        // 这两条断言一起锁住约定：同一次"向右拖 100px"（屏幕域 +0.6667s）在
        // 内核侧要取反号传进来，结果必须让 sourceStart 变小。
        const screenDeltaSec = 0.6667; // 向右拖
        const kernelCall = computeSlipWindow(clip(), -screenDeltaSec);
        expect(kernelCall?.sourceStartSec).toBeLessThan(0);

        // 反向自证：若不取反号，方向会整体反过来（这正是曾经的缺陷）。
        const wrong = computeSlipWindow(clip(), screenDeltaSec);
        expect(wrong?.sourceStartSec).toBeGreaterThan(0);
    });

    it("倒放：窗口沿指针反方向平移（dir = -1）", () => {
        const next = computeSlipWindow(clip({ reversed: true }), 0.5);
        expect(next?.sourceStartSec).toBeCloseTo(-0.5, 9);
        expect(next?.sourceEndSec).toBeCloseTo(4 - 0.5, 9);
    });

    it("播放速率参与换算（源位移 = 窗口平移量 × 速率）", () => {
        const next = computeSlipWindow(
            clip({ playbackRate: 2, sourceStartSec: 0, sourceEndSec: 8 }),
            0.5,
        );
        expect(next?.sourceStartSec).toBeCloseTo(1, 9);
    });

    it("loop：窗口两端对内容时长取模环绕", () => {
        // 内容时长 12s；起点 11 + 2 = 13 → 环绕到 1。
        const next = computeSlipWindow(
            clip({ loopEnabled: true, sourceStartSec: 11, sourceEndSec: 15 }),
            2,
        );
        expect(next?.sourceStartSec).toBeCloseTo(1, 9);
        expect(next?.sourceEndSec).toBeCloseTo(5, 9);
    });

    it("loop 且向左越界：负值也环绕到正区间（floor_mod）", () => {
        const next = computeSlipWindow(
            clip({ loopEnabled: true, sourceStartSec: 1, sourceEndSec: 5 }),
            -2,
        );
        expect(next?.sourceStartSec).toBeCloseTo(11, 9);
        expect(next?.sourceEndSec).toBeCloseTo(3, 9);
    });

    it("loop 时长未知：保持平移而不卡死", () => {
        const next = computeSlipWindow(
            clip({
                loopEnabled: true,
                sourceStartSec: 1,
                sourceEndSec: 5,
                durationFrames: null,
                sourceSampleRate: null,
                durationSec: null,
            }),
            2,
        );
        expect(next?.sourceStartSec).toBeCloseTo(3, 9);
        expect(next?.sourceEndSec).toBeCloseTo(7, 9);
    });

    it("非 loop 倒放：只整体平移、保持跨度（不被派生窗口覆盖）", () => {
        const next = computeSlipWindow(
            clip({ reversed: true, sourceStartSec: 0, sourceEndSec: 10, lengthSec: 4 }),
            1,
        );
        // 跨度 10 保持不变（倒放的 sourceEnd 是反向锚点，跨度可合法大于 len × rate）。
        expect(next?.sourceEndSec).toBeCloseTo(9, 9);
    });

    it("非法位移返回 null（调用方跳过该帧）", () => {
        expect(computeSlipWindow(clip(), Number.NaN)).toBeNull();
    });
});
