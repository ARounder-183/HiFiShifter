/**
 * slipWindow.ts 的单测。
 *
 * 【主要内容】
 * 覆盖 `toBoundarySnapClip`（媒体边界吸附视图）的归一化规则：内容时长 D 的解析
 * 优先级（帧数/采样率 → durationSec → 音高参考块覆盖值）、播放速率与长度的
 * 缺省归一化、以及「是否参与吸附」的判定字段 `isContentBearing`。
 *
 * 【为什么值得测】
 * 这几条规则决定 loop 边界吸附的候选族落在哪——分叉只在 loop Clip 跨媒体边界时
 * 体现（听感/波形层面），一旦写错极难归因；纯函数单测是最廉价的护栏。
 */
import { describe, expect, it } from "vitest";
import { toBoundarySnapClip } from "./slipWindow";

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
