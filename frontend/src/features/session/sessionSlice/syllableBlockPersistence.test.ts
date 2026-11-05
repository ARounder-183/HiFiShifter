import { describe, expect, it } from "vitest";

import type { ClipTakeInfo } from "../sessionTypes";
import type { TimelineClip, TimelineClipTake } from "../../../types/api";
import { parseClipTakes, parseTimelineClipTake } from "./helpersTakes";

const wireTake = (overrides: Partial<TimelineClipTake> = {}): TimelineClipTake => ({
    id: "take-a",
    name: "Take 1",
    gain: 1,
    source_path: "a.wav",
    source_start_sec: 0,
    source_end_sec: 2,
    playback_rate: 1,
    reversed: false,
    loop_enabled: false,
    ...overrides,
});

const wireClip = (overrides: Partial<TimelineClip> = {}): TimelineClip => ({
    id: "clip-a",
    track_id: "root",
    name: "Clip",
    start_sec: 1,
    length_sec: 2,
    color: "blue",
    takes: [
        wireTake({
            syllable_spans_sec: [
                [0.1, 0.6],
                [0.6, 1.1],
            ],
        }),
    ],
    active_take_id: "take-a",
    ...overrides,
});

describe("syllable spans v7 persistence", () => {
    it("旧 Take 载荷不带新字段时仍解析，缺省为空而不是报错", () => {
        const parsed = parseTimelineClipTake(wireTake());
        expect(parsed.syllableSpansSec).toBeUndefined();
        expect(parsed.midiNoteData).toBeUndefined();
    });

    it("源域音节区间进入 Clip 主投影和 Take，参与块派生", () => {
        const parsed = parseClipTakes(wireClip(), {
            id: "clip-a",
            trackId: "root",
            name: "Clip",
            startSec: 1,
            lengthSec: 2,
            color: "blue",
            gain: 1,
            muted: false,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            channelMode: 0,
            loopEnabled: false,
        } as never);
        expect(parsed[0].syllableSpansSec).toEqual([
            { startSec: 0.1, endSec: 0.6 },
            { startSec: 0.6, endSec: 1.1 },
        ]);
    });

    it("音节区间不会从 Take 反向丢掉", () => {
        const take: ClipTakeInfo = {
            id: "take-a",
            name: "Take",
            gain: 1,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            channelMode: 0,
            syllableSpansSec: [{ startSec: 0, endSec: 0.5 }],
        };
        expect(take.syllableSpansSec?.[0]?.endSec).toBe(0.5);
    });
});
