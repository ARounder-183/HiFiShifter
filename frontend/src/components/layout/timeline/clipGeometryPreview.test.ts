import { describe, expect, it } from "vitest";

import { buildClipGeometryPreview } from "./clipGeometryPreview";

describe("buildClipGeometryPreview", () => {
    it("投影快照包含响度映射所需的全部字段", () => {
        const preview = buildClipGeometryPreview([
            {
                id: "a",
                trackId: "t",
                startSec: 1,
                lengthSec: 2,
                gain: 0.5,
                sourceStartSec: 0.25,
                sourceEndSec: 2.25,
                playbackRate: 1.5,
                reversed: true,
                loopEnabled: true,
                durationSec: 4,
                durationFrames: 192000,
                sourceSampleRate: 48000,
                sourcePath: "/media/a.wav",
                midiNoteData: [{ startSec: 0, endSec: 3.5, note: 60 }],
            },
        ]);
        expect(preview.origin).toEqual([
            {
                clipId: "a",
                startSec: 1,
                lengthSec: 2,
                gain: 0.5,
                sourceStartSec: 0.25,
                sourceEndSec: 2.25,
                playbackRate: 1.5,
                reversed: true,
                loopEnabled: true,
                durationSec: 4,
                durationFrames: 192000,
                sourceSampleRate: 48000,
                sourcePath: "/media/a.wav",
                midiNoteData: [{ endSec: 3.5 }],
            },
        ]);
    });

    it("warp 快照与完整源几何都是深拷贝", () => {
        const spans = [{ startSec: 0, endSec: 0.5, note: 60 }];
        const takeSpans = [{ startSec: 0.5, endSec: 1, note: 62 }];
        const input = [
            {
                id: "a",
                trackId: "t",
                startSec: 0,
                lengthSec: 1,
                takes: [
                    {
                        id: "take",
                        sourceStartSec: 0,
                        sourceEndSec: 1,
                        playbackRate: 1,
                        reversed: false,
                        loopEnabled: false,
                        syllableSpansSec: takeSpans,
                    },
                ],
                syllableSpansSec: spans,
                midiNoteData: [{ startSec: 0, endSec: 1, note: 64 }],
            },
        ];
        const preview = buildClipGeometryPreview(input);
        input[0].startSec = 99;
        spans[0]!.startSec = 99;
        takeSpans[0]!.startSec = 99;
        input[0].midiNoteData[0]!.endSec = 99;
        expect(preview.origin[0]?.startSec).toBe(0);
        expect(preview.clips[0]?.syllableSpansSec?.[0]?.startSec).toBe(0);
        expect(preview.clips[0]?.takes?.[0]?.syllableSpansSec?.[0]?.startSec).toBe(0.5);
        expect(preview.origin[0]?.midiNoteData?.[0]?.endSec).toBe(1);
    });

    it("投影不含 waveform / waveformPreview", () => {
        const input = [
            {
                id: "a",
                trackId: "t",
                startSec: 0,
                lengthSec: 1,
                waveform: [1, 2, 3],
                waveformPreview: { l: [1], r: [1] },
            },
        ] as unknown as Parameters<typeof buildClipGeometryPreview>[0];
        const preview = buildClipGeometryPreview(input);
        expect("waveform" in (preview.clips[0] as object)).toBe(false);
        expect("waveformPreview" in (preview.clips[0] as object)).toBe(false);
    });
});
