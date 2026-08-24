import reducer from "./sessionSlice.js";
import {
    selectClipRemote,
    selectTrackRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
} from "./thunks/timelineThunks.js";
import { undoRemote } from "./thunks/projectThunks.js";
import { setTrackStateRemote } from "./thunks/trackThunks.js";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    const actualJson = JSON.stringify(actual);
    const expectedJson = JSON.stringify(expected);
    if (actualJson !== expectedJson) {
        throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
    }
}

function createState(): any {
    const base = reducer(undefined, {
        type: "@@INIT",
    });

    return {
        ...base,
        tracks: [
            {
                id: "track-a",
                name: "Track A",
                parentId: null,
                depth: 0,
                childTrackIds: [],
                muted: false,
                solo: false,
                volume: 1,
                composeEnabled: false,
                pitchAnalysisAlgo: "nsf_hifigan_onnx",
                color: "#4f7cff",
            },
            {
                id: "track-b",
                name: "Track B",
                parentId: null,
                depth: 0,
                childTrackIds: [],
                muted: false,
                solo: false,
                volume: 1,
                composeEnabled: false,
                pitchAnalysisAlgo: "nsf_hifigan_onnx",
                color: "#ff7a00",
            },
        ],
        clips: [
            {
                id: "clip-a",
                trackId: "track-a",
                name: "Clip A",
                startSec: 4,
                lengthSec: 2,
                color: "emerald",
                gain: 1,
                muted: false,
                sourcePath: "a.wav",
                durationSec: 8,
                sourceStartSec: 0,
                sourceEndSec: 8,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInCurve: "sine",
                fadeOutCurve: "sine",
            },
            {
                id: "clip-b",
                trackId: "track-b",
                name: "Clip B",
                startSec: 8,
                lengthSec: 3,
                color: "amber",
                gain: 1,
                muted: false,
                sourcePath: "b.wav",
                durationSec: 12,
                sourceStartSec: 0,
                sourceEndSec: 12,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInCurve: "sine",
                fadeOutCurve: "sine",
            },
        ],
        selectedTrackId: "track-a",
        selectedClipId: "clip-a",
        historyPast: [
            {
                clips: [
                    {
                        id: "clip-a",
                        trackId: "track-a",
                        name: "Clip A",
                        startSec: 1,
                        lengthSec: 2,
                        color: "emerald",
                        gain: 1,
                        muted: false,
                        sourcePath: "a.wav",
                        durationSec: 8,
                        sourceStartSec: 0,
                        sourceEndSec: 8,
                        playbackRate: 1,
                        reversed: false,
                        loopEnabled: false,
                        fadeInSec: 0,
                        fadeOutSec: 0,
                        fadeInCurve: "sine",
                        fadeOutCurve: "sine",
                    },
                    {
                        id: "clip-b",
                        trackId: "track-b",
                        name: "Clip B",
                        startSec: 8,
                        lengthSec: 3,
                        color: "amber",
                        gain: 1,
                        muted: false,
                        sourcePath: "b.wav",
                        durationSec: 12,
                        sourceStartSec: 0,
                        sourceEndSec: 12,
                        playbackRate: 1,
                        reversed: false,
                        loopEnabled: false,
                        fadeInSec: 0,
                        fadeOutSec: 0,
                        fadeInCurve: "sine",
                        fadeOutCurve: "sine",
                    },
                ],
                clipAutomation: {},
                selectedTrackId: "track-a",
                selectedClipId: "clip-a",
                selectedPointId: null,
                playheadSec: 0,
                clipWaveforms: {},
                clipPitchRanges: {},
            },
        ],
    } as any;
}

{
    const next = reducer(createState(), selectTrackRemote.pending("req-track", "track-b"));
    assertEqual(next.selectedTrackId, "track-b", "track selection updates on pending");
}

{
    const next = reducer(
        createState(),
        selectClipRemote.pending("req-clip", {
            clipId: "clip-b",
            preserveTrackFocus: false,
        }),
    );
    assertEqual(next.selectedClipId, "clip-b", "clip selection updates on pending");
    assertEqual(next.selectedTrackId, "track-b", "clip pending selection follows clip track");
}

{
    const next = reducer(
        createState(),
        selectClipRemote.pending("req-clip-preserve", {
            clipId: "clip-b",
            preserveTrackFocus: true,
        }),
    );
    assertEqual(next.selectedTrackId, "track-a", "preserveTrackFocus keeps current track");
}

{
    const next = reducer(
        createState(),
        setTrackStateRemote.pending("req-track-state", {
            trackId: "track-a",
            muted: true,
            color: "#00ffaa",
        }),
    );
    assertEqual(next.tracks[0].muted, true, "track mute updates on pending");
    assertEqual(next.tracks[0].color, "#00ffaa", "track color updates on pending");
}

{
    const next = reducer(
        createState(),
        setClipStateRemote.pending("req-clip-state", {
            clipId: "clip-a",
            name: "Renamed",
            gain: 1.5,
            fadeOutCurve: "scurve",
        }),
    );
    assertEqual(next.clips[0].name, "Renamed", "clip name updates on pending");
    assertEqual(next.clips[0].gain, 1.5, "clip gain updates on pending");
    assertEqual(next.clips[0].fadeOutCurve, "scurve", "clip fade curve updates on pending");
}

{
    const next = reducer(
        createState(),
        setClipsStateBulkRemote.pending("req-clips-bulk", {
            updates: [
                { clipId: "clip-a", muted: true },
                { clipId: "clip-b", gain: 0.5 },
            ],
            checkpoint: false,
        }),
    );
    assertEqual(next.clips[0].muted, true, "bulk mute updates on pending");
    assertEqual(next.clips[1].gain, 0.5, "bulk gain updates on pending");
}

{
    const next = reducer(createState(), undoRemote.pending("req-undo", undefined));
    assertEqual(next.clips[0].startSec, 1, "undo pending applies local snapshot immediately");
}

console.log("sessionSlice optimistic checks passed");
