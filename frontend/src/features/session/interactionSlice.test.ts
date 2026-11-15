import { describe, expect, it } from "vitest";

import {
    beginClipGeometryPreview,
    clearPianoRollLoading,
    endClipGeometryPreview,
    selectAnyPianoRollLoading,
    selectPianoRollSelection,
    setPianoRollLoading,
    setPianoRollSelection,
} from "./interactionSlice";
import interactionReducer from "./interactionSlice";

const reducer = interactionReducer;

function preview(token: string) {
    return {
        token,
        origin: [
            {
                clipId: "a",
                startSec: 0,
                lengthSec: 1,
                gain: 1,
                sourceStartSec: 0,
                sourceEndSec: 1,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
                durationSec: null,
                durationFrames: null,
                sourceSampleRate: null,
                sourcePath: null,
                midiNoteData: null,
            },
        ],
        clips: [{ id: "a", trackId: "t", startSec: 0, lengthSec: 1 }],
    };
}

describe("interactionSlice", () => {
    it("begin/end 用 token 丢弃迟到的 end", () => {
        let state = reducer(undefined, beginClipGeometryPreview(preview("second")));
        state = reducer(state, endClipGeometryPreview("first"));
        expect(state.clipGeometryPreview?.token).toBe("second");
        state = reducer(state, endClipGeometryPreview("second"));
        expect(state.clipGeometryPreview).toBeNull();
    });

    it("无 token 的 end 清空当前手势（兼容旧调用）", () => {
        let state = reducer(undefined, beginClipGeometryPreview(preview("only")));
        state = reducer(state, endClipGeometryPreview(undefined));
        expect(state.clipGeometryPreview).toBeNull();
    });

    it("加载状态按发布者记账，任何一个在加载即对外为真", () => {
        let state = reducer(undefined, setPianoRollLoading({ publisherId: "a", loading: true }));
        state = reducer(state, setPianoRollLoading({ publisherId: "b", loading: true }));
        expect(selectAnyPianoRollLoading({ interaction: state })).toBe(true);
        state = reducer(state, clearPianoRollLoading("a"));
        expect(selectAnyPianoRollLoading({ interaction: state })).toBe(true);
        state = reducer(state, clearPianoRollLoading("b"));
        expect(selectAnyPianoRollLoading({ interaction: state })).toBe(false);
    });

    it("选区可写入并按引用读取", () => {
        const info = { startFrame: 1, frameCount: 2, framePeriodMs: 5 };
        const state = reducer(undefined, setPianoRollSelection(info));
        expect(selectPianoRollSelection({ interaction: state })).toBe(info);
        const cleared = reducer(state, setPianoRollSelection(null));
        expect(selectPianoRollSelection({ interaction: cleared })).toBeNull();
    });
});
