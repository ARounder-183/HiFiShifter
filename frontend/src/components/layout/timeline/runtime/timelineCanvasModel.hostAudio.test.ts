/**
 * 渲染模型侧「等待宿主音频」标记的契约。
 *
 * 【为什么在模型层钉住】占位识别沿用 `source_path` 缺失这一既有判据（与后端
 * `retain_display_waveforms` 同源），所以这一层是"哪条 clip 被标成占位"的唯一决定点。
 * 判错会让一条**功能正常**的片段被画上斜纹、并在悬停时被告知"没有音频"。
 */
import { expect, test } from "vitest";

import { buildSparseClipRenderModel } from "./timelineCanvasModel.js";
import { createTimelineAxis } from "../../renderKernel/timelineAxis.js";

type ClipInput = Parameters<
    typeof buildSparseClipRenderModel
>[0]["visibleTrackClipsById"][string][number];

function clip(extra: Partial<ClipInput>): ClipInput {
    return {
        id: "clip",
        trackId: "track",
        name: "clip",
        startSec: 0,
        lengthSec: 1,
        gain: 1,
        playbackRate: 1,
        muted: false,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 0,
        fadeInDir: 0,
        fadeOutShape: 0,
        fadeOutDir: 0,
        ...extra,
    };
}

function flags(clips: ClipInput[]): Array<boolean | undefined> {
    return buildSparseClipRenderModel({
        visibleTracks: [{ id: "track", color: "#ff7a00" }],
        startTrackIndex: 0,
        visibleTrackClipsById: { track: clips },
        axis: createTimelineAxis({ pxPerSec: 100, viewportWidthPx: 1000 }),
        rowHeight: 48,
        selectedClipId: null,
        multiSelectedClipIds: [],
        renamingClipId: null,
    }).drawClips.map((model) => model.awaitingHostAudio);
}

test("a clip with a source file is not a placeholder", () => {
    expect(flags([clip({ id: "a", sourcePath: "ara://source" })])).toEqual([false]);
});

test("a clip with neither audio nor notes is marked as waiting for host audio", () => {
    expect(flags([clip({ id: "a" })])).toEqual([true]);
});

/* MIDI / 音高参考片段同样没有 `sourcePath`，但有音符内容可画 —— 不能标成占位。 */
test("a MIDI clip is not a placeholder", () => {
    expect(flags([clip({ id: "a", midiNoteCount: 3 })])).toEqual([false]);
});
