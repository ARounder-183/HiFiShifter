/**
 * 导入新 Clip 不得改写参数编辑器的“当前轨道组”与已有曲线。
 *
 * 【用户症状】先在同一条根轨道上调好音高，再导入一条新 Clip；新 Clip 被
 * 自动选中后，编辑器的 root 跟着切走，屏幕上的曲线变成另一条轨道的取数结果，
 * 看起来就是“新 Clip 覆盖了用户参数”。
 *
 * 【不变式】导入只新增媒体；新 Clip 进多选，但当前 root 的 selectedTrackId、
 * paramsEpoch 与已有 clipPitchCurves 都保持不变。
 */
import { test } from "vitest";

import reducer from "./sessionSlice.ts";
import { importAudioAtPosition } from "./thunks/importThunks.ts";
import type { TimelineResult } from "../../types/api";

test("导入新 Clip 不夺取参数编辑器锚点、不清空既有曲线", () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const base = {
        ...reducer(undefined, { type: "@@INIT" }),
        paramsEpoch: 11,
        editParam: "pitch",
        selectedTrackId: "root-a",
        selectedClipId: "clip-a",
        tracks: [
            {
                id: "root-a",
                name: "Root A",
                muted: false,
                solo: false,
                volume: 1,
                composeEnabled: true,
                pitchAnalysisAlgo: "nsf_hifigan_onnx",
            },
            {
                id: "root-b",
                name: "Root B",
                muted: false,
                solo: false,
                volume: 1,
                composeEnabled: true,
                pitchAnalysisAlgo: "nsf_hifigan_onnx",
            },
        ],
        clips: [
            {
                id: "clip-a",
                trackId: "root-a",
                name: "Edited Clip",
                startSec: 0,
                lengthSec: 2,
                color: "emerald",
                gain: 1,
                muted: false,
                sourceStartSec: 0,
                sourceEndSec: 2,
                playbackRate: 1,
                reversed: false,
                channelMode: 0,
                loopEnabled: false,
                fadeInSec: 0,
                fadeOutSec: 0,
                fadeInShape: 0,
                fadeOutShape: 0,
                fadeInDir: 0,
                fadeOutDir: 0,
            },
        ],
        clipPitchCurves: {
            "clip-a": {
                curveStartSec: 0,
                midiCurve: [60, 62, 64],
                framePeriodMs: 5,
            },
        },
    } as unknown as ReturnType<typeof reducer>;

    const incomingTimeline = {
        ok: true,
        bpm: 120,
        playhead_sec: 0,
        project_sec: 30,
        // 后端导入把选中切到新 Clip 所在轨道组：前端不得无条件采纳。
        selected_track_id: "root-b",
        selected_clip_id: "clip-b",
        tracks: [
            {
                id: "root-a",
                name: "Root A",
                muted: false,
                solo: false,
                volume: 1,
                compose_enabled: true,
                pitch_analysis_algo: "nsf_hifigan_onnx",
            },
            {
                id: "root-b",
                name: "Root B",
                muted: false,
                solo: false,
                volume: 1,
                compose_enabled: true,
                pitch_analysis_algo: "nsf_hifigan_onnx",
            },
        ],
        clips: [
            {
                id: "clip-a",
                track_id: "root-a",
                name: "Edited Clip",
                start_sec: 0,
                length_sec: 2,
                color: "emerald",
                gain: 1,
                muted: false,
                source_start_sec: 0,
                source_end_sec: 2,
                playback_rate: 1,
                reversed: false,
                fade_in_sec: 0,
                fade_out_sec: 0,
                fade_in_shape: 0,
                fade_out_shape: 0,
                fade_in_dir: 0,
                fade_out_dir: 0,
            },
            {
                id: "clip-b",
                track_id: "root-b",
                name: "Imported Clip",
                start_sec: 4,
                length_sec: 2,
                color: "emerald",
                gain: 1,
                muted: false,
                source_start_sec: 0,
                source_end_sec: 2,
                playback_rate: 1,
                reversed: false,
                fade_in_sec: 0,
                fade_out_sec: 0,
                fade_in_shape: 0,
                fade_out_shape: 0,
                fade_in_dir: 0,
                fade_out_dir: 0,
            },
        ],
    } as unknown as TimelineResult;

    const next = reducer(
        base,
        importAudioAtPosition.fulfilled(
            {
                ok: true,
                imported: incomingTimeline,
                newClipIds: ["clip-b"],
                playheadSec: 4,
            },
            "req-import",
            { audioPath: "new.wav", trackId: null, startSec: 4 },
        ),
    );

    assertEqual(next.selectedTrackId, "root-a", "导入不得切换参数编辑 root");
    assertEqual(next.selectedClipId, "clip-a", "导入不得夺取参数编辑器锚点");
    assertEqual(next.multiSelectedClipIds, ["clip-b"], "新 Clip 仍进入多选");
    assertEqual(next.paramsEpoch, 11, "导入保持参数刷新 epoch");
    assertEqual(next.editParam, "pitch", "导入不得切换参数");
    assertEqual(
        next.clipPitchCurves,
        {
            "clip-a": {
                curveStartSec: 0,
                midiCurve: [60, 62, 64],
                framePeriodMs: 5,
            },
        },
        "既有 Clip 的音高可视化曲线保持不变",
    );
});
