import { test } from "vitest";

import reducer from "./sessionSlice.js";
import { splitClipRemote, splitClipsAtRemote } from "./thunks/timelineThunks.js";

/**
 * DAW 惯例回归：分割后选中右段、取消左段。
 *
 * 左段继承被分割 clip 的原 id，右段为后端新建 clip 并经
 * payload.created_clip_ids 按输入顺序返回。
 */
test("features/session/sessionSlice.splitSelection.test.ts split selects the right half", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function makeClip(id: string, startSec: number) {
        return {
            id,
            track_id: "track-a",
            name: `Clip ${id}`,
            start_sec: startSec,
            length_sec: 2,
            color: "emerald",
            gain: 1,
            muted: false,
            source_path: "a.wav",
            duration_sec: 8,
            source_start_sec: 0,
            source_end_sec: 8,
            playback_rate: 1,
            reversed: false,
            fade_in_sec: 0,
            fade_out_sec: 0,
        };
    }

    function createState() {
        const base = reducer(undefined, { type: "@@INIT" }) as any;
        return {
            ...base,
            selectedTrackId: "track-a",
            selectedClipId: "clip-a",
            multiSelectedClipIds: ["clip-a"],
        };
    }

    const splitPayload = (rightIds: string[], extraClips: unknown[] = []) =>
        ({
            ok: true,
            tracks: [],
            clips: [
                makeClip("clip-a", 4),
                ...extraClips,
                ...rightIds.map((id, index) => makeClip(id, 5 + index)),
            ],
            selected_track_id: "track-a",
            selected_clip_id: rightIds[0],
            playhead_sec: 5,
            project_sec: 32,
            bpm: 120,
            disabled_group_ids: [],
            created_clip_ids: rightIds,
        }) as any;

    // 单 clip 分割：选中右段，取消左段。
    {
        const next = reducer(
            createState(),
            splitClipRemote.fulfilled(splitPayload(["clip-a-right"]), "req", {
                clipId: "clip-a",
                splitSec: 5,
            }),
        );
        assertEqual(next.selectedClipId, "clip-a-right", "single split selects the right half");
        assertEqual(
            next.multiSelectedClipIds,
            ["clip-a-right"],
            "single split replaces multi selection with the right half",
        );
    }

    // 多 clip 分割：所有右段入选，左段（原 id）取消；单选映射到被分割 clip
    // 自己的右段。
    {
        const state = createState();
        state.multiSelectedClipIds = ["clip-a", "clip-b"];
        const next = reducer(
            state,
            splitClipsAtRemote.fulfilled(
                splitPayload(["clip-a-right", "clip-b-right"], [makeClip("clip-b", 8)]),
                "req",
                { clipIds: ["clip-a", "clip-b"], splitSec: 5 },
            ),
        );
        assertEqual(next.selectedClipId, "clip-a-right", "multi split maps single selection");
        assertEqual(
            next.multiSelectedClipIds,
            ["clip-a-right", "clip-b-right"],
            "multi split selects all right halves and drops the left halves",
        );
    }

    // 多选中未被分割的成员保持选中。
    {
        const state = createState();
        state.multiSelectedClipIds = ["clip-a", "clip-z"];
        const next = reducer(
            state,
            splitClipsAtRemote.fulfilled(
                splitPayload(["clip-a-right"]),
                "req",
                { clipIds: ["clip-a"], splitSec: 5 },
            ),
        );
        assertEqual(
            next.multiSelectedClipIds,
            ["clip-z", "clip-a-right"],
            "unsplit member stays selected",
        );
    }

    // 分割点无效（后端未创建右段）：后端也不改选中，状态保持不变。
    {
        const payload = {
            ...splitPayload(["clip-a-right"]),
            created_clip_ids: null,
            selected_clip_id: "clip-a",
            clips: [makeClip("clip-a", 4)],
        };
        const next = reducer(
            createState(),
            splitClipsAtRemote.fulfilled(payload, "req", { clipIds: ["clip-a"], splitSec: 5 }),
        );
        assertEqual(next.selectedClipId, "clip-a", "no right half keeps current selection");
    }
});
