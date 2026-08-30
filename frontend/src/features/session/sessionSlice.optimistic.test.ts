import { test } from "vitest";

import reducer from "./sessionSlice.js";
import {
    selectClipRemote,
    selectTrackRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipTakeReversedRemote,
} from "./thunks/timelineThunks.js";
import { undoRemote } from "./thunks/projectThunks.js";
import { setTrackStateRemote } from "./thunks/trackThunks.js";

test("features/session/sessionSlice.optimistic.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function createState(): ReturnType<typeof reducer> {
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
        } as unknown as ReturnType<typeof reducer>;
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
                fadeOutShape: 6,
                fadeOutDir: 0.35,
            }),
        );
        assertEqual(next.clips[0].name, "Renamed", "clip name updates on pending");
        assertEqual(next.clips[0].gain, 1.5, "clip gain updates on pending");
        assertEqual(next.clips[0].fadeOutShape, 6, "clip fade shape updates on pending");
        assertEqual(next.clips[0].fadeOutDir, 0.35, "clip fade dir updates on pending");
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
        // 方向翻转的乐观换算：非 Loop 正放 Clip 的存储 se 是陈旧值（8）时，
        // 翻转必须以消费窗口 [ss, ss+len·r) = [0, 2) 推导新锚点 se —— 直接
        // 翻布尔会让倒放锚到陈旧 se，波形/音频跳到文件末段。
        const next = reducer(
            createState(),
            setClipsStateBulkRemote.pending("req-reverse", {
                updates: [{ clipId: "clip-a", reversed: true }],
                checkpoint: true,
            }),
        );
        assertEqual(next.clips[0].reversed, true, "bulk reverse pending flips direction");
        assertEqual(
            next.clips[0].sourceEndSec,
            2,
            "bulk reverse pending converts stale source end",
        );
        assertEqual(next.clips[0].sourceStartSec, 0, "bulk reverse pending keeps window start");

        // 翻回正放：以倒放消费窗口 [se−span, se) = [0, 2) 的起点换算 ss。
        const back = reducer(
            next,
            setClipsStateBulkRemote.pending("req-unreverse", {
                updates: [{ clipId: "clip-a", reversed: false }],
                checkpoint: true,
            }),
        );
        assertEqual(back.clips[0].reversed, false, "bulk unreverse pending flips direction");
        assertEqual(back.clips[0].sourceStartSec, 0, "unreverse converts to reverse window start");
        assertEqual(back.clips[0].sourceEndSec, 2, "unreverse keeps window end");

        // 显式携带源窗口的请求（粘贴/导入模板等）以调用方窗口为准，不做换算。
        const explicit = reducer(
            createState(),
            setClipsStateBulkRemote.pending("req-reverse-explicit", {
                updates: [{ clipId: "clip-a", reversed: true, sourceStartSec: 1, sourceEndSec: 3 }],
                checkpoint: true,
            }),
        );
        assertEqual(explicit.clips[0].sourceStartSec, 1, "explicit window is respected");
        assertEqual(explicit.clips[0].sourceEndSec, 3, "explicit window skips conversion");
    }

    {
        // Loop Clip 的方向翻转同样镜像后端换算：以原方向消费区间为准换算
        // 回绕锚点。10s 媒体、Loop 开（应用默认）、修剪为源 [2,4)（存储
        // se=10 为导入期陈旧值）→ 翻为倒放后 se := mod(2 + 2×1, 10) = 4，
        // 自 4 降奏覆盖 4~2s（而不是从正放锚 2 降奏到 [0,2)）。
        const loopState = createState();
        loopState.clips[0].loopEnabled = true;
        loopState.clips[0].lengthSec = 2;
        loopState.clips[0].sourceStartSec = 2;
        loopState.clips[0].sourceEndSec = 10;
        loopState.clips[0].durationSec = 10;
        const loopFlip = reducer(
            loopState,
            setClipsStateBulkRemote.pending("req-reverse-loop", {
                updates: [{ clipId: "clip-a", reversed: true }],
                checkpoint: true,
            }),
        );
        assertEqual(loopFlip.clips[0].reversed, true, "loop reverse pending flips direction");
        assertEqual(loopFlip.clips[0].sourceEndSec, 4, "loop reverse anchors at consumption end");
        assertEqual(loopFlip.clips[0].sourceStartSec, 2, "loop reverse keeps window start");
        // 翻回正放：ss := mod(4 − 2, 10) = 2，还原原消费区间起点。
        const loopBack = reducer(
            loopFlip,
            setClipsStateBulkRemote.pending("req-unreverse-loop", {
                updates: [{ clipId: "clip-a", reversed: false }],
                checkpoint: true,
            }),
        );
        assertEqual(loopBack.clips[0].reversed, false, "loop unreverse pending flips direction");
        assertEqual(
            loopBack.clips[0].sourceStartSec,
            2,
            "loop unreverse restores consumption start",
        );
    }

    {
        // 单 Take 倒放的乐观更新：inactive take 只动自身条目（按其自身速率
        // 换算窗口），flat 投影不动；active take 翻转需物化到 flat。
        const state = createState();
        state.clips[0].activeTakeId = "clip-a_take_1";
        state.clips[0].takes = [
            {
                id: "clip-a_take_1",
                name: "Take 1",
                gain: 1,
                sourceStartSec: 0,
                sourceEndSec: 2,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
                midiFillGaps: false,
            },
            {
                id: "take-2",
                name: "Take 2",
                gain: 1,
                sourceStartSec: 5,
                sourceEndSec: 20,
                playbackRate: 2,
                reversed: false,
                loopEnabled: false,
                midiFillGaps: false,
            },
        ];
        const flipInactive = reducer(
            state,
            setClipTakeReversedRemote.pending("req-take-reverse", {
                clipId: "clip-a",
                takeId: "take-2",
                reversed: true,
            }),
        );
        const take1Inactive = flipInactive.clips[0].takes?.find(
            (entry) => entry.id === "clip-a_take_1",
        );
        const take2Inactive = flipInactive.clips[0].takes?.find((entry) => entry.id === "take-2");
        if (!take1Inactive || !take2Inactive) {
            throw new Error("takes missing after inactive take flip");
        }
        assertEqual(
            flipInactive.clips[0].reversed,
            false,
            "inactive take flip leaves flat untouched",
        );
        assertEqual(take2Inactive.reversed, true, "inactive take flipped");
        // take2 自身速率 2、clip len 2 → se := 5 + 2×2 = 9（覆盖陈旧 se=20）。
        assertEqual(take2Inactive.sourceEndSec, 9, "inactive take window converted with own rate");
        assertEqual(take1Inactive.reversed, false, "other take untouched");

        const flipActive = reducer(
            flipInactive,
            setClipTakeReversedRemote.pending("req-take-reverse-active", {
                clipId: "clip-a",
                takeId: "clip-a_take_1",
                reversed: true,
            }),
        );
        const take1Active = flipActive.clips[0].takes?.find(
            (entry) => entry.id === "clip-a_take_1",
        );
        if (!take1Active) {
            throw new Error("active take missing after active take flip");
        }
        assertEqual(flipActive.clips[0].reversed, true, "active take flip materializes flat");
        assertEqual(flipActive.clips[0].sourceEndSec, 2, "active take flip converts flat window");
        assertEqual(take1Active.reversed, true, "active take entry flipped");
    }

    {
        // 撤销以后端为唯一权威：pending 阶段不做本地快照回放，状态保持不变
        // （旧实现会乐观渲染与后端检查点错位的前端快照，造成轨道视图闪屏）。
        const next = reducer(createState(), undoRemote.pending("req-undo", undefined));
        assertEqual(next.clips[0].startSec, 4, "undo pending leaves state untouched");
        assertEqual(
            next._latestHistoryOpRequestId,
            "req-undo",
            "undo pending records request id for stale-response discard",
        );
    }
});
