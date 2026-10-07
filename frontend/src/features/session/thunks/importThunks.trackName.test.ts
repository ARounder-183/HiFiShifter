/**
 * 导入媒体文件时，**因导入而新建的轨道**以首个媒体文件命名。
 *
 * 【为什么用桩后端 + 真 store 跑一遍 thunk】命名规则藏在 thunk 的建轨分支里，
 * 光测 `trackNameForMedia` 只能证明"去扩展名"这一步对，证不了"哪条轨道用了
 * 哪个文件的名字" —— 尤其 `across-tracks` 下每条新轨道要各自以**它承载的那个
 * 文件**命名（不是这批的第一个）。这里用桩 `webApi` 记录 `add_track` 收到的
 * `name`，跑真实 thunk 与 reducer。
 */

import { configureStore } from "@reduxjs/toolkit";
import { beforeEach, describe, expect, test, vi } from "vitest";

vi.mock("../../../services/webviewApi", () => {
    let seq = 0;
    return {
        webApi: {
            beginUndoGroup: vi.fn(async () => ({ ok: true })),
            endUndoGroup: vi.fn(async () => ({ ok: true })),
            importAudioItem: vi.fn(async () => ({ ok: true, clips: [{ id: "c1" }] })),
            getTimelineState: vi.fn(async () => ({ ok: true, tracks: [], clips: [] })),
            addTrackNested: vi.fn(async (payload: { name?: string }) => {
                seq += 1;
                const id = `new-${seq}`;
                return {
                    ok: true,
                    tracks: [{ id, name: payload?.name ?? "Track", parent_id: null }],
                    selected_track_id: id,
                    clips: [],
                };
            }),
        },
    };
});

import { webApi } from "../../../services/webviewApi";
import sessionReducer from "../sessionSlice";
import { importAudioAtPosition, importMultipleAudioAtPosition } from "./importThunks";

const addTrackNested = webApi.addTrackNested as unknown as ReturnType<typeof vi.fn>;

function createStore() {
    return configureStore({ reducer: { session: sessionReducer } });
}

/** 每次用例开始清掉调用记录（`addTrackNested` 内部序号可继续递增，无碍）。 */
beforeEach(() => {
    addTrackNested.mockClear();
});

/** 最近一次 `add_track` 收到的轨道名。 */
function lastNameArg(): unknown {
    const calls = addTrackNested.mock.calls;
    return (calls[calls.length - 1]?.[0] as { name?: string } | undefined)?.name;
}

describe("单文件导入到新轨道", () => {
    test("轨道名 = 该文件的主名（去扩展名）", async () => {
        const store = createStore();
        await store.dispatch(
            importAudioAtPosition({
                audioPath: "C:\\music\\vocal take 01.wav",
                trackId: null,
                startSec: 0,
            }),
        );
        expect(addTrackNested).toHaveBeenCalledTimes(1);
        expect(lastNameArg()).toBe("vocal take 01");
    });

    test("落到已有轨道时不新建轨道（也就无所谓命名）", async () => {
        const store = createStore();
        await store.dispatch(
            importAudioAtPosition({ audioPath: "C:\\a\\x.wav", trackId: "t9", startSec: 0 }),
        );
        expect(addTrackNested).not.toHaveBeenCalled();
    });
});

describe("多文件导入", () => {
    test("across-time 落在同一条新轨道 → 以这批的第一个文件命名", async () => {
        const store = createStore();
        await store.dispatch(
            importMultipleAudioAtPosition({
                audioPaths: ["C:\\a\\first.wav", "C:\\a\\second.wav"],
                mode: "across-time",
                trackId: null,
                startSec: 0,
            }),
        );
        expect(addTrackNested).toHaveBeenCalledTimes(1);
        expect(lastNameArg()).toBe("first");
    });

    test("across-tracks 需要新建时 → 每条新轨道以**它承载的那个文件**命名", async () => {
        // 初始工程已有一条根轨道（"Main"），因此第一个文件复用它、不建轨；
        // 后两个文件各自新建一条轨道 —— 名字必须分别是它们自己，而不是这批的第一个。
        const store = createStore();
        await store.dispatch(
            importMultipleAudioAtPosition({
                audioPaths: ["C:\\a\\alpha.wav", "C:\\a\\beta.wav", "C:\\a\\gamma.wav"],
                mode: "across-tracks",
                trackId: null,
                startSec: 0,
            }),
        );
        const names = addTrackNested.mock.calls.map((call) => (call[0] as { name?: string }).name);
        expect(names).toEqual(["beta", "gamma"]);
    });
});
