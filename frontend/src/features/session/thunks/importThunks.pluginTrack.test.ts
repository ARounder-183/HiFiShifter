// 插件模式导入的`trackId`三态：未指定=字段缺失（宿主决定）、null=显式新建、string=指定轨。
// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
vi.mock("../../../services/webviewApi", () => ({
    webApi: {
        beginUndoGroup: vi.fn(async () => ({ ok: true })),
        endUndoGroup: vi.fn(async () => ({ ok: true })),
        importAudioItem: vi.fn(),
        getTimelineState: vi.fn(async () => ({ ok: true, tracks: [], clips: [] })),
        addTrackNested: vi.fn(),
    },
}));
import { webApi } from "../../../services/webviewApi";
import { araOnlyCaps } from "../../../services/pluginBootstrapFixtures";
import reducer from "../sessionSlice";
import { importAudioAtPosition } from "./importThunks";

beforeEach(() => {
    vi.clearAllMocks();
    window.__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "imports",
        hostCapabilities: araOnlyCaps({ mediaImport: true }),
    };
    vi.mocked(webApi.importAudioItem).mockResolvedValue({
        ok: true,
        tracks: [{ id: "t1", name: "Track", order: 0 }],
        clips: [
            {
                id: "clip-1",
                track_id: "t1",
                name: "a.wav",
                start_sec: 0,
                length_sec: 2,
            },
        ],
        selected_track_id: "t1",
        selected_clip_id: "clip-1",
        bpm: 120,
        playhead_sec: 0,
        project_sec: 10,
        imported_clip_id: "clip-1",
    } as never);
});
afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

function store() {
    return configureStore({ reducer: { session: reducer } });
}

/**
 * **回归：插件模式下"没有指定轨道"必须是**载荷里没有这个键**，不是 `null`。**
 *
 * 【为什么这条是那个 bug 的回归】`null` 在后端（插件与独立 App 都是）指"显式新建
 * 一条轨"。前端此前把"没选中轨"也写成 `null`，于是"文件 → 导入音频"每次都新建
 * 一条宿主轨：那条轨没有被本实例采纳，它的 item 永远进不了清单（导入的 clip 停在
 * "等待宿主音频"），时间线上反而多出一行没有 region 的 sequence —— 用户看到的正是
 * "导入音频后凭空多出一条轨道，重载之后才恢复"。
 */
test("plugin mode without a chosen track sends no trackId at all", async () => {
    await store()
        .dispatch(importAudioAtPosition({ audioPath: "E:/a.wav", trackId: undefined, startSec: 0 }))
        .unwrap();
    expect(webApi.importAudioItem).toHaveBeenCalledWith("E:/a.wav", undefined, 0, undefined);
    // "没指定"与"新建"在载荷上必须是两种形状。
    expect(vi.mocked(webApi.importAudioItem).mock.calls[0][1]).toBeUndefined();
});

/** `null` 只剩一个意思：显式新建（宿主建轨 + 插件采纳）。 */
test("plugin mode keeps null as the explicit new-track request", async () => {
    await store()
        .dispatch(importAudioAtPosition({ audioPath: "E:/a.wav", trackId: null, startSec: 1 }))
        .unwrap();
    expect(vi.mocked(webApi.importAudioItem).mock.calls[0][1]).toBeNull();
});

/** 指定轨道时原样透传具体 id（这条一直是对的，钉住它防止三态被再次改写）。 */
test("plugin mode passes a concrete track id through unchanged", async () => {
    await store()
        .dispatch(importAudioAtPosition({ audioPath: "E:/a.wav", trackId: "t1", startSec: 2 }))
        .unwrap();
    expect(vi.mocked(webApi.importAudioItem).mock.calls[0][1]).toBe("t1");
});
