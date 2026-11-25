// 待定导入（item 已建成、ARA 还没认领）走完回执 → 事件两段：回执不算失败，也不许提前规格化。
// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("../../../services/webviewApi", () => ({
    webApi: {
        beginUndoGroup: vi.fn(async () => ({ ok: true })),
        endUndoGroup: vi.fn(async () => ({ ok: true })),
        importAudioItem: vi.fn(),
        getTimelineState: vi.fn(),
    },
}));
const { computeClipNormalizationGain } = vi.hoisted(() => ({
    computeClipNormalizationGain: vi.fn(),
}));
vi.mock("../clipNormalization", () => ({ computeClipNormalizationGain }));

import { webApi } from "../../../services/webviewApi";
import { araOnlyCaps } from "../../../services/pluginBootstrapFixtures";
import reducer from "../sessionSlice";
import { completePendingImport, importAudioAtPosition } from "./importThunks";

const ITEM = "{IMPORT-1}";
const CLIP = "hfs-ui-1-ara-item-{IMPORT-1}";

function timeline(clips: Array<Record<string, unknown>>) {
    return {
        ok: true,
        tracks: [{ id: "track", name: "host", order: 0 }],
        clips,
        selected_track_id: "track",
        selected_clip_id: null,
        bpm: 120,
        playhead_sec: 0,
        project_sec: 10,
    };
}

function store() {
    return configureStore({ reducer: { session: reducer } });
}

beforeEach(() => {
    vi.clearAllMocks();
    computeClipNormalizationGain.mockReset();
    window.__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "imports",
        hostCapabilities: araOnlyCaps({ mediaImport: true }),
    };
    // 回执里带的三个字段就是这条导入的全部证据。

    vi.mocked(webApi.importAudioItem).mockResolvedValue({
        ...timeline([
            {
                id: CLIP,
                track_id: "track",
                name: "pending",
                start_sec: 0,
                length_sec: 2,
            },
        ]),
        imported_item: ITEM,
        imported_clip_id: CLIP,
        import_pending: true,
    } as never);
    vi.mocked(webApi.getTimelineState).mockResolvedValue(
        timeline([
            {
                id: CLIP,
                track_id: "track",
                name: "pending",
                start_sec: 0,
                length_sec: 2,
                source_path: "C:/audio/one.wav",
            },
        ]) as never,
    );
});

afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

/**
 * 【回归】待定回执必须**成功**返回，并把它记进账本。
 *
 * 旧实现把"这一轮图里还没有它"判成 `import_audio_item_failed`，于是 item 明明建好了，
 * 用户看到的却是红色错误条 —— 而那正是本次要修的第一条症状。
 */
test("a pending reply fulfils, records the item, and does not normalize yet", async () => {
    const state = store();
    const result = (await state
        .dispatch(
            importAudioAtPosition({
                audioPath: "C:/audio/one.wav",
                trackId: "track",
                startSec: 0,
                normalizeAfterImport: true,
            }),
        )
        .unwrap()) as { ok?: boolean; pending?: boolean };
    expect(result.ok).toBe(true);
    expect(result.pending).toBe(true);
    const session = state.getState().session;
    expect(session.pendingImports).toHaveLength(1);
    expect(session.pendingImports[0].item).toBe(ITEM);
    expect(session.pendingImports[0].normalizeAfterImport).toBe(true);
    expect(session.error).toBeFalsy();
    // 这时还拿不到 source_path：算出来的永远是"跳过"，做了等于没做。
    expect(computeClipNormalizationGain).not.toHaveBeenCalled();
});

/**
 * 【回归】`import_ready` 之后：账本清空、时间线重取一次、延后的规格化补做。
 */
test("import_ready clears the pending record, refetches once, and normalizes then", async () => {
    const state = store();
    await state
        .dispatch(
            importAudioAtPosition({
                audioPath: "C:/audio/one.wav",
                trackId: "track",
                startSec: 0,
                normalizeAfterImport: true,
            }),
        )
        .unwrap();
    // 导入本身也会重取时间线（自动交叉淡化路径），把计数清零后再看转正这一次。
    vi.mocked(webApi.getTimelineState).mockClear();
    computeClipNormalizationGain.mockReturnValue(null);
    await state.dispatch(completePendingImport({ item: ITEM, clipId: CLIP })).unwrap();
    expect(webApi.getTimelineState).toHaveBeenCalledTimes(1);
    expect(state.getState().session.pendingImports).toHaveLength(0);
    // 转正那一刻才有 source_path，规格化必须在这时才算得出来。
    expect(computeClipNormalizationGain).toHaveBeenCalledTimes(1);
});
