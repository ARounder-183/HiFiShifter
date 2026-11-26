// 导入后处理只有一条路：立即成功与待定转正都必须走同一套（自动交叉淡化 → 规格化）。
// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("../../../services/webviewApi", () => ({
    webApi: {
        beginUndoGroup: vi.fn(async () => ({ ok: true })),
        endUndoGroup: vi.fn(async () => ({ ok: true })),
        importAudioItem: vi.fn(),
        getTimelineState: vi.fn(),
        setClipsStateBulk: vi.fn(async () => ({ ok: true })),
        setClipState: vi.fn(async () => ({ ok: true })),
    },
}));
const { computeClipNormalizationGain } = vi.hoisted(() => ({
    computeClipNormalizationGain: vi.fn(),
}));
vi.mock("../clipNormalization", () => ({ computeClipNormalizationGain }));

import { webApi } from "../../../services/webviewApi";
import { araOnlyCaps } from "../../../services/pluginBootstrapFixtures";
import reducer, { applyTimelinePayload, toggleAutoCrossfade } from "../sessionSlice";
import { completePendingImport, importAudioAtPosition } from "./importThunks";

const ITEM = "{IMPORT-1}";
const EXISTING = "existing-clip";
const CLIP = "hfs-ui-1-ara-item-{IMPORT-1}";

/** 两条同轨 clip 重叠 2 s：B 的 fadeIn 与 A 的 fadeOut 都该是 2。 */
function overlappingTimeline() {
    return {
        ok: true,
        tracks: [{ id: "track", name: "host", order: 0 }],
        clips: [
            {
                id: EXISTING,
                track_id: "track",
                name: "already there",
                start_sec: 0,
                length_sec: 4,
                auto_fade_in_sec: 0,
                auto_fade_out_sec: 0,
            },
            {
                id: CLIP,
                track_id: "track",
                name: "imported",
                start_sec: 2,
                length_sec: 4,
                source_path: "C:/audio/one.wav",
                auto_fade_in_sec: 0,
                auto_fade_out_sec: 0,
            },
        ],
        selected_track_id: "track",
        selected_clip_id: null,
        bpm: 120,
        playhead_sec: 0,
        project_sec: 10,
    };
}

/** 回执：同一个工程，但只有"导入前"那一条 —— 待定导入的真实形状。 */
function receiptTimeline() {
    const timeline = overlappingTimeline();
    return {
        ...timeline,
        clips: timeline.clips.slice(0, 1),
        imported_item: ITEM,
        imported_clip_id: CLIP,
        import_pending: true,
    };
}

function store() {
    return configureStore({ reducer: { session: reducer } });
}

/**
 * 把"导入前就在工程里的那条 clip"先放进 store。
 *
 * 【为什么必须放】thunk 用"导入前后的 id 差集"算 `newClipIds`。不预置的话，回执里
 * 那条**既有** clip 也会被当成"这次导入新增的"，于是规格化会对着两条 clip 各跑一遍 ——
 * 那是夹具的形状，不是产品的行为。
 */
function seedExisting(state: ReturnType<typeof store>) {
    state.dispatch(
        applyTimelinePayload({
            ...overlappingTimeline(),
            clips: overlappingTimeline().clips.slice(0, 1),
        } as never),
    );
}

/** 按发生顺序记下每一次写回，用来断言"淡化在规格化之前"。 */
function recordWrites() {
    const order: string[] = [];
    vi.mocked(webApi.setClipsStateBulk).mockImplementation((async () => {
        order.push("auto-fade");
        return overlappingTimeline();
    }) as never);
    vi.mocked(webApi.setClipState).mockImplementation((async () => {
        order.push("normalize");
        return overlappingTimeline();
    }) as never);
    return order;
}

beforeEach(() => {
    vi.clearAllMocks();
    computeClipNormalizationGain.mockReset();
    window.__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "imports",
        hostCapabilities: araOnlyCaps({ mediaImport: true }),
    };
    vi.mocked(webApi.importAudioItem).mockResolvedValue(receiptTimeline() as never);
    vi.mocked(webApi.getTimelineState).mockResolvedValue(overlappingTimeline() as never);
});

afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

/**
 * 【回归】待定导入转正时**必须**补做自动交叉淡化。
 *
 * REAPER 的每一次导入都走这条路（现场 `after_ms=101/107`）。此前这里只补了规格化，
 * 于是"导入完成即看到交叉淡化"在 ARA 里从来没有发生过。
 */
test("the pending import runs the same post-import pipeline", async () => {
    const state = store();
    await state
        .dispatch(
            importAudioAtPosition({
                audioPath: "C:/audio/one.wav",
                trackId: "track",
                startSec: 2,
                normalizeAfterImport: true,
            }),
        )
        .unwrap();
    vi.mocked(webApi.getTimelineState).mockClear();
    computeClipNormalizationGain.mockReturnValue(2);
    const order = recordWrites();

    await state.dispatch(completePendingImport({ item: ITEM, clipId: CLIP })).unwrap();

    // 淡化写回**恰好一次**，且带的是重叠量。
    expect(order.filter((entry) => entry === "auto-fade")).toHaveLength(1);
    const bulk = vi.mocked(webApi.setClipsStateBulk).mock.calls[0][0] as {
        updates: Array<{ clipId: string; autoFadeInSec: number }>;
    };
    const imported = bulk.updates.find((update) => update.clipId === CLIP);
    expect(imported?.autoFadeInSec).toBeCloseTo(2, 9);
    // …而且发生在规格化之前（规格化改 itemGain，淡化改 autoFade，两者不许抢同一步）。
    expect(order).toEqual(["auto-fade", "normalize"]);
    // 转正这一刻才第一次有 source_path，规格化必须在这时算得出来。
    expect(computeClipNormalizationGain).toHaveBeenCalledTimes(1);
});

/** 【不回归】立即成功那条路仍然只跑一次，且顺序相同。 */
test("the immediate import still runs the pipeline exactly once", async () => {
    const state = store();
    seedExisting(state);
    vi.mocked(webApi.importAudioItem).mockResolvedValue(overlappingTimeline() as never);
    computeClipNormalizationGain.mockReturnValue(2);
    const order = recordWrites();

    await state
        .dispatch(
            importAudioAtPosition({
                audioPath: "C:/audio/one.wav",
                trackId: "track",
                startSec: 2,
                normalizeAfterImport: true,
            }),
        )
        .unwrap();

    expect(order).toEqual(["auto-fade", "normalize"]);
    expect(computeClipNormalizationGain).toHaveBeenCalledTimes(1);
});

/**
 * 【回归】待定回执不是最终快照。
 *
 * 回执里的 `clips` 是"导入请求的回声"：这一刻宿主还没建 region。把它当成最终时间线
 * 套进 store，等于把"导入前 + 一个占位"写成当前图 —— 接着 `import_ready` 又换成真图，
 * 用户看到时间线闪一下。
 */
test("a pending import does not apply a timeline without its clip", async () => {
    const state = store();
    await state
        .dispatch(
            importAudioAtPosition({
                audioPath: "C:/audio/one.wav",
                trackId: "track",
                startSec: 2,
                normalizeAfterImport: true,
            }),
        )
        .unwrap();
    const ids = state.getState().session.clips.map((clip) => clip.id);
    expect(ids).not.toContain(CLIP);
    expect(ids).not.toContain(EXISTING);
    // 账本仍然记得这条待定导入 —— 它要靠 `import_ready` 收尾。
    expect(state.getState().session.pendingImports).toHaveLength(1);
});

/** 【与 §5 同一条】关闭自动交叉淡化时，管线一次都不许写 autoFade。 */
test("the pipeline skips auto crossfade when the toggle is off", async () => {
    const state = store();
    seedExisting(state);
    state.dispatch(toggleAutoCrossfade());
    vi.mocked(webApi.importAudioItem).mockResolvedValue(overlappingTimeline() as never);
    computeClipNormalizationGain.mockReturnValue(2);
    const order = recordWrites();

    await state
        .dispatch(
            importAudioAtPosition({
                audioPath: "C:/audio/one.wav",
                trackId: "track",
                startSec: 2,
                normalizeAfterImport: true,
            }),
        )
        .unwrap();

    expect(order).toEqual(["normalize"]);
    expect(webApi.setClipsStateBulk).not.toHaveBeenCalled();
});
