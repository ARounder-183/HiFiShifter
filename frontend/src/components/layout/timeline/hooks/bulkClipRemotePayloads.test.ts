import { test } from "vitest";

import {
    buildBulkClipStateUpdates,
    buildDuplicateClipsBulkPayload,
    itemGainChange,
    takeGainChange,
} from "./bulkClipRemotePayloads.js";

/**
 * **Item 层与 take 层各自只有一个构造器，且载荷里绝不出现裸 `gain`。**
 *
 * 【为什么这条测试是这一轮的核心】REAPER 每个 clip 有两个增益旋钮（item 的 `D_VOL`
 * 与 active take 的 `D_VOL`），而它们在载荷里只差一个前缀。此前载荷发的是裸 `gain`，
 * 内核把它读成 take 层、界面乐观更新写的是 item 层 —— 于是"旋钮 +12 dB → 行内编辑
 * 回 0 dB"之后，两层各自停在对方的值上（现场那对镜像读数）。这条测试钉的是：
 * 载荷里**没有**裸 `gain` 这个键可以选错。
 */
test("gain payloads name their layer", () => {
    function assertDeepEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    // 时间轴旋钮 / 行内编辑 / 双击复位 / 归一化：item 层。
    assertDeepEqual(itemGainChange(1.2), { itemGain: 1.2 }, "item layer constructor");
    assertDeepEqual(takeGainChange(4), { takeGain: 4 }, "take layer constructor");

    const updates = buildBulkClipStateUpdates({
        clipIds: ["a", "b"],
        changesById: new Map([
            ["a", itemGainChange(1.2)],
            ["b", { muted: true, fadeInSec: 0.4 }],
        ]),
    });

    assertDeepEqual(
        updates,
        [
            { clipId: "a", itemGain: 1.2 },
            { clipId: "b", muted: true, fadeInSec: 0.4 },
        ],
        "bulk state payload",
    );
    // 逐条断言键名：`gain` 一旦回来，这里就是第一个失败点。
    for (const update of updates) {
        if ("gain" in update) {
            throw new Error(`bulk update must not carry a bare gain: ${JSON.stringify(update)}`);
        }
    }

    const duplicatePayload = buildDuplicateClipsBulkPayload({
        sourceClipIds: ["a", "b"],
        deltaSec: 1.5,
        copyLinkedParams: true,
        applyAutoCrossfade: true,
        trackMode: { kind: "offset_tracks", offset: 1 },
    });

    assertDeepEqual(
        duplicatePayload,
        {
            sourceClipIds: ["a", "b"],
            deltaSec: 1.5,
            copyLinkedParams: true,
            applyAutoCrossfade: true,
            selectCreatedClips: true,
            trackMode: { kind: "offset_tracks", offset: 1 },
        },
        "bulk duplicate payload",
    );

    const stateUpdates = buildBulkClipStateUpdates({
        clipIds: ["clip-a"],
        changesById: new Map([["clip-a", { muted: false }]]),
    });

    assertDeepEqual(stateUpdates, [{ clipId: "clip-a", muted: false }], "single state update");

    const duplicatePayloadForPaste = buildDuplicateClipsBulkPayload({
        sourceClipIds: ["clip-1"],
        deltaSec: 0,
        copyLinkedParams: true,
        applyAutoCrossfade: false,
        trackMode: { kind: "same_track" },
    });

    assertDeepEqual(
        duplicatePayloadForPaste.trackMode,
        { kind: "same_track" },
        "paste can reuse duplicate bulk payload",
    );

    const duplicatePayloadWithoutRename = buildDuplicateClipsBulkPayload({
        sourceClipIds: ["clip-2"],
        deltaSec: 2,
        copyLinkedParams: true,
        applyAutoCrossfade: false,
        trackMode: { kind: "explicit_mapping", mapping: { t1: "t2" } },
        renameCopies: false,
    });

    assertDeepEqual(
        duplicatePayloadWithoutRename.renameCopies,
        false,
        "duplicate payload can preserve source names",
    );
});
