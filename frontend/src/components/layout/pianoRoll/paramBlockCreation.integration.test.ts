import { expect, test } from "vitest";
import {
    createParamBlock,
    materializeParamBlocks,
    projectParamBlockCreatePreviewFromBase,
    EMPTY_PARAM_BLOCK_STATE,
    mergeParamBlocks,
    type ParamBlockSourceClip,
} from "./paramBlocks";

function sources(): ParamBlockSourceClip[] {
    return Array.from({ length: 35 }, (_, index) => ({
        id: `clip-${index}`,
        trackId: "root",
        takeId: `take-${index}`,
        startSec: index,
        lengthSec: 1,
        sourceStartSec: 0,
        sourceEndSec: 1,
        sourceSyllableSpansSec: [
            [0, 0.5],
            [0.5, 1],
        ],
        playbackRate: 1,
    }));
}

test("creating one block after GAME preserves blocks in all other Clips", () => {
    const clips = sources();
    const before = materializeParamBlocks({
        rootTrackId: "root",
        clips,
        state: EMPTY_PARAM_BLOCK_STATE,
    });
    expect(before).toHaveLength(70);
    const created = createParamBlock({
        state: EMPTY_PARAM_BLOCK_STATE,
        clip: clips[0],
        existingBlocks: before,
        startSec: 0.1,
        endSec: 0.4,
        value: 61.5,
    });
    if (!created.ok) throw new Error(created.reason);
    const after = materializeParamBlocks({ rootTrackId: "root", clips, state: created.state });
    expect(after.filter((block) => block.startSec >= 1)).toEqual(
        before.filter((block) => block.startSec >= 1),
    );
    expect(after.find((block) => block.id === created.blockId)?.manualValue).toBe(61.5);
});

test("cross-Clip creation commits the same span as its preview and survives readback", () => {
    const clips = sources();
    const before = materializeParamBlocks({
        rootTrackId: "root",
        clips,
        state: EMPTY_PARAM_BLOCK_STATE,
    });
    const preview = projectParamBlockCreatePreviewFromBase({
        baseBlocks: before.map((block) => ({ ...block, selected: false })),
        createId: "created",
        startSec: 0.2,
        endSec: 1.4,
        value: 62.25,
    });
    const created = createParamBlock({
        state: EMPTY_PARAM_BLOCK_STATE,
        clip: clips[0],
        clips,
        blockId: "created",
        existingBlocks: before,
        startSec: 0.2,
        endSec: 1.4,
        value: 62.25,
    });
    if (!created.ok) throw new Error(created.reason);
    const after = materializeParamBlocks({
        rootTrackId: "root",
        clips,
        state: JSON.parse(JSON.stringify(created.state)),
    });
    const committed = after.find((block) => block.id === "created")!;
    const shown = preview.find((block) => block.id === "created")!;
    expect([committed.startSec, committed.endSec]).toEqual([shown.startSec, shown.endSec]);
    expect(committed.manualValue).toBe(62.25);
    expect(committed.members.map((member) => member.clipId)).toEqual(["clip-0", "clip-1"]);
    expect(after.filter((block) => block.startSec >= 2)).toEqual(
        before.filter((block) => block.startSec >= 2),
    );
});

test("creation in one Clip preserves an existing merge in unrelated Clips", () => {
    const clips = sources();
    const auto = materializeParamBlocks({
        rootTrackId: "root",
        clips,
        state: EMPTY_PARAM_BLOCK_STATE,
    });
    const merged = mergeParamBlocks({
        state: EMPTY_PARAM_BLOCK_STATE,
        blocks: auto,
        selectedBlockIds: auto
            .filter((block) => block.startSec >= 2 && block.endSec <= 4)
            .map((block) => block.id),
    });
    if (!merged.ok) throw new Error(merged.reason);
    const before = materializeParamBlocks({ rootTrackId: "root", clips, state: merged.state });
    const created = createParamBlock({
        state: merged.state,
        clip: clips[0],
        clips,
        existingBlocks: before,
        startSec: 0.1,
        endSec: 0.4,
        value: 61.5,
    });
    if (!created.ok) throw new Error(created.reason);
    const after = materializeParamBlocks({ rootTrackId: "root", clips, state: created.state });
    expect(after.filter((block) => block.startSec >= 2)).toEqual(
        before.filter((block) => block.startSec >= 2),
    );
});
