/**
 * 导入被取消时 reducer 的收尾。
 *
 * 【要钉住的性质】`{ ok: true, canceled: true, imported: null }` 必须被翻成
 * "已取消导入"，而**不是**"Import done"；且**不得**套用任何时间线快照 ——
 * 那会把用户刚撤销掉的东西又画回来。
 */

import { expect, test } from "vitest";

import reducer from "./sessionSlice.js";
import {
    importAudioAtPosition,
    importAudioFileAtPosition,
    importAudioFromPath,
    importFolderAtPosition,
    importMidiAsClip,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "./thunks/importThunks.js";

type State = ReturnType<typeof reducer>;

function createState(): State {
    return reducer(undefined, { type: "@@INIT" });
}

const canceledPayload = { ok: true, canceled: true, imported: null, newClipIds: [] };

test("importFolderAtPosition：取消 → 状态栏显示已取消，不套用快照", () => {
    const before = createState();
    const after = reducer(
        before,
        importFolderAtPosition.fulfilled(canceledPayload, "req", {} as never),
    );
    expect((after as { status?: string }).status).toBe("Import canceled");
    expect((after as { clips?: unknown[] }).clips).toEqual((before as { clips?: unknown[] }).clips);
    expect((after as { selectedClipId?: unknown }).selectedClipId).toBe(
        (before as { selectedClipId?: unknown }).selectedClipId,
    );
});

test("importMultipleAudioAtPosition：取消 → 已取消，不套用快照", () => {
    const before = createState();
    const after = reducer(
        before,
        importMultipleAudioAtPosition.fulfilled(canceledPayload, "req", {} as never),
    );
    expect((after as { status?: string }).status).toBe("Import canceled");
    expect((after as { clips?: unknown[] }).clips).toEqual((before as { clips?: unknown[] }).clips);
});

test("importMultipleAudioFilesAtPosition：取消 → 已取消，不套用快照", () => {
    const before = createState();
    const after = reducer(
        before,
        importMultipleAudioFilesAtPosition.fulfilled(canceledPayload, "req", {} as never),
    );
    expect((after as { status?: string }).status).toBe("Import canceled");
    expect((after as { clips?: unknown[] }).clips).toEqual((before as { clips?: unknown[] }).clips);
});

// 单文件导入同样带取消闸门（撤销 / 切工程后不得把快照带回来，也不得报"导入完成"）。
for (const thunk of [
    importAudioAtPosition,
    importAudioFileAtPosition,
    importAudioFromPath,
    importMidiAsClip,
]) {
    test(`${thunk.typePrefix}：取消 → 已取消，不套用快照`, () => {
        const before = createState();
        const after = reducer(before, thunk.fulfilled(canceledPayload, "req", {} as never));
        expect((after as { status?: string }).status).toBe("Import canceled");
        expect((after as { clips?: unknown[] }).clips).toEqual(
            (before as { clips?: unknown[] }).clips,
        );
    });
}
