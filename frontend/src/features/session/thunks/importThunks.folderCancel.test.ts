/**
 * 目录导入的撤销原子性（`importFolderAtPosition`）。
 *
 * 【要钉住的两条】
 * 1. **打点时机**：`begin_undo_group` 必须在 `add_track_tree` **之后**调用 ——
 *    撤销落点要是"有轨道树、无 clip"，而不是"轨道组一起消失"（那会让随后仍在
 *    跑的循环往已消失的 trackId 上灌 clip，撤销栈被写坏）。
 * 2. **取消闸门**：用户在循环进行中撤销（`cancelActiveImports`）后，循环提前
 *    退出、返回 `canceled` 且**不带**时间线快照（带回去会把刚撤销的东西画回来）。
 */

import { configureStore } from "@reduxjs/toolkit";
import { beforeEach, describe, expect, test, vi } from "vitest";

vi.mock("../../../services/webviewApi", () => ({
    webApi: {
        beginUndoGroup: vi.fn(async () => ({ ok: true })),
        endUndoGroup: vi.fn(async () => ({ ok: true })),
        addTrackTree: vi.fn(async () => ({
            createdTrackIds: ["t-folder", "t-file"],
            timeline: { ok: true, tracks: [], clips: [] },
        })),
        importAudioItem: vi.fn(async (file: string) => ({
            ok: true,
            clips: [{ id: `clip-${file}` }],
        })),
        getTimelineState: vi.fn(async () => ({ ok: true, tracks: [], clips: [] })),
    },
}));

import { webApi } from "../../../services/webviewApi";
import sessionReducer from "../sessionSlice";
import { cancelActiveImports, resetImportCancellationForTests } from "./importCancellation";
import { importFolderAtPosition } from "./importThunks";

const addTrackTree = webApi.addTrackTree as unknown as ReturnType<typeof vi.fn>;
const beginUndoGroup = webApi.beginUndoGroup as unknown as ReturnType<typeof vi.fn>;
const importAudioItem = webApi.importAudioItem as unknown as ReturnType<typeof vi.fn>;

function createStore() {
    return configureStore({ reducer: { session: sessionReducer } });
}

/** 一个文件夹 + 其中 3 个媒体文件 → specs = 文件夹轨道 + 3 条文件轨道。 */
const payload = {
    roots: [
        {
            name: "Takes",
            files: ["C:\\m\\a.wav", "C:\\m\\b.wav", "C:\\m\\c.wav"],
            children: [],
        },
    ],
    looseFiles: [],
    orderedFiles: ["C:\\m\\a.wav", "C:\\m\\b.wav", "C:\\m\\c.wav"],
    mode: "across-tracks" as const,
    createFolderTracks: true,
    trackId: null,
    startSec: 0,
};

beforeEach(() => {
    vi.clearAllMocks();
    resetImportCancellationForTests();
    addTrackTree.mockResolvedValue({
        createdTrackIds: ["t-folder", "t-a", "t-b", "t-c"],
        timeline: { ok: true, tracks: [], clips: [] },
    });
    importAudioItem.mockImplementation(async (file: string) => ({
        ok: true,
        clips: [{ id: `clip-${file}` }],
    }));
});

describe("打点时机", () => {
    test("begin_undo_group 在 add_track_tree **之前**（一次撤销撤掉整个导入，含新轨道）", async () => {
        const order: string[] = [];
        beginUndoGroup.mockImplementation(async () => {
            order.push("beginUndoGroup");
            return { ok: true };
        });
        addTrackTree.mockImplementation(async () => {
            order.push("addTrackTree");
            return {
                createdTrackIds: ["t-folder", "t-a", "t-b", "t-c"],
                timeline: { ok: true, tracks: [], clips: [] },
            };
        });

        const store = createStore();
        await store.dispatch(importFolderAtPosition(payload));
        // 打点在建树之前 = 撤销落点是"导入前"，因导入新建的轨道随之一起被撤销。
        expect(order).toEqual(["beginUndoGroup", "addTrackTree"]);
    });
});

describe("取消闸门", () => {
    test("循环进行中撤销 → 提前退出、返回 canceled、不带时间线快照", async () => {
        // 用户在第二个文件导入时按下 Ctrl+Z。
        importAudioItem.mockImplementation(async (file: string) => {
            if (file.endsWith("b.wav")) cancelActiveImports();
            return { ok: true, clips: [{ id: `clip-${file}` }] };
        });

        const store = createStore();
        const result = (await store.dispatch(importFolderAtPosition(payload))).payload as {
            canceled?: boolean;
            imported?: unknown;
            newClipIds?: string[];
        };

        expect(result.canceled).toBe(true);
        // 关键：不能带回快照 —— 否则 reducer 会把用户刚撤销掉的时间线又画回来。
        expect(result.imported).toBeNull();
        expect(result.newClipIds).toEqual([]);
        // 第三个文件不再被导入（循环在下一轮检查处退出）。
        const importedFiles = importAudioItem.mock.calls.map((call) => call[0] as string);
        expect(importedFiles).toEqual(["C:\\m\\a.wav", "C:\\m\\b.wav"]);
    });

    test("未撤销时正常收尾（返回成功与汇总）", async () => {
        const store = createStore();
        const result = (await store.dispatch(importFolderAtPosition(payload))).payload as {
            ok?: boolean;
            canceled?: boolean;
            attempted?: number;
            failedFiles?: string[];
        };
        expect(result.ok).toBe(true);
        expect(result.canceled).toBeUndefined();
        expect(result.attempted).toBe(3);
        expect(result.failedFiles).toEqual([]);
    });

    test("撤销落在「建树在途」窗口里 → 不再继续导入任何文件", async () => {
        // 用户按下 Ctrl+Z 的瞬间 add_track_tree 正在途中。`notifyHistoryJump` 会等
        // 这一步收尾后才真正跳转（因此那棵树随后被这次撤销一并还原），这里只需
        // 停止继续导入。
        addTrackTree.mockImplementation(async () => {
            cancelActiveImports();
            return {
                createdTrackIds: ["t-folder", "t-a", "t-b", "t-c"],
                timeline: { ok: true, tracks: [], clips: [] },
            };
        });

        const store = createStore();
        const result = (await store.dispatch(importFolderAtPosition(payload))).payload as {
            canceled?: boolean;
        };

        expect(result.canceled).toBe(true);
        expect(importAudioItem).not.toHaveBeenCalled();
    });
});
