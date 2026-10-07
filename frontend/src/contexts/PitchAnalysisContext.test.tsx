// @vitest-environment jsdom
/**
 * 「历史跳转后复位音高分析进度」的契约。
 *
 * 【要钉住的问题】撤销 / 跳转历史会把时间线整体换成另一份快照 —— 被导入的 clip
 * 已经不在时间线上，但后端在途的分析线程只认工程代次，仍会陆续发来进度事件。
 * 若不处理，状态栏会在撤销之后重新亮起"正在分析音高"并长时间不退。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

/** 记录型事件监听桩：把 `listen` 的回调收集起来，测试手动触发。 */
vi.mock("@tauri-apps/api/event", () => {
    const handlers = new Map<string, Set<(event: unknown) => void>>();
    return {
        listen: async (name: string, cb: (event: unknown) => void) => {
            const set = handlers.get(name) ?? new Set();
            set.add(cb);
            handlers.set(name, set);
            return () => set.delete(cb);
        },
        __handlers: handlers,
    };
});

vi.mock("../services/api", () => ({
    coreApi: { getPitchAnalysisProgress: async () => null },
}));

import * as tauriEvent from "@tauri-apps/api/event";

import { HISTORY_JUMP_EVENT } from "../features/session/historyJump";
import { PitchAnalysisProvider, usePitchAnalysis } from "./PitchAnalysisContext";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const handlers = (
    tauriEvent as unknown as {
        __handlers: Map<string, Set<(event: unknown) => void>>;
    }
).__handlers;

function emit(name: string, payload: unknown): void {
    for (const cb of handlers.get(name) ?? []) cb({ payload });
}

function Probe() {
    const state = usePitchAnalysis();
    return <span data-testid="pending">{String(state.pending)}</span>;
}

let container: HTMLDivElement;
let root: Root;

beforeEach(async () => {
    handlers.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => {
        root.render(
            <PitchAnalysisProvider>
                <Probe />
            </PitchAnalysisProvider>,
        );
    });
    // 让 `setup()` 里的动态 import + listen 注册完成。
    await act(async () => {
        await Promise.resolve();
    });
});

afterEach(async () => {
    await act(async () => {
        root.unmount();
    });
    container.remove();
});

function pendingText(): string {
    return container.querySelector('[data-testid="pending"]')?.textContent ?? "";
}

test("历史跳转后立刻复位，并忽略旧批次的进度事件", async () => {
    await act(async () => {
        emit("pitch_orig_analysis_started", { rootTrackId: "r1", key: "" });
    });
    expect(pendingText()).toBe("true");

    // 用户撤销：进度立即复位。
    await act(async () => {
        window.dispatchEvent(new CustomEvent(HISTORY_JUMP_EVENT));
    });
    expect(pendingText()).toBe("false");

    // 旧批次（分析对象已消失）的进度事件不得把它重新点亮。
    await act(async () => {
        emit("pitch_orig_analysis_progress", {
            rootTrackId: "r1",
            progress: 0.4,
            currentClipName: "gone.wav",
            completedClips: 2,
            totalClips: 5,
        });
    });
    expect(pendingText()).toBe("false");
});

test("新批次（先发 started）恢复跟踪", async () => {
    await act(async () => {
        emit("pitch_orig_analysis_started", { rootTrackId: "r1", key: "" });
        window.dispatchEvent(new CustomEvent(HISTORY_JUMP_EVENT));
    });
    expect(pendingText()).toBe("false");

    // 真正的新一批分析：started 之后进度重新被采纳。
    await act(async () => {
        emit("pitch_orig_analysis_started", { rootTrackId: "r2", key: "" });
    });
    expect(pendingText()).toBe("true");
    await act(async () => {
        emit("pitch_orig_analysis_progress", {
            rootTrackId: "r2",
            progress: 0.5,
            currentClipName: "new.wav",
            completedClips: 1,
            totalClips: 2,
        });
    });
    expect(pendingText()).toBe("true");
});

test("完成事件同样复位（无论是否经过历史跳转）", async () => {
    await act(async () => {
        emit("pitch_orig_analysis_started", { rootTrackId: "r1", key: "" });
    });
    expect(pendingText()).toBe("true");
    await act(async () => {
        emit("pitch_orig_updated", { rootTrackId: "r1" });
    });
    expect(pendingText()).toBe("false");
});
