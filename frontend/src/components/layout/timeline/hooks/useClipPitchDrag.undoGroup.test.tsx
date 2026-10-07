// @vitest-environment jsdom
/**
 * useClipPitchDrag 的 undo group 收尾契约。
 *
 * 回归点：`finish()` 曾以 `!st.base || st.currentCents === 0` 提前 return。用户拖离
 * 原点（已下发预览、已惰性开 undo group）后再拖回 0 松手时，endUndoGroup 永不调用：
 * 后端 suppress_checkpoints 永久置位，此后工程内一切撤销点被吞掉到重启；同时挂起的
 * 0 音分下发被清除，后端停在最后一个非零预览，与 UI 的 commit(0) 分叉。
 */
import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const mocks = vi.hoisted(() => ({
    beginUndoGroup: vi.fn(() => Promise.resolve()),
    endUndoGroup: vi.fn(() => Promise.resolve()),
    setParamFrames: vi.fn(() => Promise.resolve({ ok: true })),
    getParamFrames: vi.fn(() =>
        Promise.resolve({ ok: true, frame_period_ms: 5, edit: [60, 62, 0] }),
    ),
}));

vi.mock("../../../../services/webviewApi", () => ({
    webApi: {
        beginUndoGroup: mocks.beginUndoGroup,
        endUndoGroup: mocks.endUndoGroup,
        setParamFrames: mocks.setParamFrames,
        getParamFrames: mocks.getParamFrames,
    },
}));

import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { useClipPitchDrag } from "./useClipPitchDrag";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const CLIP_ID = "clip-1";
const TRACK_ID = "track-1";

function makeSession(): SessionState {
    return {
        clips: [{ id: CLIP_ID, trackId: TRACK_ID, startSec: 1, lengthSec: 2 }],
        tracks: [{ id: TRACK_ID, composeEnabled: true, pitchAnalysisAlgo: "lpc" }],
    } as unknown as SessionState;
}

type StartFn = (e: React.PointerEvent, clipId: string) => void;

let start: StartFn | null = null;

function Harness(): null {
    const sessionRef = React.useRef<SessionState>(makeSession());
    const { startClipPitchDrag } = useClipPitchDrag({
        sessionRef,
        dispatch: vi.fn() as unknown as AppDispatch,
        fineAdjustKb: { key: "", modifierOnly: true },
        formatDragTooltip: (cents) => String(cents),
    });
    React.useEffect(() => {
        start = startClipPitchDrag;
        return () => {
            start = null;
        };
    }, [startClipPitchDrag]);
    return null;
}

function downEvent(clientY: number): React.PointerEvent {
    return {
        button: 0,
        pointerId: 1,
        clientX: 100,
        clientY,
        preventDefault: () => {},
        stopPropagation: () => {},
        currentTarget: { setPointerCapture: () => {} },
    } as unknown as React.PointerEvent;
}

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    mocks.beginUndoGroup.mockClear();
    mocks.endUndoGroup.mockClear();
    mocks.setParamFrames.mockClear();
    mocks.getParamFrames.mockClear();
    start = null;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    act(() => root.render(<Harness />));
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

async function flush(times = 8): Promise<void> {
    for (let i = 0; i < times; i += 1) {
        await Promise.resolve();
    }
}

test("拖离原点再拖回原点后松手：undo group 必须关闭", async () => {
    expect(start).not.toBeNull();
    act(() => start!(downEvent(200), CLIP_ID));
    // 基准帧异步拉取完成。
    await act(async () => {
        await flush();
    });
    expect(mocks.getParamFrames).toHaveBeenCalled();

    // 拖离原点（非零音分）→ 触发节流下发（lastSentAt 为 0，wait 为 0）。
    act(() => {
        window.dispatchEvent(
            new PointerEvent("pointermove", {
                bubbles: true,
                pointerId: 1,
                clientX: 100,
                clientY: 120,
            }),
        );
    });
    // 立刻拖回原点：挂起中的 0 音分下发仍会开启 undo group。
    act(() => {
        window.dispatchEvent(
            new PointerEvent("pointermove", {
                bubbles: true,
                pointerId: 1,
                clientX: 100,
                clientY: 200,
            }),
        );
    });
    // 让节流定时器（0ms）触发：cents 为 0 也会 ensureUndoGroup → beginUndoGroup。
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        await flush();
    });
    expect(mocks.beginUndoGroup).toHaveBeenCalledTimes(1);
    expect(mocks.endUndoGroup).not.toHaveBeenCalled();

    // 松手：finish() 必须关闭已开的 undo group（旧实现因 currentCents===0 提前 return）。
    act(() => {
        window.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, pointerId: 1 }));
    });
    await act(async () => {
        await flush(16);
    });
    expect(mocks.endUndoGroup).toHaveBeenCalledTimes(1);
});
