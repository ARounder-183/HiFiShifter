// @vitest-environment jsdom
//
// 共享视觉时钟回归：时间轴与参数编辑器必须拿到**同一个**帧值；并且在任何
// onFrame 回调执行前，所有消费者的 ref 都已经写好。后者是自动滚屏 + 同步时间轴
// 时"参数编辑器光标抽动"的直接防线。
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { resetVisualPlayheadClockForTests, useVisualPlayhead } from "./useVisualPlayhead";

let rafCallbacks = new Map<number, FrameRequestCallback>();
let nextRafId = 1;
const nowMs = 1000;

beforeEach(() => {
    vi.spyOn(performance, "now").mockImplementation(() => nowMs);
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
        const id = nextRafId++;
        rafCallbacks.set(id, callback);
        return id;
    });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => {
        rafCallbacks.delete(id);
    });
    resetVisualPlayheadClockForTests();
    rafCallbacks = new Map();
    nextRafId = 1;
});

afterEach(() => {
    resetVisualPlayheadClockForTests();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

test("all consumers receive one shared frame value and refs update before callbacks", () => {
    const refs: Array<{ current: number } | null> = [null, null];
    const seen: number[] = [];
    let armed = false;

    function Probe({ index }: { index: number }) {
        const ref = useVisualPlayhead({
            syncedPlayheadSec: 0,
            syncedAtMs: nowMs - 10,
            isTransportAdvancing: true,
            onFrame: (sec) => {
                seen.push(sec);
                if (armed && index === 0) {
                    // 第二阶段之前，第二个消费者的 ref 必须已经是同一个值。
                    expect(refs[1]?.current).toBe(sec);
                }
            },
        });
        refs[index] = ref;
        return null;
    }

    const host = document.createElement("div");
    const root = createRoot(host);
    act(() => {
        root.render(
            <>
                <Probe index={0} />
                <Probe index={1} />
            </>,
        );
    });

    // 两个 hook 共享一条 rAF；订阅后应只有一个待执行帧。
    expect(rafCallbacks.size).toBe(1);
    const callbacks = Array.from(rafCallbacks.values());
    rafCallbacks.clear();
    armed = true;

    act(() => {
        callbacks[0](nowMs + 16);
    });

    const frameValues = seen.slice(-2);
    expect(frameValues).toHaveLength(2);
    expect(frameValues[0]).toBe(frameValues[1]);
    // 本帧回调结束后只排下一帧，没有每个消费者各排一条。
    expect(rafCallbacks.size).toBe(1);

    act(() => root.unmount());
});
