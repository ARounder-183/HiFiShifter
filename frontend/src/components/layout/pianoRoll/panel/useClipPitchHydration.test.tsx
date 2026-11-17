// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getClipPitchData, listeners } = vi.hoisted(() => ({
    getClipPitchData: vi.fn(),
    listeners: new Map<string, Array<(event: unknown) => void>>(),
}));

vi.mock("../../../../services/api", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../../../services/api")>();
    return {
        ...actual,
        paramsApi: { ...actual.paramsApi, getClipPitchData },
    };
});

// 宿主事件口：本组件会订阅 `pitch_orig_updated`，分析完成后再取一次参考线。
vi.mock("../../../../services/hostEvents", () => ({
    listen: async (event: string, handler: (payload: unknown) => void) => {
        const list = listeners.get(event) ?? [];
        list.push(handler);
        listeners.set(event, list);
        return () => {
            listeners.set(
                event,
                (listeners.get(event) ?? []).filter((known) => known !== handler),
            );
        };
    },
}));

import { useClipPitchHydration } from "./useClipPitchHydration";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
const dispatched: unknown[] = [];

function Probe(props: {
    editParam: string;
    show: boolean;
    clips: Array<{ id: string }>;
    curves: Record<string, unknown>;
}) {
    useClipPitchHydration({
        dispatch: ((action: unknown) => dispatched.push(action)) as never,
        editParam: props.editParam,
        clips: props.clips as never,
        session: { showClipPitchOverlay: props.show, clipPitchCurves: props.curves },
    });
    return null;
}

async function flush(): Promise<void> {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
}

beforeEach(() => {
    getClipPitchData.mockReset();
    getClipPitchData.mockResolvedValue({
        ok: true,
        payloads: [
            {
                clip_id: "b",
                curve_start_sec: 1,
                midi_curve: [60, 61],
                frame_period_ms: 5,
            },
        ],
    });
    dispatched.length = 0;
    listeners.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

describe("useClipPitchHydration", () => {
    it("只请求缺失的 pitch 曲线，并把 payload 写回 session", async () => {
        act(() =>
            root.render(
                <Probe
                    editParam="pitch"
                    show
                    clips={[{ id: "a" }, { id: "b" }, { id: "b" }]}
                    curves={{ a: { midiCurve: [60] } }}
                />,
            ),
        );
        await flush();
        expect(getClipPitchData).toHaveBeenCalledWith(["b"]);
        expect(dispatched).toHaveLength(1);
        expect(dispatched[0]).toMatchObject({
            type: "session/setClipPitchData",
            payload: { clipId: "b", curveStartSec: 1, midiCurve: [60, 61], framePeriodMs: 5 },
        });
    });

    it("分析完成后清空已请求集合并重取，参考线不会永久空白", async () => {
        act(() => root.render(<Probe editParam="pitch" show clips={[{ id: "b" }]} curves={{}} />));
        await flush();
        expect(getClipPitchData).toHaveBeenCalledTimes(1);
        // 这一轮没有 payload（分析还没完成）⇒ 前端不缓存、等事件。
        const handler = listeners.get("pitch_orig_updated")?.[0];
        expect(handler, "必须订阅分析完成事件").toBeTypeOf("function");
        act(() => handler?.({ payload: { rootTrackId: "r" } }));
        await flush();
        expect(getClipPitchData).toHaveBeenCalledTimes(2);
    });

    it("非 pitch 参数或关闭参考线时不发请求", async () => {
        act(() =>
            root.render(
                <Probe editParam="volume" show={false} clips={[{ id: "a" }]} curves={{}} />,
            ),
        );
        await flush();
        expect(getClipPitchData).not.toHaveBeenCalled();
    });
});
