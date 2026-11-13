// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getClipPitchData } = vi.hoisted(() => ({ getClipPitchData: vi.fn() }));

vi.mock("../../../../services/api", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../../../services/api")>();
    return {
        ...actual,
        paramsApi: { ...actual.paramsApi, getClipPitchData },
    };
});

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
