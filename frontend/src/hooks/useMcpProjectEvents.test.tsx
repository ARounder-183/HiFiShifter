// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const fake = vi.hoisted(() => ({
    handlers: new Map<string, (event: { payload: unknown }) => void>(),
    subscribers: new Set<() => void>(),
    session: {
        _interactionLockCount: 0,
        selectedTrackId: "vocal",
        selectedClipId: null,
        editParam: "pitch",
        visibleReferenceRootTrackIds: [],
        toolMode: "select",
    },
    dispatch: vi.fn(),
    view: vi.fn(),
    acknowledge: vi.fn(),
    reply: vi.fn(),
}));
vi.mock("../app/hooks", () => ({
    useAppDispatch: () => fake.dispatch,
    useAppStore: () => store,
}));
const store = {
    getState: () => ({ session: fake.session }),
    subscribe: (callback: () => void) => {
        fake.subscribers.add(callback);
        return () => fake.subscribers.delete(callback);
    },
};
vi.mock("../services/hostCapabilities", () => ({ isPluginMode: () => false }));
vi.mock("../services/hostEvents", () => ({
    listen: vi.fn(async (name: string, handler: (event: { payload: unknown }) => void) => {
        fake.handlers.set(name, handler);
        return () => fake.handlers.delete(name);
    }),
}));
vi.mock("../services/api/mcp", () => ({
    mcpApi: {
        view: fake.view,
        acknowledgeView: fake.acknowledge,
        replyView: fake.reply,
    },
}));
vi.mock("../features/session/thunks/transportThunks", () => ({
    fetchTimeline: {
        fulfilled: (payload: unknown) => ({ type: "timeline/fulfilled", payload }),
    },
}));
vi.mock("../features/session/thunks/timelineThunks", () => ({
    selectTrackRemote: (id: string) => ({ type: "track/select", payload: id }),
}));
vi.mock("../features/session/sessionSlice", () => ({
    bumpParamsEpoch: () => ({ type: "params/refresh" }),
    setEditParam: (payload: unknown) => ({ type: "param/select", payload }),
    setSelectedClip: (payload: unknown) => ({ type: "clip/select", payload }),
    setToolMode: (payload: unknown) => ({ type: "tool/select", payload }),
    setVisibleReferenceRootTrackIds: (payload: unknown) => ({ type: "reference/select", payload }),
}));
import { useMcpProjectEvents } from "./useMcpProjectEvents";

function renderHook(hook: () => void) {
    const host = document.createElement("div");
    const root = createRoot(host);
    function Probe() {
        hook();
        return null;
    }
    act(() => root.render(<Probe />));
    return { unmount: () => act(() => root.unmount()) };
}
async function waitFor(check: () => void) {
    await act(async () => {
        await vi.waitFor(check);
    });
}

beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    vi.clearAllMocks();
    fake.handlers.clear();
    fake.subscribers.clear();
    fake.session._interactionLockCount = 0;
    fake.view.mockResolvedValue({
        ok: true,
        revision: "r1",
        timeline: { ok: true, tracks: [], clips: [] },
    });
    fake.acknowledge.mockResolvedValue({ ok: true });
    fake.reply.mockResolvedValue({ ok: true });
    vi.stubGlobal("requestAnimationFrame", (callback: (time: number) => void) => {
        queueMicrotask(() => callback(0));
        return 1;
    });
});
afterEach(() => vi.unstubAllGlobals());

test("backend edits update the visible timeline and parameter epoch before acknowledgement", async () => {
    const { unmount } = renderHook(() => useMcpProjectEvents());
    await waitFor(() => expect(fake.acknowledge).toHaveBeenCalledWith("r1"));
    expect(fake.dispatch.mock.calls.map(([action]) => action.type)).toEqual([
        "timeline/fulfilled",
        "params/refresh",
    ]);
    fake.view.mockResolvedValue({
        ok: true,
        revision: "r2",
        timeline: { ok: true, tracks: [{ id: "agent" }], clips: [] },
    });
    await act(async () =>
        fake.handlers.get("mcp_project_changed")?.({ payload: { revision: "r2" } }),
    );
    await waitFor(() => expect(fake.acknowledge).toHaveBeenLastCalledWith("r2"));
    expect(fake.dispatch.mock.calls.at(-2)?.[0].payload.tracks[0].id).toBe("agent");
    unmount();
    expect(fake.handlers.size).toBe(0);
    expect(fake.subscribers.size).toBe(0);
});

test("MCP refresh waits for a human gesture and does not overwrite its preview", async () => {
    const { unmount } = renderHook(() => useMcpProjectEvents());
    await waitFor(() => expect(fake.acknowledge).toHaveBeenCalled());
    fake.view.mockClear();
    fake.dispatch.mockClear();
    fake.session._interactionLockCount = 1;
    await act(async () =>
        fake.handlers.get("mcp_project_changed")?.({ payload: { revision: "r2" } }),
    );
    expect(fake.view).not.toHaveBeenCalled();
    expect(fake.dispatch).not.toHaveBeenCalled();
    fake.session._interactionLockCount = 0;
    await act(async () => {
        fake.subscribers.forEach((callback) => callback());
    });
    await waitFor(() => expect(fake.view).toHaveBeenCalledTimes(1));
    unmount();
});

test("view queries return the live acknowledged GUI state and reject changes during a gesture", async () => {
    const { unmount } = renderHook(() => useMcpProjectEvents());
    await waitFor(() => expect(fake.handlers.has("mcp_view_request")).toBe(true));
    await act(async () =>
        fake.handlers.get("mcp_view_request")?.({ payload: { request_id: "read", request: {} } }),
    );
    await waitFor(() =>
        expect(fake.reply).toHaveBeenCalledWith(
            "read",
            expect.objectContaining({
                ok: true,
                active_param: "pitch",
                selected_track_id: "vocal",
            }),
        ),
    );
    fake.session._interactionLockCount = 1;
    await act(async () =>
        fake.handlers.get("mcp_view_request")?.({
            payload: { request_id: "busy", request: { active_param: "volume" } },
        }),
    );
    await waitFor(() =>
        expect(fake.reply).toHaveBeenCalledWith("busy", { ok: false, code: "VIEW_BUSY" }),
    );
    unmount();
});
