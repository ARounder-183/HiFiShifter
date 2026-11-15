// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mock = vi.hoisted(() => ({
    calls: [] as string[],
    listener: null as null | ((event: { payload?: { path?: unknown } }) => void),
}));

vi.mock("../../services/hostEvents", () => ({
    listen: vi.fn(async (_name: string, handler: typeof mock.listener) => {
        mock.calls.push("listen");
        mock.listener = handler;
        return () => undefined;
    }),
}));

vi.mock("../../services/api/project", () => ({
    projectApi: {
        markFrontendReady: vi.fn(async () => {
            mock.calls.push("ready");
            return { ok: true };
        }),
        consumeStartupProjectPath: vi.fn(async () => {
            mock.calls.push("consume");
            return { ok: true, path: "/tmp/startup.hshp" };
        }),
    },
}));

vi.mock("../../services/hostDetect", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../services/hostDetect")>();
    return { ...actual, isPluginMode: () => false };
});

import { useExternalProjectOpen } from "./useExternalProjectOpen";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
const opened: Array<[string, string]> = [];

function Probe() {
    useExternalProjectOpen((kind, path) => opened.push([kind, path]));
    return null;
}

async function flush(): Promise<void> {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
    });
}

beforeEach(() => {
    mock.calls.length = 0;
    mock.listener = null;
    opened.length = 0;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

describe("useExternalProjectOpen", () => {
    it("先注册监听、再报 ready、最后消费 pending 路径", async () => {
        await act(async () => root.render(<Probe />));
        await flush();
        expect(mock.calls).toEqual(["listen", "ready", "consume"]);
        expect(opened).toEqual([["openProject", "/tmp/startup.hshp"]]);
    });

    it("已运行时收到 Tauri Opened 事件会打开新工程", async () => {
        await act(async () => root.render(<Probe />));
        await flush();
        opened.length = 0;
        act(() => mock.listener?.({ payload: { path: "/tmp/hot.hsp" } }));
        expect(opened).toEqual([["openProject", "/tmp/hot.hsp"]]);
    });

    it("不支持的路径被忽略", async () => {
        await act(async () => root.render(<Probe />));
        await flush();
        opened.length = 0;
        act(() => mock.listener?.({ payload: { path: "/tmp/readme.txt" } }));
        expect(opened).toEqual([]);
    });
});
