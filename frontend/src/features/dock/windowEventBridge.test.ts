// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

const mock = vi.hoisted(() => ({
    emitted: [] as Array<{ event: string; payload: unknown }>,
    handler: null as null | ((event: { payload: unknown }) => void),
    activeEditor: null as string | null,
    label: "main",
}));

vi.mock("../../services/hostEvents", () => ({
    emit: vi.fn(async (event: string, payload?: unknown) => {
        mock.emitted.push({ event, payload });
    }),
    listen: vi.fn(async (_event: string, handler: (event: { payload: unknown }) => void) => {
        mock.handler = handler;
        return () => undefined;
    }),
}));

vi.mock("../../app/store", () => ({
    store: {
        getState: () => ({ interaction: { activeEditorWindow: mock.activeEditor } }),
    },
}));

vi.mock("./detachBridge", () => ({
    resolveWindowLabel: () => mock.label,
    isSatelliteWindow: () => mock.label !== "main",
}));

import {
    emitWindowEvent,
    installWindowEventBridge,
    WINDOW_EVENT_BRIDGE_EVENT,
} from "./windowEventBridge";

beforeEach(() => {
    mock.emitted.length = 0;
    mock.handler = null;
    mock.activeEditor = null;
    mock.label = "main";
});

describe("windowEventBridge", () => {
    it("目标为当前窗口时只本地 dispatch，不走桥", () => {
        const received: unknown[] = [];
        window.addEventListener("hifi:test-local", (event) => {
            received.push((event as CustomEvent).detail);
        });
        emitWindowEvent("main", "hifi:test-local", { value: 1 });
        expect(received).toEqual([{ value: 1 }]);
        expect(mock.emitted).toHaveLength(0);
    });

    it("卫星窗口发往 main 时生成目标窗口信封", () => {
        mock.label = "detached:pianoRoll";
        const received: unknown[] = [];
        window.addEventListener("hifi:test-main", (event) => {
            received.push((event as CustomEvent).detail);
        });
        emitWindowEvent("main", "hifi:test-main", { value: 2 });
        expect(received).toEqual([]);
        expect(mock.emitted).toHaveLength(1);
        expect(mock.emitted[0].event).toBe(WINDOW_EVENT_BRIDGE_EVENT);
        expect(mock.emitted[0].payload).toMatchObject({
            origin: "detached:pianoRoll",
            targetWindow: "main",
            name: "hifi:test-main",
            detail: { value: 2 },
        });
    });

    it("activeEditor 目标使用 interaction.activeEditorWindow", () => {
        mock.label = "main";
        mock.activeEditor = "detached:pianoRoll";
        emitWindowEvent("activeEditor", "hifi:test-active", { value: 3 });
        expect(mock.emitted[0].payload).toMatchObject({
            targetWindow: "detached:pianoRoll",
            name: "hifi:test-active",
        });
    });

    it("接收端只投递目标窗口且忽略自己的回声", async () => {
        installWindowEventBridge();
        await Promise.resolve();
        const received: unknown[] = [];
        window.addEventListener("hifi:test-remote", (event) => {
            received.push((event as CustomEvent).detail);
        });
        // 同一 origin：忽略。
        mock.handler?.({
            payload: { origin: "main", targetWindow: "main", name: "hifi:test-remote", detail: 1 },
        });
        expect(received).toEqual([]);
        // 别的 origin、目标 main：本地投递。
        mock.handler?.({
            payload: {
                origin: "detached:x",
                targetWindow: "main",
                name: "hifi:test-remote",
                detail: 2,
            },
        });
        expect(received).toEqual([2]);
        // 目标不是 main：忽略。
        mock.handler?.({
            payload: {
                origin: "detached:x",
                targetWindow: "detached:y",
                name: "hifi:test-remote",
                detail: 3,
            },
        });
        expect(received).toEqual([2]);
    });
});
