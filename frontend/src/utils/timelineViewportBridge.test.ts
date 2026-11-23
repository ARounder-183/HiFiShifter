// @vitest-environment jsdom
import { afterEach, expect, test, vi } from "vitest";

const { emitWindowEvent, state } = vi.hoisted(() => ({
    emitWindowEvent: vi.fn(),
    state: {
        dock: {
            layout: {
                forms: {
                    timeline: {
                        id: "timeline",
                        panelId: "timeline",
                        floating: true,
                        floatMode: "osWindow",
                    },
                },
            },
        },
    },
}));

vi.mock("../app/store", () => ({ store: { getState: () => state } }));
vi.mock("../services/hostCapabilities", () => ({ isPluginMode: () => false }));
vi.mock("../features/dock/detachBridge", () => ({
    isSatelliteWindow: () => false,
    resolveWindowLabel: () => "main",
}));
vi.mock("../features/dock/windowEventBridge", () => ({ emitWindowEvent }));

import {
    hasOsWindowEditorForm,
    installTimelineViewportBridge,
    TIMELINE_VIEWPORT_EVENT,
    TIMELINE_VIEWPORT_REQUEST_EVENT,
} from "./timelineViewportBridge";
import { timelineViewportSync } from "./timelineViewportSync";

afterEach(() => {
    timelineViewportSync.reset();
    emitWindowEvent.mockClear();
});

test("只有 osWindow 的编辑器窗体才算需要跨窗口视口桥", () => {
    expect(hasOsWindowEditorForm(state.dock.layout as never)).toBe(true);
    expect(
        hasOsWindowEditorForm({
            forms: { timeline: { id: "timeline", panelId: "timeline", floating: true } },
        } as never),
    ).toBe(false);
});

test("安装后本地视口变化会广播一次，收到 remote 事件应用但不回声", () => {
    const dispose = installTimelineViewportBridge();
    timelineViewportSync.setViewport({ scrollLeft: 120, pxPerSec: 240 });
    // rAF 合并 + 30Hz 上限：用真实 rAF 等一帧。
    return new Promise<void>((resolve) => {
        requestAnimationFrame(() => {
            expect(emitWindowEvent).toHaveBeenCalledWith(
                "all",
                TIMELINE_VIEWPORT_EVENT,
                expect.objectContaining({ scrollLeft: 120, pxPerSec: 240 }),
            );
            emitWindowEvent.mockClear();
            window.dispatchEvent(
                new CustomEvent(TIMELINE_VIEWPORT_EVENT, {
                    detail: { origin: "satellite-a", scrollLeft: 900, pxPerSec: 300 },
                }),
            );
            expect(timelineViewportSync.get().scrollLeft).toBe(900);
            expect(emitWindowEvent).not.toHaveBeenCalled();
            dispose();
            resolve();
        });
    });
});

test("卫星请求时主窗口若已播种会回发当前视口", () => {
    timelineViewportSync.setViewport({ scrollLeft: 77, pxPerSec: 155 });
    const dispose = installTimelineViewportBridge();
    emitWindowEvent.mockClear();
    window.dispatchEvent(
        new CustomEvent(TIMELINE_VIEWPORT_REQUEST_EVENT, {
            detail: { origin: "satellite-b" },
        }),
    );
    expect(emitWindowEvent).toHaveBeenCalledWith(
        "all",
        TIMELINE_VIEWPORT_EVENT,
        expect.objectContaining({ scrollLeft: 77, pxPerSec: 155 }),
    );
    dispose();
});
