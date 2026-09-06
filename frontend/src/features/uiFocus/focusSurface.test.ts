import { afterEach, describe, expect, it } from "vitest";
import {
    getActiveSurface,
    installFocusSurfaceTracking,
    resetActiveSurfaceForTests,
    resolveSurfaceFromTarget,
    updateActiveSurfaceFrom,
} from "./focusSurface";

/** 构造带 closest 链的最小元素桩（模拟 DOM 中的 target 及其祖先）。
 * 与真实 closest 一致：从自身起向上查找第一个带有该属性的节点。 */
function fakeTarget(chain: (string | null)[]): EventTarget {
    const makeNode = (i: number): {
        getAttribute: (name: string) => string | null;
        closest: (selector: string) => unknown;
    } => ({
        getAttribute: (name) => {
            void name;
            return chain[i];
        },
        closest: () => {
            for (let j = i; j < chain.length; j += 1) {
                if (chain[j] !== null) return makeNode(j);
            }
            return null;
        },
    });
    return makeNode(0) as unknown as EventTarget;
}

afterEach(() => {
    resetActiveSurfaceForTests();
});

describe("focusSurface（活动编辑表面单一事实源）", () => {
    it("pointerdown 落点在表面内时更新归属", () => {
        expect(getActiveSurface()).toBeNull();
        updateActiveSurfaceFrom(fakeTarget([null, "pianoRoll", null]));
        expect(getActiveSurface()).toBe("pianoRoll");
        updateActiveSurfaceFrom(fakeTarget(["timeline"]));
        expect(getActiveSurface()).toBe("timeline");
    });

    it("表面之外（播放条/浮动窗/对话框）不改变归属 —— DAW 编辑上下文保留", () => {
        updateActiveSurfaceFrom(fakeTarget(["timeline"]));
        updateActiveSurfaceFrom(fakeTarget([null, null, null]));
        updateActiveSurfaceFrom(null);
        expect(getActiveSurface()).toBe("timeline");
    });

    it("未知属性值视为表面之外", () => {
        updateActiveSurfaceFrom(fakeTarget(["pianoRoll"]));
        updateActiveSurfaceFrom(fakeTarget(["clipFormant"]));
        expect(getActiveSurface()).toBe("pianoRoll");
    });

    it("嵌套表面就近解析：轨道列覆盖时间轴面板", () => {
        // TimelinePanel(root=timeline) → TrackList(root=trackHeader) → 按钮
        expect(resolveSurfaceFromTarget(fakeTarget([null, "trackHeader", "timeline"]))).toBe(
            "trackHeader",
        );
        // TimelinePanel(root=timeline) → 轨道区
        expect(resolveSurfaceFromTarget(fakeTarget([null, null, "timeline"]))).toBe("timeline");
    });

    it("installFocusSurfaceTracking：pointerdown/focusin 捕获驱动归属", () => {
        const listeners: Array<{
            type: string;
            handler: (e: { target: EventTarget | null }) => void;
            capture?: boolean;
        }> = [];
        const doc = {
            addEventListener: (
                type: string,
                handler: (e: { target: EventTarget | null }) => void,
                capture?: boolean,
            ) => listeners.push({ type, handler, capture }),
            removeEventListener: (type: string) => {
                const i = listeners.findIndex((l) => l.type === type);
                if (i >= 0) listeners.splice(i, 1);
            },
        };
        (globalThis as { document?: unknown }).document = doc;
        try {
            const cleanup = installFocusSurfaceTracking();
            const pointerdown = listeners.find((l) => l.type === "pointerdown");
            const focusin = listeners.find((l) => l.type === "focusin");
            expect(pointerdown?.capture).toBe(true);
            expect(focusin?.capture).toBe(true);

            pointerdown?.handler({ target: fakeTarget(["timeline"]) });
            expect(getActiveSurface()).toBe("timeline");
            // focusin 同样驱动（键盘 Tab 导航场景）
            focusin?.handler({ target: fakeTarget(["pianoRoll"]) });
            expect(getActiveSurface()).toBe("pianoRoll");

            cleanup();
            expect(listeners.find((l) => l.type === "pointerdown")).toBeUndefined();
            expect(listeners.find((l) => l.type === "focusin")).toBeUndefined();
        } finally {
            delete (globalThis as { document?: unknown }).document;
        }
    });
});
