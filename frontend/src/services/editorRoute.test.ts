// @vitest-environment jsdom
/**
 * 编辑器路由判据的**唯一**契约：只有后端明确说 `unbound` 才算"连不上"。
 *
 * 【为什么钉死 `waiting-host`】它出现在每一次开窗的头一两秒（宿主的握手还在路上）。
 * 把它当故障，用户会在**正常**会话里被警告一次 —— 而这正是"狼来了"式提示的起点：
 * 真出问题时那条提示已经不再被当真。
 */
import { afterEach, describe, expect, it } from "vitest";

import { editorRouteIsUnbound } from "./editorRoute";

function bootstrap(editorRoute: unknown): void {
    window.__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "view-route",
        editorRoute,
    } as unknown as Window["__HFS_PLUGIN_BOOTSTRAP__"];
}

afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

describe("editorRouteIsUnbound", () => {
    it("只有 unbound 才算连不上", () => {
        for (const state of ["waiting-host", "bound", "fallback"]) {
            bootstrap({ state });
            expect(editorRouteIsUnbound(), state).toBe(false);
        }
        bootstrap({ state: "unbound", reason: "no live plug-in instance" });
        expect(editorRouteIsUnbound()).toBe(true);
    });

    it("旧后端没有这个字段时不提示", () => {
        bootstrap(undefined);
        expect(editorRouteIsUnbound()).toBe(false);
    });
});
