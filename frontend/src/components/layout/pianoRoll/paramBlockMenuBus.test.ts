import { beforeEach, describe, expect, it, vi } from "vitest";
import {
    activateParamBlockMenuContext,
    getActiveParamBlockMenuPanelId,
    getParamBlockMenuContext,
    publishParamBlockMenuContext,
    resetParamBlockMenuBusForTests,
    subscribeParamBlockMenuContext,
} from "./paramBlockMenuBus";
import type { ParamBlockMenuContext } from "./paramBlockMenuModel";

function context(panelId: string): ParamBlockMenuContext {
    return {
        panelId,
        canSplit: () => true,
        canGroup: true,
        canUngroup: false,
        canDelete: true,
        canRestoreGame: true,
        canRestoreDeleted: true,
        canNormalizePitch: false,
        hasSelection: true,
        isPitch: false,
        isVolume: false,
        isDyn: false,
    };
}

beforeEach(() => resetParamBlockMenuBusForTests());

describe("paramBlockMenuBus", () => {
    it("keeps two panels isolated and only returns the active one", () => {
        publishParamBlockMenuContext("a", context("a"));
        publishParamBlockMenuContext("b", context("b"));
        expect(getParamBlockMenuContext()?.panelId).toBe("a");
        activateParamBlockMenuContext("b");
        expect(getActiveParamBlockMenuPanelId()).toBe("b");
        expect(getParamBlockMenuContext()?.panelId).toBe("b");
    });

    it("unmounting one panel does not clear another", () => {
        publishParamBlockMenuContext("a", context("a"));
        publishParamBlockMenuContext("b", context("b"));
        activateParamBlockMenuContext("a");
        publishParamBlockMenuContext("a", null);
        expect(getActiveParamBlockMenuPanelId()).toBe("b");
        expect(getParamBlockMenuContext()?.panelId).toBe("b");
    });

    it("notifies subscribers on publish and activation", () => {
        const listener = vi.fn();
        const unsubscribe = subscribeParamBlockMenuContext(listener);
        publishParamBlockMenuContext("a", context("a"));
        publishParamBlockMenuContext("b", context("b"));
        activateParamBlockMenuContext("b");
        activateParamBlockMenuContext("a");
        expect(listener).toHaveBeenCalledTimes(4);
        unsubscribe();
        publishParamBlockMenuContext("b", null);
        expect(listener).toHaveBeenCalledTimes(4);
    });
});
