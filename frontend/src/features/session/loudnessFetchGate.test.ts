import { afterEach, describe, expect, it, vi } from "vitest";

import {
    holdLoudnessFetch,
    installLoudnessFetchGateDispatcher,
    isLoudnessFetchHeld,
    releaseLoudnessFetch,
    syncLoudnessFetchGate,
} from "./loudnessFetchGate";

afterEach(() => {
    installLoudnessFetchGateDispatcher(null);
    syncLoudnessFetchGate(false);
});

describe("loudnessFetchGate facade", () => {
    it("hold/release 立即更新同步缓存并派发 interaction action", () => {
        const dispatch = vi.fn();
        installLoudnessFetchGateDispatcher(dispatch);
        holdLoudnessFetch();
        expect(isLoudnessFetchHeld()).toBe(true);
        expect(dispatch).toHaveBeenLastCalledWith(true);
        releaseLoudnessFetch();
        expect(isLoudnessFetchHeld()).toBe(false);
        expect(dispatch).toHaveBeenLastCalledWith(false);
    });

    it("store 订阅器可把远端 action 的值同步回来", () => {
        syncLoudnessFetchGate(true);
        expect(isLoudnessFetchHeld()).toBe(true);
        syncLoudnessFetchGate(false);
        expect(isLoudnessFetchHeld()).toBe(false);
    });
});
