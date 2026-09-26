import { beforeEach, describe, expect, it, vi } from "vitest";

import {
    clearPianoRollLoading,
    getPianoRollLoading,
    resetPianoRollLoadingForTests,
    setPianoRollLoading,
    subscribePianoRollLoading,
} from "./pianoRollStatusBus";

describe("pianoRollStatusBus", () => {
    beforeEach(() => {
        resetPianoRollLoadingForTests();
    });

    it("starts idle", () => {
        expect(getPianoRollLoading()).toBe(false);
    });

    it("notifies subscribers only when the visible state actually changes", () => {
        const listener = vi.fn();
        const unsubscribe = subscribePianoRollLoading(listener);

        setPianoRollLoading("form-a", true);
        expect(getPianoRollLoading()).toBe(true);
        expect(listener).toHaveBeenCalledTimes(1);

        // 同一发布者重复上报同值：对外状态未变 → 不通知。
        setPianoRollLoading("form-a", true);
        expect(listener).toHaveBeenCalledTimes(1);

        setPianoRollLoading("form-a", false);
        expect(getPianoRollLoading()).toBe(false);
        expect(listener).toHaveBeenCalledTimes(2);

        // 已经空闲时再报 false：同样不通知。
        setPianoRollLoading("form-a", false);
        expect(listener).toHaveBeenCalledTimes(2);

        unsubscribe();
    });

    /**
     * 参数编辑器可以多开：一个实例卸载时上报 false，绝不能抹掉另一个仍在取数的实例。
     *
     * 这正是"按发布者记账"而非共用一个 boolean 的原因。
     */
    it("keeps loading while any other publisher is still loading", () => {
        const listener = vi.fn();
        const unsubscribe = subscribePianoRollLoading(listener);

        setPianoRollLoading("form-a", true);
        setPianoRollLoading("form-b", true);
        expect(getPianoRollLoading()).toBe(true);
        // 第二个发布者只是把状态从 false→true 推了一次，不产生额外通知。
        expect(listener).toHaveBeenCalledTimes(1);

        // 一个实例卸载：另一个还在取数 → 对外仍是加载中，且不通知。
        clearPianoRollLoading("form-a");
        expect(getPianoRollLoading()).toBe(true);
        expect(listener).toHaveBeenCalledTimes(1);

        // 最后一个也结束 → 收起。
        clearPianoRollLoading("form-b");
        expect(getPianoRollLoading()).toBe(false);
        expect(listener).toHaveBeenCalledTimes(2);

        unsubscribe();
    });

    /** 注销必须清理记账：否则最后一次状态会永远粘住，状态栏再也不会收起。 */
    it("clearing a loading publisher releases the state", () => {
        setPianoRollLoading("form-a", true);
        expect(getPianoRollLoading()).toBe(true);

        clearPianoRollLoading("form-a");
        expect(getPianoRollLoading()).toBe(false);

        // 重复注销是无害的（幂等）。
        clearPianoRollLoading("form-a");
        expect(getPianoRollLoading()).toBe(false);
    });

    it("unsubscribe stops notifications", () => {
        const listener = vi.fn();
        const unsubscribe = subscribePianoRollLoading(listener);
        unsubscribe();

        setPianoRollLoading("form-a", true);
        expect(listener).not.toHaveBeenCalled();
        expect(getPianoRollLoading()).toBe(true);
    });

    /** 订阅者抛异常不得影响其他订阅者（与 pianoRollSelectionBus 同策略）。 */
    it("isolates a throwing subscriber", () => {
        const healthy = vi.fn();
        const unsubscribeBad = subscribePianoRollLoading(() => {
            throw new Error("boom");
        });
        const unsubscribeGood = subscribePianoRollLoading(healthy);

        expect(() => setPianoRollLoading("form-a", true)).not.toThrow();
        expect(healthy).toHaveBeenCalledTimes(1);

        unsubscribeBad();
        unsubscribeGood();
    });
});
