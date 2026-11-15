import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    paramBlockFeedbackStore,
    publishParamBlockFeedback,
    publishParamBlockWriteFailure,
} from "./paramBlockFeedback";

const tf = (key: string) => key;

describe("paramBlockFeedback", () => {
    beforeEach(() => {
        publishParamBlockFeedback(null);
        vi.useFakeTimers();
    });
    afterEach(() => {
        vi.useRealTimers();
        publishParamBlockFeedback(null);
    });

    /*
     * 【单槽去重】连按两次同一条命令失败时会产生两条一模一样的提示，视觉上像
     * 「没反应」。同一文案 + 同一 tone 在 500ms 内只保留一条。
     */
    it("★ 同一文案同一 tone 在窗口内不重复发布", () => {
        const listener = vi.fn();
        const unsubscribe = paramBlockFeedbackStore.subscribe(listener);
        publishParamBlockFeedback({ text: "x", tone: "danger", holdMs: 10_000 });
        const afterFirst = listener.mock.calls.length;
        publishParamBlockFeedback({ text: "x", tone: "danger", holdMs: 10_000 });
        expect(listener.mock.calls.length).toBe(afterFirst);
        unsubscribe();
    });

    it("换文案或换 tone 仍然发布（不同失败要能被看见）", () => {
        const listener = vi.fn();
        const unsubscribe = paramBlockFeedbackStore.subscribe(listener);
        publishParamBlockFeedback({ text: "x", tone: "danger", holdMs: 10_000 });
        const afterFirst = listener.mock.calls.length;
        publishParamBlockFeedback({ text: "y", tone: "danger", holdMs: 10_000 });
        publishParamBlockFeedback({ text: "y", tone: "warning", holdMs: 10_000 });
        expect(listener.mock.calls.length).toBeGreaterThan(afterFirst + 1);
        unsubscribe();
    });

    it("四类失败各自一条文案，silent 不发", () => {
        publishParamBlockWriteFailure({
            failure: "rejected",
            error: { code: "c1", detail: "d1" },
            tf,
        });
        expect(paramBlockFeedbackStore.getSnapshot()).toMatchObject({
            text: "status_param_block_edit_rejected",
            tone: "warning",
        });
        publishParamBlockWriteFailure({
            failure: "conflict",
            error: { code: "c2" },
            tf,
        });
        expect(paramBlockFeedbackStore.getSnapshot()).toMatchObject({
            text: "status_param_block_conflict_resynced",
            tone: "warning",
        });
        publishParamBlockWriteFailure({
            failure: "transport",
            error: { code: "c3" },
            tf,
        });
        expect(paramBlockFeedbackStore.getSnapshot()).toMatchObject({
            text: "status_param_block_write_failed",
            tone: "danger",
        });
        publishParamBlockFeedback(null);
        publishParamBlockWriteFailure({ failure: "silent", error: { code: "c4" }, tf });
        expect(paramBlockFeedbackStore.getSnapshot()).toBeNull();
    });
});
