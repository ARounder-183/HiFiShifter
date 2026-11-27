import { describe, expect, it, vi } from "vitest";

import { GainAuditionController, type GainPreviewTarget } from "./gainAudition";

function harness() {
    const scheduled: Array<() => void> = [];
    const sent: Array<{ gestureId: string; seq: number; targets: readonly GainPreviewTarget[] }> =
        [];
    const cleared: string[] = [];
    const send = vi.fn(async (payload: (typeof sent)[number]) => {
        sent.push(payload);
    });
    const clear = vi.fn(async (gestureId: string) => {
        cleared.push(gestureId);
    });
    const controller = new GainAuditionController(
        send,
        clear,
        (callback) => {
            scheduled.push(callback);
            return scheduled.length;
        },
        () => undefined,
    );
    return { controller, scheduled, sent, cleared, send, clear };
}

const clip = (gain: number): GainPreviewTarget => ({ kind: "clip", clipId: "c", itemGain: gain });

describe("GainAuditionController", () => {
    it("同一帧内多次 preview 只发最后一次", async () => {
        const h = harness();
        h.controller.preview([clip(0.5)]);
        h.controller.preview([clip(0.7)]);
        expect(h.sent).toHaveLength(0);
        h.scheduled.shift()?.();
        await Promise.resolve();
        await Promise.resolve();
        expect(h.sent).toHaveLength(1);
        expect(h.sent[0]?.targets[0]).toMatchObject({ itemGain: 0.7 });
        expect(h.sent[0]?.seq).toBe(1);
    });

    it("在途发送期间只保留最新 pending，完成后补发", async () => {
        const h = harness();
        let release!: () => void;
        h.send.mockImplementationOnce(async (payload) => {
            h.sent.push(payload);
            await new Promise<void>((resolve) => {
                release = resolve;
            });
        });
        h.controller.preview([clip(0.2)]);
        h.scheduled.shift()?.();
        await Promise.resolve();
        h.controller.preview([clip(0.3)]);
        h.controller.preview([clip(0.4)]);
        expect(h.sent).toHaveLength(1);
        release();
        await new Promise((resolve) => setTimeout(resolve, 0));
        expect(h.sent).toHaveLength(2);
        expect(h.sent[1]?.targets[0]).toMatchObject({ itemGain: 0.4 });
        expect(h.sent[1]?.seq).toBe(2);
    });

    it("commit 先 flush 最终值，再等权威提交，最后 clear 同名手势", async () => {
        const h = harness();
        h.controller.preview([clip(0.6)]);
        const commit = Promise.resolve("ok");
        const done = h.controller.commit(commit);
        await done;
        expect(h.sent.at(-1)?.targets[0]).toMatchObject({ itemGain: 0.6 });
        expect(h.cleared).toEqual([h.sent.at(-1)?.gestureId]);
        expect(h.controller.gestureId).toBeNull();
    });

    it("cancel 丢弃 pending 并 clear；新手势的 id 不同", async () => {
        const h = harness();
        h.controller.preview([clip(0.1)]);
        const first = h.controller.gestureId;
        h.controller.cancel();
        await Promise.resolve();
        expect(h.cleared).toEqual([first]);
        h.controller.preview([clip(0.9)]);
        expect(h.controller.gestureId).not.toBe(first);
    });
});
