import { describe, expect, it, vi } from "vitest";
import { createSyllableAnalysisQueue } from "./syllableAnalysisQueue";

function deferred() {
    let resolve!: (value: boolean) => void;
    const promise = new Promise<boolean>((done) => {
        resolve = done;
    });
    return { promise, resolve };
}

describe("automatic block analysis backpressure", () => {
    it("submits a large clip batch one at a time and never fills the native slow lane", async () => {
        const clips = Array.from({ length: 100 }, (_, id) => ({
            id: String(id),
            name: `clip ${id}`,
        }));
        const gates = clips.map(() => deferred());
        const analyze = vi.fn((clip: (typeof clips)[number]) => gates[Number(clip.id)].promise);
        const running = createSyllableAnalysisQueue().runBatch({
            clips,
            analyze,
            shouldStart: () => true,
            onStart: () => {},
            onSkipped: () => {},
        });
        await Promise.resolve();
        expect(analyze).toHaveBeenCalledTimes(1);
        for (let id = 0; id < clips.length; id += 1) {
            expect(analyze).toHaveBeenCalledTimes(id + 1);
            gates[id].resolve(true);
            await Promise.resolve();
        }
        expect(await running).toEqual(clips.map((clip) => ({ clipId: clip.id, analyzed: true })));
    });

    it("shares the in-flight limit across batches created by different renders", async () => {
        const queue = createSyllableAnalysisQueue();
        const gate = deferred();
        const first = vi.fn(() => gate.promise);
        const second = vi.fn(async () => true);
        const common = { shouldStart: () => true, onStart: () => {}, onSkipped: () => {} };
        const a = queue.runBatch({ ...common, clips: [{ id: "a", name: "a" }], analyze: first });
        const b = queue.runBatch({ ...common, clips: [{ id: "b", name: "b" }], analyze: second });
        await Promise.resolve();
        expect(first).toHaveBeenCalledOnce();
        expect(second).not.toHaveBeenCalled();
        gate.resolve(true);
        await Promise.all([a, b]);
        expect(second).toHaveBeenCalledOnce();
    });

    it("finishes the submitted clip but skips queued clips after switching scope", async () => {
        const gate = deferred();
        let active = true;
        const analyze = vi.fn(() => gate.promise);
        const onSkipped = vi.fn();
        const running = createSyllableAnalysisQueue().runBatch({
            clips: [
                { id: "a", name: "a" },
                { id: "b", name: "b" },
            ],
            analyze,
            shouldStart: () => active,
            onStart: () => {},
            onSkipped,
        });
        await Promise.resolve();
        active = false;
        gate.resolve(true);
        expect(await running).toEqual([{ clipId: "a", analyzed: true }]);
        expect(analyze).toHaveBeenCalledOnce();
        expect(onSkipped).toHaveBeenCalledWith({ id: "b", name: "b" });
    });

    it("keeps processing the rest of the batch after one analysis fails", async () => {
        const analyze = vi
            .fn()
            .mockRejectedValueOnce(new Error("unavailable"))
            .mockResolvedValue(true);
        expect(
            await createSyllableAnalysisQueue().runBatch({
                clips: [
                    { id: "a", name: "a" },
                    { id: "b", name: "b" },
                ],
                analyze,
                shouldStart: () => true,
                onStart: () => {},
                onSkipped: () => {},
            }),
        ).toEqual([
            { clipId: "a", analyzed: false },
            { clipId: "b", analyzed: true },
        ]);
    });
});
