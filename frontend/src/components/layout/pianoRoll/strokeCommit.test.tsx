// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

const { setParamFrames, restoreParamFrames } = vi.hoisted(() => ({
    setParamFrames: vi.fn(async (..._args: unknown[]) => ({ ok: true })),
    restoreParamFrames: vi.fn(async (..._args: unknown[]) => ({ ok: true })),
}));

vi.mock("../../../services/api", () => ({
    paramsApi: {
        setParamFrames: (...args: unknown[]) => setParamFrames(...args),
        restoreParamFrames: (...args: unknown[]) => restoreParamFrames(...args),
    },
}));

import { projectLiveEdit } from "./strokeBuffer";
import { useLiveParamEditing } from "./useLiveParamEditing";
import type { ParamViewSegment } from "./types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/**
 * 笔画提交的边界不变量。
 *
 * ## 为什么必须有这份测试
 *
 * 提交是"预览 → 后端"的唯一通道，而它此前按**窗口下标**切片：
 *
 * ~~~ts
 * const i = f - pvStart;
 * out[f - minF] = i >= 0 && i < pvEdit.length ? pvEdit[i] : 0;   // ← 窗口外写 0
 * const present = new Array(len).fill(true);                      // ← 且声明"都改过"
 * ~~~
 *
 * 于是笔画一旦越过取数窗口（边缘滚屏画长线时必然发生），**窗口外的那段被写成 0**，
 * 而用户看到的是"画到一半线就没了"。三条不变量现在由缓冲保证，这里把它们钉住：
 *
 * 1. 写入的帧集合 == 真正画过的帧集合，**不含任何未画帧**；
 * 2. 单次 IPC 不超过 `MAX_FRAMES_PER_WRITE`，段间只有一个撤销检查点；
 * 3. 提交成功但快照未到时，覆盖层作为占位保留；快照追平才撤下。
 */

const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    while (mounted.length > 0) await mounted.pop()?.();
    vi.clearAllMocks();
});

function view(key: string, startFrame: number, edit: number[]): ParamViewSegment {
    return {
        key,
        framePeriodMs: 5,
        startFrame,
        stride: 1,
        referenceKind: "source_curve",
        orig: edit.slice(),
        edit,
    };
}

async function harness(initial: ParamViewSegment) {
    let live!: ReturnType<typeof useLiveParamEditing>;
    function Harness({ paramView }: { paramView: ParamViewSegment }) {
        live = useLiveParamEditing({
            rootTrackId: "track",
            editParam: "volume",
            pitchEnabled: true,
            paramView,
            bumpRefreshToken: () => {},
            invalidate: () => {},
        });
        return null;
    }
    const element = document.createElement("div");
    document.body.append(element);
    const root = createRoot(element);
    mounted.push(async () => {
        await act(async () => root.unmount());
        element.remove();
    });
    const render = (paramView: ParamViewSegment) =>
        act(async () => {
            root.render(<Harness paramView={paramView} />);
        });
    await render(initial);
    return { getLive: () => live, render };
}

/** 把所有 setParamFrames 调用的 (帧 → 值) 合并成一张表。 */
function writtenFrames(): Map<number, number> {
    const written = new Map<number, number>();
    for (const call of setParamFrames.mock.calls) {
        const startFrame = call[2] as number;
        const values = call[3] as number[];
        for (let i = 0; i < values.length; i += 1) written.set(startFrame + i, values[i]);
    }
    return written;
}

describe("commitStroke：跨窗口的笔画", () => {
    it("commits every drawn frame across a window swap and never writes a zero", async () => {
        const first = view("v2|track|volume|0|10|1", 0, new Array<number>(10).fill(0.5));
        const second = view("v2|track|volume|100|10|1", 100, new Array<number>(10).fill(0.5));
        const h = await harness(first);

        // 音量值域是 0..2（±6 dB）：用域内值，否则钳制会把断言变成"钳制测过了"。
        const drawnFirst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        const drawnSecond = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0];
        await act(async () => {
            h.getLive().applyDenseToLiveEdit(first, 0, drawnFirst, 0, 9, "draw");
        });
        // 取数回包换掉了窗口（正是旧实现里"轨迹消失 / 提交写 0"的触发条件）。
        await h.render(second);
        await act(async () => {
            h.getLive().applyDenseToLiveEdit(second, 100, drawnSecond, 100, 109, "draw");
        });
        await act(async () => {
            await h.getLive().commitStroke(
                [
                    { frame: 0, value: 0.1 },
                    { frame: 109, value: 2 },
                ],
                "draw",
            );
        });

        const written = writtenFrames();
        expect(Array.from(written.keys()).sort((a, b) => a - b)).toEqual([
            ...Array.from({ length: 10 }, (_, i) => i),
            ...Array.from({ length: 10 }, (_, i) => 100 + i),
        ]);
        // 画过的值逐帧到位 —— 窗口外那段既没丢，也没被写成 0。
        expect(Array.from({ length: 10 }, (_, i) => written.get(i))).toEqual(drawnFirst);
        expect(Array.from({ length: 10 }, (_, i) => written.get(100 + i))).toEqual(drawnSecond);
        expect(Array.from(written.values())).not.toContain(0);
    });

    it("writes nothing outside the drawn frames", async () => {
        const pv = view("v2|track|volume|0|8|1", 0, new Array<number>(8).fill(0.5));
        const h = await harness(pv);

        await act(async () => {
            // 只画中间 3 帧；两侧的帧必须保持"没被改过"。
            h.getLive().applyDenseToLiveEdit(pv, 2, [0.7, 0.7, 0.7], 2, 4, "draw");
        });
        await act(async () => {
            await h.getLive().commitStroke(
                [
                    { frame: 2, value: 0.7 },
                    { frame: 4, value: 0.7 },
                ],
                "draw",
            );
        });

        const written = writtenFrames();
        expect(Array.from(written.keys()).sort((a, b) => a - b)).toEqual([2, 3, 4]);
        // present 位图必须与 values 等长且全 true（显式位图，不让后端猜）。
        for (const call of setParamFrames.mock.calls) {
            expect((call[5] as boolean[]).every(Boolean)).toBe(true);
            expect((call[5] as boolean[]).length).toBe((call[3] as number[]).length);
        }
    });
});

describe("commitStroke：分段与撤销", () => {
    it("splits a long stroke into bounded writes with a single checkpoint", async () => {
        const length = 70000;
        const pv = view(`v2|track|volume|0|${length}|1`, 0, new Array<number>(length).fill(0.5));
        const h = await harness(pv);
        // 域内值：1.00 .. 1.99 循环（音量的写入值域是 0..2）。
        const dense = new Array<number>(length).fill(0).map((_, index) => 1 + (index % 100) / 100);

        await act(async () => {
            h.getLive().applyDenseToLiveEdit(pv, 0, dense, 0, length - 1, "draw");
        });
        await act(async () => {
            await h.getLive().commitStroke(
                [
                    { frame: 0, value: 1 },
                    { frame: length - 1, value: 1.99 },
                ],
                "draw",
            );
        });

        expect(setParamFrames).toHaveBeenCalledTimes(2);
        const sizes = setParamFrames.mock.calls.map((call) => (call[3] as number[]).length);
        expect(sizes).toEqual([65536, length - 65536]);
        // 只有第一段开检查点：一次 Ctrl+Z 回到画之前，而不是按段回退。
        expect(setParamFrames.mock.calls.map((call) => call[4])).toEqual([true, false]);
        expect(writtenFrames().size).toBe(length);
    });
});

describe("commitStroke：还原笔", () => {
    /** orig ≠ edit：用户已经画过这一段，右键要把它还原回原始值。 */
    function drawnView(): ParamViewSegment {
        return {
            key: "v2|track|volume|0|10|1",
            framePeriodMs: 5,
            startFrame: 0,
            stride: 1,
            referenceKind: "source_curve",
            orig: new Array<number>(10).fill(0.5),
            edit: new Array<number>(10).fill(1.5),
        };
    }

    it("previews the original values and keeps them as the placeholder", async () => {
        const pv = drawnView();
        const h = await harness(pv);

        await act(async () => {
            h.getLive().applyDenseToLiveEdit(pv, 5, null, 5, 9, "restore");
        });

        // 拖拽中：预览必须是**原始值**（0.5），而不是已提交值（1.5）—— 后者正是
        // "右键拖拽完全看不到预览"的根源（erase 让这些帧回到基底）。
        const live = h.getLive().liveEditOverrideRef.current;
        expect(live).not.toBeNull();
        const projected = live === null ? [] : Array.from(projectLiveEdit(live, pv));
        expect(projected).toEqual([1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5]);

        await act(async () => {
            await h.getLive().commitStroke(
                [
                    { frame: 5, value: 0.5 },
                    { frame: 9, value: 0.5 },
                ],
                "restore",
            );
        });

        expect(restoreParamFrames).toHaveBeenCalledWith("track", "volume", 5, 5, true);
        expect(setParamFrames).not.toHaveBeenCalled();
        // 松手后占位保留 orig（不是空缓冲）：空占位会让线先跳回 1.5，再等一次 IPC。
        expect(h.getLive().liveEditOverrideRef.current?.buffer.frameCount).toBe(5);
        expect(h.getLive().liveEditOverrideRef.current?.committed).toBe(true);

        // 快照带回还原后的值 ⇒ 占位无损退场。
        const caughtUp: ParamViewSegment = { ...pv, edit: pv.orig.slice() };
        expect(h.getLive().clearCommittedLiveEditOverrideIfCaughtUp(caughtUp)).toBe(true);
    });

    it("fills frames the gesture skipped so the placeholder covers the committed range", async () => {
        const pv = drawnView();
        const h = await harness(pv);

        // 手势只碰了第 5 帧（快速拖拽丢事件），提交范围却是 5..9。
        await act(async () => {
            h.getLive().applyDenseToLiveEdit(pv, 5, null, 5, 5, "restore");
        });
        await act(async () => {
            await h.getLive().commitStroke(
                [
                    { frame: 5, value: 0.5 },
                    { frame: 9, value: 0.5 },
                ],
                "restore",
            );
        });

        expect(h.getLive().liveEditOverrideRef.current?.buffer.frameCount).toBe(5);
    });
});

describe("提交后的占位覆盖层", () => {
    it("stays until the snapshot carries the committed values", async () => {
        const pv = view("v2|track|volume|0|4|1", 0, [0.5, 0.5, 0.5, 0.5]);
        const h = await harness(pv);

        await act(async () => {
            h.getLive().applyDenseToLiveEdit(pv, 0, [1.1, 1.2, 1.3, 1.4], 0, 3, "draw");
        });
        await act(async () => {
            await h.getLive().commitStroke(
                [
                    { frame: 0, value: 1.1 },
                    { frame: 3, value: 1.4 },
                ],
                "draw",
            );
        });

        // 提交成功：覆盖层保留为占位（否则波形会先退回旧快照再跳回来）。
        expect(h.getLive().liveEditOverrideRef.current?.committed).toBe(true);
        // 快照还是旧值 → 不能撤。
        expect(h.getLive().clearCommittedLiveEditOverrideIfCaughtUp(pv)).toBe(false);
        // 快照带上提交值 → 撤下，且是无损的。
        const caughtUp = { ...pv, edit: [1.1, 1.2, 1.3, 1.4] };
        expect(h.getLive().clearCommittedLiveEditOverrideIfCaughtUp(caughtUp)).toBe(true);
        expect(h.getLive().liveEditOverrideRef.current).toBeNull();
    });

    it("starts a fresh buffer for the next stroke", async () => {
        const pv = view("v2|track|volume|0|4|1", 0, [0.5, 0.5, 0.5, 0.5]);
        const h = await harness(pv);

        await act(async () => {
            h.getLive().applyDenseToLiveEdit(pv, 0, [1.1, 1.2, 1.3, 1.4], 0, 3, "draw");
        });
        await act(async () => {
            await h.getLive().commitStroke(
                [
                    { frame: 0, value: 1.1 },
                    { frame: 3, value: 1.4 },
                ],
                "draw",
            );
        });
        expect(h.getLive().liveEditOverrideRef.current?.buffer.frameCount).toBe(4);

        // 下一笔只画 1 帧：提交只能包含这 1 帧，不能把上一笔的 4 帧一起再写一遍。
        setParamFrames.mockClear();
        await act(async () => {
            h.getLive().applyDenseToLiveEdit(pv, 2, [1.8], 2, 2, "draw");
        });
        await act(async () => {
            await h.getLive().commitStroke([{ frame: 2, value: 1.8 }], "draw");
        });

        expect(Array.from(writtenFrames().keys())).toEqual([2]);
        expect(writtenFrames().get(2)).toBe(1.8);
    });
});
