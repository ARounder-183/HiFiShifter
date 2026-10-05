/**
 * ★ 回归：裁切 / 延伸的**源窗口方向**（含倒放）。
 *
 * ## 缺陷形态
 *
 * 用户直觉的规则**与是否倒放无关**：拖**右**边 ⇒ **左**边内容固定；拖**左**边 ⇒
 * **右**边内容固定。而"时间轴左/右端分别播放哪个源位置"在倒放时是**镜像**的
 *（后端 `state.rs:121` `clip_playback_window_sec`：倒放 `win = [se − len·r, se)`，
 * 升序消费后整体翻转 ⇒ 左端播 `se`、右端播 `se − len·r`）。
 *
 * 修复前这里按"正放"写死了 `右缘 ⇒ 动 sourceEndSec`，于是倒放 Clip：
 *
 * - 拖右缘改的是 `sourceEndSec`（那是**左**端）⇒ 左端跟着动，固定成了右端；
 * - 拖左缘改的是 `sourceStartSec`（那是**右**端）⇒ 右端跟着动，固定成了左端。
 *
 * ## 本用例钉住什么
 *
 * 不只比字段，而是**换算回消费窗口**断言"被固定的那一侧源位置逐值不变" ——
 * 这才是用户看到的东西。字段层面的对错可以有两种等价写法，消费窗口只有一种。
 */
import { describe, expect, it } from "vitest";

import { buildWaveformScene, type WaveformSceneClip } from "../../../waveform/sceneBuilder.ts";
import { createTimelineAxis } from "../renderKernel/timelineAxis.ts";
import { resolveTrimSourceWindow } from "./trimSourceWindow";

/**
 * 时间轴某一端播放的源位置（后端 `clip_playback_window_sec` 的镜像）。
 *
 * 正放 `win = [ss, ss + len·r)`，倒放 `win = [se − len·r, se)` 且输出翻转。
 */
function timelineEdgeSourceSec(args: {
    side: "left" | "right";
    reversed: boolean;
    sourceStartSec: number;
    sourceEndSec: number;
    lengthSec: number;
    rate: number;
}): number {
    const span = args.lengthSec * args.rate;
    if (args.side === "left") {
        return args.reversed ? args.sourceEndSec : args.sourceStartSec;
    }
    return args.reversed ? args.sourceEndSec - span : args.sourceStartSec + span;
}

interface Case {
    edge: "left" | "right";
    reversed: boolean;
    deltaSec: number;
    sourceStartSec: number;
    sourceEndSec: number;
    lengthSec: number;
    rate: number;
}

function resolve(args: Case) {
    return resolveTrimSourceWindow({
        edge: args.edge,
        reversed: args.reversed,
        loopEnabled: false,
        mediaDurationSec: 0,
        deltaSec: args.deltaSec,
        rate: args.rate,
        sourceStartSec: args.sourceStartSec,
        sourceEndSec: args.sourceEndSec,
    });
}

/** 拖拽后的新长度（与 `resolveTrimEdge` 同口径）。 */
function nextLength(args: Case): number {
    return args.edge === "right" ? args.lengthSec + args.deltaSec : args.lengthSec - args.deltaSec;
}

describe("resolveTrimSourceWindow：固定端不变（正放）", () => {
    const base: Case = {
        edge: "right",
        reversed: false,
        deltaSec: 1,
        sourceStartSec: 6,
        sourceEndSec: 10,
        lengthSec: 4,
        rate: 1,
    };

    it("拖右缘 ⇒ 左端（sourceStartSec）固定", () => {
        const before = timelineEdgeSourceSec({ ...base, side: "left" });
        const out = resolve(base)!;
        expect(out.sourceStartSec).toBe(6);
        expect(out.sourceEndSec).toBe(11);
        expect(
            timelineEdgeSourceSec({
                ...base,
                ...out,
                lengthSec: nextLength(base),
                side: "left",
            }),
        ).toBeCloseTo(before, 12);
    });

    it("拖左缘 ⇒ 右端（sourceEndSec）固定", () => {
        const c: Case = { ...base, edge: "left" };
        const before = timelineEdgeSourceSec({ ...c, side: "right" });
        const out = resolve(c)!;
        expect(out.sourceStartSec).toBe(7);
        expect(out.sourceEndSec).toBe(10);
        expect(
            timelineEdgeSourceSec({
                ...c,
                ...out,
                lengthSec: nextLength(c),
                side: "right",
            }),
        ).toBeCloseTo(before, 12);
    });
});

describe("★ resolveTrimSourceWindow：固定端不变（倒放）", () => {
    // 倒放、se=10、len=4、r=1 ⇒ win=[6,10]：**左端播 10、右端播 6**。
    const base: Case = {
        edge: "right",
        reversed: true,
        deltaSec: 1,
        sourceStartSec: 6,
        sourceEndSec: 10,
        lengthSec: 4,
        rate: 1,
    };

    it("★ 拖右缘 ⇒ 左端固定：动的是 sourceStartSec，且方向为负", () => {
        const before = timelineEdgeSourceSec({ ...base, side: "left" });
        expect(before).toBe(10); // 左端播 se
        const out = resolve(base)!;
        // 修复前这里会写 sourceEndSec（= 左端）⇒ 左端跟着动。
        expect(out.sourceEndSec).toBe(10);
        expect(out.sourceStartSec).toBe(5); // 6 − 1·1
        expect(
            timelineEdgeSourceSec({
                ...base,
                ...out,
                lengthSec: nextLength(base),
                side: "left",
            }),
        ).toBeCloseTo(10, 12);
        // 右端则按内容消费移动：新 win=[5,10]，右端播 5。
        expect(
            timelineEdgeSourceSec({
                ...base,
                ...out,
                lengthSec: nextLength(base),
                side: "right",
            }),
        ).toBeCloseTo(5, 12);
    });

    it("★ 拖左缘 ⇒ 右端固定：动的是 sourceEndSec，且方向为负", () => {
        const c: Case = { ...base, edge: "left" };
        const before = timelineEdgeSourceSec({ ...c, side: "right" });
        expect(before).toBe(6); // 右端播 se − len·r
        const out = resolve(c)!;
        // 修复前这里会写 sourceStartSec（= 右端）⇒ 右端跟着动。
        expect(out.sourceStartSec).toBe(6);
        expect(out.sourceEndSec).toBe(9); // 10 − 1·1
        expect(
            timelineEdgeSourceSec({
                ...c,
                ...out,
                lengthSec: nextLength(c),
                side: "right",
            }),
        ).toBeCloseTo(6, 12);
    });

    it("缩短（deltaSec 为负）同样保持固定端", () => {
        const c: Case = { ...base, deltaSec: -1 };
        const before = timelineEdgeSourceSec({ ...c, side: "left" });
        const out = resolve(c)!;
        expect(out.sourceStartSec).toBe(7);
        expect(
            timelineEdgeSourceSec({
                ...c,
                ...out,
                lengthSec: nextLength(c),
                side: "left",
            }),
        ).toBeCloseTo(before, 12);
    });
});

describe("★ resolveTrimSourceWindow：Loop（两条边不对称）", () => {
    /** Loop 的**右**端内容：`锚点 ∓ 长度×速率` 再取模。 */
    function loopRightSource(args: {
        reversed: boolean;
        sourceStartSec: number;
        sourceEndSec: number;
        lengthSec: number;
        rate: number;
        mediaDurationSec: number;
    }): number {
        const mod = (v: number, d: number): number => ((v % d) + d) % d;
        const span = args.lengthSec * args.rate;
        return args.reversed
            ? mod(args.sourceEndSec - span, args.mediaDurationSec)
            : mod(args.sourceStartSec + span, args.mediaDurationSec);
    }

    it("右缘 ⇒ null（相位锚点不动，左端内容因此固定）", () => {
        for (const reversed of [false, true]) {
            expect(
                resolveTrimSourceWindow({
                    edge: "right",
                    reversed,
                    loopEnabled: true,
                    mediaDurationSec: 8,
                    deltaSec: 1,
                    rate: 1,
                    sourceStartSec: 0.5,
                    sourceEndSec: 4,
                }),
            ).toBeNull();
        }
    });

    it("★ 左缘必须改锚点（否则右端内容会随长度跑掉）", () => {
        const out = resolveTrimSourceWindow({
            edge: "left",
            reversed: true,
            loopEnabled: true,
            mediaDurationSec: 8,
            deltaSec: 1,
            rate: 1,
            sourceStartSec: 0,
            sourceEndSec: 8,
        });
        expect(out).not.toBeNull();
        // 倒放：左缘对应 sourceEnd，位移取反 ⇒ 8 − 1 = 7。
        expect(out?.sourceEndSec).toBe(7);
        expect(out?.sourceStartSec).toBe(0);
    });

    it("★ Loop 左缘拖拽：右端内容逐值不变（正放 / 倒放 / 两个方向）", () => {
        const D = 8;
        for (const reversed of [false, true]) {
            for (const deltaSec of [-1.5, -0.5, 0.5, 1.5]) {
                for (const rate of [1, 2]) {
                    const base = {
                        reversed,
                        loopEnabled: true,
                        mediaDurationSec: D,
                        sourceStartSec: 0,
                        sourceEndSec: D,
                        lengthSec: 4,
                        rate,
                    };
                    const before = loopRightSource(base);
                    const out = resolveTrimSourceWindow({
                        edge: "left",
                        reversed,
                        loopEnabled: true,
                        mediaDurationSec: D,
                        deltaSec,
                        rate,
                        sourceStartSec: base.sourceStartSec,
                        sourceEndSec: base.sourceEndSec,
                    })!;
                    const after = loopRightSource({
                        ...base,
                        sourceStartSec: out.sourceStartSec,
                        sourceEndSec: out.sourceEndSec,
                        lengthSec: 4 - deltaSec,
                    });
                    expect(after, `reversed=${reversed} δ=${deltaSec} r=${rate}`).toBeCloseTo(
                        before,
                        9,
                    );
                }
            }
        }
    });

    it("★ Loop 左缘：锚点取模环绕到 [0, D)（防多次拖拽无界漂移）", () => {
        const out = resolveTrimSourceWindow({
            edge: "left",
            reversed: true,
            loopEnabled: true,
            mediaDurationSec: 8,
            deltaSec: -3, // 向左延伸 3s ⇒ 倒放锚点 +3 → 11 → 环绕成 3
            rate: 1,
            sourceStartSec: 0,
            sourceEndSec: 8,
        })!;
        expect(out.sourceEndSec).toBe(3);
    });

    it("★ Loop 左缘：正放动 sourceStart、倒放动 sourceEnd（镜像）", () => {
        const fwd = resolveTrimSourceWindow({
            edge: "left",
            reversed: false,
            loopEnabled: true,
            mediaDurationSec: 8,
            deltaSec: 1,
            rate: 1,
            sourceStartSec: 2,
            sourceEndSec: 6,
        })!;
        expect(fwd.sourceStartSec).toBe(3);
        expect(fwd.sourceEndSec).toBe(6);

        const rev = resolveTrimSourceWindow({
            edge: "left",
            reversed: true,
            loopEnabled: true,
            mediaDurationSec: 8,
            deltaSec: 1,
            rate: 1,
            sourceStartSec: 2,
            sourceEndSec: 6,
        })!;
        expect(rev.sourceStartSec).toBe(2);
        expect(rev.sourceEndSec).toBe(5);
    });

    it("Loop 但媒体时长未知：仍改锚点，只是不环绕", () => {
        const out = resolveTrimSourceWindow({
            edge: "left",
            reversed: true,
            loopEnabled: true,
            mediaDurationSec: 0,
            deltaSec: -3,
            rate: 1,
            sourceStartSec: 0,
            sourceEndSec: 8,
        })!;
        expect(out.sourceEndSec).toBe(11);
    });
});

describe("resolveTrimSourceWindow：非法输入", () => {
    it("deltaSec 非有限 ⇒ null（调用方跳过该帧）", () => {
        expect(
            resolveTrimSourceWindow({
                edge: "right",
                reversed: false,
                loopEnabled: false,
                mediaDurationSec: 0,
                deltaSec: Number.NaN,
                rate: 1,
                sourceStartSec: 0,
                sourceEndSec: 1,
            }),
        ).toBeNull();
    });

    it("速率非法 ⇒ 按 1 处理（与后端 pr_valid 同口径）", () => {
        const out = resolveTrimSourceWindow({
            edge: "right",
            reversed: false,
            loopEnabled: false,
            mediaDurationSec: 0,
            deltaSec: 2,
            rate: Number.NaN,
            sourceStartSec: 0,
            sourceEndSec: 4,
        })!;
        expect(out.sourceEndSec).toBe(6);
    });
});

describe("resolveTrimSourceWindow：组合速率（take 速率 ≠ 1）", () => {
    it("源位移按组合速率折算（正放）", () => {
        const out = resolve({
            edge: "right",
            reversed: false,
            deltaSec: 1,
            sourceStartSec: 6,
            sourceEndSec: 10,
            lengthSec: 4,
            rate: 2, // clip 1× × take 2×
        })!;
        expect(out.sourceEndSec).toBe(12);
        expect(out.sourceStartSec).toBe(6);
    });

    it("★ 组合速率 + 倒放：方向仍为负、幅度按组合速率", () => {
        const out = resolve({
            edge: "right",
            reversed: true,
            deltaSec: 1,
            sourceStartSec: 6,
            sourceEndSec: 10,
            lengthSec: 4,
            rate: 2,
        })!;
        expect(out.sourceStartSec).toBe(4); // 6 − 1·2
        expect(out.sourceEndSec).toBe(10);
    });
});

describe("resolveTrimSourceWindow：消费窗口层面的方向不变式", () => {
    it("★ 四种组合 × 两个方向：被拖边的**对侧**源位置逐值不变", () => {
        for (const reversed of [false, true]) {
            for (const edge of ["left", "right"] as const) {
                for (const rate of [1, 0.5, 2]) {
                    for (const deltaSec of [-1.5, -0.25, 0.25, 1.5]) {
                        const c: Case = {
                            edge,
                            reversed,
                            deltaSec,
                            sourceStartSec: 6,
                            sourceEndSec: 10,
                            lengthSec: 4,
                            rate,
                        };
                        const fixedSide = edge === "right" ? "left" : "right";
                        const before = timelineEdgeSourceSec({ ...c, side: fixedSide });
                        const out = resolve(c)!;
                        const after = timelineEdgeSourceSec({
                            reversed,
                            sourceStartSec: out.sourceStartSec,
                            sourceEndSec: out.sourceEndSec,
                            lengthSec: nextLength(c),
                            rate,
                            side: fixedSide,
                        });
                        expect(after).toBeCloseTo(before, 9);
                    }
                }
            }
        }
    });
});

/* ────────────────────────────────────────────────────────────────────────────
 * ★ 契约：裁切换算必须让**生产渲染路径**的固定端保持不变。
 *
 * 上面的用例比对的是本模块自建的消费窗口模型；这一组直接调用波形场景构建器
 * （`buildWaveformScene`，与时间轴绘制同一条路径），读取被拖边缘**对侧**实际
 * 渲染出的源位置。它是本缺陷唯一能真正抓住的层次 —— 模型的等价写法不止一种，
 * 渲染结果只有一种。
 *
 * 【曾经漏掉的】Loop clip 的**左缘**：把 Loop 一律当作"只改长度"会让左缘拖拽
 * 完全不写锚点，右端内容随长度跑掉（实测 4 → 5）。
 * ──────────────────────────────────────────────────────────────────────────── */

function renderedEdgeSourceSec(c: WaveformSceneClip, side: "left" | "right"): number {
    const scene = buildWaveformScene({
        axis: createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 2000 }),
        widthPx: 2000,
        rows: [{ topPx: 0, waveformTopPx: 0, waveformHeightPx: 60, clips: [c] }],
    });
    const segments = scene.segments;
    const seg = side === "left" ? segments[0] : segments[segments.length - 1];
    if (seg === undefined) throw new Error("no segment");
    // 段的投影方向随倒放镜像：正放「左端 → sourceStart / 右端 → sourceEnd」，
    // 倒放反之（`sourceRangeForLocal`）。
    const reversed = c.reversed === true;
    const isLeftProjectedToStart = reversed ? side === "right" : side === "left";
    return isLeftProjectedToStart ? seg.sourceStartSec : seg.sourceEndSec;
}

function trimFixture(over: Partial<WaveformSceneClip>): WaveformSceneClip {
    return {
        id: "c",
        sourcePath: "/audio.wav",
        startSec: 0,
        lengthSec: 4,
        sourceStartSec: 0,
        sourceEndSec: 8,
        durationSec: 8,
        sourceSampleRate: 44100,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        gain: 1,
        muted: false,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 0,
        fadeInDir: 0,
        fadeOutShape: 0,
        fadeOutDir: 0,
        ...over,
    };
}

describe("★ 契约：被拖边缘的对侧在**渲染结果**上保持不变", () => {
    const cases: [string, Partial<WaveformSceneClip>][] = [
        ["正放非 Loop", { reversed: false, loopEnabled: false, sourceStartSec: 2, sourceEndSec: 6 }],
        ["倒放非 Loop", { reversed: true, loopEnabled: false, sourceStartSec: 2, sourceEndSec: 6 }],
        ["正放 Loop", { reversed: false, loopEnabled: true, sourceStartSec: 0, sourceEndSec: 8 }],
        ["倒放 Loop", { reversed: true, loopEnabled: true, sourceStartSec: 0, sourceEndSec: 8 }],
    ];

    for (const [name, over] of cases) {
        for (const edge of ["left", "right"] as const) {
            for (const deltaSec of [-1.5, -0.5, 0.5, 1.5]) {
                it(`${name} · 拖${edge === "left" ? "左" : "右"}缘 δ=${deltaSec} ⇒ 对侧不变`, () => {
                    const base = trimFixture(over);
                    const fixedSide = edge === "right" ? "left" : "right";
                    const before = renderedEdgeSourceSec(base, fixedSide);

                    const out = resolveTrimSourceWindow({
                        edge,
                        reversed: base.reversed === true,
                        loopEnabled: base.loopEnabled === true,
                        mediaDurationSec: base.durationSec ?? 0,
                        deltaSec,
                        rate: base.playbackRate ?? 1,
                        sourceStartSec: base.sourceStartSec ?? 0,
                        sourceEndSec: base.sourceEndSec ?? 0,
                    });
                    const after = renderedEdgeSourceSec(
                        trimFixture({
                            ...over,
                            startSec: edge === "left" ? deltaSec : 0,
                            lengthSec: edge === "left" ? 4 - deltaSec : 4 + deltaSec,
                            ...(out === null
                                ? {}
                                : {
                                      sourceStartSec: out.sourceStartSec,
                                      sourceEndSec: out.sourceEndSec,
                                  }),
                        }),
                        fixedSide,
                    );
                    expect(after).toBeCloseTo(before, 6);
                });
            }
        }
    }
});
