/**
 * 拖拽期间响度时域映射的契约回归。
 *
 * 【为什么单测它】这是"拖拽中波形错位"修复的**唯一数学落点**。三个失败模式都
 * 只在真实拖拽里才看得见，且都表现为"波形看起来怪"而不是报错：
 *
 * - 映射算错 → 波形在拖拽中整段错位（本 bug 本身）；
 * - 该恒等的地方不恒等 → 没动的区域也跟着变形；
 * - 锁定参数线的开关没接对 → 用户曲线被凭空搬移（后端根本不会做这一步）。
 *
 * 因此这里逐条钉住：平移、裁切/拉伸的重采样比、pad 语义、增益比、以及
 * "锁定参数线关闭时曲线恒等"。
 */
import { describe, expect, it } from "vitest";

import {
    createLoudnessGeometryWarp,
    diffClipGeometryMappings,
    WARP_CURVE_PAD,
    WARP_SEGMENT_IDENTITY,
    type LoudnessRangeMapping,
    type WarpClipGeometry,
    type WarpClipOrigin,
} from "./loudnessGeometryWarp";

/** 帧周期：与工程默认一致（5 ms）。 */
const FP = 5;

function origin(id: string, startSec: number, lengthSec: number, gain = 1): WarpClipOrigin {
    return { clipId: id, startSec, lengthSec, gain };
}

function clip(id: string, startSec: number, lengthSec: number, gain = 1): WarpClipGeometry {
    return { id, startSec, lengthSec, gain };
}

/** 一段映射（默认增益不变）。 */
function mapping(
    oldStartSec: number,
    oldLengthSec: number,
    newStartSec: number,
    newLengthSec: number,
    gainScale = 1,
): LoudnessRangeMapping {
    return { oldStartSec, oldLengthSec, newStartSec, newLengthSec, gainScale };
}

describe("diffClipGeometryMappings", () => {
    it("起点变化（移动）产出一条映射", () => {
        const result = diffClipGeometryMappings([origin("c1", 0, 2)], [clip("c1", 3, 2)]);
        expect(result).toEqual([mapping(0, 2, 3, 2)]);
    });

    it("长度变化（裁切 / 拉伸）产出一条映射", () => {
        const result = diffClipGeometryMappings([origin("c1", 1, 4)], [clip("c1", 1, 2)]);
        expect(result).toEqual([mapping(1, 4, 1, 2)]);
    });

    it("几何未变则不产出映射", () => {
        expect(diffClipGeometryMappings([origin("c1", 2, 3)], [clip("c1", 2, 3)])).toEqual([]);
    });

    it("增益变化产出映射，且带增益比", () => {
        const result = diffClipGeometryMappings([origin("c1", 2, 3, 0.5)], [clip("c1", 2, 3, 1)]);
        expect(result).toEqual([mapping(2, 3, 2, 3, 2)]);
    });

    it("增益未变时增益比为 1（不引入浮点噪声）", () => {
        const result = diffClipGeometryMappings([origin("c1", 0, 1, 1)], [clip("c1", 1, 1, 1)]);
        expect(result[0]?.gainScale).toBe(1);
    });

    it("手势期间新增 / 删除的 clip 不产出映射（无旧位置或已消失）", () => {
        const result = diffClipGeometryMappings(
            [origin("gone", 0, 1), origin("kept", 0, 1)],
            [clip("kept", 1, 1), clip("fresh", 0, 1)],
        );
        expect(result).toEqual([mapping(0, 1, 1, 1)]);
    });

    it("多个 clip 各自产出映射（波纹跟随 / 编组联动同样被覆盖）", () => {
        const result = diffClipGeometryMappings(
            [origin("a", 0, 1), origin("b", 2, 1), origin("c", 4, 1)],
            [clip("a", 1, 1), clip("b", 3, 1), clip("c", 4, 1)],
        );
        expect(result).toEqual([mapping(0, 1, 1, 1), mapping(2, 1, 3, 1)]);
    });
});

describe("createLoudnessGeometryWarp", () => {
    it("没有映射 / 非法帧周期时返回 null（调用方完全跳过该路径）", () => {
        expect(
            createLoudnessGeometryWarp({ mappings: [], lockParamLines: true, framePeriodMs: FP }),
        ).toBeNull();
        expect(
            createLoudnessGeometryWarp({
                mappings: [mapping(0, 1, 1, 1)],
                lockParamLines: true,
                framePeriodMs: 0,
            }),
        ).toBeNull();
    });

    it("零长度的新范围被丢弃（搬过去也覆盖不到任何帧）", () => {
        expect(
            createLoudnessGeometryWarp({
                mappings: [mapping(0, 1, 5, 0)],
                lockParamLines: true,
                framePeriodMs: FP,
            }),
        ).toBeNull();
    });

    describe("移动（纯平移，长度不变）", () => {
        // 2 s → 4 s 起点（帧 0..400 → 帧 400..800）。
        const warp = createLoudnessGeometryWarp({
            mappings: [mapping(0, 2, 2, 2)],
            lockParamLines: false,
            framePeriodMs: FP,
        })!;

        it("新范围内的帧映射回旧位置（逐帧精确平移）", () => {
            for (let k = 0; k < 400; k += 1) {
                expect(warp.baselineFrame(400 + k)).toBe(k);
            }
        });

        it("新范围外的帧保持恒等（未受影响区域不变形）", () => {
            expect(warp.baselineFrame(0)).toBe(0);
            expect(warp.baselineFrame(399)).toBe(399);
            expect(warp.baselineFrame(800)).toBe(800);
            expect(warp.baselineFrame(1234)).toBe(1234);
        });

        it("映射段编号只在新区间内出现", () => {
            expect(warp.baselineSegment(399)).toBe(WARP_SEGMENT_IDENTITY);
            expect(warp.baselineSegment(400)).toBe(0);
            expect(warp.baselineSegment(799)).toBe(0);
            expect(warp.baselineSegment(800)).toBe(WARP_SEGMENT_IDENTITY);
        });

        it("锁定参数线关闭时用户曲线恒等（后端不会搬移它们）", () => {
            for (const f of [0, 100, 400, 600, 900]) {
                expect(warp.curveFrame(f)).toBe(f);
                expect(warp.curveSegment(f)).toBe(WARP_SEGMENT_IDENTITY);
            }
        });
    });

    describe("移动 + 锁定参数线开启", () => {
        const warp = createLoudnessGeometryWarp({
            mappings: [mapping(0, 2, 2, 2)],
            lockParamLines: true,
            framePeriodMs: FP,
        })!;

        it("新范围内曲线跟着搬（与基线同一映射）", () => {
            for (let k = 0; k < 400; k += 1) {
                expect(warp.curveFrame(400 + k)).toBe(k);
            }
            expect(warp.curveSegment(400)).toBe(0);
        });

        it("旧范围中未被新范围覆盖的帧恢复为 pad（volume→1、dyn→沿用原声）", () => {
            expect(warp.curveFrame(0)).toBe(WARP_CURVE_PAD);
            expect(warp.curveFrame(399)).toBe(WARP_CURVE_PAD);
            expect(warp.curveSegment(200)).toBe(WARP_CURVE_PAD);
        });

        it("既不在旧范围也不在新范围的帧保持恒等", () => {
            expect(warp.curveFrame(800)).toBe(800);
            expect(warp.curveFrame(50 + 800)).toBe(850);
            expect(warp.curveSegment(900)).toBe(WARP_SEGMENT_IDENTITY);
        });
    });

    describe("裁切 / 拉伸（长度变化 → 仿射重采样）", () => {
        it("重采样比与后端 resample_curve 的 (len−1) 公式逐值一致", () => {
            // 4 s → 2 s：帧 0..800 → 帧 0..400。
            const warp = createLoudnessGeometryWarp({
                mappings: [mapping(0, 4, 0, 2)],
                lockParamLines: false,
                framePeriodMs: FP,
            })!;
            const oldMaxIdx = 800 - 1;
            const newMaxIdx = 400 - 1;
            for (const k of [0, 1, 50, 100, 399]) {
                expect(warp.baselineFrame(k)).toBeCloseTo(k * (oldMaxIdx / newMaxIdx), 10);
            }
            // 端点必须精确落在旧范围两端（不因浮点漂移出界）。
            expect(warp.baselineFrame(0)).toBeCloseTo(0, 10);
            expect(warp.baselineFrame(399)).toBeCloseTo(799, 10);
        });

        it("拉伸后旧范围的尾部未被覆盖 → pad", () => {
            const warp = createLoudnessGeometryWarp({
                mappings: [mapping(0, 4, 0, 2)],
                lockParamLines: true,
                framePeriodMs: FP,
            })!;
            expect(warp.curveFrame(399)).toBeCloseTo(799, 10);
            expect(warp.curveFrame(400)).toBe(WARP_CURVE_PAD);
            expect(warp.curveFrame(799)).toBe(WARP_CURVE_PAD);
        });

        it("新范围比旧范围长（拉伸变慢）时端点同样精确", () => {
            // 2 s → 4 s：帧 0..400 → 帧 0..800。
            const warp = createLoudnessGeometryWarp({
                mappings: [mapping(0, 2, 0, 4)],
                lockParamLines: false,
                framePeriodMs: FP,
            })!;
            expect(warp.baselineFrame(0)).toBeCloseTo(0, 10);
            expect(warp.baselineFrame(799)).toBeCloseTo(399, 10);
        });
    });

    describe("增益（基线含静态 clip 增益）", () => {
        const warp = createLoudnessGeometryWarp({
            mappings: [mapping(0, 2, 0, 2, 2)],
            lockParamLines: true,
            framePeriodMs: FP,
        })!;

        it("基线按增益比缩放，且时域不变", () => {
            expect(warp.baselineScale(100)).toBe(2);
            expect(warp.baselineFrame(100)).toBe(100);
        });

        it("范围外增益比为 1", () => {
            expect(warp.baselineScale(500)).toBe(1);
        });

        it("纯增益变化不搬移用户曲线（后端不会调用时域映射）", () => {
            expect(warp.curveFrame(100)).toBe(100);
            expect(warp.curveSegment(100)).toBe(WARP_SEGMENT_IDENTITY);
        });
    });

    describe("多段与重叠", () => {
        it("多段各自独立映射（编组联动 / 波纹跟随）", () => {
            const warp = createLoudnessGeometryWarp({
                mappings: [mapping(0, 1, 2, 1), mapping(4, 1, 6, 1)],
                lockParamLines: false,
                framePeriodMs: FP,
            })!;
            expect(warp.baselineFrame(400)).toBe(0);
            expect(warp.baselineFrame(1200)).toBe(800);
            expect(warp.baselineFrame(1000)).toBe(1000);
        });

        it("新范围重叠时后者胜（复刻后端逐段写入的覆盖顺序）", () => {
            const warp = createLoudnessGeometryWarp({
                mappings: [mapping(0, 2, 2, 2), mapping(4, 2, 2, 2)],
                lockParamLines: false,
                framePeriodMs: FP,
            })!;
            // 帧 400..800 被两段同时覆盖，后一段（旧起点 4 s = 帧 800）胜出。
            expect(warp.baselineFrame(400)).toBe(800);
            expect(warp.baselineFrame(600)).toBe(1000);
        });
    });
});
