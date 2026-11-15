import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
    analysisWindowStatus,
    audibleSourceWindow,
    cutFlagsForIntersection,
    inverseProject,
    projectionForClip,
    positiveIntervalToFrames,
    projectTakeSpan,
    sourceAtProjectedFrame,
    windowIntersection,
    type ParamBlockSourceClip,
} from "./paramBlockProjection";

interface GoldenCase {
    readonly name: string;
    readonly clip: ParamBlockSourceClip;
    readonly span: {
        readonly sourceStartSec: number;
        readonly sourceEndSec: number;
        readonly cycle: number;
    };
    readonly expected: ReadonlyArray<{
        readonly cycle: number;
        readonly startFrame: number;
        readonly endFrameExclusive: number;
        readonly cutStart: boolean;
        readonly cutEnd: boolean;
    }>;
}

const fixture = JSON.parse(
    readFileSync(
        new URL("./__fixtures__/paramBlockProjection.golden.json", import.meta.url),
        "utf8",
    ),
) as { readonly framePeriodMs: number; readonly cases: readonly GoldenCase[] };

describe("paramBlockProjection", () => {
    it("matches the shared Rust golden fixture", () => {
        for (const testCase of fixture.cases) {
            const projection = projectionForClip(testCase.clip);
            const actual = projectTakeSpan(projection, testCase.span, fixture.framePeriodMs);
            expect(actual, testCase.name).toEqual(testCase.expected);
        }
    });

    it("shares window intersection and one-frame frame algebra", () => {
        expect(windowIntersection([0.5, 1.001], [1, 3])).toEqual([1, 1.001]);
        expect(windowIntersection([0.5, 1], [1, 3])).toBeNull();
        expect(positiveIntervalToFrames(40, 40, 200)).toEqual([40, 41]);
        expect(positiveIntervalToFrames(40, 42, 200)).toEqual([40, 42]);
        // 一帧的提升不得越过 Clip 可见范围的右缘（与 loop 路径同守卫）。
        expect(positiveIntervalToFrames(200, 200, 200)).toBeNull();
        expect(positiveIntervalToFrames(200, 201, 200)).toEqual([200, 201]);
        expect(cutFlagsForIntersection([0.6, 1.4], [1, 3], false)).toEqual([true, false]);
        expect(cutFlagsForIntersection([1.6, 2.4], [1, 2], true)).toEqual([true, false]);
        expect(analysisWindowStatus({ startSec: 1, endSec: 3 }, [1, 3], [1])).toBe("current");
        expect(analysisWindowStatus({ startSec: 0.5, endSec: 3 }, [1, 3], [1])).toBe("stale");
        expect(analysisWindowStatus({ startSec: 1, endSec: 3 }, [1, 3], [])).toBe("missing");
    });

    it("matches the Rust audible source window rules", () => {
        const forward = projectionForClip({
            id: "c",
            trackId: "root",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0.5,
            sourceEndSec: 8,
            playbackRate: 1,
            clipPlaybackRate: 2,
            reversed: false,
            loopEnabled: false,
        });
        expect(audibleSourceWindow(forward)).toEqual({ startSec: 0.5, endSec: 4.5 });
        const ring = projectionForClip({
            id: "c",
            trackId: "root",
            startSec: 0,
            lengthSec: 4,
            sourceStartSec: 0,
            sourceEndSec: 1,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            loopPeriodSec: 1,
            contentDurationSec: 1,
        });
        expect(audibleSourceWindow(ring)).toEqual({ startSec: 0, endSec: 1 });
        const section = projectionForClip({
            id: "c",
            trackId: "root",
            startSec: 0,
            lengthSec: 4,
            sourceStartSec: 0.25,
            sourceEndSec: 1.25,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            loopPeriodSec: 1,
            contentDurationSec: 4,
        });
        expect(audibleSourceWindow(section)).toEqual({ startSec: 0.25, endSec: 1.25 });
    });

    it("starts loop projection at the requested cycle", () => {
        const loopClip: ParamBlockSourceClip = {
            id: "c",
            trackId: "root",
            startSec: 0,
            lengthSec: 8,
            sourceStartSec: 0.266,
            sourceEndSec: 2.27,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            loopPeriodSec: 2.27,
            contentDurationSec: 2.27,
        };
        const projection = projectionForClip(loopClip);
        const span = { sourceStartSec: 0.046, sourceEndSec: 0.252, cycle: 0 };
        const full = projectTakeSpan(projection, span, 5);
        const fromCycle = projectTakeSpan(projection, span, 5, 2);
        expect(full.length).toBeGreaterThan(fromCycle.length);
        expect(fromCycle.length).toBeGreaterThan(0);
        expect(fromCycle.every((interval) => interval.cycle >= 2)).toBe(true);
        expect(fromCycle[0]?.cycle).toBe(2);
        expect(projectTakeSpan(projection, span, 5, 99)).toEqual([]);
    });

    it("sourceAtProjectedFrame stays inside the member for loop wrap and reverse", () => {
        const loopClip: ParamBlockSourceClip = {
            id: "c",
            trackId: "root",
            startSec: 0,
            lengthSec: 8,
            sourceStartSec: 0.266,
            sourceEndSec: 2.27,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            loopPeriodSec: 2.27,
            contentDurationSec: 2.27,
        };
        const projection = projectionForClip(loopClip);
        const span = { sourceStartSec: 0.046, sourceEndSec: 0.252, cycle: 0 };
        for (const interval of projectTakeSpan(projection, span, 5)) {
            for (let frame = interval.startFrame; frame <= interval.endFrameExclusive; frame += 1) {
                const source = sourceAtProjectedFrame(projection, frame, interval.cycle, 5);
                expect(source).not.toBeNull();
                expect(source!).toBeGreaterThanOrEqual(0.0459);
                expect(source!).toBeLessThanOrEqual(0.2521);
            }
        }
        const reverse: ParamBlockSourceClip = {
            ...loopClip,
            reversed: true,
            sourceStartSec: 0,
            sourceEndSec: 2,
            loopPeriodSec: 2,
            contentDurationSec: 2,
        };
        const reverseProjection = projectionForClip(reverse);
        const reverseSpan = { sourceStartSec: 0.2, sourceEndSec: 0.4, cycle: 0 };
        for (const interval of projectTakeSpan(reverseProjection, reverseSpan, 5)) {
            for (let frame = interval.startFrame; frame <= interval.endFrameExclusive; frame += 1) {
                const source = sourceAtProjectedFrame(reverseProjection, frame, interval.cycle, 5);
                expect(source).not.toBeNull();
                expect(source!).toBeGreaterThanOrEqual(0.1999);
                expect(source!).toBeLessThanOrEqual(0.4001);
            }
        }
    });

    it("inverseProject loop wrap stays inside the member instead of jumping to media end", () => {
        const clip: ParamBlockSourceClip = {
            id: "c",
            trackId: "root",
            startSec: 0,
            lengthSec: 8,
            sourceStartSec: 0.266,
            sourceEndSec: 2.27,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            loopPeriodSec: 2.27,
            contentDurationSec: 2.27,
        };
        const projection = projectionForClip(clip);
        const span = { sourceStartSec: 0.046, sourceEndSec: 0.252, cycle: 0 };
        const interval = projectTakeSpan(projection, span, 5)[0]!;
        for (let frame = interval.startFrame; frame <= interval.endFrameExclusive; frame += 1) {
            const source = inverseProject(projection, frame, interval.cycle, 5);
            expect(source).not.toBeNull();
            expect(source!).toBeGreaterThanOrEqual(0.0459);
            expect(source!).toBeLessThanOrEqual(0.2521);
        }
    });

    it("inverse projection round-trips a forward frame", () => {
        const clip: ParamBlockSourceClip = {
            id: "c",
            trackId: "root",
            startSec: 1,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
        };
        const projection = projectionForClip(clip);
        const interval = projectTakeSpan(
            projection,
            { sourceStartSec: 0.5, sourceEndSec: 1, cycle: 0 },
            5,
        )[0];
        expect(interval.startFrame).toBe(300);
    });
});
