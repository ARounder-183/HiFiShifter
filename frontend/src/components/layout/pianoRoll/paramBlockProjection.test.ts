import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
    inverseProject,
    projectionForClip,
    projectTakeSpan,
    sourceAtProjectedFrame,
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
