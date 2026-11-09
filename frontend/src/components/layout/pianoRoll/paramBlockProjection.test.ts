import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
    projectionForClip,
    projectTakeSpan,
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
