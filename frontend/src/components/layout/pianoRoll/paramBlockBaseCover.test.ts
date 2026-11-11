import { describe, expect, it } from "vitest";

import { narrowBaseCoverToMembers } from "./paramBlockBaseCover";
import type { BaseCover } from "./paramBlockTypes";

const BASE: BaseCover = {
    full: true,
    spans: [
        {
            clipId: "clip",
            takeId: "take",
            cycle: 0,
            sourceStartSec: 10,
            sourceEndSec: 20,
            syllableId: "syl",
            label: "syllable",
        },
    ],
};

describe("narrowBaseCoverToMembers", () => {
    it("keeps a full cover when a member covers the whole span", () => {
        const narrowed = narrowBaseCoverToMembers(BASE, [
            { clipId: "clip", takeId: "take", cycle: 0, sourceStartSec: 10, sourceEndSec: 20 },
        ]);
        expect(narrowed?.full).toBe(true);
        expect(narrowed?.spans[0]?.sourceStartSec).toBe(10);
        expect(narrowed?.spans[0]?.sourceEndSec).toBe(20);
    });

    it("drops full when only part of the span is owned", () => {
        const narrowed = narrowBaseCoverToMembers(BASE, [
            { clipId: "clip", takeId: "take", cycle: 0, sourceStartSec: 10, sourceEndSec: 14 },
        ]);
        expect(narrowed?.full).toBe(false);
        expect(narrowed?.spans).toHaveLength(1);
        expect(narrowed?.spans[0]?.sourceEndSec).toBe(14);
    });

    it("supports adjacent members covering the span without a gap", () => {
        const narrowed = narrowBaseCoverToMembers(BASE, [
            { clipId: "clip", takeId: "take", cycle: 0, sourceStartSec: 10, sourceEndSec: 14 },
            { clipId: "clip", takeId: "take", cycle: 0, sourceStartSec: 14, sourceEndSec: 20 },
        ]);
        expect(narrowed?.full).toBe(true);
    });

    it("does not leak covers across clip, take or cycle", () => {
        expect(
            narrowBaseCoverToMembers(BASE, [
                { clipId: "other", takeId: "take", cycle: 0, sourceStartSec: 10, sourceEndSec: 20 },
            ]),
        ).toBeUndefined();
        expect(
            narrowBaseCoverToMembers(BASE, [
                { clipId: "clip", takeId: "take", cycle: 1, sourceStartSec: 10, sourceEndSec: 20 },
            ]),
        ).toBeUndefined();
    });

    it("returns undefined for empty inputs", () => {
        expect(narrowBaseCoverToMembers(undefined, [])).toBeUndefined();
        expect(narrowBaseCoverToMembers(BASE, [])).toBeUndefined();
        expect(narrowBaseCoverToMembers({ spans: [], full: true }, [])).toBeUndefined();
    });
});
