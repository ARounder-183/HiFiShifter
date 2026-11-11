import { describe, expect, it } from "vitest";
import {
    normalizeParamBlockEditSet,
    normalizeParamBlockEditSetWithDiagnostics,
} from "./paramBlockTypes";

describe("normalizeParamBlockEditSet", () => {
    it("always materializes blocks as an array", () => {
        expect(normalizeParamBlockEditSet({ schemaVersion: 2, revision: 3 })).toEqual({
            schemaVersion: 2,
            revision: 3,
            blocks: [],
        });
        expect(normalizeParamBlockEditSet({ blocks: undefined })).toEqual({
            schemaVersion: 2,
            revision: 0,
            blocks: [],
        });
    });

    it("reports malformed groups instead of silently dropping them", () => {
        const result = normalizeParamBlockEditSetWithDiagnostics({
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "bad-group",
                    geometry: { kind: "group", children: [] },
                    createdSeq: 1,
                },
            ],
        } as never);
        expect(result.edits.blocks).toHaveLength(0);
        expect(result.diagnostics).toContain("group_too_small");
    });

    it("never throws on malformed members or baseCover and keeps valid geometry", () => {
        const result = normalizeParamBlockEditSetWithDiagnostics({
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "bad-members",
                    geometry: { kind: "take", members: [null, { clipId: "" }] },
                    createdSeq: 1,
                },
                {
                    id: "bad-cover",
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: "clip",
                                takeId: "take",
                                cycle: 0,
                                sourceStartSec: 0,
                                sourceEndSec: 0.5,
                                label: "manual",
                            },
                        ],
                    },
                    baseCover: { spans: "nope" },
                    createdSeq: 2,
                },
            ],
        } as never);
        expect(result.edits.blocks.map((block) => block.id)).toEqual(["bad-cover"]);
        expect(result.edits.blocks[0]?.baseCover).toBeUndefined();
        expect(result.diagnostics).toContain("block_missing_members");
        expect(result.diagnostics).toContain("block_malformed_base_cover");
    });

    it("degrades a single-child group to the child and rejects mixed root/take groups", () => {
        const take = {
            id: "child",
            geometry: {
                kind: "take" as const,
                members: [
                    {
                        clipId: "clip",
                        takeId: "take",
                        cycle: 0,
                        sourceStartSec: 0,
                        sourceEndSec: 0.5,
                        label: "manual" as const,
                    },
                ],
            },
            createdSeq: 1,
        };
        const root = {
            id: "root-child",
            geometry: { kind: "root" as const, span: { startSec: 0, endSec: 0.5 } },
            createdSeq: 2,
        };
        const single = normalizeParamBlockEditSetWithDiagnostics({
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "single-group",
                    geometry: { kind: "group", children: [take] },
                    createdSeq: 1,
                },
            ],
        } as never);
        expect(single.edits.blocks[0]?.id).toBe("child");
        expect(single.diagnostics).toContain("group_too_small");

        const mixed = normalizeParamBlockEditSetWithDiagnostics({
            schemaVersion: 2,
            revision: 1,
            blocks: [
                {
                    id: "mixed-group",
                    geometry: { kind: "group", children: [take, root] },
                    createdSeq: 1,
                },
            ],
        } as never);
        expect(mixed.edits.blocks).toHaveLength(0);
        expect(mixed.diagnostics).toContain("group_mixed_geometry");
    });

    it("converts legacy groupedFrom snapshots into Group children exactly once", () => {
        const member = (startSec: number) => ({
            clipId: "clip",
            takeId: "take",
            cycle: 0,
            sourceStartSec: startSec,
            sourceEndSec: startSec + 0.5,
            label: "syllable" as const,
        });
        const child = (id: string, startSec: number) => ({
            id,
            geometry: { kind: "take" as const, members: [member(startSec)] },
            createdSeq: 1,
        });
        const legacy = {
            schemaVersion: 1,
            revision: 1,
            blocks: [
                {
                    id: "legacy-group",
                    geometry: { kind: "take", members: [member(0), member(0.5)] },
                    groupedFrom: [child("a", 0), child("b", 0.5)],
                    createdSeq: 2,
                },
            ],
        };
        const normalized = normalizeParamBlockEditSet(legacy as never);
        const group = normalized.blocks[0];
        expect(group?.geometry.kind).toBe("group");
        if (group?.geometry.kind !== "group") throw new Error("expected group");
        expect(group.geometry.children.map((item) => item.id)).toEqual(["a", "b"]);
        expect("groupedFrom" in (group as object)).toBe(false);
    });
});
