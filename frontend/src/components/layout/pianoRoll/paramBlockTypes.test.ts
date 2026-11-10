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
