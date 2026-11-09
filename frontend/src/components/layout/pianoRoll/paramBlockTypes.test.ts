import { describe, expect, it } from "vitest";
import { normalizeParamBlockEditSet } from "./paramBlockTypes";

describe("normalizeParamBlockEditSet", () => {
    it("always materializes blocks as an array", () => {
        expect(normalizeParamBlockEditSet({ schemaVersion: 1, revision: 3 })).toEqual({
            schemaVersion: 1,
            revision: 3,
            blocks: [],
        });
        expect(normalizeParamBlockEditSet({ blocks: undefined })).toEqual({
            schemaVersion: 1,
            revision: 0,
            blocks: [],
        });
    });
});
