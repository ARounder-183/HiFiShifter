import { describe, expect, it } from "vitest";
import { normalizeParamBlockEditSet } from "./paramBlockTypes";
import { paramBlockClipLayoutKey } from "./useParamBlockPersistence";

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

describe("paramBlockClipLayoutKey", () => {
    it("changes with active take, loop, and analysis spans", () => {
        const base = {
            id: "clip",
            trackId: "root",
            startSec: 0,
            lengthSec: 2,
            sourceStartSec: 0,
            sourceEndSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            activeTakeId: "a",
            takes: [
                {
                    id: "a",
                    sourceStartSec: 0,
                    sourceEndSec: 2,
                    playbackRate: 1,
                    reversed: false,
                    loopEnabled: false,
                    syllableSpansSec: [{ startSec: 0, endSec: 1 }],
                },
            ],
        };
        const first = paramBlockClipLayoutKey([base]);
        expect(paramBlockClipLayoutKey([{ ...base, activeTakeId: "b" }])).not.toBe(first);
        expect(paramBlockClipLayoutKey([{ ...base, loopEnabled: true }])).not.toBe(first);
        expect(
            paramBlockClipLayoutKey([
                {
                    ...base,
                    takes: [{ ...base.takes[0], syllableSpansSec: [{ startSec: 0, endSec: 0.5 }] }],
                },
            ]),
        ).not.toBe(first);
    });
});
