import { describe, expect, it } from "vitest";
import {
    resolveParamBlockPointerIntent,
    type ParamBlockPointerPolicyInput,
} from "./paramBlockPointerPolicy";

const block = { kind: "block" as const, blockId: "run-a", logicalId: "logical-a" };
const edge = { kind: "edge" as const, blockId: "run-a", edge: "left" as const };
const transition = { kind: "transition" as const, leftId: "run-a", rightId: "run-b" };
const blank = { kind: "blank" as const };

function intent(overrides: Partial<ParamBlockPointerPolicyInput>) {
    return resolveParamBlockPointerIntent({
        button: 0,
        target: block,
        splitAtClick: false,
        createBlock: false,
        createFlat: false,
        multiSelect: false,
        rangeSelect: false,
        frame: 42,
        ...overrides,
    });
}

describe("paramBlockPointerPolicy", () => {
    it("left click block selects or drags", () => {
        expect(intent({})).toEqual({
            kind: "block-select-or-drag",
            target: block,
            mode: "replace",
        });
    });

    it("split modifier wins over Ctrl and Shift on a block", () => {
        expect(intent({ splitAtClick: true, multiSelect: true })).toEqual({
            kind: "block-split",
            target: block,
            frame: 42,
        });
        expect(intent({ splitAtClick: true, rangeSelect: true })).toMatchObject({
            kind: "block-split",
        });
    });

    it("Ctrl+Alt on a block falls back to Ctrl selection instead of splitting", () => {
        expect(intent({ splitAtClick: true, createFlat: true, multiSelect: true })).toEqual({
            kind: "block-select-or-drag",
            target: block,
            mode: "toggle",
        });
    });

    it("right button on block/edge/transition goes to morph", () => {
        expect(intent({ button: 2 })).toEqual({ kind: "block-morph", target: block });
        expect(intent({ button: 2, target: edge })).toMatchObject({ kind: "block-morph" });
        expect(intent({ button: 2, target: transition })).toMatchObject({ kind: "block-morph" });
    });

    it("right button on blank appends a range", () => {
        expect(intent({ button: 2, target: blank })).toEqual({
            kind: "background-range",
            mode: "add",
        });
    });

    it("blank drag prioritizes createFlat then create then range", () => {
        expect(intent({ target: blank, createFlat: true, createBlock: true })).toEqual({
            kind: "background-create",
            flatten: true,
        });
        expect(intent({ target: blank, createBlock: true })).toEqual({
            kind: "background-create",
            flatten: false,
        });
        expect(intent({ target: blank, multiSelect: true })).toEqual({
            kind: "background-range",
            mode: "add",
        });
    });

    it("left split on an edge is a split candidate; without the modifier it resizes", () => {
        expect(intent({ target: edge, splitAtClick: true })).toMatchObject({ kind: "block-split" });
        expect(intent({ target: edge })).toEqual({ kind: "edge-resize", target: edge });
    });

    it("transition handles resize on the left button and morph on the right", () => {
        expect(intent({ target: transition })).toEqual({
            kind: "transition-resize",
            target: transition,
        });
        expect(intent({ target: transition, button: 2 })).toMatchObject({ kind: "block-morph" });
    });
});
