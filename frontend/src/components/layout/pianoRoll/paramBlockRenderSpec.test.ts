import { describe, expect, it } from "vitest";
import { buildParamBlockRenderSpec, type ParamBlockRenderBlock } from "./paramBlockRenderSpec";

const colors = {
    borderRgba: [0.2, 0.2, 0.2, 1] as const,
    selectedRgba: [0.1, 0.5, 1, 1] as const,
    groupedRgba: [1, 0.5, 0.2, 1] as const,
};

function block(
    overrides: Partial<ParamBlockRenderBlock> &
        Pick<ParamBlockRenderBlock, "id" | "startSec" | "endSec">,
): ParamBlockRenderBlock {
    return {
        averageValue: 1,
        selected: false,
        grouped: false,
        ...overrides,
    };
}

function build(blocks: readonly ParamBlockRenderBlock[]) {
    return buildParamBlockRenderSpec({
        blocks,
        param: "volume",
        valueMin: 0,
        valueMax: 2,
        valueToY: (value) => 100 - value * 10,
        colors,
    });
}

describe("buildParamBlockRenderSpec", () => {
    it("computes contiguous flags and transition handles from final geometry", () => {
        const spec = build([
            block({ id: "a", startSec: 0, endSec: 1, transitionSec: 0.2, sourceKey: "s|t|0" }),
            block({ id: "b", startSec: 1, endSec: 2, sourceKey: "s|t|0" }),
        ]);
        expect(spec.blocks[0]).toMatchObject({
            contiguousEnd: true,
            transitionHandle: { sec: 1, leftId: "a", rightId: "b", widthSec: 0.2 },
        });
        expect(spec.blocks[1]).toMatchObject({ contiguousStart: true, transitionHandle: null });
    });

    it("does not connect non-adjacent blocks", () => {
        const spec = build([
            block({ id: "a", startSec: 0, endSec: 1 }),
            block({ id: "b", startSec: 2, endSec: 3 }),
        ]);
        expect(spec.blocks.every((item) => item.transitionHandle === null)).toBe(true);
        expect(spec.blocks.every((item) => !item.contiguousStart && !item.contiguousEnd)).toBe(
            true,
        );
    });

    it("sorts unordered input and labels logical runs without numbering", () => {
        const spec = build([
            block({ id: "b", startSec: 4, endSec: 5 }),
            block({ id: "a:1", userBlockId: "a", startSec: 2, endSec: 3, averageValue: 0.5 }),
            block({ id: "a:0", userBlockId: "a", startSec: 0, endSec: 1, averageValue: 0.5 }),
        ]);
        expect(spec.blocks.map((item) => item.id)).toEqual(["a:0", "a:1", "b"]);
        expect(spec.blocks[0]?.label).toBe("0.5");
        expect(spec.blocks[1]?.label).toBe("0.5");
        expect(spec.blocks[2]?.label).toBe("1.0");
    });

    it("does not connect adjacent blocks from different Clip/Take sources", () => {
        const spec = build([
            block({
                id: "a",
                startSec: 0,
                endSec: 1,
                transitionSec: 0.2,
                sourceKey: "A|take|0",
            }),
            block({ id: "b", startSec: 1, endSec: 2, sourceKey: "B|take|0" }),
        ]);
        expect(spec.blocks[0]).toMatchObject({ contiguousEnd: false, transitionHandle: null });
        expect(spec.blocks[1]).toMatchObject({ contiguousStart: false });
    });

    it("keeps adjacency for blocks from one source", () => {
        const spec = build([
            block({
                id: "a",
                startSec: 0,
                endSec: 1,
                transitionSec: 0.2,
                sourceKey: "A|take|0",
            }),
            block({ id: "b", startSec: 1, endSec: 2, sourceKey: "A|take|0" }),
        ]);
        expect(spec.blocks[0]).toMatchObject({
            contiguousEnd: true,
            transitionHandle: { leftId: "a", rightId: "b" },
        });
    });

    it("selected only changes the presentation field, not geometry", () => {
        const spec = build([block({ id: "a", startSec: 0, endSec: 1, selected: true })]);
        expect(spec.blocks[0]?.selected).toBe(true);
        expect(spec.blocks[0]?.startSec).toBe(0);
        expect(spec.blocks[0]?.endSec).toBe(1);
    });
});
