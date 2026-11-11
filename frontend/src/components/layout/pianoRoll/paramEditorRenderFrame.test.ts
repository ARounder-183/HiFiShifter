import { expect, test } from "vitest";
import {
    createParamEditorRenderFrame,
    resolveParamEditorBlockValue,
} from "./paramEditorRenderFrame";
import type { ParamViewSegment } from "./types";
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";
const view: ParamViewSegment = {
    key: "root|pitch|0|8",
    startFrame: 0,
    stride: 1,
    framePeriodMs: 5,
    referenceKind: "source_curve",
    orig: [60, 60, 60, 60, 66, 66, 66, 66],
    edit: [60, 60, 60, 60, 66, 66, 66, 66],
};
const blocks = [
    { id: "created", startSec: 0, endSec: 0.01, averageValue: 60, selected: true, grouped: false },
    {
        id: "unrelated",
        startSec: 0.02,
        endSec: 0.04,
        averageValue: 66,
        selected: false,
        grouped: false,
    },
];
const spec: PianoRollParamBlockSpec = {
    param: "pitch",
    curveOwned: true,
    blocks,
    valueMin: 0,
    valueMax: 127,
    borderRgba: [1, 1, 1, 1],
    selectedRgba: [1, 1, 1, 1],
    groupedRgba: [1, 1, 1, 1],
};
function preview() {
    return {
        ...view,
        edit: [64, 64, 60, 60, 70, 70, 70, 70],
        preview: {
            base: view,
            affectedRanges: [{ startFrame: 0, frameCount: 2 }],
            blockValues: new Map([["unrelated", { startSec: 0.02, endSec: 0.04, value: 66 }]]),
        },
    };
}
test("a create preview cannot move an unrelated block even if another window was mixed in", () => {
    const frame = createParamEditorRenderFrame({
        param: "pitch",
        view: preview(),
        blocks: spec,
        curves: [],
    });
    expect(resolveParamEditorBlockValue(frame, blocks[0])).toBe(64);
    expect(resolveParamEditorBlockValue(frame, blocks[1])).toBe(66);
    expect(frame.samples?.values).toBe(frame.view?.edit);
    expect(frame.baseSamples?.values).toBe(view.edit);
});
test("trimmed neighbours sample their remaining curve instead of a saved whole-block center", () => {
    const data = preview();
    data.preview.blockValues.set("unrelated", { startSec: 0, endSec: 0.04, value: 63 });
    const frame = createParamEditorRenderFrame({
        param: "pitch",
        view: data,
        blocks: spec,
        curves: [],
    });
    expect(resolveParamEditorBlockValue(frame, blocks[1])).toBe(66);
});
test("frame publication captures layout and the curve window together", () => {
    const frame = createParamEditorRenderFrame({ param: "pitch", view, blocks: spec, curves: [] });
    const next = createParamEditorRenderFrame({
        param: "pitch",
        view: { ...view, edit: view.edit.map((v) => v + 1) },
        blocks: { ...spec, blocks: [] },
        curves: [],
    });
    expect(next.id).toBeGreaterThan(frame.id);
    expect(frame.blocks?.blocks).toBe(blocks);
    expect(resolveParamEditorBlockValue(frame, blocks[1])).toBe(66);
    expect(resolveParamEditorBlockValue(next, blocks[1])).toBe(67);
});

test("sparse preview patches cannot leak changed samples outside their declared range", () => {
    const candidate = preview();
    const frame = createParamEditorRenderFrame({
        param: "pitch",
        view: candidate,
        blocks: spec,
        curves: (normalized) => [
            {
                values: normalized!.edit,
                param: "pitch",
                startFrame: normalized!.startFrame,
                stride: normalized!.stride,
                framePeriodMs: normalized!.framePeriodMs,
                rgba: [1, 1, 1, 1],
                lineWidthPx: 2.6,
                projection: "curve",
                valueToY: (value) => value,
            },
        ],
    });
    expect(frame.view?.edit).toEqual([64, 64, 60, 60, 66, 66, 66, 66]);
    expect(frame.curves[0].values).toBe(frame.samples?.values);
});
test("a late different sampling window cannot change the gesture's original domain", () => {
    const base = preview();
    const candidate: ParamViewSegment = {
        ...base,
        key: "late",
        startFrame: 100,
        stride: 4,
        edit: [99, 99],
        preview: { ...base.preview, patches: [{ startFrame: 0, values: [64, 64] }] },
    };
    const frame = createParamEditorRenderFrame({
        param: "pitch",
        view: candidate,
        blocks: spec,
        curves: [],
    });
    expect(frame.view?.startFrame).toBe(0);
    expect(frame.view?.stride).toBe(1);
    expect(frame.view?.edit).toEqual([64, 64, 60, 60, 66, 66, 66, 66]);
    expect(resolveParamEditorBlockValue(frame, blocks[1])).toBe(66);
});
