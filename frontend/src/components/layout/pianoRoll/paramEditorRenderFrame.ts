import type { ParamViewSegment } from "./types";
import type {
    PianoRollParamBlockSpec,
    PianoRollParamBlockSamples,
    PianoRollCurveLayer,
} from "./kernel/host/pianoRollKernelData";
import { resolveParamBlockFrameValue } from "./paramBlockFrame";

export interface ParamEditorRenderFrame {
    readonly id: number;
    readonly param: string;
    readonly view: ParamViewSegment | null;
    readonly blocks: PianoRollParamBlockSpec | null;
    readonly curves: readonly PianoRollCurveLayer[];
    readonly samples: PianoRollParamBlockSamples | null;
    readonly baseSamples: PianoRollParamBlockSamples | null;
}
const previewViews = new WeakMap<ParamViewSegment, ParamViewSegment>();
function composePreview(view: ParamViewSegment | null): ParamViewSegment | null {
    if (!view?.preview) return view;
    const cached = previewViews.get(view);
    if (cached) return cached;
    const { base, patches, affectedRanges } = view.preview;
    const edit = Array.from(base.edit);
    const stride = Math.max(1, base.stride);
    for (let index = 0; index < edit.length; index++) {
        const frame = base.startFrame + index * stride;
        if (
            !affectedRanges.some(
                (range) => frame >= range.startFrame && frame < range.startFrame + range.frameCount,
            )
        )
            continue;
        if (patches) {
            for (const patch of patches) {
                const value = patch.values[frame - patch.startFrame];
                if (value !== undefined && Number.isFinite(value)) edit[index] = value;
            }
        } else {
            const source = (frame - view.startFrame) / Math.max(1, view.stride);
            if (Number.isInteger(source) && Number.isFinite(view.edit[source]))
                edit[index] = view.edit[source];
        }
    }
    const result = {
        ...view,
        key: base.key,
        startFrame: base.startFrame,
        stride: base.stride,
        framePeriodMs: base.framePeriodMs,
        orig: base.orig ? Array.from(base.orig) : view.orig,
        edit,
    };
    previewViews.set(view, result);
    return result;
}
let nextFrame = 0;
export function createParamEditorRenderFrame(args: {
    param: string;
    view: ParamViewSegment | null;
    blocks: PianoRollParamBlockSpec | null;
    curves:
        | readonly PianoRollCurveLayer[]
        | ((view: ParamViewSegment | null) => readonly PianoRollCurveLayer[]);
}): ParamEditorRenderFrame {
    const samples = (
        view: Readonly<{
            edit: readonly number[];
            startFrame: number;
            stride: number;
            framePeriodMs: number;
        }> | null,
    ): PianoRollParamBlockSamples | null =>
        view
            ? {
                  values: view.edit,
                  startFrame: view.startFrame,
                  stride: view.stride,
                  framePeriodMs: view.framePeriodMs,
                  param: args.param,
                  isPitch: args.param === "pitch",
              }
            : null;
    const view = composePreview(args.view);
    return {
        id: ++nextFrame,
        param: args.param,
        blocks: args.blocks,
        view,
        curves: typeof args.curves === "function" ? args.curves(view) : args.curves,
        samples: samples(view),
        baseSamples: samples(view?.preview?.base ?? view),
    };
}

export function resolveParamEditorBlockValue(
    frame: ParamEditorRenderFrame,
    block: NonNullable<PianoRollParamBlockSpec>["blocks"][number],
): number | null {
    const preview = frame.view?.preview;
    const fp = frame.view?.framePeriodMs ?? 5;
    const affected = preview?.affectedRanges.some(
        (range) =>
            (block.startSec * 1000) / fp < range.startFrame + range.frameCount &&
            (block.endSec * 1000) / fp > range.startFrame,
    );
    const stable = preview?.blockValues.get(block.id);
    if (
        !affected &&
        stable &&
        stable.startSec === block.startSec &&
        stable.endSec === block.endSec &&
        block.frameCenterValue == null
    )
        return stable.value;
    return resolveParamBlockFrameValue({
        block,
        param: frame.param,
        curveOwned: frame.blocks?.curveOwned,
        samples: preview && !affected ? frame.baseSamples : frame.samples,
    });
}
