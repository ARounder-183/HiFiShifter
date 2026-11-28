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
    readonly currentParamDefaultValue?: number | null;
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
    /** 取数窗口视图：`samples` / `baseSamples` 用它（块高只在有真实采样的区间里算）。 */
    view: ParamViewSegment | null;
    /**
     * **曲线**用的视图：取样域已扩展到视口（见 `extendCurveViewToViewport`）。
     *
     * 【为什么要和 `view` 分开】两者在"窗口外有没有数据"上不同：曲线要画到视口边缘
     * （用户正在看的那一段），而块的中心值只能从**真实采样**里算 —— 把 NaN 与窗口外的帧
     * 喂进 `averageParamValueInSpan`，块高会在滚动时抖动。缺省等于 `view`。
     */
    curveView?: ParamViewSegment | null;
    blocks: PianoRollParamBlockSpec | null;
    currentParamDefaultValue?: number | null;
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
    // 曲线视图走同一条 preview 合成（它带着自己的 edit 数组），但**不**回写到 samples。
    const curveView = args.curveView == null ? view : composePreview(args.curveView);
    return {
        id: ++nextFrame,
        param: args.param,
        blocks: args.blocks,
        view,
        curves: typeof args.curves === "function" ? args.curves(curveView) : args.curves,
        samples: samples(view),
        baseSamples: samples(view?.preview?.base ?? view),
        currentParamDefaultValue: args.currentParamDefaultValue,
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
        fallbackValue: block.fallbackValue,
        currentParamDefaultValue: frame.currentParamDefaultValue,
    });
}
