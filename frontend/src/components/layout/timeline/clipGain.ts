/**
 * 片段**item 层**的增益 —— 时间轴旋钮编辑、header 标签展示的那一层。
 *
 * 【为什么要一个函数】REAPER 的 item 增益与 take 增益是两个旋钮。渲染、几何、标签
 * 若能各自决定读哪一层，迟早有一处读错（表现是"标签显示 0 dB，声音却是 −6 dB"）。
 * 统一从这里取，读哪一层就只有一个答案。
 *
 * 【用在哪些地方】凡是把 `ClipInfo` 投影成"视图片段"（命中块、渲染块、泳道块）的
 * 构造点：把 item 增益写进视图的 `gain` 字段，下游的样式解析与标签就自动一致——
 * 不需要给四层视图类型各加一个新字段。
 *
 * `clip_gain` 缺省（旧载荷）按 1.0；0 是合法的静音值，不能被当成"缺省"。
 */
export function clipItemGain(clip: { clip_gain?: number } | null | undefined): number {
    return clampGain(finiteOrUnity(clip?.clip_gain));
}

import { GAIN_PRODUCT_MAX, clampGain, clampTakeGain } from "../../../utils/gainDb";

type GainClipLike = {
    /** Take 级增益（active take 投影）。不是 Clip 增益。 */
    gain?: number;
    /** Item 级增益（REAPER item D_VOL）。 */
    clip_gain?: number;
};

function finiteOrUnity(value: number | undefined): number {
    return typeof value === "number" && Number.isFinite(value) ? value : 1;
}

/**
 * Active take 层增益。**保号**（负值 = 极性反转），0 是静音。
 */
export function clipTakeGain(clip: GainClipLike | null | undefined): number {
    return clampTakeGain(finiteOrUnity(clip?.gain));
}

/**
 * 可听增益（独立 App 施加）：item × take，保号。
 * ARA 里由宿主施加，此函数只用于显示/波形，不参与渲染。
 */
export function clipAudibleGain(clip: GainClipLike | null | undefined): number {
    const product = clipItemGain(clip) * clipTakeGain(clip);
    if (!Number.isFinite(product)) return 1;
    return Math.max(-GAIN_PRODUCT_MAX, Math.min(GAIN_PRODUCT_MAX, product));
}

/** 波形幅度：可听增益的绝对值（几何层要求非负）。 */
export function clipWaveformGain(clip: GainClipLike | null | undefined): number {
    return Math.abs(clipAudibleGain(clip));
}

/** 指定 Take 的波形幅度：|item × take.gain|。 */
export function clipTakeWaveformGain(
    clip: GainClipLike | null | undefined,
    takeGain: number | undefined,
): number {
    const product = clipItemGain(clip) * clampTakeGain(finiteOrUnity(takeGain));
    if (!Number.isFinite(product)) return 1;
    return Math.min(Math.abs(product), GAIN_PRODUCT_MAX);
}
