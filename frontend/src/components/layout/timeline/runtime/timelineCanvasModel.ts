/**
 * Clip 体画布的稀疏渲染模型。
 *
 * 【主要内容】把窗口内可见的 clip 元数据转换为画布绘制所需的像素几何
 * （`TimelineCanvasClipModel[]`），并判定哪些 clip 需要额外的 DOM 覆盖层。
 * 同时产出**可选**的 per-clip 细节（静音区段 / lane 分界线 / MIDI 音高折线与
 * 回绕标记）——它们的像素投影必须与 clip 几何同源，故一并在此完成。
 *
 * 【作用】clip **本体**由 canvas 绘制（数量多、重绘频繁），只有选中/悬停/
 * 重命名/重叠的 clip 才额外渲染 DOM 覆盖层以获得交互手柄。本文件负责这条
 * 分工的判定与几何产出。
 *
 * 【与其他模块的关系】（已按"渲染内核唯一路径"改造更新）
 * - 上游：`timeline/kernel/host/timelineKernelHost` 在构建帧时调用
 *   `buildSparseClipRenderModel()`，把 `drawClips` 交给
 *   `kernel/scene/clipInstances` 转成 GL 实例。
 *   旧消费者 `TimelineCanvasViewport`（绘制）与 `TrackLane`（决定 DOM 覆盖层）
 *   **已随旧渲染路径删除**；`overlayClipIdsByTrackId` 因此**无生产消费者**，仅基准脚本
 *   仍在读取（内核模式下不再有 DOM 覆盖层这一层分工）。
 *   同一改造删除了 `components/waveform/MidiPitchTrackCanvas.tsx`——MIDI 音高
 *   折线一度因此丢失，其纯数学现落在 `runtime/midiPitchCurve.ts`，由本文件投影、
 *   由 `runtime/timelineCanvasRenderer` 落笔。
 * - 横向：所有时间↔像素换算一律走 `timelineAxis.ts`，本文件不得自行
 *   执行 `sec * pxPerSec`，否则会与波形、网格产生错位。
 */

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../constants.js";
import { clipDisplayName, type ClipInfo } from "../../../../features/session/sessionTypes";
import { resolveTakeLaneLayouts } from "../takeLanes";
import { resolveClipContentDurationSec, resolveSourceEndSec } from "../../../../utils/loopRender.js";
import {
    FRAME_PERIOD_MS,
    getCachedMidiCurve,
    resolveClipLoopMarkerOffsetsSec,
    resolveLoopCycleDescriptor,
    strokeColorForClip,
} from "./midiPitchCurve.js";
import {
    durationToWidthPx,
    secToContentPx,
    secToSpanPx,
    viewportStartSec,
    type TimelineAxis,
} from "../../renderKernel/timelineAxis.js";

type SparseRenderClip = {
    id: string;
    trackId: string;
    name: string;
    startSec: number;
    lengthSec: number;
    gain: number;
    playbackRate: number;
    muted: boolean;
    /**
     * Take 集合。
     *
     * `sourcePath` 参与 lane 布局判定（只有音频 take 建 lane，见 `takeLanes`），
     * 因此这里必须保留——只声明 id / name 会让多 Take clip 在模型侧退化成单 Take。
     */
    takes?: Array<{ id: string; name: string; sourcePath?: string }>;
    activeTakeId?: string;
    midiNoteCount?: number;
    groupId?: string;
    fadeInSec: number;
    fadeOutSec: number;
    fadeInShape: number;
    fadeOutShape: number;
    fadeInDir: number;
    fadeOutDir: number;
    /** 自动交叉淡化长度（可选；缺省 0），用于“有效 fade”显示。 */
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
    /** 吸附偏移（秒，相对 Clip 起点；缺省 0）—— 左下角 ◣ 标记。 */
    snapOffsetSec?: number;
    // ── 音高折线所需的源/内容字段 ──────────────────────────────────
    // 以下字段在**生产路径**上由 `ClipInfo` 提供（恒存在），声明为可选只是
    // 为了兼容既有测试 / 基准夹具里的最小 clip 字面量；模型内部一律兜底
    // （见 `buildMidiPitchCurvePx`）。true 生产数据下不会走兜底分支。
    /** clip 调色板色名（`ClipInfo["color"]`）：决定折线描边色。 */
    color?: string;
    /** 源窗口起点（秒）；缺省 0。 */
    sourceStartSec?: number;
    /** 源窗口终点（秒，可为负 / 超界）；缺失时按音符范围估计。 */
    sourceEndSec?: number;
    /** 是否倒放。 */
    reversed?: boolean;
    /** Loop（循环源）：延伸超出源媒体区间时按周期回绕。 */
    loopEnabled?: boolean;
    sourcePath?: string;
    durationSec?: number;
    durationFrames?: number;
    sourceSampleRate?: number;
    /** MIDI 音符数据（音高参考块的权威来源）。 */
    midiNoteData?: Array<{ startSec: number; endSec: number; note: number }>;
    /** 是否用前一个有效音高填满音符空隙（后端 `fill_gaps_in_pitch_edit`）。 */
    midiFillGaps?: boolean;
};
export type TimelineCanvasClipModel = {
    id: string;
    trackId: string;
    name: string;
    leftPx: number;
    topPx: number;
    widthPx: number;
    heightPx: number;
    headerHeightPx: number;
    fadeInPx: number;
    fadeOutPx: number;
    fadeInShape: number;
    fadeOutShape: number;
    fadeInDir: number;
    fadeOutDir: number;
    selected: boolean;
    muted: boolean;
    gain: number;
    playbackRate: number;
    groupId?: string;
    isMidiClip: boolean;
    trackColor?: string;
    isRenaming: boolean;
    /** 吸附偏移（已换算为像素，相对 Clip 左缘）—— 左下角 ◣ 标记。 */
    snapOffsetPx: number;
    /**
     * 前导重叠区宽度（像素，相对 Clip 左缘）：被同轨前一个 clip 压住的部分。
     *
     * 该区在画布上按半透绘制，让下 clip 的色块与波形透出——否则两层不透明
     * 色块会叠加成脏色。
     *
     * 历史说明：这个字段此前**一直没写进本接口**，只在 `drawClips` 的推断
     * 返回类型里泄漏出去（构造处未标注类型，故 TS 未报错）。消费端
     * `drawTimelineCanvas` 却能读到它——类型与实际不符，补上声明。
     */
    leadingOverlapPx?: number;
    /**
     * 静音检测预览区段（像素，相对 clip 左缘）。
     *
     * 与旧实现 `ClipItem` 的红色覆盖层同源：后端给出的是**工程秒**区间，这里按
     * `axis` 投影为像素并**钳制到 clip 本体**（后端区域越过末端时不得把红色画到
     * 相邻 clip 上）。空数组与缺省等价（不绘制）。
     */
    silenceSpansPx?: Array<{ leftPx: number; widthPx: number }>;
    /**
     * 多 Take lane 分界线的 y 偏移（相对 **body 顶部**，CSS px）。
     *
     * 首条 lane 的顶边就是 header 边界，因此不含 0（`slice(1)`）。布局与波形面 /
     * 点击命中同源（`takeLanes.resolveTakeLaneLayouts`）——两处各算一份会让
     * 分界线与波形 lane 错位。空数组与缺省等价（不绘制）。
     */
    takeLaneSeparatorOffsetsPx?: number[];
    /**
     * MIDI / 音高参考块 body 内的音高折线点（相对该 clip 自身）。
     *
     * 只有 `midiNoteCount != null` 的 clip 有音高内容（音频 clip 画波形、MIDI
     * clip 画折线）；无音符数据时缺省（不绘制）。
     *
     * 坐标约定与 `silenceSpansPx` 同源：`x` 相对 **clip 左缘**、`y` 为
     * **body 局部**坐标（`0..bodyHeightPx`，原点在 body 顶部）。音高值域
     * （min/max note → y）已在模型侧消化，绘制端不需要知道 MIDI 值域。
     */
    midiPitchCurvePx?: Array<{ x: number; y: number }>;
    /**
     * 音高折线的描边色（由 clip 调色板色名映射，见 `strokeColorForClip`）。
     *
     * 只对 MIDI / 音高参考块有意义（仅 `midiNoteCount != null` 时产出）；
     * 与 `midiPitchCurvePx` 同生共死。
     */
    midiPitchStroke?: string;
    /**
     * Loop 回绕 / 媒体边界 "▽" 标记的 x 偏移（相对 clip 左缘，CSS px）。
     *
     * 与 `midiPitchCurvePx` 同源（`resolveClipLoopMarkerOffsetsSec`），缺省 =
     * 无标记不绘制。**同样是模型侧投影**：绘制端拿不到 `pxPerSec`。
     */
    midiLoopMarkerOffsetsPx?: number[];
};

/**
 * 静音检测预览区段 → clip 内的相对像素区间。
 *
 * 流程：逐个区间取与 clip 本体的交集（**防御性钳制**：后端区域越过 clip 末端时
 * 不得把红色画到相邻 clip 上，与旧实现 `ClipItem` 的钳制同源）→ 投影为相对像素。
 *
 * 特殊说明：只保留长度 > 0 的区间；全部无效时返回 undefined（而非空数组），
 * 让绘制端用一次 `!== undefined` 判断跳过。
 *
 * @param args.axis 统一坐标投影。
 * @param args.clipStartSec clip 起点（工程秒）。
 * @param args.clipLengthSec clip 长度（秒）。
 * @param args.segments 该 clip 的静音区间（工程秒）。
 * @returns 相对 clip 左缘的像素区间；无有效区间时为 undefined。
 */
function buildSilenceSpansPx(args: {
    axis: TimelineAxis;
    clipStartSec: number;
    clipLengthSec: number;
    segments?: ReadonlyArray<readonly [number, number]>;
}): Array<{ leftPx: number; widthPx: number }> | undefined {
    const segments = args.segments;
    if (segments === undefined || segments.length === 0) return undefined;
    const clipStartSec = Number(args.clipStartSec) || 0;
    const clipEndSec = clipStartSec + Math.max(0, Number(args.clipLengthSec) || 0);
    const spans: Array<{ leftPx: number; widthPx: number }> = [];
    for (const segment of segments) {
        const startSec = Math.max(Number(segment[0]) || 0, clipStartSec);
        const endSec = Math.min(Number(segment[1]) || 0, clipEndSec);
        if (!(endSec > startSec)) continue;
        spans.push({
            leftPx: secToSpanPx(args.axis, startSec - clipStartSec),
            widthPx: Math.max(1, secToSpanPx(args.axis, endSec - startSec)),
        });
    }
    return spans.length > 0 ? spans : undefined;
}

/**
 * 多 Take lane 分界线偏移（相对 body 顶部，CSS px）。
 *
 * 流程：直接复用 `takeLanes.resolveTakeLaneLayouts`（与波形面 / 点击命中共用同一
 * 套数学）→ 去掉首条（其顶边即 header 边界）→ 取各 lane 的 `top`。
 *
 * 特殊说明：模型侧的 clip 是 `ClipInfo` 的结构子集，字段语义一致；布局只用到
 * takes / activeTakeId，因此这里的窄化是安全的（多出的字段被忽略）。
 *
 * @param args.clip 稀疏 clip（实际来自 `ClipInfo`）。
 * @param args.showAllTakes 是否平铺全部 Take。
 * @param args.bodyHeightPx clip body 的像素高度（决定能否容纳 lane）。
 * @returns 分界线偏移；单 Take / 空间不足时为 undefined。
 */
function buildTakeLaneSeparatorOffsets(args: {
    clip: SparseRenderClip;
    showAllTakes: boolean;
    bodyHeightPx: number;
}): number[] | undefined {
    const layouts = resolveTakeLaneLayouts(
        args.clip as unknown as ClipInfo,
        args.showAllTakes,
        args.bodyHeightPx,
    );
    if (layouts === null || layouts.length <= 1) return undefined;
    return layouts.slice(1).map((lane) => lane.top);
}

/**
 * MIDI / 音高参考块的音高折线点（相对 clip 左缘 / body 顶部，CSS px）。
 *
 * 流程：
 * 1. 取音符数据：优先 `clip.midiNoteData`（权威来源，拖拽/拉伸时即时反映），
 *    缺失时回退 Redux 推来的 `args.pitchCurve`（后端 `clip_pitch_data` 事件路径）；
 * 2. 用 `midiPitchCurve.ts` 的 `getCachedMidiCurve` 生成逐帧曲线（消费窗口、
 *    playbackRate、reversed、Loop 回绕的数学全部在那边，**不在此重写**）；
 * 3. 逐帧投影：`x = secToSpanPx(axis, 帧时间 − clip 起点)`，`y` 由 MIDI 值线性
 *    映射到 body 高度内（高音在上，上下各留 10% padding）；
 * 4. 按帧步长抽稀（沿用搬迁前的 `minFrameStep`：保证相邻点至少隔 0.5px）。
 *
 * 特殊说明：
 * - `framePeriodMs` 必须与曲线生成时用的一致，否则 x 会整体缩放错位——音符
 *   路径恒为 `FRAME_PERIOD_MS`，回退路径取后端载荷自带的周期。
 * - **空隙用 `y = NaN` 编码**：多段折线共用一条数组，绘制端遇 NaN 起新子路径。
 *   若不编码，音符之间的静音会被连成一条不存在的斜线。
 * - y 的映射与搬迁前 `MidiPitchTrackCanvas` **逐行一致**
 *   （`displayH - padding - normalized * (displayH - 2 * padding)`），
 *   否则折线相对波形/后端曲线的纵向位置会整体偏移。
 * - 只产出落在 clip 自身时间跨度内的点（越界部分绘制端也会被裁剪掉）。
 * - 无数据 / 曲线不足两点时返回 `undefined`（而不是空数组），与
 *   `buildSilenceSpansPx` 同一约定，让绘制端一次 `!== undefined` 判断跳过。
 *
 * @param args.axis 统一坐标投影（唯一时间→像素来源）。
 * @param args.clip 稀疏 clip（实际来自 `ClipInfo`）。
 * @param args.bodyHeightPx clip body 的像素高度（y 映射值域）。
 * @param args.pitchCurve Redux 回退曲线（缺省 = 无回退数据）。
 * @param args.pitchRange 音高值域（缺省 `0..127`，与搬迁前同缺省）。
 * @returns 折线点（NaN 的 y 表示断点）；无内容时为 undefined。
 */
function buildMidiPitchCurvePx(args: {
    axis: TimelineAxis;
    clip: SparseRenderClip;
    bodyHeightPx: number;
    pitchCurve?: { curveStartSec: number; midiCurve: number[]; framePeriodMs: number };
    pitchRange?: { min: number; max: number };
}): Array<{ x: number; y: number }> | undefined {
    const clip = args.clip;
    // 只有 MIDI / 音高参考块有音高内容（音频 clip 画波形）。
    if (clip.midiNoteCount == null) return undefined;
    const clipLengthSec = Number(clip.lengthSec) || 0;
    if (!(clipLengthSec > 0)) return undefined;

    const playbackRate = Number(clip.playbackRate) || 1;
    let midiCurve: number[];
    let curveStartSec: number;
    let framePeriodMs: number;
    const notes = clip.midiNoteData;

    // 每 clip 只算一次内容时长 D（曲线与回绕标记共用）。
    const contentDurSec = resolveClipContentDurationSec({
        sourcePath: clip.sourcePath,
        midiNoteData: notes ?? null,
        durationFrames: clip.durationFrames,
        sourceSampleRate: clip.sourceSampleRate,
        durationSec: clip.durationSec,
    });

    if (notes && notes.length > 0) {
        // 曲线消费窗口：非 Loop 正放 = 起点+长度×速率（派生），倒放 =
        // [se−len·r, se]（锚定 se，sourceStart 不参与）；与音频渲染的窗口模型
        // 一致 —— 否则延伸过的倒放 Clip 曲线整体错位（该有声处显示为空）。
        // se 仅在**缺失/非法**时回退音符范围估计 —— 合法的 0/负值
        // （倒放静音段锚点）不得被改写，否则派生链整体错位。
        const seRaw = Number.isFinite(clip.sourceEndSec)
            ? Number(clip.sourceEndSec)
            : notes.reduce((max, n) => Math.max(max, n.endSec), 0);
        const sourceStartSec = Number(clip.sourceStartSec) || 0;
        const srcEnd = resolveSourceEndSec({
            loopEnabled: Boolean(clip.loopEnabled),
            reversed: Boolean(clip.reversed),
            sourceStartSec,
            playbackRate: Math.abs(playbackRate) || 1,
            lengthSec: clipLengthSec,
            sourceEndSec: seRaw,
        });
        const curveWinStart =
            !clip.loopEnabled && clip.reversed
                ? srcEnd - Math.max(0, clipLengthSec) * (Math.abs(playbackRate) || 1)
                : sourceStartSec;
        const loopCycle = resolveLoopCycleDescriptor({
            loopEnabled: Boolean(clip.loopEnabled),
            contentDurationSec: contentDurSec,
            sourceStartSec,
            sourceEndSec: srcEnd,
        });
        midiCurve = getCachedMidiCurve(
            notes,
            clipLengthSec,
            curveWinStart,
            srcEnd,
            playbackRate,
            Boolean(clip.reversed),
            clip.midiFillGaps ?? false,
            loopCycle,
        );
        curveStartSec = Number(clip.startSec) || 0;
        framePeriodMs = FRAME_PERIOD_MS;
    } else {
        const pitchData = args.pitchCurve;
        if (!pitchData || !pitchData.midiCurve || pitchData.midiCurve.length < 2) return undefined;
        midiCurve = pitchData.midiCurve;
        curveStartSec = Number(pitchData.curveStartSec ?? clip.startSec) || 0;
        framePeriodMs = pitchData.framePeriodMs || FRAME_PERIOD_MS;
    }
    if (!midiCurve || midiCurve.length < 2) return undefined;

    // MIDI 值 → y：与搬迁前 MidiPitchTrackCanvas 同一映射（高音在上、
    // 上下各 10% padding）。
    const displayH = args.bodyHeightPx;
    if (!(displayH > 0)) return undefined;
    const minNote = args.pitchRange?.min ?? 0;
    const maxNote = args.pitchRange?.max ?? 127;
    const noteSpan = Math.max(1, maxNote - minNote);
    const padding = displayH * 0.1;
    const yForMidi = (midiValue: number): number => {
        const normalized = (midiValue - minNote) / noteSpan;
        const y = displayH - padding - normalized * (displayH - 2 * padding);
        return Math.max(padding, Math.min(displayH - padding, y));
    };

    // 帧步长抽稀（沿用搬迁前的 minFrameStep）：保证相邻点至少隔 0.5px，
    // 否则全览缩放下每个 clip 会产出上万点。
    const frameToPx = (framePeriodMs / 1000) * args.axis.pxPerSec;
    const minFrameStep = Math.max(1, Math.floor(0.5 / Math.max(0.01, frameToPx)));
    const clipStartSec = Number(clip.startSec) || 0;
    const points: Array<{ x: number; y: number }> = [];
    const lastFrame = midiCurve.length - 1;
    for (let fi = 0; fi <= lastFrame; fi += minFrameStep) {
        const frameTimeSec = curveStartSec + (fi * framePeriodMs) / 1000;
        const localSec = frameTimeSec - clipStartSec;
        // 曲线可能比 clip 长（后端曲线覆盖整个内容）：越界部分绘制端也会
        // 裁掉，这里先剔除以免白算。
        if (localSec < 0 || localSec > clipLengthSec) continue;
        const midiValue = midiCurve[fi];
        if (!(midiValue > 0)) {
            // 空隙：写入 NaN 断点让绘制端抬起画笔（不连成不存在的斜线）。
            // 连续空隙只写一个断点（否则长静音段会白占大量数组槽位）。
            const prev = points[points.length - 1];
            if (prev !== undefined && !Number.isNaN(prev.y)) {
                points.push({ x: secToSpanPx(args.axis, localSec), y: Number.NaN });
            }
            continue;
        }
        points.push({ x: secToSpanPx(args.axis, localSec), y: yForMidi(midiValue) });
    }
    // 收尾：抽稀可能刚好跳过最后一帧，补上以免折线提前结束。仅当上一笔仍在
    // 落笔状态（非断点）时才补——断点后的孤立点画不出任何线段。
    const tailLocalSec = curveStartSec + (lastFrame * framePeriodMs) / 1000 - clipStartSec;
    if (tailLocalSec >= 0 && tailLocalSec <= clipLengthSec && midiCurve[lastFrame] > 0) {
        const tailX = secToSpanPx(args.axis, tailLocalSec);
        const prev = points[points.length - 1];
        if (prev !== undefined && !Number.isNaN(prev.y) && prev.x < tailX) {
            points.push({ x: tailX, y: yForMidi(midiCurve[lastFrame]) });
        }
    }
    return points.length >= 2 ? points : undefined;
}

/**
 * MIDI / 音高参考块的 Loop 回绕标记 x 偏移（相对 clip 左缘，CSS px）。
 *
 * 流程：`midiPitchCurve.ts` 的 `resolveClipLoopMarkerOffsetsSec` 求出 clip 局部
 * 秒 → 经 `secToSpanPx` 投影为像素。
 *
 * 特殊说明：投影必须留在模型侧——绘制端只有内容坐标、拿不到 `pxPerSec`，
 * 自行换算就会重新引入「时间 × 每秒像素」的第二次实现（统一坐标投影的
 * 强制约束）。
 *
 * @returns 标记 x 偏移；无标记时为 undefined（不绘制）。
 */
function buildMidiLoopMarkerOffsetsPx(args: {
    axis: TimelineAxis;
    clip: SparseRenderClip;
}): number[] | undefined {
    const clip = args.clip;
    if (clip.midiNoteCount == null) return undefined;
    const clipStartSec = Number(clip.startSec) || 0;
    // 只从视口左缘之前一点开始产出标记：`resolveClipLoopMarkerOffsetsSec` 会用
    // 它直接跳到可视范围内的第一个回绕点。不跳会从 clip 入口逐周期空转，长循环
    // clip 一旦超过标记上限，**深处的标记就会整体消失**（搬迁前修过的缺陷）。
    // margin 取内核的水平预构建边距量级（512px），保证窗口左侧的标记也在。
    const marginSec = 512 / Math.max(1e-9, args.axis.pxPerSec);
    const fromLocalSec = Math.max(0, viewportStartSec(args.axis) - marginSec - clipStartSec);
    const offsets = resolveClipLoopMarkerOffsetsSec({
        loopEnabled: Boolean(clip.loopEnabled),
        reversed: Boolean(clip.reversed),
        lengthSec: Number(clip.lengthSec) || 0,
        playbackRate: Number(clip.playbackRate) || 1,
        sourceStartSec: Number(clip.sourceStartSec) || 0,
        sourceEndSec: Number(clip.sourceEndSec) || 0,
        sourcePath: clip.sourcePath,
        midiNoteData: clip.midiNoteData ?? null,
        durationFrames: clip.durationFrames,
        sourceSampleRate: clip.sourceSampleRate,
        durationSec: clip.durationSec,
        fromLocalSec,
    });
    if (offsets.length === 0) return undefined;
    return offsets.map((sec) => secToSpanPx(args.axis, sec));
}

/**
 * 音高折线的描边色：复用搬迁来的 `strokeColorForClip`（按 clip 调色板色名取色）。
 *
 * 特殊说明：clip 未带调色板色名时**不做** trackColor 映射——`trackColor` 是
 * 轨道十六进制色，不是调色板色名，硬塞进去只会命中未知分支。此时交给
 * `strokeColorForClip` 的缺省回退色。
 */
function resolveMidiPitchStrokeColor(clip: SparseRenderClip): string {
    return strokeColorForClip({ color: clip.color ?? "" });
}

/**
 * 构建 clip 体画布的稀疏渲染模型。
 *
 * 流程：
 * 1. 判定需要 DOM 覆盖层的 clip（重命名中 / 悬停 / 选中 / 同组 / 同轨重叠）；
 * 2. 把每个可见 clip 的秒级字段经 `TimelineAxis` 投影为像素几何。
 *
 * 特殊说明：
 * - 输出的是**内容绝对坐标**（原点为工程 0 秒），竖直滚动由画布统一
 *   `translate` 平移，因此这里不减 scrollTopPx。
 * - 宽度与淡入淡出必须分别用 `durationToWidthPx` / `secToSpanPx`：
 *   前者带最小宽度下限（保证可命中），后者可为 0（0 淡入不应出现角标）。
 *
 * @param args.axis 统一坐标投影（唯一的时间↔像素来源）。
 * @param args.visibleTracks 窗口内可见轨道，决定绘制顺序与轨道配色。
 * @param args.startTrackIndex 窗口首行的绝对轨道索引，用于算竖直内容坐标。
 * @param args.visibleTrackClipsById 各轨道的可见 clip。
 * @param args.rowHeight 轨道行高（CSS 像素）。
 * @param args.selectedClipId / multiSelectedClipIds / renamingClipId /
 *        hoveredClipId / disabledGroupIds 影响选中态与可见性判定。
 * @returns `drawClips` 供内核构建 GL 实例；`overlayClipIdsByTrackId` 是旧 DOM 覆盖层
 *          分工的遗留产物（内核模式下无生产消费者，见文件头说明）。
 */
export function buildSparseClipRenderModel(args: {
    visibleTracks: Array<{ id: string; color?: string }>;
    /** 窗口首行的绝对轨道索引：clip body 画布使用内容绝对坐标绘制，
     * 竖直滚动时由 scrollTopPx 统一平移（与 DOM 内容层同帧提交）。 */
    startTrackIndex: number;
    visibleTrackClipsById: Record<string, SparseRenderClip[]>;
    axis: TimelineAxis;
    rowHeight: number;
    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    renamingClipId: string | null;
    hoveredClipId?: string | null;
    disabledGroupIds?: string[];
    /**
     * 每个 clip 在"前导重叠区"（被同轨前一个 clip 压住的部分）的秒数。
     * 渲染端据此在上 clip 重叠区画半透色块，让下 clip 的色块与波形都能看见——
     * 否则两层不透明色块会"叠加"成脏色。
     */
    leadingOverlapSecByClipId?: Record<string, number>;
    /**
     * 静音检测预览区段：clip id → 工程秒的 `[起, 止]` 区间数组。
     *
     * 来源是 `session.silencePreviewSegments`（静音检测对话框的实时预览），
     * 由细节层画成半透明红色。
     */
    silenceSegmentsByClipId?: Record<string, ReadonlyArray<readonly [number, number]>>;
    /**
     * 是否平铺显示全部 Take（`session.showAllTakes`）。
     *
     * 关闭时每个 clip 只画活跃 Take 的波形（波形面自行消费同一设置），lane 分界线
     * 也随之消失。缺省 false（与「不传即不画」的保守默认一致）。
     */
    showAllTakes?: boolean;
    /**
     * 后端推送的 per-clip 音高曲线（Redux `session.clipPitchCurves`）。
     *
     * **兜底数据源**：`clip.midiNoteData` 缺失时（后端 lite 轮询载荷会剥离
     * `midi_note_data`，见 `to_payload_lite`）由这里提供曲线。
     *
     * 【当前接线状态】生产调用方 `kernel/host/timelineKernelHost` **尚未传本参数**
     * （该文件不在本次改动范围内）——因此当前生效的是 `midiNoteData` 主路径。
     * 该兜底通道已具备完整的取值/投影/降级逻辑，接线时只需把
     * `session.clipPitchCurves` 透传进来即可，无需改动本文件其余逻辑。
     * 键为 clip id；缺省 = 无兜底数据（不绘制）。
     */
    clipPitchCurves?: Record<
        string,
        { curveStartSec: number; midiCurve: number[]; framePeriodMs: number }
    >;
    /**
     * per-clip 音高值域（Redux `session.clipPitchRanges`），决定 y 映射的上下界。
     *
     * 缺省 `0..127`（与搬迁前 `MidiPitchTrackCanvas` 的缺省一致，也正是后端给
     * 音高参考块写入的 `pitch_range`）。**注意**：Redux 里音频 clip 的值域是
     * 半音偏移 `-24..24`，与音高参考块的绝对音高值域语义不同——只对
     * `midiNoteCount != null` 的 clip 取值才不会错配。
     */
    clipPitchRanges?: Record<string, { min: number; max: number }>;
}): {
    drawClips: TimelineCanvasClipModel[];
    overlayClipIdsByTrackId: Record<string, string[]>;
    /**
     * 本次模型里「激活的编组」集合（由选中集合 + `disabledGroupIds` 推出）。
     *
     * 供绘制端画编组外圈描边、GL 侧算样式——调用方不应自行重算，否则两处判据会分叉。
     */
    activeGroupIds: Set<string>;
} {
    const overlayClipIds = new Set<string>();
    if (args.renamingClipId) {
        overlayClipIds.add(args.renamingClipId);
    }
    if (args.hoveredClipId) {
        overlayClipIds.add(args.hoveredClipId);
    }
    if (args.multiSelectedClipIds.length > 0) {
        for (const clipId of args.multiSelectedClipIds) {
            overlayClipIds.add(clipId);
        }
    } else if (args.selectedClipId) {
        overlayClipIds.add(args.selectedClipId);
    }

    // Expand overlay to include all clips that share a group with any overlay clip,
    // unless the group is disabled.
    //
    // 集合提到外层：调用方（渲染内核）还需要它来画「编组激活的深金外圈描边」
    // （`drawTimelineCanvas` 的 `activeGroupIds`）与 GL 侧样式。在这里重算一份
    // 会让描边与 overlay 展开的判据分叉。
    const activeGroupIds = new Set<string>();
    {
        for (const trackClips of Object.values(args.visibleTrackClipsById)) {
            for (const clip of trackClips) {
                if (
                    clip.groupId != null &&
                    overlayClipIds.has(clip.id) &&
                    !args.disabledGroupIds?.includes(clip.groupId)
                ) {
                    activeGroupIds.add(clip.groupId);
                }
            }
        }
        if (activeGroupIds.size > 0) {
            for (const trackClips of Object.values(args.visibleTrackClipsById)) {
                for (const clip of trackClips) {
                    if (clip.groupId != null && activeGroupIds.has(clip.groupId)) {
                        overlayClipIds.add(clip.id);
                    }
                }
            }
        }
    }

    const multiSelectedSet =
        args.multiSelectedClipIds.length > 0 ? new Set(args.multiSelectedClipIds) : null;

    // 重叠区可编辑性：把“同轨道存在重叠”的两个 clip 都加入 DOM overlay。
    // 否则重叠时只有“后绘制/选中的那个”有 DOM 边缘/淡入淡出手柄，较早 clip 的
    // 边缘（延长截短/拉伸/淡入淡出）完全不可达。配合 ClipItem 去掉会建立独立层叠
    // 上下文的 transform，交叉处两个 clip 的手柄都位于其它 clip body 之上、可被编辑。
    for (const trackClips of Object.values(args.visibleTrackClipsById)) {
        if (trackClips.length < 2) continue;
        // 排序后线性扫描（原实现是 O(n²) 全对比较）。
        //
        // 判据不变：仍把每一对**存在重叠**的两个 clip 都加入 overlay。
        // 按 startSec 升序后，对固定的 a 而言，一旦某个 b 的起点越过了 a 的
        // 末端，它后面所有 clip 的起点只会更靠右，必然也不与 a 重叠 ——
        // 因此可以直接 `break`，把内层从「与 a 之后的全部 clip 比较」收敛为
        // 「与 a 真正相交的那一段」。
        //
        // 复杂度：排序 O(n log n) + 扫描 O(n + 重叠对数)。单轨 400 clip 时
        // 原实现固定 8 万次比较；首尾相接的常见排布下重叠对数为 0，扫描退化为
        // O(n)。必须复制后再排序——入参数组是上游 memo 的缓存值，就地排序会
        // 破坏缓存引用并让下游 memo 每帧失效。
        const sorted = trackClips.slice().sort((a, b) => a.startSec - b.startSec);
        for (let i = 0; i < sorted.length; i += 1) {
            const a = sorted[i];
            const aStart = a.startSec;
            const aEnd = aStart + a.lengthSec;
            for (let j = i + 1; j < sorted.length; j += 1) {
                const b = sorted[j];
                const bStart = b.startSec;
                // 升序 ⇒ 越过后不可能再重叠，提前跳出内层。
                if (bStart >= aEnd - 1e-9) break;
                const bEnd = bStart + b.lengthSec;
                if (Math.min(aEnd, bEnd) > bStart + 1e-9) {
                    overlayClipIds.add(a.id);
                    overlayClipIds.add(b.id);
                }
            }
        }
    }

    const drawClips = args.visibleTracks.flatMap((track, visibleIndex) =>
        (args.visibleTrackClipsById[track.id] ?? []).map((clip) => ({
            id: clip.id,
            trackId: clip.trackId,
            name: clipDisplayName(clip),
            leftPx: secToContentPx(args.axis, clip.startSec),
            topPx: (args.startTrackIndex + visibleIndex) * args.rowHeight,
            widthPx: durationToWidthPx(args.axis, clip.lengthSec),
            heightPx: Math.max(1, args.rowHeight - CLIP_BODY_PADDING_Y),
            headerHeightPx: CLIP_HEADER_HEIGHT,
            fadeInPx: secToSpanPx(
                args.axis,
                (clip.autoFadeInSec ?? 0) > 0 ? clip.autoFadeInSec! : clip.fadeInSec,
            ),
            fadeOutPx: secToSpanPx(
                args.axis,
                (clip.autoFadeOutSec ?? 0) > 0 ? clip.autoFadeOutSec! : clip.fadeOutSec,
            ),
            fadeInShape: Number.isFinite(clip.fadeInShape) ? clip.fadeInShape : 0,
            fadeOutShape: Number.isFinite(clip.fadeOutShape) ? clip.fadeOutShape : 0,
            fadeInDir: clip.fadeInDir ?? 0,
            fadeOutDir: clip.fadeOutDir ?? 0,
            selected:
                multiSelectedSet != null
                    ? multiSelectedSet.has(clip.id)
                    : args.selectedClipId === clip.id,
            muted: clip.muted,
            gain: clip.gain,
            playbackRate: clip.playbackRate,
            groupId: clip.groupId,
            isMidiClip: clip.midiNoteCount != null,
            trackColor: track.color,
            isRenaming: clip.id === args.renamingClipId,
            snapOffsetPx: secToSpanPx(args.axis, Number(clip.snapOffsetSec) || 0),
            leadingOverlapPx: secToSpanPx(
                args.axis,
                args.leadingOverlapSecByClipId?.[clip.id] ?? 0,
            ),
            silenceSpansPx: buildSilenceSpansPx({
                axis: args.axis,
                clipStartSec: clip.startSec,
                clipLengthSec: clip.lengthSec,
                segments: args.silenceSegmentsByClipId?.[clip.id],
            }),
            takeLaneSeparatorOffsetsPx: buildTakeLaneSeparatorOffsets({
                clip,
                showAllTakes: args.showAllTakes === true,
                bodyHeightPx: Math.max(
                    1,
                    args.rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
                ),
            }),
            midiPitchCurvePx: buildMidiPitchCurvePx({
                axis: args.axis,
                clip,
                bodyHeightPx: Math.max(1, args.rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT),
                pitchCurve: args.clipPitchCurves?.[clip.id],
                pitchRange: args.clipPitchRanges?.[clip.id],
            }),
            // 描边色只在真有折线时才产出（与「缺省即不画」的约定一致，
            // 不为音频 clip 白写一个字段）。
            midiPitchStroke:
                clip.midiNoteCount != null ? resolveMidiPitchStrokeColor(clip) : undefined,
            midiLoopMarkerOffsetsPx: buildMidiLoopMarkerOffsetsPx({
                axis: args.axis,
                clip,
            }),
        })),
    );

    const overlayClipIdsByTrackId = Object.fromEntries(
        args.visibleTracks.map((track) => [
            track.id,
            (args.visibleTrackClipsById[track.id] ?? [])
                .filter((clip) => overlayClipIds.has(clip.id))
                .map((clip) => clip.id),
        ]),
    ) as Record<string, string[]>;

    return {
        drawClips,
        overlayClipIdsByTrackId,
        activeGroupIds,
    };
}
