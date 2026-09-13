/**
 * MIDI / 音高参考块的音高曲线数学（纯函数，无 React、无 DOM）。
 *
 * 【主要内容】把 MIDI 音符事件（`MidiNoteEvent[]`）铺放成**逐帧音高数组**，
 * 以及 Loop 回绕描述与回绕/媒体边界标记的 clip 局部时间求解。
 *
 * 【作用】时间线 clip 细节层据此在 MIDI / 音高参考块的 body 内画出音高折线
 * 与"▽"回绕标记（音频 clip 画波形，MIDI clip 画折线）。
 *
 * 【与其他模块的关系】
 * - 这些算法**逐字来自 `components/waveform/MidiPitchTrackCanvas.tsx`**（该组件
 *   随内核改造成为孤儿：唯一挂载点 `TrackLane` 被删除，曲线因此一度丢失）。
 *   **不要重写**：`generateMidiCurveFromNotes` 与后端
 *   `audio_engine::engine::emit_clip_pitch_data_for_clip` 的 MIDI 分支逐帧对齐
 *   （帧周期、trim 窗口、playbackRate 拉伸、reversed 镜像、Loop 锚点回绕、
 *   fillGaps 全部同一套数学）。任何"等价重写"都不会报错，只会让轨道折线与
 *   后端推送曲线 / 实际音频之间出现**恒定相位差**——极难归因。
 * - 回绕/边界标记的锚点数学与 `utils/loopRender.ts` 的 `modEuclid` /
 *   `resolveClipContentDurationSec` 是**同一套**（消费窗口的派生源终点
 *   `resolveSourceEndSec` 由模型侧 `timelineCanvasModel` 在调用本模块前求出）：
 *   周期 D 的取值链必须与波形分段、引擎音符放置保持一致（三条路径各推一份
 *   就会相位错开）。本模块只负责"求解标记位置"，绘制仍走
 *   `loopRender.drawLoopMarkers`。
 * - 上游消费者：`runtime/timelineCanvasModel.ts`（投影为像素）→
 *   `runtime/timelineCanvasRenderer.ts`（细节层落笔）。
 */

import { modEuclid, resolveClipContentDurationSec } from "../../../../utils/loopRender.js";

// ========================================
// 常量
// ========================================

/**
 * 音高曲线帧周期（毫秒）。
 *
 * **必须与后端 `emit_clip_pitch_data_for_clip` 的 `frame_period_ms = 5.0` 一致**：
 * 帧周期同时决定曲线的时间跨度（`length × fp`）与 x 投影，两侧不一致会让折线
 * 相对音频整体缩放错位。
 */
export const FRAME_PERIOD_MS = 5;

/**
 * clip 调色板色名 → 折线描边色（半透明，压在 clip 体色块上仍可辨认）。
 *
 * 色名集合与 `sessionTypes.ClipInfo["color"]` 一致；未知色名回退见
 * `strokeColorForClip`。
 */
export const CLIP_COLOR_TO_STROKE: Record<string, string> = {
    blue: "rgba(96, 165, 250, 0.85)",
    violet: "rgba(167, 139, 250, 0.85)",
    emerald: "rgba(52, 211, 153, 0.85)",
    amber: "rgba(251, 191, 36, 0.85)",
    cyan: "rgba(34, 211, 238, 0.85)",
};

// ========================================
// 工具函数
// ========================================

/**
 * 取 clip 的音高折线描边色。
 *
 * 流程：按 clip 自身的调色板色名查 `CLIP_COLOR_TO_STROKE`；未知 / 缺省色名
 * 回退为青色（与波形层缺省色一致的观感）。
 *
 * @param clip 只要求 `color` 字段（clip 调色板色名）。
 * @returns CSS 颜色串。
 */
export function strokeColorForClip(clip: { color: string }): string {
    return CLIP_COLOR_TO_STROKE[clip.color] ?? "rgba(34, 211, 238, 0.78)";
}

/**
 * Loop（循环源）回绕描述 —— 与后端 clip_loop_cycle_span_sec /
 * place_note_occurrence_frames 的锚点数学逐帧一致。
 */
export interface LoopCycleDescriptor {
    /** 回绕周期 D（源域秒）：音频 clip = 整个媒体文件时长；纯 MIDI clip = 窗口跨度。 */
    cycleSec: number;
    /** 正放锚点（原始 sourceStartSec，可为负，floor_mod 环绕）。 */
    fwdAnchorSec: number;
    /**
     * 倒放锚点末端：周期来自媒体时长时 clamp 到 D；周期退化为窗口跨度时
     * 保持原始 sourceEndSec（与后端 place_note_occurrence_in_loop 一致 ——
     * 否则 slip 窗口 [2,7] 的跨度 clamp 会把倒放相位错误平移 ss=2s）。
     */
    revAnchorEndSec: number;
    /**
     * 周期是否来自真实媒体时长：
     * - true：坐标是**文件域**，首个回绕点在 headDur = 进入段耗尽处；
     * - false（纯 MIDI 窗口跨度）：坐标是**窗口相对域**，入口即窗口起点，
     *   首个回绕点在一个完整窗口周期之后。
     */
    cycleFromMedia: boolean;
}

/**
 * 解析 Loop 回绕描述。
 *
 * 关键语义：所有可发声内容（带源媒体的 clip、以及纯音高参考块）的实际
 * 声音/音高按**整个内容** floor_mod 回绕（与 WaveformTrackCanvas / 后端
 * 引擎一致），音高曲线与回绕标记必须使用同一周期，否则相位错位；
 * contentDurationSec 为 null（连音符内容都无法确定）时退化为窗口跨度。
 *
 * @param args.loopEnabled 是否开启循环源。
 * @param args.contentDurationSec 内容时长（秒）：`resolveClipContentDurationSec`
 *        的结果；null = 退化。
 * @param args.sourceStartSec 存储的源起点（可为负）。
 * @param args.sourceEndSec 存储的源终点（可为负 / 超界）。
 * @returns 回绕描述；未开启循环或周期非正时为 null（调用方走非循环路径）。
 */
export function resolveLoopCycleDescriptor(args: {
    loopEnabled: boolean;
    /** 内容时长（秒）：resolveClipContentDurationSec 的结果；null = 退化。 */
    contentDurationSec: number | null;
    sourceStartSec: number;
    sourceEndSec: number;
}): LoopCycleDescriptor | null {
    if (!args.loopEnabled) return null;
    const mediaDur = args.contentDurationSec ?? 0;
    const windowSpan = Math.abs(Number(args.sourceEndSec ?? 0) - Number(args.sourceStartSec ?? 0));
    const cycleSec = mediaDur > 1e-9 ? mediaDur : windowSpan;
    if (!Number.isFinite(cycleSec) || cycleSec <= 1e-9) return null;
    const srcStart = Number(args.sourceStartSec ?? 0);
    const srcEnd = Number(args.sourceEndSec ?? 0);
    return {
        cycleSec,
        fwdAnchorSec: srcStart,
        // 与后端 place_note_occurrence_in_loop / trim_and_resample_midi 同约定：
        // 倒放锚点只 clamp 到媒体时长上界、**不做 max(0)** —— 负 source_end
        // （slip/左延伸可达）由消费端的 modEuclid（floor_mod）统一环绕；
        // 此处若钳到 0 会让曲线/标记与音频出现恒定相位差。
        revAnchorEndSec: mediaDur > 1e-9 ? Math.min(srcEnd, mediaDur) : srcEnd,
        cycleFromMedia: mediaDur > 1e-9,
    };
}

/**
 * 从 MIDI note data 即时生成音高曲线。
 * 逻辑与后端 emit_clip_pitch_data_for_clip 的 MIDI 分支一致，
 * 支持 source range trim、playbackRate 拉伸、reversed 倒放，
 * 以及 Loop（循环源）：按媒体时长的锚点回绕重复铺满
 * （`loopCycle` 为 null 时走非循环路径）。
 *
 * 流程：先按 `clipLengthSec / FRAME_PERIOD_MS` 分配目标帧数组（0 = 无音高），
 * 再逐音符把所有命中帧写为音高值（同帧取更高音：`noteValue > curve[frame]`），
 * 最后可选 fillGaps 用前一个有效音高填满首尾之间的空隙。
 *
 * 特殊说明：
 * - 帧索引一律 `Math.round(sec / pr * 1000 / fp)`，与后端**同一取整方式**；
 *   改成 floor/ceil 会引入最多半帧的相位差。
 * - Loop 分支按 `cycleFrames` 步进重复写入，且**不能**用窗口比较过滤音符：
 *   split 产生的"环绕窗口"（start > end）会把所有音符误判为越界而全部丢弃。
 * - 非 Loop 分支的可见性以 `[sourceStartSec, sourceEndSec)` 为准；倒放在窗口内
 *   做镜像（`srcTotalLen - rel`），与后端一致。
 *
 * @param notes 音符事件（startSec / endSec / note）。
 * @param clipLengthSec clip 长度（秒）→ 决定输出帧数。
 * @param sourceStartSec 消费窗口起点（源域秒）——非 Loop 正放 = 起点锚定；
 *        倒放由调用方传入 `[se − len·r, se]` 的派生窗口起点。
 * @param sourceEndSec 消费窗口终点（源域秒）。
 * @param playbackRate 播放倍率（非有限值 / <=0 时按 1 处理）。
 * @param reversed 是否倒放。
 * @param fillGaps 是否用前一个有效音高填满音符之间的空隙（后端
 *        `fill_gaps_in_pitch_edit` 同逻辑）。
 * @param loopCycle Loop 回绕描述；null = 非循环。
 * @returns 逐帧音高数组（0 表示该帧无音高）。
 */
export function generateMidiCurveFromNotes(
    notes: Array<{ startSec: number; endSec: number; note: number }>,
    clipLengthSec: number,
    sourceStartSec: number,
    sourceEndSec: number,
    playbackRate: number,
    reversed: boolean,
    fillGaps: boolean,
    loopCycle: LoopCycleDescriptor | null,
): number[] {
    const fp = Math.max(FRAME_PERIOD_MS, 0.1);
    const targetFrames = Math.max(1, Math.round((clipLengthSec * 1000) / fp));
    const curve = new Array<number>(targetFrames).fill(0);

    const pr = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;
    const srcTotalLen = sourceEndSec - sourceStartSec;

    for (const note of notes) {
        // Loop（循环源）：按媒体时长 D 的锚点回绕放置（与音频渲染的
        // floor_mod 映射一致）。不能用窗口比较过滤可见性 —— split 产生的
        // "环绕窗口"（start > end）会把所有音符误判为越界而全部丢弃。
        if (loopCycle && note.endSec - note.startSec > 1e-9) {
            const { cycleSec, fwdAnchorSec, revAnchorEndSec } = loopCycle;
            const u0 = reversed
                ? modEuclid(revAnchorEndSec - note.endSec, cycleSec)
                : modEuclid(note.startSec - fwdAnchorSec, cycleSec);
            const firstStartFrame = Math.round(((u0 / pr) * 1000) / fp);
            const lenFrames = Math.max(
                1,
                Math.round((((note.endSec - note.startSec) / pr) * 1000) / fp),
            );
            const cycleFrames = Math.max(1, Math.round(((cycleSec / pr) * 1000) / fp));
            const noteValue = note.note;
            for (let cycleOffset = 0; cycleOffset < targetFrames; cycleOffset += cycleFrames) {
                const writeStart = cycleOffset + firstStartFrame;
                const writeEnd = Math.min(cycleOffset + firstStartFrame + lenFrames, targetFrames);
                if (writeStart >= writeEnd) break;
                for (let frame = writeStart; frame < writeEnd; frame++) {
                    if (noteValue > curve[frame] || curve[frame] <= 0) {
                        curve[frame] = noteValue;
                    }
                }
            }
            continue;
        }

        if (note.endSec <= sourceStartSec || note.startSec >= sourceEndSec) continue;
        const relStart = Math.max(0, note.startSec - sourceStartSec);
        const relEnd = Math.min(srcTotalLen, note.endSec - sourceStartSec);
        if (relEnd <= relStart) continue;

        const [effStart, effEnd] = reversed
            ? [Math.max(0, srcTotalLen - relEnd), Math.min(srcTotalLen, srcTotalLen - relStart)]
            : [relStart, relEnd];
        if (effEnd <= effStart) continue;

        const noteStartFrame = Math.round(((effStart / pr) * 1000) / fp);
        const noteEndFrame = Math.round(((effEnd / pr) * 1000) / fp);
        const noteValue = note.note;

        // 非 Loop：单次写入（Loop 已在上方 placement 分支处理）。
        const writeEnd = Math.min(noteEndFrame, targetFrames);
        for (let frame = noteStartFrame; frame < writeEnd; frame++) {
            if (noteValue > curve[frame] || curve[frame] <= 0) {
                curve[frame] = noteValue;
            }
        }
    }

    // 填补音符之间的空隙（与后端 fill_gaps_in_pitch_edit 逻辑一致）
    if (fillGaps && curve.length > 0) {
        let first = -1;
        for (let i = 0; i < curve.length; i++) {
            if (curve[i] > 0) {
                first = i;
                break;
            }
        }
        let last = -1;
        for (let i = curve.length - 1; i >= 0; i--) {
            if (curve[i] > 0) {
                last = i;
                break;
            }
        }
        if (first >= 0 && last > first) {
            let lastPitch = 0;
            for (let i = first; i <= last; i++) {
                if (curve[i] > 0) {
                    lastPitch = curve[i];
                } else if (lastPitch > 0) {
                    curve[i] = lastPitch;
                }
            }
        }
    }

    return curve;
}

/**
 * 曲线生成缓存（F6 性能修复）：
 * generateMidiCurveFromNotes 在每个绘制帧对每个可见 MIDI clip 全量重算 ——
 * 分配 lengthSec×200 的数组（10000s clip ⇒ 2M 元素），且 Loop 分支按重复
 * 周期放大写入量。绘制帧之间同一 clip 的（notes 引用, 几何参数）几乎总是
 * 不变，直接复用上次结果即可。
 *
 * 缓存键：notes 数组**引用**（WeakMap 随 clip 释放，无泄漏）+ 几何参数串。
 * 每个 notes 引用最多保留 4 份几何变体（拖拽编辑时的中间态）。
 */
const midiCurveCache = new WeakMap<
    Array<{ startSec: number; endSec: number; note: number }>,
    Map<string, number[]>
>();
export const MIDI_CURVE_CACHE_MAX_PER_NOTES = 4;

/**
 * 带缓存的 `generateMidiCurveFromNotes`。
 *
 * 流程：按 notes 引用取内层 Map → 拼几何参数键（长度 / 窗口 / 速率 / 倒放 /
 * fillGaps / 回绕描述）→ 命中即返回，未命中则计算并按 LRU（插入序淘汰最早一份）
 * 压到每 notes 最多 `MIDI_CURVE_CACHE_MAX_PER_NOTES` 份。
 *
 * 特殊说明：键里**必须**包含全部影响结果的参数——漏掉任一项（典型是
 * `fillGaps` 或 `loopCycle`）会让拖拽中的中间态复用上一次的曲线。
 *
 * @returns 逐帧音高数组（可能是缓存里的同一引用，调用方不得就地改写）。
 */
export function getCachedMidiCurve(
    notes: Array<{ startSec: number; endSec: number; note: number }>,
    clipLengthSec: number,
    sourceStartSec: number,
    sourceEndSec: number,
    playbackRate: number,
    reversed: boolean,
    fillGaps: boolean,
    loopCycle: LoopCycleDescriptor | null,
): number[] {
    let inner = midiCurveCache.get(notes);
    if (!inner) {
        inner = new Map();
        midiCurveCache.set(notes, inner);
    }
    const key =
        `${clipLengthSec}|${sourceStartSec}|${sourceEndSec}|${playbackRate}|` +
        `${reversed ? 1 : 0}|${fillGaps ? 1 : 0}|` +
        (loopCycle
            ? `${loopCycle.cycleSec}|${loopCycle.fwdAnchorSec}|${loopCycle.revAnchorEndSec}|${loopCycle.cycleFromMedia ? 1 : 0}`
            : "null");
    const hit = inner.get(key);
    if (hit) return hit;
    const curve = generateMidiCurveFromNotes(
        notes,
        clipLengthSec,
        sourceStartSec,
        sourceEndSec,
        playbackRate,
        reversed,
        fillGaps,
        loopCycle,
    );
    if (inner.size >= MIDI_CURVE_CACHE_MAX_PER_NOTES) {
        const oldest = inner.keys().next().value;
        if (oldest !== undefined) inner.delete(oldest);
    }
    inner.set(key, curve);
    return curve;
}

/**
 * Loop 回绕标记 / 媒体边界标记的 **clip 局部时间**（秒，相对 clip 起点）。
 *
 * 流程（与搬迁前 `MidiPitchTrackCanvas` 的标记段逐行一致）：
 * 1. 用 `resolveClipContentDurationSec` 取内容时长 D，解析"曲线用"源终点
 *    （`resolveSourceEndSec`：非 Loop 正放为 `ss + len·r` 派生窗口）；
 * 2. **Loop**：周期 `D/|rate|` 的每个回绕点。头部进入段耗尽处 headDur
 *    （周期来自媒体时 = `(D − floor_mod(anchor, D)) / rate`，倒放取锚点末端；
 *    退化为窗口跨度时 = 一个完整周期）及此后每个整周期；
 * 3. **非 Loop**：源媒体 / 音符内容在本 clip 内的真实起止位置
 *    （`b ∈ {0, D}` 按消费方向投影：正放 `(b − ss)/r`、倒放 `(se − b)/r`）；
 * 4. 两端点（`t ≈ 0` 与 `t ≈ lengthSec`）不产出——与 `loopRender` 的
 *    "恰好在 clip 起点/终点的回绕点不绘制"约定一致。
 *
 * 特殊说明：
 * - 一律用 `modEuclid`（floor_mod）而不是 clamp 归一化锚点：存储在域外的
 *   锚点（slip / 左延伸）必须环绕，否则标记与音频相位错开。
 * - `fromLocalSec` 供调用方跳过视口左侧的整周期（避免长循环 clip 逐周期空转）；
 *   默认 0 = 从 clip 入口开始，与搬迁前传视图起点时的行为一致。
 * - 标记数量上限 4096（与搬迁前同值），防止极端循环周期下产出爆炸。
 *
 * @param args.loopEnabled 是否循环源。
 * @param args.reversed 是否倒放。
 * @param args.lengthSec clip 长度（秒）。
 * @param args.playbackRate 播放倍率（非有限值 / 0 按 1）。
 * @param args.sourceStartSec / args.sourceEndSec 存储的源窗口（可为负 / 超界）。
 * @param args.sourcePath / args.midiNoteData / args.durationFrames /
 *        args.sourceSampleRate / args.durationSec 内容时长 D 的取值链输入
 *        （与 `resolveClipContentDurationSec` 同签名）。
 * @param args.fromLocalSec 只从该 clip 局部时间开始产出（默认 0）。
 * @returns clip 局部秒数组（升序）；无标记时为空数组。
 */
export function resolveClipLoopMarkerOffsetsSec(args: {
    loopEnabled: boolean;
    reversed: boolean;
    lengthSec: number;
    playbackRate: number;
    sourceStartSec: number;
    sourceEndSec: number;
    sourcePath?: string | null;
    midiNoteData?: ReadonlyArray<{ endSec: number }> | null;
    durationFrames?: number | null;
    sourceSampleRate?: number | null;
    durationSec?: number | null;
    fromLocalSec?: number;
}): number[] {
    const lengthSec = Math.max(0, Number(args.lengthSec) || 0);
    if (!(lengthSec > 1e-9)) return [];
    const markers: number[] = [];
    const contentDurSec = resolveClipContentDurationSec({
        sourcePath: args.sourcePath,
        midiNoteData: args.midiNoteData ?? null,
        durationFrames: args.durationFrames,
        sourceSampleRate: args.sourceSampleRate,
        durationSec: args.durationSec,
    });
    const rateRaw = Math.abs(Number(args.playbackRate ?? 1) || 1);
    const rate = rateRaw < 1e-6 ? 1 : rateRaw;

    // ── Loop：按回绕周期铺标记 ──
    // 周期与曲线平铺一致：内容时长 D（标记描述用**存储**的 sourceEndSec，
    // 与曲线消费窗口的派生 se 是两回事，见搬迁前的注释）。
    const markerCycle = resolveLoopCycleDescriptor({
        loopEnabled: Boolean(args.loopEnabled),
        contentDurationSec: contentDurSec,
        sourceStartSec: args.sourceStartSec,
        sourceEndSec: Number(args.sourceEndSec) || 0,
    });
    const markerBodyDur = markerCycle ? markerCycle.cycleSec / rate : 0;
    if (markerBodyDur > 0 && lengthSec > markerBodyDur + 1e-6) {
        // 标记必须锚定在**实际回绕点**：与 WaveformTrackCanvas 的分段边界一致 ——
        // - 周期来自媒体：头部进入段耗尽处（headDur）及此后每个整文件周期边界；
        // - 纯 MIDI 窗口跨度（窗口相对域）：入口即窗口起点，首个回绕点在一个
        //   完整周期之后。
        // 不能一律用 k·周期：窗口不从文件原点进入时（trim/split），标记会与
        // 音频/曲线相位错开。
        const desc = markerCycle;
        const headDur = !desc
            ? 0
            : desc.cycleFromMedia
              ? (args.reversed
                    ? desc.revAnchorEndSec
                    : desc.cycleSec - modEuclid(desc.fwdAnchorSec, desc.cycleSec)) / rate
              : markerBodyDur;
        // 直接跳到起点之后的第一个回绕点：既避免从 clip 入口逐周期空转数千次，
        // 也修复"深入长循环 clip 后标记消失"（旧实现受 guard<8192 限制）。
        const fromLocalSec = Math.max(0, Number(args.fromLocalSec ?? 0) || 0);
        const k0 = Math.max(
            0,
            Math.ceil((fromLocalSec - headDur - 1e-6) / markerBodyDur),
        );
        for (
            let markerT = headDur + k0 * markerBodyDur;
            markerT < lengthSec - 1e-6 && markers.length < 4096;
            markerT += markerBodyDur
        ) {
            // 恰好在 clip 起点的回绕点不绘制（loopRender 约定；倒放整文件
            // Loop 的 revAnchor=D 时 headDur=0 会命中）。
            if (markerT <= 1e-6) continue;
            markers.push(markerT);
        }
        return markers;
    }

    // ── 非 Loop：媒体/内容边界标记 ──
    // 循环节 = 源媒体（或音符内容）在该 Clip 内的真实起始/终止位置（音频与
    // 静音的分界线），落在 Clip 内部时绘制。投影按**消费方向**：正放
    // t=(b−ss)/r；倒放 t=(se−b)/r（倒放锚定窗口终点，与 WaveformTrackCanvas
    // 一致）。
    if (contentDurSec == null) return markers;
    const srcEndResolved = Number(args.sourceEndSec) || 0;
    for (const b of [0, contentDurSec]) {
        const tLocal = args.reversed
            ? (srcEndResolved - b) / rate
            : (b - (Number(args.sourceStartSec) || 0)) / rate;
        if (tLocal <= 1e-6 || tLocal >= lengthSec - 1e-6) continue;
        markers.push(tLocal);
    }
    return markers;
}
