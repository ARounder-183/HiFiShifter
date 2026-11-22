/**
 * MIDI 音符 -> Clip 局部时间块的唯一几何实现。
 *
 * 时间轴绘制、命中密度与后续 piano-roll 预览都必须走这里，否则
 * “Loop 重复 / 倒放镜像 / 速率缩放 / 裁剪窗口”会在几个组件里各算一份。
 */
import { modEuclid, resolvePlaybackWindowSec } from "../../../../utils/loopRender";

export interface MidiNoteLike {
    startSec: number;
    endSec: number;
    note: number;
    velocity?: number;
    channel?: number;
    muted?: boolean;
}

export interface MidiNoteOccurrence {
    startSec: number;
    endSec: number;
    note: number;
    velocity: number;
    channel: number;
    muted: boolean;
}

export interface MidiNoteLoopCycle {
    cycleSec: number;
    fwdAnchorSec: number;
    revAnchorEndSec: number;
}

/**
 * 把源域音符铺到 Clip 局部时间。
 *
 * - 非 Loop：按真实消费窗口投影，倒放从窗口末端反向镜像；
 * - Loop：每个音符按回绕锚点求首个出现位置，再以 cycle/rate 重复；
 *   横跨回绕边界的音符保持为一段连续块（听到的也是连续尾音）。
 */
export function buildMidiNoteOccurrences(args: {
    notes: ReadonlyArray<MidiNoteLike>;
    clipLengthSec: number;
    playbackRate: number;
    reversed: boolean;
    loopEnabled: boolean;
    sourceStartSec: number;
    sourceEndSec: number;
    loopCycle?: MidiNoteLoopCycle | null;
    maxOccurrences?: number;
}): MidiNoteOccurrence[] {
    const clipLength = Math.max(0, Number(args.clipLengthSec) || 0);
    if (clipLength <= 1e-9 || args.notes.length === 0) return [];
    const rateRaw = Math.abs(Number(args.playbackRate) || 1);
    const rate = rateRaw > 1e-6 ? rateRaw : 1;
    const loopCycle = args.loopEnabled ? args.loopCycle : null;
    const cycle = loopCycle && loopCycle.cycleSec > 1e-9 ? loopCycle.cycleSec : null;
    const period = cycle != null ? cycle / rate : 0;
    const window = resolvePlaybackWindowSec({
        loopEnabled: Boolean(args.loopEnabled),
        reversed: args.reversed,
        sourceStartSec: Number(args.sourceStartSec) || 0,
        sourceEndSec: Number(args.sourceEndSec) || 0,
        playbackRate: rate,
        lengthSec: clipLength,
    });
    const maxOccurrences = Math.max(1, args.maxOccurrences ?? 50_000);

    const out: MidiNoteOccurrence[] = [];
    const push = (startSec: number, endSec: number, note: MidiNoteLike): boolean => {
        const start = Math.max(0, startSec);
        const end = Math.min(clipLength, endSec);
        if (end <= start + 1e-9) return true;
        out.push({
            startSec: start,
            endSec: end,
            note: Number(note.note),
            velocity: Math.max(0, Math.min(127, Number(note.velocity ?? 100) || 0)),
            channel: Math.max(0, Math.min(15, Number(note.channel ?? 0) || 0)),
            muted: Boolean(note.muted),
        });
        return out.length < maxOccurrences;
    };

    for (const note of args.notes) {
        const nStart = Number(note.startSec);
        const nEnd = Number(note.endSec);
        if (!Number.isFinite(nStart) || !Number.isFinite(nEnd) || nEnd <= nStart + 1e-9) {
            continue;
        }
        if (!Number.isFinite(Number(note.note))) continue;
        if (loopCycle && cycle != null) {
            const duration = (nEnd - nStart) / rate;
            const first = args.reversed
                ? modEuclid(loopCycle.revAnchorEndSec - nEnd, cycle) / rate
                : modEuclid(nStart - loopCycle.fwdAnchorSec, cycle) / rate;
            for (let start = first; start < clipLength - 1e-9; start += period) {
                if (!push(start, start + duration, note)) return out;
            }
            continue;
        }
        const start = args.reversed
            ? (window.winEndSec - nEnd) / rate
            : (nStart - window.winStartSec) / rate;
        const end = args.reversed
            ? (window.winEndSec - nStart) / rate
            : (nEnd - window.winStartSec) / rate;
        if (!push(start, end, note)) return out;
    }
    return out;
}
