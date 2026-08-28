/**
 * clipPitchDrag.ts - 音频块"垂直拖拽调音高"的纯计算。
 *
 * 交互语义：按住音高调整修饰键（默认 Alt+Shift）并上下拖拽 Clip 波形，
 * 将该 Clip 时间范围内（[startSec, startSec+lengthSec)）的 pitch 参数帧
 * 整体平移拖拽量对应的音分。
 *
 * pitch 参数帧模型（与钢琴卷帘/移调对话框一致）：
 * - 帧值为 MIDI 音高（浮点），音分偏移 = deltaCents / 100；
 * - 值 0 表示无声/未分析帧，移调时保持为 0（与 edit.transposeCents 相同）。
 */

/** 垂直拖拽灵敏度：音分 / 像素（50px = 1 半音）。 */
export const PITCH_DRAG_CENTS_PER_PX = 2;

/**
 * 把微调（advanceFineAxisDrag 输出的累计垂直位移）换算为音分偏移。
 * 向上拖拽（clientY 减小）→ 音高升高；0 位移归一化为 +0。
 */
export function computePitchDragCents(adjustedDeltaY: number): number {
    const cents = -adjustedDeltaY * PITCH_DRAG_CENTS_PER_PX;
    return cents === 0 ? 0 : cents;
}

/**
 * 对基准 pitch 帧数组应用整体音分偏移。
 * 无声帧（0）保持为 0；返回新数组，不改写输入。
 */
export function shiftPitchFrames(base: number[], deltaSemitones: number): number[] {
    if (deltaSemitones === 0) return base.slice();
    return base.map((v) => (v === 0 ? 0 : v + deltaSemitones));
}

/** 拖拽 ToolTips 的音分文本（与 formatGainDbValue 同风格）。 */
export function formatPitchDragCents(cents: number): string {
    const rounded = Math.round(cents * 10) / 10;
    if (Math.abs(rounded) < 0.05) return "0";
    const sign = rounded > 0 ? "+" : "";
    const value = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
    return `${sign}${value}`;
}
