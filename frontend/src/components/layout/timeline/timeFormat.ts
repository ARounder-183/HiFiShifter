/**
 * 时间轴时间单位格式化与自适应标尺刻度。
 *
 * 支持四种时间单位：
 * - barBeats     小节.节拍.小单位（1000 小单位 = 1 节拍）
 * - barDivisions 小节.切分（切分由“网格”设置决定）
 * - seconds      绝对秒
 * - clock        时:分:秒.毫秒
 */
import type { TimeUnit, TimeUnitChoice } from "../../../features/session/sessionTypes.ts";
import { gridStepBeats } from "./grid.ts";

export type { TimeUnit, TimeUnitChoice } from "../../../features/session/sessionTypes.ts";

export const TIME_UNITS: readonly TimeUnit[] = [
    "barBeats",
    "barDivisions",
    "seconds",
    "clock",
] as const;

export const TIME_UNIT_CHOICES: readonly TimeUnitChoice[] = [
    "none",
    ...TIME_UNITS,
] as const;

export interface TimeFormatContext {
    bpm: number;
    beatsPerBar: number;
    grid: string;
}

export interface RulerTick {
    /** 拍数（1 拍 = 四分音符）。 */
    beat: number;
    /** 秒数。 */
    sec: number;
    /** 主时间单位标签。 */
    primaryLabel: string;
    /** 副时间单位标签；副单位为“不使用”或与主单位相同时为 null。 */
    secondaryLabel: string | null;
    /** 是否为小节起始位置。 */
    isBarStart: boolean;
}

const BEATS_PER_BAR_DEFAULT = 4;

function normalizeBeatsPerBar(beatsPerBar: number): number {
    return Math.max(1, Math.round(beatsPerBar || BEATS_PER_BAR_DEFAULT));
}

export function beatFromSec(sec: number, bpm: number): number {
    return (Math.max(0, sec) * Math.max(1, bpm)) / 60;
}

export function secFromBeat(beat: number, bpm: number): number {
    return (Math.max(0, beat) * 60) / Math.max(1, bpm);
}

/**
 * 小单位 = 直接舍去小数部分后的整数（保留三位，前补 0）。
 * 标尺模式：舍去后与原值不完全相等时追加 `..`。
 * 光标模式：同样只保留三位整数，不追加 `..`。
 */
function formatSubdivision(sub: number, mode: "ruler" | "cursor"): string {
    const truncated = Math.floor(sub + 1e-9);
    const text = String(truncated).padStart(3, "0");
    if (mode === "cursor") return text;
    const exact = Math.abs(truncated - sub) < 1e-6;
    return exact ? text : `${text}..`;
}

/**
 * 小节.节拍.小单位
 *
 * 例：
 * - 2.1（小节起始）
 * - 1.3（2 分音符位置）
 * - 1.1.500（8 分音符位置）
 * - 1.1.333..（三连音位置，小单位经过舍去并不精确）
 */
export function formatBarBeatsLabel(
    beat: number,
    beatsPerBar = BEATS_PER_BAR_DEFAULT,
    mode: "ruler" | "cursor" = "ruler",
): string {
    const bpb = normalizeBeatsPerBar(beatsPerBar);
    const safeBeat = Math.max(0, beat);
    let barIndex = Math.floor(safeBeat / bpb);
    const inBarBeat = safeBeat - barIndex * bpb;
    let beatIndex = Math.floor(inBarBeat);
    const frac = inBarBeat - beatIndex;
    if (Math.abs(frac) < 1e-9) {
        if (mode === "cursor") {
            // 光标时间需要固定位数对齐，始终保留三位小单位。
            return `${barIndex + 1}.${beatIndex + 1}.000`;
        }
        return `${barIndex + 1}.${beatIndex + 1}`;
    }
    // 消除浮点误差：当拍位置已经无限接近下一拍时，按下一拍处理。
    if (frac > 1 - 1e-9) {
        beatIndex += 1;
        if (beatIndex >= bpb) {
            barIndex += 1;
            beatIndex = 0;
        }
        if (mode === "cursor") {
            return `${barIndex + 1}.${beatIndex + 1}.000`;
        }
        return `${barIndex + 1}.${beatIndex + 1}`;
    }
    return `${barIndex + 1}.${beatIndex + 1}.${formatSubdivision(frac * 1000, mode)}`;
}

/**
 * 单小节切分数。正常网格为整数；附点网格可能是循环小数。
 */
export function gridDivisionsPerBar(grid: string, beatsPerBar: number): number {
    const step = Math.max(1e-9, gridStepBeats(grid));
    return Math.max(1, normalizeBeatsPerBar(beatsPerBar) / step);
}

function formatDivisionCount(count: number): string {
    if (!Number.isFinite(count)) return "1";
    if (Math.abs(count - Math.round(count)) < 1e-9) {
        return String(Math.round(count));
    }
    return String(Number(count.toFixed(4)));
}

/**
 * 小节.切分
 *
 * 例：`1.17/32` 表示第 1 小节内，以 1/32 网格计的第 17 个切分。
 * 附点网格的“单小节切分数”可能是小数（如 2.6667），三连音网格通常是整数。
 */
export function formatBarDivisionsLabel(beat: number, ctx: TimeFormatContext): string {
    const bpb = normalizeBeatsPerBar(ctx.beatsPerBar);
    const step = Math.max(1e-9, gridStepBeats(ctx.grid));
    const safeBeat = Math.max(0, beat);
    const barIndex = Math.floor(safeBeat / bpb);
    const inBarBeat = safeBeat - barIndex * bpb;
    // 切分编号采用舍去（截断）：当前位置落在第几个网格切分内，就显示第几个。
    const index = Math.max(1, Math.floor(inBarBeat / step + 1e-9) + 1);
    const divisions = gridDivisionsPerBar(ctx.grid, bpb);
    return `${barIndex + 1}.${index}/${formatDivisionCount(divisions)}`;
}

function trimTrailingZeros(value: string): string {
    if (!value.includes(".")) return value;
    return value.replace(/0+$/, "").replace(/\.$/, "");
}

/**
 * 秒：标尺保留小数点后 4 位并采取舍去；不精确时追加 `..`。
 */
export function formatSecondsRuler(sec: number): string {
    const safeSec = Math.max(0, sec);
    const truncated = Math.floor(safeSec * 10000) / 10000;
    const text = trimTrailingZeros(String(truncated));
    const exact = Math.abs(safeSec - truncated) < 1e-9;
    return exact ? text : `${text}..`;
}

/**
 * 秒：光标位置保留小数点后 3 位并四舍五入，始终显示三位小数以对齐文本，不标记 `..`。
 */
export function formatSecondsCursor(sec: number): string {
    const rounded = Math.round(Math.max(0, sec) * 1000) / 1000;
    return rounded.toFixed(3);
}

/**
 * 时:分:秒.毫秒。小时为 0 时省略；毫秒始终 3 位，采用舍去。
 */
export function formatClockLabel(sec: number): string {
    const totalMs = Math.floor(Math.max(0, sec) * 1000);
    const ms = totalMs % 1000;
    const totalSec = Math.floor(totalMs / 1000);
    const seconds = totalSec % 60;
    const totalMin = Math.floor(totalSec / 60);
    const minutes = totalMin % 60;
    const hours = Math.floor(totalMin / 60);
    const msText = String(ms).padStart(3, "0");
    const body = `${minutes}:${seconds}.${msText}`;
    return hours > 0 ? `${hours}:${body}` : body;
}

/**
 * 将某个拍位置格式化为指定时间单位（标尺场景）。
 */
export function formatRulerTick(unit: TimeUnit, beat: number, ctx: TimeFormatContext): string {
    switch (unit) {
        case "barBeats":
            return formatBarBeatsLabel(beat, ctx.beatsPerBar, "ruler");
        case "barDivisions":
            return formatBarDivisionsLabel(beat, ctx);
        case "seconds":
            return formatSecondsRuler(secFromBeat(beat, ctx.bpm));
        case "clock":
            return formatClockLabel(secFromBeat(beat, ctx.bpm));
    }
}

/**
 * 将某个秒位置格式化为指定时间单位（光标场景，使用光标精度规则）。
 */
export function formatCursorUnit(unit: TimeUnit, sec: number, ctx: TimeFormatContext): string {
    switch (unit) {
        case "barBeats":
            return formatBarBeatsLabel(beatFromSec(sec, ctx.bpm), ctx.beatsPerBar, "cursor");
        case "barDivisions":
            return formatBarDivisionsLabel(beatFromSec(sec, ctx.bpm), ctx);
        case "seconds":
            return formatSecondsCursor(sec);
        case "clock":
            return formatClockLabel(sec);
    }
}

/**
 * 播放光标时间：`{主} / {副}`；副单位未使用或与主单位相同时仅显示主单位。
 */
export function formatCursorTime(
    primary: TimeUnit,
    secondary: TimeUnitChoice,
    sec: number,
    ctx: TimeFormatContext,
): { primaryLabel: string; secondaryLabel: string | null; combined: string } {
    const primaryLabel = formatCursorUnit(primary, sec, ctx);
    const secondaryLabel =
        secondary !== "none" && secondary !== primary
            ? formatCursorUnit(secondary, sec, ctx)
            : null;
    return {
        primaryLabel,
        secondaryLabel,
        combined: secondaryLabel ? `${primaryLabel} / ${secondaryLabel}` : primaryLabel,
    };
}

/**
 * 候选标尺刻度步长（拍）。升序排列，始终包含小节步长。
 *
 * 标准网格使用常用音符阶梯；附点/三连音网格额外加入网格步长及其倍增，
 * 使标签始终落在网格线上，同时支持逐步细化到用户设置的“网格”精度。
 * 若网格步长本身大于一个小节（如 `1/1d`），则一并加入网格步长的倍增。
 */
export function rulerStepCandidates(grid: string, beatsPerBar: number): number[] {
    const gridStep = Math.max(1e-9, gridStepBeats(grid));
    const bpb = normalizeBeatsPerBar(beatsPerBar);
    const set = new Set<number>();
    const standard = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4];
    for (const step of standard) {
        if (step >= gridStep - 1e-9) set.add(step);
    }
    set.add(bpb);
    const isStandard =
        standard.some((step) => Math.abs(step - gridStep) < 1e-9) ||
        Math.abs(gridStep - bpb) < 1e-9;
    if (!isStandard) {
        for (let multiplier = 1; multiplier <= 8; multiplier *= 2) {
            const step = gridStep * multiplier;
            if (step <= bpb + 1e-9) set.add(step);
        }
        if (gridStep > bpb + 1e-9) {
            for (let multiplier = 1; multiplier <= 16; multiplier *= 2) {
                set.add(gridStep * multiplier);
            }
        }
    }
    return [...set].sort((a, b) => a - b);
}

/**
 * 根据当前水平缩放自动选择标尺刻度步长（拍）。
 *
 * 规则：选择“间距不小于 minLabelSpacingPx”的最细候选步长；
 * 最大精度受“网格”限制；全部不满足（过度缩小）时回退到小节。
 */
export function selectRulerStep(args: {
    pxPerBeat: number;
    grid: string;
    beatsPerBar: number;
    minLabelSpacingPx: number;
}): number {
    const { pxPerBeat, grid, beatsPerBar, minLabelSpacingPx } = args;
    const candidates = rulerStepCandidates(grid, beatsPerBar);
    const spacing = Math.max(24, Math.min(600, minLabelSpacingPx));
    const gridStep = Math.max(1e-9, gridStepBeats(grid));
    for (const step of candidates) {
        if (step >= gridStep - 1e-9 && step * pxPerBeat >= spacing - 1e-9) {
            return step;
        }
    }
    // 缩小到候选阶梯之外时，按 2 的幂继续拉大间隔：
    // 每 2 个小节、每 4 个小节、每 8 个小节……以此类推，可持续无限放大。
    // 当网格步长本身大于一个小节时，以网格步长为基础倍增，保证标签仍落在网格线上。
    const bpb = normalizeBeatsPerBar(beatsPerBar);
    const baseUnit = gridStep > bpb + 1e-9 ? gridStep : bpb;
    let step = baseUnit;
    while (step * pxPerBeat < spacing - 1e-9) {
        step *= 2;
    }
    return step;
}

/**
 * 生成当前可见范围内的标尺刻度（主/副标签），并按拍升序返回。
 * 刻度始终落在网格线上；小节起始位置会额外补入并标记为强刻度。
 */
export function buildRulerTicks(args: {
    pxPerSec: number;
    scrollLeft: number;
    viewportWidth: number;
    projectSec: number;
    bpm: number;
    beatsPerBar: number;
    grid: string;
    primaryUnit: TimeUnit;
    secondaryUnit: TimeUnitChoice;
    minLabelSpacingPx: number;
}): RulerTick[] {
    const {
        pxPerSec,
        scrollLeft,
        viewportWidth,
        projectSec,
        bpm,
        beatsPerBar,
        grid,
        primaryUnit,
        secondaryUnit,
        minLabelSpacingPx,
    } = args;
    const bpb = normalizeBeatsPerBar(beatsPerBar);
    const secPerBeat = 60 / Math.max(1, bpm);
    const pxPerBeat = Math.max(1e-9, secPerBeat * Math.max(0, pxPerSec));
    const step = selectRulerStep({
        pxPerBeat,
        grid,
        beatsPerBar: bpb,
        minLabelSpacingPx,
    });
    const totalBeats = Math.max(0, projectSec / secPerBeat);
    const bufferPx = Math.max(320, Math.max(0, viewportWidth) * 0.5);
    const leftPx = Math.max(0, scrollLeft - bufferPx);
    const rightPx = scrollLeft + Math.max(0, viewportWidth) + bufferPx;
    const leftBeat = leftPx / pxPerBeat;
    const rightBeat = rightPx / pxPerBeat;
    const ctx: TimeFormatContext = { bpm, beatsPerBar: bpb, grid };
    const showSecondary =
        secondaryUnit !== "none" && secondaryUnit !== primaryUnit;
    const ticks = new Map<number, RulerTick>();

    const addTick = (beat: number, forcedBarStart: boolean) => {
        if (!Number.isFinite(beat) || beat < -1e-9 || beat > totalBeats + 1e-9) return;
        const key = Math.round(beat * 1e9) / 1e9;
        const isBarStart =
            forcedBarStart ||
            Math.abs(key / bpb - Math.round(key / bpb)) < 1e-9;
        const existing = ticks.get(key);
        if (existing) {
            if (isBarStart) existing.isBarStart = true;
            return;
        }
        ticks.set(key, {
            beat: key,
            sec: key * secPerBeat,
            primaryLabel: formatRulerTick(primaryUnit, key, ctx),
            secondaryLabel: showSecondary
                ? formatRulerTick(secondaryUnit as TimeUnit, key, ctx)
                : null,
            isBarStart,
        });
    };

    const firstTickIndex = Math.max(0, Math.floor(leftBeat / step));
    const lastTickIndex = Math.max(firstTickIndex, Math.ceil(rightBeat / step));
    for (let index = firstTickIndex; index <= lastTickIndex; index += 1) {
        addTick(index * step, false);
    }

    // 只有当刻度步长小于一个小节时，才额外补入每个小节起始刻度；
    // 否则（每 2/4/8… 个小节才显示一个标签），标签只来自刻度序列本身，
    // 避免缩小后每个小节的文本挤成一团。
    if (step < bpb - 1e-9) {
        const firstBarIndex = Math.max(0, Math.floor(leftBeat / bpb));
        const lastBarIndex = Math.max(firstBarIndex, Math.ceil(rightBeat / bpb));
        for (let barIndex = firstBarIndex; barIndex <= lastBarIndex; barIndex += 1) {
            addTick(barIndex * bpb, true);
        }
    }

    return [...ticks.values()].sort((a, b) => a.beat - b.beat);
}
