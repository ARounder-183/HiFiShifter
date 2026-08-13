import { gridStepBeats } from "./grid";
import { rulerStepCandidates } from "./timeFormat";

/**
 * 根据视口宽度限制网格线的绘制密度。
 *
 * 网格设置得很细或水平缩放很小时，原始网格步长会远小于 1px，
 * 若仍按 1px 下限绘制，宽屏幕上每层可能生成上千条 SVG path 命令。
 *
 * 与时间轴标尺一致，这里不使用“任意 N 条网格线抽一条”的做法，而是
 * 从音乐上均匀的候选步长（网格细分、小节、小节整数倍）中选出满足
 * 最小像素间距的最细步长，避免在 3/4 拍、附点/三连音等网格下出现
 * 前后间距不均或与小节不对齐的情况。
 */

export const MAX_WEAK_GRID_LINES = 160;
export const MAX_STRONG_GRID_LINES = 48;

export const MIN_WEAK_GRID_LINE_SPACING_PX = 8;
export const MIN_STRONG_GRID_LINE_SPACING_PX = 16;

export function resolveGridLineSpacing(
    viewportWidth: number,
    maxLines: number,
    minSpacingPx: number,
): number {
    const viewport = Math.max(1, viewportWidth);
    const safeMaxLines = Math.max(1, maxLines);
    return Math.max(Math.max(1, minSpacingPx), viewport / safeMaxLines);
}

/**
 * 选择弱网格线的显示步长（拍）。
 *
 * 候选步长与标尺共用同一套 `rulerStepCandidates`：
 * - 优先使用最细、且屏幕间距不小于 minSpacingPx 的音乐步长；
 * - 网格步长本身过密时，逐级放大到候选阶梯；
 * - 缩得过小时按小节整数倍继续放大。
 */
export function selectUniformGridStepBeats(args: {
    pxPerBeat: number;
    grid: string;
    beatsPerBar: number;
    minSpacingPx: number;
}): number {
    const bpb = Math.max(1, Math.round(args.beatsPerBar));
    const pxPerBeat = Math.max(1e-9, args.pxPerBeat);
    const spacing = Math.max(1, args.minSpacingPx);
    const gridStep = Math.max(1e-9, gridStepBeats(args.grid));
    const candidates = rulerStepCandidates(args.grid, bpb);

    for (const step of candidates) {
        if (step >= gridStep - 1e-9 && step * pxPerBeat >= spacing - 1e-9) {
            return step;
        }
    }

    let step = bpb;
    while (step * pxPerBeat < spacing - 1e-9) {
        step *= 2;
    }
    return step;
}

export function selectStrongGridBarMultiple(
    barStepPx: number,
    minSpacingPx: number,
): number {
    const stepPx = Math.max(1e-9, barStepPx);
    const spacing = Math.max(1, minSpacingPx);
    let multiple = 1;
    while (multiple * stepPx < spacing - 1e-9) {
        multiple *= 2;
    }
    return multiple;
}

export interface GridLineSamplingPlan {
    weakStepPx: number;
    strongStepPx: number;
}

export function resolveGridLineSamplingPlan(args: {
    pxPerBeat: number;
    grid: string;
    beatsPerBar: number;
    viewportWidth: number;
}): GridLineSamplingPlan {
    const weakSpacing = resolveGridLineSpacing(
        args.viewportWidth,
        MAX_WEAK_GRID_LINES,
        MIN_WEAK_GRID_LINE_SPACING_PX,
    );
    const strongSpacing = resolveGridLineSpacing(
        args.viewportWidth,
        MAX_STRONG_GRID_LINES,
        MIN_STRONG_GRID_LINE_SPACING_PX,
    );
    const weakStepBeats = selectUniformGridStepBeats({
        pxPerBeat: args.pxPerBeat,
        grid: args.grid,
        beatsPerBar: args.beatsPerBar,
        minSpacingPx: weakSpacing,
    });
    const barStepPx = Math.max(1e-9, args.pxPerBeat * Math.max(1, args.beatsPerBar));
    const strongBarMultiple = selectStrongGridBarMultiple(barStepPx, strongSpacing);

    return {
        weakStepPx: weakStepBeats * args.pxPerBeat,
        strongStepPx: barStepPx * strongBarMultiple,
    };
}
