/**
 * 时间线统一刻度源（网格线与标尺刻度的唯一来源）。
 *
 * 【主要内容】按当前投影生成可见范围内的刻度序列，每个刻度同时携带：时间、
 * 内容坐标 x、是否小节起点、是否作为强网格线、是否显示标签、主/副标签文本。
 *
 * 【作用】消除网格与标尺"各自算一套"的错位。历史问题是两者用不同的步长选择
 * 入口，且分属不同坐标系：
 * - 网格走 `resolveGridLineSamplingPlan`（beat 域，按 pxPerBeat 推算步长），
 *   Tempo Map 下另有 `buildTempoGridLineXsForViewport` 一条路径；
 * - 标尺走 `buildRulerTicks`（时间域，按 sec 生成，`sec * pxPerSec` 定位）。
 * Tempo Map 下 beat 与像素不再是线性关系，两条路径必然分叉。
 *
 * 现在两者消费同一份 tick：网格画出全部刻度，标尺只渲染其中标记了
 * `showLabel` 的部分。标尺刻度因此天然是网格线的子集，不可能错位。
 *
 * 【与其他模块的关系】
 * - 上游：`useTimelineState` 在渲染期生成，供 `TimeRuler` 与 `BackgroundGrid`
 *   同时使用；参数编辑器在接入 axis 后复用同一函数（P3）。
 * - 复用：`buildTempoGridLines()` 负责逐段生成（Tempo Map 与均匀网格都支持），
 *   `selectUniformGridStepBeats` / `selectRulerStep` 负责步长选择。
 * - 依赖：`timelineAxis.ts` 提供内容坐标投影。
 */

import type { GridSize } from "../../../../features/session/sessionTypes.ts";
import type { TempoMap } from "../../../../utils/tempoMap.ts";
import { buildTempoGridLines, secToBeat } from "../../../../utils/tempoMap.ts";
import type { TimeUnit, TimeUnitChoice } from "../timeFormat.ts";
import {
    formatRulerTick,
    formatTempoRulerTick,
    selectRulerStep,
    type TimeFormatContext,
} from "../timeFormat.ts";
import { gridStepBeats } from "../grid.ts";
import {
    MIN_STRONG_GRID_LINE_SPACING_PX,
    MIN_WEAK_GRID_LINE_SPACING_PX,
    resolveGridLineSpacing,
    selectStrongGridBarMultiple,
    selectUniformGridStepBeats,
} from "../gridLineSampling.ts";
import { secToContentPx, type TimelineAxis } from "./timelineAxis.js";

/** 弱网格线的最大条数（与 gridLineSampling 的密度上限一致）。 */
const MAX_WEAK_GRID_LINES = 160;

/** 一个刻度：网格线与标尺刻度的最小公共单位。 */
export interface TimelineTick {
    /** 工程时间（秒）。 */
    readonly sec: number;
    /** 全局拍号坐标（Tempo Map 下为分段折算值）。 */
    readonly beat: number;
    /** 内容坐标 x（CSS 像素），由 axis 投影得到，网格与标尺共用。 */
    readonly contentPx: number;
    /** 是否为小节起点。 */
    readonly isBarStart: boolean;
    /** 是否作为强网格线绘制（小节过密时按 stride 抽取，避免糊成一片）。 */
    readonly isStrongGridLine: boolean;
    /** 是否渲染标尺标签。标尺只渲染这部分与小节的刻度，保持标签密度稳定。 */
    readonly showLabel: boolean;
    /** 主单位标签文本。 */
    readonly primaryLabel: string;
    /** 副单位标签文本（未启用副单位时为 null）。 */
    readonly secondaryLabel: string | null;
}

/**
 * 生成可见范围内的统一刻度序列（按时间升序，无重复）。
 *
 * 流程：
 * 1. 按视口（含缓冲）确定生成范围；
 * 2. 选择弱网格步长与强网格 stride（Tempo Map 按视口跨度估算，避免长工程
 *    + 细网格时先生成数百万条线）；
 * 3. 调 `buildTempoGridLines` 生成 `{sec, isBar}` 序列（Swing 与强线抽取都在
 *    其中完成）；
 * 4. 按实际条数做密度兜底（分段对齐会让估算偏少，逐步加粗直到达标）；
 * 5. 用 axis 投影为内容坐标，并按标签步长标记 `showLabel`。
 *
 * 特殊说明：
 * - `contentPx` 一律经 `secToContentPx` 投影，禁止在本文件外用
 *   `sec * pxPerSec` 重新计算。
 * - 生成范围用缓冲像素而非视口边界，避免滚动时边缘刻度闪烁。
 *
 * @param args.axis 统一坐标投影。
 * @param args.bpm 工程 BPM（无 Tempo Map 时的拍速基准）。
 * @param args.beatsPerBar 每小节拍数。
 * @param args.grid 网格细分（如 "1/8"）。
 * @param args.primaryUnit / secondaryUnit 标尺主/副时间单位。
 * @param args.minLabelSpacingPx 标尺标签的最小像素间距。
 * @param args.minGridSpacingPx 弱网格线的最小像素间距（缺省 8）。
 * @param args.swingPercent Swing 强度（0-100），作用于弱网格奇数格。
 * @param args.tempoMap 速度映射；为 null 或空时走均匀网格。
 * @returns 升序刻度数组。
 */
export function buildTimelineTicks(args: {
    axis: TimelineAxis;
    bpm: number;
    beatsPerBar: number;
    grid: GridSize | string;
    primaryUnit: TimeUnit;
    secondaryUnit: TimeUnitChoice;
    minLabelSpacingPx: number;
    minGridSpacingPx?: number;
    swingPercent?: number;
    tempoMap?: TempoMap | null;
}): TimelineTick[] {
    const axis = args.axis;
    const pxPerSec = axis.pxPerSec;
    const bpm = Math.max(1, args.bpm);
    const beatsPerBar = Math.max(1, Math.round(args.beatsPerBar || 4));
    const grid = args.grid;
    const tempoMap = args.tempoMap ?? null;
    const hasTempoMap = Boolean(tempoMap && tempoMap.points.length > 0);

    const bufferPx = Math.max(320, axis.viewportWidthPx * 0.5);
    const leftPx = Math.max(0, axis.scrollLeftPx - bufferPx);
    const rightPx = axis.scrollLeftPx + axis.viewportWidthPx + bufferPx;
    const startSec = Math.max(0, leftPx / pxPerSec);
    const endSec = Math.max(startSec, rightPx / pxPerSec);

    const secPerBeat = 60 / bpm;
    const pxPerBeat = Math.max(1e-9, secPerBeat * pxPerSec);

    // ── 1. 步长选择 ────────────────────────────────────────────────
    let stepBeats: number;
    let strongStride: number;

    if (hasTempoMap) {
        // 以真实视口跨度估步长：若按"生成范围"估算，右侧空白区会让工程内的
        // 网格越估越粗（生成范围随空白无限变长）。
        const viewportSpanSec = Math.max(1e-9, axis.viewportWidthPx / pxPerSec);
        const spanBeats = Math.max(
            1e-9,
            secToBeat(tempoMap, startSec + viewportSpanSec, bpm) -
                secToBeat(tempoMap, startSec, bpm),
        );
        const maxWeak = Math.max(
            1,
            Math.min(
                MAX_WEAK_GRID_LINES,
                Math.floor(
                    axis.viewportWidthPx /
                        Math.max(1, args.minGridSpacingPx ?? MIN_WEAK_GRID_LINE_SPACING_PX),
                ) || MAX_WEAK_GRID_LINES,
            ),
        );
        stepBeats = Math.max(1e-9, gridStepBeats(grid));
        while (spanBeats / stepBeats > maxWeak) {
            stepBeats *= 2;
        }
        const maxStrong = Math.max(1, Math.ceil(maxWeak / 3));
        strongStride = Math.max(1, Math.ceil(spanBeats / beatsPerBar / maxStrong));
    } else {
        const weakSpacing = resolveGridLineSpacing(
            axis.viewportWidthPx,
            MAX_WEAK_GRID_LINES,
            Math.max(1, args.minGridSpacingPx ?? MIN_WEAK_GRID_LINE_SPACING_PX),
        );
        const strongSpacing = resolveGridLineSpacing(
            axis.viewportWidthPx,
            MAX_WEAK_GRID_LINES / 3,
            MIN_STRONG_GRID_LINE_SPACING_PX,
        );
        // selectUniformGridStepBeats 内部取 min(rulerStep, gridStep*2^n)，
        // 保证网格不会比标尺更粗。
        stepBeats = selectUniformGridStepBeats({
            pxPerBeat,
            grid,
            beatsPerBar,
            minSpacingPx: weakSpacing,
        });
        strongStride = selectStrongGridBarMultiple(pxPerBeat * beatsPerBar, strongSpacing);
    }

    // ── 2. 生成 + 密度兜底 ─────────────────────────────────────────
    const buildLines = () =>
        buildTempoGridLines({
            startSec,
            endSec,
            map: tempoMap,
            stepBeats,
            fallbackBpm: bpm,
            fallbackBeatsPerBar: beatsPerBar,
            swingPercent: args.swingPercent ?? 0,
            strongStride,
        });

    // 无 Tempo Map 时 buildTempoGridLines 会画出每一条小节线——它只在 Tempo Map
    // 分支应用 strongStride。这里按**全局**小节序号统一抽取：用全局序号而不是
    // "可见范围内的第几条"，抽取相位才不会随滚动/缩放跳变导致线条闪烁。
    const buildFilteredLines = () => {
        const raw = buildLines();
        if (hasTempoMap || strongStride <= 1) return raw;
        return raw.filter((line) => {
            if (!line.isBar) return true;
            const barIndex = Math.round(line.sec / secPerBeat / beatsPerBar);
            return barIndex % strongStride === 0;
        });
    };

    let lines = buildFilteredLines();
    const maxWeakLines = MAX_WEAK_GRID_LINES;
    const maxStrongLines = Math.max(1, Math.ceil(maxWeakLines / 3));
    // 分段对齐会让按跨度估算的步长偏细，按实际条数逐步加粗直到达标。
    let guard = 0;
    while (
        guard < 64 &&
        (lines.filter((line) => !line.isBar).length > maxWeakLines ||
            lines.filter((line) => line.isBar).length > maxStrongLines)
    ) {
        guard += 1;
        if (lines.filter((line) => !line.isBar).length > maxWeakLines) {
            stepBeats *= 2;
        } else {
            strongStride *= 2;
        }
        lines = buildFilteredLines();
    }

    // ── 3. 标签步长与格式化 ────────────────────────────────────────
    // Tempo Map 下 beat 是分段折算值（非均匀），不能用均匀 beat 域选出的
    // labelStepBeats 去做整除判定——旧 buildRulerTicks 的 Tempo Map 路径正是
    // 逐段局部对齐生成刻度的。这里改用时间域步长：用 fallback BPM 把 beat 步长
    // 折算为秒，再按 sec 判定是否落在标签位置。
    const labelStepBeats = selectRulerStep({
        pxPerBeat,
        grid,
        beatsPerBar,
        minLabelSpacingPx: args.minLabelSpacingPx,
    });
    /** 时间域标签步长（秒）。Tempo Map 下用它判定，均匀网格下等价于 beat 域。 */
    const labelStepSec = labelStepBeats * secPerBeat;
    const ctx: TimeFormatContext = { bpm, beatsPerBar, grid, tempoMap };
    const showSecondary =
        args.secondaryUnit !== "none" && args.secondaryUnit !== args.primaryUnit;

    // 弱线与小节线会落在同一秒（小节起点本身就是一条弱线位置），必须合并成
    // 单个刻度、小节样式优先。不去重的后果是标尺出现间距为 0 的相邻刻度，
    // 触发 labelHidden（间距 < 26px）把标签整片隐藏，只剩一堆裸竖线。
    const merged = new Map<number, { sec: number; isBar: boolean }>();
    for (const line of lines) {
        const key = Math.round(line.sec * 1e6) / 1e6;
        const existing = merged.get(key);
        if (existing) {
            existing.isBar = existing.isBar || line.isBar;
            continue;
        }
        merged.set(key, { sec: line.sec, isBar: line.isBar });
    }

    const ticks: TimelineTick[] = [];
    for (const entry of merged.values()) {
        const beat = hasTempoMap ? secToBeat(tempoMap, entry.sec, bpm) : entry.sec / secPerBeat;
        // 标签只落在标签步长的整数倍上。这里**不能**把小节起点无条件计入：
        // 缩小时小节间距会小到放不下标签，标尺便会只剩一堆没有文字的竖线。
        // 小节通过 isBarStart 影响刻度样式（2px 强线 + 加粗文字），而不是额外
        // 增加刻度数量——与旧 buildRulerTicks 的语义一致。
        //
        // Tempo Map 下必须用时间域判定：beat 是分段折算值（非均匀），而
        // labelStepBeats 是在均匀 beat 域选出的，两者不在同一坐标系。用 sec 域
        // 的 labelStepSec（= labelStepBeats * fallbackSecPerBeat）做整除，与旧
        // buildRulerTicks Tempo Map 路径的逐段局部对齐语义一致。
        const stepsFromOrigin = hasTempoMap
            ? entry.sec / labelStepSec
            : beat / labelStepBeats;
        const onLabelStep = Math.abs(stepsFromOrigin - Math.round(stepsFromOrigin)) < 1e-6;
        ticks.push({
            sec: entry.sec,
            beat,
            contentPx: secToContentPx(axis, entry.sec),
            isBarStart: entry.isBar,
            isStrongGridLine: entry.isBar,
            showLabel: onLabelStep,
            primaryLabel: hasTempoMap
                ? formatTempoRulerTick(args.primaryUnit, entry.sec, ctx)
                : formatRulerTick(args.primaryUnit, beat, ctx),
            secondaryLabel: showSecondary
                ? hasTempoMap
                    ? formatTempoRulerTick(args.secondaryUnit as TimeUnit, entry.sec, ctx)
                    : formatRulerTick(args.secondaryUnit as TimeUnit, beat, ctx)
                : null,
        });
    }

    // merged 的迭代顺序即 lines 的顺序（已升序），显式排序以消除对上游顺序的
    // 隐式依赖。
    ticks.sort((a, b) => a.sec - b.sec);

    return ticks;
}
