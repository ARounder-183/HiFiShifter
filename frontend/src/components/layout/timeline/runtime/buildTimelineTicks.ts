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
import { buildTempoGridLines, secToBeat, tempoMapSegments } from "../../../../utils/tempoMap.ts";
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
    const labelStepBeats = selectRulerStep({
        pxPerBeat,
        grid,
        beatsPerBar,
        minLabelSpacingPx: args.minLabelSpacingPx,
    });
    const ctx: TimeFormatContext = { bpm, beatsPerBar, grid, tempoMap };
    const showSecondary = args.secondaryUnit !== "none" && args.secondaryUnit !== args.primaryUnit;

    // ── 3b. Tempo Map 的标签位置：显式枚举，而非对线做整除判定 ─────
    // Tempo Map 下网格线逐段局部对齐生成（段内第 k 条 = 段起点 +
    // k*stepBeats*(60/段BPM)），段与段的秒间距随 BPM 变化，不存在任何全局常量
    // 步长能整除这些秒位——这正是旧 buildRulerTicks 为 Tempo Map 单独按段生成
    // 刻度的原因。
    //
    // 这里按**与 buildTempoGridLines 完全相同的公式**（含同一顺序的浮点运算与
    // swing 偏移）显式枚举出标签秒位，再用集合匹配决定哪条线带标签。相比"把线
    // 的秒位 round 成索引再取模"，枚举法不会把一批不同的线折叠成同一个索引：
    // 曾出现 3/4 拍段内 stepBeats=32、每条小节线的局部索引 3m/32 全部 round 成
    // 0，于是相邻 1 秒（2px）的小节线整批被判为标签，标尺因间距小于 26px 隐藏
    // 阈值而整片空白。
    //
    // 与均匀网格路径的对称点：均匀路径用严格整除（|v-round(v)|<1e-6）天然排除
    // 不在标签栅格上的小节线；枚举法在这里起到同样的作用。
    // 直接判空而非用 hasTempoMap 布尔量：后者无法让 TS 收窄 tempoMap 的类型。
    const labelSegments =
        tempoMap && tempoMap.points.length > 0
            ? tempoMapSegments(
                  tempoMap,
                  Math.max(endSec, tempoMap.points[tempoMap.points.length - 1].positionSec),
              )
            : [];

    // 逐段计算标签 stride：段内一步的像素宽度随该段 BPM 变化，若用全局 fallback
    // BPM 统一取 stride，快段（BPM 高 → 每拍像素少）的标签会挤成一团。这与旧
    // buildRulerTicks 每段按自己的 segPxPerBeat 选 step 的做法一致。
    // stride 只依赖 (tempoMap, pxPerSec, grid, minLabelSpacingPx)，与视口滚动
    // 无关，因此标签相位在滚动时保持稳定。
    const segmentLabelStrides = labelSegments.map((segment) => {
        const segPxPerBeat = (60 / Math.max(1, segment.point.bpm)) * Math.max(0, pxPerSec);
        const segStep = selectRulerStep({
            pxPerBeat: segPxPerBeat,
            grid,
            beatsPerBar: Math.max(1, segment.beatsPerBar),
            minLabelSpacingPx: args.minLabelSpacingPx,
        });
        return Math.max(1, Math.round(segStep / stepBeats));
    });

    const swingPercent = Math.max(0, Math.min(100, args.swingPercent ?? 0));
    /** 标签秒位集合，键为 sec 四舍五入到 1e-6。 */
    const labelSecKeys = new Set<number>();
    if (hasTempoMap) {
        // 跨段传递上一个已接受的标签位置。每段都在自己的起点重新锚定相位，
        // 因此段 A 的末个标签与段 B 的首个标签（m=0）可能挨得极近——实测
        // pps=8 处仅 16px，低于标尺 26px 的隐藏阈值，边界附近会成片空白。
        let lastLabelPx: number | null = null;
        for (let i = 0; i < labelSegments.length; i += 1) {
            const segment = labelSegments[i];
            const stride = segmentLabelStrides[i];
            const segBpm = Math.max(1, segment.point.bpm);
            const segSecPerBeat = 60 / segBpm;
            const stepSec = stepBeats * segSecPerBeat;
            if (!Number.isFinite(stepSec) || stepSec <= 1e-12) continue;
            // 只枚举可见范围：段起点可能远在视口左侧，全段枚举会退化成 O(段长)。
            const fromSec = Math.max(startSec, segment.startSec);
            const toSec = Math.min(endSec, segment.endSec);
            if (toSec < fromSec - 1e-9) continue;
            const firstM = Math.max(0, Math.ceil((fromSec - segment.startSec) / stepSec - 1e-9));
            const lastM = Math.floor((toSec - segment.startSec) / stepSec + 1e-9);
            // 相位锚定到 stride 的整数倍（相对段起点），保证滚动/缩放不跳变。
            const startM = Math.ceil(firstM / stride) * stride;
            for (let m = startM; m <= lastM; m += stride) {
                // 与 buildTempoGridLines 的 swingAt 同式同序，确保浮点结果一致。
                const swing =
                    swingPercent > 0 && m % 2 !== 0
                        ? (swingPercent / 100) * 0.5 * stepBeats * segSecPerBeat
                        : 0;
                const sec = segment.startSec + m * stepBeats * segSecPerBeat + swing;
                // add() 内部按 [startSec, endSec] 裁剪，这里同样只收范围内的值。
                if (!Number.isFinite(sec) || sec < startSec - 1e-9 || sec > endSec + 1e-9) {
                    continue;
                }
                // 最小间距约束：段边界处跳过与上一个标签挨太近的候选。段内间距
                // 均匀，正常情况下不会触发；触发即说明该处标签会重叠。
                const px = secToContentPx(axis, sec);
                if (lastLabelPx !== null && px - lastLabelPx < args.minLabelSpacingPx) continue;
                lastLabelPx = px;
                labelSecKeys.add(Math.round(sec * 1e6) / 1e6);
            }
        }
    }

    /** 匹配标签秒位（±1 个量化单位，吸收浮点末位差异）。 */
    const isTempoLabelPosition = (sec: number): boolean => {
        const base = Math.round(sec * 1e6);
        return (
            labelSecKeys.has(base / 1e6) ||
            labelSecKeys.has((base - 1) / 1e6) ||
            labelSecKeys.has((base + 1) / 1e6)
        );
    };

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
        // Tempo Map 下改按段内局部索引抽取（见 3b）。均匀网格下 beat 与 sec 线性
        // 相关，保持原有的整数倍判定。
        const onLabelStep = hasTempoMap
            ? isTempoLabelPosition(entry.sec)
            : Math.abs(beat / labelStepBeats - Math.round(beat / labelStepBeats)) < 1e-6;
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
