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
import {
    buildTempoGridLines,
    pointIndexAtSec,
    secToBeat,
    tempoMapSegments,
} from "../../../../utils/tempoMap.ts";
import type { TimeUnit, TimeUnitChoice } from "../timeFormat.ts";
import {
    formatRulerTick,
    formatTempoRulerTick,
    rulerStepCandidates,
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
import { secToContentPx, type TimelineAxis } from "../../renderKernel/timelineAxis.js";

/** 弱网格线的最大条数（与 gridLineSampling 的密度上限一致）。 */
const MAX_WEAK_GRID_LINES = 160;

/**
 * 标尺标签的让位阈值（px）：相邻带标签刻度的间距低于它时，左侧标签让位
 * （渲染层不渲染，生成期也把它标记为 `showLabel: false`），保证右侧标签完整可见。
 *
 * 【为什么是"最小可读宽度"而不是用户设定的标签间距】让位的目的是**避免文字
 * 重叠**，不是预留一整个标签位。旧实现把变化点的让位半径取成
 * `max(minLabelSpacingPx, 26)`（默认 110px、可调到 320px），一个变化点就会清掉
 * 左右共 2×320px 内的全部常规标签 —— 实测在 121px 标称间距下挖出 290px 的空洞
 * （2.40 倍）、327px 下挖出 785px（2.40 倍）。用户看到的就是"某段之内的标尺
 * 刻度线与文本消失"，且是否触发取决于缩放档位，于是放大/缩小或滚动又"回来"。
 *
 * 标签自身的宽度由 `TimelineTick.labelMaxWidth`（到下一个保留标签的间距 − 6）
 * 单独保证，与本阈值各司其职。
 */
export const RULER_LABEL_HIDDEN_GAP_PX = 26;

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
    /**
     * 是否渲染标尺标签。标尺只渲染这部分与小节的刻度，保持标签密度稳定。
     * Tempo Map 变化点位置的刻度强制携带标签（见 buildTimelineTicks §4）。
     */
    readonly showLabel: boolean;
    /**
     * 标签文本的最大宽度（px）：到**下一条带标签刻度**的间距减去留白；
     * 没有下一条时为 `null`（不限宽）。
     *
     * 【为什么由生成器算】渲染层曾经在渲染期两两比较"可见切片里的相邻标签"：
     * 切片边界随滚动移动，于是同一个标签的宽度（甚至可见性）会随滚动位置变化 ——
     * 这正是"标尺文字时有时无"的来源之一。间距只依赖刻度序列本身，应当在生成
     * 阶段一次算定。
     */
    readonly labelMaxWidth: number | null;
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
/**
 * 刻度窗口的量化步长（CSS 像素）。
 *
 * `timelineTicks` 与 `TimeRulerMarks` 都按内容坐标自行做可见范围二分，并各
 * 自带缓冲，因此喂给它们一个**量化后**的滚动位置是安全的：锚点 ≤ 真实
 * scrollLeft < 锚点 + 步长，只要把取刻度用的视口宽加上一个步长，覆盖区间就
 * 必然包含真实视口。这样滚动期间刻度数组与标尺子树都不必每帧重算重渲染。
 *
 * 约束：步长必须小于下游 `TimeRulerMarks` 的缓冲
 * （`max(320, viewportWidth * 0.5)`），否则标尺窗口会漏刻度。
 * 该不变式由 `buildTimelineTicks.windowing.test.ts` 锁住。
 */
export const TICK_WINDOW_STEP_PX = 256;

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

    // ── 2. 生成 ────────────────────────────────────────────────────
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

    // ── 2b. 密度兜底（解析式，与滚动位置无关）─────────────────────
    // 旧实现按**生成范围**里的实际条数加粗。生成范围 = 视口 + 两侧各
    // `max(320, viewportWidthPx / 2)` 的缓冲，比视口宽约 3.5 倍；而预算
    // `MAX_WEAK_GRID_LINES` 是按**视口**定的。于是同一缩放下，只要滚动位置让
    // 缓冲"变长"（左端不再被 0 钳住），条数就越过阈值、`stepBeats` 整档翻倍 ——
    // 实测 pxPerSec=40、视口 1500 时，网格步长在 scrollLeft 704→768 之间从
    // 20px 跳到 40px。那正是"水平滚动一定距离后标尺刻度与文本又回来了"的根因：
    // 步长跳档 ⇒ 标签栅格随之跳档。
    //
    // 现在按**视口跨度**解析估算条数：视口宽度与预算都不随滚动变化，于是
    // `stepBeats` / `strongStride` 只依赖 (缩放, 网格设置, 视口宽, Tempo Map)。
    // 无 Tempo Map 时这两条循环恒不触发（`selectUniformGridStepBeats` /
    // `selectStrongGridBarMultiple` 已按视口宽度保证达标），保留它们只是把
    // 不变量写死在代码里。
    const maxWeakLines = MAX_WEAK_GRID_LINES;
    const maxStrongLines = Math.max(1, Math.ceil(maxWeakLines / 3));
    const viewportSpanBeats = hasTempoMap
        ? Math.max(
              1e-9,
              secToBeat(tempoMap, startSec + axis.viewportWidthPx / pxPerSec, bpm) -
                  secToBeat(tempoMap, startSec, bpm),
          )
        : axis.viewportWidthPx / pxPerBeat;
    let guard = 0;
    while (guard < 64 && viewportSpanBeats / stepBeats > maxWeakLines) {
        stepBeats *= 2;
        guard += 1;
    }
    while (guard < 64 && viewportSpanBeats / (beatsPerBar * strongStride) > maxStrongLines) {
        strongStride *= 2;
        guard += 1;
    }
    const lines = buildFilteredLines();

    // ── 3. 标签栅格：网格步长的 2 的幂倍 ───────────────────────────
    // 【为什么必须是 stepBeats 的整数倍】标尺刻度是网格刻度的**子集**（既有
    // 不变量）。标签位置若落在没有网格线的拍位上，标尺会画出一条网格里没有的
    // 竖线；而"按秒/拍整除判定"在附点、三连音网格下必然失配 —— 实测 1/8d 网格
    // 的标签间距被放大到标称值的 3.00 倍（每三个标签位只有一个落得到网格线上）。
    //
    // 【为什么取 2 的幂倍】`stepBeats` 会因密度兜底整体翻倍（只乘 2），而集合
    // `{stepBeats × 2^k}` 对翻倍是**不变的** —— 于是即便网格步长随 Tempo Map 的
    // 密度估算变化，标签栅格也不会跟着跳档（"标签忽有忽无"）。
    //
    // 【Swing 时至少取 2】swing 把**奇数索引**的线整体平移最多半步，其秒位不再
    // 对应整数拍；标签文字由秒位反推拍值，落在奇数索引上会渲染成 "1.2.300"。
    // 强制偶数倍后标签只落在未被平移的线上，间距恰好等于 swing 下的实际间距
    // （与旧实现的实际输出一致，但不再依赖"按秒判定恰好排除奇数线"这一巧合）。
    const spacingPx = Math.max(24, Math.min(600, args.minLabelSpacingPx));
    const swingOn = (args.swingPercent ?? 0) > 0;
    const labelStrideFor = (pxPerBeatForSegment: number, beatsPerBarForSegment: number): number => {
        const stepPx = stepBeats * Math.max(1e-9, pxPerBeatForSegment);
        const minStride = swingOn ? 2 : 1;
        // 候选 stride（以 stepBeats 为单位）：
        // 1) 音乐候选阶梯中恰好是 stepBeats 整数倍的那些 —— 优先。它们与小节对齐
        //    （3/4 拍下取 3 拍 / 6 拍，标签落在小节起点上，而不是 2 的幂给出的
        //    "3.3" 这种非小节位置）；
        // 2) stepBeats 的 2 的幂倍 —— 兜底。附点 / 三连音网格的音乐阶梯与网格
        //    栅格不整除（1/8d 的候选是 1,2,4,8… 而网格步长是 0.75 的倍数），
        //    此时必须靠它保证标签仍落在网格线上。
        const strides: number[] = [];
        for (const candidate of rulerStepCandidates(grid, beatsPerBarForSegment)) {
            const ratio = candidate / stepBeats;
            const rounded = Math.round(ratio);
            if (rounded >= 1 && Math.abs(ratio - rounded) < 1e-9) strides.push(rounded);
        }
        for (let multiplier = 1; multiplier <= 1 << 16; multiplier *= 2) strides.push(multiplier);
        strides.sort((a, b) => a - b);
        for (const stride of strides) {
            if (stride >= minStride && stride * stepPx >= spacingPx - 1e-9) return stride;
        }
        let stride = minStride;
        while (stride * stepPx < spacingPx - 1e-9) stride *= 2;
        return stride;
    };
    /** 本视口的标签栅格步长（以 `stepBeats` 为单位）。 */
    const labelStride = labelStrideFor(pxPerBeat, beatsPerBar);
    const ctx: TimeFormatContext = { bpm, beatsPerBar, grid, tempoMap };
    const showSecondary = args.secondaryUnit !== "none" && args.secondaryUnit !== args.primaryUnit;

    // ── 3b. Tempo Map 的逐段标签步长 ───────────────────────────────
    // 段内一步的像素宽度随该段 BPM 变化：用全局 BPM 统一取 stride 会让快段
    // （BPM 高 → 每拍像素少）的标签挤成一团。这里每段按自己的 segPxPerBeat 取
    // "网格步长的 2 的幂倍"，与均匀路径同一套规则（见 §3）。
    //
    // 【为什么不再显式枚举秒位再集合匹配】旧实现枚举 `段起点 + m*stepBeats*段秒每拍`
    // 并用 `round(sec*1e6)` 建集合、匹配时容许 ±1e-6。它在两处不可靠：
    // - stride 由 `round(segStep / stepBeats)` 得出，是**非 2 的幂**（实测 1/8d
    //   网格下 16/0.75 → round 21），把"音乐候选阶梯"与"网格栅格"两个不同步长的
    //   集合硬拼在一起，二者不整除时标签就落不到网格线上；
    // - 跨段的最小间距约束用 `continue` 跳过候选却**不推进** `lastLabelPx`，
    //   于是段首的空白被成倍放大（实测 121px 标称间距下挖出 291px 的空洞）。
    // 现在标签判定改为按**索引**取模（索引与 stride 都是整数，判定精确，且
    // swing 平移过的线不会被误判），间距交给 §5 的统一让位规则处理。
    const labelSegments =
        tempoMap && tempoMap.points.length > 0
            ? tempoMapSegments(
                  tempoMap,
                  Math.max(endSec, tempoMap.points[tempoMap.points.length - 1].positionSec),
              )
            : [];
    const segmentLabelStrides = labelSegments.map((segment) =>
        labelStrideFor(
            (60 / Math.max(1, segment.point.bpm)) * Math.max(0, pxPerSec),
            Math.max(1, segment.beatsPerBar),
        ),
    );

    // 弱线与小节线会落在同一秒（小节起点本身就是一条弱线位置），必须合并成
    // 单个刻度、小节样式优先。不去重的后果是标尺出现间距为 0 的相邻刻度，
    // 触发 labelHidden（间距 < 26px）把标签整片隐藏，只剩一堆裸竖线。
    //
    // `index` 随合并保留（小节线落在弱线栅格上时携带等价弱线索引）：标签栅格
    // 只认弱线索引，见 `TempoGridLine.index` 的说明。
    const merged = new Map<number, { sec: number; isBar: boolean; index: number | undefined }>();
    for (const line of lines) {
        const key = Math.round(line.sec * 1e6) / 1e6;
        const existing = merged.get(key);
        if (existing) {
            existing.isBar = existing.isBar || line.isBar;
            if (existing.index === undefined) existing.index = line.index;
            continue;
        }
        merged.set(key, { sec: line.sec, isBar: line.isBar, index: line.index });
    }

    const ticks: TimelineTick[] = [];
    for (const entry of merged.values()) {
        const beat = hasTempoMap ? secToBeat(tempoMap, entry.sec, bpm) : entry.sec / secPerBeat;
        // 标签只落在标签栅格的整数倍上。这里**不能**把小节起点无条件计入：
        // 缩小时小节间距会小到放不下标签，标尺便会只剩一堆没有文字的竖线。
        // 小节通过 isBarStart 影响刻度样式（2px 强线 + 加粗文字），而不是额外
        // 增加刻度数量——与旧 buildRulerTicks 的语义一致。
        const stride =
            tempoMap && tempoMap.points.length > 0
                ? segmentLabelStrides[pointIndexAtSec(tempoMap, entry.sec)]
                : labelStride;
        const onLabelStep =
            entry.index !== undefined && stride !== undefined && entry.index % stride === 0;
        ticks.push({
            sec: entry.sec,
            beat,
            contentPx: secToContentPx(axis, entry.sec),
            isBarStart: entry.isBar,
            isStrongGridLine: entry.isBar,
            showLabel: onLabelStep,
            labelMaxWidth: null,
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

    // ── 4. 变化点位置强制展示标尺值 ────────────────────────────────
    // 变化点（Tempo Map 段起点）是工程的时间地标，必须带标签：读不出它的位置时
    // 用户只能看到旗帜悬在两个标尺值之间。这里只做"强制显示"。
    //
    // 【为什么不再顺手清场】旧实现在这里额外隐藏了距变化点
    // `max(minLabelSpacingPx, 26)`（默认 110、可调到 320）以内的**所有**常规
    // 标签，理由是"保证变化点标签的显示宽度"。但标签宽度已由 §5 算出的
    // `labelMaxWidth`（到下一个保留标签的间距 − 6）保证，不需要清场；而清场半径
    // 按**用户设定的标签间距**取，等于把一整个标签位整段挖掉 —— 实测在 121px
    // 标称间距下挖出 290px 的空洞（2.40 倍）、在 327px 下挖出 785px（2.40 倍）。
    // 那正是"某段之内的标尺刻度线与文本消失"的直接机制：是否触发取决于当前
    // 缩放档位下常规标签与变化点的相对位置，于是放大/缩小或滚动一下又"回来"。
    // 现在让位统一交给 §5（阈值 = 文字重叠宽度），且变化点标签优先保留。
    const changePointKeys = new Set<number>();
    if (tempoMap && tempoMap.points.length > 0) {
        for (const point of tempoMap.points) {
            changePointKeys.add(Math.round(point.positionSec * 1e6));
        }
    }
    /** 该秒位是否就是（或落在量化误差内的）一个 Tempo Map 变化点。 */
    const isChangePointSec = (sec: number): boolean => {
        if (changePointKeys.size === 0) return false;
        const key = Math.round(sec * 1e6);
        return (
            changePointKeys.has(key) || changePointKeys.has(key - 1) || changePointKeys.has(key + 1)
        );
    };
    const forcedLabel = new Set<number>();
    for (let i = 0; i < ticks.length; i += 1) {
        if (!isChangePointSec(ticks[i].sec)) continue;
        forcedLabel.add(i);
        if (!ticks[i].showLabel) ticks[i] = { ...ticks[i], showLabel: true };
    }

    // ── 5. 标签版式：重叠让位（收敛级联）+ 计算最大宽度 ─────────────
    // 这一步曾在渲染层（TimeRulerMarks）做，判据是"**可见切片**里相邻标签的间距"。
    // 切片边界随滚动移动 ⇒ 同一个标签的可见性会随滚动位置改变（右边缘的标签没有
    // "下一条"，因此永不被隐藏；它一进入切片内部就可能被隐藏）—— 这正是"标尺
    // 文字时有时无"的机制之一。间距只取决于刻度序列本身，必须一次算定。
    //
    // 让位规则：从左到右单趟推进，间距小于"文字重叠宽度"时**左侧让位**（保证右侧
    // 标签完整），并与新的最近保留者**继续比较**（级联收敛）。旧实现只有一条
    // `if`，一次只让掉一个：连续三条挨得近时，让掉一条后剩下两条仍小于阈值，
    // 渲染层再各隐藏一次，最终两条都看不见。
    //
    // 变化点标签优先：间距不足时若左侧是变化点，改让**右侧**（本次候选）—— 否则
    // §4 刚强制打开的变化点标签会被这里立刻关掉。
    const labeledIndexes: number[] = [];
    for (let i = 0; i < ticks.length; i += 1) if (ticks[i].showLabel) labeledIndexes.push(i);
    const surviving: number[] = [];
    for (const index of labeledIndexes) {
        let yielded = false;
        for (;;) {
            const previous = surviving[surviving.length - 1];
            if (
                previous === undefined ||
                ticks[index].contentPx - ticks[previous].contentPx >= RULER_LABEL_HIDDEN_GAP_PX
            ) {
                break;
            }
            if (forcedLabel.has(previous) && !forcedLabel.has(index)) {
                ticks[index] = { ...ticks[index], showLabel: false, labelMaxWidth: 0 };
                yielded = true;
                break;
            }
            ticks[previous] = { ...ticks[previous], showLabel: false, labelMaxWidth: 0 };
            surviving.pop();
        }
        if (!yielded) surviving.push(index);
    }
    // 版式：宽度取到**下一条保留标签**的间距（减去留白）；最后一条不限宽。
    for (let k = 0; k < surviving.length; k += 1) {
        const index = surviving[k];
        const nextIndex = surviving[k + 1];
        const maxWidth =
            nextIndex === undefined
                ? null
                : Math.max(0, ticks[nextIndex].contentPx - ticks[index].contentPx - 6);
        ticks[index] = { ...ticks[index], labelMaxWidth: maxWidth };
    }

    // ── 6. 兜底：绝不整片无标签 ────────────────────────────────────
    // 上面所有规则都是"隐藏"方向的操作，极端参数下可能把所有标签都隐藏掉，用户
    // 看到的就是"标尺文字整片消失"。恢复按最粗的粒度（小节起点）取候选，保证
    // 任何缩放/网格组合下至少每 `minLabelSpacingPx` 有一个标签；窗口内完全没有
    // 小节起点时（Tempo Map 段极短等）才退回全部刻度，维持"绝不整片无标签"。
    // 恢复标签的宽度与 §5 同一约定：到下一条恢复标签的间距减去留白、最后一条
    // 不限宽 —— 若不限宽（null），相邻两条恢复标签会互相重叠。
    if (ticks.length > 0 && !ticks.some((tick) => tick.showLabel)) {
        const candidates: number[] = [];
        for (let i = 0; i < ticks.length; i += 1) if (ticks[i].isBarStart) candidates.push(i);
        if (candidates.length === 0) {
            for (let i = 0; i < ticks.length; i += 1) candidates.push(i);
        }
        const restored: number[] = [];
        for (const index of candidates) {
            const last = restored[restored.length - 1];
            if (
                last !== undefined &&
                ticks[index].contentPx - ticks[last].contentPx < args.minLabelSpacingPx
            ) {
                continue;
            }
            restored.push(index);
        }
        for (let k = 0; k < restored.length; k += 1) {
            const index = restored[k];
            const nextIndex = restored[k + 1];
            const maxWidth =
                nextIndex === undefined
                    ? null
                    : Math.max(0, ticks[nextIndex].contentPx - ticks[index].contentPx - 6);
            ticks[index] = { ...ticks[index], showLabel: true, labelMaxWidth: maxWidth };
        }
    }

    return ticks;
}
