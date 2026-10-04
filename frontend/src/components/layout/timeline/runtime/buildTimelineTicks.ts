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
import { tickWindowBufferPx } from "./tickWindow.js";

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

/**
 * 标签间距**上限**的倍数（相对 `minLabelSpacingPx`）—— 密度补充的触发阈值。
 *
 * 【为什么必须有上限】`labelStrideFor` 选的是满足 `stride * stepPx >= spacingPx`
 * 的**最小** stride，而 stride 只能取 2 的幂 / 音乐候选阶梯，因此实际间距在
 * `[spacingPx, 2 × spacingPx)` 之间漂移；Tempo Map 下各段 stride 独立选择，
 * 段边界拼接后可达 4×。`TimeRulerMarks` 只渲染 `showLabel` 的刻度 ⇒ 间距过大
 * 的那一段**既没有刻度线也没有文本**，正是"某段之内的标尺刻度与文本消失"。
 *
 * 【为什么取 2】2 恰好是栅格自身的固有粒度（相邻档位是 2 的幂）。取 2 意味着
 * 补充只在栅格**确实**超出其固有粒度时才介入，正常档位不触发、不会平白加密。
 */
const LABEL_DENSITY_TARGET_MULT = 2;

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
    /**
     * 该刻度是否可作为**密度补充**（§5b）的候选（**内部字段**，渲染层不消费）。
     *
     * 两个条件缺一不可：
     * - **落在弱线栅格上**（`index !== undefined`）。附点 / 三连音网格下，小节线
     *   常常不落在弱线栅格上（例如 1/8d 的网格步长是 0.75 拍，而小节线每 4 拍），
     *   把这种线当标签会往"均匀的弱线栅格"里插进一条完全不同步的线 —— 实测会把
     *   附点网格刚修好的"间距均匀"重新打乱（退化成 3.00×）。
     * - **不被 swing 平移**（swing 平移段内奇数弱线索引的线，其秒位不对应整数拍，
     *   标签文字会渲染成 `1.2.300`）。
     */
    readonly densityEligible?: boolean;
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
 * 刻度窗口的量化步长（CSS 像素）—— 定义与理由见 `tickWindow.ts`。
 *
 * 此处**转出**而非就地定义：内核的提交步长、标尺的切片缓冲都与它是同一族
 * 常量，必须能互相引用而不产生循环依赖（`tickAxis` 已从本模块导入它）。
 */
export { TICK_WINDOW_STEP_PX } from "./tickWindow.js";

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

    // 缓冲与 `TimeRulerMarks` 的切片共用同一公式（见 `tickWindow.ts`）：两处
    // 一旦分叉，切片就会比生成范围更宽，切出不存在的区间 —— 表现为标尺露白。
    const bufferPx = tickWindowBufferPx(axis.viewportWidthPx);
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
            if (stride < minStride || stride * stepPx < spacingPx - 1e-9) continue;
            // swing 下必须取偶数：标签判定用全局索引取模（见 §3b），而 swing 把
            // **奇数索引**的线整体平移半步。若 stride 为奇数，标签会落回被平移的
            // 线上，文字被渲染成 "1.2.300"。取偶数后，标签恒落在未被平移的线上。
            if (swingOn && stride % 2 !== 0) continue;
            return stride;
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
    // 现在标签判定改为按**全局索引**取模（索引与 stride 都是整数，判定精确，且
    // swing 平移过的线不会被误判），间距交给 §5 的统一让位规则处理。
    //
    // 【为什么索引必须是全局的】旧实现让 `buildTempoGridLines` 按**段内**编号，
    // 于是每个段起点的索引恒为 0，`0 % stride === 0` 对任何 stride 都成立 ——
    // **每个变化点都无条件获得标签**，完全忽略 `minLabelSpacingPx`。变化点区因此
    // 变成"变化点间距"的密集栅格，与尾段的真实栅格首尾拼接（实测 pps=8 时
    // 58px / 193px，3.31×），用户看到的就是"某段之内的刻度与文本消失"。索引全局
    // 单调后，段起点只在其恰好落在栅格上时才带标签；变化点标签由 §4 显式补充。
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
    // 各段起点对应的**全局弱线索引**，作为该段标签栅格的相位基准。必须与
    // `buildTempoGridLines` 的 `segWeakOffset` 用同一公式（同一 stepBeats 与 bpm
    // 基准），否则相位对不上。
    const segmentStartIndices = labelSegments.map((segment) =>
        Math.round(secToBeat(tempoMap, segment.startSec, bpm) / stepBeats),
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
        const segmentIndex =
            tempoMap && tempoMap.points.length > 0 ? pointIndexAtSec(tempoMap, entry.sec) : -1;
        const stride = segmentIndex >= 0 ? segmentLabelStrides[segmentIndex] : labelStride;
        // 标签栅格：全局索引取模即可跨段连续（索引全局单调，见 `TempoGridLine.index`）。
        // swing 打开时再整体偏移 `段起点索引 % 2`，把栅格对齐到段内**偶数**索引
        // （未被 swing 平移）的线上 —— 与 §3b 强制偶数 stride 配合，标签拍值恒为整数。
        const phase =
            segmentIndex >= 0 && swingOn ? ((segmentStartIndices[segmentIndex] % 2) + 2) % 2 : 0;
        const onLabelStep =
            entry.index !== undefined &&
            stride !== undefined &&
            (((entry.index - phase) % stride) + stride) % stride === 0;
        // swing 平移的是**段内奇数**弱线索引的线（`buildTempoGridLines` 的
        // `swingAt(segBpm, k)` 按段内 k 的奇偶判定）。段内索引 = 全局索引 − 段起点
        // 索引；无 Tempo Map 时全局索引即段内索引。密度补充（§5b）据此排除这些线。
        const localWeakIndex =
            entry.index === undefined
                ? undefined
                : segmentIndex >= 0
                  ? entry.index - segmentStartIndices[segmentIndex]
                  : entry.index;
        const swung = swingOn && localWeakIndex !== undefined && Math.abs(localWeakIndex % 2) === 1;
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
            densityEligible: entry.index !== undefined && !swung,
        });
    }

    // merged 的迭代顺序即 lines 的顺序（已升序），显式排序以消除对上游顺序的
    // 隐式依赖。
    ticks.sort((a, b) => a.sec - b.sec);

    // ── 4. 变化点位置强制展示标尺值 ────────────────────────────────
    // 变化点（Tempo Map 段起点）是工程的时间地标，必须带标签：读不出它的位置时
    // 用户只能看到旗帜悬在两个标尺值之间。
    //
    // 【为什么不再顺手清场】旧实现在这里额外隐藏了距变化点
    // `max(minLabelSpacingPx, 26)`（默认 110、可调到 320）以内的**所有**常规
    // 标签，理由是"保证变化点标签的显示宽度"。但标签宽度已由 §5 算出的
    // `labelMaxWidth`（到下一个保留标签的间距 − 6）保证，不需要清场；而清场半径
    // 按**用户设定的标签间距**取，等于把一整个标签位整段挖掉 —— 实测在 121px
    // 标称间距下挖出 290px 的空洞（2.40 倍）、在 327px 下挖出 785px（2.40 倍）。
    //
    // 【为什么强制标签也要限量】变化点可能比 `minLabelSpacingPx` 密得多（例如
    // 每 7s 一个变化点 + 中等缩放）。若全部强制显示，变化点区就变成"变化点间距"
    // 的密集栅格，与其余部分的均匀栅格首尾拼接 —— 正是"某段之内的刻度与文本消失"
    // 的另一种形态（实测 pps=8 时 58px / 193px）。因此强制标签之间也满足
    // `minLabelSpacingPx`：过密时只保留每隔约 `minLabelSpacingPx` 的那一个。
    //
    // 【为什么按整张地图选、而不是按已生成的刻度选】选择必须只依赖**绝对位置**，
    // 否则生成范围（随滚动移动）会改变"哪一个是第一个"，强制标签的相位随滚动
    // 跳变 —— 那正是要消除的闪烁。
    const forcedLabel = new Set<number>();
    if (tempoMap && tempoMap.points.length > 0) {
        const tickIndexByKey = new Map<number, number>();
        for (let i = 0; i < ticks.length; i += 1) {
            tickIndexByKey.set(Math.round(ticks[i].sec * 1e6), i);
        }
        let lastForcedPx: number | null = null;
        for (const point of tempoMap.points) {
            const px = secToContentPx(axis, point.positionSec);
            if (lastForcedPx !== null && px - lastForcedPx < spacingPx - 1e-9) continue;
            lastForcedPx = px;
            const key = Math.round(point.positionSec * 1e6);
            const index =
                tickIndexByKey.get(key) ??
                tickIndexByKey.get(key - 1) ??
                tickIndexByKey.get(key + 1);
            if (index === undefined) continue;
            forcedLabel.add(index);
            if (!ticks[index].showLabel) ticks[index] = { ...ticks[index], showLabel: true };
        }
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
    // ── 5b. 密度补充：约束标签间距的**上限** ──────────────────────
    // 【为什么必须有这一趟】§3~§5 全都是"隐藏"方向的操作：§3 选的是满足
    // `>= spacingPx` 的**最小** stride（只能取 2 的幂 / 音乐候选阶梯，因此实际
    // 间距可漂到 2×），§5 只剔除过密的（< 26px），§6 只在**全部**无标签时兜底。
    // **没有任何一处约束"相邻标签是否离得太远"** —— 于是视口里可以出现 300~450px
    // 没有标签的段（请求 110px），而标尺只渲染带标签的刻度 ⇒ 那一段既没有刻度线
    // 也没有文本。实测最坏 2.73×（无 Tempo Map）/ 4.09×（密集 Tempo Map）。
    //
    // 【为什么从既有刻度里补、而不是改选级】选级必须取整数 stride 才能保证
    // "标签落在网格线上"（既有不变量）；改小 stride 又会违反 `minLabelSpacingPx`。
    // 正确做法是先按栅格选、再从**既有刻度**里补 —— 后者保证"标尺刻度 ⊆ 网格刻度"
    // 结构上成立（绝不凭空造一条网格里没有的线）。
    //
    // 【为什么单趟左→右、按"每对相邻标签各自处理"】处理某一对 (a, b) 时只看 a、b
    // 自身的绝对位置，与其它对、与生成窗口无关。若改成"每次挑最大间距"的全局贪心，
    // 处理顺序会随生成窗口（随滚动移动）变化，同一刻度在不同窗口下可能得到不同的
    // 标签 —— 那正是要消除的闪烁。
    //
    // 【为什么不会造出过密的标签 / 不会打乱栅格】候选必须同时满足：
    // - `densityEligible`：落在**弱线栅格**上且未被 swing 平移（见该字段的说明）。
    //   附点 / 三连音网格下的小节线不落在弱线栅格上，插进来会把"间距均匀"打乱。
    // - 距两侧都 `>= RULER_LABEL_HIDDEN_GAP_PX`：不会触发 §5 的最小间距约束。
    // 因此补充后无需再跑一趟让位。
    const labelDensityTargetPx = spacingPx * LABEL_DENSITY_TARGET_MULT;
    const denseSurviving = [...surviving].sort((a, b) => ticks[a].contentPx - ticks[b].contentPx);
    const densityLabeled = new Set(denseSurviving);
    const densityMinGapPx = RULER_LABEL_HIDDEN_GAP_PX;
    let densityCursor = 1;
    let densityInsertions = 0;
    // 每次插入都占用一个此前未标注的刻度，故插入次数天然有界；上限再加一道保险。
    const densityMaxInsertions = ticks.length + 1;
    while (densityCursor < denseSurviving.length && densityInsertions <= densityMaxInsertions) {
        const leftIndex = denseSurviving[densityCursor - 1];
        const rightIndex = denseSurviving[densityCursor];
        const leftPx = ticks[leftIndex].contentPx;
        const rightPx2 = ticks[rightIndex].contentPx;
        if (rightPx2 - leftPx <= labelDensityTargetPx + 1e-9) {
            densityCursor += 1;
            continue;
        }
        // 理想落点：自左标签起一个目标间距（左偏，确定性）。
        const idealPx = Math.min(leftPx + labelDensityTargetPx, rightPx2 - densityMinGapPx);
        let bestIndex = -1;
        let bestScore = -Infinity;
        for (let i = 0; i < ticks.length; i += 1) {
            if (densityLabeled.has(i)) continue;
            // 只从**弱线栅格**上、且未被 swing 平移的刻度里挑（见 `densityEligible`）。
            if (ticks[i].densityEligible !== true) continue;
            const x = ticks[i].contentPx;
            if (x <= leftPx + densityMinGapPx - 1e-9) continue;
            if (x >= rightPx2 - densityMinGapPx + 1e-9) continue;
            // 优先级：小节起点 > 整数拍 > 其它；同级取最靠近理想落点的。
            const priority =
                (ticks[i].isBarStart ? 2 : 0) +
                (Math.abs(ticks[i].beat - Math.round(ticks[i].beat)) < 1e-6 ? 1 : 0);
            const score = priority * 1e6 - Math.abs(x - idealPx);
            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }
        if (bestIndex < 0) {
            // 这一对之间没有可补的刻度（网格本身就很稀）——属物理必然，跳过。
            densityCursor += 1;
            continue;
        }
        densityLabeled.add(bestIndex);
        denseSurviving.splice(densityCursor, 0, bestIndex);
        densityInsertions += 1;
        // 不推进游标：继续检查 (左标签 → 新标签) 这一对是否仍超限。
    }
    for (const index of denseSurviving) {
        if (!ticks[index].showLabel) ticks[index] = { ...ticks[index], showLabel: true };
    }

    // 版式：宽度取到**下一条保留标签**的间距（减去留白）；最后一条不限宽。
    // 必须在密度补充**之后**算，新增标签才拿得到正确宽度。
    for (let k = 0; k < denseSurviving.length; k += 1) {
        const index = denseSurviving[k];
        const nextIndex = denseSurviving[k + 1];
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
                // 与 §3 / §5b 同一口径：用**钳制后**的 `spacingPx`，而不是原始入参。
                // 旧写法用 `args.minLabelSpacingPx`，用户设了极端值时两处间距不一致。
                ticks[index].contentPx - ticks[last].contentPx < spacingPx
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
