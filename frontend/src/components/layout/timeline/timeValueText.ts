/**
 * timeValueText — 时间量 Tooltip 的**主/副单位**组合。
 *
 * ## 为什么单独成模块
 *
 * 时间轴上"把某个时间量按用户的主/副时间单位显示"这件事，此前在四个地方各写了一遍：
 *
 * - `timeFormat.formatFadeLengthTooltip`（淡化长度）
 * - `TempoMapRulerRow` 的纯文本版与富内容版（逐字重复）
 * - `TimeRuler` 的 Tempo Map 变化点提示
 *
 * 判据 `副单位 !== "none" && 副单位 !== 主单位` 与 `{主} / {副}` 的拼接散落各处，
 * 每新增一个"显示时间量"的 UI 就要再抄一次 —— 本模块是这一层的唯一落点。
 *
 * ## 两种时间量的口径**不同**，不得抹平
 *
 * | | 时长（零基点） | 时刻（绝对位置） |
 * |---|---|---|
 * | 例 | 淡化长度、**吸附偏移** | 播放头、Tempo Map 变化点、**吸附偏移的位置** |
 * | 函数 | {@link formatDurationText} → `formatDurationUnit` | {@link formatPositionText} → `formatCursorUnit` |
 * | Tempo Map | **不感知** | **感知** |
 *
 * - 时长**不该**做 Tempo Map 分段积分：时长的音乐学定义依赖它所在的起点，而相对
 *   时长 UI 一律按工程全局 BPM 静态折算（见 `formatDurationUnit` 的论证）。
 *   因此 {@link formatDurationText} 的签名里**根本没有** `tempoMap` —— 从类型上
 *   断掉"顺手把绝对位置当长度显示"这一类误用。
 * - 时刻**必须**感知 Tempo Map，否则与标尺 / 光标显示对不上。
 *
 * @see TimeValueFormatContext
 */
import { formatCursorUnit, formatDurationUnit } from "./timeFormat";
import type { FadeLengthFormatContext, TimeUnit } from "./timeFormat";
import type { TempoMap } from "../../../utils/tempoMap";
import type { MessageKey } from "../../../i18n/messages";

/**
 * 时间量格式化上下文：主/副单位 + 计时参数。
 *
 * 继承 {@link FadeLengthFormatContext}（淡化侧现成的对象因此可以直接传入，零改动），
 * 只额外补一个可选的 `tempoMap` —— 时刻格式化需要它，时长按定义忽略它。
 */
export interface TimeValueFormatContext extends FadeLengthFormatContext {
    /** 仅**时刻**格式化使用（见文件头的口径表）。 */
    tempoMap?: TempoMap | null;
}

/** 副时间单位是否参与展示：已启用且与主单位不同。 */
export function hasSecondaryUnit(ctx: TimeValueFormatContext): boolean {
    return ctx.secondaryTimeUnit !== "none" && ctx.secondaryTimeUnit !== ctx.primaryTimeUnit;
}

/** `{主} / {副}`；`secondary` 为 null 时只给主单位。 */
function joinUnits(primary: string, secondary: string | null): string {
    return secondary === null ? primary : `${primary} / ${secondary}`;
}

/**
 * **时长**（零基点，不做 Tempo Map 分段积分）→ `{主}` 或 `{主} / {副}`。
 */
export function formatDurationText(
    durationSec: number,
    ctx: TimeValueFormatContext,
): string {
    const main = formatDurationUnit(ctx.primaryTimeUnit, durationSec, ctx);
    if (!hasSecondaryUnit(ctx)) return main;
    return joinUnits(main, formatDurationUnit(ctx.secondaryTimeUnit as TimeUnit, durationSec, ctx));
}

/**
 * **时刻**（绝对位置，Tempo Map 感知）→ `{主}` 或 `{主} / {副}`。
 */
export function formatPositionText(sec: number, ctx: TimeValueFormatContext): string {
    const main = formatCursorUnit(ctx.primaryTimeUnit, sec, ctx);
    if (!hasSecondaryUnit(ctx)) return main;
    return joinUnits(main, formatCursorUnit(ctx.secondaryTimeUnit as TimeUnit, sec, ctx));
}

/** 小于此值（秒）的位移视为"没有移动"，不展示增量。 */
const DELTA_EPSILON_SEC = 1e-6;

/**
 * **带符号的时长**（位移量）→ `+{时长}` / `-{时长}`。
 *
 * 时长格式化器把负值钳到 0（时长不可为负），因此符号在这里单独处理：取绝对值走
 * 同一套主/副单位格式化，再前置 `+` / `-`。与 `formatGainDbValue` 的符号约定一致。
 */
export function formatSignedDurationText(
    deltaSec: number,
    ctx: TimeValueFormatContext,
): string {
    const safe = Number.isFinite(deltaSec) ? deltaSec : 0;
    const sign = safe < 0 ? "-" : "+";
    return `${sign}${formatDurationText(Math.abs(safe), ctx)}`;
}

/** 取值器：与 `useI18n().tVars` 同形（`{name}` 插值，全部出现处都替换）。 */
export type TimeValueLabelLookup = (
    key: MessageKey,
    vars: Record<string, string>,
) => string;

/**
 * 吸附偏移的 Tooltip 文本（悬停 / 拖拽共用**唯一**判定点）。
 *
 * 三种形态：
 *
 * ```text
 * 未拖拽 且 偏移 = 0  →  {标签}                                // 位置恒等于 Clip 起点，无信息量
 * 未拖拽 且 偏移 ≠ 0  →  吸附偏移：{主} / {副}
 *                        位置：{主} / {副}
 * 拖拽中（有位移）    →  吸附偏移：{主} / {副} [{±主} / {±副}]
 *                        位置：{主} / {副} [{±主} / {±副}]
 * ```
 *
 * 【为什么"偏移 = 0 只给标签"不适用于拖拽】拖拽中用户正在**操作**这个量：
 * 把它拖回 0 时恰恰最需要看到 `0.000` 与本次位移，此时退回裸标签反而像功能失效。
 * 因此该规则只在"没有位移"时生效 —— 判据落在**位移**而不是偏移值上。
 *
 * 【为什么整段文案进词典而不是拼标签+值】语序与冒号形态是文案而非代码：
 * en-US 的 `"Position: "` 带尾空格、ko-KR 用 `"위치: "`、CJK 用全角「：」，
 * 硬拼会把这些差异冻死在代码里。词典里只需三个占位符 —— `{offset}` /
 * `{position}` / `{delta}` 的**值**已经含主副组合（由上面两个格式化器产出），
 * 因此不必为"有副单位 / 无副单位"各写一份文案。
 *
 * @param offsetSec 吸附偏移（相对 Clip 起点的时长，秒；负值按 0 处理）。
 * @param positionSec 该吸附偏移在**整条时间轴**上的绝对位置（秒）。
 * @param deltaSec 本次拖拽相对按下时的位移（秒）；悬停传 `null`。
 */
export function buildSnapOffsetInfoText(args: {
    offsetSec: number;
    positionSec: number;
    deltaSec: number | null;
    formatCtx: TimeValueFormatContext;
    t: TimeValueLabelLookup;
}): string {
    const offsetSec = Math.max(0, Number.isFinite(args.offsetSec) ? args.offsetSec : 0);
    const dragging =
        args.deltaSec !== null &&
        Number.isFinite(args.deltaSec) &&
        Math.abs(args.deltaSec) > DELTA_EPSILON_SEC;
    if (!dragging && !(offsetSec > 0)) return args.t("clip_snap_offset", {});
    const offset = formatDurationText(offsetSec, args.formatCtx);
    const position = formatPositionText(args.positionSec, args.formatCtx);
    if (!dragging) {
        return args.t("clip_snap_offset_value", { offset, position });
    }
    return args.t("clip_snap_offset_value_drag", {
        offset,
        position,
        delta: formatSignedDurationText(args.deltaSec as number, args.formatCtx),
    });
}
