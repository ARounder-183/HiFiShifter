/**
 * crossfadeGripWindow.ts — 交叉点抓手拖拽的几何（共享纯函数）。
 *
 * 【主要内容】
 * 把「交叉点抓手的水平位移」换算为**两侧 clip** 的目标几何（起点 / 长度 /
 * 源窗口），并在反向模式下额外给出两侧淡变长度的按比例缩放结果。
 *
 * 【为什么单独成模块】
 * 这段逻辑原先只存在于旧实现的 `useEditDrag` 内（`type === "crossfade_edges"`）。
 * 渲染内核的抓手手势需要**同一份**语义；两处各写一份会在「循环 / 倒放 / 非 Loop
 * 派生窗口」这几条分支上分叉——而这些差异只在波形与听感上体现（源窗口错了但
 * 长度是对的，画面上完全看不出来），极难归因。
 *
 * 【两种模式（与旧实现逐条一致）】
 * - **默认**：A 的右缘与 B 的左缘**同向**移动同一位移，重叠长度不变。
 * - **反向**（`modifier.crossfadeGrip`，默认 Ctrl/⌘ 按住）：两侧向**相反**方向
 *   移动，重叠长度改变，两侧淡变长度按「新重叠 / 原重叠」比例缩放。
 *
 * 【源窗口的三条分支（两侧各自判定，与 slip 手势同一套语义）】
 * 1. `loopEnabled`：只改长度（内容按周期回绕）或把锚点在媒体域内环绕推进；
 * 2. 非 Loop **正放**：派生窗口 —— `sourceEndSec = sourceStartSec + 长度 × 速率`，
 *    越出媒体的部分渲染静音（尾静音 / 前导静音）；
 * 3. 非 Loop **倒放**：`sourceStartSec`（右缘）或 `sourceEndSec`（左缘）是反向
 *    锚点，按位移反方向推进，越界部分同样是静音。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 的 `handleKernelCrossfadeGripPreview/Commit`。
 * - 依赖：`slipWindow` 的 `readSlipClip`（源窗口字段归一化，单一事实来源）。
 * - 独立性：纯函数，无 React / DOM / Redux 依赖，可直接单测。
 */

/** 抓手拖拽两侧的**按下时**几何快照。 */
export interface CrossfadeGripClipBase {
    readonly id: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    readonly playbackRate: number;
    readonly loopEnabled: boolean;
    readonly reversed: boolean;
    /** 媒体 / 内容时长（秒）；<= 0 表示未知（环绕退化为不环绕）。 */
    readonly mediaDurationSec: number;
}

/** 抓手拖拽的换算参数。 */
export interface CrossfadeGripArgs {
    /** 前一个 clip（右缘随拖拽移动）。 */
    readonly earlier: CrossfadeGripClipBase;
    /** 后一个 clip（左缘随拖拽移动）。 */
    readonly later: CrossfadeGripClipBase;
    /** 原始水平位移（秒，向右为正）。 */
    readonly deltaSec: number;
    /** 是否反向模式（`modifier.crossfadeGrip` 按住）。 */
    readonly opposite: boolean;
    /** 按下时的重叠长度（秒）——反向模式的缩放比例基准。 */
    readonly baseOverlapSec: number;
    /** 按下时 A 的**生效**淡出长度（秒）。 */
    readonly earlierFadeOutSec: number;
    /** 按下时 B 的**生效**淡入长度（秒）。 */
    readonly laterFadeInSec: number;
    /** A 的淡出当前是否由自动交叉淡化提供（决定缩放写 auto 还是 manual 字段）。 */
    readonly earlierFadeOutAuto: boolean;
    /** B 的淡入当前是否由自动交叉淡化提供。 */
    readonly laterFadeInAuto: boolean;
}

/** 一侧的目标几何补丁（只含需要改写的字段）。 */
export interface CrossfadeGripPatch {
    readonly clipId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly sourceStartSec?: number;
    readonly sourceEndSec?: number;
}

/** 一侧的淡变更新（反向模式下的比例缩放）。 */
export interface CrossfadeGripFadeUpdate {
    readonly clipId: string;
    readonly fadeInSec?: number;
    readonly fadeOutSec?: number;
    readonly autoFadeInSec?: number;
    readonly autoFadeOutSec?: number;
}

/** 抓手拖拽的换算结果。 */
export interface CrossfadeGripResult {
    /** 实际生效的位移（秒，已按两侧的可行区间钳制）。 */
    readonly deltaSec: number;
    readonly earlier: CrossfadeGripPatch;
    readonly later: CrossfadeGripPatch;
    /** 反向模式下的淡变缩放结果；默认模式为空数组。 */
    readonly fades: CrossfadeGripFadeUpdate[];
}

/** 反向模式下允许的最小重叠（秒）：再小就不是交叉淡化而是"擦肩"。 */
const MIN_CROSSFADE_OVERLAP_SEC = 0.0002;

/** 把值环绕进 `[0, d)`（媒体域）；`d` 未知时原样返回。 */
function wrapIntoMediaDomain(valueSec: number, mediaDurationSec: number): number {
    const d = Number(mediaDurationSec);
    if (!(d > 1e-9)) return valueSec;
    let v = valueSec % d;
    if (v < 0) v += d;
    return v;
}

/** 取正数速率（非法值退化为 1，避免除零）。 */
function safeRate(rate: number): number {
    return Number.isFinite(rate) && rate > 0 ? rate : 1;
}

/**
 * 计算一次交叉点抓手拖拽的目标几何。
 *
 * @param args 见 `CrossfadeGripArgs`。
 * @returns 目标几何与淡变缩放；两侧可行区间无交集（或输入非法）时返回 null
 *   （调用方应放弃本次预览）。
 */
export function computeCrossfadeGrip(args: CrossfadeGripArgs): CrossfadeGripResult | null {
    const a = args.earlier;
    const b = args.later;
    const rawDelta = Number(args.deltaSec);
    if (!Number.isFinite(rawDelta)) return null;
    const aRate = safeRate(a.playbackRate);
    const bRate = safeRate(b.playbackRate);
    const baseOverlapSec = Math.max(0, Number(args.baseOverlapSec) || 0);

    // ── 可行区间：A 右缘与 B 左缘共享同一位移，任一顶到极限两者一起停 ──
    let minDelta = -a.lengthSec;
    let maxDelta = Number.POSITIVE_INFINITY;
    // B.start = baseStart + bStartSign × delta。
    const bStartSign = args.opposite ? -1 : 1;
    if (args.opposite) {
        // 反向：向左拖（delta<0）把 B 起点向右推（裁短 B），向右拖反之。
        minDelta = Math.max(minDelta, -b.lengthSec);
        maxDelta = Math.min(maxDelta, b.startSec);
        // 反向模式下必须保持两侧仍然重叠：newOverlap = baseOverlap + 2×delta >= min。
        minDelta = Math.max(minDelta, (MIN_CROSSFADE_OVERLAP_SEC - baseOverlapSec) / 2);
    } else {
        // 同向：B 左缘随 A 右缘一起平移，重叠长度不变。
        minDelta = Math.max(minDelta, -b.startSec);
        maxDelta = Math.min(maxDelta, b.lengthSec);
    }
    if (minDelta > maxDelta) return null;
    const delta = Math.min(maxDelta, Math.max(minDelta, rawDelta));

    // ── A：右缘（结束位置）移动 delta（语义同 trim_right）──
    const earlier: CrossfadeGripPatch = (() => {
        if (a.loopEnabled) {
            // Loop：只改长度，源窗口不变（内容按周期回绕）。
            return {
                clipId: a.id,
                startSec: a.startSec,
                lengthSec: Math.max(0, a.lengthSec + delta),
            };
        }
        if (a.reversed) {
            // 倒放非 Loop：右缘延伸向下消费窗口起点（→0），耗尽后的部分为静音尾巴。
            const nextTrimStart = Math.min(
                a.sourceEndSec,
                Math.max(0, a.sourceStartSec - delta * aRate),
            );
            return {
                clipId: a.id,
                startSec: a.startSec,
                lengthSec: Math.max(0, a.lengthSec + delta),
                sourceStartSec: nextTrimStart,
            };
        }
        // 正放非 Loop：派生窗口 —— source_end = 起点 + 长度 × 速率。
        const nextLen = Math.max(0, a.lengthSec + delta);
        return {
            clipId: a.id,
            startSec: a.startSec,
            lengthSec: nextLen,
            sourceEndSec: a.sourceStartSec + nextLen * aRate,
        };
    })();

    // ── B：左缘（起始位置）移动 bStartSign × delta（语义同 trim_left）──
    const startDelta = bStartSign * delta;
    const later: CrossfadeGripPatch = (() => {
        const shortensFromLeft = startDelta > 0;
        const nextStart = b.startSec + startDelta;
        const nextLength = Math.max(0, b.lengthSec - startDelta);
        if (b.loopEnabled && !shortensFromLeft) {
            // Loop + 左缘**左移**（延伸）：锚点回退并环绕，内容保持锚定。
            return b.reversed
                ? {
                      clipId: b.id,
                      startSec: Math.max(0, nextStart),
                      lengthSec: nextLength,
                      sourceEndSec: wrapIntoMediaDomain(
                          b.sourceEndSec - startDelta * bRate,
                          b.mediaDurationSec,
                      ),
                  }
                : {
                      clipId: b.id,
                      startSec: Math.max(0, nextStart),
                      lengthSec: nextLength,
                      sourceStartSec: wrapIntoMediaDomain(
                          b.sourceStartSec + startDelta * bRate,
                          b.mediaDurationSec,
                      ),
                  };
        }
        if (b.loopEnabled) {
            // Loop + 左缘**右移**（裁短）：锚点向前环绕推进。
            return b.reversed
                ? {
                      clipId: b.id,
                      startSec: nextStart,
                      lengthSec: nextLength,
                      sourceEndSec: wrapIntoMediaDomain(
                          b.sourceEndSec - startDelta * bRate,
                          b.mediaDurationSec,
                      ),
                  }
                : {
                      clipId: b.id,
                      startSec: nextStart,
                      lengthSec: nextLength,
                      sourceStartSec: wrapIntoMediaDomain(
                          b.sourceStartSec + startDelta * bRate,
                          b.mediaDurationSec,
                      ),
                  };
        }
        if (b.reversed) {
            // 倒放非 Loop：左缘对应窗口终点（source_end）。越界部分为前导静音。
            const nextTrimEnd = b.sourceEndSec - startDelta * bRate;
            const actualDeltaTimeline = (b.sourceEndSec - nextTrimEnd) / bRate;
            return {
                clipId: b.id,
                startSec: b.startSec + actualDeltaTimeline,
                lengthSec: Math.max(0, b.lengthSec - actualDeltaTimeline),
                sourceEndSec: nextTrimEnd,
            };
        }
        // 正放非 Loop：左缘对应窗口起点（source_start）。越界部分为前导静音。
        const nextTrimStart = b.sourceStartSec + startDelta * bRate;
        const actualDeltaTimeline = (nextTrimStart - b.sourceStartSec) / bRate;
        return {
            clipId: b.id,
            startSec: b.startSec + actualDeltaTimeline,
            lengthSec: Math.max(0, b.lengthSec - actualDeltaTimeline),
            sourceStartSec: nextTrimStart,
        };
    })();

    // ── 反向模式：按「新重叠 / 原重叠」比例缩放两侧淡变 ──
    // 自动淡化 → 写 auto 字段（因此 auto 始终 == 重叠长度）；手动淡化 → 写手动字段
    // （保持原有比例）。默认模式（同向）重叠不变，无需缩放。
    const fades: CrossfadeGripFadeUpdate[] = [];
    if (args.opposite && baseOverlapSec > 0.001) {
        const newOverlap = Math.max(MIN_CROSSFADE_OVERLAP_SEC, baseOverlapSec + 2 * delta);
        const ratio = newOverlap / baseOverlapSec;
        const aFade = args.earlierFadeOutSec * ratio;
        const bFade = args.laterFadeInSec * ratio;
        if (args.earlierFadeOutAuto) {
            fades.push({ clipId: a.id, autoFadeOutSec: aFade });
        } else {
            fades.push({ clipId: a.id, fadeOutSec: aFade });
        }
        if (args.laterFadeInAuto) {
            fades.push({ clipId: b.id, autoFadeInSec: bFade });
        } else {
            fades.push({ clipId: b.id, fadeInSec: bFade });
        }
    }

    return { deltaSec: delta, earlier, later, fades };
}
